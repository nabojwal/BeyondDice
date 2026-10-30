"""Evaluate encoded SPIDER held-out segmentations; never resample or alter masks."""
import argparse
import csv
import hashlib
import itertools
import json
from importlib.metadata import version
from pathlib import Path
import tempfile
import shutil
import numpy as np

METRICS = ('dice', 'iou', 'hd95_mm', 'assd_mm', 'surface_dice')
LABELS = {'background': 0, **{f'vertebra_{i}': i for i in range(1, 10)},
          'spinal_canal': 10, **{f'disc_{i+190}': i for i in range(11, 20)}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def label_metrics(gt, pred, spacing_zyx, tolerance):
    ng, npred = int(gt.sum()), int(pred.sum())
    tp = int(np.count_nonzero(gt & pred))
    out = dict(gt_voxels=ng, pred_voxels=npred, tp=tp, fp=npred-tp, fn=ng-tp)
    if not ng and not npred:
        return dict(out, status='both_absent', **{m: None for m in METRICS})
    out.update(dice=2*tp/(ng+npred), iou=tp/(ng+npred-tp))
    if not ng or not npred:
        return dict(out, status='missed' if ng else 'false_positive',
                    hd95_mm=float('inf'), assd_mm=float('inf'), surface_dice=0.)
    # Crop both masks together; the surface-distance library supplies boundary padding.
    coords = np.nonzero(gt | pred)
    crop = tuple(slice(int(c.min()), int(c.max())+1) for c in coords)
    from surface_distance import metrics as surface
    sd = surface.compute_surface_distances(gt[crop], pred[crop], spacing_zyx)
    # Pooled surface-area weighted symmetric average, not mean of directed means.
    area_g, area_p = sd['surfel_areas_gt'], sd['surfel_areas_pred']
    assd = (np.dot(sd['distances_gt_to_pred'], area_g) +
            np.dot(sd['distances_pred_to_gt'], area_p)) / (area_g.sum()+area_p.sum())
    return dict(out, status='both_present', hd95_mm=float(surface.compute_robust_hausdorff(sd, 95)),
                assd_mm=float(assd), surface_dice=float(surface.compute_surface_dice_at_tolerance(sd, tolerance)))


def load_mask(path):
    import SimpleITK as sitk
    im = sitk.ReadImage(str(path))
    if im.GetDimension() != 3 or im.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f'Expected scalar 3D mask: {path}')
    a = sitk.GetArrayFromImage(im)
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f'Noninteger/nonfinite mask: {path}')
    if not set(np.unique(a)).issubset(set(LABELS.values())):
        raise ValueError(f'Unknown encoded label: {path}; do not use original source IDs')
    spacing = np.asarray(im.GetSpacing())
    direction = np.asarray(im.GetDirection()).reshape(3,3)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid spacing')
    if not np.isfinite(direction).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('Direction must be orthonormal for spacing-based surface distances')
    if not np.isfinite(im.GetOrigin()).all():
        raise ValueError('Invalid origin')
    return im, a


def check_geometry(a, b, tolerance_mm):
    if a.GetSize() != b.GetSize():
        raise ValueError('Image dimensions differ')
    displacements = []
    for index in itertools.product(*[(0, n-1) for n in a.GetSize()]):
        displacements.append(np.linalg.norm(np.asarray(a.TransformIndexToPhysicalPoint(index))-
                                            b.TransformIndexToPhysicalPoint(index)))
    error = float(max(displacements))
    if error > tolerance_mm:
        raise ValueError(f'Geometry mismatch: maximum corresponding-corner displacement {error} mm')
    return error


def subject_map(path, cases):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        case, subject = row['case_id'].strip(), row['subject_id'].strip()
        if not case or not subject or case in result:
            raise ValueError('Empty or duplicate entry in subject map')
        result[case] = subject
    if set(result) != set(cases):
        raise ValueError('Subject map must contain exactly the held-out case IDs, one row per scan')
    return result


def summary(rows, n_boot, seed):
    """Equal-subject estimand: average eligible sequences within each subject first."""
    output = []
    for label in range(1,20):
        selected = [r for r in rows if r['label_id'] == label]
        subjects = sorted({r['subject_id'] for r in selected})
        status_counts = {s: sum(r['status']==s for r in selected)
                         for s in ('both_absent','missed','false_positive','both_present')}
        for metric in METRICS:
            vals = np.array([np.nan if r[metric] is None else r[metric] for r in selected])
            defined = vals[~np.isnan(vals)]
            finite = defined[np.isfinite(defined)]
            by_subject = []
            for subject in subjects:
                v = [r[metric] for r in selected if r['subject_id']==subject and r[metric] is not None]
                by_subject.append(float(np.mean(v)) if v else np.nan)
            by_subject = np.asarray(by_subject)
            available = by_subject[~np.isnan(by_subject)]
            point = float(np.mean(available)) if len(available) else None
            # Resample whole subjects; every sequence within a subject stays together.
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(n_boot):
                sample = by_subject[rng.integers(0,len(subjects),len(subjects))]
                sample = sample[~np.isnan(sample)]
                if len(sample): boots.append(float(np.mean(sample)))
            ci = [None, None]
            if boots and len(available) >= 2:
                # Empirical order statistics avoid interpolation of infinity.
                ordered = np.sort(boots)
                ci = [float(ordered[int(np.floor(.025*(len(ordered)-1)))]),
                      float(ordered[int(np.ceil(.975*(len(ordered)-1)))])]
            output.append(dict(label_id=label, label_name=selected[0]['label_name'], metric=metric,
                               n_scans=len(selected), **status_counts,
                               n_defined=len(defined), n_infinite=int(np.isinf(defined).sum()),
                               n_finite=len(finite), scan_mean=float(np.mean(defined)) if len(defined) else None,
                               finite_only_scan_mean=float(np.mean(finite)) if len(finite) else None,
                               n_subjects_total=len(subjects), n_subjects_defined=len(available),
                               subject_mean=point, subject_ci95_low=ci[0], subject_ci95_high=ci[1],
                               bootstrap_valid_replicates=len(boots)))
    return output


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(args):
    if not np.isfinite(args.surface_tolerance_mm) or args.surface_tolerance_mm <= 0:
        raise ValueError('Surface Dice tolerance must be positive and finite')
    if args.bootstrap < 100 or args.geometry_tolerance_mm < 0 or not np.isfinite(args.geometry_tolerance_mm):
        raise ValueError('Invalid bootstrap count or geometry tolerance')
    raw, pred, out = args.raw, args.predictions, args.output
    dataset = read_json(raw/'dataset.json')
    mapping = read_json(raw/'spider_label_mapping.json')
    contract = read_json(raw/'spider_contract.json')
    manifest = read_json(raw/'spider_source_manifest.json')
    if dataset['labels'] != LABELS or mapping['labels'] != LABELS:
        raise ValueError('Expected exact 20-class Dataset012 encoded label mapping')
    if (digest(dataset) != contract['dataset_sha256'] or
        digest(manifest) != contract['source_manifest_sha256'] or
        digest({k:v for k,v in mapping.items() if k!='sha256'}) != mapping['sha256'] or
        mapping['sha256'] != contract['label_mapping_sha256']):
        raise ValueError('Dataset/label mapping/source manifest contract mismatch')
    cases = sorted(manifest['test'])
    if len(cases) != 87 or set(cases) & set(manifest['train']):
        raise ValueError('Expected exactly 87 held-out scans with no training-case overlap')
    expected = {case+'.mha' for case in cases}
    for directory in (raw/'labelsTs', pred):
        if {p.name for p in directory.glob('*.mha')} != expected:
            raise ValueError(f'Missing or extra .mha segmentations in {directory}')
    groups = subject_map(args.subject_map, cases)
    if out.exists(): raise FileExistsError(f'Output exists; preserving: {out}')
    rows, provenance = [], []
    for i,case in enumerate(cases,1):
        gp, pp = raw/'labelsTs'/f'{case}.mha', pred/f'{case}.mha'
        gi,g = load_mask(gp); pi,p = load_mask(pp)
        delta = check_geometry(gi,pi,args.geometry_tolerance_mm)
        spacing = tuple(reversed(gi.GetSpacing())) # SimpleITK xyz -> NumPy zyx.
        for name,label in sorted(LABELS.items(), key=lambda kv:kv[1]):
            if label == 0: continue
            row = dict(case_id=case, subject_id=groups[case], label_id=label, label_name=name, source_label_id=(100 if label==10 else label+190 if label>10 else label),
                       surface_tolerance_mm=args.surface_tolerance_mm)
            row.update(label_metrics(g==label,p==label,spacing,args.surface_tolerance_mm))
            rows.append(row)
        provenance.append(dict(case_id=case, gt_sha256=sha(gp), prediction_sha256=sha(pp),
                               spacing_xyz_mm=list(gi.GetSpacing()), max_corner_displacement_mm=delta))
        print(f'[{i}/87] {case}', flush=True)
    aggregate = summary(rows,args.bootstrap,args.seed)
    metadata = dict(schema_version=1, scans=87, foreground_labels=19, rows=len(rows),
                    subject_count=len(set(groups.values())), subject_map_sha256=sha(args.subject_map),
                    surface_tolerance_mm=args.surface_tolerance_mm,
                    geometry_tolerance_mm=args.geometry_tolerance_mm,
                    bootstrap_replicates=args.bootstrap, seed=args.seed,
                    bootstrap_estimand='equal-subject mean of eligible within-subject sequence means',
                    empty_policy='both absent: undefined; one absent: overlap/surface Dice 0, distances infinity',
                    hd95_definition='maximum of two directed surface-area-weighted 95th percentiles',
                    assd_definition='pooled bidirectional surface-area-weighted mean distance',
                    versions={k:version(k) for k in ('numpy','scipy','SimpleITK','surface-distance')},
                    evaluator_sha256=sha(__file__), cases=provenance)
    if args.inference_manifest:
        metadata['inference'] = read_json(args.inference_manifest)
        metadata['inference_manifest_sha256'] = sha(args.inference_manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.spider-eval-',dir=out.parent))
    try:
        write_csv(temp/'per_scan_per_label.csv',rows)
        write_csv(temp/'per_label_summary.csv',aggregate)
        (temp/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        if out.exists(): raise FileExistsError(out)
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True); raise
    print(f'Complete: {len(rows)} rows -> {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','predictions','output','subject-map'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--surface-tolerance-mm',type=float,required=True)
    p.add_argument('--geometry-tolerance-mm',type=float,default=.001)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--inference-manifest',type=Path)
    evaluate(p.parse_args())

if __name__=='__main__': main()
