"""Validate and install the SPIDER internal patient holdout; never use official validation."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'spider_baseline.json'


def digest(value):
    """Hash JSON by content, independent of indentation and dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def patient_id(case):
    """Validate the local SPIDER convention: numeric patient plus MRI sequence."""
    match = re.fullmatch(r'([1-9][0-9]*)_(t1|t2|t2_SPACE)', case)
    if not match:
        raise ValueError(f'Invalid SPIDER case identifier: {case!r}')
    return match[1]


def make_split(csv_path, fraction=0.2, seed=42):
    """Derive a deterministic patient holdout using official training rows only."""
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    csv_path = Path(csv_path)
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'new_file_name', 'subset'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires new_file_name and subset columns')
        rows = list(reader)
    groups = {'training': {}, 'validation': {}}
    seen = set()
    for row in rows:
        case = (row.get('new_file_name') or '').strip()
        subset = (row.get('subset') or '').strip().lower()
        patient = patient_id(case)
        if case in seen:
            raise ValueError(f'Duplicate case: {case}')
        seen.add(case)
        if subset not in groups:
            raise ValueError(f'Unknown subset: {subset}')
        groups[subset].setdefault(patient, []).append(case)
    if set(groups['training']) & set(groups['validation']):
        raise ValueError('Official train/validation patient leakage')
    patients = sorted(groups['training'], key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    count = math.ceil(len(patients) * fraction)
    if not 0 < count < len(patients):
        raise ValueError('Too few patients for split')
    held_out = set(patients[:count])
    split = {'train': [], 'val': []}
    for patient, cases in groups['training'].items():
        split['val' if patient in held_out else 'train'].extend(cases)
    split = {key: sorted(value) for key, value in split.items()}
    result = [split]
    report = {
        'seed': seed, 'validation_patient_fraction': fraction,
        'algorithm': 'sort patients by SHA256(seed:patient), first ceil(N*fraction) to validation',
        'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'split_sha256': digest(result),
        'train_patients': len(patients) - count, 'val_patients': count,
        'train_scans': len(split['train']), 'val_scans': len(split['val']),
        'official_training_scans': sum(map(len, groups['training'].values())),
        'official_training_patients': len(patients),
        'official_validation_scans_excluded': sum(map(len, groups['validation'].values())),
        'official_validation_patients_excluded': len(groups['validation']),
    }
    return result, report


def validate_split(splits, expected_cases=None, approved_hash=None):
    """Reject duplicates, patient leakage, incomplete coverage, and altered designs."""
    if not isinstance(splits, list) or len(splits) != 1 or set(splits[0]) != {'train', 'val'}:
        raise ValueError('Expected exactly one train/val fold')
    train, val = splits[0]['train'], splits[0]['val']
    if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
        raise ValueError('Both partitions must be nonempty lists')
    cases = train + val
    patients_train = {patient_id(case) for case in train}
    patients_val = {patient_id(case) for case in val}
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate or overlapping cases')
    if patients_train & patients_val:
        raise ValueError('Patient leakage in internal split')
    if expected_cases is not None and set(cases) != set(expected_cases):
        raise ValueError('Split does not exactly cover dataset cases')
    if approved_hash is not None and digest(splits) != approved_hash:
        raise ValueError('Split differs from the verified baseline design')
    return train, val


def discover_dataset(raw_root, dataset_id=11):
    """Find exactly one real dataset directory for the requested ID."""
    candidates = sorted(p for p in Path(raw_root).glob(f'Dataset{dataset_id:03d}_*') if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(f'Expected one Dataset{dataset_id:03d}_* directory, found {len(candidates)}')
    return candidates[0]


def verify_dataset(raw, pre, splits):
    """Check metadata, exact raw pairing and identifiers in completed preprocessing."""
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset = json.loads((raw / 'dataset.json').read_text(encoding='utf-8-sig'))
    if dataset['file_ending'] != '.mha' or dataset['numTraining'] != 360:
        raise ValueError('Expected 360 .mha training cases')
    if dataset['channel_names'] != {'0': 'MRI'}:
        raise ValueError('Expected one MRI channel')
    labels = {name.replace(' ', '_'): value for name, value in dataset['labels'].items()}
    if labels != {'background': 0, 'vertebrae': 1, 'intervertebral_discs': 2, 'spinal_canal': 3}:
        raise ValueError('Unexpected semantic label mapping')
    expected = set(splits[0]['train'] + splits[0]['val'])
    for folder, suffix in [('imagesTr', '_0000.mha'), ('labelsTr', '.mha')]:
        actual = {p.name for p in (raw / folder).iterdir() if p.is_file()}
        if actual != {case + suffix for case in expected}:
            raise ValueError(f'CSV does not exactly match {folder} filenames')
    plans = json.loads((pre / 'nnUNetPlans.json').read_text())
    if plans['dataset_name'] != raw.name:
        raise ValueError('Plans belong to a different dataset')
    pre_dataset = json.loads((pre / 'dataset.json').read_text())
    if any(pre_dataset[key] != dataset[key] for key in ['labels', 'channel_names', 'file_ending', 'numTraining']):
        raise ValueError('Raw and preprocessed metadata disagree')
    identifiers = {}
    manager = PlansManager(plans)
    for name in plans['configurations']:
        config = manager.get_configuration(name)
        folder = pre / config.data_identifier
        if folder.name in identifiers:
            continue
        dataset_class = infer_dataset_class(str(folder))
        found = dataset_class.get_identifiers(str(folder))
        if len(found) != len(expected) or set(found) != expected:
            raise ValueError(f'Preprocessed cases disagree in {folder}')
        identifiers[folder.name] = len(found)
    return identifiers


def install_split(pre, splits, report, results_root=None, dataset_name=None):
    """Accept an identical split; preserve conflicts and report potentially dependent runs."""
    path = Path(pre) / 'splits_final.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8-sig'))
        if existing != splits:
            runs = []
            if results_root and dataset_name:
                runs = [str(p) for p in (Path(results_root) / dataset_name).rglob('checkpoint_*.pth')]
            raise FileExistsError(f'Different split preserved at {path}. Dependent checkpoints: {runs}')
        status = 'Identical split already installed'
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(splits, handle, indent=2)
            handle.write('\n')
        status = 'Installed split'
    report_path = Path(pre) / 'spider_split_report.json'
    if not report_path.exists():
        with report_path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    return status, path


def main():
    """Verify the approved design and install fold 0 without preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    splits, report = make_split(args.csv, config['validation_patient_fraction'], config['seed'])
    validate_split(splits, approved_hash=config['split_sha256'])
    if any(report[key] != value for key, value in config['counts'].items()):
        raise ValueError('Local CSV counts differ from the approved study design')
    raw = discover_dataset(os.environ['nnUNet_raw'], config['dataset_id'])
    pre = Path(os.environ['nnUNet_preprocessed']) / raw.name
    report['preprocessed_case_counts'] = verify_dataset(raw, pre, splits)
    status, path = install_split(pre, splits, report, os.environ.get('nnUNet_results'), raw.name)
    print(f'{status}: {path}')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
