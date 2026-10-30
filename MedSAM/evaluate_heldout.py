#!/usr/bin/env python3
"""Held-out multi-sequence, oracle-box MedSAM native-3D evaluation.

Ground-truth masks determine oracle box location AND the list of target instances.
Therefore NOT an automatic detection benchmark. Test labels are not used for training,
model selection or threshold tuning. Run preflight before inference.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REGEX = r'(?i)^(?P<patient>.+?)_(?P<modality>t1|t2|t2_space)$'
FIELDS = ['run','patient_id','series_id','modality','label','structure','status','gt_voxels','pred_voxels','dice','iou','hd95_mm','assd_mm','surface_dice','gt_volume_mm3','pred_volume_mm3','volume_error_pct']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,default=str)); temp.replace(path)

def audit(test_dir, split_file, schema_file, expected=87):
    from patient_manifests import discover_pairs
    from instance_dataset import read_label_schema
    from mri_preprocessing import assert_same_geometry, _axes
    test_dir=Path(test_dir)
    rec=discover_pairs(test_dir/'images',test_dir/'masks',r'(?i)^(?P<patient>.+?)_(?:t1|t2|t2_space)$')
    split=json.loads(Path(split_file).read_text())
    trainval=set(split['assignments'])
    testids={r['patient_id'] for r in rec}
    overlap=sorted(trainval & testids)
    if overlap: raise ValueError(f'TEST LEAKAGE: patient IDs in TrainData and TestData: {overlap}')
    if len(rec)!=expected: raise ValueError(f'Expected {expected} test series, found {len(rec)}')
    schema=read_label_schema(schema_file)
    from collections import Counter
    kinds=Counter(); label_counts=Counter(); shape_counts=Counter()
    issues=[]
    for r in rec:
        try:
            image,mask,spacing,origin,direction=assert_same_geometry(r['image'],r['mask'])
            _axes(direction) # orientation supported by training geometry
            if not np.isfinite(image).all(): raise ValueError('nonfinite image values')
            labels=np.unique(mask)
            unknown=set(map(int,labels)) - ({0}|set(schema))
            if unknown: raise ValueError(f'Labels absent from label_schema.json: {sorted(unknown)}')
            if len(labels)==1 and labels[0]==0: raise ValueError('empty reference mask')
            for lbl in labels:
                if int(lbl)>0:label_counts[int(lbl)]+=1;kinds[schema[int(lbl)]]+=1
            shape_counts[tuple(image.shape)]+=1
        except Exception as exc: issues.append({'series_id':r['series_id'],'error':repr(exc)})
    report={'series':len(rec),'patients':len(testids),'modalities':dict(Counter(r['modality_designation'] for r in rec)),
            'trainval_patients':len(trainval),'patient_overlap':overlap,'schema_labels':len(schema),
            'present_label_counts':dict(label_counts),'present_structure_counts':dict(kinds),'errors':issues,
            'series_ids':[r['series_id'] for r in rec], 'split_sha256':sha(split_file),'schema_sha256':sha(schema_file)}
    if issues: raise ValueError(f'{len(issues)} test series failed audit; first: {issues[:3]}')
    return rec,schema,report

def surfaces(gt,pred,spacing_xyz, direction, tolerance_mm):
    """Symmetric physical surface distance, NN face-boundary voxel-centre approximation.

    For anisotropic grids results approximate mesh surfaces; explicitly document this.
    """
    from scipy import ndimage as ndi
    if not gt.any() and not pred.any():return 0.,0.,1.
    if not gt.any() or not pred.any():return float('nan'),float('nan'),0.
    # ITK x,y,z spacing => NumPy z,y,x. Axis permutation/reflection preserves distances.
    spacing=tuple(float(v) for v in spacing_xyz[::-1])
    kernel=ndi.generate_binary_structure(3,1)
    gsurf=gt & ~ndi.binary_erosion(gt,structure=kernel,border_value=0)
    psurf=pred & ~ndi.binary_erosion(pred,structure=kernel,border_value=0)
    d_to_p=ndi.distance_transform_edt(~psurf,sampling=spacing)[gsurf]
    d_to_g=ndi.distance_transform_edt(~gsurf,sampling=spacing)[psurf]
    combined=np.concatenate((d_to_p,d_to_g))
    return (float(np.percentile(combined,95)),float(combined.mean()),
            float(((d_to_p<=tolerance_mm).sum()+(d_to_g<=tolerance_mm).sum())/(len(d_to_p)+len(d_to_g))))

def score_one(gt,pred,spacing_xyz,direction,tolerance_mm):
    gt=gt.astype(bool);pred=pred.astype(bool)
    a=int(gt.sum()); b=int(pred.sum()); tp=int((gt & pred).sum()); voxel_volume=float(np.prod(spacing_xyz))
    dice=(2*tp/(a+b)) if a+b else 1.
    iou=(tp/(a+b-tp)) if a+b-tp else 1.
    hd,assd,sd=surfaces(gt,pred,spacing_xyz,direction,tolerance_mm)
    return {'gt_voxels':a,'pred_voxels':b,'dice':dice,'iou':iou,'hd95_mm':hd,'assd_mm':assd,'surface_dice':sd,
            'gt_volume_mm3':a*voxel_volume,'pred_volume_mm3':b*voxel_volume,
            'volume_error_pct':100*(b-a)/a if a else float('nan')}

def restore_native(mask_hw, geo, volume, label):
    """Insert 2D displayed plane into native array order, exact inversion of extract_sagittal."""
    raw=mask_hw
    if geo.flip_rows:raw=np.flip(raw,axis=0)
    if geo.flip_columns:raw=np.flip(raw,axis=1)
    remaining=[i for i in range(3) if i!=geo.sagittal_axis_zyx]
    raw=np.transpose(raw,( [geo.row_axis_zyx,geo.column_axis_zyx].index(remaining[0]),
                            [geo.row_axis_zyx,geo.column_axis_zyx].index(remaining[1]) ))
    plane=[slice(None)]*3;plane[geo.sagittal_axis_zyx]=geo.slice_index
    # First matching instance gets preference in overlaps, with stable ascending label ordering.
    v=volume[tuple(plane)];v[(raw.astype(bool)) & (v==0)]=label

def predict_volume(model,rec,schema,device):
    import torch
    import torch.nn.functional as F
    from skimage.transform import resize
    from mri_preprocessing import assert_same_geometry,extract_sagittal,normalize_medsam,resize_labels,_axes
    from box_prompts import reference_box,to_model_box
    image,gt,spacing,origin,direction=assert_same_geometry(rec['image'],rec['mask'])
    pred=np.zeros(gt.shape,dtype=np.int32)
    sag,*_= _axes(direction)
    present_labels=sorted(set(map(int,np.unique(gt))) & set(schema))
    with torch.inference_mode():
        for index in range(gt.shape[sag]):
            plane_img,geo=extract_sagittal(image,spacing_xyz=spacing,origin_lps=origin,direction_xyz=direction,slice_index=index)
            plane_gt,_=extract_sagittal(gt,spacing_xyz=spacing,origin_lps=origin,direction_xyz=direction,slice_index=index)
            slice_labels=[int(x) for x in np.unique(plane_gt) if int(x) in schema]
            if not slice_labels:continue
            tensor=normalize_medsam(plane_img,1024).unsqueeze(0).to(device)
            embedding=model.image_encoder(tensor)  # once per sagittal slice
            pe=model.prompt_encoder.get_dense_pe()
            resized=resize_labels(plane_gt,1024)
            for label in sorted(slice_labels):
                prompt=reference_box(plane_gt,label,padding=0)
                if prompt is None: continue
                box=to_model_box(prompt,geo,device=device).reshape(1,1,4)
                # Match training's union correction when nearest-neighbor resize extends beyond box.
                support=np.nonzero(resized==label)
                if len(support[0])==0:raise ValueError(f'Resizing erased label {label} in {rec["series_id"]} slice {index}')
                bb=box.reshape(4).clone()
                bb[0]=min(float(bb[0]),float(support[1].min()))
                bb[1]=min(float(bb[1]),float(support[0].min()))
                bb[2]=max(float(bb[2]),float(support[1].max()+1))
                bb[3]=max(float(bb[3]),float(support[0].max()+1))
                bb=bb.clamp(0,1024).reshape(1,1,4)
                sparse,dense=model.prompt_encoder(points=None,boxes=bb,masks=None)
                low,_=model.mask_decoder(image_embeddings=embedding,image_pe=pe,
                    sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
                # Match training: 256 -> 1024 logits first, threshold at 0, then resample to native.
                logits=F.interpolate(low,size=(1024,1024),mode='bilinear',align_corners=False)
                # Native resampling follows inverse_probabilities used for geometry mapping.
                native=F.interpolate(logits,size=geo.native_hw,mode='bilinear',align_corners=False)
                binary=(native[0,0]>=0).cpu().numpy()
                restore_native(binary,geo,pred,label)
    return pred,gt,spacing,origin,direction,present_labels

def load_model(pretrained,run_ckpt,device,split_sha):
    import torch
    from medsam_setup import load_medsam
    model=load_medsam(pretrained,device=device)
    # Trusted local training checkpoint only; weights_only=False needed for NumPy RNG objects.
    payload=torch.load(run_ckpt,map_location='cpu',weights_only=False)
    if 'decoder' not in payload or 'provenance' not in payload:
        raise ValueError(f'Not a locked MedSAM decoder checkpoint: {run_ckpt}')
    prov=payload['provenance']
    if prov.get('pretrained_sha256')!=sha(pretrained):raise ValueError('Pretrained checkpoint hash mismatch')
    # split_sha is the manifest inventory digest, not the JSON file digest.
    if prov.get('split_inventory_sha256')!=split_sha:raise ValueError('Round 2 split inventory mismatch')
    model.mask_decoder.load_state_dict(payload['decoder'],strict=True)
    model.eval()
    return model,{'best_epoch':payload.get('best_epoch'),'best_validation_dice':payload.get('best_score'),
                  'checkpoint_sha256':sha(run_ckpt)}

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0]) if rows else fields
    if not fields:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def summarize(rows,keynames):
    # Only GT-present label rows; no GT-absent score inflation.
    keys=['dice','iou','hd95_mm','assd_mm','surface_dice','volume_error_pct']
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keynames)].append(r)
    out=[]
    for group,rr in sorted(groups.items()):
        d={k:v for k,v in zip(keynames,group)}
        d['n_labels']=len(rr);d['n_scans']=len({r['series_id'] for r in rr});d['n_patients']=len({r['patient_id'] for r in rr})
        for m in keys:
            vals=np.array([float(r[m]) for r in rr],float);vals=vals[np.isfinite(vals)]
            d[f'{m}_mean']=float(vals.mean()) if len(vals) else float('nan')
            d[f'{m}_sd']=float(vals.std(ddof=1)) if len(vals)>1 else float('nan')
        out.append(d)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-dir',default='./TestData');p.add_argument('--split',default='./manifests_round2/patient_train_val.json')
    p.add_argument('--schema',default='./label_schema.json');p.add_argument('--expected',type=int,default=87)
    p.add_argument('--pretrained',default='./checkpoints/medsam_vit_b.pth')
    p.add_argument('--runs',default='./runs_train2');p.add_argument('--out',default='./heldout_evaluation')
    p.add_argument('--device',default='cuda:0');p.add_argument('--surface-tolerance-mm',type=float,default=1.0)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--one-run',default='',help='Optional K5_seed1 etc for debugging')
    args=p.parse_args();out=Path(args.out)
    if args.surface_tolerance_mm<=0:raise ValueError('Surface tolerance must be positive')
    records,schema,report=audit(args.test_dir,args.split,args.schema,args.expected)
    atomic_json(out/'test_audit.json',report)
    print('AUDIT PASS:',len(records),'scans,',report['patients'],'patients,',report['modalities'],flush=True)
    if args.audit_only:return
    from patient_manifests import _digest
    import torch,SimpleITK as sitk
    from mri_preprocessing import read_mha
    split=json.loads(Path(args.split).read_text())
    runs=[args.one_run] if args.one_run else [f'K{k}_seed{s}' for k in (5,10,20) for s in (1,2,3)]
    if not Path(args.pretrained).is_file():raise FileNotFoundError(args.pretrained)
    # Verify every run's best checkpoint before processing any test scan.
    for r in runs:
        if not re.fullmatch(r'K(?:5|10|20)_seed[123]',r):raise ValueError(f'Invalid experiment name: {r}')
        if not (Path(args.runs)/r/'best.pth').is_file():raise FileNotFoundError(f'Missing best checkpoint for {r}')
    for run in runs:
        directory=out/run
        if (directory/'COMPLETE.json').exists():raise FileExistsError(f'Results already finalized: {directory}')
        directory.mkdir(parents=True,exist_ok=True)
        ckpt=Path(args.runs)/run/'best.pth'
        model,metadata=load_model(args.pretrained,ckpt,args.device,split['inventory_sha256'])
        print('RUN',run,'best epoch',metadata['best_epoch'],flush=True)
        rows=[]; series_stats=[]
        for n,rec in enumerate(records,1):
            pred,gt,spacing,origin,direction,present=predict_volume(model,rec,schema,args.device)
            if args.save_predictions:
                reference=sitk.ReadImage(rec['image']);res=sitk.GetImageFromArray(pred)
                res.CopyInformation(reference)
                dest=directory/'predictions';dest.mkdir(exist_ok=True)
                sitk.WriteImage(res,str(dest/(rec['series_id']+'.mha')),useCompression=True)
            for label in present:
                scores=score_one(gt==label,pred==label,spacing,direction,args.surface_tolerance_mm)
                rows.append({'run':run,'patient_id':rec['patient_id'],'series_id':rec['series_id'],
                             'modality':rec['modality_designation'],'label':label,'structure':schema[label],
                             'status':'gt_present',**scores})
            # Detection accounting for GT-present instances only; oracle protocol is NOT automatic detection.
            series_stats.append({'series_id':rec['series_id'],'patient_id':rec['patient_id'],
                                 'modality':rec['modality_designation'],'n_gt_labels':len(present),
                                 'n_pred_labels_detected':sum((pred==label).any() for label in present)})
            write_csv(directory/'per_scan_per_label.csv',rows,FIELDS)
            write_csv(directory/'scan_inventory.csv',series_stats)
            print(f'{run} [{n}/{len(records)}] {rec["series_id"]} labels={len(present)}',flush=True)
        write_csv(directory/'per_scan_summary.csv',summarize(rows,['run','series_id','patient_id','modality']))
        write_csv(directory/'per_patient_summary.csv',summarize(rows,['run','patient_id']))
        write_csv(directory/'per_modality_summary.csv',summarize(rows,['run','modality']))
        write_csv(directory/'per_structure_summary.csv',summarize(rows,['run','structure']))
        write_csv(directory/'per_label_summary.csv',summarize(rows,['run','label','structure']))
        write_csv(directory/'overall_summary.csv',summarize(rows,['run']))
        atomic_json(directory/'COMPLETE.json',{'run':run,'series':len(records),'patients':report['patients'],
                    'labels_evaluated':len(rows),'oracle_prompt':True,'surface_tolerance_mm':args.surface_tolerance_mm,
                    'surface_method':'symmetric surface voxel-center EDT, scipy 6-connectivity',**metadata})
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print('HELDOUT EVALUATION COMPLETE',flush=True)
if __name__=='__main__':main()
