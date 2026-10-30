"""Validation-only oracle-prompt 3D MedSAM evaluation. No training performed.

The 42-person validation cohort is model-selection data, not independent test data.
Identical boxes and native preprocessing for zero-shot and all few-shot checkpoints.
"""
from __future__ import annotations
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from box_prompts import reference_box,to_model_box
from medsam_setup import load_medsam
from mri_preprocessing import (assert_same_geometry,extract_sagittal,normalize_medsam,resize_labels,
                               inverse_probabilities)
from part7_reconstruction import NativeVolume,VolumeAssembler,VALID_LABELS
from part7_metrics import score_instances,detection_summary


def make_model(pretrained,checkpoint,device):
    model=load_medsam(pretrained,device=device)
    if checkpoint is not None:
        obj=torch.load(checkpoint,map_location='cpu',weights_only=False)  # trusted own checkpoints only
        model.mask_decoder.load_state_dict(obj['decoder'],strict=True)
    model.eval()
    return model


@torch.inference_mode()
def infer_instance(model,image_embedding,geo,box_native,device):
    from box_prompts import PromptBox
    prompt=PromptBox(tuple(float(x) for x in box_native),None,'external')
    box=to_model_box(prompt,geo,device=device)
    sparse,dense=model.prompt_encoder(points=None,boxes=box,masks=None)
    low,_=model.mask_decoder(image_embeddings=image_embedding,
       image_pe=model.prompt_encoder.get_dense_pe(),
       sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    logits=F.interpolate(low,size=(1024,1024),mode='bilinear',align_corners=False)
    native=inverse_probabilities(logits.sigmoid(),geo)[0,0].cpu().numpy()
    return native


def infer_volume(model,img,gt,spacing,origin,direction,device,threshold=0.5):
    geometry=NativeVolume(img.shape,spacing,origin,direction)
    assembler=VolumeAssembler(geometry,threshold)
    # Derive plane axis from Part 2, supporting nonidentity axis-aligned orientations.
    _,probe=extract_sagittal(img,spacing_xyz=spacing,origin_lps=origin,
                             direction_xyz=direction,slice_index=0)
    count=img.shape[probe.sagittal_axis_zyx]
    for slice_index in range(count):
        sl,geo=extract_sagittal(img,spacing_xyz=spacing,origin_lps=origin,
                                direction_xyz=direction,slice_index=slice_index)
        gt_sl,_=extract_sagittal(gt,spacing_xyz=spacing,origin_lps=origin,
                                direction_xyz=direction,slice_index=slice_index)
        ids=sorted(set(map(int,np.unique(gt_sl)))-{0})
        if not ids:continue
        if set(ids)-VALID_LABELS:raise ValueError(f'Unexpected GT labels: {set(ids)-VALID_LABELS}')
        x=normalize_medsam(sl)[None].to(device=device)
        embedding=model.image_encoder(x)
        for label in ids:
            # Reference-derived boxes are for oracle-only, not independent inference.
            prompt=reference_box(gt_sl,label,padding=0)
            if prompt is None:continue
            # Match Part 3's training-time box union with nearest-neighbor resized target.
            model_box=to_model_box(prompt,geo,device='cpu').reshape(4).numpy()
            resized=resize_labels(gt_sl,1024)==label
            yy,xx=np.nonzero(resized)
            if not len(xx):
                raise ValueError(f'Instance {label} vanishes under model resizing')
            union=np.array([min(float(model_box[0]),float(xx.min())),
                            min(float(model_box[1]),float(yy.min())),
                            max(float(model_box[2]),float(xx.max()+1)),
                            max(float(model_box[3]),float(yy.max()+1))],dtype=float)
            union[[0,2]]=np.clip(union[[0,2]],0,1024)
            union[[1,3]]=np.clip(union[[1,3]],0,1024)
            native_box=(union[[0,2]]*geo.native_hw[1]/1024).tolist()
            native_y=(union[[1,3]]*geo.native_hw[0]/1024).tolist()
            box_native=(native_box[0],native_y[0],native_box[1],native_y[1])
            native=infer_instance(model,embedding,geo,box_native,device)
            assembler.insert(label,native,geo,{'prompt':'reference-oracle',
                'slice_index':slice_index,'box_native_xyxy':box_native})
    return assembler


def evaluate_validation(args):
    from patient_manifests import discover_pairs,prepare_split
    from mri_preprocessing import read_mha
    records=discover_pairs(args.images,args.masks,args.patient_regex)
    split=prepare_split(records,args.split,seed=42)
    val_ids={p for p,s in split['assignments'].items() if s=='val'}
    if not val_ids:raise ValueError('No validation patients')
    chosen=[r for r in records if r['patient_id'] in val_ids]
    if {r['patient_id'] for r in chosen}!=val_ids:raise ValueError('Validation files absent')
    if args.patient_limit:
        print('SMOKE TEST ONLY: subset of validation cohort, not full evaluation')
        chosen=chosen[:args.patient_limit]
    if args.model=='zero_shot':ckpt=None
    else:
        if args.k not in (5,10,20) or args.seed not in (1,2,3):raise ValueError('Invalid K/seed')
        ckpt=Path(args.runs)/f'K{args.k}_seed{args.seed}'/'best.pth'
        if not ckpt.is_file():raise FileNotFoundError(ckpt)
    model=make_model(args.pretrained,ckpt,args.device)
    exp='zero_shot' if ckpt is None else f'K{args.k}_seed{args.seed}'
    out=Path(args.output)/'validation_oracle'/exp
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for rec in chosen:
        pid=rec['patient_id'];sid=rec['series_id']
        img,gt,sp,org,direc=assert_same_geometry(rec['image'],rec['mask'])
        started=time.monotonic()
        assembler=infer_volume(model,img,gt,sp,org,direc,args.device,args.threshold)
        path=assembler.write_mha(out/f'{sid}.mha')
        metrics=score_instances(gt,assembler.labels,sp)
        detection=detection_summary(metrics)
        for metric in metrics:
            rows.append({'patient_id':pid,'series_id':sid,'experiment':exp,
                         'cohort':'validation (used for checkpoint selection)',
                         'prompt':'reference-oracle','threshold':args.threshold,
                         'checkpoint':str(ckpt) if ckpt else str(args.pretrained),
                         'runtime_sec':time.monotonic()-started,**metric})
        (out/f'{sid}.provenance.json').write_text(json.dumps({
            'patient_id':pid,'series_id':sid,'checkpoint':str(ckpt) if ckpt else None,
            'pretrained':str(args.pretrained),'cohort':'validation (NOT TEST)',
            'prompt_protocol':'reference-oracle','threshold':args.threshold,
            'spacing_xyz':sp,'origin_lps':org,'direction_xyz':direc,
            'labels':sorted(set(map(int,np.unique(assembler.labels)))-{0}),
            'detections':detection,'prediction_mha':str(path),
            'instance_sources':assembler.sources},indent=2,default=str))
        print(f'{exp} patient={pid} series={sid} instances={len(metrics)} '
              f'detection={detection["detection_rate"]} runtime={time.monotonic()-started:.1f}s',flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))
    print('RESULTS:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--images',default='TrainData/images');p.add_argument('--masks',default='TrainData/masks')
    p.add_argument('--split',default='manifests/patient_train_val.json')
    p.add_argument('--pretrained',default='checkpoints/medsam_vit_b.pth')
    p.add_argument('--patient-regex',default=r'(?P<patient>[^_]+)_t2')
    p.add_argument('--runs',default='runs');p.add_argument('--output',default='part7_outputs')
    p.add_argument('--model',choices=('zero_shot','few_shot'),required=True)
    p.add_argument('--k',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--device',default='cuda:0');p.add_argument('--patient-limit',type=int,default=0)
    args=p.parse_args()
    evaluate_validation(args)
