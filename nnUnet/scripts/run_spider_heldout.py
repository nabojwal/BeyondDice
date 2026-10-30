"""One-GPU inference from DDP7 best checkpoint followed by held-out evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version
from datetime import datetime, timezone
from evaluate_spider_test import LABELS, sha, read_json, subject_map, evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject-map',type=Path,required=True)
    p.add_argument('--surface-tolerance-mm',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--results-root',type=Path)
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    raw=root/'Models/nnUNet/nnUNet_raw/Dataset012_SPIDERInstances'
    results=args.results_root or root/'Models/nnUNet/nnUNet_results_ddp7'
    trainer='nnUNetTrainerSPIDERInstancesDDP7'; plans='nnUNetPlansSPIDERDDP7'; config='3d_fullres_ddp7'
    model=results/'Dataset012_SPIDERInstances'/f'{trainer}__{plans}__{config}'
    checkpoint=model/'fold_0/checkpoint_best.pth'
    import numpy as np
    import torch
    import SimpleITK
    import surface_distance
    if version('nnunetv2')!='2.8.1': raise RuntimeError('Requires nnunetv2==2.8.1')
    if not os.getenv('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly ONE allocated GPU; physical GPU 0 is restricted')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if not np.isfinite(args.surface_tolerance_mm) or args.surface_tolerance_mm<=0 or args.bootstrap<100:
        raise ValueError('Invalid tolerance or bootstrap count')
    if not checkpoint.is_file() or not (model/'fold_0/checkpoint_final.pth').is_file():
        raise FileNotFoundError('Wait for full training completion: best and final checkpoints required')
    dataset=read_json(raw/'dataset.json')
    if dataset != read_json(model/'dataset.json') or dataset['labels']!=LABELS:
        raise ValueError('Model/raw dataset label mismatch')
    model_plans=read_json(model/'plans.json')
    if model_plans['plans_name']!=plans or model_plans['dataset_name']!='Dataset012_SPIDERInstances':
        raise ValueError('Unexpected model plans')
    from training.nnunet.data.spider_ddp7_contract import validate_contract
    pre=root/'Models/nnUNet/nnUNet_preprocessed/Dataset012_SPIDERInstances'
    validate_contract(pre, model_plans, dataset, root/'training/nnunet/trainers/nnUNetTrainerSPIDERInstances.py')
    manifest=read_json(raw/'spider_source_manifest.json');cases=sorted(manifest['test'])
    if len(cases)!=87 or set(cases)&set(manifest['train']): raise ValueError('Unexpected held-out cases')
    subject_map(args.subject_map,cases)
    if {f.name for f in (raw/'imagesTs').glob('*.mha')}!={c+'_0000.mha' for c in cases}:
        raise ValueError('Missing/extra held-out images')
    if {f.name for f in (raw/'labelsTs').glob('*.mha')}!={c+'.mha' for c in cases}:
        raise ValueError('Missing/extra held-out labels')
    for case in cases:
        if sha(raw/'imagesTs'/f'{case}_0000.mha')!=manifest['test'][case]['image_sha256']:
            raise ValueError(f'Held-out image changed: {case}')
    args.output.mkdir(parents=True,exist_ok=False);args.output=args.output.resolve()
    command=[str(Path(sys.executable).parent/'nnUNetv2_predict'),'-i',str(raw/'imagesTs'),
             '-o',str(args.output/'predictions'),'-d','12','-p',plans,'-c',config,
             '-f','0','-tr',trainer,'-chk','checkpoint_best.pth','-npp','1','-nps','1','-device','cuda']
    env=os.environ.copy()
    env.update(nnUNet_raw=str(raw.parent),nnUNet_preprocessed=str(pre.parent),nnUNet_results=str(results.resolve()),
               nnUNet_extTrainer=str(root/'training/nnunet/trainers'),nnUNet_wandb_enabled='0',
               PYTHONPATH=str(root)+os.pathsep+env.get('PYTHONPATH',''))
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha(checkpoint),
                checkpoint='checkpoint_best.pth',fold=0,trainer=trainer,plans=plans,configuration=config,
                model_plans_sha256=sha(model/'plans.json'),model_dataset_sha256=sha(model/'dataset.json'),
                subject_map_sha256=sha(args.subject_map),surface_tolerance_mm=args.surface_tolerance_mm,
                bootstrap=args.bootstrap,command=command,tta=True,sliding_window_step_size=.5,
                nnunet_version=version('nnunetv2'),torch_version=version('torch'),completion='pending')
    provenance=args.output/'inference_manifest.json';provenance.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(command,env=env,cwd=root,check=True)
    if sha(checkpoint)!=record['checkpoint_sha256']: raise RuntimeError('Checkpoint changed during inference')
    record.update(completion='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    provenance.write_text(json.dumps(record,indent=2)+'\n')
    evaluate(argparse.Namespace(raw=raw,predictions=args.output/'predictions',output=args.output/'metrics',
              subject_map=args.subject_map,surface_tolerance_mm=args.surface_tolerance_mm,
              geometry_tolerance_mm=.001,bootstrap=args.bootstrap,seed=42,inference_manifest=provenance))

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1])); main()
