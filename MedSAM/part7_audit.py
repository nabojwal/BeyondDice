"""Read-only audit of frozen MedSAM few-shot checkpoints and patient cohorts.
Only run against checkpoints you created/trust: training files contain pickle RNG state.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def audit(root:Path,split_path:Path,pretrained:Path,out:Path):
    split=json.loads(split_path.read_text())
    assign=split['assignments']
    train={k for k,v in assign.items() if v=='train'}
    val={k for k,v in assign.items() if v=='val'}
    if train & val or len(assign)!=210 or len(train)!=168 or len(val)!=42:
        raise ValueError('Unexpected or overlapping patient split')
    source_sha=file_hash(pretrained)
    reports=[]
    for k in (5,10,20):
        for seed in (1,2,3):
            name=f'K{k}_seed{seed}'
            p=root/name/'best.pth'
            if not p.is_file():raise FileNotFoundError(p)
            # PyTorch 2.11: weights_only=True can reject numpy RNG state.
            # Use weights_only=False exclusively for trusted experiment artifacts.
            obj=torch.load(p,map_location='cpu',weights_only=False)
            c=obj.get('config',{}); provenance=obj.get('provenance',{})
            shot=provenance.get('shot_manifest',{})
            if c.get('k')!=k or c.get('seed')!=seed or shot.get('patient_ids') is None:
                raise ValueError(f'{name}: checkpoint/shot identity mismatch')
            if set(shot['patient_ids'])-train or set(shot['patient_ids'])&val or len(shot['patient_ids'])!=k:
                raise ValueError(f'{name}: patient sampling/leakage error')
            if provenance.get('pretrained_sha256')!=source_sha:
                raise ValueError(f'{name}: incompatible pretrained weights')
            if provenance.get('split_inventory_sha256')!=split.get('inventory_sha256'):
                raise ValueError(f'{name}: split fingerprint differs')
            if 'reference-oracle' not in provenance.get('prompt_protocol',''):
                raise ValueError(f'{name}: unsupported prompt provenance')
            if sorted(provenance.get('validation_ids',[]))!=sorted(val):
                raise ValueError(f'{name}: validation cohort differs')
            if not isinstance(obj.get('decoder'),dict) or not obj['decoder']:
                raise ValueError(f'{name}: missing decoder state')
            reports.append({'experiment':name,'k':k,'seed':seed,
                'checkpoint':str(p),'checkpoint_sha256':file_hash(p),
                'epoch':obj.get('epoch'),'best_epoch':obj.get('best_epoch'),
                'best_score':obj.get('best_score'),'pretrained_sha256':source_sha,
                'prompt_protocol':provenance['prompt_protocol'],
                'train_patients':len(shot['patient_ids']), 'validation_patients':len(val),
                'decoder_tensors':len(obj['decoder'])})
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps({'cohort':'training-validation only (not independent test)',
        'split_fingerprint':split['inventory_sha256'],'checkpoints':reports},indent=2))
    print('PASS: nine checkpoint identity/provenance audits; validation cohort is NOT test')
    print(f'Audit: {out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',default='runs');parser.add_argument('--split',default='manifests/patient_train_val.json')
    parser.add_argument('--pretrained',default='checkpoints/medsam_vit_b.pth')
    parser.add_argument('--output',default='part7_outputs/checkpoint_audit.json')
    args=parser.parse_args()
    audit(Path(args.runs),Path(args.split),Path(args.pretrained),Path(args.output))
