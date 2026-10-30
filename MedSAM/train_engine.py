"""Part 5: decoder-only MedSAM training and patient-level oracle-box validation."""
from __future__ import annotations
import csv, hashlib, json, os, random, time, warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from decoder_adaptation import (configure_decoder_only, set_decoder_train_mode,
                                 decoder_optimizer, forward_decoder_only)
from segmentation_loss import bce_soft_dice

@dataclass
class RunConfig:
    images: str = './TrainData/images'
    masks: str = './TrainData/masks'
    label_schema: str = './label_schema.json'
    split_manifest: str = './manifests/patient_train_val.json'
    shot_dir: str = './manifests/fewshot_train_val'
    checkpoint: str = './checkpoints/medsam_vit_b.pth'
    patient_regex: str = r'(?P<patient>[^_]+)_t2'
    seed: int = 42
    k: int = 1
    epochs: int = 3
    batch_size: int = 1
    workers: int = 0
    lr: float = 1e-5
    weight_decay: float = 0.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    accum_steps: int = 1
    clip_norm: float = 0.0
    val_frequency: int = 1
    patience: int = 0
    device: str = 'cpu'
    amp: bool = False
    deterministic: bool = True
    experiment: str = 'part5_smoke'
    output_root: str = './runs'
    train_limit: int = 4
    val_patients_limit: int = 2
    val_examples_per_patient: int = 2
    resume: str = ''
    split_seed: int = 42
    wandb_mode: str = 'disabled'  # online | offline | disabled
    wandb_project: str = 'MedSAM-SPIDER-FewShot'
    wandb_group: str = 'DecoderOnly-T2'
    wandb_log_every: int = 20  # optimizer steps, plus epoch-end
    wandb_system_stats: bool = True

def check_config(c: RunConfig) -> None:
    if any(x < 1 for x in (c.epochs,c.batch_size,c.accum_steps,c.val_frequency,c.k)):
        raise ValueError('epochs/batch_size/accum_steps/val_frequency/k must be positive')
    if min(c.train_limit,c.val_patients_limit,c.val_examples_per_patient)<0:
        raise ValueError('Sample limits must be >= 0; zero means unlimited')
    if c.wandb_mode not in ('online','offline','disabled') or c.wandb_log_every < 1:
        raise ValueError('Invalid W&B mode or logging frequency')
    if c.workers < 0 or c.patience < 0 or c.clip_norm < 0 or c.lr <= 0 or c.weight_decay < 0:
        raise ValueError('Invalid optimization configuration')
    if c.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if c.amp and not c.device.startswith('cuda'):
        raise ValueError('AMP here requires CUDA')

def seed_all(seed:int, deterministic:bool=True)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark=False

def sha256(path:str)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest_parameters(module:torch.nn.Module)->str:
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_capture()->dict:
    result={'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
    if torch.cuda.is_available():result['cuda']=torch.cuda.get_rng_state_all()
    return result

def rng_restore(s:dict)->None:
    torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python'])
    if 'cuda' in s and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])

def patient_metric(rows:list[dict])->dict:
    """Mean instance Dice within each patient; mean of patient means is primary."""
    group=defaultdict(list);structures=defaultdict(lambda:defaultdict(list))
    for r in rows:
        p=str(r['patient_id']);d=float(r['dice']);t=str(r['structure_type'])
        if not np.isfinite(d) or not 0<=d<=1:raise ValueError('Invalid per-instance Dice')
        group[p].append(d);structures[t][p].append(d)
    if not group:raise ValueError('No validation examples')
    per_patient={p:float(np.mean(v)) for p,v in group.items()}
    per_structure={s:float(np.mean([np.mean(v) for v in by_patient.values()])) for s,by_patient in structures.items()}
    return {'patient_mean_dice':float(np.mean(list(per_patient.values()))),
            'per_structure_patient_mean_dice':per_structure,'patient_count':len(group),
            'evaluated_instances':len(rows)}

def batch_to_device(batch:dict,device:torch.device)->dict:
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def dice_per_instance(logits:torch.Tensor, mask:torch.Tensor)->list[float]:
    """Thresholded binary 2D Dice, per prompted instance; empty/empty = 1."""
    prediction=logits.detach().float() >= 0
    target=mask.detach() >= .5
    inter=(prediction & target).sum(dim=(1,2,3)).float()
    denom=prediction.sum(dim=(1,2,3)).float()+target.sum(dim=(1,2,3)).float()
    scores=torch.where(denom>0,2*inter/denom,torch.ones_like(denom))
    return scores.cpu().tolist()

def segmentation_counts(logits:torch.Tensor,mask:torch.Tensor)->dict[str,torch.Tensor]:
    """Per-example binary confusion counts at sigmoid threshold 0.5 (logit 0)."""
    prediction=logits.detach().float()>=0
    truth=mask.detach()>=0.5
    dims=(1,2,3)
    return {'tp':(prediction & truth).sum(dims).float(),
            'fp':(prediction & ~truth).sum(dims).float(),
            'fn':(~prediction & truth).sum(dims).float(),
            'tn':(~prediction & ~truth).sum(dims).float()}

def segmentation_scores(counts:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
    tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
    ones=torch.ones_like(tp)
    dice=torch.where(2*tp+fp+fn>0,2*tp/(2*tp+fp+fn).clamp_min(1),ones)
    iou=torch.where(tp+fp+fn>0,tp/(tp+fp+fn).clamp_min(1),ones)
    precision=torch.where(tp+fp>0,tp/(tp+fp).clamp_min(1),torch.where(fn==0,ones,torch.zeros_like(tp)))
    recall=torch.where(tp+fn>0,tp/(tp+fn).clamp_min(1),ones)
    return {'dice':dice,'iou':iou,'precision':precision,'recall':recall,
            'accuracy':(tp+tn)/(tp+fp+fn+tn).clamp_min(1),
            'sensitivity':recall,
            'specificity':torch.where(tn+fp>0,tn/(tn+fp).clamp_min(1),ones)}

class SafeWandb:
    """Optional logger; any W&B failure leaves local CSV/JSON authoritative."""
    def __init__(self,cfg:RunConfig,root:Path,metadata:dict):
        self.run=None
        if cfg.wandb_mode=='disabled':return
        try:
            import wandb
            self.run=wandb.init(project=cfg.wandb_project,group=cfg.wandb_group,
                name=cfg.experiment,dir=str(root),mode=cfg.wandb_mode,
                id=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16],
                resume='allow' if cfg.resume else 'never',
                config={**asdict(cfg),**metadata},
                tags=['T2-designated','ViT-B','decoder-only',f'K{cfg.k}',f'seed{cfg.seed}'],
                settings=wandb.Settings(x_disable_stats=not cfg.wandb_system_stats))
            wandb.define_metric('epoch')
            wandb.define_metric('train/*',step_metric='epoch')
            wandb.define_metric('val/*',step_metric='epoch')
        except Exception as exc:
            warnings.warn(f'W&B initialization failed; local logging continues: {exc}')
            self.run=None
    def log(self,row:dict,step:int|None=None):
        if self.run is None:return
        try:
            self.run.log(row,step=step)
        except Exception as exc:
            warnings.warn(f'W&B logging failed; local logging continues: {exc}')
    def finish(self):
        if self.run is not None:
            try:self.run.finish()
            except Exception as exc:warnings.warn(f'W&B finish failed: {exc}')

def gpu_diagnostics(device:torch.device)->dict:
    if device.type!='cuda':return {}
    result={'gpu/allocated_gib':torch.cuda.memory_allocated(device)/2**30,
            'gpu/reserved_gib':torch.cuda.memory_reserved(device)/2**30,
            'gpu/max_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
            'gpu/max_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30}
    try:
        import pynvml
        pynvml.nvmlInit()
        index=device.index if device.index is not None else torch.cuda.current_device()
        handle=pynvml.nvmlDeviceGetHandleByIndex(index)
        utilization=pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory=pynvml.nvmlDeviceGetMemoryInfo(handle)
        result.update({'gpu/utilization_pct':utilization.gpu,
                       'gpu/memory_controller_pct':utilization.memory,
                       'gpu/device_used_gib':memory.used/2**30})
        try:result['gpu/temperature_c']=pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU)
        except Exception:pass
        try:result['gpu/power_w']=pynvml.nvmlDeviceGetPowerUsage(handle)/1000
        except Exception:pass
    except Exception:pass  # optional; W&B itself also records system statistics
    return result

def _autocast(enabled:bool):
    return torch.autocast(device_type='cuda',dtype=torch.float16,enabled=enabled)

def train_epoch(model, loader, optimizer, scaler, cfg:RunConfig, device:torch.device,logger=None,epoch:int=0,global_step:int=0)->dict:
    set_decoder_train_mode(model);optimizer.zero_grad(set_to_none=True)
    sums=np.zeros(3,dtype=np.float64);n=0;norms=[];steps=0
    total_batches=len(loader)
    metric_sums=defaultdict(float);structure_sums=defaultdict(lambda:defaultdict(float));structure_counts=defaultdict(int)
    for ix,batch in enumerate(loader):
        batch=batch_to_device(batch,device);size=batch['image'].shape[0]
        with _autocast(cfg.amp):
            pred=forward_decoder_only(model,batch)
            loss=bce_soft_dice(pred.logits,batch['mask'],bce_weight=cfg.bce_weight,dice_weight=cfg.dice_weight)
        if not torch.isfinite(loss.total):raise FloatingPointError(f'Nonfinite loss batch {ix}')
        # Scale by examples in accumulation window (handles short final window).
        start=(ix//cfg.accum_steps)*cfg.accum_steps
        group_count=min(cfg.accum_steps,total_batches-start)
        scaler.scale(loss.total/group_count).backward()
        sums += np.asarray([loss.bce.item(),loss.dice.item(),loss.total.item()])*size;n+=size
        with torch.no_grad():
            metric_batch=segmentation_scores(segmentation_counts(pred.logits,batch['mask']))
            for key,values in metric_batch.items():
                metric_sums[key]+=values.sum().item()
                for j,kind in enumerate(batch['structure_type']):
                    structure_sums[str(kind)][key]+=values[j].item()
            for kind in batch['structure_type']:structure_counts[str(kind)]+=1
        if (ix+1)%cfg.accum_steps==0 or ix+1==total_batches:
            scaler.unscale_(optimizer)
            grads=[p.grad for p in model.mask_decoder.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError('Missing or nonfinite decoder gradients')
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])).item()
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
            norms.append(norm)
            if cfg.clip_norm:clip_grad_norm_(model.mask_decoder.parameters(),cfg.clip_norm,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
            if logger and ((global_step+steps)%cfg.wandb_log_every==0):
                row={'train/step_loss_total':float(loss.total.item()),
                     'train/step_bce':float(loss.bce.item()),
                     'train/step_dice_loss':float(loss.dice.item()),
                     'train/step_gradient_norm':float(norm),
                     'train/lr':float(optimizer.param_groups[0]['lr']),
                     'train/examples_seen_epoch':n,'epoch_progress':epoch+(ix+1)/total_batches,
                     **gpu_diagnostics(device)}
                logger.log(row,step=global_step+steps)
    result={'train_bce':sums[0]/n,'train_dice_loss':sums[1]/n,'train_total':sums[2]/n,
            'gradient_norm_mean':float(np.mean(norms)),'samples':n,'optimizer_steps':steps}
    result.update({f'train_{key}':value/n for key,value in metric_sums.items()})
    result['per_structure_train_metrics']={kind:{key:v/structure_counts[kind] for key,v in measurements.items()}
                                          for kind,measurements in structure_sums.items()}
    return result

@torch.no_grad()
def validate(model,loader,cfg:RunConfig,device:torch.device)->dict:
    model.image_encoder.eval();model.prompt_encoder.eval();model.mask_decoder.eval()
    sums=np.zeros(3,dtype=np.float64);n=0;rows=[]
    for batch in loader:
        batch=batch_to_device(batch,device);bs=batch['image'].shape[0]
        with _autocast(cfg.amp):
            pred=forward_decoder_only(model,batch)
            loss=bce_soft_dice(pred.logits,batch['mask'],bce_weight=cfg.bce_weight,dice_weight=cfg.dice_weight)
        if not torch.isfinite(loss.total):raise FloatingPointError('Nonfinite validation loss')
        sums+=np.asarray([loss.bce.item(),loss.dice.item(),loss.total.item()])*bs;n+=bs
        scores=dice_per_instance(pred.logits,batch['mask'])
        measures=segmentation_scores(segmentation_counts(pred.logits,batch['mask']))
        for j,(pid,kind,d) in enumerate(zip(batch['patient_id'],batch['structure_type'],scores)):
            rows.append({'patient_id':pid,'structure_type':kind,'dice':d,
                         **{key:values[j].item() for key,values in measures.items() if key!='dice'}})
    scores=patient_metric(rows)
    for key in ('iou','precision','recall','accuracy','specificity'):
        per_patient=defaultdict(list)
        for r in rows:per_patient[str(r['patient_id'])].append(r[key])
        scores[f'patient_mean_{key}']=float(np.mean([np.mean(v) for v in per_patient.values()]))
    return {'val_bce':sums[0]/n,'val_dice_loss':sums[1]/n,'val_total':sums[2]/n,**scores}

def atomic_save(obj:dict,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp);os.replace(tmp,path)

def save_checkpoint(path:Path,model,optimizer,scaler,cfg,epoch,global_step,best,best_epoch,patience_count,provenance):
    atomic_save({'schema_version':1,'decoder':model.mask_decoder.state_dict(),
        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
        'epoch':epoch,'global_step':global_step,'best_score':best,'best_epoch':best_epoch,
        'patience_count':patience_count,'config':asdict(cfg),'rng':rng_capture(),
        'provenance':provenance,'torch_version':torch.__version__,
        'numpy_version':np.__version__},path)

def read_checkpoint(path:str|Path,model,optimizer,scaler,provenance:dict)->dict:
    # weights_only=False is required for NumPy RNG state; use only trusted local checkpoints.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['provenance']!=provenance:
        raise ValueError('Resume provenance mismatch: split, K-shot, checkpoint or prompt protocol')
    model.mask_decoder.load_state_dict(payload['decoder'],strict=True)
    optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
    rng_restore(payload['rng'])
    return payload

def write_logs(out:Path,rows:list[dict])->None:
    out.mkdir(parents=True,exist_ok=True)
    (out/'history.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf8')
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k],dict)))
        with (out/'history.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for r in rows:writer.writerow({k:r.get(k,'') for k in keys})

def run_engine(cfg:RunConfig)->None:
    """Run bounded smoke experiment with persistent epoch checkpoints."""
    check_config(cfg)
    from patient_manifests import discover_pairs,prepare_split,fewshot_manifests
    from instance_dataset import read_label_schema,build_instance_index,MedSAMInstanceDataset,create_loader
    from medsam_setup import load_medsam
    seed_all(cfg.seed,cfg.deterministic)
    root=Path(cfg.output_root)/cfg.experiment
    if root.exists() and not cfg.resume:raise FileExistsError(f'{root} exists; specify --resume for same experiment or choose another --experiment')
    root.mkdir(parents=True,exist_ok=True)
    records=discover_pairs(cfg.images,cfg.masks,cfg.patient_regex)
    split=prepare_split(records,cfg.split_manifest,seed=cfg.split_seed)
    shot=fewshot_manifests(split,cfg.shot_dir,seeds=[cfg.seed],k_values=(cfg.k,))[(cfg.seed,cfg.k)]
    training=shot['patient_ids'];validation=sorted(p for p,v in split['assignments'].items() if v=='val')
    if cfg.val_patients_limit:validation=validation[:cfg.val_patients_limit]
    if set(training)&set(validation):raise ValueError('Patient leakage')
    schema=read_label_schema(cfg.label_schema)
    train_entries,train_report=build_instance_index(records,split['assignments'],training,schema)
    # Validation indexer requires 'train' patients; use a validated isolated assignment
    # view only for *indexing* validation targets, not for sampling or splitting.
    val_view={**split['assignments'],**{p:'train' for p in validation}}
    val_entries,val_report=build_instance_index(records,val_view,validation,schema)
    if cfg.train_limit:train_entries=train_entries[:cfg.train_limit]
    selected=[];per_patient=defaultdict(int)
    for entry in val_entries:
        p=entry['patient_id']
        if cfg.val_examples_per_patient==0 or per_patient[p]<cfg.val_examples_per_patient:
            selected.append(entry);per_patient[p]+=1
    if len(per_patient)!=len(validation):raise ValueError('Validation patient without eligible instances')
    train_data=MedSAMInstanceDataset(train_entries)
    val_data=MedSAMInstanceDataset(selected)
    # Recreate train shuffle each epoch from seed+epoch; no persistent sampler state needed.
    val_loader=create_loader(val_data,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.workers,seed=cfg.seed)
    model=load_medsam(cfg.checkpoint,device=cfg.device)
    configure_decoder_only(model)
    device=next(model.mask_decoder.parameters()).device
    optimizer=decoder_optimizer(model,lr=cfg.lr,weight_decay=cfg.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp)
    prompt_protocol='reference-oracle boxes extracted from validation GT; 2D instance Dice, per-instance -> patient mean -> cohort mean'
    provenance={'pretrained_sha256':sha256(cfg.checkpoint),'split_inventory_sha256':split['inventory_sha256'],
                'shot_manifest':shot,'train_ids':training,'validation_ids':validation,
                'train_selected':[(e['patient_id'],e['series_id'],e['slice_index'],e['instance_id']) for e in train_entries],
                'validation_selected':[(e['patient_id'],e['series_id'],e['slice_index'],e['instance_id']) for e in selected],
                'prompt_protocol':prompt_protocol,'metric':'mean over per-patient means of binary 2D Dice at threshold 0.5'}
    config_file=root/'configuration.json'
    config_normal=asdict(cfg);config_normal.pop('resume')
    if cfg.resume:
        old=json.loads(config_file.read_text(encoding='utf8'))
        if {k:v for k,v in old.items() if k!='epochs'}!={k:v for k,v in config_normal.items() if k!='epochs'}:
            raise ValueError('Configuration mismatch on resume (only epochs may increase)')
    else:
        config_file.write_text(json.dumps(config_normal,indent=2),encoding='utf8')
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    start=1;steps=0;best=-1.;best_epoch=0;stall=0
    if cfg.resume:
        ckpt=read_checkpoint(cfg.resume,model,optimizer,scaler,provenance)
        start=ckpt['epoch']+1;steps=ckpt['global_step'];best=ckpt['best_score'];best_epoch=ckpt['best_epoch'];stall=ckpt['patience_count']
    history_file=root/'history.json'
    history=json.loads(history_file.read_text(encoding='utf8')) if cfg.resume else []
    logger=SafeWandb(cfg,root,{'pretrained_sha256':provenance['pretrained_sha256'],
        'split_inventory_sha256':provenance['split_inventory_sha256'],
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_patients':len(training),'validation_patients':len(validation),
        'train_examples':len(train_entries),'validation_examples':len(selected)})
    frozen_before=(digest_parameters(model.image_encoder),digest_parameters(model.prompt_encoder))
    for epoch in range(start,cfg.epochs+1):
        t0=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        train_loader=create_loader(train_data,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.workers,seed=cfg.seed+epoch)
        train_stats=train_epoch(model,train_loader,optimizer,scaler,cfg,device,logger,epoch,steps);steps+=train_stats['optimizer_steps']
        line={'epoch':epoch,**train_stats,'lr':optimizer.param_groups[0]['lr']}
        if epoch%cfg.val_frequency==0:
            val_stats=validate(model,val_loader,cfg,device);line.update(val_stats)
            score=val_stats['patient_mean_dice']
            if score>best:
                best=score;best_epoch=epoch;stall=0
                save_checkpoint(root/'best.pt',model,optimizer,scaler,cfg,epoch,steps,best,best_epoch,stall,provenance)
            else:stall+=1
        line['duration_sec']=time.perf_counter()-t0
        if device.type=='cuda':line['peak_cuda_bytes']=torch.cuda.max_memory_allocated(device)
        line['global_step']=steps
        history.append(line)
        event={'epoch':epoch,'global_step':steps}
        for key,value in line.items():
            if isinstance(value,(int,float)) and key not in ('epoch','global_step'):
                prefix='val/' if key.startswith('val_') or key.startswith('patient_mean_') else 'train/'
                event[prefix+key]=float(value)
        for kind,value in line.get('per_structure_patient_mean_dice',{}).items():
            event[f'val/structure/{kind}/patient_mean_dice']=value
        for kind,measurements in line.get('per_structure_train_metrics',{}).items():
            for metric,value in measurements.items():event[f'train/structure/{kind}/{metric}']=value
        event.update(gpu_diagnostics(device))
        logger.log(event,step=steps)
        save_checkpoint(root/'last.pt',model,optimizer,scaler,cfg,epoch,steps,best,best_epoch,stall,provenance)
        write_logs(root,history)
        print('EPOCH',json.dumps(line,allow_nan=False),flush=True)
        if cfg.patience and stall>=cfg.patience:
            print('EARLY STOP: patience',cfg.patience);break
    frozen_after=(digest_parameters(model.image_encoder),digest_parameters(model.prompt_encoder))
    if frozen_before!=frozen_after:raise AssertionError('Frozen encoder weights changed')
    # Validate that final decoder state round-trips exactly through the last checkpoint.
    sample=batch_to_device(next(iter(val_loader)),device)
    model.mask_decoder.eval()
    with torch.no_grad():
        baseline=forward_decoder_only(model,sample).logits.detach().cpu()
    saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    decoder_cpu={k:v.detach().cpu().clone() for k,v in model.mask_decoder.state_dict().items()}
    model.mask_decoder.load_state_dict(saved['decoder'],strict=True)
    with torch.no_grad():
        reloaded=forward_decoder_only(model,sample).logits.detach().cpu()
    torch.testing.assert_close(baseline,reloaded,rtol=1e-5,atol=1e-6)
    model.mask_decoder.load_state_dict(decoder_cpu,strict=True)
    print('LAST CHECKPOINT PREDICTION RELOAD: PASS')
    print('FROZEN ENCODERS UNCHANGED: PASS')
    print('BEST EPOCH:',best_epoch,'PATIENT-MEAN 2D DICE:',best)
    print('CHECKPOINTS:',root/'best.pt',root/'last.pt')
    plot_curves(root)
    logger.finish()
    print('TRAINING FINISHED')


def plot_curves(output_dir: str | Path) -> None:
    """Plot exported history without requiring matplotlib for model training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_dir);rows=json.loads((out/'history.json').read_text(encoding='utf8'))
    fig,ax=plt.subplots(figsize=(8,5))
    for key in ('train_total','val_total','patient_mean_dice'):
        x=[r['epoch'] for r in rows if key in r]
        y=[r[key] for r in rows if key in r]
        if x:ax.plot(x,y,marker='o',label=key)
    ax.set_xlabel('Epoch');ax.set_ylabel('Value');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
