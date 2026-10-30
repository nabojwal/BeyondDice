"""Part 4: isolated tests plus one real, strict-checkpoint MedSAM optimizer step."""
from __future__ import annotations
import argparse
import hashlib
import random
import unittest
import numpy as np
import torch
from torch import nn
from decoder_adaptation import (configure_decoder_only,parameter_report,decoder_optimizer,
                                 forward_decoder_only,set_decoder_train_mode)
from segmentation_loss import bce_soft_dice


def digest(module:nn.Module)->str:
    """SHA-256 of parameter values, using one CPU tensor at a time."""
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode('utf-8'))
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

class DummyPrompt(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Parameter(torch.ones(1))
    def forward(self,points,boxes,masks):
        b=boxes.shape[0];device=boxes.device
        return torch.ones(b,2,256,device=device)*self.proj,torch.ones(b,256,64,64,device=device)*self.proj
    def get_dense_pe(self):return torch.zeros(1,256,64,64,device=self.proj.device)
class DummyImage(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1))
    def forward(self,x):return torch.nn.functional.adaptive_avg_pool2d(x.mean(1,keepdim=True),64).expand(-1,256,-1,-1)*self.weight
class DummyDecoder(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(0.2))
    def forward(self,*,image_embeddings,image_pe,sparse_prompt_embeddings,dense_prompt_embeddings,multimask_output):
        x=image_embeddings[:,0:1]+dense_prompt_embeddings[:,0:1]
        logits=torch.nn.functional.interpolate(x,size=(256,256),mode='bilinear',align_corners=False)*self.weight
        return logits,logits.mean((2,3))
class DummySam(nn.Module):
    def __init__(self):
        super().__init__();self.image_encoder=DummyImage();self.prompt_encoder=DummyPrompt();self.mask_decoder=DummyDecoder()

class Part4UnitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);self.model=configure_decoder_only(DummySam())
        self.batch={'image':torch.rand(1,3,1024,1024),
                    'box':torch.tensor([[100.,100.,200.,200.]]),
                    'mask':torch.zeros(1,1,1024,1024)}
        self.batch['mask'][:,:,100:200,100:200]=1
    def test_freezing_and_optimizer(self):
        r=parameter_report(self.model)
        self.assertEqual(r['total']['trainable'],1)
        self.assertEqual(r['total']['frozen'],2)
        opt=decoder_optimizer(self.model)
        self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},
                         {id(p) for p in self.model.mask_decoder.parameters()})
    def test_forward_and_loss_and_step(self):
        set_decoder_train_mode(self.model)
        before=(digest(self.model.image_encoder),digest(self.model.prompt_encoder))
        p=self.model.mask_decoder.weight
        old=p.detach().clone()
        opt=decoder_optimizer(self.model,lr=1e-2)
        opt.zero_grad(set_to_none=True)
        out=forward_decoder_only(self.model,self.batch)
        self.assertEqual(out.logits.shape,self.batch['mask'].shape)
        self.assertEqual(out.low_res_logits.shape,(1,1,256,256))
        result=bce_soft_dice(out.logits,self.batch['mask'])
        self.assertTrue(torch.isfinite(result.total).item())
        result.total.backward()
        self.assertIsNotNone(p.grad)
        self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertTrue(all(q.grad is None for m in (self.model.image_encoder,self.model.prompt_encoder) for q in m.parameters()))
        opt.step()
        self.assertFalse(torch.equal(old,p.detach()))
        self.assertEqual(before,(digest(self.model.image_encoder),digest(self.model.prompt_encoder)))
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)
    def test_errors(self):
        with self.assertRaises(ValueError):forward_decoder_only(self.model,{**self.batch,'box':torch.zeros(1,4)})
        with self.assertRaises(ValueError):bce_soft_dice(torch.zeros(1,1,256,256),self.batch['mask'])
        with self.assertRaises(ValueError):bce_soft_dice(torch.zeros(1,1,2,2),torch.full((1,1,2,2),2.))
        with self.assertRaises(ValueError):decoder_optimizer(self.model,lr=0)


def real_step(args:argparse.Namespace)->None:
    from medsam_setup import load_medsam
    from patient_manifests import discover_pairs,prepare_split,fewshot_manifests
    from instance_dataset import read_label_schema,build_instance_index,MedSAMInstanceDataset,create_loader
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    records=discover_pairs(args.images,args.masks,args.patient_regex,args.series_regex)
    split=prepare_split(records,args.split_manifest,seed=args.split_seed)
    selected=fewshot_manifests(split,args.shot_dir,seeds=[args.seed],k_values=(args.k,))[(args.seed,args.k)]['patient_ids']
    entries,counts=build_instance_index(records,split['assignments'],selected,read_label_schema(args.label_schema))
    print('PATIENTS:',selected,'INDEXED EXAMPLES:',counts['samples'])
    dataset=MedSAMInstanceDataset(entries)
    loader=create_loader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,seed=args.seed)
    batch=next(iter(loader))
    print('REAL BATCH PATIENTS:',batch['patient_id'],'INSTANCE IDs:',batch['instance_id'])
    print('INPUT SHAPES:',{k:tuple(batch[k].shape) for k in ('image','mask','box')})
    model=configure_decoder_only(load_medsam(args.checkpoint,args.device))
    set_decoder_train_mode(model)
    report=parameter_report(model)
    print('PARAMETERS:',report)
    device=next(model.parameters()).device
    batch={**batch,**{k:batch[k].to(device) for k in ('image','mask','box')}}
    opt=decoder_optimizer(model,lr=args.lr,weight_decay=args.weight_decay)
    ids={id(p) for g in opt.param_groups for p in g['params']}
    decoder_ids={id(p) for p in model.mask_decoder.parameters()}
    assert ids==decoder_ids and len(ids)==len(decoder_ids),'Optimizer contains incorrect parameters'
    enc_before=digest(model.image_encoder);prompt_before=digest(model.prompt_encoder)
    decoder_before={name:p.detach().clone() for name,p in model.mask_decoder.named_parameters()}
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters()),'Residual grads before backward'
    out=forward_decoder_only(model,batch)
    losses=bce_soft_dice(out.logits,batch['mask'],bce_weight=args.bce_weight,dice_weight=args.dice_weight)
    print('LOGITS:',tuple(out.logits.shape),'LOW RES:',tuple(out.low_res_logits.shape))
    print(f'LOSS BCE={losses.bce.item():.8f} DICE={losses.dice.item():.8f} TOTAL={losses.total.item():.8f}')
    losses.total.backward()
    grad_count=0;grad_norm_sq=0.;grad_max=0.
    for name,p in model.mask_decoder.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise AssertionError(f'Nonfinite gradient: {name}')
            norm=float(torch.linalg.vector_norm(p.grad.detach().float()).item())
            grad_norm_sq+=norm*norm;grad_max=max(grad_max,float(p.grad.detach().abs().max().item()))
            if norm>0:grad_count+=1
    assert grad_count>0 and grad_norm_sq>0,'No nonzero decoder gradients'
    assert all(p.grad is None for m in (model.image_encoder,model.prompt_encoder) for p in m.parameters())
    print(f'DECODER GRADIENTS: nonzero_tensors={grad_count}, l2_norm={grad_norm_sq**.5:.8g}, max_abs={grad_max:.8g}')
    opt.step()
    updated=sum(not torch.equal(p.detach(),decoder_before[name]) for name,p in model.mask_decoder.named_parameters())
    assert updated>0,'Decoder weights did not change'
    assert digest(model.image_encoder)==enc_before,'Image encoder changed'
    assert digest(model.prompt_encoder)==prompt_before,'Prompt encoder changed'
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters()),'Gradients remain after clearing'
    print(f'FROZEN WEIGHTS UNCHANGED: PASS; UPDATED DECODER TENSORS: {updated}; ZERO GRAD CLEAR: PASS')
    print('PART 4 ONE-STEP INTEGRATION: PASS',device)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint');p.add_argument('--images');p.add_argument('--masks')
    p.add_argument('--label-schema');p.add_argument('--split-manifest');p.add_argument('--shot-dir')
    p.add_argument('--patient-regex',default=r'(?P<patient>[^_]+)_t2')
    p.add_argument('--series-regex',default=r'.*_t2$')
    p.add_argument('--split-seed',type=int,default=42);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--k',type=int,choices=(1,5,10,20),default=1)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='auto');p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=0.0)
    p.add_argument('--bce-weight',type=float,default=1.0);p.add_argument('--dice-weight',type=float,default=1.0)
    a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Part4UnitTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if a.checkpoint:
        if not all((a.images,a.masks,a.label_schema,a.split_manifest,a.shot_dir)):
            p.error('Real integration requires --images --masks --label-schema --split-manifest --shot-dir')
        real_step(a)
    else:print('REAL SPIDER INTEGRATION: NOT RUN (provide --checkpoint and dataset arguments)')
