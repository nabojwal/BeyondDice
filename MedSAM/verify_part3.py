"""Synthetic tests and optional local SPIDER index/model smoke checks."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from patient_manifests import discover_pairs, prepare_split, fewshot_manifests
from instance_dataset import build_instance_index, MedSAMInstanceDataset, create_loader


def fixtures():
    records=[{'patient_id':f'p{i:02d}','series_id':f'p{i:02d}_t2',
              'image':f'/fake/p{i:02d}_t2.mha','mask':f'/fake-mask/p{i:02d}_t2.mha'} for i in range(30)]
    return records


def fake_pair(*paths):
    arr=np.zeros((8,12,3),dtype=np.int16)
    seg=np.zeros_like(arr)
    arr[1:7,2:10,:]=100
    seg[1:4,3:7,1]=201
    seg[4:7,6:10,1]=100
    return arr,seg,(2.,.6,.5),(0.,0.,0.),(1.,0.,0.,0.,1.,0.,0.,0.,1.)


class Part3Tests(unittest.TestCase):
    def test_discovery_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'images';b=Path(tmp)/'masks';a.mkdir();b.mkdir()
            for f in ('01_t2.mha','02_t2.mha','01_t1.mha'):
                (a/f).touch();(b/f).touch()
            rec=discover_pairs(a,b,r'(?P<patient>\d+)_t2')
            self.assertEqual([r['patient_id'] for r in rec],['01','02'])
    def test_patient_disjoint_and_preserve_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixtures();p=Path(tmp)/'split.json'
            split=prepare_split(records,p,seed=17)
            self.assertEqual(len(split['assignments']),30)
            self.assertEqual(split,prepare_split(records,p,seed=999))
            self.assertRaises(ValueError,prepare_split,records[:-1],p,seed=17)
            self.assertEqual(set(split['assignments'].values()),{'train','val'})
    def test_nested_reproducible_fewshot_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixtures();sp=prepare_split(records,Path(tmp)/'split.json',seed=0)
            d=fewshot_manifests(sp,Path(tmp)/'shots',seeds=[7,8])
            for seed in [7,8]:
                self.assertTrue(all(set(d[(seed,k)]['patient_ids'])<=set(d[(seed,k2)]['patient_ids'])
                                    for k,k2 in [(1,5),(5,10),(10,20)]))
                self.assertTrue(all(sp['assignments'][p]=='train' for p in d[(seed,20)]['patient_ids']))
            self.assertEqual(d,fewshot_manifests(sp,Path(tmp)/'shots',seeds=[7,8]))
            self.assertRaises(ValueError,fewshot_manifests,sp,Path(tmp)/'bad',seeds=[1],k_values=(999,))
    def test_instance_preservation_loader_and_alignment(self):
        rec=fixtures()[:2]
        assignments={r['patient_id']:'train' for r in rec}
        with patch('instance_dataset.assert_same_geometry',side_effect=fake_pair):
            entries,report=build_instance_index(rec,assignments,list(assignments),
                                               {201:'intervertebral_disc',100:'spinal_canal'})
            self.assertEqual(len(entries),4)
            self.assertEqual(report['per_structure']['intervertebral_disc'],2)
            ds=MedSAMInstanceDataset(entries,model_size=64)
            loader=create_loader(ds,batch_size=2,shuffle=False,seed=5)
            batches=list(loader)
            self.assertEqual(len(batches),2)
            for b in batches:
                self.assertEqual(tuple(b['image'].shape),(2,3,64,64))
                self.assertEqual(tuple(b['mask'].shape),(2,1,64,64))
                self.assertEqual(tuple(b['box'].shape),(2,4))
                self.assertEqual(b['image'].dtype,torch.float32)
                self.assertEqual(b['mask'].dtype,torch.float32)
                self.assertEqual(b['box'].dtype,torch.float32)
                self.assertTrue(torch.isfinite(b['image']).all())
                self.assertGreaterEqual(float(b['image'].min()),0)
                self.assertLessEqual(float(b['image'].max()),1)
                for j in range(2):
                    ys,xs=torch.where(b['mask'][j,0]>0)
                    self.assertGreater(len(xs),0)
                    x0,y0,x1,y1=b['box'][j].tolist()
                    self.assertLessEqual(x0,float(xs.min()));self.assertLessEqual(y0,float(ys.min()))
                    self.assertGreaterEqual(x1,float(xs.max()+1));self.assertGreaterEqual(y1,float(ys.max()+1))
                    self.assertIn(b['instance_id'][j],(100,201))
                    self.assertEqual(b['prompt_source'][j],'reference-oracle')
                    self.assertEqual(b['geometry'][j]['native_hw'],(8,12))
    def test_legacy_three_way_manifest_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixtures()
            path=Path(tmp)/'existing.json'
            # A valid two-way file is immutable, even if the seed changes.
            split=prepare_split(records,path,seed=12)
            initial=path.read_bytes()
            assert prepare_split(records,path,seed=88)==split
            self.assertEqual(path.read_bytes(),initial)
            # Legacy test assignments may not be silently converted.
            payload=json.loads(path.read_text(encoding='utf-8'))
            patient=next(iter(payload['assignments']))
            payload['assignments'][patient]='test'
            path.write_text(json.dumps(payload),encoding='utf-8')
            old_bytes=path.read_bytes()
            with self.assertRaises(ValueError):
                prepare_split(records,path,seed=12)
            self.assertEqual(path.read_bytes(),old_bytes)

    def test_reject_test_patient(self):
        r=fixtures()[:2]
        with self.assertRaises(ValueError):
            build_instance_index(r,{r[0]['patient_id']:'train',r[1]['patient_id']:'test'},
                                 [r[0]['patient_id'],r[1]['patient_id']],{201:'intervertebral_disc'})


def real_audit(args):
    records=discover_pairs(args.images,args.masks,args.patient_regex,args.series_regex)
    split=prepare_split(records,args.split_manifest,seed=args.split_seed)
    selection=fewshot_manifests(split,args.shot_dir,seeds=args.seeds,k_values=tuple(args.ks))
    from instance_dataset import read_label_schema
    labels=read_label_schema(args.label_schema)
    print('DISCOVERED SERIES:',len(records),'PATIENTS:',len(set(r['patient_id'] for r in records)))
    print('SPLIT PATIENT COUNTS:',{s:sum(x==s for x in split['assignments'].values()) for s in ('train','val')})
    for (seed,k),spec in sorted(selection.items()):
        entries,report=build_instance_index(records,split['assignments'],spec['patient_ids'],labels)
        print('SEED',seed,'K',k,'PATIENT_IDS',','.join(spec['patient_ids']))
        print('SAMPLE_COUNTS',json.dumps(report,sort_keys=True))
        if (seed,k)==(args.seeds[0],args.ks[0]) and entries:
            ds=MedSAMInstanceDataset(entries)
            loader=create_loader(ds,batch_size=args.batch_size,num_workers=args.workers,seed=seed)
            for i,b in enumerate(loader):
                print('BATCH',i,'image',tuple(b['image'].shape),'mask',tuple(b['mask'].shape),
                      'box',tuple(b['box'].shape),'patients',b['patient_id'])
                if i>=args.smoke_batches-1:break
            if args.checkpoint:
                from medsam_setup import load_medsam
                model=load_medsam(args.checkpoint,args.device)
                first=ds[0]
                with torch.inference_mode():
                    device=next(model.parameters()).device
                    image_embed=model.image_encoder(first['image'][None].to(device))
                    sparse,dense=model.prompt_encoder(points=None,boxes=first['box'][None,None].to(device),masks=None)
                    logits,_=model.mask_decoder(image_embeddings=image_embed,
                        image_pe=model.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,multimask_output=False)
                assert tuple(logits.shape)==(1,1,256,256) and torch.isfinite(logits).all()
                print('MEDSAM INPUT COMPATIBILITY: PASS',device)
            if args.overlay:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle
                from mri_preprocessing import extract_sagittal
                from box_prompts import reference_box
                output=Path(args.overlay);output.mkdir(parents=True,exist_ok=True)
                # one representative example per structure, preferably from distinct patients
                seen=set();used_patients=set()
                for entry in entries:
                    kind=entry['structure_type']
                    if kind in seen or (entry['patient_id'] in used_patients and len(used_patients)<len(set(e['patient_id'] for e in entries))):continue
                    item=ds[entries.index(entry)]
                    from mri_preprocessing import assert_same_geometry
                    volume,gt,sp,org,direction=assert_same_geometry(entry['image'],entry['mask'])
                    opts=dict(spacing_xyz=sp,origin_lps=org,direction_xyz=direction,
                              slice_index=entry['slice_index'])
                    native,geo=extract_sagittal(volume,**opts)
                    native_lbl,_=extract_sagittal(gt,**opts)
                    ref=reference_box(native_lbl,entry['instance_id'])
                    nb=ref.xyxy
                    img=item['image'][0].numpy();mask=item['mask'][0].numpy()>0
                    box=item['box'].numpy();fig,ax=plt.subplots(1,2,figsize=(12,6))
                    ax[0].imshow(native,cmap='gray')
                    ax[0].contour((native_lbl==entry['instance_id']).astype(float),levels=[.5],colors=['red'])
                    ax[0].add_patch(Rectangle((nb[0],nb[1]),nb[2]-nb[0],nb[3]-nb[1],fill=False,edgecolor='cyan'))
                    ax[0].set_title('Native MRI / mask / oracle box')
                    ax[1].imshow(img,cmap='gray',vmin=0,vmax=1)
                    ax[1].contour(mask.astype(float),levels=[.5],colors=['red'])
                    ax[1].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='cyan'))
                    ax[1].set_title('MedSAM 1024 / mask / oracle box')
                    fig.suptitle(f"{entry['patient_id']} | {kind} | label {entry['instance_id']}")
                    fig.tight_layout();p=output/f"{kind}_{entry['patient_id']}.png";fig.savefig(p,dpi=120);plt.close(fig)
                    print('QC OVERLAY',p);seen.add(kind);used_patients.add(entry['patient_id'])
                    if len(seen)==3:break

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--images');parser.add_argument('--masks')
    parser.add_argument('--patient-regex',default=r'(?P<patient>[^_]+)_t2')
    parser.add_argument('--series-regex',default=r'(?i)^(.+)_t2$')
    parser.add_argument('--split-manifest',default='manifests/patient_split.json')
    parser.add_argument('--shot-dir',default='manifests/fewshot')
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--ks',nargs='+',type=int,default=[1,5,10,20])
    parser.add_argument('--label-schema')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--smoke-batches',type=int,default=2)
    parser.add_argument('--checkpoint');parser.add_argument('--device',default='cpu')
    parser.add_argument('--overlay')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Part3Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():sys.exit(1)
    if args.images or args.masks:
        if not (args.images and args.masks and args.label_schema):parser.error('Real audit needs --images, --masks, --label-schema')
        real_audit(args)
