"""Lazy patient-filtered, instance-aware MedSAM examples; no training logic."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mri_preprocessing import assert_same_geometry, extract_sagittal, normalize_medsam, resize_labels, _axes
from box_prompts import reference_box, to_model_box

TYPES={'vertebral_body','intervertebral_disc','spinal_canal'}


def read_label_schema(path: str | Path) -> dict[int,str]:
    """Require explicit per-label anatomical categories; never infer ordinal level."""
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw,dict) or not raw:raise ValueError('Expected nonempty label mapping')
    mapping={int(k):v for k,v in raw.items()}
    if any(k<=0 or v not in TYPES for k,v in mapping.items()):
        raise ValueError(f'Each positive label must map to one of {sorted(TYPES)}')
    return mapping


def build_instance_index(records: list[dict], assignments: dict, patients: list[str],
                         label_mapping: dict[int,str], *, min_pixels: int=1) -> tuple[list[dict],dict]:
    """Index nonempty slice/label pairs from exclusively selected training patients.

    Indexing reads each volume once and does not store full arrays. Missing labels
    are counted; unrelated labels are ignored (never silently relabeled).
    """
    selected=set(patients)
    if len(selected)!=len(patients) or not selected:
        raise ValueError('Patient selection must be unique and nonempty')
    if any(assignments.get(p)!='train' for p in selected):
        raise ValueError('Patient is missing from train split')
    if not label_mapping or any(k<=0 or v not in TYPES for k,v in label_mapping.items()):
        raise ValueError('Explicit valid instance label mapping required')
    if min_pixels<1:raise ValueError('min_pixels must be >=1')
    entries=[];excluded=Counter(); patient_counts=Counter(); structure_counts=Counter()
    included_series=set()
    for record in records:
        if record['patient_id'] not in selected:continue
        if record['series_id'] in included_series:raise ValueError('Duplicate series')
        included_series.add(record['series_id'])
        image, gt, sp,org,direction=assert_same_geometry(record['image'],record['mask'])
        sag,*_= _axes(direction)
        for idx in range(gt.shape[sag]):
            mask_hw,_=extract_sagittal(gt,spacing_xyz=sp,origin_lps=org,
                                     direction_xyz=direction,slice_index=idx)
            sizes=dict(zip(*np.unique(mask_hw,return_counts=True)))
            for label,kind in sorted(label_mapping.items()):
                n=int(sizes.get(label,0))
                if n<min_pixels:
                    excluded['empty' if n==0 else 'below_min_pixels']+=1
                    continue
                entry={**record,'slice_index':idx,'instance_id':label,'structure_type':kind,
                       'native_foreground_pixels':n}
                entries.append(entry)
                patient_counts[record['patient_id']]+=1
                structure_counts[kind]+=1
    report={'samples':len(entries),'per_patient':dict(sorted(patient_counts.items())),
            'per_structure':dict(sorted(structure_counts.items())),
            'excluded':dict(excluded),'selected_patients':sorted(selected),
            'indexed_series':len(included_series)}
    return entries,report


class MedSAMInstanceDataset(Dataset):
    """Returns a single binary target for exactly one explicit integer instance label.

    Each worker lazily loads a volume pair once and reuses that pair for subsequent
    items in that worker. Avoids serializing cached volume arrays between workers.
    """
    def __init__(self, entries: list[dict], *, padding: float=0, model_size:int=1024):
        if not entries:raise ValueError('Dataset contains no eligible positive examples')
        if padding<0 or model_size<=0:raise ValueError('Invalid padding or model size')
        self.entries=list(entries);self.padding=padding;self.model_size=model_size
        self._cache_key=None;self._cache=None
    def __len__(self):return len(self.entries)
    def __getstate__(self):
        s=self.__dict__.copy();s['_cache_key']=None;s['_cache']=None;return s
    def __getitem__(self, i:int)->dict:
        record=self.entries[i]
        key=(record['image'],record['mask'])
        if self._cache_key!=key:
            self._cache=assert_same_geometry(*key)
            self._cache_key=key
        image,seg,sp,org,direction=self._cache
        options=dict(spacing_xyz=sp,origin_lps=org,direction_xyz=direction,
                     slice_index=record['slice_index'],model_size=self.model_size)
        img,geometry=extract_sagittal(image,**options)
        lbl,mask_geometry=extract_sagittal(seg,**options)
        if geometry!=mask_geometry:raise AssertionError('Image/mask geometry mismatch')
        label=int(record['instance_id'])
        native_target=lbl==label
        if not native_target.any():raise ValueError('Positive instance disappeared')
        prompt=reference_box(lbl,label,padding=self.padding)
        if prompt is None:raise ValueError('Empty reference prompt')
        transformed=resize_labels(lbl,self.model_size)==label
        if not transformed.any():raise ValueError('Resampling removed instance; unsupported small target')
        box=to_model_box(prompt,geometry).reshape(4)
        # NN target support must be enclosed even when boundary rounding occurs.
        yy,xx=np.nonzero(transformed)
        native_scaled=box.numpy()
        union=np.array([min(float(native_scaled[0]),float(xx.min())),
                        min(float(native_scaled[1]),float(yy.min())),
                        max(float(native_scaled[2]),float(xx.max()+1)),
                        max(float(native_scaled[3]),float(yy.max()+1))],dtype=np.float32)
        union[[0,2]]=np.clip(union[[0,2]],0,self.model_size)
        union[[1,3]]=np.clip(union[[1,3]],0,self.model_size)
        image_tensor=normalize_medsam(img,self.model_size)
        mask_tensor=torch.from_numpy(transformed.astype(np.float32)[None].copy())
        out={'image':image_tensor,'mask':mask_tensor,'box':torch.from_numpy(union.copy()),
             'patient_id':record['patient_id'],'series_id':record['series_id'],
             'slice_index':int(record['slice_index']),'instance_id':label,
             'structure_type':record['structure_type'],
             'prompt_source':'reference-oracle',
             'geometry':asdict(geometry),'image_path':record['image'],
             'mask_path':record['mask']}
        return out


def collate_instances(batch: list[dict])->dict:
    """Tensor stack only for equal-sized arrays; variable geometries remain dictionaries."""
    return {k:(torch.stack([b[k] for b in batch]) if k in ('image','mask','box')
               else [b[k] for b in batch]) for k in batch[0]}


def _seed_worker(worker_id: int)->None:
    seed=torch.initial_seed() % 2**32
    np.random.seed(seed);random.seed(seed)


def create_loader(dataset:MedSAMInstanceDataset, *,batch_size:int=2,shuffle:bool=True,
                  num_workers:int=0,seed:int=0)->DataLoader:
    if batch_size<1 or num_workers<0:raise ValueError('Invalid DataLoader arguments')
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
                      collate_fn=collate_instances,worker_init_fn=_seed_worker,generator=generator)
