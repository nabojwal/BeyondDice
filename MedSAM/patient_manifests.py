"""Immutable patient-disjoint manifests and nested patient-level K-shot selections."""
from __future__ import annotations
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def discover_pairs(images: str | Path, masks: str | Path, patient_regex: str,
                   series_regex: str = r'(?i)^(.+)_t2$') -> list[dict]:
    """Only discover T2-designated exact filename pairs; IDs come from explicit regex.

    patient_regex must have named group 'patient'. series_regex must match
    the entire basename (without .mha); by default only *_t2 is accepted.
    """
    images, masks = Path(images), Path(masks)
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError('Both image and mask directories must exist')
    pr = re.compile(patient_regex)
    sr = re.compile(series_regex)
    if 'patient' not in pr.groupindex:
        raise ValueError("patient_regex requires a named (?P<patient>...) group")
    mask_files = {p.name.lower(): p.resolve() for p in masks.glob('*.mha')}
    records = []
    unmatched = []
    for image in sorted(images.glob('*.mha')):
        stem = image.name[:-4]
        if not sr.fullmatch(stem):
            continue
        match = pr.fullmatch(stem)
        if match is None:
            raise ValueError(f'Cannot extract patient identifier from {stem}')
        if image.name.lower() not in mask_files:
            unmatched.append(image.name)
            continue
        records.append(dict(patient_id=match.group('patient'), series_id=stem,
                            image=str(image.resolve()), mask=str(mask_files[image.name.lower()]),
                            modality_designation='t2', modality_source='filename'))
    if unmatched:
        raise ValueError(f'Missing matching masks: {unmatched[:10]}')
    if not records:
        raise ValueError('No matching T2-designated MRI/mask pairs found')
    if len({r['series_id'] for r in records}) != len(records):
        raise ValueError('Duplicate series identifiers')
    return records


def validate_split(records: list[dict], assignments: dict[str, str]) -> None:
    eligible = {r['patient_id'] for r in records}
    if set(assignments) != eligible:
        raise ValueError(f'Split assignment mismatch: missing={sorted(eligible-set(assignments))}, '
                         f'extra={sorted(set(assignments)-eligible)}')
    if set(assignments.values()) - {'train', 'val'}:
        raise ValueError('Unsupported split name')
    # All records with identical patient_id resolve to one unique split by construction.
    if any(not r.get('patient_id') or not r.get('series_id') for r in records):
        raise ValueError('Empty patient or series identifier')


def prepare_split(records: list[dict], path: str | Path, *, seed: int,
                  ratios: tuple[float, float] = (0.8, 0.2)) -> dict:
    """Create immutable patient-level train/val manifest; never create test split.

    Existing incompatible manifests (including old train/val/test manifests)
    are rejected without changing their contents. Held-out test IDs, which
    are not in the supplied inventory, are never included in the result.
    """
    path = Path(path)
    ids = sorted({r['patient_id'] for r in records})
    inventory = sorted((r['patient_id'], r['series_id'], r['image'], r['mask'])
                       for r in records)
    fingerprint = _digest(inventory)
    if len(ids) < 2:
        raise ValueError('At least two distinct patients are required for train/val')
    if len(ratios) != 2 or any(v <= 0 for v in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError('Train and validation ratios must be positive and sum to one')
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('inventory_sha256') != fingerprint:
            raise ValueError('Existing split inventory differs; file preserved')
        validate_split(records, payload['assignments'])
        if set(payload['assignments'].values()) != {'train', 'val'}:
            raise ValueError('Existing manifest is not two-way train/val; file preserved')
        return payload
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = min(len(ids)-1, max(1, int(len(ids)*ratios[0])))
    assignments = {p: ('train' if i < n_train else 'val')
                   for i, p in enumerate(ids)}
    validate_split(records, assignments)
    payload = {'schema_version': 2, 'seed': seed, 'ratios': list(ratios),
               'split_policy': 'train_val_only_heldout_test_external',
               'inventory_sha256': fingerprint,
               'assignments': dict(sorted(assignments.items())),
               'series': [{**r, 'split': assignments[r['patient_id']]} for r in records]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return payload

def fewshot_manifests(split: dict, output_dir: str | Path, *, seeds: list[int],
                      k_values: tuple[int,...] = (1,5,10,20)) -> dict:
    """One deterministic patient permutation per seed yields nested subsets."""
    out=Path(output_dir)
    train=sorted(p for p,v in split['assignments'].items() if v=='train')
    ks=sorted(set(k_values))
    if not ks or ks[0]<1 or max(ks)>len(train):
        raise ValueError(f'Few-shot K values {ks} invalid for {len(train)} training patients')
    if len(set(seeds)) != len(seeds): raise ValueError('Repeated seed')
    result={}
    for seed in seeds:
        permutation=train.copy();random.Random(seed).shuffle(permutation)
        for k in ks:
            file=out/f'seed_{seed}_k_{k}.json'
            payload={'schema_version':1,'source_split_sha256':split['inventory_sha256'],
                     'source_assignments_sha256':_digest(split['assignments']),
                     'seed':seed,'K':k,'patient_ids':permutation[:k],
                     'sampling_unit':'patient','nested':True}
            if file.exists():
                existing=json.loads(file.read_text(encoding='utf-8'))
                if existing!=payload: raise ValueError(f'Existing few-shot manifest conflict: {file}')
            else:
                out.mkdir(parents=True,exist_ok=True)
                with file.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
            result[(seed,k)]=payload
    return result
