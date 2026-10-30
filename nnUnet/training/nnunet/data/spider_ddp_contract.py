"""Immutable, separately named eight-GPU experiment derived from the supplied plans."""
import copy
import hashlib
import json
from pathlib import Path

PLAN_NAME = 'nnUNetPlansSPIDERDDP8'
CONFIGURATION = '3d_fullres_ddp8'
CONTRACT_FILE = 'spider_ddp8_contract.json'
WORLD_SIZE = 8
GLOBAL_BATCH = 16
SOURCE_PLANS_SHA256 = '429c08768cc5a3a3418775fc9dffaef0060efeb1b0f1a2e7a69c57a23da1930a'
SOURCE_TRAINER_SHA256 = 'f3ea52419882cddb35338e984777eccb5bd79c02ae221a20d9abb17ce417e693'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hash(path):
    # Windows CRLF versus Linux LF does not constitute a source change.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode()).hexdigest()


def derived_plans(source):
    if digest(source) != SOURCE_PLANS_SHA256:
        raise ValueError('Source nnUNetPlans.json differs from the reviewed upload')
    plans = copy.deepcopy(source)
    plans['plans_name'] = PLAN_NAME
    plans['configurations'][CONFIGURATION] = {
        'inherits_from': '3d_fullres', 'batch_size': GLOBAL_BATCH}
    return plans


def validate_contract(pre, plans, dataset, parent_source):
    pre = Path(pre)
    original = read_json(pre / 'spider_contract.json')
    source = read_json(pre / 'nnUNetPlans.json')
    manifest = read_json(pre / CONTRACT_FILE)
    mapping = read_json(pre / 'spider_label_mapping.json')
    payload = {k: v for k, v in mapping.items() if k != 'sha256'}
    clean_plans = {k: v for k, v in plans.items() if k != 'continue_training'}
    checks = [
        digest(source) == original['plans_sha256'] == SOURCE_PLANS_SHA256,
        clean_plans == derived_plans(source),
        digest(clean_plans) == manifest['derived_plans_sha256'],
        digest(original) == manifest['source_contract_sha256'],
        digest(dataset) == original['dataset_sha256'],
        dataset['labels'] == mapping['labels'],
        digest(payload) == mapping['sha256'] == original['label_mapping_sha256'],
        source_hash(parent_source) == SOURCE_TRAINER_SHA256 == manifest['parent_trainer_sha256'],
        manifest['world_size'] == WORLD_SIZE,
        manifest['global_batch_size'] == GLOBAL_BATCH,
        manifest['configuration'] == CONFIGURATION,
        manifest['plans_name'] == PLAN_NAME,
        source['dataset_name'] == original['dataset_name'] == 'Dataset012_SPIDERInstances',
    ]
    if not all(checks):
        raise ValueError('DDP contract, mapping, plans, dataset or parent trainer changed')
    return dict(original, plans_sha256=digest(clean_plans), ddp=manifest)
