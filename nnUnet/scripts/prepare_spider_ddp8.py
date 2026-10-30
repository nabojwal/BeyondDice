"""Add an eight-GPU plan/contract without modifying the single-GPU experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.nnunet.data.spider_ddp_contract import (
    PLAN_NAME, CONFIGURATION, CONTRACT_FILE, GLOBAL_BATCH, WORLD_SIZE,
    SOURCE_TRAINER_SHA256, SOURCE_PLANS_SHA256, derived_plans, digest, read_json,
    source_hash, validate_contract)
from training.nnunet.data.spider_instances_split import validate_split


def prepare(pre, parent_source):
    pre = Path(pre)
    source = read_json(pre / 'nnUNetPlans.json')
    contract = read_json(pre / 'spider_contract.json')
    dataset = read_json(pre / 'dataset.json')
    splits = read_json(pre / 'splits_final.json')
    validate_split(splits, approved_hash=contract['split_sha256'])
    if source_hash(parent_source) != SOURCE_TRAINER_SHA256:
        raise ValueError('Parent trainer differs from the exact reviewed upload')
    if contract['plans_sha256'] != SOURCE_PLANS_SHA256 or digest(dataset) != contract['dataset_sha256']:
        raise ValueError('Single-GPU plans/dataset no longer match the finalized contract')
    plans = derived_plans(source)
    manifest = {
        'schema_version': 1, 'plans_name': PLAN_NAME, 'configuration': CONFIGURATION,
        'world_size': WORLD_SIZE, 'global_batch_size': GLOBAL_BATCH,
        'source_contract_sha256': digest(contract), 'derived_plans_sha256': digest(plans),
        'parent_trainer_sha256': SOURCE_TRAINER_SHA256,
        'training_schedule': '1000 epochs x 250 global optimizer steps; 4000 patches per epoch',
        'validation_schedule': '50 patch steps per rank per epoch; 800 patches globally',
        'initial_lr': .01, 'lr_scaling': 'none',
    }
    targets = {pre / f'{PLAN_NAME}.json': plans, pre / CONTRACT_FILE: manifest}
    # Check all collisions before writing any file.
    for path, content in targets.items():
        if path.exists() and read_json(path) != content:
            raise FileExistsError(f'Conflicting file preserved: {path}')
    for path, content in targets.items():
        if not path.exists():
            with path.open('x', encoding='utf-8') as handle:
                json.dump(content, handle, indent=2)
                handle.write('\n')
    validate_contract(pre, plans, dataset, parent_source)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprocessed-dataset', type=Path)
    args = parser.parse_args()
    pre = args.preprocessed_dataset or Path(os.environ['nnUNet_preprocessed']) / 'Dataset012_SPIDERInstances'
    root = Path(__file__).resolve().parents[1]
    parent_source = root / 'training/nnunet/trainers/nnUNetTrainerSPIDERInstances.py'
    # Validate preprocessing coverage before publishing the new contract.
    from importlib.metadata import version
    if version('nnunetv2') != '2.8.1':
        raise RuntimeError('Requires nnunetv2==2.8.1')
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    source = read_json(pre / 'nnUNetPlans.json')
    folder = pre / PlansManager(source).get_configuration('3d_fullres').data_identifier
    cases = infer_dataset_class(str(folder)).get_identifiers(str(folder))
    original = read_json(pre / 'spider_contract.json')
    validate_split(read_json(pre / 'splits_final.json'), cases, original['split_sha256'])
    plans = prepare(pre, parent_source)
    resolved = PlansManager(plans).get_configuration(CONFIGURATION)
    if resolved.batch_size != GLOBAL_BATCH or resolved.data_identifier != folder.name:
        raise ValueError('Derived configuration did not resolve to expected preprocessing/batch size')
    print(json.dumps({'plans': PLAN_NAME, 'configuration': CONFIGURATION, 'world_size': WORLD_SIZE,
                      'global_batch_size': GLOBAL_BATCH, 'batch_per_gpu': GLOBAL_BATCH // WORLD_SIZE,
                      'preprocessed_cases': len(cases), 'preprocessing_reused': resolved.data_identifier,
                      'original_files_modified': False}, indent=2))


if __name__ == '__main__':
    main()
