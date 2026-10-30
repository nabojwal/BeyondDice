"""Read-only dataset checks and CPU trainer construction; no training or W&B run."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from importlib.metadata import version

import torch
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

from training.nnunet.data.spider_split import (
    CONFIG, discover_dataset, make_split, validate_split, verify_dataset)


def main():
    """Verify the local split, processed identifiers and both external trainer classes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    args = parser.parse_args()
    approved = json.loads(CONFIG.read_text())
    raw = discover_dataset(os.environ['nnUNet_raw'])
    pre = Path(os.environ['nnUNet_preprocessed']) / raw.name
    splits, report = make_split(args.csv)
    validate_split(splits, approved_hash=approved['split_sha256'])
    if json.loads((pre / 'splits_final.json').read_text()) != splits:
        raise ValueError('Installed split differs from local CSV derivation')
    report['preprocessed_case_counts'] = verify_dataset(raw, pre, splits)
    plans = json.loads((pre / 'nnUNetPlans.json').read_text())
    dataset = json.loads((pre / 'dataset.json').read_text())
    found = {}
    previous_results = os.environ.get('nnUNet_results')
    previous_logger = os.environ.get('nnUNet_wandb_enabled')
    try:
        with tempfile.TemporaryDirectory(prefix='spider_api_check_') as tmp:
            os.environ['nnUNet_results'] = tmp
            for name in ['nnUNetTrainerSPIDER', 'nnUNetTrainerSPIDER_5epochs']:
                with contextlib.redirect_stdout(io.StringIO()):
                    cls = recursive_find_trainer_class_by_name(name)
                    trainer = cls(copy.deepcopy(plans), '3d_fullres', 0, dataset, torch.device('cpu'))
                    train, val = trainer.do_split()
                if trainer.was_initialized or trainer.network is not None or trainer._wb is not None:
                    raise AssertionError('Verification unexpectedly initialized training')
                found[name] = {'train_scans': len(train), 'val_scans': len(val),
                               'epochs': trainer.num_epochs,
                               'train_iterations': trainer.num_iterations_per_epoch,
                               'validation_iterations': trainer.num_val_iterations_per_epoch}
    finally:
        for key, old in [('nnUNet_results', previous_results), ('nnUNet_wandb_enabled', previous_logger)]:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    report.update(dataset=raw.name, trainers=found,
                  versions={name: version(name) for name in ['nnunetv2', 'torch', 'numpy', 'wandb']},
                  cuda_available=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
