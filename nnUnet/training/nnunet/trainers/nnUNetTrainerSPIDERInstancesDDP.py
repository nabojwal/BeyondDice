"""Single-node, exactly eight-GPU SPIDER experiment; requires nnU-Net 2.8.1.

Adds rank-safe orchestration. Loss, optimizer, augmentation and metric reduction
remain the upstream DDP implementations. Parent single-GPU source is unchanged.
"""
import inspect
import json
import os
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from training.nnunet.trainers.nnUNetTrainerSPIDERInstances import nnUNetTrainerSPIDERInstances
from training.nnunet.data.spider_ddp_coordination import coordinated_call
from training.nnunet.data.spider_ddp_contract import (
    CONFIGURATION, GLOBAL_BATCH, WORLD_SIZE, digest, source_hash, validate_contract)


class nnUNetTrainerSPIDERInstancesDDP(nnUNetTrainerSPIDERInstances):
    """1000 epochs, 250 global training steps/epoch, global batch 16."""

    def _contract(self):
        return validate_contract(
            self.preprocessed_dataset_folder_base, self.plans_manager.plans,
            self.dataset_json, inspect.getfile(nnUNetTrainerSPIDERInstances.__init__))

    def _identity(self):
        identity = super()._identity()
        identity.update(
            ddp_world_size=WORLD_SIZE, global_batch_size=GLOBAL_BATCH,
            ddp_trainer_sha256=source_hash(__file__),
            compile_enabled=self._do_i_compile(),
            parent_trainer_sha256=self._contract()['ddp']['parent_trainer_sha256'])
        return identity

    def _preflight(self):
        if not self.is_ddp or not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError('Use single-node nnUNetv2_train -num_gpus 8 for this trainer')
        if self.device.type != 'cuda' or self.local_rank != dist.get_rank():
            raise RuntimeError('Expected single-node CUDA rank assignment')
        if self.configuration_name != CONFIGURATION or self.configuration_manager.batch_size != GLOBAL_BATCH:
            raise ValueError('Use the separately prepared 3d_fullres_ddp8 configuration')
        self.do_split()
        return {'identity': self._identity(), 'resume': self._resume_requested,
                'loaded': self._checkpoint_loaded, 'epoch': self.current_epoch,
                'training_completed': getattr(self, '_training_completed', False),
                'wandb_mode': os.getenv('WANDB_MODE', 'offline'),
                'wandb_project': os.getenv('WANDB_PROJECT', 'BeyondDice-SPIDER'),
                'wandb_entity': os.getenv('WANDB_ENTITY')}

    def _guard_run(self):
        states = coordinated_call(self._preflight)
        if any(state != states[0] for state in states):
            raise RuntimeError('Ranks disagree on experiment, checkpoint, epoch or W&B settings')
        def guard():
            out = Path(self.output_folder)
            path = out / 'spider_run_identity.json'
            identity = states[0]['identity']
            if self._resume_requested and not self._checkpoint_loaded:
                raise FileNotFoundError('Resume requested but no checkpoint was loaded')
            if self._checkpoint_loaded or getattr(self, '_training_completed', False):
                if not path.is_file() or json.loads(path.read_text()) != identity:
                    raise ValueError('Missing or changed DDP run identity; preserve this run')
            elif path.exists() or list(out.glob('checkpoint_*.pth')):
                raise FileExistsError('Existing DDP run: resume with --c or select a fresh results root')
            else:
                out.mkdir(parents=True, exist_ok=True)
                with path.open('x') as handle:
                    json.dump(identity, handle, indent=2)
        coordinated_call(guard, root_only=True)

    def _resolved_config(self):
        config = super()._resolved_config()
        # self.batch_size is rank-local after upstream initialization.
        config['distributed'] = {
            'world_size': WORLD_SIZE, 'backend': dist.get_backend(),
            'global_batch_size': GLOBAL_BATCH, 'batch_size_per_rank': GLOBAL_BATCH // WORLD_SIZE,
            'training_patches_per_epoch': GLOBAL_BATCH * self.num_iterations_per_epoch,
            'validation_patches_per_epoch': GLOBAL_BATCH * self.num_val_iterations_per_epoch,
            'rank_zero_logging_only': True,
            'oversampling_policy': 'upstream deterministic allocation across global batch',
            'learning_rate_policy': 'unchanged upstream initial LR; no linear scaling',
            'export_processes_per_rank': int(os.getenv('nnUNet_def_n_proc', '8')),
            'ddp_trainer_sha256': source_hash(__file__),
        }
        return config

    def _start_wandb(self):
        # Only rank zero creates snapshots, online state or a W&B run.
        coordinated_call(lambda: super(nnUNetTrainerSPIDERInstancesDDP, self)._start_wandb(), root_only=True)

    def on_train_start(self):
        """Pinned upstream startup sequence, with shared metadata written once.

        Do not call parent on_train_start: its upstream implementation writes
        plans.json/dataset.json/fingerprint concurrently from all ranks in 2.8.1.
        """
        self._guard_run()
        if not self.was_initialized:
            self.initialize()  # Includes DDP collectives; launcher handles failures.

        def create_loaders():
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        coordinated_call(create_loaders)
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)

        def unpack_and_write():
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder, overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify=True)
            out = Path(self.output_folder_base)
            out.mkdir(parents=True, exist_ok=True)
            for name, payload in [('plans.json', self.plans_manager.plans), ('dataset.json', self.dataset_json)]:
                (out / name).write_text(json.dumps(payload, indent=2))
            shutil.copyfile(Path(self.preprocessed_dataset_folder_base) / 'dataset_fingerprint.json',
                            out / 'dataset_fingerprint.json')
            self.plot_network_architecture()
            self._save_debug_information()
        coordinated_call(unpack_and_write, root_only=True)
        self._start_wandb()

    def on_epoch_end(self):
        # Upstream already aggregates TP/FP/FN and losses across ranks before
        # this hook. Parent logs those global metrics only where _wb is set.
        coordinated_call(lambda: super(nnUNetTrainerSPIDERInstancesDDP, self).on_epoch_end())

    def on_train_end(self):
        # No rank can load checkpoint_best/final until rank zero finishes writes.
        coordinated_call(lambda: super(nnUNetTrainerSPIDERInstancesDDP, self).on_train_end())
        self._training_completed = True

    def perform_actual_validation(self, save_probabilities=False):
        # Also enforce identity for --val, which bypasses on_train_start.
        self._guard_run()
        # Upstream shards case identifiers, waits for exports and calculates
        # full-volume summary on rank zero. W&B training has already closed.
        result = super().perform_actual_validation(save_probabilities)
        dist.barrier()
        return result


class nnUNetTrainerSPIDERInstancesDDP_5epochs(nnUNetTrainerSPIDERInstancesDDP):
    """Separate DDP smoke: 5 epochs, 10 training and 5 validation steps each."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
        self.save_every = 1
