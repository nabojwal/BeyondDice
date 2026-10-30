"""nnU-Net 2.8.1 baseline with a fixed SPIDER holdout and opt-in W&B uploads."""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import warnings
from importlib.metadata import version

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from training.nnunet.data.spider_instances_split import digest, validate_split


def plain(value):
    """Convert numerical settings to strict JSON; undefined Dice becomes null."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported configuration value: {type(value).__name__}')


def wandb_settings():
    """Disable automatic console, code, Git and machine/host metadata collection."""
    import wandb
    return wandb.Settings(
        console='off', disable_code=True, save_code=False, disable_git=True,
        disable_job_creation=True, x_disable_meta=True, x_disable_stats=True,
        x_disable_machine_info=True)


class nnUNetTrainerSPIDERInstances(nnUNetTrainer):
    """Inherit the standard training recipe; change only split enforcement and logging."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        if version('nnunetv2') != '2.8.1':
            raise RuntimeError('This integration requires nnunetv2==2.8.1')
        # Upstream initializes its optional W&B logger in the constructor.
        # Disable it before super(): this integration logs only allowlisted fields.
        os.environ['nnUNet_wandb_enabled'] = '0'
        self._resume_requested = bool(plans.get('continue_training', False))
        plans = dict(plans, continue_training=self._resume_requested)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._wb = None
        self._checkpoint_loaded = False

    def _plans_digest(self):
        # CLI resume flag is execution state, not part of the model plan.
        return digest({k: v for k, v in self.plans_manager.plans.items()
                       if k != 'continue_training'})

    def _contract(self):
        base = Path(self.preprocessed_dataset_folder_base)
        contract = json.loads((base / 'spider_contract.json').read_text())
        mapping = json.loads((base / 'spider_label_mapping.json').read_text())
        mapping_payload = {k: v for k, v in mapping.items() if k != 'sha256'}
        if (digest(mapping_payload) != contract['label_mapping_sha256'] or
                mapping['sha256'] != contract['label_mapping_sha256'] or
                mapping['labels'] != self.dataset_json['labels'] or
                digest(self.dataset_json) != contract['dataset_sha256'] or
                self._plans_digest() != contract['plans_sha256'] or
                self.plans_manager.plans['dataset_name'] != contract['dataset_name']):
            raise ValueError('Dataset, mapping or plans differ from finalized preparation')
        return contract

    def do_split(self):
        """Require the approved fold and exact coverage of preprocessed cases."""
        if self.fold != 0:
            raise ValueError('SPIDER requires internal fold 0; all and other folds are forbidden')
        config = self._contract()
        path = Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'
        splits = json.loads(path.read_text())
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        cases = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
        train, val = validate_split(splits, cases, config['split_sha256'])
        return train, val

    def load_checkpoint(self, filename_or_checkpoint):
        """Use upstream file restoration; reject its broken dictionary overload explicitly."""
        # In installed 2.8.1 the dictionary overload leaves 'checkpoint' unbound.
        # All supported training/resume/validation CLI paths pass filenames.
        if not isinstance(filename_or_checkpoint, (str, Path)):
            raise TypeError('nnU-Net 2.8.1 checkpoint restoration requires a filename')
        super().load_checkpoint(str(filename_or_checkpoint))
        self._checkpoint_loaded = True

    def _identity(self):
        """Bind a local run to its split, plans, trainer, configuration and full schedule."""
        split = json.loads((Path(self.preprocessed_dataset_folder_base) / 'splits_final.json').read_text())
        validate_split(split, approved_hash=self._contract()['split_sha256'])
        return {
            'trainer': type(self).__name__, 'configuration': self.configuration_name,
            'fold': self.fold, 'split_sha256': digest(split),
            'plans_sha256': self._plans_digest(),
            'label_mapping_sha256': self._contract()['label_mapping_sha256'],
            'source_manifest_sha256': self._contract()['source_manifest_sha256'],
            'dataset_sha256': digest(self.dataset_json),
            'schedule': [self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch],
        }

    def _guard_run(self):
        """Fail before starting workers when a run would be overwritten or resumed unsafely."""
        if self.is_ddp:
            raise RuntimeError('SPIDER integration supports single-GPU training only')
        self.do_split()
        out = Path(self.output_folder)
        identity_path = out / 'spider_run_identity.json'
        identity = self._identity()
        checkpoints = list(out.glob('checkpoint_*.pth'))
        if self._resume_requested and not self._checkpoint_loaded:
            raise FileNotFoundError('Resume requested but no checkpoint was loaded; refusing fresh training')
        if self._checkpoint_loaded:
            if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
                raise ValueError('Resume identity missing or changed; preserve this run and use matching settings')
        elif checkpoints or identity_path.exists():
            raise FileExistsError('Existing SPIDER run: use --c to resume or select a new results root')
        else:
            with identity_path.open('x', encoding='utf-8') as handle:
                json.dump(identity, handle, indent=2)

    def _resolved_config(self):
        """Return aggregate settings only: no case lists, patient rows, paths or credentials."""
        settings = self.configuration_manager.configuration
        allowed = (
            'batch_size', 'patch_size', 'spacing', 'normalization_schemes',
            'use_mask_for_norm', 'preprocessor_name', 'batch_dice', 'architecture',
            'resampling_fn_data', 'resampling_fn_seg', 'resampling_fn_probabilities',
            'resampling_fn_data_kwargs', 'resampling_fn_seg_kwargs', 'resampling_fn_probabilities_kwargs')
        training = {key: getattr(self, key) for key in (
            'num_epochs', 'initial_lr', 'weight_decay', 'batch_size',
            'num_iterations_per_epoch', 'num_val_iterations_per_epoch',
            'oversample_foreground_percent', 'probabilistic_oversampling',
            'save_every', 'enable_deep_supervision')}
        sources = [nnUNetTrainer.get_training_transforms,
                   nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size,
                   nnUNetTrainer._build_loss]
        approved = self._contract()
        return plain({
            'trainer': type(self).__name__, 'dataset_id': approved['dataset_id'], 'fold': self.fold,
            'configuration': self.configuration_name, 'channel_names': {'0': 'MRI'},
            'labels': self.dataset_json['labels'],
            'label_mapping_sha256': approved['label_mapping_sha256'],
            'source_manifest_sha256': approved['source_manifest_sha256'],
            'resolved_configuration': {key: settings[key] for key in allowed if key in settings},
            'transpose_forward': self.plans_manager.plans.get('transpose_forward'),
            'transpose_backward': self.plans_manager.plans.get('transpose_backward'),
            'training': training,
            'optimizer': {'class': type(self.optimizer).__name__, 'defaults': self.optimizer.defaults},
            'scheduler': {'class': type(self.lr_scheduler).__name__,
                          'state': self.lr_scheduler.state_dict()},
            'loss_class': type(self.loss).__name__, 'loss': repr(self.loss),
            'deep_supervision_scales': self._get_deep_supervision_scales(),
            'parameter_count': sum(parameter.numel() for parameter in self.network.parameters()),
            'amp': self.grad_scaler is not None,
            'versions': {name: version(name) for name in ['nnunetv2', 'torch', 'numpy', 'wandb']},
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
            'compile_enabled': self._do_i_compile(),
            'augmentation_workers': int(os.getenv('nnUNet_n_proc_DA', '2')),
            'inference_mirroring_allowed_axes': self.inference_allowed_mirroring_axes,
            'split_sha256': approved['split_sha256'], 'split_counts': approved['counts'],
            'implementation_sha256': {
                method.__name__: hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
                for method in sources},
            'resume_epoch': self.current_epoch,
            'checkpoint_selection': 'EMA foreground patch Dice; not full-volume Dice',
            'reproducibility': 'deterministic patient split; standard stochastic nnU-Net training',
        })

    def _start_wandb(self):
        """Persist safe settings locally and initialize a mode-specific W&B run."""
        import wandb
        config = self._resolved_config()
        out = Path(self.output_folder)
        # Retain initial and restart snapshots rather than replacing the original record.
        import uuid
        snapshot = out / f'spider_resolved_config_epoch{self.current_epoch}_{uuid.uuid4().hex[:8]}.json'
        snapshot.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        mode = os.getenv('WANDB_MODE', 'offline')
        if mode not in {'online', 'offline', 'disabled'}:
            raise ValueError('WANDB_MODE must be online, offline or disabled')
        if mode == 'disabled':
            return
        project = os.getenv('WANDB_PROJECT', 'BeyondDice-SPIDER')
        entity = os.getenv('WANDB_ENTITY') or None
        state_file = out / 'spider_wandb_online.json'
        if self.current_epoch > 0 and mode == 'online':
            if not state_file.is_file():
                raise FileNotFoundError('No online W&B run ID for resume; continue offline or preserve this run')
            state = json.loads(state_file.read_text())
            if state['project'] != project or state['entity'] != entity:
                raise ValueError('W&B project/entity changed during resume')
            run_id, resume = state['id'], 'must'
        else:
            run_id, resume = uuid.uuid4().hex[:8], 'never' if mode == 'online' else None
        if wandb.run is not None:
            raise RuntimeError('Another W&B run is already active in this process')
        try:
            self._wb = wandb.init(
                project=project, entity=entity, name=os.getenv('WANDB_NAME') or type(self).__name__,
                id=run_id, resume=resume, mode=mode, dir=str(out), config=config,
                settings=wandb_settings(), allow_val_change=True)
        except BaseException:
            # A failed SDK initialization can leave a partially created run.
            if wandb.run is not None:
                try:
                    wandb.run.finish(exit_code=1)
                except Exception as error:
                    warnings.warn(f'W&B initialization cleanup failed: {type(error).__name__}')
            raise
        if mode == 'online':
            state_file.write_text(json.dumps({'id': run_id, 'project': project, 'entity': entity}), encoding='utf-8')
        self._wb.define_metric('epoch')
        self._wb.define_metric('*', step_metric='epoch')
        # No artifacts, console logs, files, volumes or patient metadata are uploaded.

    def on_train_start(self):
        """Perform standard initialization followed by restricted experiment logging."""
        self._guard_run()
        super().on_train_start()
        self._start_wandb()

    def on_epoch_end(self):
        """Log completed zero-based epochs without changing standard checkpoint selection."""
        epoch = self.current_epoch
        super().on_epoch_end()
        if self._wb is None:
            return
        # Online resume can replay epochs newer than the last saved checkpoint.
        # W&B history cannot rewind: retain already-logged steps and log new epochs only.
        if epoch < self._wb.step:
            return
        def get(key):
            return self.logger.get_value(key, step=epoch)
        metrics = {
            'epoch': epoch, 'train/loss': get('train_losses'),
            'val/patch_loss': get('val_losses'), 'val/patch_dice_mean': get('mean_fg_dice'),
            'val/patch_dice_ema': get('ema_fg_dice'), 'train/lr': get('lrs'),
            'train/epoch_seconds': get('epoch_end_timestamps') - get('epoch_start_timestamps'),
        }
        names = {value: name for name, value in self.dataset_json['labels'].items()}
        for label, dice in zip(self.label_manager.foreground_labels, get('dice_per_class_or_region'), strict=True):
            metrics[f'val/patch_dice_{names[label]}'] = dice
        self._wb.log(plain(metrics), step=epoch)
        self._wb.summary['best_ema_patch_dice'] = plain(self._best_ema)

    def run_training(self):
        """Close W&B and worker processes on failures without masking the original error."""
        failed = True
        try:
            # A completed --c run should proceed to CLI validation, not rewrite checkpoints.
            if self._checkpoint_loaded and self.current_epoch >= self.num_epochs:
                self._guard_run()
                failed = False
                return
            result = super().run_training()
            failed = False
            return result
        finally:
            if failed:
                for loader in (getattr(self, 'dataloader_train', None), getattr(self, 'dataloader_val', None)):
                    finish = getattr(loader, '_finish', None)
                    if callable(finish):
                        try:
                            finish()
                        except Exception as error:
                            warnings.warn(f'Worker cleanup failed: {type(error).__name__}')
            if self._wb is not None:
                try:
                    self._wb.finish(exit_code=1 if failed else 0)
                except Exception as error:
                    warnings.warn(f'W&B cleanup failed: {type(error).__name__}')
                finally:
                    self._wb = None


class nnUNetTrainerSPIDERInstances_5epochs(nnUNetTrainerSPIDERInstances):
    """Separate smoke experiment: 5 epochs with 10 train / 5 development iterations."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
        self.save_every = 1
