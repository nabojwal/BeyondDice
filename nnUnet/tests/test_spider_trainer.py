"""Exercise installed nnU-Net lifecycle APIs on CPU; W&B calls are mocked."""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import wandb
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from training.nnunet.trainers.nnUNetTrainerSPIDER import (
    nnUNetTrainerSPIDER, nnUNetTrainerSPIDER_5epochs, plain, wandb_settings)


class SpiderTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            'nnUNet_preprocessed': str(self.root / 'pre'),
            'nnUNet_results': str(self.root / 'results'), 'nnUNet_compile': 'false',
            'nnUNet_wandb_enabled': '0', 'WANDB_MODE': 'offline',
            'WANDB_PROJECT': 'test-project', 'WANDB_ENTITY': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.no_upload = patch('wandb.init', side_effect=AssertionError('Unexpected W&B initialization'))
        self.no_upload.start()
        self.addCleanup(self.no_upload.stop)

    def construct(self, cls=nnUNetTrainerSPIDER):
        plans = {'dataset_name': 'Dataset011_TEST', 'plans_name': 'nnUNetPlans',
                 'configurations': {'3d_fullres': {'data_identifier': 'test',
                                                  'batch_size': 2, 'patch_size': [8, 8, 8],
                                                  'spacing': [1., 1., 1.], 'architecture': {}}}}
        dataset = {'channel_names': {'0': 'MRI'}, 'labels': {
            'background': 0, 'vertebrae': 1, 'intervertebral discs': 2, 'spinal canal': 3}}
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = cls(plans, '3d_fullres', 0, dataset, torch.device('cpu'))
        self.assertNotIn('continue_training', plans)
        return trainer

    def test_constructor_and_recipe_inheritance(self):
        baseline = self.construct()
        smoke = self.construct(nnUNetTrainerSPIDER_5epochs)
        self.assertEqual(list(inspect.signature(nnUNetTrainerSPIDER).parameters),
                         list(inspect.signature(nnUNetTrainer).parameters))
        self.assertEqual((baseline.num_epochs, baseline.num_iterations_per_epoch,
                          baseline.num_val_iterations_per_epoch), (1000, 250, 50))
        self.assertEqual((smoke.num_epochs, smoke.num_iterations_per_epoch,
                          smoke.num_val_iterations_per_epoch, smoke.save_every), (5, 10, 5, 1))
        self.assertNotEqual(baseline.output_folder, smoke.output_folder)
        for name in ['build_network_architecture', '_build_loss', 'configure_optimizers',
                     'get_training_transforms', 'train_step', 'validation_step', 'perform_actual_validation']:
            self.assertIs(getattr(nnUNetTrainerSPIDER, name), getattr(nnUNetTrainer, name))
        self.assertFalse(baseline.was_initialized)

    def test_actual_logger_epoch_metrics(self):
        trainer = self.construct()
        trainer.print_to_log_file = Mock()
        trainer.save_checkpoint = Mock()
        trainer.logger.plot_progress_png = Mock()
        run = Mock(step=0)
        run.summary = {}
        trainer._wb = run
        values = {'train_losses': .2, 'val_losses': .3, 'mean_fg_dice': .7,
                  'lrs': .01, 'epoch_start_timestamps': 3.,
                  'dice_per_class_or_region': [.6, .7, np.nan]}
        for key, value in values.items():
            trainer.logger.log(key, value, 0)
        with patch('nnunetv2.training.nnUNetTrainer.nnUNetTrainer.time', return_value=13.):
            trainer.on_epoch_end()
        self.assertEqual(trainer.current_epoch, 1)
        metrics = run.log.call_args.args[0]
        self.assertEqual(metrics['epoch'], 0)
        self.assertEqual(metrics['train/epoch_seconds'], 10)
        self.assertEqual(metrics['val/patch_dice_intervertebral_discs'], .7)
        self.assertIsNone(metrics['val/patch_dice_spinal_canal'])
        self.assertEqual(run.log.call_args.kwargs['step'], 0)
        trainer.save_checkpoint.assert_called_once()
        self.assertEqual(trainer._best_ema, .7)

    def test_checkpoint_file_roundtrip_no_forward_pass(self):
        trainer = self.construct()
        trainer.network = torch.nn.Linear(2, 4)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=.01, momentum=.99)
        trainer.was_initialized = True
        trainer.current_epoch = 3
        trainer._best_ema = .7
        trainer.inference_allowed_mirroring_axes = (0, 1, 2)
        expected = {key: value.clone() for key, value in trainer.network.state_dict().items()}
        path = self.root / 'checkpoint.pth'
        trainer.save_checkpoint(str(path))
        with torch.no_grad():
            trainer.network.weight.zero_()
        trainer.load_checkpoint(path)
        self.assertEqual(trainer.current_epoch, 4)
        self.assertTrue(trainer._checkpoint_loaded)
        self.assertEqual(trainer._best_ema, .7)
        for key, value in trainer.network.state_dict().items():
            self.assertTrue(torch.equal(expected[key], value))
        payload = torch.load(path, weights_only=False, map_location='cpu')
        self.assertEqual(payload['trainer_name'], 'nnUNetTrainerSPIDER')
        self.assertEqual(payload['init_args']['configuration'], '3d_fullres')
        with self.assertRaises(TypeError):
            trainer.load_checkpoint({})

    def test_run_guard_and_resume_identity(self):
        trainer = self.construct()
        trainer.do_split = Mock()
        trainer._identity = Mock(return_value={'identity': 'one'})
        trainer._guard_run()
        with self.assertRaises(FileExistsError):
            trainer._guard_run()
        trainer._resume_requested = True
        with self.assertRaises(FileNotFoundError):
            trainer._guard_run()
        trainer._checkpoint_loaded = True
        trainer._guard_run()
        trainer._identity.return_value = {'identity': 'changed'}
        with self.assertRaises(ValueError):
            trainer._guard_run()

    def test_split_rejects_nonzero_fold_and_altered_design(self):
        trainer = self.construct()
        trainer.fold = 'all'
        with self.assertRaises(ValueError):
            trainer.do_split()
        trainer.fold = 0
        folder = Path(trainer.preprocessed_dataset_folder_base)
        folder.mkdir(parents=True)
        (folder / 'splits_final.json').write_text(json.dumps([{'train': ['1_t1'], 'val': ['2_t1']}]))
        trainer.dataset_class = SimpleNamespace(get_identifiers=lambda path: ['1_t1', '2_t1'])
        with self.assertRaisesRegex(ValueError, 'verified baseline'):
            trainer.do_split()

    def test_cleanup_does_not_mask_training_failure(self):
        trainer = self.construct()
        run = Mock()
        run.finish.side_effect = RuntimeError('cleanup failure')
        trainer._wb = run
        trainer.dataloader_train = Mock()
        with patch.object(nnUNetTrainer, 'run_training', side_effect=RuntimeError('training failure')):
            with self.assertWarns(UserWarning):
                with self.assertRaisesRegex(RuntimeError, 'training failure'):
                    trainer.run_training()
        run.finish.assert_called_once_with(exit_code=1)
        trainer.dataloader_train._finish.assert_called_once()
        self.assertIsNone(trainer._wb)

    def test_wandb_modes_resume_and_settings(self):
        trainer = self.construct()
        trainer._resolved_config = Mock(return_value={'safe': 1})
        run = Mock(step=0)
        with patch('wandb.init', return_value=run) as initialize:
            trainer._start_wandb()
            self.assertEqual(initialize.call_args.kwargs['mode'], 'offline')
            self.assertIsNone(initialize.call_args.kwargs['resume'])
            self.assertFalse((Path(trainer.output_folder) / 'spider_wandb_online.json').exists())
        with patch.dict(os.environ, {'WANDB_MODE': 'online'}):
            with patch('wandb.init', return_value=run) as initialize:
                trainer._start_wandb()
                run_id = initialize.call_args.kwargs['id']
                self.assertEqual(initialize.call_args.kwargs['resume'], 'never')
                trainer.current_epoch = 5
                trainer._start_wandb()
                self.assertEqual(initialize.call_args.kwargs['id'], run_id)
                self.assertEqual(initialize.call_args.kwargs['resume'], 'must')
                with patch.dict(os.environ, {'WANDB_PROJECT': 'different'}):
                    with self.assertRaises(ValueError):
                        trainer._start_wandb()
        settings = wandb_settings()
        self.assertEqual(settings.console, 'off')
        for key in ['disable_code', 'disable_git', 'x_disable_meta', 'x_disable_stats',
                    'x_disable_machine_info', 'disable_job_creation']:
            self.assertTrue(getattr(settings, key))
        self.assertFalse(settings.save_code)
        run.log_artifact.assert_not_called()

    def test_replayed_online_epoch_is_not_relogged(self):
        trainer = self.construct()
        trainer._wb = Mock(step=10)
        trainer._wb.summary = {}
        with patch.object(nnUNetTrainer, 'on_epoch_end'):
            trainer.on_epoch_end()
        trainer._wb.log.assert_not_called()

    def test_disabled_logging_and_missing_online_identity(self):
        trainer = self.construct()
        trainer._resolved_config = Mock(return_value={'safe': 1})
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled'}):
            trainer._start_wandb()
        self.assertIsNone(trainer._wb)
        trainer.current_epoch = 5
        with patch.dict(os.environ, {'WANDB_MODE': 'online'}):
            with self.assertRaises(FileNotFoundError):
                trainer._start_wandb()

    def test_initialization_failure_closes_wandb(self):
        trainer = self.construct()
        trainer._resolved_config = Mock(return_value={'safe': 1})
        run = Mock()
        run.define_metric.side_effect = RuntimeError('metric setup failed')
        def start_and_fail():
            trainer._start_wandb()
        with patch('wandb.init', return_value=run):
            with patch.object(nnUNetTrainer, 'run_training', side_effect=start_and_fail):
                with self.assertRaisesRegex(RuntimeError, 'metric setup failed'):
                    trainer.run_training()
        run.finish.assert_called_once_with(exit_code=1)
        self.assertIsNone(trainer._wb)

    def test_predictor_checkpoint_initialization_without_inference(self):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        trainer = self.construct()
        folder = self.root / 'inference_fixture'
        (folder / 'fold_0').mkdir(parents=True)
        (folder / 'plans.json').write_text(json.dumps(trainer.plans_manager.plans))
        (folder / 'dataset.json').write_text(json.dumps(trainer.dataset_json))
        network = torch.nn.Linear(2, 4)
        torch.save({'network_weights': network.state_dict(),
                    'trainer_name': 'nnUNetTrainerSPIDER',
                    'init_args': {'configuration': '3d_fullres'},
                    'inference_allowed_mirroring_axes': (0, 1, 2)},
                   folder / 'fold_0' / 'checkpoint_final.pth')
        trainer_dir = Path(__file__).resolve().parents[1] / 'training/nnunet/trainers'
        with patch.dict(os.environ, {'nnUNet_extTrainer': str(trainer_dir)}):
            predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
            # Exercise real discovery/restore, replacing only network construction.
            def build_network_architecture(plans_manager, configuration_manager,
                                           num_input_channels, num_output_channels,
                                           enable_deep_supervision=True):
                return torch.nn.Linear(2, 4)
            with patch.object(nnUNetTrainer, 'build_network_architecture',
                              staticmethod(build_network_architecture)):
                with contextlib.redirect_stdout(io.StringIO()):
                    predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,))
        self.assertTrue(torch.equal(predictor.network.weight, network.weight))

    def test_partial_sdk_initialization_cleanup(self):
        trainer = self.construct()
        trainer._resolved_config = Mock(return_value={'safe': 1})
        run = Mock()
        def failed_init(**kwargs):
            wandb.run = run
            raise RuntimeError('SDK startup failed')
        with patch('wandb.run', None):
            with patch('wandb.init', side_effect=failed_init):
                with self.assertRaisesRegex(RuntimeError, 'SDK startup failed'):
                    trainer._start_wandb()
        run.finish.assert_called_once_with(exit_code=1)

    def test_config_excludes_case_paths_and_secrets(self):
        trainer = self.construct()
        trainer.dataset_json['dataset'] = {'secret_patient': {'images': ['secret_path']}}
        trainer.dataset_json['credential'] = 'do-not-upload'
        trainer.batch_size = 2
        trainer.network = torch.nn.Linear(2, 4)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=.01)
        trainer.lr_scheduler = SimpleNamespace(state_dict=lambda: {'epoch': 0})
        trainer.loss = torch.nn.MSELoss()
        trainer._get_deep_supervision_scales = lambda: [[1, 1, 1]]
        payload = json.dumps(trainer._resolved_config(), allow_nan=False)
        for forbidden in ['secret_patient', 'secret_path', 'do-not-upload', str(self.root)]:
            self.assertNotIn(forbidden, payload)
        self.assertIsNone(plain(float('nan')))


if __name__ == '__main__':
    unittest.main()
