"""Contract and logging unit tests with lightweight upstream/W&B stand-ins; no GPU claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.nnunet.data.spider_instances_split import digest


def trainer_class():
    class Base:
        def on_epoch_end(self):
            self.current_epoch += 1
    torch = types.ModuleType('torch')
    torch.device = lambda *args: None
    base = types.ModuleType('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
    base.nnUNetTrainer = Base
    data = types.ModuleType('nnunetv2.training.dataloading.nnunet_dataset')
    data.infer_dataset_class = lambda folder: None
    spec = importlib.util.spec_from_file_location('trainer_under_test', ROOT / 'training/nnunet/trainers/nnUNetTrainerSPIDERInstances.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch, base.__name__: base, data.__name__: data}):
        spec.loader.exec_module(module)
    return module.nnUNetTrainerSPIDERInstances


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cls = trainer_class()
        self.obj = self.cls.__new__(self.cls)
        self.obj.preprocessed_dataset_folder_base = str(self.root)
        self.obj.output_folder = str(self.root)
        self.obj.dataset_json = {'labels': {'background': 0, 'vertebra_1': 1, 'spinal_canal': 2, 'disc_201': 3}}
        plans = {'dataset_name': 'Dataset012_SPIDERInstances'}
        self.obj.plans_manager = types.SimpleNamespace(plans=plans)
        mapping = {'labels': self.obj.dataset_json['labels'], 'source_to_encoded': {'0': 0, '1': 1, '100': 2, '201': 3}}
        mapping['sha256'] = digest(mapping)
        split = [{'train': ['1_t1'], 'val': ['2_t1']}]
        self.contract = {'label_mapping_sha256': mapping['sha256'], 'dataset_sha256': digest(self.obj.dataset_json),
                         'plans_sha256': digest(plans), 'dataset_name': plans['dataset_name'],
                         'split_sha256': digest(split), 'source_manifest_sha256': 'testhash'}
        for name, payload in [('spider_contract.json', self.contract), ('spider_label_mapping.json', mapping),
                              ('splits_final.json', split)]:
            (self.root / name).write_text(json.dumps(payload))
        self.obj.fold = 0
        self.obj.configuration_name = '3d_fullres'
        self.obj.num_epochs = 1000
        self.obj.num_iterations_per_epoch = 250
        self.obj.num_val_iterations_per_epoch = 50
        self.obj.is_ddp = False
        self.obj._checkpoint_loaded = False
        self.obj._resume_requested = False
        self.obj.do_split = lambda: (['1_t1'], ['2_t1'])

    def test_contract_accepts_only_matching_labels_and_plans(self):
        self.assertEqual(self.obj._contract(), self.contract)
        self.obj.dataset_json['labels']['disc_201'] = 9
        with self.assertRaises(ValueError):
            self.obj._contract()

    def test_resume_flag_does_not_change_plan_identity(self):
        identity = self.obj._identity()
        self.obj.plans_manager.plans['continue_training'] = True
        self.assertEqual(identity, self.obj._identity())

    def test_fresh_run_overwrite_rejected(self):
        self.obj._guard_run()
        with self.assertRaises(FileExistsError):
            self.obj._guard_run()

    def test_resume_without_checkpoint_rejected(self):
        self.obj._resume_requested = True
        with self.assertRaises(FileNotFoundError):
            self.obj._guard_run()

    def test_matching_resume_accepted_changed_schedule_rejected(self):
        self.obj._guard_run()
        self.obj._checkpoint_loaded = True
        self.obj._resume_requested = True
        self.obj._guard_run()
        self.obj.num_epochs = 200
        with self.assertRaises(ValueError):
            self.obj._guard_run()

    def test_dynamic_per_label_wandb_metrics(self):
        obj = self.obj
        obj.current_epoch = 0
        obj._best_ema = .2
        values = {'train_losses': .5, 'val_losses': .6, 'mean_fg_dice': .2, 'ema_fg_dice': .2,
                  'lrs': .01, 'epoch_end_timestamps': 15, 'epoch_start_timestamps': 5,
                  'dice_per_class_or_region': [.1, .2, float('nan')]}
        obj.logger = types.SimpleNamespace(get_value=lambda key, step: values[key])
        obj.label_manager = types.SimpleNamespace(foreground_labels=[1, 2, 3])
        history = []
        obj._wb = types.SimpleNamespace(step=0, summary={}, log=lambda data, step: history.append(data))
        obj.on_epoch_end()
        self.assertEqual(history[0]['val/patch_dice_vertebra_1'], .1)
        self.assertIsNone(history[0]['val/patch_dice_disc_201'])
        self.assertEqual(history[0]['train/epoch_seconds'], 10)
        self.assertEqual(obj.current_epoch, 1)


if __name__ == '__main__':
    unittest.main()
