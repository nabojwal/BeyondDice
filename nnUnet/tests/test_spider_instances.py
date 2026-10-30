import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import SimpleITK as sitk
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import preprocess_spider_masks as masks
import prepare_spider_instances as prep
from training.nnunet.data.spider_instances_split import validate_split


def write(path, values=(0, 1, 2, 100, 201, 202)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.resize(np.array(values, dtype=np.uint16), (3, 4, 5)))
    image.SetSpacing((.5, .6, 2.))
    image.SetOrigin((1., 2., 3.))
    image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    sitk.WriteImage(image, str(path))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_roundtrip_preserves_instances_and_geometry(self):
        source = self.root / 'source.mha'
        write(source)
        mapping = masks.fit_mapping([source])
        masks.convert_mask(source, self.root / 'encoded.mha', mapping)
        masks.convert_mask(self.root / 'encoded.mha', self.root / 'restored.mha', mapping, True)
        a, aa = masks.read_mask(source)
        b, bb = masks.read_mask(self.root / 'restored.mha')
        self.assertTrue(np.array_equal(aa, bb))
        self.assertTrue(masks.same_geometry(a, b))
        self.assertEqual(len(mapping['labels']), 6)
        self.assertEqual(set(mapping['labels'].values()), set(range(6)))

    def test_unknown_test_label_rejected(self):
        write(self.root / 'train.mha', (0, 1, 100, 201))
        write(self.root / 'test.mha', (0, 2))
        mapping = masks.fit_mapping([self.root / 'train.mha'])
        with self.assertRaisesRegex(ValueError, 'Unknown labels'):
            masks.convert_mask(self.root / 'test.mha', self.root / 'out.mha', mapping)

    def test_mapping_tampering_rejected(self):
        write(self.root / 'a.mha')
        mapping = masks.fit_mapping([self.root / 'a.mha'])
        mapping['source_to_encoded']['2'] = 1
        with self.assertRaises(ValueError):
            masks.validate_mapping(mapping)

    def test_fractional_masks_rejected(self):
        sitk.WriteImage(sitk.GetImageFromArray(np.full((2, 2, 2), 1.5)), str(self.root / 'a.mha'))
        with self.assertRaises(ValueError):
            masks.read_mask(self.root / 'a.mha')

    def test_unrecognized_label_rejected(self):
        write(self.root / 'a.mha', (0, 101))
        with self.assertRaises(ValueError):
            masks.fit_mapping([self.root / 'a.mha'])

    def test_split_grouped_and_deterministic(self):
        cases = [f'{i}_{s}' for i in range(1, 11) for s in ('t1', 't2')]
        a = prep.split_cases(cases)
        self.assertEqual(a, prep.split_cases(list(reversed(cases))))
        train, val = validate_split(a, cases)
        self.assertEqual(len(val), 4)
        self.assertFalse({prep.group(c) for c in train} & {prep.group(c) for c in val})

    def test_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_split([{'train': ['1_t1'], 'val': ['1_t2']}])

    def setup_data(self):
        for root, groups in [('train', range(1, 6)), ('test', range(10, 12))]:
            for i in groups:
                for seq in ['t1', 't2']:
                    write(self.root / root / 'images' / f'{i}_{seq}.mha')
                    write(self.root / root / 'masks' / f'{i}_{seq}mask.mha')
        env = {k: str(self.root / k) for k in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']}
        args = types.SimpleNamespace(train=self.root / 'train', test=self.root / 'test', fraction=.2, seed=42)
        return env, args

    def test_prepare_end_to_end_and_collision(self):
        env, args = self.setup_data()
        original = prep.sha_file(self.root / 'train/masks/1_t1mask.mha')
        with patch.dict(os.environ, env):
            prep.prepare(args)
            raw = Path(env['nnUNet_raw']) / prep.NAME
            dataset = prep.read_json(raw / 'dataset.json')
            self.assertEqual(dataset['numTraining'], 10)
            self.assertEqual(len(list((raw / 'imagesTs').glob('*.mha'))), 4)
            validate_split(prep.read_json(raw / 'splits_final.json'), prep.paired(args.train))
            self.assertEqual(original, prep.sha_file(self.root / 'train/masks/1_t1mask.mha'))
            with self.assertRaises(FileExistsError):
                prep.prepare(args)

    def test_finalize_checks_preprocessing_and_detects_changed_raw(self):
        env, args = self.setup_data()
        class Manager:
            def __init__(self, plans):
                self.plans = plans
            def get_configuration(self, name):
                return types.SimpleNamespace(data_identifier='nnUNetPlans_3d_fullres')
        data = types.ModuleType('nnunetv2.training.dataloading.nnunet_dataset')
        data.infer_dataset_class = lambda folder: types.SimpleNamespace(
            get_identifiers=lambda path: list(prep.paired(args.train)))
        plans_module = types.ModuleType('nnunetv2.utilities.plans_handling.plans_handler')
        plans_module.PlansManager = Manager
        with patch.dict(os.environ, env), patch.dict(sys.modules, {data.__name__: data, plans_module.__name__: plans_module}):
            prep.prepare(args)
            raw = Path(env['nnUNet_raw']) / prep.NAME
            pre = Path(env['nnUNet_preprocessed']) / prep.NAME
            pre.mkdir(parents=True)
            prep.write_json(pre / 'dataset.json', prep.read_json(raw / 'dataset.json'))
            prep.write_json(pre / 'nnUNetPlans.json', {'dataset_name': prep.NAME})
            prep.finalize(types.SimpleNamespace())
            self.assertEqual(prep.read_json(pre / 'spider_contract.json')['plans_sha256'],
                             prep.digest({'dataset_name': prep.NAME}))
            prep.finalize(types.SimpleNamespace())  # Identical installation is idempotent.
            write(raw / 'labelsTr/1_t1.mha', (0, 1))
            with self.assertRaisesRegex(ValueError, 'Raw training inputs changed'):
                prep.finalize(types.SimpleNamespace())

    def test_train_test_overlap_rejected(self):
        env, args = self.setup_data()
        write(args.test / 'images/1_t1.mha')
        write(args.test / 'masks/1_t1mask.mha')
        with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'overlap'):
            prep.prepare(args)

    def test_geometry_mismatch_rejected(self):
        env, args = self.setup_data()
        path = args.train / 'images/1_t1.mha'
        image = sitk.ReadImage(str(path))
        image.SetOrigin((99, 2, 3))
        sitk.WriteImage(image, str(path))
        with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'geometry'):
            prep.prepare(args)

    def test_unpaired_files_rejected(self):
        _, args = self.setup_data()
        (args.train / 'masks/1_t1mask.mha').unlink()
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            prep.paired(args.train)

    def test_duplicate_normalized_name_rejected(self):
        _, args = self.setup_data()
        write(args.train / 'images/1_t1_0000.mha')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prep.paired(args.train)


if __name__ == '__main__':
    unittest.main()
