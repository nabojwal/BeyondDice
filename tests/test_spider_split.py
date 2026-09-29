"""Patient split regression tests; synthetic fixtures, no preprocessing or training."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from training.nnunet.data.spider_split import (
    digest, discover_dataset, install_split, make_split, patient_id, validate_split, verify_dataset)


class SpiderSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / 'overview.csv'
        self.rows = [(f'{p}_{sequence}', 'training') for p in range(1, 11) for sequence in ['t1', 't2']]
        self.rows += [('11_t2_SPACE', 'validation')]
        self.write_csv(self.rows)

    def write_csv(self, rows):
        with self.csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['new_file_name', 'subset'])
            writer.writerows(rows)

    def test_determinism_and_official_exclusion(self):
        splits, report = make_split(self.csv)
        self.write_csv(list(reversed(self.rows)))
        self.assertEqual(splits, make_split(self.csv)[0])
        train, val = validate_split(splits, expected_cases=[c for c, s in self.rows if s == 'training'])
        self.assertFalse({patient_id(c) for c in train} & {patient_id(c) for c in val})
        self.assertEqual((report['train_patients'], report['val_patients']), (8, 2))
        self.assertEqual(report['official_validation_patients_excluded'], 1)
        self.assertNotIn('11_t2_SPACE', train + val)

    def test_csv_errors(self):
        for rows in [[], self.rows + [self.rows[0]],
                     self.rows + [('1_t2_SPACE', 'validation')],
                     [('01_t1', 'training')], [('1_unknown', 'training')],
                     [('1_t1', 'test')]]:
            with self.subTest(rows=rows[-1:] if rows else []):
                self.write_csv(rows)
                with self.assertRaises(ValueError):
                    make_split(self.csv)

    def test_invalid_fraction(self):
        for fraction in [0, 1, -0.1, float('nan')]:
            with self.assertRaises(ValueError):
                make_split(self.csv, fraction)

    def test_validation_rejects_changes(self):
        split = [{'train': ['1_t1'], 'val': ['2_t1']}]
        validate_split(split, approved_hash=digest(split))
        for bad in [[], [{'train': ['1_t1'], 'val': ['1_t2']}],
                    [{'train': ['1_t1'], 'val': ['1_t1']}]]:
            with self.assertRaises(ValueError):
                validate_split(bad)
        with self.assertRaises(ValueError):
            validate_split(split, expected_cases=['3_t1'])
        with self.assertRaises(ValueError):
            validate_split(split, approved_hash='changed')

    def test_install_identical_and_preserve_conflict(self):
        splits, report = make_split(self.csv)
        install_split(self.root, splits, report)
        before = (self.root / 'splits_final.json').read_bytes()
        install_split(self.root, splits, report)
        self.assertEqual(before, (self.root / 'splits_final.json').read_bytes())
        altered = [{'train': splits[0]['val'], 'val': splits[0]['train']}]
        result = self.root / 'results' / 'Dataset011_TEST' / 'fold_0'
        result.mkdir(parents=True)
        checkpoint = result / 'checkpoint_best.pth'
        checkpoint.write_bytes(b'preserve')
        with self.assertRaisesRegex(FileExistsError, 'checkpoint_best'):
            install_split(self.root, altered, report, self.root / 'results', 'Dataset011_TEST')
        self.assertEqual(before, (self.root / 'splits_final.json').read_bytes())
        self.assertEqual(checkpoint.read_bytes(), b'preserve')

    def test_discovery_and_missing_pair(self):
        with self.assertRaises(ValueError):
            discover_dataset(self.root)
        raw = self.root / 'Dataset011_TEST'
        raw.mkdir()
        self.assertEqual(discover_dataset(self.root), raw)
        (raw / 'imagesTr').mkdir()
        (raw / 'labelsTr').mkdir()
        metadata = {'file_ending': '.mha', 'numTraining': 360,
                    'channel_names': {'0': 'MRI'},
                    'labels': {'background': 0, 'vertebrae': 1, 'intervertebral discs': 2, 'spinal canal': 3}}
        (raw / 'dataset.json').write_text(json.dumps(metadata))
        (raw / 'imagesTr' / '1_t1_0000.mha').touch()
        with self.assertRaisesRegex(ValueError, 'labelsTr'):
            verify_dataset(raw, self.root, [{'train': ['1_t1'], 'val': []}])
        (self.root / 'Dataset011_OTHER').mkdir()
        with self.assertRaises(ValueError):
            discover_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
