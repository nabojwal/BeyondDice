"""Synthetic on-disk checks for SPIDER semantic mask preprocessing."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from scripts.preprocess_spider_masks import convert_spider_masks


class SpiderMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.source.mkdir()
        self.output = self.root / 'output'

    def write_mask(self, name, values):
        image = sitk.GetImageFromArray(np.asarray(values).reshape(1, 1, -1))
        image.SetSpacing((0.7, 1.3, 2.9))
        image.SetOrigin((-12.3, 4.5, 67.8))
        angle = 0.3
        image.SetDirection((np.cos(angle), -np.sin(angle), 0.,
                            np.sin(angle), np.cos(angle), 0., 0., 0., 1.))
        path = self.source / name
        sitk.WriteImage(image, str(path))
        return path, image

    def test_mapping_geometry_order_and_source_preservation(self):
        values = [0] + list(range(1, 26)) + list(range(201, 226)) + [100]
        path, source = self.write_mask('100_t1mask.mha', values)
        original = path.read_bytes()
        self.write_mask('099.mha', [0])
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            outputs = convert_spider_masks(self.source, self.output)
        self.assertEqual([p.name for p in outputs], ['099.mha', '100_t1.mha'])
        self.assertEqual(report.getvalue().splitlines(), [
            '099.mha: labels [0]', '100_t1.mha: labels [0, 1, 2, 3]'])
        saved = sitk.ReadImage(str(outputs[1]))
        self.assertEqual(saved.GetPixelID(), sitk.sitkUInt8)
        self.assertEqual(saved.GetSize(), source.GetSize())
        for getter in ['GetSpacing', 'GetOrigin', 'GetDirection']:
            np.testing.assert_allclose(getattr(saved, getter)(), getattr(source, getter)(),
                                       rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(saved).ravel(),
                                      [0] + [1] * 25 + [2] * 25 + [3])
        self.assertEqual(path.read_bytes(), original)
        self.assertIn(b'CompressedData = True', outputs[1].read_bytes()[:1024])

    def test_non_3d_masks_rejected(self):
        for dimension in [2, 4]:
            with self.subTest(dimension=dimension):
                image = sitk.Image([2] * dimension, sitk.sitkUInt8)
                sitk.WriteImage(image, str(self.source / 'bad.mha'))
                with self.assertRaisesRegex(ValueError, 'bad.mha: expected a 3D mask'):
                    convert_spider_masks(self.source, self.output)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_vector_mask_rejected(self):
        image = sitk.GetImageFromArray(np.zeros((2, 2, 2, 3), dtype=np.uint8),
                                       isVector=True)
        sitk.WriteImage(image, str(self.source / 'vector.mha'))
        with self.assertRaisesRegex(ValueError, 'vector.mha: expected a scalar mask'):
            convert_spider_masks(self.source, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_saved_voxel_mismatch_rejected(self):
        self.write_mask('case.mha', [0, 1, 201, 100])
        write_image = sitk.WriteImage

        def write_corrupted(image, filename, compress):
            array = sitk.GetArrayFromImage(image)
            # Preserve the label set, dtype, and geometry but swap two classes.
            array[0, 0, 1], array[0, 0, 2] = array[0, 0, 2], array[0, 0, 1]
            corrupted = sitk.GetImageFromArray(array)
            corrupted.CopyInformation(image)
            write_image(corrupted, filename, compress)

        with patch('scripts.preprocess_spider_masks.sitk.WriteImage',
                   side_effect=write_corrupted):
            with self.assertRaisesRegex(ValueError, 'saved voxel labels differ from conversion'):
                convert_spider_masks(self.source, self.output)

    def test_empty_input(self):
        with self.assertRaisesRegex(FileNotFoundError, r'No \*\.mha files found'):
            convert_spider_masks(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_destinations_preflight(self):
        self.write_mask('casemask.mha', [0])
        self.write_mask('case.mha', [0])
        with self.assertRaisesRegex(ValueError, 'Duplicate destination'):
            convert_spider_masks(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_destination_preflight(self):
        self.write_mask('a.mha', [0])
        self.write_mask('zmask.mha', [0])
        self.output.mkdir()
        existing = self.output / 'z.mha'
        existing.write_bytes(b'preserve this')
        with self.assertRaises(FileExistsError):
            convert_spider_masks(self.source, self.output)
        self.assertEqual(existing.read_bytes(), b'preserve this')
        self.assertFalse((self.output / 'a.mha').exists())

    def test_unexpected_labels(self):
        for value in [-1, 26, 101, 200, 226, 1.5, 201.5, float('nan')]:
            with self.subTest(value=value):
                self.write_mask('bad.mha', [value])
                with self.assertRaisesRegex(ValueError, 'bad.mha: unexpected labels'):
                    convert_spider_masks(self.source, self.output)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_midway_failure_keeps_earlier_output(self):
        self.write_mask('a.mha', [100])
        self.write_mask('b.mha', [26])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                convert_spider_masks(self.source, self.output)
        self.assertTrue((self.output / 'a.mha').exists())
        self.assertFalse((self.output / 'b.mha').exists())


if __name__ == '__main__':
    unittest.main()
