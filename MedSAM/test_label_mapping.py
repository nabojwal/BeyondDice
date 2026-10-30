import unittest
import numpy as np
from inspect_prepare_spider_masks import to_semantic, label_counts

class TestLabels(unittest.TestCase):
    def test_mapping(self):
        arr = np.array([0, 1, 25, 100, 201, 225], dtype=np.int16).reshape(1, 2, 3)
        self.assertEqual(to_semantic(arr).flatten().tolist(), [0, 1, 1, 3, 2, 2])
        self.assertEqual(to_semantic(arr).dtype, np.uint8)
        self.assertEqual(label_counts(arr)[201], 1)
    def test_unknown(self):
        with self.assertRaises(ValueError):
            to_semantic(np.array([999]).reshape(1, 1, 1))
    def test_noninteger(self):
        with self.assertRaises(TypeError):
            label_counts(np.zeros((1, 1, 1), dtype=np.float32))
    def test_original_unchanged(self):
        arr = np.array([1, 2, 201, 202, 100], dtype=np.int16).reshape(1, 1, 5)
        before = arr.copy()
        to_semantic(arr)
        np.testing.assert_array_equal(arr, before)

if __name__ == '__main__':
    unittest.main(verbosity=2)
