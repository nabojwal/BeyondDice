# SPIDER semantic segmentation targets

`scripts/preprocess_spider_masks.py` merges SPIDER instances into compressed
`uint8` MetaImage masks for nnU-Net semantic segmentation:

| Original values | Output label | nnU-Net label name |
| --- | --- | --- |
| 0 | 0 | background |
| 1-25 | 1 | vertebrae (whole vertebrae, including vertebral arches) |
| 201-225 | 2 | intervertebral discs |
| 100 | 3 | spinal canal |

Instance identities are removed in the outputs. Original `.mha` masks are
preserved. Separate radiological annotations, including Pfirrmann and Modic
grades, are not read or modified. This utility prepares segmentation targets
only; it does not create a complete nnU-Net dataset or configure training.

## Usage

The nnU-Net environment includes the required `SimpleITK` and `numpy` dependencies.
For a standalone environment, install them with `python -m pip install SimpleITK numpy`.
From the repository root:

```powershell
.\.venvs\nnunet\Scripts\python.exe scripts\preprocess_spider_masks.py --input-dir data\spider\masks --output-dir outputs\semantic_masks
```

The original Kaggle paths can be supplied explicitly:

```sh
python scripts/preprocess_spider_masks.py --input-dir /kaggle/input/datasets/nabojwalacharjee/spider/masks/masks/ --output-dir /kaggle/working/semantic_masks
```

Or import the function from the repository root:

```python
from scripts.preprocess_spider_masks import convert_spider_masks

output_paths = convert_spider_masks('data/spider/masks', 'outputs/semantic_masks')
```

The utility searches non-recursively for `*.mha`, processes sorted paths, and
raises `FileNotFoundError` for empty input. A trailing `mask` is removed from
each stem: `100_t1mask.mha` becomes `100_t1.mha`. Duplicate destinations and
all existing destination paths are rejected before conversion starts.
Inputs must be scalar 3D masks. Unexpected labels (including fractional values)
are rejected.

Each saved mask is reopened to verify dimensions, spacing, origin, direction,
`uint8` storage, exact voxel-by-voxel equality with the converted array, and labels
restricted to `{0, 1, 2, 3}`. Geometry comparisons
allow floating-point serialization differences (`rtol=1e-6`, `atol=1e-5`).
Individual scans need not contain every class. Each generated filename and its
unique saved labels are printed.

Conversion is **not transactional**: an invalid mask or write/verification failure
midway may leave earlier outputs in place. Use a fresh destination for a rerun.

## Verification

```powershell
.\.venvs\nnunet\Scripts\python.exe -m unittest discover -s tests -v
```

Synthetic tests cover the full label mapping, geometry after saving, compression,
source preservation, sorted processing, missing classes, empty input, duplicate
destinations, overwrite prevention, invalid labels, non-3D/vector inputs, saved
voxel corruption, and partial progress on failure.


## Real-data validation

See [SPIDER_VALIDATION.md](SPIDER_VALIDATION.md) for the three-mask real-data
check, exact class counts, overwrite test, and 3D Slicer overlay spot-check.

The current converter writes compressed .mha to match the active nnU-Net dataset.
For training and the internal patient split, use [the canonical workflow](training/nnunet/README.md).
Earlier NIfTI validation results remain documented as historical checks.
