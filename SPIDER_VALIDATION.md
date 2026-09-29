# Real SPIDER preprocessing validation

Validated on 2026-09-29 with three real masks from
`D:/Courses/MD5001/Mini Project/SPIDERDataset/masks`.
Outputs are in the sibling `semantic_test_masks` directory.

The CLI expects a directory, so the selected masks were copied to a temporary
input directory. Original mask SHA-256 hashes were unchanged after validation.

| Case | Background (0) | Vertebrae (1) | IVDs (2) | Canal (3) |
| --- | ---: | ---: | ---: | ---: |
| 1_t1 | 12,659,855 | 220,640 | 32,406 | 34,299 |
| 100_t1 | 7,882,170 | 261,871 | 46,801 | 43,762 |
| 100_t2 | 7,882,170 | 261,871 | 46,801 | 43,762 |

All output counts exactly match the combined input counts for their classes.
An independent lookup table verified every voxel: 0 -> 0, 1-25 -> 1,
201-225 -> 2, and 100 -> 3. All outputs are 3D uint8 and preserve source
geometry within the documented floating-point tolerances. Their dimensions,
spacing, origin, and direction also match the corresponding MRI volumes.

A second invocation with the same input and output directories raised
FileExistsError before conversion; output hashes remained unchanged.

The implementation explicitly requires GetDimension() == 3 and compares
the saved array against the converted array using np.array_equal.

## 3D Slicer overlay spot-check

In 3D Slicer 5.12.4, each saved labelmap was overlaid on its corresponding
MRI. Slicer-composited sagittal, axial, and coronal slice images were exported
and visually inspected at a foreground-centered location for each case.
Red denotes vertebrae, green discs, and blue spinal canal. No visible global
shift or flip was observed; overlays follow the corresponding anatomy.
This is a spot-check, not an exhaustive review of all slices or annotation quality.
Optional Slicer modules were disabled for this run because an unrelated
installed extension failed during initial startup.

Local artifacts (ignored by Git):
- [Numerical report](outputs/spider_validation/report.json)
- [Overlay contact sheet](outputs/spider_validation/overlay_review.jpg)
- [Validation runner](outputs/spider_validation/validate_real.py)
- [Slicer capture script](outputs/spider_validation/slicer_overlays.py)

The validation runner intentionally refuses a repeat conversion into the
now-populated destination. Select a fresh output directory for another run.

The existing nine synthetic tests cover additional failure cases and label
values absent from these scans. These three scans do not validate the entire dataset.
