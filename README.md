# Beyond Dice

Quantitative-Biomarker Validation of Foundation-Model Segmentation for Lumbar Spine MRI.

## Training workflow

The integrated SPIDER baseline targets nnU-Net 2.8.1 in the existing `.venvs/nnunet/`.
Follow [the canonical nnU-Net guide](training/nnunet/README.md) for setup, smoke training,
full baseline, resumption and internal full-volume validation.
MedSAM remains a separate scaffold in its own environment.

The verified internal split is **289 scans / 143 patients** for training and
**71 scans / 36 patients** for development validation. The **87 official validation
scans / 39 patients** remain excluded from training and model selection.
The split has been installed; existing preprocessing was reused. No GPU training,
inference or live W&B upload was started during integration.

## Layout

```text
training/
  nnunet/
    data/spider_split.py             # Canonical split derivation and installation
    trainers/nnUNetTrainerSPIDER.py   # Baseline and separate smoke trainer
    configs/spider_baseline.json     # Approved split hash/counts and experiment IDs
    verify_setup.py                  # Read-only data and CPU API verification
  medsam/                           # Fine-tuning scaffold, separate environment
  common/                           # Shared framework-independent helpers
scripts/
  setup_nnunet.ps1                   # External discovery, paths, Windows execution settings
  preprocess_spider_masks.py         # .mha semantic mask utility
  split_spider_dataset.py            # Historical official split copier
tests/                              # unittest regression suite
requirements/                       # Environment-specific dependencies and locks
Models/nnUNet/                      # Ignored local raw/preprocessed/results directories
Models/MedSAM/                      # Ignored local pretrained weights
outputs/                            # Ignored reports and original draft archive
external/                           # Ignored upstream checkouts
.venvs/                             # Ignored isolated environments
```

The active nnU-Net dataset is `Models/nnUNet/nnUNet_raw/Dataset011_SPIDER/`.
Source code is kept outside the ignored data/results directories.
Earlier `Data/nnUNet/` instructions are historical.

## Safe checks (no training or W&B upload)

```powershell
. .\scripts\setup_nnunet.ps1
.\.venvs\nnunet\Scripts\python.exe -m training.nnunet.verify_setup --csv "D:\Courses\MD5001\Mini Project\SPIDERDataset\overview.csv"
.\.venvs\nnunet\Scripts\python.exe -m unittest discover -s tests -v
```

Use the explicitly named interpreter rather than relying on shell activation.
Set up external trainer discovery again in prediction/validation shells.

## Documentation

- [Canonical nnU-Net commands and study split](training/nnunet/README.md)
- [Environment setup](ENVIRONMENTS.md)
- [MedSAM layout](training/medsam/README.md)
- [SPIDER preprocessing](SPIDER_PREPROCESSING.md)
- [Initial real-data validation](SPIDER_VALIDATION.md)
- [Project history](PROJECT_HISTORY.md)
