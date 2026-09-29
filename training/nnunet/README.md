# SPIDER nnU-Net baseline: canonical workflow

Run from the BeyondDice repository root using `.venvs/nnunet/Scripts/python.exe`.
Integration was checked on **2026-09-30** against **nnunetv2 2.8.1**,
**torch 2.6.0+cu124**, and **wandb 0.30.0**. CUDA is available on the
NVIDIA RTX 2000 Ada. Neither training nor inference nor a live W&B run was started.

## Integrated files

- `data/spider_split.py`: derive, validate and safely install the internal split.
- `configs/spider_baseline.json`: approved split hash, counts and experiment identifiers.
- `trainers/nnUNetTrainerSPIDER.py`: baseline and smoke classes.
- `verify_setup.py`: dataset checks and CPU-only trainer construction.
- [scripts/setup_nnunet.ps1](../../scripts/setup_nnunet.ps1): PowerShell environment setup.
- `tests/test_spider_split.py` and `tests/test_spider_trainer.py`: regression tests.

The supplied draft is preserved locally in `outputs/integration_archive/spider_next_steps/`.
The external-trainer mechanism is used; installed nnU-Net source files were not edited.
Both training and inference require the same setup script so the custom class remains discoverable.

## Experimental split

| Partition | Patients | Scans | Use |
| --- | ---: | ---: | --- |
| Internal training | 143 | 289 | Gradient updates |
| Internal development validation | 36 | 71 | Checkpoint selection and development |
| Official validation | 39 | 87 | Held out until the model and analysis are fixed |

The official training pool contains 179 patients and 360 scans. Patient identity is
the positive integer prefix of `<patient>_t1`, `<patient>_t2`, or
`<patient>_t2_SPACE`; no patient overlaps either internal partition or official validation.
All sequences for one patient stay together, with each sequence a separate single-channel MRI case.

The draft's proposed counts and exact membership were independently reproduced from local
`overview.csv`. Patients are sorted by SHA256 of `42:<patient>`; the first
ceil(179 * 0.2) = 36 patients form development validation. This is a deterministic
grouped holdout, not pathology, sex, sequence or scanner stratification. The split seed
does not make stochastic nnU-Net training deterministic.

The exact split is installed at
`Models/nnUNet/nnUNet_preprocessed/Dataset011_SPIDER/splits_final.json`.
Its aggregate provenance is in `spider_split_report.json` beside it.
Only **fold 0** is permitted: `all` and other folds are rejected. The trainer checks
the approved split hash and exact preprocessed identifier coverage each time.

The 87 official validation scans are excluded from training, patch validation,
checkpoint selection and tuning. They are not passed to the commands below.

## Existing preprocessing

The actual dataset is `Dataset011_SPIDER` under `Models/nnUNet/`, superseding
earlier `Data/nnUNet/` examples. All 360 imagesTr/labelsTr pairs match the official
training CSV rows exactly; images are `<case>_0000.mha` and masks `<case>.mha`.
Labels: background 0, vertebrae 1, intervertebral discs 2, spinal canal 3.

Preprocessing already exists for 2d, 3d_lowres and 3d_fullres, each with 360 cases.
Do not rerun preprocessing for this integration. The plans also define
3d_cascade_fullres through inheritance; the baseline uses **3d_fullres**, not a cascade.
Its resolved plan has batch size 2, patch size [64, 192, 192], and spacing approximately
[0.6373272, 0.6373172, 0.625] in the plan's axis order.

## Setup and local checks

```powershell
. .\scripts\setup_nnunet.ps1
.\.venvs\nnunet\Scripts\python.exe -m training.nnunet.data.spider_split --csv "D:\Courses\MD5001\Mini Project\SPIDERDataset\overview.csv"
.\.venvs\nnunet\Scripts\python.exe -m training.nnunet.verify_setup --csv "D:\Courses\MD5001\Mini Project\SPIDERDataset\overview.csv"
.\.venvs\nnunet\Scripts\python.exe -m unittest discover -s tests -v
```

Setup uses repository-relative paths, sets `nnUNet_extTrainer` and `PYTHONPATH`,
disables compilation on Windows, sets two augmentation workers, and disables the
upstream W&B logger. Existing nnU-Net/PyTorch versions are retained.
W&B was added with existing dependency constraints; dependencies are pinned in
`requirements/nnunet-lock.txt`. No credentials belong in config files.

Split installation accepts an identical existing split. A different split is preserved
and the installer lists potentially dependent checkpoints before stopping. Do not
replace it under existing runs. No previous split or checkpoints were present at installation.

## Smoke test: next execution

```powershell
. .\scripts\setup_nnunet.ps1
$env:WANDB_MODE = 'offline'
$env:WANDB_NAME = 'SPIDER-3d-fullres-smoke'
.\.venvs\nnunet\Scripts\nnUNetv2_train.exe 11 3d_fullres 0 -tr nnUNetTrainerSPIDER_5epochs --val_best
```

This is an actual GPU run: 5 epochs, 10 training and 5 development iterations per epoch,
with saving every epoch. The standard CLI then performs full-volume validation on the
**71 internal development cases**, which can take substantially longer than training.
It uses checkpoint_best.pth because of `--val_best`.
The smoke trainer's class name creates a separate output folder; never use smoke weights
as the baseline. Inspect checkpoints, patch curves and `validation/summary.json`.

## Full baseline

After the smoke run succeeds:

```powershell
. .\scripts\setup_nnunet.ps1
$env:WANDB_MODE = 'offline'
$env:WANDB_NAME = 'SPIDER-3d-fullres-baseline'
.\.venvs\nnunet\Scripts\nnUNetv2_train.exe 11 3d_fullres 0 -tr nnUNetTrainerSPIDER --val_best
```

The base architecture, loss, SGD optimizer, learning-rate schedule, augmentations,
deep supervision and **1000 epochs / 250 train / 50 development iterations** remain
inherited unchanged. Compilation and worker count are execution settings.
These memory settings still need verification during the first actual GPU run.

Outputs are in:
`Models/nnUNet/nnUNet_results/Dataset011_SPIDER/nnUNetTrainerSPIDER__nnUNetPlans__3d_fullres/fold_0/`.
The smoke model folder instead starts with `nnUNetTrainerSPIDER_5epochs__`.

## Resume

For an interrupted baseline with an existing checkpoint:

```powershell
. .\scripts\setup_nnunet.ps1
$env:WANDB_MODE = 'offline'
.\.venvs\nnunet\Scripts\nnUNetv2_train.exe 11 3d_fullres 0 -tr nnUNetTrainerSPIDER --c --val_best
```

Use the same results root, plans, configuration, fold, trainer, and approved split.
The local run identity guards these plus the schedule. A resume without a loaded
checkpoint fails rather than silently starting fresh. Existing runs cannot be
replaced by a non-resume invocation. A failure before the first checkpoint requires
a fresh results root (retain the failed run for inspection).

nnU-Net's checkpoint search order is final, latest, then best. Training resumes at
the saved next epoch. A completed checkpoint skips the training loop and proceeds
to CLI validation. Do not launch simultaneous processes into the same output folder.

The installed 2.8.1 loader has a broken dictionary overload; this trainer explicitly
accepts checkpoint filenames/Paths only, which is what all documented CLI commands use.
The standard predictor independently restores checkpoint weights and discovers this class.

## Internal full-volume validation

The training command already runs this after completion. To recover validation after
a completed training run, use the following only when no validation output exists:

```powershell
. .\scripts\setup_nnunet.ps1
$validationDir = Join-Path $env:nnUNet_results 'Dataset011_SPIDER/nnUNetTrainerSPIDER__nnUNetPlans__3d_fullres/fold_0/validation'
if (Test-Path -LiteralPath $validationDir) { throw 'Validation outputs already exist; inspect/preserve them before a deliberate rerun.' }
.\.venvs\nnunet\Scripts\nnUNetv2_train.exe 11 3d_fullres 0 -tr nnUNetTrainerSPIDER --val --val_best
```

This requires checkpoint_final.pth, then loads checkpoint_best.pth for validation.
For the smoke experiment substitute its trainer name in the path and command.
This is internal development evaluation only. Full-volume predictions and summary
metrics are local; validation-only execution never initializes this W&B logger.

## W&B logging and resumption

Offline is the default. To explicitly enable live uploads for a future run, authenticate
interactively using `.\.venvs\nnunet\Scripts\python.exe -m wandb login`, then set
`$env:WANDB_MODE = 'online'`. Set `WANDB_ENTITY` only if required.
Use `WANDB_MODE=disabled` to retain local config snapshots without a W&B run.

The integration sends aggregate allowlisted training settings, architecture/resampling
settings, optimizer/scheduler settings, version information, split counts/hash,
implementation hashes and epoch metrics. It does not upload dataset dictionaries,
case lists, CSV rows, patient metadata, image/mask volumes, artifacts, credentials
or arbitrary files. Automatic console capture, code/Git capture, machine metadata
and system monitoring are disabled. Use non-sensitive run names.
Resolved configs are saved locally as `spider_resolved_config_epoch<epoch>_<id>.json`.

Online resumes require the recorded run ID and unchanged W&B project/entity, using
`resume='must'`. Offline resumes create separate local segments and never replace
the online identity file. Resuming offline training online without a recorded online
identity is rejected. W&B cannot rewind history when nnU-Net resumes an older
checkpoint: already-logged epoch steps are retained/skipped until training catches up.
Recovered computation may differ because baseline training is stochastic.

Epoch indices are zero-based. Epoch train loss, validation patch loss, foreground
patch Dice, EMA patch Dice, per-class patch Dice, learning rate and elapsed seconds
are logged. Absent-class NaN Dice is represented as null, not falsely reported as zero.
Best checkpoint selection remains **EMA foreground patch Dice**. This is not
full-volume Dice. Full-volume validation follows after the W&B training run closes
and writes `validation/summary.json` locally. HD95, ASSD and surface Dice remain
future evaluation work.

W&B closes on success or failure; augmentation workers are cleaned up on errors.
Cleanup failures do not replace the original training exception.

## Legacy scripts and verification limits

`scripts/preprocess_spider_masks.py` is the original-mask conversion utility and
currently outputs .mha. `scripts/split_spider_dataset.py` copies the official
train/validation partitions; it does not create the internal holdout and should not
be rerun on the populated split. `scripts/json_create.py` is retired: its old
import-time write and obsolete path are preserved only in the local integration archive.
The supplied package's commands are superseded by this document.

Verified locally: syntax/imports, 28 regression tests, deterministic local CSV membership,
idempotent split installation, raw/preprocessed coverage, both external trainer classes,
real logger API and epoch indexing, a CPU checkpoint save/load without forward/backward,
predictor initialization using a synthetic checkpoint and a stub network (no prediction),
resume guards, SDK settings and mocked W&B cleanup.
Real GPU training, actual model checkpoint recovery, memory use, live/offline W&B runtime,
authentication/upload and full-volume inference remain unexecuted.
