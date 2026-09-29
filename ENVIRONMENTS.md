# Research Environments

Project: **Beyond Dice: Quantitative-Biomarker Validation of Foundation-Model Segmentation for Lumbar Spine MRI**

Three isolated environments live under `.venvs/`. They use the installed Python 3.11.9 on native Windows. The local GPU is an NVIDIA RTX 2000 Ada with 16 GB VRAM and driver 551.61.

| Environment | Purpose | Framework selection |
| --- | --- | --- |
| `nnunet` | nnU-Net v2 preprocessing, training, inference | nnunetv2 2.8.1; PyTorch 2.6.0 / torchvision 0.21.0, CUDA 12.4 |
| `medsam` | Original MedSAM preprocessing, fine-tuning, inference | PyTorch 2.0.1 / torchvision 0.15.2, CUDA 11.8; NumPy 1.26.4; MONAI 1.3.2 |
| `evaluation` | Shared MRI measurements, segmentation metrics, statistics, plotting | SimpleITK, NiBabel, pydicom, SciPy, scikit-image, pandas, surface-distance, statsmodels, Pingouin |

MedSAM is pinned to official source revision `b9db486e77bf9a2c9c11cc061c63f54110d520b6`, before the custom installer that added NumPy 2 and CUDA 12 constraints. Its source checkout is in `external/MedSAM/`. This is original MedSAM, not MedSAM2. Python 3.11 is used instead of the README's suggested 3.10; local smoke checks establish compatibility for the exercised paths.

## Select an environment

From PowerShell in the project folder, activate one environment:

```powershell
. .\.venvs\nnunet\Scripts\Activate.ps1
# Or:
. .\.venvs\medsam\Scripts\Activate.ps1
# Or:
. .\.venvs\evaluation\Scripts\Activate.ps1
```

Use `deactivate` before switching. Activation is optional: you can invoke `.venvs\<name>\Scripts\python.exe` directly. In VS Code, use **Python: Select Interpreter**, then choose that executable.

## Notebooks

Launch JupyterLab with the evaluation interpreter to access all three locally registered kernels:

```powershell
.\.venvs\evaluation\Scripts\python.exe -m jupyterlab
```

Choose **Beyond Dice (nnunet)**, **Beyond Dice (medsam)**, or **Beyond Dice (evaluation)**. Kernel registrations are stored within the evaluation environment and must be recreated if that environment is rebuilt or the project is moved.

## nnU-Net working directories

Before running nnU-Net commands, set its working directories in the current PowerShell session:

```powershell
$env:nnUNet_raw = Join-Path $PWD 'Models/nnUNet/nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $PWD 'Models/nnUNet/nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $PWD 'Models/nnUNet/nnUNet_results'
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw, $env:nnUNet_preprocessed, $env:nnUNet_results
$env:nnUNet_compile = 'false'
```

Compilation is disabled for the native Windows training setup. Dataset conversion, training plans, and batch sizes still depend on the actual MRI data and available GPU memory.

## Reproducibility

- `requirements/<name>.txt` describes the requested packages.
- `requirements/*-constraints.txt` protects the selected model stacks.
- `requirements/medsam-source.txt` records the exact upstream source revision.
- `requirements/<name>-lock.txt` records all installed package versions after validation.
- Environments, external source checkouts, data, checkpoints, and outputs are excluded from Git.

To rebuild, create a Python 3.11 environment and install the corresponding lock file. For nnU-Net and MedSAM, install the GPU frameworks first from their official indexes:

```powershell
# Run with the appropriate environment's Python executable.
# nnunet:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# medsam:
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# Then use the matching lock file, e.g.:
python -m pip install -r requirements/nnunet-lock.txt
```

The MedSAM lock includes a Git dependency; Git and internet access are required when rebuilding. Lock files are specific to this Windows/Python setup, not portable guarantees for every OS.

## Verification

Validated locally on 2026-09-29: dependency checks passed for all three environments; all runtime checks described below passed. GPU checks used the RTX 2000 Ada. Pandas 2.2.3 is pinned in nnU-Net and evaluation because Windows Application Control blocked a pandas 3.0.6 binary. Evaluation uses compatible Pingouin 0.5.5. MedSAM emits an upstream MONAI `pkg_resources` deprecation warning; setuptools is pinned below 81 in its version snapshot.


Run `.venvs\<name>\Scripts\python.exe scripts\verify_environment.py <name>` to check the corresponding environment, and `python -m pip check` using that environment's interpreter to check dependency consistency.

The checks exercise MRI arrays, CUDA forward/backward operations for both model environments, a synthetic MedSAM forward pass with random weights, and surface Dice / intraclass correlation calculations in the evaluation environment. They do not establish model accuracy or validate a full training run.

Pretrained checkpoints and research datasets are not bundled. MedSAM requires a checkpoint for meaningful predictions; nnU-Net requires trained weights or training on the selected dataset. Biomarker definitions and the study-specific evaluation pipeline remain to be implemented.

## Upstream references

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [Pinned MedSAM revision](https://github.com/bowang-lab/MedSAM/tree/b9db486e77bf9a2c9c11cc061c63f54110d520b6)
- [Official PyTorch version combinations](https://pytorch.org/get-started/previous-versions/)

## SPIDER mask preprocessing

See [SPIDER_PREPROCESSING.md](SPIDER_PREPROCESSING.md) for semantic target conversion and verification.

## Training source layout

Project training code belongs in `training/nnunet/` and `training/medsam/`.
See [README.md](README.md) for the source and artifact layout. These packages
provide the integrated nnU-Net baseline and a MedSAM scaffold. Follow
[the canonical nnU-Net workflow](training/nnunet/README.md) for external trainer setup
and exact smoke, baseline, resume and validation commands.

W&B 0.30.0 was added to nnunet on 2026-09-30 under existing dependency constraints.
PyTorch and nnU-Net versions were unchanged; pip check passed. No W&B run was started.
