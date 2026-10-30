# Beyond Dice

**Quantitative-Biomarker Validation of Foundation-Model Segmentation for Lumbar Spine MRI**

Research code for comparing **MedSAM** and **nnU-Net** on lumbar spine MRI from the SPIDER dataset, with a focus on segmentation quality and the reliability of downstream intervertebral disc (IVD) measurements.

## Repository structure

- **[MedSAM/](MedSAM/)** — MedSAM implementation, adaptation and evaluation workflows.
- **[nnUnet/](nnUnet/)** — nnU-Net training, SPIDER preprocessing and held-out evaluation workflows.

See the documentation inside each folder for model-specific setup and execution instructions.

## Experiment tracking

**[Weights & Biases — Beyond Dice report](https://forge.coreweave.com/wandb/nabojwal_dl1/BeyondDice-SPIDER/reports/Beyond-Dice--VmlldzoxODEzNzQ3Ng)**

Viewing the report may require W&B access permissions.

## Data and reproducibility

The experiments use the **SPIDER lumbar spine MRI dataset**. Dataset volumes, trained checkpoints and local experimental artifacts are not included in this repository; refer to the component directories for configuration and workflow details.
