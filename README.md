# Beyond Dice

**A Comparative Analysis Framework for Segmentation and Biomarker Validation in Lumbar Spine MRI**

Research code for comparing **MedSAM** and **nnU-Net** on lumbar spine MRI from the SPIDER dataset, with a focus on segmentation quality and the reliability of downstream intervertebral disc (IVD) measurements.

## Repository structure

- **[MedSAM/](MedSAM/)** — MedSAM implementation, adaptation and evaluation workflows.
- **[nnUnet/](nnUnet/)** — nnU-Net training, SPIDER preprocessing and held-out evaluation workflows.

See the documentation inside each folder for model-specific setup and execution instructions.

## Experiment tracking

**[Weights & Biases — Beyond Dice report](https://forge.coreweave.com/wandb/nabojwal_dl1/BeyondDice-SPIDER/reports/Beyond-Dice--VmlldzoxODEzNzQ3Ng)**

## License

Original BeyondDice project code is licensed under the [GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`), unless otherwise indicated. Third-party components retain their original licenses; the bundled [MedSAM source](MedSAM/MedSAM/LICENSE), for example, is distributed under the Apache License 2.0.
