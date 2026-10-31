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

Original code developed for BeyondDice is distributed under the GNU General Public License version 3.0 only (GPL-3.0-only), subject to the rights of its respective copyright holders. Third-party components, including MedSAM and code derived from other upstream projects, retain their applicable copyright notices, attribution requirements, and licenses. The repository-level license does not supersede those obligations.
