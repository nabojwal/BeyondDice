# MedSAM training code

Use `.venvs/medsam/` and the pinned original MedSAM source in
`external/MedSAM/`. See [environment details](../../ENVIRONMENTS.md).

| Location | Intended contents |
| --- | --- |
| `data/` | Slice datasets, image/mask transforms, and prompt construction |
| `trainers/` | Fine-tuning loops, losses, validation, and checkpoint handling |
| `configs/` | Reproducible experiment settings |
| Future `train.py` | Project fine-tuning entry point, when implemented |

Recommended local artifact paths, created when needed:

- Prepared data: `Data/MedSAM/`
- Pretrained weights: `Models/MedSAM/`
- Run logs, checkpoints, and predictions: `outputs/medsam/<run_name>/`

Keep training adaptations in this package so they are versioned independently
of the ignored upstream checkout. Avoid importing the nnU-Net model package;
its environment uses a different PyTorch stack.

The SPIDER masks use background 0, vertebrae 1, discs 2, and canal 3. A MedSAM
data adapter must explicitly define slice selection, per-class targets and
prompts, and image/mask transforms. Preserve patient-level train/validation
separation when generating slices.

This scaffold does not implement those adapters or a training loop. Checkpoint
selection, freezing policy, losses, and evaluation remain study decisions.
