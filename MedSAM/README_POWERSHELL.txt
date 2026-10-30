PART 6 LOCKED STUDY — IMPORTANT

Copy train_engine_locked.py, run_locked.py, launch_locked.py into the MedSAM
repository root alongside the unchanged Parts 1–4 modules.

Requires existing files:
 medsam_setup.py, patient_manifests.py, instance_dataset.py,
 mri_preprocessing.py, box_prompts.py, decoder_adaptation.py,
 segmentation_loss.py, TrainData/, checkpoints/medsam_vit_b.pth,
 label_schema.json, manifests/patient_train_val.json

Preflight (PowerShell):
 python .\test_locked.py
 python .\launch_locked.py
 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
 python -m pip show wandb

Pilot K5 seed1 (online W&B, after wandb login):
 python .\launch_locked.py --pilot-only --execute --wandb-mode online --device cuda:0

The launch command is equivalent to:
 python .\run_locked.py --k 5 --seed 1 --epochs 100 --patience 15 --min-delta 0.001 --val-frequency 1 --train-limit 0 --val-patients-limit 0 --val-examples-per-patient 0 --device cuda:0 --wandb-mode online --experiment K5_seed1 --output-root .\runs

If interrupted, resume ONLY this experiment using:
 python .\run_locked.py --k 5 --seed 1 --epochs 100 --patience 15 --min-delta 0.001 --val-frequency 1 --train-limit 0 --val-patients-limit 0 --val-examples-per-patient 0 --device cuda:0 --wandb-mode online --experiment K5_seed1 --output-root .\runs --resume .\runs\K5_seed1\last.pth

Only after evaluating pilot and explicitly approving the matrix:
 python .\launch_locked.py --execute --pilot-approved --skip-verified-pilot --wandb-mode online --device cuda:0

CAUTION: --skip-verified-pilot requires the pilot's best/last checkpoints,
history and config files. Review pilot results manually before approval.
The launcher refuses all other pre-existing run directories.

Local outputs per run:
  runs/K5_seed1/configuration.json, provenance.json, history.csv,
  history.json, best.pth, last.pth, curves.png

Policy: seeds 1,2,3; K 5,10,20; 9 runs; frozen encoders, decoder only;
100 maximum epochs, full validation every epoch, patience=15,
min_delta=0.001; best checkpoint selected by highest patient mean Dice.
Initial optimization defaults preserved from Part 5: AdamW, 1e-5 LR,
batch size 1, weight decay 0, AMP off; these were not tuned.

No independent held-out test patient files are accessed.
Original individual patient IDs remain in LOCAL provenance/checkpoint JSON,
not in the W&B init config. Before enabling checkpoint artifact upload,
check whether including local provenance with patient IDs is permitted.

LIMITATIONS: Full real SPIDER experiments and W&B online integration cannot
be executed in this environment without the full data/checkpoint and account.
Source engine carries no embedding cache; a full 42-patient validation each
epoch may remain computationally expensive.
