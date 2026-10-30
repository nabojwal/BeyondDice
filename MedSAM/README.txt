Enhanced metrics for the uploaded Part 5 train_engine.py. Copy train_engine.py and run_logged.py to your project root. Existing modules Parts 1-4 remain unchanged.

PowerShell:
python -m pip install wandb
wandb login
python .\test_logging_metrics.py
python .\run_logged.py --images .\TrainData\images --masks .\TrainData\masks --label-schema .\label_schema.json --split-manifest .\manifests\patient_train_val.json --shot-dir .\manifests\fewshot_train_val --checkpoint .\checkpoints\medsam_vit_b.pth --seed 42 --k 1 --epochs 100 --batch-size 1 --workers 0 --lr 0.0001 --patience 15 --train-limit 0 --val-patients-limit 0 --val-examples-per-patient 0 --device cuda:0 --wandb-mode online --wandb-log-every 10 --experiment pilot_K1_seed42_metrics_v2 --output-root .\runs

Optional telemetry: python -m pip install nvidia-ml-py
GPU utilization is instantaneous NVML sampling (not average utilization); memory allocation is PyTorch memory. W&B system monitoring is enabled by default.
IMPORTANT: This engine is the modified uploaded train_engine.py, NOT the separate train_engine_part6.py. run_logged.py explicitly imports this module.
Do not replace code during an active run; wait until it ends or stop deliberately, then use a new experiment name. Existing part 6 checkpoints need engine schema reconciliation before resuming with new fields.
