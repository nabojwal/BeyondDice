# Dot-source from the BeyondDice checkout. Does not overwrite the existing setup script.
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $projectRoot 'Models\nnUNet\nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $projectRoot 'Models\nnUNet\nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $projectRoot 'Models\nnUNet\nnUNet_results'
$env:nnUNet_extTrainer = Join-Path $projectRoot 'training\nnunet\trainers'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH" }
else { $env:PYTHONPATH = $projectRoot }
$env:nnUNet_n_proc_DA = '2'
$env:nnUNet_wandb_enabled = '0'
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = 'offline' }
$env:WANDB_PROJECT = 'BeyondDice-SPIDER'
