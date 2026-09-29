# Dot-source from any directory. No training, data movement, or W&B login occurs.
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $projectRoot 'Models/nnUNet/nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $projectRoot 'Models/nnUNet/nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $projectRoot 'Models/nnUNet/nnUNet_results'
$env:nnUNet_extTrainer = Join-Path $projectRoot 'training/nnunet/trainers'
$env:nnUNet_compile = 'false'
$env:nnUNet_n_proc_DA = '2'
$env:nnUNet_wandb_enabled = '0'
if (($env:PYTHONPATH -split ';') -notcontains $projectRoot) {
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$projectRoot;$env:PYTHONPATH" } else { $projectRoot }
}
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = 'offline' }
if (-not $env:WANDB_PROJECT) { $env:WANDB_PROJECT = 'BeyondDice-SPIDER' }
