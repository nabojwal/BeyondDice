# Default preview. -Execute permanently deletes only the displayed old checkpoint files.
# Stop any old Dataset011 training process before using -Execute.
[CmdletBinding()]
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
if (-not $env:nnUNet_results) { throw 'Dot-source setup_spider_instances.ps1 first.' }
$dataset = Join-Path $env:nnUNet_results 'Dataset011_SPIDER'
$names = @('nnUNetTrainerSPIDER__nnUNetPlans__3d_fullres',
           'nnUNetTrainerSPIDER_5epochs__nnUNetPlans__3d_fullres')
$checkpoints = @('checkpoint_best.pth', 'checkpoint_final.pth', 'checkpoint_latest.pth')
$targets = @()
foreach ($name in $names) {
    $fold = Join-Path (Join-Path $dataset $name) 'fold_0'
    foreach ($checkpoint in $checkpoints) {
        $candidate = Join-Path $fold $checkpoint
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $targets += $candidate }
    }
}
$targets | ForEach-Object { Write-Output $_ }
if ($Execute) {
    foreach ($target in $targets) { Remove-Item -LiteralPath $target -ErrorAction Stop }
    Write-Output "Deleted $($targets.Count) old checkpoint files."
} else { Write-Output "Preview: $($targets.Count) files. Add -Execute to delete them." }
