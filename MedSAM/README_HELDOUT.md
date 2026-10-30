# MedSAM Round 2 — Independent 87-scan TestData evaluation

This is a **new held-out testing workflow**, separate from training/validation. It is configured for `TestData/images/*.mha` and `TestData/masks/*.mha` and the **360-series / 179-patient** Round 2 model training split.

## Deployment

Place `evaluate_heldout.py` in `~/MedSAM/` alongside the existing `patient_manifests.py`, `instance_dataset.py`, `mri_preprocessing.py`, `medsam_setup.py` and `box_prompts.py`. Use the Python environment that successfully ran training, with `scipy`, `scikit-image`, `SimpleITK`, `torch` and the official MedSAM code installed.

First run the preflight; it **does not run the model**, alter patient data or read/modify checkpoints:

```bash
cd ~/MedSAM
source .sam/bin/activate
python evaluate_heldout.py --audit-only --test-dir ./TestData --expected 87 \
 --split ./manifests_round2/patient_train_val.json \
 --schema ./label_schema.json --out ./heldout_evaluation
```

The audit fails closed on any duplicate/missing or wrong-number-of-scans, train/val vs test patient-ID overlap, geometric mismatch, unsupported orientations, unreadable/nonfinite MRI volume, unknown nonzero label, or all-background mask. IMPORTANT: this audit verifies **ID-derived patient independence**; if patient IDs were anonymized inconsistently across data sources, cross-dataset de-identification linkage must be independently checked.

Once all nine best checkpoints are saved, run on an authorized GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -u evaluate_heldout.py \
  --test-dir ./TestData --expected 87 \
  --split ./manifests_round2/patient_train_val.json \
  --schema ./label_schema.json \
  --pretrained ./checkpoints/medsam_vit_b.pth \
  --runs ./runs_train2 --out ./heldout_evaluation \
  --device cuda:0 --surface-tolerance-mm 1.0 \
  --save-predictions 2>&1 | tee heldout_evaluation.log
```

The **1.0 mm surface tolerance is a proposed default, not yet specified by the project proposal**. Fix it prospectively before accessing test performance, and report it alongside HD95/ASSD. Prefer a defined surface tolerance appropriate to clinical use.

For a functionality check without executing all nine checkpoint evaluations, use `--one-run K5_seed1` with a **separate output directory**. Do not select or optimize checkpoints on its test results. To skip storing output MHA files, omit `--save-predictions`.

## Outputs

`heldout_evaluation/test_audit.json` contains the strict patient overlap / 87 scan / geometry / modality / label audit. Per each completed experiment:

- `per_scan_per_label.csv`: one row per scan and GT-present integer label; Dice, IoU, symmetric 95th percentile surface distance (HD95 mm), symmetric average surface distance (ASSD mm), surface Dice with specified mm tolerance, GT/pred volume in mm³ and relative volume error.
- `per_scan_summary.csv`, `per_patient_summary.csv`, `per_modality_summary.csv`, `per_label_summary.csv`, `per_structure_summary.csv`, and `overall_summary.csv`: macro-mean and SD across **GT-present label observations**. These are **descriptive macros**, not pooled voxel metrics, and labels from the same patient are correlated.
- `scan_inventory.csv`: GT-present and predicted GT-label counts. This is **oracle-prompt conditional detection**, not automatic detection sensitivity; it cannot measure proposals for GT-absent targets.
- `predictions/<series_id>.mha`: predicted 3D instance-label segmentation with original reference spatial information when enabled.
- `COMPLETE.json`: metadata, best checkpoint fingerprint, and expected run count.

## Locked semantics

1. The official pretrained ViT-B MedSAM checkpoint initializes the encoders; **only `payload['decoder']` from the trained `best.pth` is restored** with a strict state-dict load. No test-set weight updates.
2. Segmentation is **GT-derived oracle bounding-box prompted**. The reference segmentation is used to define visible classes and prompt boxes. This cannot be presented as fully automated or unprompted detection or segmentation.
3. Preprocessing uses the original `extract_sagittal`, `normalize_medsam` and `resize_labels` routines. Per-label binary logits are thresholded at 0 and restored to their original sagittal planes; deterministic ascending-label precedence handles overlapping binary predictions. Report this overlap priority because it can affect 3D scores.
4. Surface metrics are based on 6-connected *boundary-voxel centers* and anisotropic Euclidean distance-transform sampling, **not continuous triangulated mesh distances**. Interpret HD95 and ASSD accordingly; this is one documented definition, and it should be fixed before test evaluation.
5. For a GT-present label with **empty prediction**, Dice and IoU = 0, Surface Dice = 0, HD95/ASSD = NaN (undefined). These NaN values are excluded from aggregate distance means, so **report the count of failed detections alongside distance summaries**. Both empty yields Dice/IoU/SurfaceDice 1 but is not present in this GT-present-only report.
6. Volumes are physical 3D voxel counts times voxel volume. These are valid volumetric biomarkers. **Disc height, disc height index, inter-rater ICC and Bland–Altman for anatomical height are not computed in this release**: defensible definitions require explicit measurement landmarks, level assignment, and a reference measurement protocol. Do not fabricate them from pixel bounding-box heights.

## Operational safeguards

- Does **not** modify `TrainData`, `TestData`, training checkpoints, or manifests.
- Refuses to overwrite a final `COMPLETE.json`.
- Stop and investigate any audit failure, CUDA OOM, or read error rather than silently excluding cases.
- Run this pipeline **only after training checkpoint selection is finalized**; do not compare all nine held-out scores to pick a winner without labeling such selection exploratory.
- Multiple scans can belong to one patient. Statistical confidence intervals and hypothesis tests should be clustered at patient level.
