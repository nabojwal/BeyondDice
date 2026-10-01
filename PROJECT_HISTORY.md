# Project History

**Research project:** Beyond Dice: Quantitative-Biomarker Validation of Foundation-Model Segmentation for Lumbar Spine MRI

This document records the project's stages, changes, decisions, and progress. Add an entry whenever the project reaches a milestone or changes direction. Keep older entries so the history remains traceable.

## Current State

- **Stage:** Full nnU-Net baseline stopped after completed epoch index 522
- **Last updated:** 2026-10-01
- **Status:** The user stopped baseline training after epoch 522. The log confirms zero-based epoch 522 completed and epoch 523 began; completion of epoch 523 is not recorded. Best/latest checkpoints remain local. The planned 1000-epoch run and full-volume baseline validation are incomplete/unconfirmed. The 87 official validation scans remain held out. MedSAM remains a scaffold.

## Stage History

### Stage 1 - Initial Documentation

- **Date:** 2026-09-29
- **Status:** Documentation initialized
- **Starting state:** The project folder was empty, with no source files or Git repository available to establish earlier history.
- **Changes:** Created `PROJECT_HISTORY.md` to record the project's development stages.
- **Decision:** Record verified changes and explicitly identify any unknown earlier history.
- **Validation:** Confirmed that the folder contained no existing project files before creating this document.
- **Open items:** Project objectives, requirements, technology choices, and implementation milestones are not yet recorded.
- **Next steps:** Document the project objectives and requirements, then record subsequent development stages below.

### Stage 2 - Research Project Named

- **Date:** 2026-09-29
- **Status:** Completed (title recorded)
- **Starting state:** Project history initialized without a research project title.
- **Changes:** Recorded the user-provided title: "Beyond Dice: Quantitative-Biomarker Validation of Foundation-Model Segmentation for Lumbar Spine MRI".
- **Decision:** Use this title to identify the research project in this history.
- **Validation:** Title matches the wording supplied by the user.
- **Open items:** Detailed research questions, datasets, foundation models, quantitative biomarkers, and validation methods remain to be specified.
- **Next steps:** Document the research objectives and study design.

### Stage 3 - Local Version Control

- **Date:** 2026-09-29
- **Status:** Completed
- **Changes:** Initialized the local Git repository on `main` and created the initial documentation commit `19f2428` (`docs: add initial project history`).
- **Validation:** Initial commit succeeded and the working directory was clean immediately afterward.

### Stage 4 - Research Environments

- **Date:** 2026-09-29
- **Status:** Completed (setup and smoke checks)
- **Changes:** Created separate Python 3.11 environments for nnU-Net, original MedSAM, and shared evaluation; saved dependency locks and registered notebook kernels.
- **Model stacks:** nnU-Net 2.8.1 / PyTorch 2.6 / CUDA 12.4; MedSAM / PyTorch 2.0.1 / CUDA 11.8. Isolated and pinned dependencies to address compatibility issues.
- **Validation:** All dependency checks passed. Verified GPU operations, a synthetic MedSAM forward pass, MRI handling, surface Dice, and intraclass correlation.
- **Details:** See [ENVIRONMENTS.md](ENVIRONMENTS.md) for versions, source revision, and usage.
- **Next steps:** Add datasets and pretrained weights, define the validation protocol, and implement biomarker measurements. Full training and model accuracy remain unvalidated.

### Stage 5 - SPIDER Semantic Mask Preprocessing

- **Date:** 2026-09-29
- **Status:** Completed (utility and synthetic checks)
- **Changes:** Added an importable converter and CLI in scripts/preprocess_spider_masks.py, explicit NumPy/SimpleITK dependencies, and SPIDER_PREPROCESSING.md usage documentation.
- **Behavior:** Merges vertebral and disc instances into four semantic classes, preserves sources, checks destination conflicts, rejects unknown labels, and verifies saved geometry and label storage.
- **Validation:** Six synthetic tests passed in the nnU-Net environment, covering all label values, saved geometry and compression, naming/order, source preservation, empty input, conflicts, invalid labels, and non-transactional failure behavior. CLI help verified.
- **Follow-up validation (2026-09-29):** Required scalar 3D masks and exact saved voxel equality. Confirmed the actual script has no Markdown copying artifacts or redundant empty-list assignment. All nine synthetic tests passed, including non-3D/vector rejection and saved voxel corruption detection; CLI help verified.
- **Limitations at this stage:** Real-data validation was pending (subsequently completed for three masks in Stage 6). Full nnU-Net dataset assembly/training remain unvalidated. Conversion may leave earlier outputs after a later failure.

### Stage 6 - Real SPIDER Preprocessing Validation

- **Date:** 2026-09-29
- **Status:** Completed (three-volume spot-check)
- **Validation:** Converted 1_t1, 100_t1, and 100_t2 into SPIDERDataset/semantic_test_masks. Independently verified exact mapping, class voxel counts, uint8 storage, source/MRI geometry, source preservation, and overwrite refusal with unchanged output hashes.
- **Visual check:** Inspected Slicer 5.12.4 composited sagittal, axial, and coronal overlays for all three cases; no visible global shift or flip.
- **Details:** See [SPIDER_VALIDATION.md](SPIDER_VALIDATION.md). Numerical reports and overlay images are retained locally under outputs/spider_validation.
- **Limitations:** Sampled slices only; full-dataset validation, annotation quality assessment, dataset assembly, and training remain pending.

### Stage 7 - nnU-Net Environment Activation Verified

- **Date:** 2026-09-29
- **Status:** Completed (activation check)
- **Action:** Activated the existing environment using `. .\.venvs\nnunet\Scripts\Activate.ps1` from the project root.
- **Validation:** Confirmed that `sys.executable` resolved to `D:\Courses\MD5001\Mini Project\BeyondDice\.venvs\nnunet\Scripts\python.exe`.
- **Scope:** Activation was verified in the command shell used for this check. Other terminals require their own activation; this does not establish persistent activation in the user's terminal.
- **Next steps:** Convert and validate the remaining SPIDER masks, assemble the complete nnU-Net dataset, and define the training and evaluation protocol.

### Stage 8 - nnU-Net Launcher Repair

- **Date:** 2026-09-29
- **Status:** Completed
- **Reported issue:** nnU-Net command launchers referenced the former Code project path instead of BeyondDice.
- **Action:** Ran `python -m pip install --force-reinstall --no-deps nnunetv2==2.8.1` using the BeyondDice nnU-Net interpreter to regenerate launchers without reinstalling dependencies.
- **Validation:** The regenerated `nnUNetv2_plan_and_preprocess.exe -h` exited successfully. `python -m pip check` reported "No broken requirements found."
- **Outcome:** Environment recreation was unnecessary. Existing environment-variable configuration was not repeated.

### Stage 9 - SPIDER Split Dataset Metadata

- **Date:** 2026-09-29
- **Changes:** Created `Data/nnUNet/nnUNet_raw/split_dataset/dataset.json` for 360 single-channel MRI training cases in `.mha` format; 87 validation cases remain held out.
- **Labels:** Background 0, vertebrae 1, intervertebral discs 2, spinal canal 3.
- **Validation:** All training image/mask paths resolved through nnU-Net's loader; no case or patient overlap between splits.
- **Next steps:** Adopt `DatasetNNN_Name` naming for CLI discovery, run full integrity checks and preprocessing, then configure training. The JSON does not configure the held-out cases as a training validation fold.

### Stage 10 - Model Training Source Layout

- **Date:** 2026-09-29
- **Changes:** Added `training/nnunet/` and `training/medsam/`, each with data adapters, trainers, and configuration locations; added `training/common/` for shared helpers. Created a root README and model-specific guidance.
- **Organization:** Existing scripts and datasets remain in place. Documented current `Data/nnUNet/` paths and ignored local data and model weights while keeping training source visible to Git.
- **Validation:** Scaffold imports passed in both isolated model environments; ignore rules and whitespace checked.
- **Scope:** Structure only; training entry points, adapters, experiment settings, and model training remain pending.

### Stage 11 - Dataset011 Image Naming Repair

- **Date:** 2026-09-29
- **Status:** Fixed; training-data integrity check passed.
- **Current location:** `Models/nnUNet/nnUNet_raw/Dataset011_SPIDER/`. This supersedes the earlier `Data/nnUNet/nnUNet_raw/split_dataset/` layout recorded in Stage 9.
- **Inconsistency:** The current `dataset.json` declares 360 cases and uses standard folder discovery rather than Stage 9's explicit path mapping. Images in `imagesTr/` lacked the required channel suffix. nnU-Net stripped five characters from each stem, yielding only 33 distinct, truncated case IDs.
- **Fix:** Renamed all 360 training images from `<case>.mha` to `<case>_0000.mha` after checking destination conflicts and matching labels. For example, `imagesTr/100_t1.mha` became `imagesTr/100_t1_0000.mha`; `labelsTr/100_t1.mha` remained unchanged. Voxel contents and geometry were not modified.
- **Validation:** nnU-Net detected all 360 cases, and its full training-data integrity check passed using two workers. Twelve image-mask pairs triggered direction-matrix warnings; the largest absolute matrix-entry difference was approximately `1.19e-7`. These warnings were measured, not corrected, and are not an exhaustive visual alignment assessment.
- **Artifacts:** Rename manifest: `outputs/nnunet11_filename_repair.json`; warning details: `outputs/nnunet11_direction_warnings.json`.
- **Next steps:** Run `nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity`. Only the standalone integrity check was run during this repair; planning/preprocessing and training remain pending.

### Stage 12 - nnU-Net Planning and Preprocessing Documentation

- **Date:** 2026-09-29
- **Command:** `nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity`
- **Why it is needed:** Checks dataset integrity, extracts dataset properties, automatically selects preprocessing and network settings, and prepares resampled/normalized training data for Dataset011_SPIDER.
- **Confirmed outcome:** The earlier standalone integrity check passed for 360 training cases, with 12 small direction-matrix warnings (Stage 11). The full command's terminal output has not been supplied, so completion of planning/preprocessing and its actual outputs remain unconfirmed. No preprocessing or training command was run for this documentation update.
- **Expected generated artifacts:** Successful execution writes to `Models/nnUNet/nnUNet_preprocessed/Dataset011_SPIDER/`: `dataset_fingerprint.json` (dataset statistics used for planning), `nnUNetPlans.json` (experiment settings), dataset metadata, and configuration-specific preprocessed images, targets, and case properties. The configurations produced depend on the planner's decisions.
- **About nnUNetPlans.json:** This is nnU-Net's automatically designed experiment blueprint for SPIDER. It specifies target voxel spacing, resampling and normalization methods, patch size, planned batch size, network architecture, and available configurations. Actual settings must be read from the generated file; the planned batch size still needs runtime validation on the training hardware.
- **Use in the future pipeline:** Training reads the plans to instantiate the network and load matching preprocessed data. Inference uses the trained model's plans and metadata for consistent preprocessing and restoration of predictions to the original image space. This command does not train a model or create trained checkpoints. Cross-validation splits are generally established during training unless supplied beforehand.
- **Pending record:** Confirm completion, generated configurations, target spacing, patch sizes, batch sizes, and any additional warnings from the terminal output or saved artifacts before marking preprocessing complete.

### Stage 13 - Internal Split and External SPIDER Trainers

- **Date:** 2026-09-30
- **Environment:** Verified existing nnunet interpreter, nnunetv2 2.8.1, torch 2.6.0+cu124 and CUDA on RTX 2000 Ada. Added wandb 0.30.0 under existing version constraints; no existing dependency versions changed and pip check passed.
- **Preprocessing confirmed:** Located Dataset011_SPIDER in Models/nnUNet. Existing plans and 2d, 3d_lowres and 3d_fullres data each cover all 360 official training cases. No preprocessing was repeated. The 3d_fullres plan uses batch size 2, patch [64, 192, 192], spacing approximately [0.6373272, 0.6373172, 0.625].
- **Split:** Independently reproduced the supplied seed-42 SHA256 patient holdout from local overview.csv: 289 scans/143 patients for internal training, 71 scans/36 patients for development validation. All 87 official validation scans/39 patients are excluded. Numeric patient prefixes and all three sequence suffixes were verified, with no duplicate cases, missing pairs or patient overlap. Installed the exact split and provenance report under the preprocessed dataset; identical reruns are accepted. No prior split or dependent training checkpoints existed.
- **Integration:** Canonical code is in training/nnunet/data/spider_split.py, trainers/nnUNetTrainerSPIDER.py, configs/spider_baseline.json and verify_setup.py. scripts/setup_nnunet.ps1 configures external trainer discovery and Windows execution settings. Original supplied files and the retired metadata writer are preserved under outputs/integration_archive.
- **Corrections:** Replaced unavailable W&B generate_id API; added strict split/design checks, run identity and resume guards, JSON-safe metrics, privacy allowlists, disabled automatic console/code/metadata capture, and failure cleanup. Explicitly restrict checkpoint loading to filenames because installed 2.8.1 has a broken dictionary overload. Replaced globally monkeypatched draft tests with installed-API regression tests. Updated stale mask tests to the current .mha output format without changing converter behavior.
- **Baseline preserved:** Architecture, loss, optimizer, augmentations, deep supervision and 1000 epochs with 250/50 iterations remain inherited. The separate smoke class uses 5 epochs and 10/5 iterations. Epoch patch Dice/EMA determines best checkpoint selection; full-volume internal validation is separate and local.
- **Verification:** 28 tests passed, plus real-data split regeneration and idempotence, raw/preprocessed coverage, discovery and CPU construction of both trainers, syntax/import checks and dependency checks. Checkpoint and predictor restoration used synthetic CPU fixtures without training or prediction; W&B initialization/logging was mocked.
- **Limits and next step:** GPU memory, actual training/checkpoint recovery, W&B runtime/upload and full-volume inference remain unexecuted. Run the offline smoke command in [the canonical guide](training/nnunet/README.md); inspect its outputs before the baseline. Official validation must not be used for tuning or checkpoint selection.

### Stage 14 - Smoke Training and Full-Volume Validation Completed

- **Date:** 2026-09-30; validation completed at 01:36:21.752065 (run log timestamp).
- **Experiment:** Dataset011_SPIDER, 3d_fullres, fold 0, trainer nnUNetTrainerSPIDER_5epochs. Resolved settings confirm 5 epochs, 10 training iterations and 5 development-validation iterations per epoch, batch size 2.
- **Outcome:** The terminal output and saved validation/summary.json agree: mean foreground full-volume validation Dice = **0.06561332397992369** (approximately 0.0656). The summary contains **71 internal development volumes**, drawn from the approved 36-patient development partition.
- **Per-class full-volume Dice:** Vertebrae (1): 0.19680551894230527; intervertebral discs (2): 0.0000003336493548806311; spinal canal (3): 0.000034119348110884753.
- **Artifacts:** Saved under Models/nnUNet/nnUNet_results/Dataset011_SPIDER/nnUNetTrainerSPIDER_5epochs__nnUNetPlans__3d_fullres/fold_0/: checkpoint_best.pth, checkpoint_final.pth, progress.png, training log, resolved configuration, and validation/summary.json.
- **Interpretation:** This establishes successful execution of the short training-to-full-volume-validation pipeline. The low Dice is a smoke-run result, not a trained baseline performance claim. Full-volume Dice is distinct from the epoch-level patch Dice/EMA used for checkpoint selection.
- **Experimental boundary:** The 87 official validation scans remain excluded from training and model selection. Smoke outputs remain separate from baseline outputs.
- **Next steps:** Inspect smoke predictions, logs and metric curves before starting the standard 1000-epoch baseline. No training, inference or W&B upload was rerun for this history update.

### Stage 15 - Full Baseline Startup and Resolved Experiment Blueprint

- **Date/source:** 2026-09-30; user-supplied startup output includes timestamp 01:53:41.404340. This entry records that output, not a new training invocation.
- **Command:** `nnUNetv2_train 11 3d_fullres 0 -tr nnUNetTrainerSPIDER --val_best`.
- **Startup outcome:** nnU-Net found `nnUNetTrainerSPIDER` through `nnUNet_extTrainer` at `D:/Courses/MD5001/Mini Project/BeyondDice/training/nnunet/trainers` and selected `cuda:0`. The initial message that the trainer was not found in the built-in package is expected fallback behavior; external discovery succeeded.
- **Upstream plans advisory:** The CLI reports that `nnUNetPlans` uses the older default plans and points to the [upstream residual-encoder preset recommendations](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md). This is an informational recommendation, not a startup failure. The current baseline retains the existing `ExperimentPlanner` / `PlainConvUNet` design; no preset migration, architecture change or replanning was performed. Any future residual-encoder comparison should be a separately identified experiment.
- **Schedule and split:** The integrated baseline inherits 1000 epochs with 250 training and 50 development-validation iterations per epoch. Fold 0 uses 289 training scans/143 patients and 71 internal development scans/36 patients. The 87 official validation scans remain excluded from training and model selection. `--val_best` requests full-volume internal validation using the best checkpoint after training; best-checkpoint selection remains based on EMA foreground patch Dice.
- **Runtime messages:** `do_dummy_2d_data_aug: False`; pinned-memory loading enabled on device 0 for both loaders. The configured Windows workflow disables compilation; this does not change the planned architecture.

**Resolved 3d_fullres configuration**

| Setting | Recorded value |
| --- | --- |
| Data identifier | `nnUNetPlans_3d_fullres` |
| Preprocessor | `DefaultPreprocessor` |
| Batch size | `2` |
| Patch size | `[64, 192, 192]` |
| Median image size in voxels | `[135.0, 441.0, 448.0]` |
| Target spacing | `[0.6373272056900671, 0.6373172056900671, 0.625]` in plan axis order |
| Normalization | `ZScoreNormalization`; `use_mask_for_norm: [False]` |
| Batch Dice | `True` |
| Data resampling | `resample_data_or_seg_to_shape`; `is_seg=False`, `order=3` |
| Segmentation resampling | `resample_data_or_seg_to_shape`; `is_seg=True`, `order=1` |
| Probability resampling | `resample_data_or_seg_to_shape`; `is_seg=False`, `order=1` |
| Shared resampling options | `order_z=0`, `force_separate_z=None` |

**Network architecture**

- **Class:** `dynamic_network_architectures.architectures.unet.PlainConvUNet`, using `torch.nn.modules.conv.Conv3d` with convolution bias enabled.
- **Stages/features:** 6 stages; features `[32, 64, 128, 256, 320, 320]`.
- **Kernels/strides:** `[3, 3, 3]` kernels at all stages; strides `[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]`.
- **Convolutions:** Encoder counts `[2, 2, 2, 2, 2, 2]`; decoder counts `[2, 2, 2, 2, 2]`.
- **Normalization/activation:** `InstanceNorm3d(eps=1e-5, affine=True)` and `LeakyReLU(inplace=True)`; no dropout (`dropout_op=None`, `dropout_op_kwargs=None`). The plan imports `conv_op`, `norm_op`, `dropout_op` and `nonlin` by their recorded names.

**Global nnUNetPlans.json settings**

- **Dataset/plans:** `Dataset011_SPIDER` / `nnUNetPlans`; planner `ExperimentPlanner`; reader/writer `SimpleITKIO`; label manager `LabelManager`.
- **Original median geometry after transpose:** Spacing `[3.313382213432355, 0.6373172056900671, 0.625]`; shape `[24, 448, 448]`.
- **Axis transformations:** `transpose_forward=[2, 0, 1]`; `transpose_backward=[1, 2, 0]`. Shape and spacing vectors above refer to the logged plan axis order.
- **Foreground intensity statistics, channel 0:** Minimum -1000.0; maximum 3096.0; mean 783.4005126953125; median 639.0; standard deviation 886.1395874023438; 0.5th percentile -863.0; 99.5th percentile 3096.0. These are recorded dataset statistics, not a claim that the MRI channel uses CT normalization or intensity clipping.
- **Required citation recorded from startup:** Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211.
- **Status/next record:** Startup/configuration are confirmed by the supplied output; completed epochs, baseline convergence and final full-volume validation metrics are not established by this excerpt. Record those from subsequent logs and saved artifacts. No running process, dataset, plans or training settings were changed for this documentation update.

### Stage 16 - Baseline Training Stopped and Development Checkpoint Saved

- **Date:** 2026-10-01.
- **User report:** Training was stopped after 522 epochs. The latest baseline log records completed zero-based epoch index 522 (523 epochs if counted from index 0), followed by the start of epoch 523 at 12:57:14.818577; its completion is not recorded.
- **Experiment:** Dataset011_SPIDER, 3d_fullres, fold 0, nnUNetTrainerSPIDER; planned schedule remains 1000 epochs.
- **Last completed epoch metrics:** Training loss -0.8351; development patch loss -0.8402; foreground patch Dice: vertebrae 0.9388, intervertebral discs 0.9078, spinal canal 0.9271. New best EMA foreground patch Dice: 0.9225000143051147. These are patch metrics, not full-volume validation results.
- **Local evidence:** training_log_2026_10_1_00_01_07.txt, checkpoint_best.pth (modified 12:57:14), and checkpoint_latest.pth (modified 11:51:58), under Models/nnUNet/nnUNet_results/Dataset011_SPIDER/nnUNetTrainerSPIDER__nnUNetPlans__3d_fullres/fold_0/. No checkpoint_final.pth was present during inspection. Checkpoint contents were not loaded for this documentation update.
- **Resume consideration:** The periodic latest checkpoint is older than the best checkpoint. Standard --c prefers latest over best when no final checkpoint exists, so it may resume before the last logged epoch. Inspect the saved epoch before choosing a recovery checkpoint; no checkpoints were renamed or overwritten here.
- **Scope:** Training was not restarted. Full-volume baseline validation and final performance remain unconfirmed. The official validation set remains excluded from model selection. Source/documentation are committed to Git; datasets, checkpoints and run artifacts remain ignored and local.

## Future Entry Template

Copy this template into **Stage History** for each new stage, in chronological order. Update **Current State** to reflect the latest stage. Use the actual date of each change; do not infer dates for undocumented work.

### Stage N - Stage Name

- **Date:** YYYY-MM-DD
- **Status:** Planned / In progress / Completed / Blocked
- **Starting state:** What existed at the beginning of this stage.
- **Changes:** What was added, modified, or removed, including relevant file paths or commit references if available.
- **Decisions and reasons:** Important choices and why they were made.
- **Validation:** Checks performed and their results, or "Not yet validated."
- **Open items:** Known issues, limitations, or unresolved questions.
- **Next steps:** Work to carry into the next stage.
