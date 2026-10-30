# Part 7 — incremental implementation (not yet complete)

## What is verified here
- `part7_reconstruction.py`: native z/y/x instance reassembly, explicit overlap priority, MHA geometry writer.
- `part7_metrics.py`: 3D Dice, symmetric pooled surface ASD with spacing in mm, **ID-presence** detection accounting. Empty cases return `None` rather than artificial perfect Dice/ASD.
- `part7_audit.py`: read-only provenance/identity verification for all nine **trusted** decoder checkpoints against the copied split and pretrained image encoder hash.
- `part7_inference.py`: validation-cohort **oracle-box** inference for one chosen checkpoint/zero-shot arm; same box preprocessing as Part 3. It is not a blind/independent test evaluation.
- `test_part7_core.py`: geometry and metric tests.

## Installation
Copy only these five `part7_*.py` files and `test_part7_core.py` to the root of your existing `~/MedSAM` source tree. Do not replace Parts 1–6.

```bash
cd ~/MedSAM
source .sam/bin/activate
python -m unittest -v test_part7_core
python part7_audit.py
# one patient oracle-only validation smoke test; may take time
CUDA_VISIBLE_DEVICES=6 python part7_inference.py --model zero_shot --device cuda:0 --patient-limit 1
# after a successful zero-shot smoke test
CUDA_VISIBLE_DEVICES=6 python part7_inference.py --model few_shot --k 20 --seed 1 --device cuda:0 --patient-limit 1
```

**Do not launch all ten evaluation arms until the first patient geometry, masks, and qualitative overlays are visually verified.**

The inference command intentionally requires `--model`. There is no `--test` option; `validation_oracle` appears in output paths and results. Seed variation must not be collapsed into a single best seed selected on evaluation data.

## Unverified / intentionally incomplete
- Full 42-person inference and all ten-arm comparisons have not run here: this environment does not contain your live cluster checkpoints or 210 T2 volumes.
- A truly independent labeled T2 test cohort has not been established.
- Official SPIDER scoring code has been examined: its Dice and symmetric ASD are surface-derived, but its detection/matching and aggregation policy must be cross-checked numerically before leaderboard-equivalent claims. Here `detection_rate` means **exact reference label-ID presence**, not the official structure-matching rule.
- Oracle prompts are derived from reference labels; no independent box generator is available, so no automatic-detection claim is supported.
- W&B evaluation logging, pooled seed summaries, qualitative overlays, checkpoint load smoke test on H100, physical MHA roundtrip test, and a full SPIDER evaluator differential test remain pending.
- `torch.load(weights_only=False)` must only read your own trusted checkpoints because training serialized NumPy/Python RNG state.
- The basic overlap policy favors the largest probability and then the smallest label; further clinical review of instance continuity is required.

## Review focus
The Part 3 training Dataset uses a box that encloses both the original native GT bbox and the nearest-neighbor resized GT. `part7_inference.py` mirrors that exact union for oracle prompts. Note that the inference 1024→native interpolated probability differs from thresholding a 1024 binary label before resampling: this decision must be held constant across all models.
