PART 3 UPDATE: Training/validation only
========================================
Copy patient_manifests.py and verify_part3.py into your MedSAM project root.
Keep existing Part 1/Part 2 modules and instance_dataset.py unchanged.
Do NOT include held-out test data in --images or --masks.

The code creates immutable manifest assignments. It does NOT relocate image files.
Default split: 80% training / 20% validation, by unique patient ID, seed 42.
If manifests/patient_split.json already exists as a three-way split, the program
will refuse to overwrite it. Use a new manifest path and manually check
any existing training/validation assignments before making a new split.

Windows PowerShell (from MedSAM project root):
python .\verify_part3.py `
  --images '.\TrainData\images' `
  --masks '.\TrainData\masks' `
  --patient-regex '(?P<patient>[^_]+)_t2' `
  --label-schema '.\label_schema.json' `
  --split-manifest '.\manifests\patient_train_val.json' `
  --shot-dir '.\manifests\fewshot_train_val' `
  --split-seed 42 `
  --seeds 42 `
  --ks 1 5 10 20 `
  --batch-size 2 `
  --workers 0

Inspect the created JSON to see actual patient IDs/counts. Requires at least
20 eligible training patients to generate K=20. No held-out test count or
patient IDs are fabricated; held-out test remains outside the repo.
