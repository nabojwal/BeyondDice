"""Split SPIDER images and semantic masks using the official CSV split.

Output:
    split_dataset/Image/Train/
    split_dataset/Image/Val/
    split_dataset/Mask/Train/
    split_dataset/Mask/Val/

Original files are preserved. Copying is not transactional:
an interruption during copying may leave partial outputs.
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import shutil


BASE = Path(r"D:\Courses\MD5001\Mini Project\SPIDERDataset")

CSV_PATH = BASE / "overview.csv"
IMAGE_DIR = BASE / "images"
MASK_DIR = BASE / "semantic_test_masks"
OUTPUT_DIR = BASE / "split_dataset"

SPLITS = {
    "training": "Train",
    "validation": "Val",
}


def find_one(folder, names):
    """Find exactly one file, ignoring duplicate candidate names."""
    names = list(dict.fromkeys(names))
    matches = [
        folder / name
        for name in names
        if (folder / name).is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            f"No matching file in {folder}.\n"
            f"Expected one of: {names}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matching files in {folder}:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def prepare_jobs(rows):
    """Validate all records and destinations before copying."""
    if not rows:
        raise ValueError("The CSV contains no records.")

    jobs = []
    seen_cases = set()
    seen_destinations = set()
    patient_splits = defaultdict(set)
    counts = Counter()

    for row_number, row in enumerate(rows, start=2):
        case = (row.get("new_file_name") or "").strip()
        subset = (row.get("subset") or "").strip().lower()

        if not case:
            raise ValueError(f"CSV row {row_number}: missing case identifier.")

        if "/" in case or "\\" in case or case in {".", ".."}:
            raise ValueError(f"Invalid case identifier: {case!r}")

        if subset not in SPLITS:
            raise ValueError(f"{case}: unknown subset {subset!r}")

        if case in seen_cases:
            raise ValueError(f"Duplicate case in CSV: {case}")
        seen_cases.add(case)

        split = SPLITS[subset]
        patient_id = case.split("_", 1)[0]
        patient_splits[patient_id].add(split)

        image = find_one(
            IMAGE_DIR,
            [
                f"{case}_Img.mha",
                f"{case}.mha",
                f"{case}_0000.mha",
            ],
        )
        mask = find_one(MASK_DIR, [f"{case}.mha"])

        for category, source in (("Image", image), ("Mask", mask)):
            destination = OUTPUT_DIR / category / split / source.name
            destination_key = destination.resolve()

            if destination_key in seen_destinations:
                raise ValueError(
                    f"Duplicate destination: {destination}"
                )
            seen_destinations.add(destination_key)

            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite: {destination}\n"
                    "Choose a new output directory."
                )

            jobs.append((source, destination))

        counts[split] += 1

    overlap = sorted(
        patient
        for patient, splits in patient_splits.items()
        if len(splits) > 1
    )
    if overlap:
        raise ValueError(
            f"Patients present in both Train and Val: {overlap}"
        )

    return jobs, counts


def main():
    """Validate dataset inputs and copy the official training/validation split."""
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    for folder in (IMAGE_DIR, MASK_DIR):
        if not folder.is_dir():
            raise NotADirectoryError(f"Source folder not found: {folder}")

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        required_columns = {"new_file_name", "subset"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing_columns)}"
            )

        rows = list(reader)

    jobs, counts = prepare_jobs(rows)

    # Create all four folders after validation succeeds.
    for category in ("Image", "Mask"):
        for split in SPLITS.values():
            (OUTPUT_DIR / category / split).mkdir(
                parents=True, exist_ok=True
            )

    for source, destination in jobs:
        shutil.copy2(source, destination)

    print(f"Train: {counts['Train']} image-mask pairs")
    print(f"Val:   {counts['Val']} image-mask pairs")
    print(f"Total files copied: {len(jobs)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
