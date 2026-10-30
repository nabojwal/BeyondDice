"""Audit SPIDER .mha instance labels and optionally export verified MetaImage masks.

Original labels: 0 background; 1..25 vertebrae; 100 canal; 201..225 discs.
The semantic export is for nnU-Net/comparison, NOT MedSAM instance prompting.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

VERTEBRA = frozenset(range(1, 26))
DISC = frozenset(range(201, 226))
VALID = frozenset({0, 100}) | VERTEBRA | DISC


def mapping_label(label: int) -> str:
    if label == 0:
        return "background"
    if label in VERTEBRA:
        return "vertebrae_whole"
    if label == 100:
        return "spinal_canal"
    if label in DISC:
        return "intervertebral_disc"
    raise ValueError(f"Unknown SPIDER label {label}")


def label_counts(mask: np.ndarray) -> dict[int, int]:
    """Return exact voxel counts, rejecting invalid or non-integral annotations."""
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D scalar mask, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(f"Expected integer mask, got {mask.dtype}")
    ids, counts = np.unique(mask, return_counts=True)
    result = {int(k): int(v) for k, v in zip(ids, counts)}
    unknown = sorted(set(result) - VALID)
    if unknown:
        raise ValueError(f"Unknown mask labels: {unknown}; source mask not modified")
    return result


def to_semantic(mask: np.ndarray) -> np.ndarray:
    """Preserve original shape; return 0 background, 1 vertebrae, 2 discs, 3 canal."""
    label_counts(mask)
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[(mask >= 1) & (mask <= 25)] = 1
    out[(mask >= 201) & (mask <= 225)] = 2
    out[mask == 100] = 3
    return out


def geometry(image: Any) -> dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_lps_mm": list(image.GetOrigin()),
        "direction_lps": list(image.GetDirection()),
    }


def check_geometry(src: Any, dst: Any) -> None:
    a, b = geometry(src), geometry(dst)
    if a["size_xyz"] != b["size_xyz"]:
        raise AssertionError("Output size differs from source")
    for field in ("spacing_xyz_mm", "origin_lps_mm", "direction_lps"):
        if not np.allclose(a[field], b[field], rtol=1e-6, atol=1e-5):
            raise AssertionError(f"Output {field} differs from source")


def process(input_dir: Path, report_path: Path, output_dir: Path | None,
            export: str, images_dir: Path | None) -> None:
    import SimpleITK as sitk

    paths = sorted(input_dir.glob("*.mha"))
    if not paths:
        raise FileNotFoundError(f"No .mha masks found in {input_dir}")
    if export != "none" and output_dir is None:
        raise ValueError("--output-dir required when --export is not none")
    if output_dir is not None and input_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory cannot equal source mask directory")
    targets = ([] if export == "none" else [output_dir / p.name for p in paths])
    if any(t.exists() for t in targets):
        raise FileExistsError("One or more output masks already exist; no files overwritten")
    if output_dir is not None and export != "none":
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    rows: list[dict[str, Any]] = []
    summary: Counter[int] = Counter()
    for path in paths:
        src = sitk.ReadImage(str(path))
        if src.GetDimension() != 3 or src.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{path}: expected a scalar 3D MetaImage mask")
        original = sitk.GetArrayFromImage(src)
        counts = label_counts(original)
        summary.update(counts)
        if images_dir is not None:
            img_path = images_dir / path.name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing paired image: {img_path}")
            check_geometry(src, sitk.ReadImage(str(img_path)))

        for lid, voxels in counts.items():
            rows.append({"mask_file": path.name, "label_id": lid,
                         "structure_type": mapping_label(lid), "voxel_count": voxels,
                         "volume_mm3": float(voxels * np.prod(src.GetSpacing()))})

        print(f"{path.name}: {counts} | size={src.GetSize()} | spacing={src.GetSpacing()}")
        if export != "none":
            payload = to_semantic(original) if export == "semantic" else original
            result = sitk.GetImageFromArray(payload, isVector=False)
            result.CopyInformation(src)
            target = output_dir / path.name
            sitk.WriteImage(result, str(target), useCompression=True)
            saved = sitk.ReadImage(str(target))
            check_geometry(src, saved)
            if not np.array_equal(sitk.GetArrayFromImage(saved), payload):
                raise AssertionError(f"Voxel mismatch after writing {target}")
            print(f"  VERIFIED: {target}")

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mask_file", "label_id", "structure_type",
                                               "voxel_count", "volume_mm3"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path = report_path.with_suffix(".summary.json")
    if summary_path.exists():
        raise FileExistsError(f"Summary report already exists: {summary_path}")
    summary_path.write_text(json.dumps({"files": len(paths), "export": export,
                                        "aggregate_voxel_counts": dict(sorted(summary.items()))},
                                       indent=2), encoding="utf-8")
    print(f"Reports: {report_path}, {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--export", choices=("none", "instance", "semantic"), default="none")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    process(args.input_dir, args.report, args.output_dir, args.export, args.images_dir)


if __name__ == "__main__":
    main()
