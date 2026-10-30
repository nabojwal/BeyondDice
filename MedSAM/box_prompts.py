"""Explicit instance-preserving box prompts; xyxy continuous pixel-edge coordinates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from mri_preprocessing import SliceGeometry

@dataclass(frozen=True)
class PromptBox:
    xyxy: tuple[float, float, float, float]
    label_id: int | None
    source: str  # 'reference-oracle' or 'external'


def validate_box(box, hw: tuple[int, int], *, clip: bool = True) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Box must contain 4 finite xyxy values")
    h, w = hw
    if clip:
        values[[0, 2]] = np.clip(values[[0, 2]], 0, w)
        values[[1, 3]] = np.clip(values[[1, 3]], 0, h)
    elif min(values[0], values[1]) < 0 or values[2] > w or values[3] > h:
        raise ValueError("Box outside native image")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Box is empty or degenerate after clipping")
    return tuple(map(float, values))


def reference_box(mask_hw: np.ndarray, label_id: int, *, padding: float = 0,
                  jitter_xyxy: tuple[float, float, float, float] = (0, 0, 0, 0)) -> PromptBox | None:
    """Oracle ONLY: derive one instance box from an explicit integer label.

    No semantic class merging; absent label -> None. Jitter in native pixels.
    A label must represent a unique anatomical instance in the supplied mask.
    """
    m = np.asarray(mask_hw)
    if m.ndim != 2 or not np.issubdtype(m.dtype, np.integer):
        raise ValueError("Expected 2D integer label mask")
    if not isinstance(label_id, (int, np.integer)) or label_id <= 0:
        raise ValueError("Use a strictly positive explicit instance label")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("Padding must be finite and nonnegative")
    j = np.asarray(jitter_xyxy, dtype=float)
    if j.shape != (4,) or not np.isfinite(j).all():
        raise ValueError("jitter_xyxy must have four finite coordinates")
    yy, xx = np.nonzero(m == label_id)
    if not len(xx):
        return None
    # Half-open intervals include last occupied pixel and extend to image boundary.
    b = np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=float)
    b += np.array([-padding, -padding, padding, padding]) + j
    return PromptBox(validate_box(b, m.shape), int(label_id), "reference-oracle")


def external_box(box, hw: tuple[int, int]) -> PromptBox:
    """Boxes generated independently of ground-truth reference masks."""
    return PromptBox(validate_box(box, hw), None, "external")


def to_model_box(prompt: PromptBox, geometry: SliceGeometry, device="cpu") -> torch.Tensor:
    """Return float32 [1,1,4] MedSAM prompt input; pixel-edge xyxy."""
    if prompt.source not in {"external", "reference-oracle"}:
        raise ValueError("Unrecognized prompt provenance")
    x0, y0, x1, y1 = validate_box(prompt.xyxy, geometry.native_hw)
    mx0, my0 = geometry.point_to_model(x0, y0)
    mx1, my1 = geometry.point_to_model(x1, y1)
    result = torch.tensor([[[mx0, my0, mx1, my1]]], device=device, dtype=torch.float32)
    return result


def box_to_native(model_xyxy, geometry: SliceGeometry):
    b = np.asarray(model_xyxy, dtype=float).reshape(4)
    x0, y0 = geometry.point_to_native(float(b[0]), float(b[1]))
    x1, y1 = geometry.point_to_native(float(b[2]), float(b[3]))
    return validate_box((x0, y0, x1, y1), geometry.native_hw)
