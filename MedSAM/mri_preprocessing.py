"""Geometry-preserving sagittal slice extraction for original SPIDER .mha files.

Coordinates: SimpleITK physical LPS; display rows increase inferiorly (-S),
columns increase posteriorly (+P). Only axis-aligned or near-axis-aligned
volumes accepted; no hidden oblique resampling. NumPy volume arrays are z,y,x.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class SliceGeometry:
    volume_shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction_xyz: tuple[float, ...]
    sagittal_axis_zyx: int
    row_axis_zyx: int
    column_axis_zyx: int
    flip_rows: bool
    flip_columns: bool
    slice_index: int
    native_hw: tuple[int, int]
    model_hw: tuple[int, int] = (1024, 1024)

    def voxel_zyx(self, row: int, column: int) -> tuple[int, int, int]:
        """Map displayed pixel indices back to original volume indices."""
        h, w = self.native_hw
        if not (0 <= row < h and 0 <= column < w):
            raise ValueError("Pixel index outside native slice")
        indices = [0, 0, 0]
        indices[self.sagittal_axis_zyx] = self.slice_index
        indices[self.row_axis_zyx] = h - 1 - row if self.flip_rows else row
        indices[self.column_axis_zyx] = w - 1 - column if self.flip_columns else column
        return tuple(indices)

    def physical_lps(self, row: int, column: int) -> np.ndarray:
        z, y, x = self.voxel_zyx(row, column)
        return np.asarray(self.origin_lps) + np.asarray(self.direction_xyz).reshape(3, 3) @ (
            np.asarray([x, y, z]) * np.asarray(self.spacing_xyz)
        )

    def point_to_model(self, x: float, y: float) -> tuple[float, float]:
        h, w = self.native_hw
        mh, mw = self.model_hw
        return x * mw / w, y * mh / h

    def point_to_native(self, x: float, y: float) -> tuple[float, float]:
        h, w = self.native_hw
        mh, mw = self.model_hw
        return x * w / mw, y * h / mh


def _axes(direction_xyz: tuple[float, ...], tolerance: float = 0.95):
    d = np.asarray(direction_xyz, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(d).all() or not np.allclose(d.T @ d, np.eye(3), atol=1e-4):
        raise ValueError("Direction matrix is invalid or nonorthogonal")
    # Columns of d are ITK x,y,z; physical rows are L,P,S.
    best = np.argmax(np.abs(d), axis=1)
    if len(set(best.tolist())) != 3 or np.any(np.abs(d[np.arange(3), best]) < tolerance):
        raise ValueError("Oblique geometry: explicit resampling required (not performed)")
    # physical X defines sagittal axis; physical Z defines vertical display axis;
    # physical Y defines horizontal display axis.
    itk_to_numpy = lambda axis: 2 - int(axis)
    sagittal = itk_to_numpy(best[0]); row = itk_to_numpy(best[2]); column = itk_to_numpy(best[1])
    # display +row points toward physical inferior (-S); +col toward posterior (+P).
    flip_row = d[2, best[2]] > 0
    flip_col = d[1, best[1]] < 0
    return sagittal, row, column, bool(flip_row), bool(flip_col)


def extract_sagittal(volume_zyx: np.ndarray, *, spacing_xyz: tuple[float, float, float],
                     origin_lps: tuple[float, float, float], direction_xyz: tuple[float, ...],
                     slice_index: int, model_size: int = 1024) -> tuple[np.ndarray, SliceGeometry]:
    """Return displayed 2D slice and invertible index geometry; no interpolation."""
    a = np.asarray(volume_zyx)
    if a.ndim != 3 or any(n <= 0 for n in a.shape):
        raise ValueError("Expected nonempty 3D volume in z,y,x order")
    if len(spacing_xyz) != 3 or any(v <= 0 or not np.isfinite(v) for v in spacing_xyz):
        raise ValueError("Invalid voxel spacing")
    if len(origin_lps) != 3 or not np.isfinite(origin_lps).all():
        raise ValueError("Invalid physical origin")
    if model_size <= 0:
        raise ValueError("Invalid model size")
    sag, row_axis, col_axis, flip_r, flip_c = _axes(direction_xyz)
    if not 0 <= slice_index < a.shape[sag]:
        raise IndexError(f"Sagittal slice {slice_index} outside axis {sag}, size {a.shape[sag]}")
    # Take slice; reorder surviving axes to (display row, display column).
    remaining = [i for i in range(3) if i != sag]
    sl = np.take(a, slice_index, axis=sag)
    sl = np.transpose(sl, (remaining.index(row_axis), remaining.index(col_axis)))
    if flip_r: sl = np.flip(sl, axis=0)
    if flip_c: sl = np.flip(sl, axis=1)
    sl = np.ascontiguousarray(sl)
    geo = SliceGeometry(tuple(a.shape), tuple(spacing_xyz), tuple(origin_lps), tuple(direction_xyz),
                        sag, row_axis, col_axis, flip_r, flip_c, slice_index, sl.shape,
                        (model_size, model_size))
    return sl, geo


def read_mha(path: str | Path):
    """Read original SPIDER .mha volume and spatial information (SimpleITK)."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Install SimpleITK: python -m pip install SimpleITK") from exc
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected 3D image, got {image.GetDimension()}D")
    arr = sitk.GetArrayFromImage(image)
    return arr, tuple(image.GetSpacing()), tuple(image.GetOrigin()), tuple(image.GetDirection())


def load_sagittal_mha(path: str | Path, slice_index: int, model_size: int = 1024):
    array, spacing, origin, direction = read_mha(path)
    return extract_sagittal(array, spacing_xyz=spacing, origin_lps=origin,
                            direction_xyz=direction, slice_index=slice_index, model_size=model_size)


def normalize_medsam(slice_hw: np.ndarray, model_size: int = 1024) -> torch.Tensor:
    """MedSAM inference-style cubic resize, uint8 quantization, min-max [0,1].

    The upstream demo uses skimage resize order=3, preserve_range=True,
    anti_aliasing=True -> uint8 -> min-max normalization. This method follows
    that exact ordering; quantization is intentional and documented.
    """
    from skimage.transform import resize
    a = np.asarray(slice_hw, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite 2D MRI slice")
    # Preserve upstream semantics (quantization after resize), without uint8 overflow
    resized = resize(a, (model_size, model_size), order=3, preserve_range=True,
                     anti_aliasing=True, mode="reflect")
    # MRI floats may be negative or exceed 255; upstream preparation first scales
    # MR intensities into [0,255], avoiding modulo wrap on uint8 conversion.
    # For consistent nonnegative inputs already in uint8 range, this is identity.
    lo, hi = float(np.min(resized)), float(np.max(resized))
    if hi <= lo:
        output = np.zeros_like(resized, dtype=np.float32)
    else:
        eight_bit = np.clip(np.rint((resized - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
        out = eight_bit.astype(np.float32)
        den = max(float(out.max() - out.min()), 1e-8)
        output = (out - out.min()) / den
    return torch.from_numpy(np.repeat(output[None, :, :], 3, axis=0).copy()).to(torch.float32)


def resize_labels(mask_hw: np.ndarray, model_size: int = 1024) -> np.ndarray:
    """Nearest-neighbor resample of integer labels; never interpolate categories."""
    a = np.asarray(mask_hw)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Expected 2D integer-valued label mask")
    from skimage.transform import resize
    return resize(a, (model_size, model_size), order=0, preserve_range=True,
                  anti_aliasing=False).astype(a.dtype)


def inverse_probabilities(prediction: torch.Tensor, geometry: SliceGeometry) -> torch.Tensor:
    """Map probability array (...,1024,1024) back to native displayed H,W."""
    if tuple(prediction.shape[-2:]) != geometry.model_hw:
        raise ValueError("Prediction does not match model geometry")
    leading = prediction.shape[:-2]
    x = prediction.reshape(-1, 1, *geometry.model_hw).float()
    y = F.interpolate(x, size=geometry.native_hw, mode="bilinear", align_corners=False)
    return y.reshape(*leading, *geometry.native_hw)


def assert_same_geometry(image_path: str | Path, mask_path: str | Path, atol: float = 1e-5):
    """Fail closed on even subtly mismatched image/mask physical grids."""
    image, sp, org, direction = read_mha(image_path)
    mask, msp, morg, mdirection = read_mha(mask_path)
    if image.shape != mask.shape or not all(np.allclose(a, b, atol=atol, rtol=0)
        for a, b in ((sp, msp), (org, morg), (direction, mdirection))):
        raise ValueError("Image/mask grids differ; registration/resampling required")
    return image, mask, sp, org, direction
