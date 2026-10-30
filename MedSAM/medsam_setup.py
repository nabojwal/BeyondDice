"""Initialize the authentic MedSAM ViT-B architecture with pretrained weights.

Uses the official bowang-lab/MedSAM ``segment_anything`` package from this
workspace when available, without requiring a PYTHONPATH setting.
No dataset, optimizer, or training behavior is implemented here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
from torch import nn


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto', 'cpu', 'cuda', or 'cuda:N' without silent fallback."""
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        requested = "cuda:0"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device: {requested!r}") from exc
    if device.type != "cuda":
        raise ValueError("Supported --device choices: auto, cpu, cuda, cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    index = device.index if device.index is not None else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} requested, but only {torch.cuda.device_count()} available"
        )
    return device


def load_medsam(checkpoint_path: str | Path, device: str = "auto") -> nn.Module:
    """Build official MedSAM ViT-B and strictly load the entire checkpoint.

    The official ``build_sam.py`` calls ``load_state_dict(state_dict)`` with
    ``strict=True`` (the PyTorch default), detecting missing/unexpected keys
    and incompatible parameter shapes. No key rewriting or partial loading.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found: {path}\n"
            "Download the official medsam_vit_b.pth and supply --checkpoint."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"MedSAM checkpoint is empty: {path}")

    target_device = resolve_device(device)
    repository_dir = Path(__file__).resolve().parent
    for candidate in (repository_dir, repository_dir / "MedSAM"):
        if (candidate / "segment_anything" / "__init__.py").is_file():
            candidate_path = str(candidate)
            if candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)
            break

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "Cannot import the official MedSAM segment_anything package. "
            "Clone https://github.com/bowang-lab/MedSAM and run this script "
            "from its repository root, or add that root to PYTHONPATH. "
            "Do not substitute the pip 'segment-anything' package when "
            "reproducibility against the MedSAM repository is required."
        ) from exc

    try:
        # The official registry implementation performs full strict loading.
        model = sam_model_registry["vit_b"](checkpoint=str(path))
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        raise RuntimeError(
            f"Failed to construct/load official MedSAM ViT-B from {path}. "
            "Checkpoint keys and/or tensor shapes may not match. "
            "The original exception contains the exact incompatibilities."
        ) from exc

    model = model.to(device=target_device)
    model.eval()
    return model


def component_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count parameters without double-counting nested parameters."""
    names = ("image_encoder", "prompt_encoder", "mask_decoder")
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters())
              for name in names}
    counts["total"] = sum(p.numel() for p in model.parameters())
    if counts["total"] != sum(counts[name] for name in names):
        raise AssertionError("Model contains unaccounted or shared parameters")
    return counts
