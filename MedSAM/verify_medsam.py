"""Strict checkpoint-load and synthetic forward sanity test for official MedSAM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from medsam_setup import component_parameter_counts, load_medsam


def verify(checkpoint: Path, device: str) -> None:
    model = load_medsam(checkpoint_path=checkpoint, device=device)
    parameter = next(model.parameters())
    counts = component_parameter_counts(model)

    print(f"PyTorch version:      {torch.__version__}")
    print(f"Checkpoint:           {checkpoint.expanduser().resolve()}")
    print("Architecture:         official MedSAM ViT-B")
    print("Checkpoint loading:   PASS (strict=True, official builder)")
    print(f"Device:               {parameter.device}")
    print(f"Parameter dtype:      {parameter.dtype}")
    for name in ("image_encoder", "prompt_encoder", "mask_decoder", "total"):
        print(f"Parameters/{name:<15} {counts[name]:>12,}")
    print(f"Model class:          {type(model).__name__}")
    print(f"Image encoder class:  {type(model.image_encoder).__name__}")
    print(f"Prompt encoder class: {type(model.prompt_encoder).__name__}")
    print(f"Mask decoder class:   {type(model.mask_decoder).__name__}")

    # Match official MedSAM_Inference.py: normalized 3-channel image, no
    # application of SAM.pixel_mean/std, bounding boxes on the 1024 grid.
    image = torch.rand((1, 3, 1024, 1024), device=parameter.device,
                       dtype=parameter.dtype)
    boxes = torch.tensor([[[256.0, 256.0, 768.0, 768.0]]],
                         device=parameter.device, dtype=parameter.dtype)
    assert boxes.shape == (1, 1, 4)
    assert bool((boxes[..., 2:] > boxes[..., :2]).all())
    assert bool((boxes >= 0).all() and (boxes <= 1024).all())

    with torch.inference_mode():
        image_embeddings = model.image_encoder(image)
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_logits, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        probabilities = torch.sigmoid(low_res_logits)
        full_resolution = F.interpolate(probabilities, size=(1024, 1024),
                                        mode="bilinear", align_corners=False)

    expected = {
        "image_embeddings": ((1, 256, 64, 64), image_embeddings),
        "sparse_embeddings": ((1, 2, 256), sparse),
        "dense_embeddings": ((1, 256, 64, 64), dense),
        "low_res_logits": ((1, 1, 256, 256), low_res_logits),
        "iou_predictions": ((1, 1), iou_predictions),
        "probabilities": ((1, 1, 256, 256), probabilities),
        "full_resolution": ((1, 1, 1024, 1024), full_resolution),
    }
    for name, (shape, tensor) in expected.items():
        actual = tuple(tensor.shape)
        if actual != shape:
            raise AssertionError(f"{name} shape: expected {shape}, got {actual}")
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f"{name} contains NaN or Inf")
        print(f"{name:<20} shape={str(actual):<24} finite=PASS")

    if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
        raise AssertionError("Sigmoid probabilities are outside [0, 1]")
    print("Synthetic inference: PASS")
    print("Overall verification: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local official medsam_vit_b.pth file")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, cuda, or cuda:N (default: auto)")
    args = parser.parse_args()
    try:
        verify(args.checkpoint, args.device)
    except Exception as exc:
        print(f"Verification FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
