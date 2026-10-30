"""Decoder-only MedSAM adaptation, respecting official three-module architecture."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ForwardOutput:
    logits: torch.Tensor
    low_res_logits: torch.Tensor
    iou_predictions: torch.Tensor


def configure_decoder_only(model: nn.Module) -> nn.Module:
    """Freeze both encoders and train only mask decoder; do not alter weights."""
    for name in ('image_encoder', 'prompt_encoder'):
        module = getattr(model, name)
        module.requires_grad_(False)
        module.eval()
    model.mask_decoder.requires_grad_(True)
    model.mask_decoder.train()
    return model


def set_decoder_train_mode(model: nn.Module) -> None:
    """Call after any upstream model.train() to retain frozen encoder eval mode."""
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.train()


def parameter_report(model: nn.Module) -> dict[str, dict[str, float | int]]:
    report = {}
    for name in ('image_encoder','prompt_encoder','mask_decoder'):
        params = tuple(getattr(model, name).parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        report[name] = {'total':total,'trainable':trainable,'frozen':total-trainable,
                        'trainable_pct':100 * trainable/total if total else 0.0}
    total=sum(d['total'] for d in report.values())
    trainable=sum(d['trainable'] for d in report.values())
    report['total']={'total':total,'trainable':trainable,'frozen':total-trainable,
                     'trainable_pct':100*trainable/total if total else 0.0}
    return report


def decoder_optimizer(model: nn.Module, *, lr: float = 1e-5, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    if lr <= 0 or weight_decay < 0:
        raise ValueError('lr must be positive and weight_decay nonnegative')
    decoder=list(model.mask_decoder.parameters())
    if not decoder or not all(p.requires_grad for p in decoder):
        raise ValueError('Mask decoder must have all parameters trainable')
    if any(p.requires_grad for m in (model.image_encoder,model.prompt_encoder) for p in m.parameters()):
        raise ValueError('Encoders have trainable parameters; configure_decoder_only first')
    return torch.optim.AdamW(decoder,lr=lr,weight_decay=weight_decay)


def forward_decoder_only(model: nn.Module, batch: dict) -> ForwardOutput:
    """Box-prompted forward; full-resolution float logits match target mask shape."""
    image, box, mask = (batch[key] for key in ('image','box','mask'))
    if image.ndim!=4 or image.shape[1:]!=(3,1024,1024):
        raise ValueError(f'Expected image [B,3,1024,1024], got {tuple(image.shape)}')
    b=image.shape[0]
    if box.shape!=(b,4) or mask.ndim!=4 or mask.shape[:2]!=(b,1) or mask.shape[-2:]!=(1024,1024):
        raise ValueError(f'Invalid box/mask shapes: {tuple(box.shape)}, {tuple(mask.shape)}')
    if not image.is_floating_point() or image.dtype!=torch.float32 or box.dtype!=torch.float32:
        raise TypeError('Images and boxes must be float32')
    if not mask.is_floating_point():
        raise TypeError('Mask must be floating-point binary target')
    if not all(torch.isfinite(t).all().item() for t in (image,box,mask)):
        raise ValueError('Nonfinite input')
    if image.min()<0 or image.max()>1 or not torch.all((mask==0)|(mask==1)):
        raise ValueError('Input image must lie in [0,1] and mask must be binary')
    if not ((box[:,0]>=0)&(box[:,1]>=0)&(box[:,2]<=1024)&(box[:,3]<=1024)&
            (box[:,2]>box[:,0])&(box[:,3]>box[:,1])).all():
        raise ValueError('Invalid pixel-edge xyxy box')
    device=next(model.mask_decoder.parameters()).device
    if any(t.device!=device for t in (image,box,mask)):
        raise ValueError(f'Input tensors must be on {device}')
    if any(p.requires_grad for m in (model.image_encoder,model.prompt_encoder) for p in m.parameters()):
        raise ValueError('Both encoders must be frozen')
    with torch.no_grad():
        img_embed=model.image_encoder(image)
        sparse,dense=model.prompt_encoder(points=None,boxes=box[:,None,:],masks=None)
        image_pe=model.prompt_encoder.get_dense_pe()
    low,iou=model.mask_decoder(image_embeddings=img_embed,image_pe=image_pe,
        sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    if low.shape!=(b,1,256,256) or iou.shape!=(b,1):
        raise AssertionError(f'Unexpected MedSAM output {tuple(low.shape)}, {tuple(iou.shape)}')
    logits=F.interpolate(low,size=mask.shape[-2:],mode='bilinear',align_corners=False)
    if not torch.isfinite(logits).all() or not torch.isfinite(iou).all():
        raise FloatingPointError('Nonfinite MedSAM decoder output')
    return ForwardOutput(logits,low,iou)
