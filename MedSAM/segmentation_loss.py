"""Stable BCE-with-logits plus per-example soft Dice, for binary masks."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor


def bce_soft_dice(logits:torch.Tensor,target:torch.Tensor,*,bce_weight:float=1.0,
                  dice_weight:float=1.0,smooth:float=1.0)->LossOutput:
    if logits.shape!=target.shape or logits.ndim!=4 or logits.shape[1]!=1:
        raise ValueError(f'Expected matching [B,1,H,W]; got {logits.shape} and {target.shape}')
    if not logits.is_floating_point() or not target.is_floating_point():
        raise TypeError('Logits and target must be floating-point')
    if logits.device!=target.device:raise ValueError('Device mismatch')
    if bce_weight<0 or dice_weight<0 or bce_weight+dice_weight<=0 or smooth<=0:
        raise ValueError('Invalid loss weights or smoothing')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite logits or targets')
    if not torch.all((target==0)|(target==1)):
        raise ValueError('Non-binary target mask')
    target=target.to(logits.dtype)
    bce=F.binary_cross_entropy_with_logits(logits,target)
    prob=logits.sigmoid()
    dims=(1,2,3)
    intersection=(prob*target).sum(dims)
    dice=1-(2*intersection+smooth)/(prob.sum(dims)+target.sum(dims)+smooth)
    dice=dice.mean()
    total=bce_weight*bce+dice_weight*dice
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss')
    return LossOutput(total,bce,dice)
