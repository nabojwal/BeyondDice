"""3D binary and label-wise metrics. Missing-structure conventions are explicit.

ASD implements SPIDER's symmetric pooled-surface-distance concept using
full connectivity; not an assertion of leaderboard equivalence for matching.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from part7_reconstruction import VALID_LABELS


def dice_3d(gt, pred):
    a=np.asarray(gt,dtype=bool);b=np.asarray(pred,dtype=bool)
    if a.shape!=b.shape or a.ndim!=3: raise ValueError('Expected same-shape 3D arrays')
    denominator=int(a.sum())+int(b.sum())
    if denominator==0: return None  # explicit both-empty N/A
    return float(2*np.count_nonzero(a&b)/denominator)


def asd_mm(gt, pred, spacing_xyz):
    a=np.asarray(gt,dtype=bool);b=np.asarray(pred,dtype=bool)
    if a.ndim!=3 or a.shape!=b.shape: raise ValueError('Expected same-shape 3D arrays')
    spacing=np.asarray(tuple(spacing_xyz),dtype=float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or np.any(spacing<=0):
        raise ValueError('Invalid XYZ spacing')
    if not a.any() or not b.any(): return None  # missing: not a finite ASD
    kernel=generate_binary_structure(3,3)
    ba=a ^ binary_erosion(a,structure=kernel)
    bb=b ^ binary_erosion(b,structure=kernel)
    # NumPy z,y,x -> ITK xyz reversed
    spacing_zyx=tuple(spacing[::-1])
    d_ab=distance_transform_edt(~ba,sampling=spacing_zyx)[bb]
    d_ba=distance_transform_edt(~bb,sampling=spacing_zyx)[ba]
    return float(np.concatenate((d_ab,d_ba)).mean())


def score_instances(reference, prediction, spacing_xyz, label_ids=sorted(VALID_LABELS)):
    gt=np.asarray(reference);pred=np.asarray(prediction)
    if gt.shape!=pred.shape or gt.ndim!=3: raise ValueError('Volumes must match')
    unknown=set(map(int,np.unique(gt))) | set(map(int,np.unique(pred)))
    if unknown-{0}-VALID_LABELS: raise ValueError(f'Unexpected label IDs: {unknown-{0}-VALID_LABELS}')
    result=[]
    for label in label_ids:
        a=gt==label;b=pred==label
        if not a.any() and not b.any(): continue
        result.append({'label_id':label, 'reference_present':bool(a.any()),
                       'prediction_present':bool(b.any()),
                       'dice_3d':dice_3d(a,b),
                       'asd_mm':asd_mm(a,b,spacing_xyz),
                       'status':'matched' if a.any() and b.any() else ('missed' if a.any() else 'false_positive')})
    return result


def detection_summary(rows):
    positives=sum(bool(r['reference_present']) for r in rows)
    found=sum(r['status']=='matched' for r in rows)
    false_positives=sum(r['status']=='false_positive' for r in rows)
    return {'reference_structures':positives,'detected_structures':found,
            'detection_rate':found/positives if positives else None,
            'false_positive_structures':false_positives,
            'note':'ID-presence detection, not official SPIDER matching'}
