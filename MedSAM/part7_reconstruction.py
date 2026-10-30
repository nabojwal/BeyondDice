"""Native-space, instance-aware SPIDER reconstruction (NumPy arrays use z,y,x).

Overlap winner: largest probability at each voxel; exact ties use lowest label ID.
An absent slice is background. Oracle prompts are not independent detections.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from mri_preprocessing import SliceGeometry

VALID_LABELS = frozenset(range(1, 26)) | {100} | frozenset(range(201, 226))

@dataclass(frozen=True)
class NativeVolume:
    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction_xyz: tuple[float, ...]

    @classmethod
    def from_image(cls, image):
        import SimpleITK as sitk
        if image.GetDimension() != 3: raise ValueError('Expected 3D image')
        return cls(tuple(reversed(image.GetSize())),tuple(image.GetSpacing()),
                   tuple(image.GetOrigin()),tuple(image.GetDirection()))

    def assert_geo(self, geo: SliceGeometry):
        if (geo.volume_shape_zyx != self.shape_zyx or
            not np.allclose(geo.spacing_xyz,self.spacing_xyz) or
            not np.allclose(geo.origin_lps,self.origin_lps) or
            not np.allclose(geo.direction_xyz,self.direction_xyz)):
            raise ValueError('Slice native geometry differs from destination volume')


class VolumeAssembler:
    def __init__(self, reference: NativeVolume, threshold: float = 0.5):
        if not (0 < threshold < 1): raise ValueError('Threshold must lie in (0,1)')
        self.reference = reference
        self.threshold = threshold
        self.labels = np.zeros(reference.shape_zyx,dtype=np.uint16)
        self.confidence = np.full(reference.shape_zyx,-np.inf,dtype=np.float32)
        self.sources = []

    def insert(self, label_id: int, probability_hw: np.ndarray, geo: SliceGeometry,
               provenance: dict | None = None):
        if label_id not in VALID_LABELS: raise ValueError(f'Invalid SPIDER label {label_id}')
        self.reference.assert_geo(geo)
        prob=np.asarray(probability_hw,dtype=np.float32)
        if prob.shape!=geo.native_hw or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
            raise ValueError('Expected finite native-size probability in [0,1]')
        # Layout transpose and flips exactly inverse to extract_sagittal.
        native=prob
        if geo.flip_rows: native=np.flip(native,axis=0)
        if geo.flip_columns: native=np.flip(native,axis=1)
        remaining=[i for i in range(3) if i!=geo.sagittal_axis_zyx]
        native=np.transpose(native, [ [geo.row_axis_zyx,geo.column_axis_zyx].index(a) for a in remaining])
        index=[slice(None)]*3
        index[geo.sagittal_axis_zyx]=geo.slice_index
        index=tuple(index)
        conf=self.confidence[index]; lab=self.labels[index]
        mask=native>=self.threshold
        winner=mask & ((native>conf)|((native==conf)&(label_id<lab)))
        conf[winner]=native[winner];lab[winner]=label_id
        self.sources.append({'label_id':int(label_id),'slice_index':int(geo.slice_index),
                             'accepted_voxels_at_insert':int(winner.sum()),'provenance':provenance or {}})

    def write_mha(self, path: str | Path):
        import SimpleITK as sitk
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        out=sitk.GetImageFromArray(self.labels)
        out.SetSpacing(self.reference.spacing_xyz)
        out.SetOrigin(self.reference.origin_lps)
        out.SetDirection(self.reference.direction_xyz)
        sitk.WriteImage(out,str(p),useCompression=True)
        return p


def verify_mha_geometry(path: str | Path, expected: NativeVolume):
    import SimpleITK as sitk
    image=sitk.ReadImage(str(path))
    assert NativeVolume.from_image(image)==expected, 'Export geometry mismatch'
    return sitk.GetArrayFromImage(image)
