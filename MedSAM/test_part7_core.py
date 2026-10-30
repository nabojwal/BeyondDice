import unittest
import tempfile
from pathlib import Path
import numpy as np
from mri_preprocessing import extract_sagittal
from part7_reconstruction import NativeVolume,VolumeAssembler,verify_mha_geometry
from part7_metrics import dice_3d,asd_mm,score_instances,detection_summary

class TestReconstruction(unittest.TestCase):
    def setUp(self):
        self.shape=(4,5,6)
        self.ref=NativeVolume(self.shape,(2.,3.,4.),(8.,-3.,1.),(1.,0.,0.,0.,1.,0.,0.,0.,1.))
        self.zeros=np.zeros(self.shape,dtype=np.uint8)
    def geometry(self, n=0, direction=None):
        return extract_sagittal(self.zeros,spacing_xyz=self.ref.spacing_xyz,
            origin_lps=self.ref.origin_lps,direction_xyz=direction or self.ref.direction_xyz,
            slice_index=n)[1]
    def test_slice_axis_mapping(self):
        for axis in range(self.shape[2]):
            geo=self.geometry(axis)
            native=np.zeros(geo.native_hw,dtype=np.float32)
            native[1:3,1:4]=.8
            a=VolumeAssembler(self.ref)
            a.insert(201,native,geo)
            restored,_=extract_sagittal(a.labels,spacing_xyz=self.ref.spacing_xyz,
                origin_lps=self.ref.origin_lps,direction_xyz=self.ref.direction_xyz,slice_index=axis)
            np.testing.assert_array_equal(restored,(native>=.5)*201)
    def test_direction_and_flips(self):
        ref=NativeVolume(self.shape,self.ref.spacing_xyz,self.ref.origin_lps,
                         (-1.,0.,0.,0.,-1.,0.,0.,0.,-1.))
        _,geo=extract_sagittal(self.zeros,spacing_xyz=ref.spacing_xyz,
            origin_lps=ref.origin_lps,direction_xyz=ref.direction_xyz,slice_index=2)
        a=VolumeAssembler(ref)
        p=np.zeros(geo.native_hw,dtype=np.float32);p[0,0]=1
        a.insert(100,p,geo)
        sl,_=extract_sagittal(a.labels,spacing_xyz=ref.spacing_xyz,
            origin_lps=ref.origin_lps,direction_xyz=ref.direction_xyz,slice_index=2)
        self.assertEqual(sl[0,0],100)
    def test_overlap_tie_and_missing(self):
        g=self.geometry(1); a=VolumeAssembler(self.ref)
        p=np.ones(g.native_hw,dtype=np.float32)*.8
        a.insert(202,p,g);a.insert(201,p,g)
        self.assertTrue(np.all(a.labels[:,:,1]==201))
        self.assertFalse(a.labels[:,:,0].any())
        a.insert(202,np.ones(g.native_hw,dtype=np.float32)*.9,g)
        self.assertTrue(np.all(a.labels[:,:,1]==202))
    def test_invalid_probability(self):
        g=self.geometry();a=VolumeAssembler(self.ref)
        with self.assertRaises(ValueError):a.insert(201,np.ones(g.native_hw)*1.1,g)
    def test_mha_roundtrip(self):
        try:import SimpleITK
        except ImportError:self.skipTest('SimpleITK unavailable')
        g=self.geometry();a=VolumeAssembler(self.ref)
        a.insert(1,np.ones(g.native_hw,dtype=np.float32),g)
        with tempfile.TemporaryDirectory() as temp:
            p=a.write_mha(Path(temp)/'pred.mha')
            np.testing.assert_array_equal(verify_mha_geometry(p,self.ref),a.labels)

class TestMetrics(unittest.TestCase):
    def test_identical_anisotropic(self):
        a=np.zeros((8,8,8),dtype=bool);a[2:5,2:6,1:4]=1
        self.assertEqual(dice_3d(a,a),1.)
        self.assertEqual(asd_mm(a,a,(.5,1.,3.)),0.)
    def test_shift_physical_units(self):
        a=np.zeros((9,9,9),dtype=bool);b=np.zeros_like(a)
        a[3,3,3]=1;b[4,3,3]=1
        self.assertEqual(asd_mm(a,b,(.5,1.,3.)),3.)
    def test_empty_missing_and_detection(self):
        a=np.zeros((3,3,3),dtype=np.uint16);b=a.copy();a[1,1,1]=201;b[1,1,2]=100
        rows=score_instances(a,b,(1.,1.,1.))
        self.assertEqual(detection_summary(rows)['detection_rate'],0.)
        self.assertIsNone(dice_3d(np.zeros_like(a),np.zeros_like(b)))
        self.assertIsNone(asd_mm(a==201,b==201,(1,1,1)))

if __name__=='__main__':unittest.main()
