"""Run: python verify_medsam_mri_pipeline.py [--image path.mha --mask path.mha --slice-index N --label-id K] [--checkpoint path.pth]."""
import argparse
from pathlib import Path
import unittest
import numpy as np
import torch
from mri_preprocessing import (extract_sagittal, normalize_medsam, resize_labels,
                               inverse_probabilities, assert_same_geometry)
from box_prompts import reference_box, external_box, to_model_box, box_to_native, validate_box

D = (1.,0.,0.,0.,1.,0.,0.,0.,1.)
ARGS = None

def synthetic_case():
    # ITK x index maps to NumPy axis 2, so sagittal axis is zyx axis 2.
    a = np.arange(7 * 13 * 5, dtype=np.float32).reshape(7,13,5)
    mask = np.zeros_like(a, dtype=np.uint16)
    mask[1:6,3:9,2] = 201
    mask[0:3,9:13,2] = 100
    return a, mask

class Part2Tests(unittest.TestCase):
    def setUp(self):
        a, mask = synthetic_case()
        self.s, self.g = extract_sagittal(a, spacing_xyz=(3.3,.6,.6), origin_lps=(11,22,33),
                                          direction_xyz=D, slice_index=2)
        self.m, _ = extract_sagittal(mask, spacing_xyz=(3.3,.6,.6), origin_lps=(11,22,33),
                                     direction_xyz=D, slice_index=2)

    def test_axis_orientation_and_physical_geometry(self):
        self.assertEqual(self.g.sagittal_axis_zyx, 2)
        self.assertEqual(self.g.native_hw, (7,13))
        self.assertEqual(self.g.voxel_zyx(0,0), (6,0,2))
        self.assertEqual(self.s[0,0], (np.arange(7*13*5).reshape(7,13,5))[6,0,2])
        np.testing.assert_allclose(self.g.physical_lps(0,0), (17.6,22,36.6))

    def test_axis_permutation_and_flip(self):
        # ITK x maps to physical superior, ITK z maps to physical left-right;
        # physical posterior maps to -ITK y.
        d = (0.,0.,1., 0.,-1.,0., 1.,0.,0.)
        arr = np.arange(3*4*5).reshape(3,4,5)
        out, g = extract_sagittal(arr, spacing_xyz=(1,2,3),origin_lps=(0,0,0),
                                  direction_xyz=d,slice_index=1)
        self.assertEqual(g.sagittal_axis_zyx, 0)
        self.assertEqual(out.shape, (5,4))
        self.assertEqual(g.voxel_zyx(0,0), (1,3,4))
        self.assertEqual(out[0,0], arr[1,3,4])

    def test_oblique_rejected(self):
        angle = np.pi/4
        d = (np.cos(angle),-np.sin(angle),0,np.sin(angle),np.cos(angle),0,0,0,1)
        with self.assertRaises(ValueError):
            extract_sagittal(np.zeros((5,5,5)), spacing_xyz=(1,1,1),
                             origin_lps=(0,0,0),direction_xyz=d,slice_index=0)

    def test_image_range_and_constant(self):
        t = normalize_medsam(self.s, model_size=96)
        self.assertEqual(t.shape, (3,96,96))
        self.assertEqual(t.dtype, torch.float32)
        self.assertTrue(torch.isfinite(t).all())
        self.assertGreaterEqual(float(t.min()),0)
        self.assertLessEqual(float(t.max()),1)
        np.testing.assert_array_equal(t[0],t[1])
        self.assertEqual(int(torch.count_nonzero(normalize_medsam(np.ones((5,7))*42,96))),0)

    def test_mask_alignment_exact_source(self):
        b = reference_box(self.m,201)
        self.assertIsNotNone(b)
        self.assertEqual(b.xyxy,(3.,1.,9.,6.))
        self.assertEqual(resize_labels(self.m,1024).dtype, self.m.dtype)
        self.assertEqual(set(np.unique(resize_labels(self.m,96)).tolist()), {0,100,201})
        # Backwards voxel mapping recovers original voxel and hence the original label.
        for r,c in np.argwhere(self.m==201):
            self.assertEqual(self.m[r,c], synthetic_case()[1][self.g.voxel_zyx(int(r),int(c))])

    def test_round_trip_and_boundary(self):
        b = reference_box(self.m,100,padding=20)
        self.assertEqual(b.xyxy,(0.,0.,13.,7.))
        native = external_box((1.5,2.25,12.,6.5),self.g.native_hw)
        model = to_model_box(native,self.g)
        self.assertEqual(tuple(model.shape),(1,1,4))
        np.testing.assert_allclose(box_to_native(model.numpy(),self.g),native.xyxy,atol=1e-6)
        self.assertEqual(validate_box((-4,-3,300,400),self.g.native_hw),(0.,0.,13.,7.))

    def test_empty_degenerate_perturbation(self):
        self.assertIsNone(reference_box(self.m,999))
        with self.assertRaises(ValueError): external_box((2,3,2,5),self.g.native_hw)
        with self.assertRaises(ValueError): external_box((-5,0,-1,5),self.g.native_hw)
        with self.assertRaises(ValueError): reference_box(self.m,201,jitter_xyxy=(200,0,-200,0))
        self.assertEqual(reference_box(self.m,201,padding=2).xyxy,(1.,0.,11.,7.))

    def test_inverse_probability_dimensions(self):
        p = torch.rand(1,1,1024,1024)
        self.assertEqual(inverse_probabilities(p,self.g).shape,(1,1,7,13))

    def test_real_sample_if_supplied(self):
        if not ARGS.image:
            self.skipTest('Real SPIDER .mha input not provided')
        image, labels, spacing, origin, direction = assert_same_geometry(ARGS.image,ARGS.mask)
        sl,g = extract_sagittal(image,spacing_xyz=spacing,origin_lps=origin,
                                direction_xyz=direction,slice_index=ARGS.slice_index)
        ml,_ = extract_sagittal(labels,spacing_xyz=spacing,origin_lps=origin,
                                direction_xyz=direction,slice_index=ARGS.slice_index)
        self.assertEqual(sl.shape,ml.shape)
        self.assertTrue(torch.isfinite(normalize_medsam(sl)).all())
        self.assertIsNotNone(reference_box(ml,ARGS.label_id))
        print(f'REAL SPIDER: volume={image.shape}, spacing={spacing}, origin={origin}, '
              f'direction={direction}, sagittal_axis_zyx={g.sagittal_axis_zyx}, native_hw={sl.shape}')
        if ARGS.overlay:
            make_overlay(sl,ml,reference_box(ml,ARGS.label_id),g,ARGS.overlay)


def make_overlay(img,mask,prompt,geometry,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    resized = normalize_medsam(img,geometry.model_hw[0])[0].numpy()
    mask_resized = resize_labels(mask,geometry.model_hw[0])
    b = prompt.xyxy
    mb = to_model_box(prompt,geometry).numpy().reshape(-1)
    fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,im,m,bounds,title in ((axs[0],img,mask,b,'Native MRI / mask / box'),
                                  (axs[1],resized,mask_resized,mb,'MedSAM 1024 / mask / box')):
        ax.imshow(im,cmap='gray'); ax.imshow(np.ma.masked_where(m!=prompt.label_id,m==prompt.label_id),
                    cmap='autumn',alpha=.3,interpolation='nearest')
        x0,y0,x1,y1=bounds
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='cyan',linewidth=1.5))
        ax.set_title(title)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=140)
    plt.close(fig)


def check_model(checkpoint,device):
    # Use the verified Part 1 module without modifying architecture or training flags.
    from medsam_setup import load_medsam
    model = load_medsam(checkpoint,device)
    if isinstance(model,tuple): model=model[0]
    dev = next(model.parameters()).device
    input_tensor=normalize_medsam(np.random.default_rng(43).uniform(0,255,(47,69)).astype(np.float32))
    _, geo=extract_sagittal(np.zeros((47,69,3),np.float32),spacing_xyz=(3,1,1),
                            origin_lps=(0,0,0),direction_xyz=D,slice_index=1)
    box=to_model_box(external_box((5,8,50,39),geo.native_hw),geo,device=dev)
    with torch.inference_mode():
        emb=model.image_encoder(input_tensor.unsqueeze(0).to(dev))
        sparse,dense=model.prompt_encoder(points=None,boxes=box,masks=None)
        logits,iou=model.mask_decoder(image_embeddings=emb,image_pe=model.prompt_encoder.get_dense_pe(),
                                      sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,
                                      multimask_output=False)
    assert emb.shape==(1,256,64,64) and logits.shape==(1,1,256,256) and iou.shape==(1,1)
    assert all(torch.isfinite(t).all().item() for t in (emb,sparse,dense,logits,iou))
    print('PART 1 MODEL INPUT COMPATIBILITY: PASS',dev)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--slice-index',type=int,default=0)
    parser.add_argument('--label-id',type=int,default=201)
    parser.add_argument('--overlay',type=Path,default=Path('outputs/part2_overlay.png'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cpu')
    ARGS=parser.parse_args()
    if bool(ARGS.image) != bool(ARGS.mask): parser.error('--image and --mask required together')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Part2Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    if not ARGS.image:
        a,m=synthetic_case()
        sl,g=extract_sagittal(a,spacing_xyz=(3.3,.6,.6),origin_lps=(11,22,33),
                              direction_xyz=D,slice_index=2)
        ml,_=extract_sagittal(m,spacing_xyz=(3.3,.6,.6),origin_lps=(11,22,33),
                              direction_xyz=D,slice_index=2)
        make_overlay(sl,ml,reference_box(ml,201),g,ARGS.overlay)
        print('SYNTHETIC OVERLAY:',ARGS.overlay)
    if ARGS.checkpoint: check_model(ARGS.checkpoint,ARGS.device)
    else: print('PART 1 MODEL INPUT COMPATIBILITY: NOT RUN (provide --checkpoint)')
