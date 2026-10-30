"""Validate installed environments using synthetic data; no model weights required."""
import argparse
import importlib.metadata as metadata
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', choices=['nnunet', 'medsam', 'evaluation'])
    args = parser.parse_args()
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk
    import scipy
    import skimage
    import pydicom
    report = {'environment': args.environment, 'python': sys.version.split()[0],
              'numpy': np.__version__}
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    volume[4:12, 4:12, 4:12] = 1
    image = nib.Nifti1Image(volume, np.diag([1., 1., 2., 1.]))
    assert image.header.get_zooms() == (1., 1., 2.)
    assert sitk.GetArrayFromImage(sitk.GetImageFromArray(volume)).sum() == 512
    if args.environment == 'evaluation':
        import pandas as pd
        import pingouin as pg
        import statsmodels.api
        import sklearn
        import surface_distance
        import matplotlib
        import seaborn
        distances = surface_distance.compute_surface_distances(
            volume.astype(bool), volume.astype(bool), spacing_mm=(1., 1., 2.))
        assert surface_distance.compute_surface_dice_at_tolerance(distances, 1.) == 1.
        table = pd.DataFrame({'subject': np.tile(np.arange(10), 2),
                              'method': np.repeat(['reference', 'model'], 10),
                              'value': np.concatenate([np.arange(10, dtype=float), np.arange(10) + np.array([.1, -.1] * 5)])})
        icc = pg.intraclass_corr(data=table, targets='subject', raters='method', ratings='value')
        assert .99 < icc.loc[icc['Type'] == 'ICC2', 'ICC'].iloc[0] < 1.
        report['checks'] = ['MRI image handling', 'surface Dice', 'intraclass correlation']
    else:
        import torch
        import torchvision
        assert torch.cuda.is_available(), 'CUDA is unavailable'
        report.update(torch=torch.__version__, torchvision=torchvision.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        layer = torch.nn.Conv2d(1, 2, 3, padding=1).cuda()
        layer(torch.randn(1, 1, 32, 32, device='cuda')).square().mean().backward()
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.from_numpy(np.ones(2, dtype=np.float32)).numpy().sum() == 2
        if args.environment == 'nnunet':
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
            from dynamic_network_architectures.architectures.unet import PlainConvUNet
            from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
            report['nnunetv2'] = metadata.version('nnunetv2')
            report['checks'] = ['trainer and predictor imports', 'CUDA convolution backward', 'NumPy bridge']
        else:
            import monai
            import cv2
            import cc3d
            # Match upstream ViT-B architecture directly: its factory requires a checkpoint.
            from functools import partial
            from segment_anything.modeling import (
                Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer)
            model = Sam(
                image_encoder=ImageEncoderViT(
                    depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12,
                    patch_size=16, qkv_bias=True, use_rel_pos=True,
                    global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256),
                prompt_encoder=PromptEncoder(
                    embed_dim=256, image_embedding_size=(64, 64),
                    input_image_size=(1024, 1024), mask_in_chans=16),
                mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                  mlp_dim=2048, num_heads=8),
                    transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
                pixel_mean=[123.675, 116.28, 103.53],
                pixel_std=[58.395, 57.12, 57.375]).cuda().eval()
            with torch.inference_mode():
                prediction = model([{'image': torch.zeros(3, 1024, 1024, device='cuda'),
                                     'original_size': (64, 64),
                                     'boxes': torch.tensor([[100., 100., 800., 800.]], device='cuda')}],
                                   multimask_output=False)
            assert prediction[0]['masks'].shape == (1, 1, 64, 64)
            assert torch.isfinite(prediction[0]['low_res_logits']).all()
            report['monai'] = monai.__version__
            report['checks'] = ['MedSAM synthetic forward (random weights)', 'CUDA convolution backward', 'NumPy bridge']
        torch.cuda.synchronize()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
