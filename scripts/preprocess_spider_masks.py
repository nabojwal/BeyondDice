"""Convert SPIDER instance masks into four-class nnU-Net segmentation targets."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def convert_spider_masks(input_dir, output_dir):
    """Convert sorted *.mha masks, returning output Paths and reporting labels.

    Original masks are preserved. All destination conflicts are checked before
    conversion; other failures may leave previously converted masks in place.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mask_paths = sorted(input_dir.glob('*.mha'))
    if not mask_paths:
        raise FileNotFoundError(f'No *.mha files found in {input_dir}')

    # output_paths = [output_dir / f"{p.stem.removesuffix('mask')}.nii.gz"
    #                 for p in mask_paths]

    output_paths = [output_dir / f"{p.stem.removesuffix('mask')}.mha"
                    for p in mask_paths]


    counts = Counter(path.resolve() for path in output_paths)
    duplicates = [path for path, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError('Duplicate destination filenames:\n'
                         + '\n'.join(str(path) for path in duplicates))
    existing = [path for path in output_paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            'Conversion stopped to avoid overwriting existing masks:\n'
            + '\n'.join(str(path) for path in existing)
            + '\nChoose a new output folder.')

    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path, output_path in zip(mask_paths, output_paths):
        image = sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3:
            raise ValueError(f'{mask_path.name}: expected a 3D mask')
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f'{mask_path.name}: expected a scalar mask')
        mask = sitk.GetArrayFromImage(image)
        vertebrae = np.isin(mask, np.arange(1, 26))
        discs = np.isin(mask, np.arange(201, 226))
        canal = mask == 100
        valid = (mask == 0) | vertebrae | discs | canal
        if not np.all(valid):
            raise ValueError(f'{mask_path.name}: unexpected labels '
                             f'{np.unique(mask[~valid]).tolist()}')
        semantic = np.zeros(mask.shape, dtype=np.uint8)
        semantic[vertebrae] = 1
        semantic[discs] = 2
        semantic[canal] = 3
        result = sitk.GetImageFromArray(semantic, isVector=False)
        result.CopyInformation(image)
        sitk.WriteImage(result, str(output_path), True)

        saved = sitk.ReadImage(str(output_path))
        saved_array = sitk.GetArrayFromImage(saved)
        labels = np.unique(saved_array)
        if not np.array_equal(saved_array, semantic):
            raise ValueError(
                f'{output_path.name}: saved voxel labels differ from conversion')
        geometry_matches = saved.GetSize() == image.GetSize() and all(
            np.allclose(source, actual, rtol=1e-6, atol=1e-5)
            for source, actual in [
                (image.GetSpacing(), saved.GetSpacing()),
                (image.GetOrigin(), saved.GetOrigin()),
                (image.GetDirection(), saved.GetDirection())])
        if not geometry_matches:
            raise ValueError(f'{output_path.name}: saved geometry differs from source')
        if saved.GetPixelID() != sitk.sitkUInt8 or not np.all(np.isin(labels, [0, 1, 2, 3])):
            raise ValueError(f'{output_path.name}: invalid saved dtype or labels')
        print(f'{output_path.name}: labels {labels.tolist()}')
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Directory containing SPIDER *.mha instance masks')
    # parser.add_argument('--output-dir', required=True, type=Path,
    #                     help='Destination for compressed semantic *.nii.gz masks')

    parser.add_argument('--output-dir',required=True,type=Path,
                        help='Destination for compressed semantic *.mha masks')

    args = parser.parse_args()
    convert_spider_masks(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
