"""Lossless label-ID encoding for SPIDER; never merges vertebrae or discs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_mask(path):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f'Expected scalar 3D mask: {path}')
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f'Noninteger/nonfinite mask: {path}')
    if array.min() < 0 or array.max() > 65535:
        raise ValueError(f'Mask outside uint16 range: {path}')
    return image, array.astype(np.uint16)


def label_name(value):
    if value == 0:
        return 'background'
    if 1 <= value < 100:
        return f'vertebra_{value}'
    if value == 100:
        return 'spinal_canal'
    if 201 <= value <= 65535:
        return f'disc_{value}'
    raise ValueError(f'Unrecognized source label {value}; check annotation schema')


def fit_mapping(paths):
    values = {0}
    for path in paths:
        _, array = read_mask(path)
        values.update(map(int, np.unique(array)))
    if len(values) < 2:
        raise ValueError('No foreground in training masks')
    labels = {label_name(value): index for index, value in enumerate(sorted(values))}
    payload = {'schema_version': 1, 'source_to_encoded': {str(v): i for i, v in enumerate(sorted(values))},
               'labels': labels, 'fitted_on': 'TrainData only'}
    payload['sha256'] = digest(payload)
    return payload


def validate_mapping(mapping):
    payload = {k: v for k, v in mapping.items() if k != 'sha256'}
    if digest(payload) != mapping['sha256']:
        raise ValueError('Label mapping hash mismatch')
    forward = {int(k): int(v) for k, v in mapping['source_to_encoded'].items()}
    if forward.get(0) != 0 or sorted(forward.values()) != list(range(len(forward))):
        raise ValueError('Mapping must be one-to-one, consecutive, and preserve background')
    if {label_name(k): v for k, v in forward.items()} != mapping['labels']:
        raise ValueError('Mapping names disagree')
    return forward


def convert_mask(source, destination, mapping, inverse=False):
    forward = validate_mapping(mapping)
    table = {v: k for k, v in forward.items()} if inverse else forward
    image, array = read_mask(source)
    unknown = set(map(int, np.unique(array))) - table.keys()
    if unknown:
        raise ValueError(f'Unknown labels in {source}: {sorted(unknown)}; never map these to background')
    lookup = np.zeros(65536, dtype=np.uint16)
    for old, new in table.items():
        lookup[old] = new
    output = lookup[array]
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = sitk.GetImageFromArray(output)
    result.CopyInformation(image)
    sitk.WriteImage(result, str(destination), True)
    check, check_array = read_mask(destination)
    if not np.array_equal(check_array, output) or not same_geometry(image, check):
        raise ValueError(f'Written mask verification failed: {destination}')


# def same_geometry(a, b):
#     return (a.GetSize() == b.GetSize() and
#             all(np.allclose(x, y, rtol=0, atol=1e-5) for x, y in
#                 [(a.GetSpacing(), b.GetSpacing()), (a.GetOrigin(), b.GetOrigin()),
#                  (a.GetDirection(), b.GetDirection())]))

def same_geometry(a, b):
    """Accept matching grids within 0.001 mm; never modify geometry."""
    from itertools import product

    if a.GetDimension() != 3 or b.GetDimension() != 3:
        return False
    if a.GetSize() != b.GetSize():
        return False

    # For these affine image grids, corner comparisons capture the
    # maximum displacement between corresponding voxel centres.
    for corner in product(*[(0, n - 1) for n in a.GetSize()]):
        pa = np.asarray(a.TransformIndexToPhysicalPoint(corner))
        pb = np.asarray(b.TransformIndexToPhysicalPoint(corner))
        distance = np.linalg.norm(pa - pb)
        if not np.isfinite(distance) or distance > 0.001:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mapping', required=True, type=Path)
    parser.add_argument('--fit', action='store_true', help='Fit ONLY on training masks')
    parser.add_argument('--inverse', action='store_true', help='Restore source IDs in predictions')
    args = parser.parse_args()
    files = sorted(args.input_dir.glob('*.mha'))
    if not files:
        parser.error('No .mha masks found')
    if args.output_dir.exists() or (args.fit and args.mapping.exists()):
        parser.error('Output/mapping already exists; use a fresh destination')
    if args.fit and args.inverse:
        parser.error('--fit and --inverse are incompatible')
    mapping = fit_mapping(files) if args.fit else json.loads(args.mapping.read_text())
    validate_mapping(mapping)
    if args.fit:
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        with args.mapping.open('x') as handle:
            json.dump(mapping, handle, indent=2)
    for source in files:
        convert_mask(source, args.output_dir / source.name, mapping, args.inverse)
    print(f'Converted {len(files)} masks; originals preserved')


if __name__ == '__main__':
    main()
