"""Prepare Dataset012 from TrainData/TestData, then finalize after nnU-Net preprocessing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
import SimpleITK as sitk
from preprocess_spider_masks import (convert_mask, digest, fit_mapping, read_mask,
                                     same_geometry, validate_mapping)

NAME = 'Dataset012_SPIDERInstances'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def case_id(path, mask=False):
    name = path.stem
    if mask:
        name = re.sub(r'(?:_?mask)$', '', name)
    else:
        name = name.removesuffix('_0000')
    if not re.fullmatch(r'[1-9][0-9]*_(t1|t2|t2_SPACE)', name):
        raise ValueError(f'Unexpected filename {path.name}; expected patient_sequence[mask].mha')
    return name


def paired(root):
    collections = []
    for folder, is_mask in [('images', False), ('masks', True)]:
        directory = Path(root) / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.iterdir())
        if not files or any(not p.is_file() or p.suffix != '.mha' for p in files):
            raise ValueError(f'{directory}: require a flat, nonempty folder of .mha files')
        entries = {}
        for path in files:
            case = case_id(path, is_mask)
            if case in entries:
                raise ValueError(f'Duplicate normalized identifier: {case}')
            entries[case] = path
        collections.append(entries)
    images, masks = collections
    if images.keys() != masks.keys():
        raise ValueError(f'Unpaired images/masks: {sorted(images.keys() ^ masks.keys())}')
    return {case: (images[case], masks[case]) for case in sorted(images)}


def group(case):
    return case.split('_', 1)[0]


def split_cases(cases, fraction=.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('Holdout fraction must be between 0 and 1')
    groups = sorted({group(c) for c in cases},
                    key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    count = math.ceil(len(groups) * fraction)
    if not 0 < count < len(groups):
        raise ValueError('Too few patient groups')
    held = set(groups[:count])
    return [{part: sorted(c for c in cases if (group(c) in held) == (part == 'val'))
             for part in ['train', 'val']}]


def roots():
    paths = {key: Path(os.environ[key]).resolve() for key in
             ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']}
    if len(set(paths.values())) != 3:
        raise ValueError('nnU-Net roots must be distinct')
    return paths


def prepare(args):
    locations = roots()
    for root in locations.values():
        if list(root.glob('Dataset012_*')):
            raise FileExistsError(f'Dataset012 already exists under {root}; do not overwrite it')
    train, test = paired(args.train), paired(args.test)
    if {group(c) for c in train} & {group(c) for c in test}:
        raise ValueError('Patient-prefix overlap between TrainData and TestData')
    split = split_cases(train, args.fraction, args.seed)
    mapping = fit_mapping(pair[1] for pair in train.values())
    allowed = set(validate_mapping(mapping))
    manifest = {}
    for partition, pairs in [('train', train), ('test', test)]:
        manifest[partition] = {}
        for case, (image_path, mask_path) in pairs.items():
            image = sitk.ReadImage(str(image_path))
            mask, array = read_mask(mask_path)
            if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError(f'Expected scalar 3D MRI: {image_path}')
            if not same_geometry(image, mask):
                raise ValueError(f'Image/mask geometry mismatch: {case}; review rather than overwrite metadata')
            unknown = set(map(int, np.unique(array))) - allowed
            if unknown:
                raise ValueError(f'{partition}/{case}: IDs absent from training: {sorted(unknown)}')
            manifest[partition][case] = {'image_sha256': sha_file(image_path),
                                        'mask_sha256': sha_file(mask_path),
                                        'source_label_ids': list(map(int, np.unique(array)))}
    counts = {f'{part}_scans': len(split[0][part]) for part in ['train', 'val']}
    counts.update({f'{part}_patients': len({group(c) for c in split[0][part]}) for part in ['train', 'val']})
    counts.update(test_scans=len(test), test_patients=len({group(c) for c in test}))
    dataset = {'channel_names': {'0': 'MRI'}, 'labels': mapping['labels'],
               'numTraining': len(train), 'file_ending': '.mha'}
    contract = {'schema_version': 1, 'dataset_id': 12, 'dataset_name': NAME,
                'split_sha256': digest(split), 'label_mapping_sha256': mapping['sha256'],
                'dataset_sha256': digest(dataset), 'source_manifest_sha256': digest(manifest),
                'counts': counts, 'seed': args.seed, 'validation_patient_fraction': args.fraction,
                'grouping': 'numeric filename prefix; must represent the same patient across sequences',
                'label_semantics': 'one channel per source ID; consistent IDs across scans required'}
    raw = locations['nnUNet_raw']
    raw.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.spider012-stage-', dir=raw))
    try:
        for partition, pairs in [('Tr', train), ('Ts', test)]:
            (stage / f'images{partition}').mkdir()
            (stage / f'labels{partition}').mkdir()
            for case, (image_path, mask_path) in pairs.items():
                destination = stage / f'images{partition}' / f'{case}_0000.mha'
                shutil.copy2(image_path, destination)
                if sha_file(destination) != manifest['train' if partition == 'Tr' else 'test'][case]['image_sha256']:
                    raise ValueError('Image copy checksum mismatch')
                convert_mask(mask_path, stage / f'labels{partition}' / f'{case}.mha', mapping)
        for filename, payload in [('dataset.json', dataset), ('spider_label_mapping.json', mapping),
                                  ('splits_final.json', split), ('spider_contract.json', contract),
                                  ('spider_source_manifest.json', manifest)]:
            write_json(stage / filename, payload)
        # Hash generated inputs so finalize can detect changes since preparation.
        generated = {str(p.relative_to(stage)).replace('\\', '/'): sha_file(p)
                     for folder in ['imagesTr', 'labelsTr'] for p in sorted((stage / folder).iterdir())}
        write_json(stage / 'spider_raw_checksums.json', generated)
        stage.rename(raw / NAME)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({'dataset': str(raw / NAME), 'counts': counts,
                      'labels': mapping['labels'], 'next': 'Run fresh planning/preprocessing, then finalize'}, indent=2))


def finalize(args):
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    locations = roots()
    raw, pre = locations['nnUNet_raw'] / NAME, locations['nnUNet_preprocessed'] / NAME
    contract, dataset = read_json(raw / 'spider_contract.json'), read_json(raw / 'dataset.json')
    split, mapping = read_json(raw / 'splits_final.json'), read_json(raw / 'spider_label_mapping.json')
    validate_mapping(mapping)
    if digest(dataset) != contract['dataset_sha256'] or digest(split) != contract['split_sha256']:
        raise ValueError('Prepared dataset/split changed')
    if mapping['sha256'] != contract['label_mapping_sha256'] or mapping['labels'] != dataset['labels']:
        raise ValueError('Prepared mapping changed')
    if digest(read_json(raw / 'spider_source_manifest.json')) != contract['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    checksums = read_json(raw / 'spider_raw_checksums.json')
    actual = {str(p.relative_to(raw)).replace('\\', '/'): sha_file(p)
              for folder in ['imagesTr', 'labelsTr'] for p in sorted((raw / folder).iterdir())}
    if actual != checksums:
        raise ValueError('Raw training inputs changed; do not reuse preprocessing')
    if read_json(pre / 'dataset.json') != dataset:
        raise ValueError('Preprocessed dataset metadata differs')
    plans = read_json(pre / 'nnUNetPlans.json')
    if plans['dataset_name'] != NAME:
        raise ValueError('Plans belong to a different dataset')
    manager = PlansManager(plans)
    folder = pre / manager.get_configuration('3d_fullres').data_identifier
    cases = infer_dataset_class(str(folder)).get_identifiers(str(folder))
    expected = split[0]['train'] + split[0]['val']
    if len(cases) != len(expected) or set(cases) != set(expected):
        raise ValueError('Preprocessing incomplete or has unexpected cases')
    contract['plans_sha256'] = digest(plans)
    payloads = {'splits_final.json': split, 'spider_contract.json': contract,
                'spider_label_mapping.json': mapping}
    for filename, payload in payloads.items():
        target = pre / filename
        if target.exists() and read_json(target) != payload:
            raise FileExistsError(f'Conflicting existing metadata: {target}')
    for filename, payload in payloads.items():
        if not (pre / filename).exists():
            write_json(pre / filename, payload)
    print('Verified 3d_fullres preprocessing and installed split/contract. Run the smoke test next.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--train', type=Path, required=True)
    prep.add_argument('--test', type=Path, required=True)
    prep.add_argument('--fraction', type=float, default=.2)
    prep.add_argument('--seed', type=int, default=42)
    commands.add_parser('finalize')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else finalize)(args)


if __name__ == '__main__':
    main()
