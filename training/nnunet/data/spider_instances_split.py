"""Split checks shared by the separate-label SPIDER experiment."""
import hashlib
import json
import re

def digest(value):
    """Hash JSON by content, independent of indentation and dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def patient_id(case):
    """Validate the local SPIDER convention: numeric patient plus MRI sequence."""
    match = re.fullmatch(r'([1-9][0-9]*)_(t1|t2|t2_SPACE)', case)
    if not match:
        raise ValueError(f'Invalid SPIDER case identifier: {case!r}')
    return match[1]


def validate_split(splits, expected_cases=None, approved_hash=None):
    """Reject duplicates, patient leakage, incomplete coverage, and altered designs."""
    if not isinstance(splits, list) or len(splits) != 1 or set(splits[0]) != {'train', 'val'}:
        raise ValueError('Expected exactly one train/val fold')
    train, val = splits[0]['train'], splits[0]['val']
    if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
        raise ValueError('Both partitions must be nonempty lists')
    cases = train + val
    patients_train = {patient_id(case) for case in train}
    patients_val = {patient_id(case) for case in val}
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate or overlapping cases')
    if patients_train & patients_val:
        raise ValueError('Patient leakage in internal split')
    if expected_cases is not None and set(cases) != set(expected_cases):
        raise ValueError('Split does not exactly cover dataset cases')
    if approved_hash is not None and digest(splits) != approved_hash:
        raise ValueError('Split differs from the verified baseline design')
    return train, val


