
import argparse
from pathlib import Path
import SimpleITK as sitk


def inspect_mri(path: str) -> None:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    reader.ReadImageInformation()

    print(f"File: {file_path}")
    print(f"Dimensions: {reader.GetSize()}")
    print(f"Spacing (mm): {reader.GetSpacing()}")
    print(f"Origin: {reader.GetOrigin()}")
    print(f"Direction: {reader.GetDirection()}")

    keys = sorted(reader.GetMetaDataKeys())

    print("\nAvailable metadata:")
    for key in keys:
        print(f"{key}: {reader.GetMetaData(key)}")

    sequence_keys = (
        "0008|103e",  # Series Description
        "0018|1030",  # Protocol Name
        "0018|0080",  # Repetition Time
        "0018|0081",  # Echo Time
        "0018|0020",  # Scanning Sequence
        "0018|0021",  # Sequence Variant
    )

    print("\nSequence metadata:")
    found = False

    for key in sequence_keys:
        if reader.HasMetaDataKey(key):
            print(f"{key}: {reader.GetMetaData(key)}")
            found = True

    if not found:
        print(
            "No explicit DICOM sequence fields found. "
            "MRI contrast cannot be confirmed from this "
            "file's metadata alone."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    inspect_mri(args.image)
