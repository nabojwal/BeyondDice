
from pathlib import Path
import numpy as np
import SimpleITK as sitk

mask_path = Path(
    r"D:\Courses\MD5001\Mini Project\MedSAM"
    r"\TrainData\masks\1_t1.mha"
)

mask = sitk.ReadImage(str(mask_path))
array = sitk.GetArrayFromImage(mask)

print("Mask shape (Z,Y,X):", array.shape)
print("Spacing (X,Y,Z):", mask.GetSpacing())
print("Direction:", mask.GetDirection())
print("Origin:", mask.GetOrigin())

labels, counts = np.unique(array, return_counts=True)

print("\nLabels in entire volume:")
for label, count in zip(labels, counts):
    print(f"Label {label}: {count:,} voxels")

print("\nSagittal slices containing label 201:")

# Applicable only to this file if its sagittal axis is X.
for index in range(array.shape[2]):
    count = np.count_nonzero(array[:, :, index] == 201)
    if count:
        print(f"Slice {index}: {count} pixels")
