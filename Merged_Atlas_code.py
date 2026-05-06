import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
import abagen

# === Correct Paths to input atlases ===
atlas1_path = '/data/users3/jalaparthi1/rabcdsmri_Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_1mm.nii'
atlas2_path = '/data/users3/jalaparthi1/rabcdsmri_atl-NettekovenSym32_space-MNI152NLin2009cSymC_dseg_forreslice.nii'

# === Load atlases ===
atlas1_img = nib.load(atlas1_path)
atlas2_img = nib.load(atlas2_path)

# === Resample cerebellum atlas to match base atlas ===
atlas2_resampled = resample_to_img(atlas2_img, atlas1_img, interpolation='nearest')

# === Get data arrays ===
atlas1_data = atlas1_img.get_fdata().astype(int)
atlas2_data = atlas2_resampled.get_fdata().astype(int)

# === Relabel cerebellum regions (start from 455) ===
atlas2_data[atlas2_data > 0] += 454

# === Merge: keep cerebellum label where it's non-zero ===
merged_data = np.where(atlas2_data > 0, atlas2_data, atlas1_data)

# === Create and save merged NIfTI ===
merged_atlas_path = '/data/users3/jalaparthi1/Merged_Atlas.nii.gz'  # Corrected path
merged_img = nib.Nifti1Image(merged_data, atlas1_img.affine, atlas1_img.header)
nib.save(merged_img, merged_atlas_path)

print(f"Merged NIfTI saved as '{merged_atlas_path}'")

# === Load merged atlas ===
atlas_img = nib.load(merged_atlas_path)
atlas_data = atlas_img.get_fdata()
affine = atlas_img.affine

# === Get all unique region labels (exclude 0) ===
unique_labels = np.unique(atlas_data)
unique_labels = unique_labels[unique_labels != 0]

# === Build table with ID, hemisphere, structure ===
atlas_info = []

for label in unique_labels:
    coords = np.array(np.where(atlas_data == label)).T
    voxel_coords = coords[0]
    # MNI coordinates for the voxel
    mni_coords = affine.dot(np.append(voxel_coords, 1))[:3]  # This is MNI coordinates, not centroids
    hemisphere = 'L' if mni_coords[0] < 0 else 'R'

    # Classifying the region into anatomical structures
    if 1 <= label <= 54:
        structure = 'subcortex'
    elif 55 <= label <= 454:
        structure = 'cortex'
    elif 455 <= label <= 486:
        structure = 'cerebellum'
    else:
        structure = 'unknown'

    atlas_info.append({
        'id': int(label),
        'hemisphere': hemisphere,
        'structure': structure
    })

# === Save region metadata to CSV ===
merged_atlas_info_path = '/data/users3/jalaparthi1/Merged_Atlas_Info.csv'  # Corrected path
df = pd.DataFrame(atlas_info)
df.to_csv(merged_atlas_info_path, index=False)

print(f"CSV saved as '{merged_atlas_info_path}'")

# === Load region info for expression analysis ===
atlas_info_df = pd.read_csv(merged_atlas_info_path)

# === Get expression data with centroid filling ===
expression = abagen.get_expression_data(
    atlas=merged_atlas_path,
    atlas_info=atlas_info_df,
    missing='centroids'  # Use centroid-based interpolation for missing regions
)

# === Merge in labels for clarity ===
expression = expression.reset_index()  # 'label' becomes a column
expression = expression.merge(atlas_info_df, left_on='label', right_on='id', how='left')

# === Save expression data to CSV ===
merged_atlas_expression_output_path = '/data/users3/jalaparthi1/Merged_Atlas_Expression_output.csv'  # Corrected path
expression.to_csv(merged_atlas_expression_output_path, index=False)

print(f"Expression data saved to '{merged_atlas_expression_output_path}'")
