import pandas as pd
import nibabel as nib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------- PATHS ----------------------
base_dir = "/data/users3/jalaparthi1"
expression_file = os.path.join(base_dir, "Merged_Atlas_Expression_output.csv") 
atlas_path = os.path.join(base_dir, "Merged_Atlas.nii.gz")  # Atlas file path
output_dir = os.path.join(base_dir, "SZ_HC")  
os.makedirs(output_dir, exist_ok=True)

# ---------------------- LOAD EXPRESSION DATA ----------------------
df = pd.read_csv(expression_file)

# Check if 'label' column is present in the data
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in expression CSV.")

# Genes of interest (list you provided earlier)
genes_of_interest = [
    "MAP7", "ELMO1", "ZKSCAN8", "ANKRD27", "MDGA1", "TOB2", "MIRLET7G", "MIR124-1",
    "PLCG1", "S100A12", "RPS6KA3", "FRMPD3", "NUP214", "TNRC6A", "BARHL2", "PAM",
    "CNR1", "FRS2", "PPP1CC", "LAMB1", "RTN4RL1", "SLC22A1", "TNRC6C", "SKP2", "SFXN2",
    "FGF5", "RBL2", "TFCP2L1", "NR1D1", "ZNF350", "DIP2C", "SLC6A10P", "VEPH1", "SUPT6H",
    "DCLK1", "AKR1C1", "ZNF57", "MIR29B2", "RFT1", "BMP3", "ZNF655", "STMN2", "TPCN1",
    "DNAH5", "TNRC6B", "DHDDS", "CHGB", "ADCYAP1R1", "C14orf159", "MTMR3", "OAZ1", "AACS",
    "NIM1", "TMEM56", "TBL1XR1", "NKAPP1", "CHD6", "PRDM1", "CATSPERG", "MAP3K5", "MAX",
    "pRbBP-39", "PIK3R1", "KIFAP3", "SCG2", "DLG5", "TCTE1", "SORBS2", "CCDC77", "ZBTB16",
    "AKR1C2", "AGPAT4-IT1", "C4orf22", "MRAP2", "CNOT4", "GNAZ", "SH3GL3", "NYNRIN", "ELL",
    "MIR181B1", "MIR15A", "HML-2", "ZBTB34", "NISCH", "TMEM164", "CPNE5", "MIR148B",
    "FBXL21P", "ZNF146", "GOSR2", "RBM27", "XRN1", "ATRIP", "MAGI3", "CEP95", "ARHGAP21",
    "BCL2", "MCM5", "INSR", "FLJ39257", "ITGA8", "ZBTB34", "ZNF805", "ZNF585A", "ANKRD31",
    "MAVS", "MIR495", "CHPT1", "NREP", "TIGD1"
]

# ---------------------- FILTER DATASET ----------------------

# Determine which genes of interest are present in the dataset
base_columns = ['label']  # Add any other required metadata columns here (e.g., 'id', 'name')
available_genes = [g for g in genes_of_interest if g in df.columns]

# Warn if some genes are missing
missing_genes = [g for g in genes_of_interest if g not in df.columns]
if missing_genes:
    print(f"Warning: {len(missing_genes)} genes not found in file and will be skipped.")
    print(f"Missing genes: {missing_genes}")

# Filter the DataFrame to include only the available genes
filtered_df = df[base_columns + available_genes]

# Save the filtered gene expression data
filtered_output_path = os.path.join(output_dir, "SZ_HC_Filtered_Gene_Expression.csv")
filtered_df.to_csv(filtered_output_path, index=False)
print(f"Filtered expression data saved to: {filtered_output_path}")


# ---------------------- PCA ----------------------

# Remove the 'label' column for PCA
region_ids = df["label"].values
expression_data = df.drop(columns=["label"])

# Standardize the data (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(expression_data)

# Apply PCA (Principal Component Analysis)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# ---------------------- SAVE REGION PCA SCORES ----------------------
region_scores = pd.DataFrame(
    X_pca[:, :19],  # Top 19 principal components
    index=region_ids,
    columns=[f"PC{i+1}" for i in range(19)]
)
region_scores.index.name = "RegionID"

# ---------------------- SAVE GENE LOADINGS ----------------------
loadings = pd.DataFrame(
    pca.components_[:19].T,
    index=expression_data.columns,
    columns=[f"PC{i+1}" for i in range(19)]
)
loadings.index.name = "Gene"

# ---------------------- FLIP SIGN FOR PCs 1,2,3 BASED ON PEAK MAG ----------------------
pcs_to_check = [1, 2, 3]

for pc in pcs_to_check:
    pc_col = f"PC{pc}"
    scores = region_scores[pc_col]
    max_pos = scores[scores > 0].max() if any(scores > 0) else 0
    max_neg = scores[scores < 0].min() if any(scores < 0) else 0  # negative value

    if abs(max_neg) > abs(max_pos):
        region_scores[pc_col] = -region_scores[pc_col]
        loadings[pc_col] = -loadings[pc_col]
        print(f"Flipped sign of {pc_col} due to higher peak negative magnitude.")

# ---------------------- SAVE RESULTS ----------------------
region_scores_path = os.path.join(output_dir, "SZ_HC_Region_PC_Matrix.csv")
region_scores.to_csv(region_scores_path)
print(f"PCA scores saved: {region_scores_path}")

loadings_path = os.path.join(output_dir, "SZ_HC_Gene_PC_Matrix.csv")
loadings.to_csv(loadings_path)
print(f"Gene loadings saved: {loadings_path}")

# ---------------------- SAVE EXPLAINED VARIANCE ----------------------
explained_variance = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "ExplainedVarianceRatio": pca.explained_variance_ratio_,
    "CumulativeExplainedVariance": np.cumsum(pca.explained_variance_ratio_)
})
variance_path = os.path.join(output_dir, "SZ_HC_PCA_Variance.csv")
explained_variance.to_csv(variance_path, index=False)
print(f"Explained variance saved: {variance_path}")

# ---------------------- LOAD ATLAS ----------------------
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
affine = atlas_img.affine

# ---------------------- MAP TOP 3 PCs TO NIFTI ----------------------
for pc in pcs_to_check:
    pc_col = f"PC{pc}"
    scores = region_scores[pc_col].values

    pc_volume = np.zeros_like(atlas_data)

    for region_idx, score in zip(region_ids, scores):
        pc_volume[atlas_data == region_idx] = score

    pc_nifti = nib.Nifti1Image(pc_volume, affine)
    pc_nii_path = os.path.join(output_dir, f"SZ_HC_PC{pc}_map.nii.gz")
    nib.save(pc_nifti, pc_nii_path)
    print(f"NIfTI saved: {pc_nii_path}")

# ---------------------- SAVE ALL GENE CONTRIBUTORS BY ABS WEIGHT FOR PC1–PC3 ----------------------
contrib_dir = os.path.join(output_dir, "SZ_HC_PC_Contributors")
os.makedirs(contrib_dir, exist_ok=True)

for pc in pcs_to_check:
    pc_col = f"PC{pc}"
    sorted_genes = loadings[[pc_col]].reindex(loadings[pc_col].abs().sort_values(ascending=False).index)
    full_path = os.path.join(contrib_dir, f"all_genes_sorted_by_abs_weight_{pc_col}.csv")
    sorted_genes.to_csv(full_path)
    print(f"Saved all genes sorted by abs weight for {pc_col} at {full_path}")
