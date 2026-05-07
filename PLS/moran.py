# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore, pearsonr
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cdist

# BrainSpace Moran randomization
from brainspace.null_models.moran import moran_randomization
from brainspace.utils.parcellation import reduce_by_labels

# ================== CONFIG ==================
n_components = 21
n_perm = 1000
alpha = 0.05
z_thresh = 2.5
out_dir = "pls_moran_output_brainspace"
os.makedirs(out_dir, exist_ok=True)

# ================== PATHS ==================
gene_path     = "/data/users3/jalaparthi1/Merged_Atlas_Expression_output.csv"
contrast_path = "/data/users3/jalaparthi1/CBCL_ADHD65_CONTRAST_CORR/CBCL_ADHD65_minus_Control_Contrast_Template.nii"
mask_path     = "/data/users2/bbharatula1/ABCD-score/mask.nii"
atlas_path    = "/data/users3/jalaparthi1/Merged_Atlas.nii.gz"
coords_path   = "/data/users3/jalaparthi1/region_middle_voxels.csv"

# ================== LOAD DATA ==================
atlas_img  = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)
region_ids = np.unique(atlas_data)[1:]  # skip background
n_regions = len(region_ids)

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

contrast_img = nib.load(contrast_path)
contrast_data = contrast_img.get_fdata()

# Compute regional mean Y
masked_voxels = np.where(mask_data > 0, contrast_data, np.nan)
Y = np.array([np.nanmean(masked_voxels[atlas_data == rid]) for rid in region_ids]).reshape(-1, 1)

# Load gene expression
df = pd.read_csv(gene_path)
df = df.drop(columns=["label","id","hemisphere","structure"], errors="ignore")
X = df.select_dtypes(include=[float, int]).values
gene_names = df.columns.tolist()

# Standardize
X_scaled = StandardScaler().fit_transform(X)
Y_scaled = StandardScaler().fit_transform(Y)

# ================== SPATIAL WEIGHT MATRIX ==================
coords_df = pd.read_csv(coords_path)
coords = coords_df["MiddleVoxelMNI"].str.strip("()").str.split(",", expand=True).astype(float).values
D = cdist(coords, coords, metric="euclidean")
W = 1 / (D + np.eye(D.shape[0]))
np.fill_diagonal(W, 0)

# ================== MEM VARIABLES FOR MORAN ==================
evals, mem = np.linalg.eigh(W)
mem = mem[:, evals > 1e-10]
evals = evals[evals > 1e-10]

# ================== RUN OBSERVED PLS ==================
pls = PLSRegression(n_components=n_components)
pls.fit(X_scaled, Y_scaled)

X_scores = pls.x_scores_
Y_scores = pls.y_scores_
X_weights = pls.x_weights_

true_corrs = np.array([pearsonr(X_scores[:, i], Y_scores[:, i])[0] for i in range(n_components)])
true_r2 = r2_score(Y_scaled, pls.predict(X_scaled))

print("Observed R²:", true_r2)
print("Observed component correlations:", true_corrs)

# ================== NULL DISTRIBUTION VIA MORAN ==================
null_r2 = np.zeros(n_perm)
null_corrs = np.zeros((n_perm, n_components))

for p in range(n_perm):
    # Moran-randomized Y (returns 1D)
    Y_null = moran_randomization(Y_scaled.ravel(), mem, n_rep=1).ravel()
    Y_null = Y_null.reshape(-1, 1)

    pls_null = PLSRegression(n_components=n_components)
    pls_null.fit(X_scaled, Y_null)
    Y_pred_null = pls_null.predict(X_scaled).ravel()

    null_r2[p] = r2_score(Y_null, Y_pred_null)
    null_corrs[p, :] = [pearsonr(pls_null.x_scores_[:, i], pls_null.y_scores_[:, i])[0]
                        for i in range(n_components)]

# ================== EMPIRICAL P-VALUES ==================
p_r2 = (1 + np.sum(null_r2 >= true_r2)) / (1 + n_perm)
p_corrs = [(1 + np.sum(null_corrs[:, j] >= true_corrs[j])) / (1 + n_perm)
           for j in range(n_components)]

# Bonferroni correction
rej, p_bonf, _, _ = multipletests(p_corrs, alpha=alpha, method="bonferroni")

# ================== SAVE STATS ==================
stats_df = pd.DataFrame({
    "Component": np.arange(1, n_components+1),
    "r_obs": true_corrs,
    "p_empirical": p_corrs,
    "p_bonf": p_bonf,
    "Significant_Bonferroni": ["Yes" if x else "No" for x in rej]
})
stats_df.to_csv(os.path.join(out_dir, "PLS_component_stats.csv"), index=False)

overall_stats = {
    "true_r2": float(true_r2),
    "p_r2": float(p_r2),
    "true_corrs": true_corrs.tolist(),
    "p_corrs": p_corrs
}
pd.DataFrame([overall_stats]).to_csv(os.path.join(out_dir, "PLS_overall_stats.csv"), index=False)

print(stats_df)
print("Significant components:", stats_df.loc[stats_df.Significant_Bonferroni=="Yes","Component"].tolist())

# ================== IMPORTANT GENES ==================
important_genes = {}
for c in stats_df.loc[stats_df.Significant_Bonferroni=="Yes","Component"]:
    idx = c-1
    wts = X_weights[:, idx]
    z_wts = zscore(wts)
    sig_idx = np.where(np.abs(z_wts) >= z_thresh)[0]
    genes = [gene_names[i] for i in sig_idx]
    important_genes[c] = genes
    pd.Series(genes).to_csv(os.path.join(out_dir, f"important_genes_comp_{c}.csv"), index=False)

# Save combined gene list
if important_genes:
    all_genes = pd.concat([pd.DataFrame({"Component": c, "Gene": genes})
                           for c, genes in important_genes.items()])
    all_genes.to_csv(os.path.join(out_dir, "PLS_significant_genes.csv"), index=False)

# ================== COMMON GENES ==================
if important_genes:
    common_genes = set.intersection(*[set(gs) for gs in important_genes.values()])
else:
    common_genes = set()
pd.Series(list(common_genes)).to_csv(os.path.join(out_dir, "PLS_common_genes.csv"), index=False)

# ================== SAVE BRAIN MAPS ==================
for c in stats_df.loc[stats_df.Significant_Bonferroni=="Yes","Component"]:
    idx = c-1
    comp_map = np.zeros(atlas_data.shape)
    for i, rid in enumerate(region_ids):
        comp_map[atlas_data == rid] = Y_scores[i, idx]
    nib.save(nib.Nifti1Image(comp_map, atlas_img.affine),
             os.path.join(out_dir, f"PLS_comp_{c}_map.nii.gz"))

print("DONE Moran randomization + PLS complete!")
