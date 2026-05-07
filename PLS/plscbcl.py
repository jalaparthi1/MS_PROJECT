import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import multipletests

# ================== PATHS ==================
gene_path = "/data/users3/jalaparthi1/Merged_Atlas_Expression_output.csv"
contrast_path = "/data/users3/jalaparthi1/CBCL_ADHD65_CONTRAST_CORR/CBCL_ADHD65_minus_Control_Contrast_Template.nii"
mask_path = "/data/users2/bbharatula1/ABCD-score/mask.nii"
atlas_path = "/data/users3/jalaparthi1/Merged_Atlas.nii.gz"

outdir = "pls_results_no_perm_bonferroni"
os.makedirs(outdir, exist_ok=True)

# ================== PARAMETERS ==================
n_components = 21
alpha = 0.05
z_thresh = 2.5

# ================== LOAD GENE EXPRESSION ==================
gene_df = pd.read_csv(gene_path)
numeric_cols = gene_df.select_dtypes(include=[np.number]).columns
X = gene_df[numeric_cols].values
print("X shape:", X.shape)

# ================== LOAD ATLAS / CONTRAST ==================
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

contrast_img = nib.load(contrast_path)
contrast_data = contrast_img.get_fdata()

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

# ================== EXTRACT REGIONAL Y ==================
region_ids = np.unique(atlas_data)
region_ids = region_ids[region_ids > 0]

Y = []
for rid in region_ids:
    vox = contrast_data[(atlas_data == rid) & (mask_data > 0)]
    Y.append(np.nanmean(vox) if len(vox) > 0 else np.nan)

Y = np.array(Y).reshape(-1, 1)

valid = ~np.isnan(Y[:, 0])
X = X[valid]
Y = Y[valid]
region_ids = region_ids[valid]
print("Y shape:", Y.shape)

# ================== SCALE ==================
Xz = StandardScaler().fit_transform(X)
Yz = StandardScaler().fit_transform(Y)

# ================== PLS ==================
pls = PLSRegression(n_components=n_components)
pls.fit(Xz, Yz)

Xs = pls.x_scores_   # <-- Brain scores
Ys = pls.y_scores_
Xw = pls.x_weights_

# ================== COMPONENT CORRELATIONS ==================
r_vals, p_vals = [], []
n = Xs.shape[0]

for i in range(n_components):
    r, _ = pearsonr(Xs[:, i], Ys[:, i])
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * (1 - t.cdf(abs(t_stat), df=n - 2))
    r_vals.append(r)
    p_vals.append(p)

r_vals = np.array(r_vals)
p_vals = np.array(p_vals)

# Compute true R2 for observed PLS
true_r2 = float(r2_score(Yz, pls.predict(Xz)))
p_r2 = None  # no permutation here

# ================== BONFERRONI ==================
rej, p_bonf, _, _ = multipletests(p_vals, alpha=alpha, method="bonferroni")

# ================== SAVE COMPONENT STATS ==================
stats_df = pd.DataFrame({
    "Component": np.arange(1, n_components + 1),
    "r_obs": r_vals,
    "p_raw": p_vals,
    "p_bonf": p_bonf,
    "Significant_Bonferroni": ["Yes" if x else "No" for x in rej]
})
stats_df.to_csv(f"{outdir}/PLS_component_stats.csv", index=False)
print(stats_df)

sig_components = stats_df.loc[stats_df["Significant_Bonferroni"]=="Yes", "Component"].tolist()
print("Significant components:", sig_components)

# ================== SAVE R2 AND CORRELATION METRICS ==================
metrics = {
    "true_r2": true_r2,
    "p_r2": p_r2,
    "true_corrs": r_vals.tolist(),
    "p_corrs": p_vals.tolist()
}
pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "PLS_r2_and_corr_metrics.csv"), index=False)

# ================== IMPORTANT GENES ==================
important_genes = {}

for c in sig_components:
    idx = c - 1
    w = Xw[:, idx]
    z = zscore(w)
    sel = np.where(np.abs(z) >= z_thresh)[0]
    genes = numeric_cols[sel]
    important_genes[c] = genes
    pd.Series(genes).to_csv(f"{outdir}/important_genes_component_{c}.csv", index=False)

# ================== COMMON GENES ==================
common_genes = set.intersection(*map(set, important_genes.values())) if important_genes else set()
pd.Series(list(common_genes)).to_csv(f"{outdir}/common_genes.csv", index=False)
print("Common genes:", common_genes)

# ================== SAVE BRAIN MAPS ==================
for c in sig_components:
    idx = c - 1
    region_scores = Xs[:, idx]
    comp_map = np.zeros(atlas_data.shape)
    for i, rid in enumerate(region_ids):
        comp_map[atlas_data == rid] = region_scores[i]
    out_img = nib.Nifti1Image(comp_map, atlas_img.affine)
    nib.save(out_img, f"{outdir}/PLS_component_{c}_map.nii.gz")

print("DONE! PLS stats, genes, and maps saved.")
