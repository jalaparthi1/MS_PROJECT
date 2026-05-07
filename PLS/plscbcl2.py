import numpy as np
import pandas as pd
import nibabel as nib
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests

# ================== PATHS ==================
gene_path = "/data/users3/jalaparthi1/Merged_Atlas_Expression_output.csv"
contrast_path = "/data/users3/jalaparthi1/CBCL_ADHD65_CONTRAST_CORR/CBCL_ADHD65_minus_Control_Contrast_Template.nii"
mask_path = "/data/users2/bbharatula1/ABCD-score/mask.nii"
atlas_path = "/data/users3/jalaparthi1/Merged_Atlas.nii.gz"

outdir = "pls_results_with_perm"
os.makedirs(outdir, exist_ok=True)

# ================== PARAMETERS ==================
n_components = 21
n_perm = 1000
alpha = 0.05
z_thresh = 2.5
rng = np.random.default_rng(42)

# ================== LOAD GENE EXPRESSION ==================
gene_df = pd.read_csv(gene_path)
numeric_cols = gene_df.select_dtypes(include=[np.number]).columns
X = gene_df[numeric_cols].values

# ================== LOAD IMAGES ==================
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

contrast_data = nib.load(contrast_path).get_fdata()
mask_data = nib.load(mask_path).get_fdata()

# ================== BUILD REGIONAL Y ==================
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

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ================== Z-SCORE ==================
Xz = StandardScaler().fit_transform(X)
Yz = StandardScaler().fit_transform(Y)

# ================== OBSERVED PLS ==================
pls = PLSRegression(n_components=n_components)
pls.fit(Xz, Yz)

Xs = pls.x_scores_
Ys = pls.y_scores_
Xw = pls.x_weights_

obs_r = np.array([pearsonr(Xs[:, i], Ys[:, i])[0] for i in range(n_components)])
true_r2 = r2_score(Yz, pls.predict(Xz))

# ================== PERMUTATION TEST ==================
perm_r = np.zeros((n_perm, n_components))
perm_r2 = np.zeros(n_perm)

for p in range(n_perm):
    Y_perm = rng.permutation(Yz[:, 0]).reshape(-1, 1)

    pls_perm = PLSRegression(n_components=n_components)
    pls_perm.fit(Xz, Y_perm)

    Xs_p = pls_perm.x_scores_
    Ys_p = pls_perm.y_scores_

    perm_r2[p] = r2_score(Y_perm, pls_perm.predict(Xz))
    for i in range(n_components):
        perm_r[p, i] = pearsonr(Xs_p[:, i], Ys_p[:, i])[0]

# ================== EMPIRICAL P-VALUES ==================
p_r2 = (1 + np.sum(perm_r2 >= true_r2)) / (1 + n_perm)
p_corrs = [(1 + np.sum(np.abs(perm_r[:, j]) >= np.abs(obs_r[j]))) / (1 + n_perm)
           for j in range(n_components)]

# ================== BONFERRONI ==================
rej, p_bonf, _, _ = multipletests(p_corrs, alpha=alpha, method="bonferroni")

# ================== SAVE STATS ==================
stats_df = pd.DataFrame({
    "Component": np.arange(1, n_components + 1),
    "r_obs": obs_r,
    "p_corrs": p_corrs,
    "p_bonf": p_bonf,
    "Significant_Bonferroni": ["Yes" if x else "No" for x in rej]
})
stats_df.to_csv(f"{outdir}/PLS_component_stats_with_perm.csv", index=False)

# Save overall model stats
overall_stats = {
    "true_r2": float(true_r2),
    "p_r2": float(p_r2),
    "true_corrs": obs_r.tolist(),
    "p_corrs": p_corrs
}
pd.Series(overall_stats).to_csv(f"{outdir}/PLS_overall_stats.csv")

print(stats_df)
print("Significant components:", stats_df.loc[stats_df["Significant_Bonferroni"]=="Yes","Component"].tolist())

# ================== IMPORTANT GENES ==================
important_genes = {}
sig_components = stats_df.loc[stats_df["Significant_Bonferroni"]=="Yes","Component"].tolist()

for c in sig_components:
    idx = c - 1
    w = Xw[:, idx]
    z = (w - w.mean()) / w.std()
    sel = np.where(np.abs(z) >= z_thresh)[0]
    genes = numeric_cols[sel]
    important_genes[c] = genes
    pd.Series(genes).to_csv(f"{outdir}/important_genes_component_{c}.csv", index=False)

# ================== COMMON GENES ==================
if len(important_genes) > 1:
    common_genes = set.intersection(*map(set, important_genes.values()))
else:
    common_genes = set()

pd.Series(list(common_genes)).to_csv(f"{outdir}/common_genes.csv", index=False)

# ================== SAVE BRAIN MAPS ==================
for c in sig_components:
    idx = c - 1
    region_scores = Xs[:, idx]

    comp_map = np.zeros(atlas_data.shape)
    for i, rid in enumerate(region_ids):
        comp_map[atlas_data == rid] = region_scores[i]

    nib.save(nib.Nifti1Image(comp_map, atlas_img.affine),
             f"{outdir}/PLS_component_{c}_map.nii.gz")

print("DONE (PLS + permutations + Bonferroni + R² saved).")
