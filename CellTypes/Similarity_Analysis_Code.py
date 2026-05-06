import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh

# === PATHS ===
base_dir = '/data/users3/jalaparthi1'
atlas_path = os.path.join(base_dir, 'Merged_Atlas.nii.gz')

# === PARAMETERS ===
diffusion_alpha = 0.01
num_components = 3
diffusion_time = 1

# === FUNCTIONS ===
def compute_similarity_matrix(data, method):
    if method == "euclidean":
        dist = cdist(data, data, metric='euclidean')
        return 1 / (1 + dist)
    elif method == "cosine":
        return cosine_similarity(data)
    else:
        raise ValueError("Invalid similarity method")

def compute_degree_matrix(A):
    return np.diag(np.sum(A, axis=1))

def compute_laplacian_matrix(A):
    D = compute_degree_matrix(A)
    L = D - A
    return L, D

def compute_diffusion_operator(A, alpha=0.01):
    d_alpha = np.power(np.sum(A, axis=1), -alpha)
    D_alpha_inv = np.diag(d_alpha)
    W_alpha = D_alpha_inv @ A @ D_alpha_inv
    D_alpha = np.diag(np.sum(W_alpha, axis=1))
    D_alpha_inv2 = np.diag(1.0 / (np.diag(D_alpha) + 1e-10))
    return D_alpha_inv2 @ W_alpha

def compute_eigenvectors_LE(A, num_components=3):
    L, D = compute_laplacian_matrix(A)
    eigvals, eigvecs = eigh(L, D)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, 1:num_components + 1]

def compute_eigenvectors_DM(P_alpha, num_components=3, t=1):
    eigvals, eigvecs = np.linalg.eig(P_alpha)
    eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
    idx = np.argsort(-eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvecs[:, 1:num_components + 1] * (eigvals[1:num_components + 1] ** t)

def save_csv(data, path):
    pd.DataFrame(data).to_csv(path, index=False)

def map_to_atlas_and_save(eigenvectors, atlas_data, atlas_affine, output_path_prefix):
    unique_regions = np.unique(atlas_data)
    unique_regions = unique_regions[unique_regions != 0]
    sorted_regions = np.sort(unique_regions)

    if len(sorted_regions) != eigenvectors.shape[0]:
        raise ValueError(f"Mismatch: {len(sorted_regions)} atlas regions vs {eigenvectors.shape[0]} eigenvectors")

    for i in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, i]
        brain_map = np.zeros_like(atlas_data, dtype=np.float32)

        for val, region in zip(vector, sorted_regions):
            brain_map[atlas_data == region] = val

        mask = brain_map != 0
        values = brain_map[mask]
        if np.any(values):
            scaled = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
            brain_map[mask] = scaled

        img = nib.Nifti1Image(brain_map, affine=atlas_affine)
        nib.save(img, f"{output_path_prefix}_Component{i + 1}.nii.gz")

# === LOAD ATLAS ONCE ===
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
atlas_affine = atlas_img.affine

# === MAIN EXECUTION FOR MULTIPLE SUPERCLUSTERS ===
for sc_id in range(5, 32):  # SC5 → SC31
    print(f"\n================= SUPERCLUSTER {sc_id} =================")

    input_dir = os.path.join(base_dir, f'supercluster{sc_id}')
    output_dir = os.path.join(input_dir, 'Similarity_Analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Load gene expression for this SC
    gene_expr_file = os.path.join(input_dir, f'supercluster{sc_id}_Filtered_Gene_Expression.csv')
    if not os.path.exists(gene_expr_file):
        print(f"File missing: {gene_expr_file}, skipping.")
        continue
    gene_expression_data = pd.read_csv(gene_expr_file, index_col=0)
    print(f"Gene expression shape for SC{sc_id}: {gene_expression_data.shape}")

    for sim_method in ["euclidean", "cosine"]:
        print(f"--- Processing {sim_method.upper()} ---")

        A = compute_similarity_matrix(gene_expression_data, sim_method)

        # --- LE ---
        le_dir = os.path.join(output_dir, "LE", sim_method.capitalize())
        os.makedirs(le_dir, exist_ok=True)

        save_csv(A, os.path.join(le_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_Affinity.csv'))
        le_eigenvectors = compute_eigenvectors_LE(A, num_components=num_components)
        save_csv(le_eigenvectors, os.path.join(le_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_LE_Eigenvectors.csv'))
        map_to_atlas_and_save(le_eigenvectors, atlas_data, atlas_affine,
                              os.path.join(le_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_LE_Mapped'))

        # --- DM ---
        dm_dir = os.path.join(output_dir, "DM", sim_method.capitalize())
        os.makedirs(dm_dir, exist_ok=True)

        P_alpha = compute_diffusion_operator(A, alpha=diffusion_alpha)
        save_csv(P_alpha, os.path.join(dm_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_DiffusionOperator.csv'))
        dm_eigenvectors = compute_eigenvectors_DM(P_alpha, num_components=num_components, t=diffusion_time)
        save_csv(dm_eigenvectors, os.path.join(dm_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_DM_Eigenvectors.csv'))
        map_to_atlas_and_save(dm_eigenvectors, atlas_data, atlas_affine,
                              os.path.join(dm_dir, f'Supercluster{sc_id}_{sim_method.capitalize()}_DM_Mapped'))

