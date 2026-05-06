

import os
import sys
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh

# -------------------- CONFIG --------------------
INPUT_DIR = '/data/users3/jalaparthi1/SZ_HC'
OUTPUT_DIR = '/data/users3/jalaparthi1/SZ_HC/Similarity_Analysis'
ATLAS_PATH = '/data/users3/jalaparthi1/Merged_Atlas.nii.gz'

DIFFUSION_ALPHA = 0.01
NUM_COMPONENTS = 3
DIFFUSION_TIME = 1

# Numerical epsilon for stability
EPS = 1e-10
# ------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_similarity_matrix(data: pd.DataFrame, method: str) -> np.ndarray:
    """
    data: rows = regions / samples, columns = genes/features
    method: "euclidean" or "cosine"
    returns: similarity matrix (n_regions x n_regions) normalized to [0,1]
    """
    X = data.values
    if method == "euclidean":
        dist = cdist(X, X, metric='euclidean')
        # convert distance to similarity in (0,1]
        sim = 1.0 / (1.0 + dist)
        return sim
    elif method == "cosine":
        sim = cosine_similarity(X)
        # numerical clipping then convert cos similarity [-1,1] -> [0,1]
        sim = np.clip(sim, -1.0, 1.0)
        sim = 1.0 - np.arccos(sim) / np.pi
        return sim
    else:
        raise ValueError("Unknown similarity method: " + str(method))

def compute_degree_matrix(A: np.ndarray) -> np.ndarray:
    return np.diag(np.sum(A, axis=1))

def compute_laplacian_matrix(A: np.ndarray):
    D = compute_degree_matrix(A)
    L = D - A
    return L, D

def compute_diffusion_operator(A: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Anisotropic diffusion operator similar to BrainSpace implementation."""
    row_sums = np.sum(A, axis=1)
    # protect zeros
    d_alpha = np.power(row_sums + EPS, -alpha)
    D_alpha_inv = np.diag(d_alpha)
    W_alpha = D_alpha_inv @ A @ D_alpha_inv
    D_alpha = np.diag(np.sum(W_alpha, axis=1))
    D_alpha_inv2 = np.diag(1.0 / (np.diag(D_alpha) + EPS))
    P_alpha = D_alpha_inv2 @ W_alpha
    return P_alpha

def sign_align(eigvecs: np.ndarray) -> np.ndarray:
    """
    Align eigenvector signs so the element with maximum absolute value is positive.
    This mirrors BrainSpace sign correction.
    """
    if eigvecs.ndim != 2:
        return eigvecs
    max_inds = np.abs(eigvecs).argmax(axis=0)
    signs = np.sign(eigvecs[max_inds, range(eigvecs.shape[1])])
    signs[signs == 0] = 1.0
    return eigvecs * signs

def compute_eigenvectors_LE(A: np.ndarray, num_components: int = 3) -> np.ndarray:
    L, D = compute_laplacian_matrix(A)
    # generalized eigenproblem L v = lambda D v
    eigvals, eigvecs = eigh(L, D)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    # skip first trivial eigenvector
    vecs = eigvecs[:, 1:num_components + 1]
    vecs = sign_align(vecs)
    return vecs

def compute_eigenvectors_DM(P: np.ndarray, num_components: int = 3, t: int = 1) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P)
    eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
    idx = np.argsort(-eigvals)  # descending
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    # drop the first eigenvector (largest eigenvalue ~1)
    vecs = eigvecs[:, 1:num_components + 1]
    vals = eigvals[1:num_components + 1]
    # scale by diffusion time
    vecs = vecs * (vals ** t)[None, :]
    vecs = sign_align(vecs)
    return vecs

def save_csv(data: np.ndarray, path: str, index=None, columns=None):
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(path, index=(index is not None))

def map_to_atlas_and_save(eigenvectors: np.ndarray, atlas_data: np.ndarray,
                          atlas_affine: np.ndarray, region_ids: np.ndarray,
                          output_prefix: str):
    """
    eigenvectors: shape (n_regions, n_components)
    atlas_data: 3D label image
    region_ids: 1D array of region ids corresponding to rows of eigenvectors
    """
    n_regions, n_components = eigenvectors.shape
    unique_regions = np.unique(atlas_data)
    unique_regions = unique_regions[unique_regions != 0]

    # We will map using the provided region_ids -> atlas region ids.
    # region_ids must have length n_regions and contain values found in atlas_data.
    if len(region_ids) != n_regions:
        raise ValueError("region_ids length mismatch with eigenvectors rows")

    # Validate that provided region_ids exist in atlas
    missing = set(region_ids) - set(unique_regions)
    if missing:
        print(f"Warning: some region_ids are not present in atlas: {sorted(list(missing))}")
        # still proceed — those rows will map to nothing

    for comp in range(n_components):
        vec = eigenvectors[:, comp]
        # create brain map
        brain_map = np.zeros_like(atlas_data, dtype=np.float32)
        for val, rid in zip(vec, region_ids):
            brain_map[atlas_data == rid] = val

        mask = brain_map != 0
        if np.any(mask):
            vals = brain_map[mask]
            mn, mx = vals.min(), vals.max()
            if abs(mx - mn) < EPS:
                # constant map -> set to zeros
                brain_map[mask] = 0.0
            else:
                scaled = (vals - mn) / (mx - mn + EPS)
                brain_map[mask] = scaled

        out_img = nib.Nifti1Image(brain_map, affine=atlas_affine)
        out_path = f"{output_prefix}_Component{comp + 1}.nii.gz"
        nib.save(out_img, out_path)
        print(f"  Saved NIfTI: {out_path}")

def infer_region_ids_from_gene_index(gene_df: pd.DataFrame, atlas_regions: np.ndarray):
    """
    Attempt to infer mapping from gene_expression index to atlas region ids.
    Strategies tried in order:
      1. If index dtype is integer and set(index) subset of atlas_regions -> use that order.
      2. If index strings look numeric -> convert and check.
      3. Else assume order of rows corresponds to sorted atlas regions.
    Returns an array of region ids with length == n_rows
    """
    idx = gene_df.index
    unique_atlas = set(atlas_regions)
    # strategy 1: integer index matching atlas ids
    try:
        if np.issubdtype(idx.dtype, np.integer):
            idx_vals = np.array(idx, dtype=int)
            if set(idx_vals).issubset(unique_atlas):
                return idx_vals
    except Exception:
        pass

    # strategy 2: numeric-like strings
    try:
        idx_numeric = np.array([int(x) for x in idx])
        if set(idx_numeric).issubset(unique_atlas):
            return idx_numeric
    except Exception:
        pass

    # strategy 3: if number of rows equals number of non-zero atlas regions, map by sorted order
    sorted_regions = np.sort(list(unique_atlas))
    if len(sorted_regions) == gene_df.shape[0]:
        return sorted_regions

    # fallback: raise so user can check
    raise ValueError("Unable to infer mapping between gene-expression rows and atlas region ids. "
                     "Please ensure your gene-expression CSV rows either contain atlas region ids "
                     "or their row count matches the number of atlas regions.")

def process_sz_hc(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, atlas_path=ATLAS_PATH,
                  diffusion_alpha=DIFFUSION_ALPHA, num_components=NUM_COMPONENTS, diffusion_time=DIFFUSION_TIME):
    print("Loading atlas...")
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    atlas_affine = atlas_img.affine
    atlas_unique = np.unique(atlas_data)
    atlas_unique = atlas_unique[atlas_unique != 0]
    print(f"Atlas loaded: {atlas_path}  (nonzero regions: {len(atlas_unique)})")

    ensure_dir(output_dir)

    # Find gene expression file(s) in the input dir
    gene_expr_candidates = [f for f in os.listdir(input_dir) if f.endswith("_Filtered_Gene_Expression.csv")]
    if not gene_expr_candidates:
        print("No *_Filtered_Gene_Expression.csv files found in input directory:", input_dir)
        return

    # If multiple, process all (but you likely have single SZ_HC_Filtered_Gene_Expression.csv)
    for gfile in gene_expr_candidates:
        try:
            gpath = os.path.join(input_dir, gfile)
            print(f"\nProcessing file: {gpath}")
            gene_df = pd.read_csv(gpath, index_col=0)
            print(f"  Gene expression shape: {gene_df.shape}")

            # infer mapping from gene rows to atlas region ids
            try:
                region_ids = infer_region_ids_from_gene_index(gene_df, atlas_unique)
                print("  Inferred region mapping from gene-expression index.")
            except ValueError as e:
                print("  ERROR inferring mapping:", e)
                raise

            base_name = os.path.splitext(os.path.basename(gfile))[0]
            sample_output_dir = os.path.join(output_dir, base_name)
            ensure_dir(sample_output_dir)

            for sim_method in ["euclidean", "cosine"]:
                print(f"\n  --- {sim_method.upper()} ---")
                A = compute_similarity_matrix(gene_df, sim_method)
                # clamp to [0,1]
                A = np.clip(A, 0.0, 1.0)

                # Save affinity matrix
                aff_path = os.path.join(sample_output_dir, f"{base_name}_{sim_method}_Affinity.csv")
                save_csv(A, aff_path)
                print(f"  Saved affinity CSV: {aff_path}")

                # Laplacian Eigenmaps
                le_dir = os.path.join(sample_output_dir, "LE", sim_method.capitalize())
                ensure_dir(le_dir)
                le_prefix = os.path.join(le_dir, f"{base_name}_{sim_method}")
                le_evecs = compute_eigenvectors_LE(A, num_components=num_components)
                save_csv(le_evecs, f"{le_prefix}_LE_Eigenvectors.csv")
                print(f"  Saved LE eigenvectors CSV: {le_prefix}_LE_Eigenvectors.csv")
                map_to_atlas_and_save(le_evecs, atlas_data, atlas_affine, region_ids, f"{le_prefix}_LE_Mapped")

                # Diffusion Maps
                dm_dir = os.path.join(sample_output_dir, "DM", sim_method.capitalize())
                ensure_dir(dm_dir)
                dm_prefix = os.path.join(dm_dir, f"{base_name}_{sim_method}")
                P_alpha = compute_diffusion_operator(A, alpha=diffusion_alpha)
                save_csv(P_alpha, f"{dm_prefix}_DiffusionOperator.csv")
                print(f"  Saved diffusion operator CSV: {dm_prefix}_DiffusionOperator.csv")
                dm_evecs = compute_eigenvectors_DM(P_alpha, num_components=num_components, t=diffusion_time)
                save_csv(dm_evecs, f"{dm_prefix}_DM_Eigenvectors.csv")
                print(f"  Saved DM eigenvectors CSV: {dm_prefix}_DM_Eigenvectors.csv")
                map_to_atlas_and_save(dm_evecs, atlas_data, atlas_affine, region_ids, f"{dm_prefix}_DM_Mapped")

            print(f"\n  Completed processing for {gfile}. Outputs in: {sample_output_dir}")

        except Exception as exc:
            print(f"\nERROR processing {gfile}: {exc}")
            traceback.print_exc()
            # continue with next file rather than aborting entire run
            continue

    print("\nAll processing finished.")

if __name__ == "__main__":
    # Allow overriding paths via env vars if desired
    in_dir = os.environ.get('SZ_HC_INPUT_DIR', INPUT_DIR)
    out_dir = os.environ.get('SZ_HC_OUTPUT_DIR', OUTPUT_DIR)
    atlas = os.environ.get('SZ_HC_ATLAS', ATLAS_PATH)

    process_sz_hc(input_dir=in_dir, output_dir=out_dir, atlas_path=atlas,
                  diffusion_alpha=DIFFUSION_ALPHA, num_components=NUM_COMPONENTS, diffusion_time=DIFFUSION_TIME)
