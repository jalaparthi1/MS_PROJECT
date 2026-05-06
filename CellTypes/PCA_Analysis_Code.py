import pandas as pd
import nibabel as nib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------- PATHS ----------------------
base_dir = "/data/users3/jalaparthi1"
expression_file = os.path.join(base_dir, "Merged_Atlas_Expression_output.csv") 
atlas_path = os.path.join(base_dir, "Merged_Atlas.nii.gz")

# ---------------------- SUPERCLUSTER GENE LISTS ----------------------
supercluster_genes = {
    

    "supercluster15": [
        "NPY", "CORT", "SST", "NOS1", "TLL2", "CRHBP", "TENT5A", "TACR1", "LHX6", "FBN2",
        "PDGFD", "SNTG2", "SOX6", "CDCA7", "IGFBP4", "CRABP1", "KLB", "HTR3A", "PLSCR5", 
        "HAPLN1", "EXOC1L", "IL1RAPL2", "AGTR1", "NDNF", "TP73", "AIM2", "PRPH", "SCUBE1", 
        "GNG8", "GPR151", "CHRNB3", "POU4F1", "SLC5A7", "TRPM8", "ACOXL", "F13A1", "TES", 
        "HOXD3", "PDLIM1", "NKX6-1", "FAM83B", "ASAH2", "EBF3", "SIM1", "ONECUT1", "FOXA2", 
        "LHX1", "FSHR", "EBF1", "PPP1R1C", "HOXB3", "IGFBP7", "SLC6A5", "GLRA1", "LGI2", 
        "EN1", "ADGRD2", "PTH2", "LHX9", "MAB21L2", "PRDM6", "DLK1", "C1QL1", "ONECUT3", 
        "EBF2", "DRGX", "CNGA3", "LTBP2", "DIAPH3", "LMX1B", "MAFA", "HOXB7", "PHOX2B", 
        "TLX3", "SKOR2", "NHLH1", "TAC3", "PAX2", "NPPC", "PAX8", "GBX1", "GPC3", "SALL3", 
        "HOXB2", "HOXA3", "GBX2", "TFAP2A", "CRABP1", "NDNF", "PAX5", "EMX2", "GATA3", 
        "PVALB", "HAS2", "FABP4", "HPGD", "ST8SIA6", "SAMD11", "SIX3", "GATA2", "PAX7", 
        "LMO1", "TFAP2B", "EMX2", "LEF1", "PAX3", "ADAMTS20", "IGFL1", "OTX2", "IRX2", 
        "DMBX1", "COL6A2", "IRX3", "FST", "KCNJ16", "OTP", "S100A10", "ZPLD1", "ECEL1", 
        "TRPC7", "EOMES", "GLP1R", "CPA6", "TFAP2D", "HNF4G", "TMEM176B", "WDR72", "KLHL1", 
        "IGDCC4", "SHOX2", "SLC17A6", "NCAPG", "TAC1", "FAM9B", "BARHL1", "BNC2", "NTS", 
        "PGM5", "OTOS", "SHISAL2B", "TAGLN2", "CALB2", "NPB", "LMX1A", "CALCB", "CALCA", 
        "CBLN1", "GPR151", "MMRN1", "LRRC55", "TRH", "SIM1", "PITX2", "AGTR1", "ARHGAP36", 
        "C14orf39", "TFPI", "CALCR", "ISL1", "COL15A1", "GLI2", "GPR101", "HEPACAM2", 
        "NPBWR1", "PENK", "FST", "RSPO3", "KCNJ5", "DMRTA2", "LHX5", "FOXA1", "CDH23", 
        "PANCR", "GIPR", "CARTPT", "IRX3", "CRNDE", "AR", "TMEM255A", "GABRE", "IRX6", 
        "TACR3", "PGR", "HOXA3", "F13A1", "SCTR", "VSX2", "APELA", "ADCYAP1", "NMB", 
        "TACR3", "LMO1", "SST", "PAX6", "EGFL6", "ABI3BP", "NFIA", "LHX8", "FGF10", "STAC", 
        "DIAPH3", "ELFN1", "ISL1", "INHBB", "NPS", "IRX4", "TH", "SLC6A3", "SLC18A2", 
        "SLC18A1", "ASB4", "FEV", "TPH2", "SLC17A8", "RLN3", "SLC6A4", "DDC", "AMIGO2", 
        "PHOX2B", "SLC6A2", "GAL", "PHOX2A", "DBH", "RP1", "CHAT", "NTRK1", "OR8A1", "CNBD1", 
        "LMNTD1", "P2RX2", "SLC10A4", "FRMD7", "TRDN", "KANK3", "CACNG5", "SP8", "ROR2", 
        "SCGN", "CER1", "TAC3", "TRH", "NANOS1", "NPR3", "CHRNA3", "AVP", "OXT", "HCRT", 
        "PGF", "PPP1R17", "PRDM13", "RERGL", "RADX", "PIK3C2G", "SLITRK6", "PMCH", "TBX3", 
        "GAL", "HMX3", "SIX6", "DIRAS3", "ZCCHC12", "HMX2", "GHRH", "FEZF1", "DGKK", "PRLR", 
        "SYTL4", "ANKRD34B", "GABRQ", "IGSF1", "ESR1", "TMEM114", "STK26", "CYP19A1", 
        "SCN5A", "SLC7A3", "CITED1", "FREM2", "ZIC2", "PRDM12", "PRDM16", "ZIC5", "TBX3", 
        "POMC", "NR5A2", "HDC", "TECRL", "QRFPR", "GMNC"
    ]
}

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv(expression_file)
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in expression CSV.")

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
affine = atlas_img.affine

# ---------------------- PROCESS EACH SUPERCLUSTER ----------------------
for sc_name, genes_of_interest in supercluster_genes.items():
    print(f"\nProcessing {sc_name}...")
    
    # Create output directory
    output_dir = os.path.join(base_dir, f"{sc_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter available genes
    available_genes = [g for g in genes_of_interest if g in df.columns]
    missing_genes = [g for g in genes_of_interest if g not in df.columns]
    
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in {sc_name}")
        if len(missing_genes) > 10:
            print(f"First 10 missing: {missing_genes[:10]}")
        else:
            print(f"Missing genes: {missing_genes}")
    
    if len(available_genes) < 2:
        print(f"Skipping {sc_name} - not enough available genes ({len(available_genes)})")
        continue
    
    # Filter dataset
    filtered_df = df[['label'] + available_genes]
    filtered_output_path = os.path.join(output_dir, f"{sc_name}_Filtered_Gene_Expression.csv")
    filtered_df.to_csv(filtered_output_path, index=False)
    
    # PCA (use min(3, n_genes, n_samples))
    n_components = min(3, len(available_genes), len(df))
    print(f"Running PCA with {n_components} components for {sc_name}")
    
    region_ids = df["label"].values
    expression_data = df[available_genes]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression_data)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save region PCA scores
    region_scores = pd.DataFrame(
        X_pca,
        index=region_ids, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    region_scores.index.name = "RegionID"
    
    # Save gene loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=expression_data.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    loadings.index.name = "Gene"
    
    # Flip sign for PCs 1–3 if needed
    for pc in range(1, n_components+1):
        pc_col = f"PC{pc}"
        scores = region_scores[pc_col]
        max_pos = scores[scores > 0].max() if any(scores > 0) else 0
        max_neg = scores[scores < 0].min() if any(scores < 0) else 0
        if abs(max_neg) > abs(max_pos):
            region_scores[pc_col] = -region_scores[pc_col]
            loadings[pc_col] = -loadings[pc_col]
            print(f"Flipped sign of {pc_col} for {sc_name}")
    
    # Save results
    region_scores.to_csv(os.path.join(output_dir, f"{sc_name}_Region_PC_Matrix.csv"))
    loadings.to_csv(os.path.join(output_dir, f"{sc_name}_Gene_PC_Matrix.csv"))
    
    explained_variance = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_,
        "CumulativeExplainedVariance": np.cumsum(pca.explained_variance_ratio_)
    })
    explained_variance.to_csv(os.path.join(output_dir, f"{sc_name}_PCA_Variance.csv"), index=False)
    
    # Map PCs to NIfTI
    for pc in range(1, n_components+1):
        pc_col = f"PC{pc}"
        scores = region_scores[pc_col].values
        pc_volume = np.zeros_like(atlas_data)
        for region_idx, score in zip(region_ids, scores):
            pc_volume[atlas_data == region_idx] = score
        pc_nifti = nib.Nifti1Image(pc_volume, affine)
        nib.save(pc_nifti, os.path.join(output_dir, f"{sc_name}_PC{pc}_map.nii.gz"))
    
    # Save gene contributors
    contrib_dir = os.path.join(output_dir, f"{sc_name}_PC_Contributors")
    os.makedirs(contrib_dir, exist_ok=True)
    
    for pc in range(1, n_components+1):
        pc_col = f"PC{pc}"
        sorted_genes = loadings[[pc_col]].assign(abs_weight=loadings[pc_col].abs())
        sorted_genes = sorted_genes.sort_values("abs_weight", ascending=False).drop(columns="abs_weight")
        sorted_genes.to_csv(os.path.join(contrib_dir, f"all_genes_sorted_by_abs_weight_{pc_col}.csv"))
    
    print(f"Completed processing {sc_name}")

print("\nAll superclusters processed successfully!")
