import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ===========================================================
# PATHS
# ===========================================================
corr_path = "/data/users3/jalaparthi1/CBCL_ADHD65_CONTRAST_CORR/CBCL65_contrast_corr_measure.csv"
cbcl_path = "/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/mental-health/mh_p_cbcl.csv"
cog_path  = "/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/neurocognition/nc_y_nihtb.csv"
meta_path = "/data/users2/bbharatula1/ABCD-score/meta_data_cog_beh.csv"

out_dir = "/data/users3/jalaparthi1/CBCL65_GLM_ALLPHENOS_New/"
os.makedirs(out_dir, exist_ok=True)

# ===========================================================
# LOAD DATA
# ===========================================================
print("Loading correlation scores...")
corr = pd.read_csv(corr_path, low_memory=False)
print("Loading CBCL phenotypes...")
cbcl = pd.read_csv(cbcl_path, low_memory=False)
print("Loading cognitive phenotypes...")
cog  = pd.read_csv(cog_path, low_memory=False)
print("Loading metadata (age, sex)...")
meta = pd.read_csv(meta_path, low_memory=False)

print("\nData shapes:")
print("corr:", corr.shape)
print("cbcl:", cbcl.shape)
print("cog :", cog.shape)
print("meta:", meta.shape)

# ===========================================================
# FIX SUBJECT IDs
# ===========================================================
def fix_id(x):
    x = str(x)
    if x.startswith("NDAR") and not x.startswith("NDAR_"):
        return "NDAR_" + x[4:]
    return x

for df in [cbcl, cog, corr, meta]:
    if "src_subject_id" in df.columns:
        df["subjectkey"] = df["src_subject_id"].apply(fix_id)

# ===========================================================
# REMOVE DUPLICATES
# ===========================================================
cbcl = cbcl.drop_duplicates(subset=["subjectkey", "eventname"])
cog  = cog.drop_duplicates(subset=["subjectkey", "eventname"])
corr = corr.drop_duplicates(subset=["subjectkey"])

print("\nAfter removing duplicates:")
print("corr:", corr.shape)
print("cbcl:", cbcl.shape)
print("cog :", cog.shape)
print("meta:", meta.shape)

# ===========================================================
# BASELINE FILTER
# ===========================================================
cbcl_base = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]
cog_base  = cog[cog["eventname"]  == "baseline_year_1_arm_1"]

print("\nCBCL baseline:", cbcl_base.shape)
print("COG baseline :", cog_base.shape)

# ===========================================================
# MERGE CORR + PHENOTYPES
# ===========================================================
merged = corr.merge(cbcl_base, on="subjectkey", how="inner")
merged = merged.merge(cog_base, on="subjectkey", how="inner")
print("\nMerged shape (before meta):", merged.shape)

# ===========================================================
# MERGE META DATA (AGE & SEX)
# ===========================================================
meta_base = meta[["subjectkey", "interview_age", "demo_sex_v2"]]
merged = merged.drop(columns=[c for c in ["interview_age", "demo_sex_v2"] if c in merged.columns], errors='ignore')
merged = merged.merge(meta_base, on="subjectkey", how="left")

# Convert to numeric and check missing
merged["interview_age"] = pd.to_numeric(merged["interview_age"], errors="coerce")
merged["demo_sex_v2"] = pd.to_numeric(merged["demo_sex_v2"], errors="coerce")

if merged["interview_age"].isnull().sum() > 0 or merged["demo_sex_v2"].isnull().sum() > 0:
    print("Warning: Missing values in 'interview_age' or 'demo_sex_v2'")

# ===========================================================
# SPECIFIC PHENOTYPE VARIABLES
# ===========================================================
phenotypes_to_run = [
    # CBCL
    "cbcl_scr_syn_anxdep_t",
    "cbcl_scr_07_ocd_t",
    "cbcl_scr_07_sct_t",
    "cbcl_scr_07_stress_t",
    "cbcl_scr_syn_social_t",
    "cbcl_scr_syn_thought_t",
    "cbcl_scr_syn_attention_t",
    "cbcl_scr_syn_rulebreak_t",
    "cbcl_scr_syn_aggressive_t",
    "cbcl_scr_syn_somatic_t",
    "cbcl_scr_syn_internal_t",
    "cbcl_scr_syn_external_t",
    "cbcl_scr_syn_totprob_t",
    # COG
    "nihtbx_cardsort_agecorrected",
    "nihtbx_cryst_agecorrected",
    "nihtbx_flanker_agecorrected",
    "nihtbx_list_agecorrected",
    "nihtbx_pattern_agecorrected",
    "nihtbx_picture_agecorrected",
    "nihtbx_picvocab_agecorrected",
    "nihtbx_reading_agecorrected",
    "nihtbx_fluidcomp_agecorrected",
    "nihtbx_totalcomp_agecorrected",
]

# ===========================================================
# COVARIATES
# ===========================================================
corr_var = "CBCL65_contrast_score"  # predictor
covariates = [corr_var, "interview_age", "demo_sex_v2"]

# ===========================================================
# RUN GLM
# ===========================================================
results = []

for pheno in phenotypes_to_run:
    if pheno not in merged.columns:
        print(f"Skipping {pheno} (not in dataframe)")
        continue

    glm_df = merged[covariates + [pheno]].copy()
    glm_df = glm_df.apply(pd.to_numeric, errors="coerce").dropna()

    # Skip small N or constant phenotype
    if glm_df.shape[0] < 20:
        print(f"Skipping {pheno} (N too small: {glm_df.shape[0]})")
        continue
    if glm_df[pheno].var() == 0:
        print(f"Skipping {pheno} (constant phenotype)")
        continue

    y = glm_df[pheno]
    X = sm.add_constant(glm_df[covariates])

    try:
        model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    except Exception as e:
        print(f"Skipping {pheno} due to GLM error: {e}")
        continue

    # Calculate R-squared manually
    y_mean = y.mean()
    ss_total = ((y - y_mean)**2).sum()
    ss_res = ((y - model.fittedvalues)**2).sum()
    rsq = 1 - ss_res/ss_total

    # Save individual summary
    with open(os.path.join(out_dir, f"GLM_{pheno}.txt"), "w") as f:
        f.write(model.summary().as_text())

    # Append results
    results.append([
        pheno,
        model.params.get(corr_var, np.nan),
        model.pvalues.get(corr_var, np.nan),
        rsq,
        glm_df.shape[0]
    ])

# ===========================================================
# SAVE RESULTS
# ===========================================================
res_df = pd.DataFrame(results, columns=["phenotype", "beta_corr", "p_corr", "rsq", "n_samples"])
res_df.to_csv(os.path.join(out_dir, "ALL_GLM_RESULTS.csv"), index=False)

# ===========================================================
# SAVE MERGED DATAFRAME
# ===========================================================
merged.to_csv(os.path.join(out_dir, "merged_dataframe.csv"), index=False)

print("\n==============================================")
print("CBCL65 GLM pipeline complete")
print(f"Results & merged dataframe saved → {out_dir}")
print("==============================================")
