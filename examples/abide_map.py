# abide_map.py
# ABIDE Parcellation
# Output: (150 timepoints × 414 ROIs) fMRI + standardized phenotype

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import datasets
from gt_map import GlasserTianParcellator, create_analysis_phenotype
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("dataset/nilearn_data")
OUTPUT_DIR = Path("dataset/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ABIDE I TR mapping (official)
ABIDE_I_TR = {
    'Caltech': 2.0, 'CMU': 2.0, 'KKI': 1.5, 'MaxMun': 3.0,
    'NYU': 2.0, 'Olin': 1.5, 'SBL': 2.2, 'SDSU': 2.0,
    'Stanford': 2.0, 'Trinity': 2.0, 'UCLA': 3.0, 'UM': 2.0,
    'USM': 2.0, 'PITT': 1.5, 'Yale': 2.0, 'Utah': 2.0
}

TARGET_TR = 2.0
TARGET_DURATION = 300.0  # 5 minutes → 150 timepoints


# =============================================================================
# HELPER: Extract TRs from ABIDE phenotype
# =============================================================================
def prepare_tr_values(phenotypic_df):
    """Robustly extract TR using ABIDE site mapping."""
    site_col = None
    for col in phenotypic_df.columns:
        clean = col.upper().replace('_', '').replace('ID', '')
        if clean in ['SITE', 'SITEID']:
            site_col = col
            break
    if site_col is None:
        for col in phenotypic_df.columns:
            if 'site' in col.lower():
                site_col = col
                break
    if site_col is None:
        raise ValueError(f"SITE_ID not found. Columns: {list(phenotypic_df.columns)}")

    tr_values = phenotypic_df[site_col].map(ABIDE_I_TR)
    unmapped = tr_values.isna()
    if unmapped.any():
        logger.warning(f"⚠️ {unmapped.sum()} subjects have unmapped sites → using TR=2.0")
        tr_values = tr_values.fillna(2.0)
    if tr_values.isna().all():
        raise ValueError("All TR values are NaN!")

    return tr_values.astype(np.float32).values


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    logger.info("📥 Loading ABIDE data...")
    abide = datasets.fetch_abide_pcp(
        data_dir=DATA_DIR,
        pipeline="cpac",
        derivatives=["func_preproc"],
        quality_checked=True,
        verbose=0
    )

    logger.info("🔍 Preparing TR values...")
    tr_values = prepare_tr_values(abide.phenotypic)

    # ✅ Use gt_map with BUNDLED ATLASES (no path needed!)
    logger.info("🚀 Parcellating with Glasser+Tian atlases (bundled)...")
    parcellator = GlasserTianParcellator()  # ← No atlas_dir required!

    processed_data, valid_indices = parcellator.process_dataset(
        fmri_paths=abide.func_preproc,
        tr_values=tr_values,
        n_jobs=-1,
        target_tr=TARGET_TR,
        target_duration=TARGET_DURATION
    )

    # Filter phenotype to valid subjects
    valid_phenotype = abide.phenotypic.iloc[valid_indices].reset_index(drop=True)

    # --- Save fMRI: (N, 150, 414) ---
    data_array = np.stack(processed_data, axis=0).astype(np.float32)
    subject_ids = valid_phenotype['SUB_ID'].astype(str).values
    max_len = max(len(sid) for sid in subject_ids)
    subject_ids_safe = np.array(subject_ids, dtype=f'U{max_len}')

    np.savez_compressed(
        OUTPUT_DIR / "fmri_ASD.npz",
        data=data_array,
        subject_ids=subject_ids_safe
    )

    # --- Save standardized phenotype using gt_map helper ---
    pheno_analysis = create_analysis_phenotype(
        valid_phenotype,
        eid_col='SUB_ID',
        age_col='AGE_AT_SCAN',
        sex_col='SEX',
        target_col='DX_GROUP',
        sex_male_value=1,
        target_positive_value=1
    )
    # Rename target column to match your pipeline
    pheno_analysis = pheno_analysis.rename(columns={'Target': 'ASD'})
    pheno_analysis.to_csv(OUTPUT_DIR / "pheno_ASD.csv", index=False)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ABIDE Parcellation Complete!")
    logger.info(f"Processed: {len(processed_data)} / {len(abide.func_preproc)} subjects")
    logger.info(f"Saved fMRI: {OUTPUT_DIR / 'fmri_ASD.npz'}")
    logger.info(f"Saved phenotype: {OUTPUT_DIR / 'pheno_ASD.csv'}")
    logger.info(f"Shape per subject: ({int(TARGET_DURATION / TARGET_TR)}, 414)")
    logger.info(f"Duration: {TARGET_DURATION}s | TR: {TARGET_TR}s")
    logger.info("=" * 60)

    print("\nSample phenotype:")
    print(pheno_analysis.head(3))
    print(f"\nClass balance (ASD=1): {pheno_analysis['ASD'].value_counts().to_dict()}")


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
