# adhd200_map.py
# ADHD-200 Parcellation using gt_map
# Output: (150 timepoints × 414 ROIs) fMRI + standardized run-level phenotype


import numpy as np
import pandas as pd
from pathlib import Path
from gt_map import GlasserTianParcellator
from nilearn import image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_ROOT = Path('/nilearn_data/ADHD200')
PHENO_PATH = DATA_ROOT / "adhd200.csv"
OUTPUT_DIR = DATA_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADHD_SITE_TR = {
    1: 2.0,   # Peking
    3: 2.0,   # KKI
    4: 2.0,   # NeuroIMAGE
    5: 2.0,   # NYU
    6: 2.5,   # OHSU
}

TARGET_TR = 2.0
TARGET_DURATION = 300.0  # → 150 timepoints


# =============================================================================
# HELPER: Safely get number of timepoints
# =============================================================================
def get_n_timepoints(img_path):
    """Return number of timepoints; 1 if 3D, T if 4D, 0 if invalid."""
    img = image.load_img(img_path)
    if len(img.shape) == 3:
        return 1
    elif len(img.shape) == 4:
        return img.shape[3]
    else:
        return 0


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    logger.info("Loading ADHD-200 phenotype...")
    df = pd.read_csv(PHENO_PATH)
    logger.info(f"Starting with {len(df)} fMRI runs (all subjects × all runs).")

    # Rebuild fMRI paths
    def get_fMRI_path(row):
        sid = str(row["ScanDir ID"])
        run = int(row["run"])
        filename = f"{sid}.nii.gz" if run == 1 else f"{sid}_run{run}.nii.gz"
        full_path = DATA_ROOT / filename
        return str(full_path) if full_path.exists() else None

    df["fMRI_path"] = df.apply(get_fMRI_path, axis=1)
    initial_n = len(df)
    df = df.dropna(subset=["fMRI_path"]).reset_index(drop=True)
    logger.info(f"Kept {len(df)} runs with existing fMRI files (dropped {initial_n - len(df)}).")

    # Pre-screen for valid 4D fMRI with ≥20 timepoints
    logger.info("Pre-screening runs for valid fMRI with ≥20 timepoints...")
    valid_mask = []
    fmri_paths_filtered = []
    tr_values_filtered = []

    for idx, row in df.iterrows():
        fp = row["fMRI_path"]
        tr = ADHD_SITE_TR.get(row["Site"], 2.0)
        try:
            n_tps = get_n_timepoints(fp)
            if n_tps >= 20:
                valid_mask.append(True)
                fmri_paths_filtered.append(fp)
                tr_values_filtered.append(tr)
            else:
                logger.warning(f"⚠️ Skipping {Path(fp).name}: only {n_tps} timepoints")
                valid_mask.append(False)
        except Exception as e:
            logger.warning(f"⚠️ Skipping {Path(fp).name}: load error – {e}")
            valid_mask.append(False)

    df = df[valid_mask].reset_index(drop=True)
    fmri_paths = fmri_paths_filtered
    tr_values = np.array(tr_values_filtered, dtype=np.float32)
    logger.info(f"Proceeding with {len(fmri_paths)} valid fMRI runs.")

    # Use gt_map with BUNDLED ATLASES — no ATLAS_DIR needed!
    logger.info("Parcellating with Glasser+Tian atlases (bundled)...")
    parcellator = GlasserTianParcellator()  # ← Clean, no path required

    processed_data, valid_indices = parcellator.process_dataset(
        fmri_paths=fmri_paths,
        tr_values=tr_values,
        n_jobs=-1,
        target_tr=TARGET_TR,
        target_duration=TARGET_DURATION
    )

    valid_df = df.iloc[valid_indices].reset_index(drop=True)

    # Create run-level IDs
    def make_run_id(row):
        sid = str(row["ScanDir ID"])
        run = int(row["run"])
        return f"{sid}_run{run}" if run != 1 else sid

    valid_df["run_id"] = valid_df.apply(make_run_id, axis=1)

    # Save outputs
    data_array = np.stack(processed_data, axis=0).astype(np.float32)
    run_ids = valid_df["run_id"].astype(str).values
    max_len = max(len(rid) for rid in run_ids)
    run_ids_safe = np.array(run_ids, dtype=f'U{max_len}')

    np.savez_compressed(
        OUTPUT_DIR / "fmri_ADHD_all_runs.npz",
        data=data_array,
        subject_ids=run_ids_safe
    )

    # Build phenotype
    pheno_adhd = pd.DataFrame({
        'eid': valid_df["run_id"].astype(str),
        'subject_id': valid_df["ScanDir ID"].astype(str),
        'run': valid_df["run"].astype(int),
        'Age': valid_df["Age"],
        'Sex': valid_df["Gender"],
        'ADHD': (valid_df["DX"] != 0).astype(int)
    })
    pheno_adhd.to_csv(OUTPUT_DIR / "pheno_ADHD_all_runs.csv", index=False)

    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("ADHD-200 Parcellation (All Runs) Complete!")
    logger.info(f"Processed: {len(processed_data)} fMRI runs")
    logger.info(f"Output shape per run: (150, 414)")
    logger.info("=" * 60)

    print("\nSample phenotype (run-level):")
    print(pheno_adhd.head(3))
    print(f"\nClass balance (ADHD=1): {pheno_adhd['ADHD'].value_counts().to_dict()}")
    print(f"\nTotal unique subjects: {pheno_adhd['subject_id'].nunique()}")


if __name__ == "__main__":
    main()
