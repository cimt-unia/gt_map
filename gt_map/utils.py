# gt_map/utils.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

# Default standardization targets
DEFAULT_TARGET_TR = 2.0
DEFAULT_TARGET_DURATION = 300.0  # 5 minutes → 150 timepoints


def resample_timeseries(ts, original_tr, target_tr=DEFAULT_TARGET_TR,
                        target_duration=DEFAULT_TARGET_DURATION, kind='cubic'):
    if ts.ndim != 2:
        raise ValueError(f"Input must be 2D (T, n_rois), got shape {ts.shape}")
    
    original_time = np.arange(ts.shape[0]) * original_tr
    if original_time[-1] < target_duration:
        return None

    target_n = int(target_duration / target_tr)
    target_time = np.arange(target_n) * target_tr
    resampled = np.zeros((target_n, ts.shape[1]), dtype=np.float32)

    for roi_idx in range(ts.shape[1]):
        interp_func = interp1d(
            original_time, ts[:, roi_idx],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True
        )
        resampled[:, roi_idx] = interp_func(target_time)

    return resampled


def create_analysis_phenotype(
    phenotype_df: pd.DataFrame,
    eid_col: str = 'SUB_ID',
    age_col: str = 'AGE_AT_SCAN',
    sex_col: str = 'SEX',
    target_col: str = 'DX_GROUP',
    sex_male_value=1,
    target_positive_value=1
):
    return pd.DataFrame({
        'eid': phenotype_df[eid_col],
        'Age': phenotype_df[age_col],
        'Sex': (phenotype_df[sex_col] == sex_male_value).astype(int),
        'Target': (phenotype_df[target_col] == target_positive_value).astype(int)
    })
