# gt_map/core.py
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from joblib import Parallel, delayed
from .utils import resample_timeseries, DEFAULT_TARGET_TR, DEFAULT_TARGET_DURATION
import logging

logger = logging.getLogger(__name__)


class GlasserTianParcellator:
    """Efficient parcellation using Glasser (360) + Tian (54) atlases → 414 ROIs."""

    def __init__(self, atlas_dir=None):
        """
        Initialize the parcellator.
        
        Parameters
        ----------
        atlas_dir : str or Path, optional
            Path to directory containing:
                - glasser_360_MNI152NLin6Asym.nii.gz
                - tian_subcortex_54_MNI152NLin6Asym.nii
                - roi_labels.csv
            If None, uses the bundled atlas data included in the package.
        """
        if atlas_dir is None:
            # Lazy import to avoid circular dependency
            from . import get_bundled_atlas_dir
            atlas_dir = get_bundled_atlas_dir()
        
        self.atlas_dir = Path(atlas_dir)
        self.glasser_nii = self.atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz"
        self.tian_nii = self.atlas_dir / "tian_subcortex_54_MNI152NLin6Asym.nii"
        self.roi_labels_csv = self.atlas_dir / "roi_labels.csv"
        self._validate_files()

    def _validate_files(self):
        missing = [f for f in [self.glasser_nii, self.tian_nii, self.roi_labels_csv] if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing atlas files: {missing}")
        roi_df = pd.read_csv(self.roi_labels_csv)
        if len(roi_df) != 414 or 'roi_name' not in roi_df.columns:
            raise ValueError("ROI labels must have exactly 414 rows with 'roi_name' column")

    def parcellate_subject(self, fmri_path: str, tr: float,
                          resample_atlases: bool = True,
                          target_tr: float = DEFAULT_TARGET_TR,
                          target_duration: float = DEFAULT_TARGET_DURATION):
        fmri_img = image.load_img(fmri_path)

        if resample_atlases:
            resample_kwargs = {
                'target_img': fmri_img,
                'interpolation': 'nearest',
                'force_resample': True,
                'copy_header': True
            }
            glasser_res = image.resample_to_img(self.glasser_nii, **resample_kwargs)
            tian_res = image.resample_to_img(self.tian_nii, **resample_kwargs)
        else:
            glasser_res = self.glasser_nii
            tian_res = self.tian_nii

        masker_kwargs = {
            'standardize': False,
            't_r': tr,
            'detrend': True,
            'low_pass': None,
            'high_pass': None,
            'memory': "nilearn_cache",
            'verbose': 0
        }

        g_ts = NiftiLabelsMasker(glasser_res, **masker_kwargs).fit_transform(fmri_img)
        t_ts = NiftiLabelsMasker(tian_res, **masker_kwargs).fit_transform(fmri_img)

        if g_ts.ndim == 1:
            g_ts = g_ts[np.newaxis, :]
        if t_ts.ndim == 1:
            t_ts = t_ts[np.newaxis, :]

        ts = np.concatenate([g_ts, t_ts], axis=1)

        resampled = resample_timeseries(ts, tr, target_tr=target_tr, target_duration=target_duration)
        if resampled is None:
            logger.debug(f"Skipping {Path(fmri_path).name}: insufficient duration (<{target_duration}s)")
            return None

        resampled = (resampled - resampled.mean(axis=0)) / (resampled.std(axis=0) + 1e-8)
        return resampled

    def process_dataset(self, fmri_paths, tr_values, n_jobs=-1,
                        target_tr=DEFAULT_TARGET_TR,
                        target_duration=DEFAULT_TARGET_DURATION):
        logger.info(f"Processing {len(fmri_paths)} subjects with n_jobs={n_jobs}")

        test_img = image.load_img(fmri_paths[0])
        glasser_img = image.load_img(self.glasser_nii)
        tian_img = image.load_img(self.tian_nii)
        resample = not (
            np.allclose(test_img.affine, glasser_img.affine, atol=1e-3) and
            np.allclose(test_img.affine, tian_img.affine, atol=1e-3)
        )

        if resample:
            logger.warning("Atlases misaligned with fMRI data — will resample atlases")
        else:
            logger.info("Atlases already aligned with fMRI data — skipping resampling")
            
        def _process_one(idx):
            return idx, self.parcellate_subject(
                fmri_paths[idx], tr_values[idx], resample,
                target_tr=target_tr, target_duration=target_duration
            )

        if n_jobs == 1:
            results = [_process_one(i) for i in range(len(fmri_paths))]
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_process_one)(i) for i in range(len(fmri_paths))
            )

        processed_data, valid_indices = [], []
        for idx, data in results:
            if data is not None:
                processed_data.append(data)
                valid_indices.append(idx)

        logger.info(f"✅ Successfully processed {len(processed_data)}/{len(fmri_paths)} subjects")
        return processed_data, valid_indices

    def get_roi_labels(self):
        return pd.read_csv(self.roi_labels_csv)