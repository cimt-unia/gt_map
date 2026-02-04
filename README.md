# Glasser-Tian fMRI Atlas

A robust, memory-efficient fMRI parcellation tool using:
- **Glasser et al. (2016)**: 360 cortical regions  
- **Tian et al. (2020)**: 54 subcortical regions  
→ **Total: 414 ROIs**

GT-MAP enables reproducible time series extraction across heterogeneous datasets—without discarding subjects due to non-standard TR or short scan durations.  
**Atlases are bundled with the package**, so no external downloads are required.

---

## Features

- **TR-flexible**: Accepts any repetition time (`TR`)
- **Resamples first, standardizes after**: Methodologically correct; preserves signal integrity
- **No temporal filtering**: Avoids `padlen` errors on short or degenerate runs
- **Parallel batch processing**: Scales across subjects with `joblib`
- **Memory-safe**: Uses nilearn caching; no full-dataset loading
- **Robust validation**: Automatically skips scans shorter than target duration (no crash)

Designed for ABIDE, UK Biobank, ADHD-200, and other multi-site fMRI studies.

---

## 📦 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/cimt-unia/gt_map.git
```

> **Requirements**: Python ≥3.9, `nilearn≥0.9`, `numpy`, `pandas`, `scipy`, `joblib`

---

## Usage

### Single subject (using bundled atlases)
```python
from gt_map import GlasserTianParcellator

# No atlas_dir needed — uses built-in atlases
parcellator = GlasserTianParcellator()

ts = parcellator.parcellate_subject(
    fmri_path="sub-01_task-rest_bold.nii.gz",
    tr=2.5,                     # actual TR in seconds
    target_tr=2.0,              # desired TR
    target_duration=300.0       # 5 minutes → 150 timepoints
)
# ts.shape → (150, 414)
```

### Use custom atlases (optional)
```python
parcellator = GlasserTianParcellator(atlas_dir="/path/to/custom/atlases")
```

### Batch processing
```python
fmri_files = ["sub-01.nii.gz", "sub-02.nii.gz", ...]
trs = [2.5, 2.0, ...]

data, valid_idx = parcellator.process_dataset(
    fmri_paths=fmri_files,
    tr_values=trs,
    n_jobs=8
)
# data: list of (150, 414) arrays
# valid_idx: indices of successfully processed subjects
```

### Phenotype standardization
```python
from gt_map import create_analysis_phenotype

phenotype = create_analysis_phenotype(
    df,
    eid_col='SUB_ID',
    age_col='AGE_AT_SCAN',
    sex_col='SEX',
    target_col='DX_GROUP'
)
# Output: ['eid', 'Age', 'Sex', 'Target']
```

---

## Bundled Atlas Details

The package includes the following files in **MNI152NLin6Asym** space:
- `glasser_360_MNI152NLin6Asym.nii.gz` (cortical, 360 ROIs)
- `tian_subcortex_54_MNI152NLin6Asym.nii` (subcortical, 54 ROIs)
- `roi_labels.csv` (414 rows, column: `roi_name`)

> These are derived from publicly available sources (see Acknowledgments).  
> The bundled data enables **zero-configuration usage** while ensuring reproducibility.

---

## Scientific Rationale

Many pipelines **reject** subjects with non-standard TR or short durations. GT-MAP **repairs** them by:
1. Resampling time series to a common temporal grid (fixed TR and duration),
2. Applying standardization *only after* resampling (to avoid distorting interpolated values),
3. Skipping only those scans truly too short (< target duration).

This maximizes usable data while maintaining methodological rigor—embodying a **“Repair—Not Reject”** philosophy.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## 🙌 Acknowledgments

- **Glasser et al. (2016)**  
  *A multi-modal parcellation of human cerebral cortex*  
  Nature 536, 171–178. https://doi.org/10.1038/nature18933  

- **Tian et al. (2020)**  
  *Topographic organization of the human subcortex unveiled with functional connectivity gradients*  
  Nature Neuroscience 23, 1421–1432. https://doi.org/10.1038/s41593-020-00711-6  
  Atlas files available at: https://github.com/yetianmed/subcortex

- **Nilearn team** for robust, open-source neuroimaging utilities  
  https://nilearn.github.io/


<img width="893" height="370" alt="image" src="https://github.com/user-attachments/assets/f6c7a6a6-f8ab-423e-b2fb-52f023ce39e0" />






