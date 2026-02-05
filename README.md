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

### Example
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




### Visuals



```python
from gt_map import plot_selected_rois, plot_two_roi_connectivity

# ROI plot
plot_selected_rois([0, 1, 403, 405])

# Connectivity plot
plot_two_roi_connectivity(35, 413, weight=0.99)
```

These functions render brain maps overlaid with selected ROIs or pairwise connectivity strengths, facilitating rapid quality control and exploratory analysis.



---



## Details

The package includes the following files in **MNI152NLin6Asym** space:
- `glasser_360_MNI152NLin6Asym.nii.gz` (cortical, 360 ROIs)
- `tian_subcortex_54_MNI152NLin6Asym.nii` (subcortical, 54 ROIs)
- `roi_labels.csv` (414 rows, column: `roi_name`)

> These are derived from publicly available sources (see Acknowledgments).  
> The bundled data enables **zero-configuration usage** while ensuring reproducibility.




---



## References

- **Glasser et al. (2016)**  
  *A multi-modal parcellation of human cerebral cortex*  
  Nature 536, 171–178. https://doi.org/10.1038/nature18933  

- **Tian et al. (2020)**  
  *Topographic organization of the human subcortex unveiled with functional connectivity gradients*  
  Nature Neuroscience 23, 1421–1432. https://doi.org/10.1038/s41593-020-00711-6  
  Atlas files available at: https://github.com/yetianmed/subcortex

- **Nilearn team** for robust, open-source neuroimaging utilities  
  https://nilearn.github.io/












