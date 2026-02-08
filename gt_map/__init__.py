# gt_map/__init__.py

import importlib.resources as resources
from pathlib import Path
from .core import GlasserTianParcellator
from .utils import resample_timeseries, create_analysis_phenotype
from .config import set_thread_limit
from .viz_2d import plot_selected_rois, plot_two_roi_connectivity
from .viz_3d import plot_roi_3d_connectivity

def get_bundled_atlas_dir() -> Path:
    try:
        with resources.path("gt_map.data", "roi_networks.csv") as p:
            return p.parent
    except Exception as e:
        raise FileNotFoundError(
            "Bundled atlas files not found. "
            "This may occur if the package was not installed properly. "
            "Please install via: pip install git+https://github.com/cimt-unia/gt_map.git"
        ) from e

__version__ = "0.1.0"

__all__ = [
    "GlasserTianParcellator",
    "resample_timeseries",
    "create_analysis_phenotype",
    "set_thread_limit",
    "get_bundled_atlas_dir",
    "plot_selected_rois", 
    "plot_two_roi_connectivity",
    "plot_roi_3d_connectivity"  
]
