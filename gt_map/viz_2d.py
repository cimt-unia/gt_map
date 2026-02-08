# gt_map/viz_2d.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Union, List, Tuple, Optional, Any
import warnings

# Internal imports
from .core import GlasserTianParcellator
from .viz_3d import _get_roi_coords  # ← FIXED: Import shared coordinate function

# Suppress NumPy masked array warnings that commonly occur with neuroimaging data
warnings.filterwarnings('ignore', message='.*converting a masked element to nan.*', category=UserWarning)


def _get_adaptive_cut_coords(atlas_img: nib.Nifti1Image, labels: List[int]) -> Tuple[int, int, int]:
    """
    Compute center-of-mass cut coordinates for selected labels.
    
    Parameters
    ----------
    atlas_img : nib.Nifti1Image
        The atlas NIfTI image
    labels : List[int]
        ROI labels to compute center of mass for
        
    Returns
    -------
    Tuple[int, int, int]
        MNI coordinates (x, y, z) rounded to integers
    """
    atlas_data = atlas_img.get_fdata().astype(int)
    mask = np.isin(atlas_data, labels)
    
    if not np.any(mask):
        return (0, 0, 0)
    
    voxels = np.argwhere(mask)
    com_vox = voxels.mean(axis=0)
    com_mni = np.dot(atlas_img.affine, np.append(com_vox, 1))[:3]
    
    return tuple(com_mni.round().astype(int))


def _get_roi_center(atlas_img: nib.Nifti1Image, roi_label: int) -> Tuple[int, int, int]:
    """
    Compute MNI coordinates (x, y, z) of an ROI's center of mass.
    
    Parameters
    ----------
    atlas_img : nib.Nifti1Image
        The atlas NIfTI image
    roi_label : int
        The ROI label to find center for
        
    Returns
    -------
    Tuple[int, int, int]
        MNI coordinates (x, y, z) rounded to integers
    """
    data = atlas_img.get_fdata().astype(int)
    mask = (data == roi_label)
    
    if not np.any(mask):
        return (0, 0, 0)
    
    voxels = np.argwhere(mask)
    com_vox = voxels.mean(axis=0)
    com_mni = np.dot(atlas_img.affine, np.append(com_vox, 1))[:3]
    
    return tuple(com_mni.round().astype(int))


def _create_legend_top_right(ax: plt.Axes, roi_names_full: List[str], colors: List) -> None:
    """
    Add a compact legend in the top-right corner.
    
    Parameters
    ----------
    ax : plt.Axes
        The axes to add legend to
    roi_names_full : List[str]
        ROI names for legend labels
    colors : List
        Colors for legend markers
    """
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
               markersize=10, linewidth=0, label=name)
        for color, name in zip(colors, roi_names_full)
    ]
    ax.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(1.8, 1.0),
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor='gray',
        borderpad=0.5,
        handletextpad=0.3
    )


def _plot_atlas_rois(
    atlas_img: nib.Nifti1Image,
    labels: List[int],
    roi_indices: List[int],
    roi_df: pd.DataFrame,
    title: str,
    cut_coords: Tuple[int, int, int],
    cmap_name: str
) -> None:
    """
    Plot ROIs from a single atlas.
    
    Parameters
    ----------
    atlas_img : nib.Nifti1Image
        The atlas image
    labels : List[int]
        Atlas-specific labels to plot
    roi_indices : List[int]
        Global ROI indices for metadata lookup
    roi_df : pd.DataFrame
        DataFrame with ROI metadata
    title : str
        Plot title
    cut_coords : Tuple[int, int, int]
        MNI coordinates for slice positions
    cmap_name : str
        Matplotlib colormap name
    """
    atlas_data = atlas_img.get_fdata().astype(int)
    mask = np.isin(atlas_data, labels)
    
    # Fix for single ROI grey color issue:
    # - Single ROI: normalize to intensity=1 for vivid color
    # - Multiple ROIs: preserve original labels for differentiation
    if len(labels) == 1:
        selected_data = np.where(mask, 1, 0).astype(np.int32)
    else:
        selected_data = np.where(mask, atlas_data, 0).astype(np.int32)
    
    selected_img = new_img_like(atlas_img, selected_data)

    plt.figure(figsize=(12, 8))
    plotting.plot_roi(
        roi_img=selected_img,
        title=title,
        cut_coords=cut_coords,
        display_mode="ortho",
        black_bg=False,
        colorbar=False,
        cmap=cmap_name
    )

    roi_names_full = [roi_df.iloc[idx]['roi_name'] for idx in roi_indices]

    # Handle different matplotlib versions
    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    n = len(roi_names_full)
    # Single ROI: use vivid color from colormap's upper range
    # Multiple ROIs: distribute across colormap
    if n == 1:
        colors = [cmap(0.8)]  # Use bright end of colormap
    else:
        colors = cmap(np.linspace(0, 1, n))

    _create_legend_top_right(plt.gca(), roi_names_full, colors)
    plotting.show()


def plot_selected_rois(
    indices: Union[int, List[int]],
    title_prefix: str = "Selected ROIs",
    glasser_cut_coords: Tuple[int, int, int] = (5, -80, 10),
    tian_cut_coords: Tuple[int, int, int] = (0, 10, -8),
    parcellator: Optional[GlasserTianParcellator] = None
) -> None:
    """
    Plot selected ROIs by global index (0–413).
    
    Parameters
    ----------
    indices : int or List[int]
        Global ROI indices to plot (0–413)
    title_prefix : str, default="Selected ROIs"
        Prefix for plot titles
    glasser_cut_coords : Tuple[int, int, int], default=(5, -80, 10)
        Cut coordinates for cortical plots (auto-adjusted if default)
    tian_cut_coords : Tuple[int, int, int], default=(0, 10, -8)
        Cut coordinates for subcortical plots (auto-adjusted if default)
    parcellator : GlasserTianParcellator, optional
        Existing parcellator instance (creates new if None)
        
    Raises
    ------
    ValueError
        If any index is outside range 0–413
    FileNotFoundError
        If roi_networks.csv is missing
    """
    # Normalize to list
    if isinstance(indices, int):
        indices = [indices]

    # Validate indices
    for idx in indices:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")

    # Use provided or instantiate new parcellator
    parc = parcellator or GlasserTianParcellator()

    # Load metadata
    roi_df_path = parc.atlas_dir / "roi_networks.csv"
    if not roi_df_path.exists():
        raise FileNotFoundError(f"Missing roi_networks.csv at {roi_df_path}")
    roi_df = pd.read_csv(roi_df_path)

    # Separate cortical (0-359) and subcortical (360-413)
    glasser_indices = [i for i in indices if 0 <= i <= 359]
    tian_indices = [i for i in indices if 360 <= i <= 413]

    # Print user-friendly info
    print(f"\nPlotting {len(indices)} ROI{'s' if len(indices) != 1 else ''}:")
    for idx in sorted(indices):
        row = roi_df.iloc[idx]
        full_name = row['region_full_name']
        roi_code = row['roi_name']
        print(f"  • {full_name}, {roi_code}, (Index: {idx})")

    # Plot cortical ROIs
    if glasser_indices:
        glasser_labels = [idx + 1 for idx in glasser_indices]
        DEFAULT_GLASSER = (5, -80, 10)
        
        # Auto-adjust cut coords if using default
        if glasser_cut_coords == DEFAULT_GLASSER:
            glasser_img = nib.load(parc.glasser_nii)
            cut_coords = _get_adaptive_cut_coords(glasser_img, glasser_labels)
        else:
            cut_coords = glasser_cut_coords

        _plot_atlas_rois(
            atlas_img=nib.load(parc.glasser_nii),
            labels=glasser_labels,
            roi_indices=glasser_indices,
            roi_df=roi_df,
            title="Cortical",
            cut_coords=cut_coords,
            cmap_name="coolwarm"
        )

    # Plot subcortical ROIs — NOW WITH ADAPTIVE COORDINATES!
    if tian_indices:
        tian_labels = [idx - 360 + 1 for idx in tian_indices]
        DEFAULT_TIAN = (0, 10, -8)  # ← Define default
        
        # Auto-adjust cut coords if using default
        if tian_cut_coords == DEFAULT_TIAN:
            tian_img = nib.load(parc.tian_nii)
            cut_coords = _get_adaptive_cut_coords(tian_img, tian_labels)
        else:
            cut_coords = tian_cut_coords

        _plot_atlas_rois(
            atlas_img=nib.load(parc.tian_nii),
            labels=tian_labels,
            roi_indices=tian_indices,
            roi_df=roi_df,
            title="Subcortical",
            cut_coords=cut_coords,  # ← Now adaptive!
            cmap_name="cividis"
        )


def plot_roi_connectivity_2d(
    indices: Union[int, List[int]],
    matrix: np.ndarray,
    parcellator: Optional[GlasserTianParcellator] = None,
    title: str = None,
    edge_cmap: str = 'RdYlBu_r',
    node_size: int = 80,
    display_mode: str = 'ortho'
) -> None:
    """
    Plot connectivity between N ROIs using nilearn.plot_connectome.
    
    Parameters
    ----------
    indices : int or list of int
        Global ROI indices (0-413)
    matrix : np.ndarray
        Full NxN connectivity matrix
    parcellator : GlasserTianParcellator, optional
        Parcellator instance (creates new if None)
    title : str, optional
        Plot title
    edge_cmap : str
        Colormap for edges (use diverging for correlation matrices)
    node_size : int
        Size of node markers
    display_mode : str
        Nilearn display mode ('ortho', 'z', 'x', etc.)
    """
    # Normalize indices
    if isinstance(indices, int):
        indices = [indices]
    
    # Validate inputs
    if len(indices) < 2:
        raise ValueError("At least 2 ROIs required")
    for idx in indices:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")
    if matrix.shape[0] <= max(indices):
        raise ValueError("Matrix too small for given indices")

    # Initialize parcellator
    parc = parcellator or GlasserTianParcellator()
    roi_df = pd.read_csv(parc.atlas_dir / "roi_networks.csv")

    # Get coordinates and colors
    coords = []
    node_colors = []
    print("\nPlotting connections:")
    
    for idx in indices:
        if 0 <= idx <= 359:  # Glasser (cortical)
            atlas_img = nib.load(parc.glasser_nii)
            label = idx + 1
            color = '#e41a1c'  # Red (cortical)
        else:  # Tian (subcortical)
            atlas_img = nib.load(parc.tian_nii)
            label = idx - 360 + 1
            color = '#377eb8'  # Blue (subcortical)
        
        coord = _get_roi_center(atlas_img, label)
        coords.append(coord)
        node_colors.append(color)
        
        name = roi_df.iloc[idx]['region_full_name']
        print(f"  • {name} (Idx {idx}) at {coord}")

    # Extract submatrix
    sub_indices = np.array(indices)
    adj_sub = matrix[np.ix_(sub_indices, sub_indices)].copy()
    np.fill_diagonal(adj_sub, 0)

    # Auto-generate title
    if title is None:
        rois = [roi_df.iloc[i]['roi_name'] for i in indices]
        title = " ↔ ".join(rois[:3]) + ("..." if len(rois) > 3 else "")

    # Plot
    plotting.plot_connectome(
        adjacency_matrix=adj_sub,
        node_coords=coords,
        node_color=node_colors,
        node_size=node_size,
        edge_cmap=edge_cmap,
        edge_vmin=np.min(adj_sub),
        edge_vmax=np.max(adj_sub),
        display_mode=display_mode,
        title=title,
        black_bg=False,
        colorbar=True
    )
    plotting.show()
    print("✅ Plot displayed.")
