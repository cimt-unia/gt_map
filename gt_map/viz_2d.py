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
    cmap_name: str,
    node_colors: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    dpi: int = 300
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
    node_colors : List[str], optional
        Custom hex/RGB colors for each ROI (e.g., ['#FF0000', '#00FF00']).
        If None, uses colormap.
    output_file : str, optional
        Path to save the figure (e.g., "figure.png"). If None, displays interactively.
    dpi : int, default=300
        Resolution for saved figures.
    """
    atlas_data = atlas_img.get_fdata().astype(int)
    
    # Assign sequential IDs (1, 2, 3...) to ensure consistent coloring
    selected_data = np.zeros_like(atlas_data, dtype=np.int32)
    for i, lbl in enumerate(labels):
        selected_data[atlas_data == lbl] = i + 1
    
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
    n = len(roi_names_full)

    # Generate colors to match nilearn's internal mapping:
    # nilearn uses Normalize(vmin=1, vmax=n) → label i maps to cmap((i-1)/(n-1)) for i=1..n
    if node_colors is not None:
        if len(node_colors) != n:
            raise ValueError(f"Expected {n} colors, got {len(node_colors)}")
        colors = [plt.matplotlib.colors.to_rgba(c) for c in node_colors]
    else:
        try:
            cmap = plt.colormaps.get_cmap(cmap_name)
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap_name)
        if n == 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(i / (n - 1)) for i in range(n)]  # i = 0 → label 1, i = n-1 → label n

    _create_legend_top_right(plt.gca(), roi_names_full, colors)

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
        plt.close()
    else:
        plotting.show()


def plot_selected_rois(
    indices: Union[int, List[int]],
    title_prefix: str = "Selected ROIs",
    glasser_cut_coords: Tuple[int, int, int] = (5, -80, 10),
    tian_cut_coords: Tuple[int, int, int] = (0, 10, -8),
    parcellator: Optional[GlasserTianParcellator] = None,
    node_colors: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300
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
    node_colors : List[str], optional
        Custom colors for all ROIs (length must match indices). 
        Order: [color_for_indices[0], color_for_indices[1], ...]
    output_dir : str or Path, optional
        Directory to save figures. If None, displays interactively.
    dpi : int, default=300
        Resolution for saved figures
        
    Raises
    ------
    ValueError
        If any index is outside range 0–413 or color length mismatch
    FileNotFoundError
        If roi_networks.csv is missing
    """
    if isinstance(indices, int):
        indices = [indices]

    for idx in indices:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")

    parc = parcellator or GlasserTianParcellator()
    roi_df_path = parc.atlas_dir / "roi_networks.csv"
    if not roi_df_path.exists():
        raise FileNotFoundError(f"Missing roi_networks.csv at {roi_df_path}")
    roi_df = pd.read_csv(roi_df_path)

    glasser_indices = [i for i in indices if 0 <= i <= 359]
    tian_indices = [i for i in indices if 360 <= i <= 413]

    print(f"\nPlotting {len(indices)} ROI{'s' if len(indices) != 1 else ''}:")
    for idx in sorted(indices):
        row = roi_df.iloc[idx]
        full_name = row['region_full_name']
        roi_code = row['roi_name']
        print(f"  • {full_name}, {roi_code}, (Index: {idx})")

    # Split colors by atlas type if provided
    glasser_colors = None
    tian_colors = None
    if node_colors is not None:
        if len(node_colors) != len(indices):
            raise ValueError(f"node_colors length ({len(node_colors)}) must match indices length ({len(indices)})")
        # Map colors to indices by position
        idx_to_color = dict(zip(indices, node_colors))
        glasser_colors = [idx_to_color[idx] for idx in glasser_indices]
        tian_colors = [idx_to_color[idx] for idx in tian_indices]

    # Plot cortical ROIs
    if glasser_indices:
        glasser_labels = [idx + 1 for idx in glasser_indices]
        DEFAULT_GLASSER = (5, -80, 10)
        if glasser_cut_coords == DEFAULT_GLASSER:
            glasser_img = nib.load(parc.glasser_nii)
            cut_coords = _get_adaptive_cut_coords(glasser_img, glasser_labels)
        else:
            cut_coords = glasser_cut_coords

        output_file = None
        if output_dir:
            output_file = Path(output_dir) / f"{title_prefix.replace(' ', '_').lower()}_cortical.png"

        _plot_atlas_rois(
            atlas_img=nib.load(parc.glasser_nii),
            labels=glasser_labels,
            roi_indices=glasser_indices,
            roi_df=roi_df,
            title="Cortical",
            cut_coords=cut_coords,
            cmap_name="coolwarm",
            node_colors=glasser_colors,
            output_file=output_file,
            dpi=dpi
        )

    # Plot subcortical ROIs
    if tian_indices:
        tian_labels = [idx - 360 + 1 for idx in tian_indices]
        DEFAULT_TIAN = (0, 10, -8)
        if tian_cut_coords == DEFAULT_TIAN:
            tian_img = nib.load(parc.tian_nii)
            cut_coords = _get_adaptive_cut_coords(tian_img, tian_labels)
        else:
            cut_coords = tian_cut_coords

        output_file = None
        if output_dir:
            output_file = Path(output_dir) / f"{title_prefix.replace(' ', '_').lower()}_subcortical.png"

        _plot_atlas_rois(
            atlas_img=nib.load(parc.tian_nii),
            labels=tian_labels,
            roi_indices=tian_indices,
            roi_df=roi_df,
            title="Subcortical",
            cut_coords=cut_coords,
            cmap_name="cividis",
            node_colors=tian_colors,
            output_file=output_file,
            dpi=dpi
        )


def plot_roi_connectivity_2d(
    roi_index_1: int,
    roi_index_2: int,
    weight: float = 1.0,
    title: Optional[str] = None,
    parcellator: Optional["GlasserTianParcellator"] = None,
    node_colors: Optional[List[str]] = None
) -> None:
    """
    Plot a single connection between two ROIs.
    
    Parameters
    ----------
    roi_index_1, roi_index_2 : int
        Global ROI indices (0-413)
    weight : float
        Connection strength (for coloring; optional)
    title : str, optional
        Plot title
    parcellator : GlasserTianParcellator, optional
        Parcellator instance (creates new if None)
    node_colors : List[str], optional
        Custom colors for the two nodes (e.g., ['#FF0000', '#0000FF'])
        If None, uses default: red=cortical, blue=subcortical
    """
    # Validate indices
    for idx in [roi_index_1, roi_index_2]:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")
    
    # Initialize parcellator
    parc = parcellator or GlasserTianParcellator()
    roi_df = pd.read_csv(parc.atlas_dir / "roi_networks.csv")
    
    # Get atlas images and labels
    def get_atlas_and_label(idx):
        if 0 <= idx <= 359:  # Glasser
            return nib.load(parc.glasser_nii), idx + 1
        else:  # Tian
            return nib.load(parc.tian_nii), idx - 360 + 1
    
    img1, label1 = get_atlas_and_label(roi_index_1)
    img2, label2 = get_atlas_and_label(roi_index_2)
    
    # Get MNI coordinates
    coord1 = _get_roi_center(img1, label1)
    coord2 = _get_roi_center(img2, label2)
    
    # Print info
    name1 = roi_df.iloc[roi_index_1]['region_full_name']
    name2 = roi_df.iloc[roi_index_2]['region_full_name']
    print(f"\nPlotting connection:")
    print(f"  • {name1} (Idx {roi_index_1}) ↔ {name2} (Idx {roi_index_2})")
    print(f"  • MNI coordinates: {coord1} ↔ {coord2}")
    
    # Build 2x2 adjacency matrix
    adj_matrix = np.array([
        [0, weight],
        [weight, 0]
    ])
    
    # Node colors: custom or default (red=cortical, blue=subcortical)
    if node_colors is None:
        color1 = 'red' if roi_index_1 <= 359 else 'blue'
        color2 = 'red' if roi_index_2 <= 359 else 'blue'
        node_colors = [color1, color2]
    else:
        if len(node_colors) != 2:
            raise ValueError("node_colors must contain exactly 2 color values")
    
    # Plot
    if title is None:
        title = f"{roi_df.iloc[roi_index_1]['roi_name']} ↔ {roi_df.iloc[roi_index_2]['roi_name']}"
    
    plotting.plot_connectome(
        adjacency_matrix=adj_matrix,
        node_coords=[coord1, coord2],
        node_color=node_colors,
        node_size=80,
        edge_cmap='RdYlBu_r',
        edge_vmin=-1,
        edge_vmax=1,
        display_mode='ortho',
        title=title,
        black_bg=False
    )
    plotting.show()
    print("✅ Connection plot displayed.")
