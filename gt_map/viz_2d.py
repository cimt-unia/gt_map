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
    parcellator: Optional[GlasserTianParcellator] = None,
    use_full_names_in_legend: bool = False,
    save_dir: Optional[str] = None
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
    use_full_names_in_legend : bool, default=False
        If True, legend displays full anatomical name + hemisphere + system.
    save_dir : str, optional
        Directory to save high-quality PNGs (300 DPI). If None, plots are displayed.
        
    Raises
    ------
    ValueError
        If any index is outside range 0–413
    FileNotFoundError
        If roi_networks.csv is missing
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from nilearn.image import new_img_like
    from matplotlib.lines import Line2D

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

    # Helper: create integrated plot for one atlas type
    def _plot_single_atlas(indices_list, is_cortical=True):
        if not indices_list:
            return

        # Map global indices → atlas labels
        if is_cortical:
            atlas_img = nib.load(parc.glasser_nii)
            labels = [idx + 1 for idx in indices_list]
            title = "Cortical"
            cmap_name = "coolwarm"
            DEFAULT_CUT = (5, -80, 10)
            user_cut = glasser_cut_coords
        else:
            atlas_img = nib.load(parc.tian_nii)
            labels = [idx - 360 + 1 for idx in indices_list]
            title = "Subcortical"
            cmap_name = "cividis"
            DEFAULT_CUT = (0, 10, -8)
            user_cut = tian_cut_coords

        # Adaptive cut coords
        if user_cut == DEFAULT_CUT:
            cut_coords = _get_adaptive_cut_coords(atlas_img, labels)
        else:
            cut_coords = user_cut

        # Build mask image
        atlas_data = atlas_img.get_fdata().astype(int)
        mask = np.isin(atlas_data, labels)
        if len(labels) == 1:
            selected_data = np.where(mask, 1, 0).astype(np.int32)
        else:
            selected_data = np.where(mask, atlas_data, 0).astype(np.int32)
        selected_img = new_img_like(atlas_img, selected_data)

        # Get unique non-zero labels for coloring
        unique_labels = np.unique(selected_data[selected_data > 0])
        n = len(unique_labels)

        # Colormap
        try:
            cmap = plt.colormaps.get_cmap(cmap_name)
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap_name)

        # Generate consistent colors: map label → color
        if n == 1:
            # Use fixed position for single ROI
            label_colors = {unique_labels[0]: cmap(0.8)}
        else:
            # Normalize label IDs to [0,1]
            norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
            label_colors = {lbl: cmap(norm(lbl)) for lbl in unique_labels}

        # Map global index → color
        roi_colors = []
        legend_labels = []
        for idx in indices_list:
            if is_cortical:
                lbl = idx + 1
            else:
                lbl = idx - 360 + 1
            roi_colors.append(label_colors[lbl])
            
            if use_full_names_in_legend:
                row = roi_df.iloc[idx]
                label_str = f"{row['region_full_name']} ({row['hemisphere']}; {row['functional_system']})"
            else:
                label_str = roi_df.iloc[idx]['roi_name']
            legend_labels.append(label_str)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        display = plotting.plot_roi(
            roi_img=selected_img,
            title=title,
            cut_coords=cut_coords,
            display_mode="ortho",
            black_bg=False,
            colorbar=False,
            cmap=cmap_name,
            figure=fig
        )

        # Custom legend with synchronized colors
        handles = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                   markersize=10, linewidth=0, label=label)
            for color, label in zip(roi_colors, legend_labels)
        ]
        ax = plt.gca()
        ax.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(1.8, 1.0),
            fontsize=9,
            frameon=True,
            fancybox=True,
            edgecolor='gray'
        )

        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = "cortical" if is_cortical else "subcortical"
            path = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_{suffix}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"✅ Saved: {path}")
        else:
            plotting.show()

    # Plot both types
    _plot_single_atlas(glasser_indices, is_cortical=True)
    _plot_single_atlas(tian_indices, is_cortical=False)

    if save_dir:
        total = len(glasser_indices) + len(tian_indices)
        if total > 0:
            print(f"📊 Completed: {title_prefix}")

def plot_selected_rois_o1(
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
