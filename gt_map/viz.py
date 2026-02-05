# gt_map/viz.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Union, List, Tuple, Optional
import warnings

# Internal imports
from .core import GlasserTianParcellator

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
        Cut coordinates for subcortical plots
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
            print(f"  → Auto-adjusted cortical cut_coords to: {cut_coords}")
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

    # Plot subcortical ROIs
    if tian_indices:
        tian_labels = [idx - 360 + 1 for idx in tian_indices]
        _plot_atlas_rois(
            atlas_img=nib.load(parc.tian_nii),
            labels=tian_labels,
            roi_indices=tian_indices,
            roi_df=roi_df,
            title="Subcortical",
            cut_coords=tian_cut_coords,
            cmap_name="cividis"
        )


def plot_two_roi_connectivity(
    roi_index_1: int,
    roi_index_2: int,
    weight: float = 1.0,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    parcellator: Optional[GlasserTianParcellator] = None
) -> None:
    """
    Plot a single connection between two ROIs with beautiful purple aesthetics.
    
    Parameters
    ----------
    roi_index_1, roi_index_2 : int
        Global ROI indices (0–413)
    weight : float, default=1.0
        Connection strength (typically -1 to 1)
    title : str, optional
        Plot title (auto-generated from ROI names if None)
    save_path : str or Path, optional
        Path to save figure (displays only if None)
    parcellator : GlasserTianParcellator, optional
        Existing parcellator instance (creates new if None)
        
    Raises
    ------
    ValueError
        If any index is outside range 0–413
    """
    # Validate indices
    for idx in [roi_index_1, roi_index_2]:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")

    # Use provided or instantiate new parcellator
    parc = parcellator or GlasserTianParcellator()
    
    # Load ROI metadata
    roi_df_path = parc.atlas_dir / "roi_networks.csv"
    if not roi_df_path.exists():
        raise FileNotFoundError(f"Missing roi_networks.csv at {roi_df_path}")
    roi_df = pd.read_csv(roi_df_path)

    # Helper: get atlas image and label for an index
    def get_atlas_and_label(idx: int) -> Tuple[nib.Nifti1Image, int]:
        """Return (atlas_img, atlas_label) for global index."""
        if 0 <= idx <= 359:  # Glasser (cortical)
            return nib.load(parc.glasser_nii), idx + 1
        else:  # Tian (subcortical)
            return nib.load(parc.tian_nii), idx - 360 + 1

    img1, label1 = get_atlas_and_label(roi_index_1)
    img2, label2 = get_atlas_and_label(roi_index_2)

    # Get MNI coordinates for each ROI
    coord1 = _get_roi_center(img1, label1)
    coord2 = _get_roi_center(img2, label2)

    # Extract ROI metadata
    name1 = roi_df.iloc[roi_index_1]['region_full_name']
    name2 = roi_df.iloc[roi_index_2]['region_full_name']
    roi_code1 = roi_df.iloc[roi_index_1]['roi_name']
    roi_code2 = roi_df.iloc[roi_index_2]['roi_name']

    # Build 2x2 adjacency matrix
    adj_matrix = np.array([[0, weight], [weight, 0]])

    # Node colors: purple aesthetic
    color1 = "#96336E"  # Deep magenta-purple
    color2 = "#5b85df"  # Complementary blue-purple
    node_colors = [color1, color2]

    # Create figure with light background
    fig = plt.figure(figsize=(18, 12), facecolor='#faf9fc')

    # Plot connectome
    display = plotting.plot_connectome(
        adjacency_matrix=adj_matrix,
        node_coords=[coord1, coord2],
        node_color=node_colors,
        node_size=150,
        edge_cmap="Purples",
        edge_vmin=-1,
        edge_vmax=1,
        edge_threshold=None,
        display_mode='ortho',
        black_bg=False,
        alpha=0.4,
        edge_kwargs={'linewidth': 3.0, 'alpha': 0.9},
        node_kwargs={
            'edgecolors': '#2d1b4e',
            'linewidths': 2.5,
            'alpha': 0.95,
        },
        colorbar=True,
        figure=fig
    )

    # Create legend labels
    legend_labels = [
        f"{roi_code1} • {name1}",
        f"{roi_code2} • {name2}"
    ]

    # Create legend handles with styled markers
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color1, markersize=20,
               markeredgecolor='#2d1b4e', markeredgewidth=2,
               label=legend_labels[0], linestyle=''),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color2, markersize=20,
               markeredgecolor='#2d1b4e', markeredgewidth=2,
               label=legend_labels[1], linestyle='')
    ]

    # Add styled legend
    legend = fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.3, 0.99),
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor="#240b39",
        facecolor='#fdfbff',
        framealpha=0.95,
        borderpad=1,
        labelspacing=1.2
    )
    
    # Style legend text
    for text in legend.get_texts():
        text.set_color("#110931")
        text.set_fontfamily('sans-serif')

    # Ensure background color consistency
    fig.patch.set_facecolor('#faf9fc')

    # Add custom title if provided
    if title:
        fig.suptitle(title, fontsize=24, color="#110931", y=0.98)

    # Save or display
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='#faf9fc',
            edgecolor='none'
        )
        print(f"💾 Saved to {save_path}")

    plt.show()
