# gt_map/viz.py

"""
Visualization utilities for Glasser-Tian parcellation.
Provides high-level plotting of selected ROIs with adaptive slicing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
from matplotlib.lines import Line2D

# Internal imports — relative to package
from .core import GlasserTianParcellator


def _get_adaptive_cut_coords(atlas_img, labels):
    """Compute center-of-mass cut coordinates for selected labels."""
    atlas_data = atlas_img.get_fdata().astype(int)
    mask = np.isin(atlas_data, labels)
    if not np.any(mask):
        return (0, 0, 0)
    voxels = np.argwhere(mask)
    com_vox = voxels.mean(axis=0)
    com_mni = np.dot(atlas_img.affine, np.append(com_vox, 1))[:3]
    return tuple(com_mni.round().astype(int))


def _create_legend_top_right(ax, roi_names_full, colors):
    """Add a compact legend in the top-right corner."""
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


def _plot_atlas_rois(atlas_img, labels, roi_indices, roi_df, title, cut_coords, cmap_name):
    """Plot ROIs from a single atlas."""
    atlas_data = atlas_img.get_fdata().astype(int)
    mask = np.isin(atlas_data, labels)
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

    roi_names_full = [f"{roi_df.iloc[idx]['roi_name']}" for idx in roi_indices]

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    n = len(roi_names_full)
    colors = cmap(np.linspace(0, 1, n)) if n > 1 else [cmap(0.5)]

    _create_legend_top_right(plt.gca(), roi_names_full, colors)
    plotting.show()


def plot_selected_rois(
    indices,
    title_prefix="Selected ROIs",
    glasser_cut_coords=(5, -80, 10),
    tian_cut_coords=(0, 10, -8),
    parcellator=None
):
    """
    Plot selected ROIs by global index (0–413).
    
    Parameters
    ----------
    indices : int or list of int
        Global ROI indices (0–359: Glasser; 360–413: Tian).
    title_prefix : str
        Prefix for plot titles.
    glasser_cut_coords : tuple or None
        Cut coordinates for cortical view. If default `(5, -80, 10)` is used,
        adaptive coordinates are computed automatically.
    tian_cut_coords : tuple
        Cut coordinates for subcortical view.
    parcellator : GlasserTianParcellator, optional
        Reuse an existing instance; if None, a new one is created.
    """
    if isinstance(indices, int):
        indices = [indices]

    for idx in indices:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")

    # Use provided or instantiate new parcellator
    parc = parcellator or GlasserTianParcellator()
    roi_df = pd.read_csv(parc.atlas_dir / "roi_networks.csv")

    glasser_indices = [i for i in indices if 0 <= i <= 359]
    tian_indices = [i for i in indices if 360 <= i <= 413]

    print(f"\nPlotting {len(indices)} ROIs:")
    for idx in sorted(indices):
        row = roi_df.iloc[idx]
        full_name = row['region_full_name']
        roi_code = row['roi_name']
        print(f"  • {full_name}, {roi_code}, (Index: {idx})")

    # Cortical
    if glasser_indices:
        glasser_labels = [idx + 1 for idx in glasser_indices]
        DEFAULT_GLASSER = (5, -80, 10)
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
            title=f"{title_prefix} (Cortical)",
            cut_coords=cut_coords,
            cmap_name="coolwarm"
        )

    # Subcortical
    if tian_indices:
        tian_labels = [idx - 360 + 1 for idx in tian_indices]
        _plot_atlas_rois(
            atlas_img=nib.load(parc.tian_nii),
            labels=tian_labels,
            roi_indices=tian_indices,
            roi_df=roi_df,
            title=f"{title_prefix} (Subcortical)",
            cut_coords=tian_cut_coords,
            cmap_name="cividis"
        )

    