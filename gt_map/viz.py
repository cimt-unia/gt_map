# gt_map/viz.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
from matplotlib.lines import Line2D
from pathlib import Path

# Internal imports
from .core import GlasserTianParcellator
from . import get_bundled_atlas_dir


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


def _get_roi_center(atlas_img, roi_label):
    """Compute MNI coordinates (x, y, z) of an ROI's center of mass."""
    data = atlas_img.get_fdata().astype(int)
    mask = (data == roi_label)
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
    """
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

    # Separate cortical and subcortical
    glasser_indices = [i for i in indices if 0 <= i <= 359]
    tian_indices = [i for i in indices if 360 <= i <= 413]

    # Print user-friendly info
    print(f"\nPlotting {len(indices)} ROIs:")
    for idx in sorted(indices):
        row = roi_df.iloc[idx]
        full_name = row['region_full_name']
        roi_code = row['roi_name']
        print(f"  • {full_name}, {roi_code}, (Index: {idx})")

    # Plot cortical
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

    # Plot subcortical
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


def plot_two_roi_connectivity(
    roi_index_1, roi_index_2, weight=1.0, title=None, save_path=None, parcellator=None
):
    """
    Plot a single connection between two ROIs with beautiful purple aesthetics.
    
    Parameters
    ----------
    roi_index_1, roi_index_2 : int
        Global ROI indices (0–413)
    weight : float
        Connection strength (-1 to 1)
    title : str, optional
        Plot title (if None, auto-generated from ROI names)
    save_path : str or Path, optional
        Save figure to this path
    parcellator : GlasserTianParcellator, optional
        Reuse an existing instance; if None, a new one is created.
    """
    # Validate indices
    for idx in [roi_index_1, roi_index_2]:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0–413)")

    # Use provided or instantiate new parcellator
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

    # ROI info
    name1 = roi_df.iloc[roi_index_1]['region_full_name']
    name2 = roi_df.iloc[roi_index_2]['region_full_name']
    roi_code1 = roi_df.iloc[roi_index_1]['roi_name']
    roi_code2 = roi_df.iloc[roi_index_2]['roi_name']

    # Build adjacency matrix
    adj_matrix = np.array([[0, weight], [weight, 0]])

    # Node colors: purple scheme
    color1 = "#96336E" if roi_index_1 <= 359 else "#96336E"
    color2 = "#5b85df" if roi_index_2 <= 359 else "#5b85df"
    node_colors = [color1, color2]

    # figure
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

    # Legend
    legend_labels = [
        f"{roi_code1} • {name1}",
        f"{roi_code2} • {name2}"
    ]

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=color1, markersize=20,
               markeredgecolor='#2d1b4e', markeredgewidth=2,
               label=legend_labels[0], linestyle=''),
        Line2D([0], [0], marker='o', color='w', 
               markerface, markersize=20,
               markeredgecolor='#2d1b4e', markeredgewidth=2,
               label=legend_labels[1], linestyle='')
    ]

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
    for text in legend.get_texts():
        text.set_color("#110931")
        text.set_fontfamily('sans-serif')

    # Style axes
    for ax in fig.get_axes():
        ax.tick_params(colors='#110931', labelsize=15)
        if ax.get_ylabel():
            ax.set_ylabel('Connection Strength', color='#110931', fontsize=11, fontweight='semibold')

    fig.patch.set_facecolor('#faf9fc')

    # Save
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
