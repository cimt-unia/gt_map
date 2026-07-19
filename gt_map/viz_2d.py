# gt_map/viz_2d.py
"""2D visualization for GT atlas ROIs using the unified NIfTI."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import new_img_like
import nibabel as nib
from pathlib import Path
from typing import Union, List, Optional
from matplotlib.patches import Patch
from matplotlib.colors import to_hex

from .core import GlasserTianParcellator


def plot_gt_rois(
    indices: Union[int, List[int]],
    title: Optional[str] = None,
    label_type: str = "roi_name",
    cmap: str = 'Set1',
    alpha: float = 0.7,
    show: bool = True,
    save_to: Optional[str] = None,
) -> None:
    """Plot GT atlas ROIs on MNI template using ortho view.

    Parameters
    ----------
    indices : int or list of int
        GT ROI indices (0-413). Maps to NIfTI labels 1-414.
    title : str, optional
        Plot title. Auto-generated from label_type if None. Use '' for no title.
    label_type : str
        Column from roi_labels.csv: 'roi_name', 'region_full_name', or 'full'.
    cmap : str
        Matplotlib colormap for ROIs.
    alpha : float
        Opacity of the ROI overlay (0-1).
    show : bool
        If True, display the figure.
    save_to : str or Path, optional
        If provided, save the figure to this path.
    """
    if isinstance(indices, int):
        indices = [indices]

    parc = GlasserTianParcellator()
    atlas_path = parc.atlas_dir / "GT_414ROIs_atlas.nii.gz"
    labels_path = parc.atlas_dir / "roi_labels.csv"

    if not atlas_path.exists():
        raise FileNotFoundError(f"GT atlas not found at {atlas_path}")

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    roi_df = pd.read_csv(labels_path)

    # Build mask with sequential values for colormap
    mask = np.zeros(atlas_data.shape, dtype=np.int32)
    for i, idx in enumerate(indices):
        mask[atlas_data == idx + 1] = i + 1

    mask_img = new_img_like(atlas_img, mask)

    # Center view on the midpoint of all selected ROIs
    all_voxels = np.argwhere(mask > 0)
    if len(all_voxels) == 0:
        raise ValueError("None of the requested labels found in the atlas.")
    com = nib.affines.apply_affine(atlas_img.affine, all_voxels.mean(axis=0))
    cut_coords = tuple(com.round().astype(int))

    # Build title
    if title is None:
        if len(indices) == 1:
            row = roi_df.iloc[indices[0]]
            if label_type == "full":
                title = f"{row['hemisphere']} {row['region_full_name']}"
            else:
                title = str(row[label_type])
        else:
            title = f"GT: {len(indices)} ROIs"
    elif title == "":
        title = None

    plotting.plot_roi(
        mask_img,
        title=title,
        cut_coords=cut_coords,
        display_mode='ortho',
        cmap=cmap,
        alpha=alpha,
        dim=-0.5,
        black_bg=False,
        draw_cross=True,
        radiological=False,
        colorbar=False,
        output_file=save_to,
    )

    # Color-matched legend for multi-ROI plots
    if len(indices) > 1:
        cmap_obj = plt.colormaps[cmap]
        legend_patches = []
        for i, idx in enumerate(indices):
            row = roi_df.iloc[idx]
            color = to_hex(cmap_obj(i / max(len(indices) - 1, 1)))
            legend_patches.append(Patch(color=color, label=row['roi_name']))

        plt.gcf().legend(
            handles=legend_patches,
            loc='upper left',
            bbox_to_anchor=(1.02, 0.9),
            frameon=True,
            fontsize=8,
            title="ROIs",
            title_fontsize=9,
        )
        plt.gcf().subplots_adjust(right=0.72)

    if show and save_to is None:
        plotting.show()


def plot_gt_connectivity_2d(
    roi_index_1: int,
    roi_index_2: int,
    weight: float = 1.0,
    title: Optional[str] = None,
    node_colors: Optional[List[str]] = None,
    show: bool = True,
) -> None:
    """Plot a single connection between two GT ROIs on MNI template.

    Parameters
    ----------
    roi_index_1, roi_index_2 : int
        Global ROI indices (0-413).
    weight : float
        Connection strength.
    title : str, optional
        Plot title. Auto-generated if None.
    node_colors : List[str], optional
        Custom colors for the two nodes.
    show : bool
        If True, display the figure.
    """
    for idx in [roi_index_1, roi_index_2]:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0-413)")

    parc = GlasserTianParcellator()
    atlas_img = nib.load(parc.atlas_dir / "GT_414ROIs_atlas.nii.gz")
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    roi_df = pd.read_csv(parc.atlas_dir / "roi_labels.csv")

    coords = []
    for idx in [roi_index_1, roi_index_2]:
        mask = atlas_data == idx + 1
        if not np.any(mask):
            raise ValueError(f"Label {idx + 1} not found in atlas.")
        vox_center = np.argwhere(mask).mean(axis=0)
        mni = nib.affines.apply_affine(atlas_img.affine, vox_center)
        coords.append(mni)

    if title is None:
        title = f"{roi_df.iloc[roi_index_1]['roi_name']} ↔ {roi_df.iloc[roi_index_2]['roi_name']}"

    if node_colors is None:
        node_colors = ['red' if i <= 359 else 'blue' for i in [roi_index_1, roi_index_2]]

    adj_matrix = np.array([[0, weight], [weight, 0]])

    plotting.plot_connectome(
        adjacency_matrix=adj_matrix,
        node_coords=coords,
        node_color=node_colors,
        node_size=80,
        edge_cmap='RdYlBu_r',
        edge_vmin=-1,
        edge_vmax=1,
        display_mode='ortho',
        title=title,
        black_bg=False,
    )

    if show:
        plotting.show()
