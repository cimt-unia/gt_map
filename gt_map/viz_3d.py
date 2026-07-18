# gt_map/viz_3d.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from nilearn.datasets import load_fsaverage
from functools import lru_cache
from typing import Optional, Union, List, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import nibabel as nib


# --- CONFIGURATION: NATURE STYLE ---
STYLE = {
    'bg_color': 'rgba(255,255,255,1)',
    'font_family': "Helvetica, Arial, sans-serif",
    'font_color': '#222222',
    'brain_color': '#f4f4f4',
    'lighting': {
        'ambient': 0.4,
        'diffuse': 0.5,
        'roughness': 0.8,
        'specular': 0.1,
        'fresnel': 0.2
    }
}

@lru_cache(maxsize=1)
def _get_fsaverage_mesh():
    """Load and cache fsaverage surface."""
    return load_fsaverage(mesh='fsaverage5')

@lru_cache(maxsize=1)
def _get_unified_atlas(parc: Any) -> nib.Nifti1Image:
    """Load the unified GT_414ROIs atlas from the bundled data directory.
    
    Falls back to on-the-fly merge of Glasser + Tian if the unified file
    hasn't been added to the data directory yet.
    """
    from nilearn import image as nimg
    
    unified_path = parc.atlas_dir / "GT_414ROIs_atlas.nii.gz"
    if unified_path.exists():
        return nib.load(unified_path)
    
    # Fallback: merge Glasser + Tian on the fly
    glasser = nib.load(parc.atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz")
    tian_raw = nib.load(parc.atlas_dir / "tian_subcortex_54_MNI152NLin6Asym.nii")
    tian = nimg.resample_to_img(tian_raw, glasser, interpolation='nearest')
    
    g_data = glasser.get_fdata().astype(np.int32)
    t_data = tian.get_fdata().astype(np.int32)
    t_data[t_data > 0] += 360
    
    unified_data = np.where(t_data > 0, t_data, g_data)
    return nib.Nifti1Image(unified_data.astype(np.int32), glasser.affine, glasser.header)


def _get_roi_coords(parc: Any, indices: List[int]) -> np.ndarray:
    """Extract MNI coordinates from the unified GT atlas.
    
    Parameters
    ----------
    parc : GlasserTianParcellator
        Parcellator instance with atlas_dir pointing to bundled data.
    indices : List[int]
        Global ROI indices (0-413). Maps directly to NIfTI labels 1-414.
    
    Returns
    -------
    np.ndarray of shape (len(indices), 3)
    """
    atlas = _get_unified_atlas(parc)
    data = atlas.get_fdata().astype(np.int32)
    affine = atlas.affine
    
    coords = []
    for idx in indices:
        if not (0 <= idx <= 413):
            raise ValueError(f"Index {idx} out of range (0-413)")
        
        label = idx + 1  # CSV index 0 → NIfTI label 1
        mask = (data == label)
        
        if np.any(mask):
            vox_center = np.argwhere(mask).mean(axis=0)
            mni_center = np.dot(affine, np.append(vox_center, 1))[:3]
            coords.append(mni_center)
        else:
            coords.append([0, 0, 0])
    
    return np.array(coords)


def plot_roi_connectivity_3d(
    indices: Union[int, List[int]],
    matrix: np.ndarray,
    parcellator: Any,
    top_n: Optional[int] = None,
    edge_cmap: str = 'Purples',
    node_cmap: str = 'Pastel1',
    show_labels_on_map: bool = False,
    show_legend: bool = True,
    show_colorbar: bool = True,
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """3D brain connectivity plot with Nature-style aesthetics.
    
    Parameters
    ----------
    indices : int or List[int]
        Global ROI indices (0-413) to display as nodes.
    matrix : np.ndarray
        Square connectivity matrix (414 x 414).
    parcellator : GlasserTianParcellator
        Initialized parcellator with bundled atlas data.
    top_n : int, optional
        Show only the top N strongest edges.
    edge_cmap : str
        Matplotlib colormap for edges.
    node_cmap : str
        Plotly qualitative colormap or matplotlib colormap name for nodes.
    show_labels_on_map : bool
        Display ROI short names on the 3D brain.
    show_legend : bool
        Display the legend panel.
    show_colorbar : bool
        Display the edge weight colorbar.
    title : str
        Plot title.
    save_path : str or Path, optional
        If provided, save the figure as HTML.
    
    Returns
    -------
    go.Figure
    """
    if isinstance(indices, int):
        indices = [indices]
    if parcellator is None:
        raise ValueError("❌ Parcellator required.")

    # Load ROI metadata and coordinates
    roi_df = pd.read_csv(parcellator.atlas_dir / "roi_labels.csv")
    selected_coords = _get_roi_coords(parcellator, indices)
    roi_info = roi_df.iloc[indices].reset_index(drop=True)

    # 1. Process Edges & Nodes
    edges = []
    node_degrees = np.zeros(len(indices))
    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            if i < j:
                weight = matrix[idx1, idx2]
                if weight != 0 and np.isfinite(weight):
                    edges.append({
                        'i': i, 'j': j, 'weight': weight,
                        'coord1': selected_coords[i], 'coord2': selected_coords[j]
                    })
                    node_degrees[i] += abs(weight)
                    node_degrees[j] += abs(weight)

    edges = sorted(edges, key=lambda x: abs(x['weight']))[-top_n:] if top_n else edges

    # 2. Figure Setup
    fig = go.Figure()

    # A. Brain Mesh
    fsaverage = _get_fsaverage_mesh()
    mesh = fsaverage.pial
    verts = np.vstack([mesh.parts['left'].coordinates, mesh.parts['right'].coordinates])
    faces = np.vstack([
        mesh.parts['left'].faces,
        mesh.parts['right'].faces + len(mesh.parts['left'].coordinates)
    ])
    
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=STYLE['brain_color'], opacity=0.15, showlegend=False,
        lighting=STYLE['lighting'], hoverinfo='skip'
    ))

    # B. Edges
    if edges:
        weights = [e['weight'] for e in edges]
        max_w = max(weights) if weights else 1
        cmap_func = plt.get_cmap(edge_cmap)
        
        for edge in edges:
            color_hex = to_hex(cmap_func(edge['weight'] / max_w))
            width = 2 + (edge['weight'] / max_w) * 6
            fig.add_trace(go.Scatter3d(
                x=[edge['coord1'][0], edge['coord2'][0]],
                y=[edge['coord1'][1], edge['coord2'][1]],
                z=[edge['coord1'][2], edge['coord2'][2]],
                mode='lines', line=dict(color=color_hex, width=width),
                opacity=0.9, showlegend=False, hoverinfo='skip'
            ))

        if show_colorbar:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(
                    colorscale=edge_cmap, cmin=0, cmax=max_w, showscale=True,
                    colorbar=dict(
                        thickness=25, len=0.4, x=0.80, y=0.25,
                        xanchor='center', yanchor='bottom', orientation='v',
                        tickfont=dict(family=STYLE['font_family'], size=11, color=STYLE['font_color']),
                        outlinewidth=0
                    )
                ),
                showlegend=False
            ))

    # C. Nodes & Legend
    if hasattr(px.colors.qualitative, node_cmap):
        palette = getattr(px.colors.qualitative, node_cmap)
    else:
        palette = px.colors.sample_colorscale(node_cmap, len(indices))

    for i, row in roi_info.iterrows():
        color = palette[i % len(palette)]
        max_deg = node_degrees.max() if node_degrees.max() > 0 else 1
        size = 8 + (node_degrees[i] / max_deg) * 15
        
        fig.add_trace(go.Scatter3d(
            x=[selected_coords[i, 0]], y=[selected_coords[i, 1]], z=[selected_coords[i, 2]],
            mode='markers+text' if show_labels_on_map else 'markers',
            marker=dict(size=size, color=color, line=dict(color='white', width=2)),
            text=[row['roi_name']] if show_labels_on_map else None,
            textposition="top center",
            name=f"{row['roi_name']} • {row['region_full_name']}",
            hoverinfo='text',
            hovertext=f"{row['region_full_name']}<br>Degree: {node_degrees[i]:.2f}",
            showlegend=show_legend
        ))

    # 3. Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>", y=0.90, x=0.5, xanchor='center',
            font=dict(size=32, family=STYLE['font_family'], color=STYLE['font_color'])
        ),
        width=1200, height=800,
        paper_bgcolor=STYLE['bg_color'],
        scene=dict(
            domain=dict(x=[0.0, 0.70], y=[0.1, 0.9]),
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=0, y=1.8, z=0), up=dict(x=0, y=0, z=1)),
            bgcolor=STYLE['bg_color']
        ),
        legend=dict(
            yanchor="top", y=0.85, xanchor="center", x=0.80,
            bgcolor="rgba(255,255,255,0.5)", bordercolor="#e5e5e5", borderwidth=1,
            font=dict(size=12, family=STYLE['font_family'], color=STYLE['font_color']),
            itemsizing='constant', tracegroupgap=5,
        ) if show_legend else None,
        margin=dict(t=80, b=50, l=0, r=0),
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
    
    return fig
