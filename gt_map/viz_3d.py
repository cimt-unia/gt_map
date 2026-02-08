# gt_map/viz_3d.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from nilearn.datasets import load_fsaverage
from functools import lru_cache
from typing import Optional, Union, List, Any
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import nibabel as nib


# --- CONFIGURATION: NATURE STYLE ---
STYLE = {
    'bg_color': 'rgba(255,255,255,1)',   # Pure White
    'font_family': "Helvetica, Arial, sans-serif",
    'font_color': '#222222',             # Soft Black
    'brain_color': '#f4f4f4',            # Matte Bone White
    'lighting': {
        'ambient': 0.4,
        'diffuse': 0.5,
        'roughness': 0.8,                # Matte finish
        'specular': 0.1,                 # Low reflection
        'fresnel': 0.2
    }
}

@lru_cache(maxsize=1)
def _get_fsaverage_mesh():
    """Load and cache fsaverage surface."""
    return load_fsaverage(mesh='fsaverage5')

def _get_roi_coords(parc: Any, indices: List[int]) -> np.ndarray:
    """Extract MNI coordinates from Glasser/Tian NIfTIs."""
    coords = []
    try:
        glasser_img = nib.load(parc.glasser_nii)
        tian_img = nib.load(parc.tian_nii)
    except (AttributeError, FileNotFoundError) as e:
        raise RuntimeError(f"❌ Atlas files missing or Parcellator invalid: {e}")

    g_data = glasser_img.get_fdata()
    t_data = tian_img.get_fdata()

    for idx in indices:
        if 0 <= idx <= 359:
            label, data, affine = idx + 1, g_data, glasser_img.affine
        elif 360 <= idx <= 413:
            label, data, affine = idx - 360 + 1, t_data, tian_img.affine
        else:
            raise ValueError(f"Index {idx} out of range (0-413)")

        mask = (data == label)
        if np.any(mask):
            vox_center = np.argwhere(mask).mean(axis=0)
            mni_center = np.dot(affine, np.append(vox_center, 1))[:3]
            coords.append(mni_center)
        else:
            coords.append([0, 0, 0])
    return np.array(coords)


def plot_roi_3d_connectivity(
    indices: Union[int, List[int]],
    matrix: np.ndarray,
    parcellator: Any,
    top_n: Optional[int] = None,
    edge_cmap: str = 'Purples',
    node_cmap: str = 'Pastel1',         # Set to 'Plasma', 'Viridis', or 'Prism'
    show_labels_on_map: bool = False, # Toggle names on the brain
    show_legend: bool = True,         # Toggle full names legend
    show_colorbar: bool = True,       # ← NEW: Toggle colorbar visibility
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    3D brain connectivity plot with bottom colorbar and customizable aesthetics.
    """
    if isinstance(indices, int): indices = [indices]
    if parcellator is None: raise ValueError("❌ Parcellator required.")

    # Load Data
    roi_df = pd.read_csv(parcellator.atlas_dir / "roi_networks.csv")
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
                    edges.append({'i': i, 'j': j, 'weight': weight, 
                                  'coord1': selected_coords[i], 'coord2': selected_coords[j]})
                    node_degrees[i] += abs(weight)
                    node_degrees[j] += abs(weight)

    edges = sorted(edges, key=lambda x: abs(x['weight']))[-top_n:] if top_n else edges

    # 2. Figure Setup
    fig = go.Figure()

    # A. Brain Mesh
    fsaverage = _get_fsaverage_mesh()
    mesh = fsaverage.pial
    verts = np.vstack([mesh.parts['left'].coordinates, mesh.parts['right'].coordinates])
    faces = np.vstack([mesh.parts['left'].faces, mesh.parts['right'].faces + len(mesh.parts['left'].coordinates)])
    
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
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

        # Colorbar: Only add if show_colorbar=True
        if show_colorbar:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(
                    colorscale=edge_cmap, cmin=0, cmax=max_w, showscale=True,
                    colorbar=dict(
                        title=dict(text='Connection Strength', side='top'),
                        thickness=25,
                        len=0.4,
                        x=0.85,        # ← PERFECT ALIGNMENT: matches legend x
                        y=0.25,        # ← BOTTOM RIGHT position
                        xanchor='center',
                        yanchor='bottom',
                        orientation='v',
                        tickfont=dict(family=STYLE['font_family'], size=11, color=STYLE['font_color']),
                        outlinewidth=0
                    )
                ),
                showlegend=False
            ))

    # C. Nodes & Legend
    # Get node colors based on user cmap choice
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


    # 3. Layout: Professional Right-Column Alignment
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            y=0.90,  # ↑ Improved vertical spacing (was 0.96)
            x=0.5,
            xanchor='center',
            font=dict(size=32, family=STYLE['font_family'], color=STYLE['font_color'])  # ↑ Larger font
        ),
        width=1200,
        height=800,
        paper_bgcolor=STYLE['bg_color'],
        scene=dict(
            domain=dict(x=[0.0, 0.70], y=[0.1, 0.9]),  # ← BRAIN: LEFT 70%, VERTICALLY CENTERED
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=0, y=1.8, z=0), up=dict(x=0, y=0, z=1)),  # ← TRUE FRONTAL VIEW (was y=-1.5)
            bgcolor=STYLE['bg_color']
        ),
        legend=dict(
            yanchor="top",
            y=0.85,           # ← TOP OF RIGHT COLUMN
            xanchor="center",
            x=0.85,           # ← PERFECT ALIGNMENT: matches colorbar x
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="#e5e5e5",
            borderwidth=1,
            font=dict(size=12, family=STYLE['font_family'], color=STYLE['font_color']),
            itemsizing='constant',
            tracegroupgap=5,
        ) if show_legend else None,
        margin=dict(t=80, b=50, l=0, r=0),  # ← NO MARGINS
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
    
    return fig

'''
# Usage Example

from gt_map import GlasserTianParcellator 
parc = GlasserTianParcellator()

# Create a Connectivity Matrix
dummy_matrix = np.random.rand(414, 414)

fig = plot_roi_3d_connectivity(
    indices=[10, 50, 150, 250, 380],
    matrix=dummy_matrix,
    parcellator=parc,
    edge_cmap='Purples',        
    node_cmap='Pastel1',     
    show_labels_on_map=True,  # See short names on map
    show_legend=True,         # See full names on legend
    show_colorbar=True,       # ← NEW: Set to False to hide colorbar
    top_n=10,
    title="BIO MARKER ANALYSIS",
)
fig.show()

'''
