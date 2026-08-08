"""Headless Plotly 3D rendering helpers for PENGWIN fragments."""

import numpy as np
import plotly.graph_objects as go

from fracturelens.core.geometry import MIN_VOXELS_FOR_MESH, FragmentMesh, build_fragment_mesh, fracture_proximity_intensity, fragment_bounding_box
from fracturelens.core.io import CATEGORY_DISPLAY_NAMES, decode_label

CATEGORY_BASE_COLOR = {1: (200, 50, 50), 2: (50, 110, 210), 3: (50, 170, 75)}
FRACTURE_HIGHLIGHT_COLOR = (255, 210, 30)


def make_colorscale(base_rgb: tuple[int, int, int]) -> list[list[float | str]]:
    """Build the fracture-highlight-to-base Plotly colorscale."""
    hi = FRACTURE_HIGHLIGHT_COLOR
    return [[0.0, f"rgb({hi[0]},{hi[1]},{hi[2]})"], [1.0, f"rgb({base_rgb[0]},{base_rgb[1]},{base_rgb[2]})"]]


def build_fractured_bones_figure(label_vol: np.ndarray, spacing_xyz: tuple[float, float, float], fractured_category_ids: list[int], mesh_smooth_iterations: int = 6, fragment_meshes: dict[int, FragmentMesh] | None = None) -> go.Figure:
    """Build a static Plotly figure for fractured bones only."""
    vz, vy, vx = spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]
    voxel_spacing = (vz, vy, vx); voxel_volume = vz * vy * vx
    traces = []
    for cid in fractured_category_ids:
        labels = sorted(int(v) for v in np.unique(label_vol) if int(v) and decode_label(int(v))[0] == cid)
        if len(labels) <= 1:
            continue
        bbox = fragment_bounding_box(np.isin(label_vol, labels)); crop = label_vol[bbox]
        for label in labels:
            mask = crop == label
            if int(mask.sum()) < MIN_VOXELS_FOR_MESH:
                continue
            mesh = fragment_meshes.get(label) if fragment_meshes is not None else None
            if mesh is None:
                mesh = build_fragment_mesh(mask, voxel_volume, mesh_smooth_iterations)
            intensity = fracture_proximity_intensity(mesh.verts, crop, label, labels, voxel_spacing)
            verts = mesh.verts.copy()
            verts[:, 0] = (mesh.verts[:, 0] + bbox[0].start) * vz
            verts[:, 1] = (mesh.verts[:, 1] + bbox[1].start) * vy
            verts[:, 2] = (mesh.verts[:, 2] + bbox[2].start) * vx
            frag_id = decode_label(label)[1]
            traces.append(go.Mesh3d(x=verts[:, 2], y=verts[:, 1], z=verts[:, 0], i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2], intensity=intensity, intensitymode="vertex", colorscale=make_colorscale(CATEGORY_BASE_COLOR.get(cid, (150,150,150))), cmin=0, cmax=1, showscale=False, name=f"{CATEGORY_DISPLAY_NAMES[cid]} - fragment {frag_id}", flatshading=False))
    fig = go.Figure(data=traces)
    fig.update_layout(title="3D fractured-bone render", scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)", aspectmode="data"), margin=dict(l=0,r=0,t=40,b=0))
    return fig
