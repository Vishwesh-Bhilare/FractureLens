"""Headless Plotly 3D rendering helpers for PENGWIN fragments."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from fracturelens.core.geometry import MIN_VOXELS_FOR_MESH, FragmentMesh, build_fragment_mesh, fracture_proximity_intensity, fragment_bounding_box
from fracturelens.core.io import CATEGORY_DISPLAY_NAMES, decode_label

CATEGORY_BASE_COLOR = {1: (200, 50, 50), 2: (50, 110, 210), 3: (50, 170, 75)}
FRACTURE_HIGHLIGHT_COLOR = (255, 210, 30)
CATEGORY_COLOR_HEX = {cid: f"#{r:02x}{g:02x}{b:02x}" for cid, (r, g, b) in CATEGORY_BASE_COLOR.items()}
FRACTURE_HIGHLIGHT_COLOR_HEX = f"#{FRACTURE_HIGHLIGHT_COLOR[0]:02x}{FRACTURE_HIGHLIGHT_COLOR[1]:02x}{FRACTURE_HIGHLIGHT_COLOR[2]:02x}"


@dataclass(frozen=True)
class TraceMeta:
    """Metadata used to build Plotly button visibility masks."""
    position: str
    is_fractured_bone: bool


def make_colorscale(base_rgb: tuple[int, int, int]) -> list[list[float | str]]:
    """Build the fracture-highlight-to-base Plotly colorscale."""
    hi = FRACTURE_HIGHLIGHT_COLOR
    return [[0.0, f"rgb({hi[0]},{hi[1]},{hi[2]})"], [1.0, f"rgb({base_rgb[0]},{base_rgb[1]},{base_rgb[2]})"]]


def build_color_key() -> list[dict[str, str]]:
    """Return the static HTML color key for 3D bone renders."""
    return [
        {"name": CATEGORY_DISPLAY_NAMES[cid], "hex": CATEGORY_COLOR_HEX[cid]}
        for cid in CATEGORY_DISPLAY_NAMES
        if cid in CATEGORY_COLOR_HEX
    ] + [{"name": "Fracture surface", "hex": FRACTURE_HIGHLIGHT_COLOR_HEX}]


def build_case_figure(label_vol: np.ndarray, spacing_xyz: tuple[float, float, float], mesh_smooth_iterations: int = 6, fragment_meshes: dict[int, FragmentMesh] | None = None, explode_mm: float = 15.0) -> go.Figure:
    """Build an interactive Plotly figure for every labeled bone fragment in a case."""
    vz, vy, vx = spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]
    voxel_spacing = (vz, vy, vx)
    voxel_volume = vz * vy * vx
    traces: list[go.Mesh3d] = []
    trace_meta: list[TraceMeta] = []
    seen_legend_groups: set[tuple[int, str]] = set()

    for cid in CATEGORY_DISPLAY_NAMES:
        labels = sorted(int(v) for v in np.unique(label_vol) if int(v) and decode_label(int(v))[0] == cid)
        if not labels:
            continue

        cat_mask = np.isin(label_vol, labels)
        bbox = fragment_bounding_box(cat_mask)
        crop = label_vol[bbox]
        cat_centroid = np.argwhere(cat_mask[bbox]).mean(axis=0)
        is_fractured_bone = len(labels) > 1
        category_name = CATEGORY_DISPLAY_NAMES[cid]
        colorscale = make_colorscale(CATEGORY_BASE_COLOR.get(cid, (150, 150, 150)))

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

            frag_centroid = np.argwhere(mask).mean(axis=0)
            direction = frag_centroid - cat_centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-6 and is_fractured_bone:
                direction = direction / norm
            else:
                direction = np.zeros(3)
            offset_mm = direction * explode_mm
            frag_id = decode_label(label)[1]

            for position, offset, initially_visible in (
                ("assembled", np.zeros(3), True),
                ("exploded", offset_mm, False),
            ):
                verts_positioned = verts.copy()
                verts_positioned[:, 0] += offset[0]
                verts_positioned[:, 1] += offset[1]
                verts_positioned[:, 2] += offset[2]
                legend_key = (cid, position)
                showlegend = legend_key not in seen_legend_groups
                seen_legend_groups.add(legend_key)
                traces.append(go.Mesh3d(
                    x=verts_positioned[:, 2], y=verts_positioned[:, 1], z=verts_positioned[:, 0],
                    i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                    intensity=intensity, intensitymode="vertex", colorscale=colorscale,
                    cmin=0, cmax=1, showscale=False, name=category_name,
                    legendgroup=f"{category_name}-{position}", showlegend=showlegend,
                    visible=initially_visible, flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.75, specular=0.4, roughness=0.55, fresnel=0.15),
                    lightposition=dict(x=100, y=200, z=300),
                    hovertemplate=f"{category_name} - fragment {frag_id}<extra></extra>",
                ))
                trace_meta.append(TraceMeta(position=position, is_fractured_bone=is_fractured_bone))

    visible_all_assembled = [t.position == "assembled" for t in trace_meta]
    visible_all_exploded = [t.position == "exploded" for t in trace_meta]
    visible_fractured_assembled = [t.position == "assembled" and t.is_fractured_bone for t in trace_meta]
    visible_fractured_exploded = [t.position == "exploded" and t.is_fractured_bone for t in trace_meta]

    has_fractured_bone = any(t.is_fractured_bone for t in trace_meta)
    if has_fractured_bone:
        buttons = [
            dict(label="All · Assembled", method="restyle", args=[{"visible": visible_all_assembled}]),
            dict(label="All · Exploded", method="restyle", args=[{"visible": visible_all_exploded}]),
            dict(label="Fractured · Assembled", method="restyle", args=[{"visible": visible_fractured_assembled}]),
            dict(label="Fractured · Exploded", method="restyle", args=[{"visible": visible_fractured_exploded}]),
        ]
    else:
        buttons = [
            dict(label="Assembled", method="restyle", args=[{"visible": visible_all_assembled}]),
            dict(label="Exploded", method="restyle", args=[{"visible": visible_all_exploded}]),
        ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D bone-fragment render",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)", aspectmode="data", bgcolor="rgb(245,245,248)"),
        paper_bgcolor="white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(type="buttons", direction="right", x=0.02, y=1.08, xanchor="left", yanchor="top", buttons=buttons)],
    )
    return fig
