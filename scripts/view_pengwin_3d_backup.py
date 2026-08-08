"""
3D viewer for PENGWIN pelvic bone fragments, with fracture-surface highlighting.

For each fragment, builds a 3D surface mesh (marching cubes on a lightly
smoothed volume, for a cleaner surface than raw voxel blockiness) and colors
it on a gradient: bright orange/yellow exactly where the surface is close to
a DIFFERENT fragment of the same bone (i.e. the actual fracture line), fading
to the bone's normal color away from the break. A button lets you toggle
between the assembled pelvis and an "exploded" view that pulls fragments
apart so gaps are unambiguous.

Usage (run from anywhere -- default --root resolves relative to this file):
    python view_pengwin_3d.py 009
    python view_pengwin_3d.py 009 --downsample 1           # full resolution
    python view_pengwin_3d.py 009 --explode 20              # bigger explode gap (mm)
    python view_pengwin_3d.py 009 --smooth 0                # disable smoothing, raw voxel look
    python view_pengwin_3d.py 009 --root /custom/path/to/data

Requires:
    pip install SimpleITK scikit-image plotly numpy scipy
    pip install kaleido   # optional, only needed for --png export
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt, find_objects, gaussian_filter
import plotly.graph_objects as go

# scripts/view_pengwin_3d.py -> project_root/data
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"

CATEGORIES = {1: "SA (sacrum)", 2: "LI (left hip)", 3: "RI (right hip)"}
CATEGORY_BASE_COLOR = {
    1: (200, 50, 50),    # sacrum -> red family
    2: (50, 110, 210),   # left hip -> blue family
    3: (50, 170, 75),    # right hip -> green family
}
FRACTURE_HIGHLIGHT_COLOR = (255, 210, 30)  # bright orange/yellow
FRACTURE_HIGHLIGHT_RANGE_MM = 12.0  # distance at which highlight fully fades out


def decode_label(label: int) -> tuple[int, int]:
    category_id = (label - 1) // 10 + 1
    fragment_id = (label - 1) % 10 + 1
    return category_id, fragment_id


def load_volume(path: Path) -> tuple[np.ndarray, tuple]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (x, y, z)
    return arr, spacing


def make_colorscale(base_rgb):
    hi = FRACTURE_HIGHLIGHT_COLOR
    return [[0.0, f"rgb({hi[0]},{hi[1]},{hi[2]})"],
            [1.0, f"rgb({base_rgb[0]},{base_rgb[1]},{base_rgb[2]})"]]


def smoothed_mesh(binary_mask: np.ndarray, sigma: float):
    """Marching cubes on a Gaussian-smoothed field instead of the raw binary mask.

    Smoothing the field before thresholding at 0.5 removes the "staircase"
    look inherent to voxel data and gives noticeably higher-quality, more
    anatomically plausible surfaces -- without needing higher-resolution
    input data. sigma=0 skips smoothing (raw voxel look).
    """
    field = binary_mask.astype(np.float32)
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma)
    return marching_cubes(field, level=0.5)


def main():
    parser = argparse.ArgumentParser(description="3D render of PENGWIN bone fragments with fracture highlighting.")
    parser.add_argument("case_id", help="Case id, e.g. 009")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                         help=f"Dataset root folder (default: {DEFAULT_ROOT})")
    parser.add_argument("--downsample", type=int, default=1,
                         help="Downsample factor (1=full res default, 2=half res for speed)")
    parser.add_argument("--explode", type=float, default=15.0,
                         help="How far (mm) fragments move apart in exploded view (default 15)")
    parser.add_argument("--smooth", type=float, default=0.8,
                         help="Gaussian smoothing sigma (voxels) applied before meshing for "
                              "cleaner surfaces; 0 disables smoothing (default 0.8)")
    parser.add_argument("--png", action="store_true",
                         help="Also export a high-res static PNG snapshot (requires `pip install kaleido`)")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    lbl_path = root / "labels" / f"{args.case_id}.mha"
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label not found: {lbl_path}")

    label_vol, spacing = load_volume(lbl_path)
    print(f"Loaded {lbl_path.name}: shape={label_vol.shape}, spacing={spacing}")

    ds = args.downsample
    if ds > 1:
        label_vol = label_vol[::ds, ::ds, ::ds]
    # spacing given as (x, y, z); volume axes are (z, y, x)
    vz, vy, vx = spacing[2] * ds, spacing[1] * ds, spacing[0] * ds
    voxel_spacing = (vz, vy, vx)  # matches array axis order (z, y, x)

    unique_labels = sorted(int(l) for l in np.unique(label_vol) if l != 0)
    if not unique_labels:
        print("No labeled voxels found in this volume.")
        return

    # group fragment labels by parent bone category
    labels_by_category: dict[int, list[int]] = {}
    for label in unique_labels:
        cat_id, _ = decode_label(label)
        labels_by_category.setdefault(cat_id, []).append(label)

    print(f"\nBuilding meshes for {len(unique_labels)} fragment(s) "
          f"(smoothing sigma={args.smooth})...")
    traces_assembled = []
    traces_exploded = []
    seen_categories = set()

    for cat_id, frag_labels in labels_by_category.items():
        cat_name = CATEGORIES.get(cat_id, f"unknown({cat_id})")
        base_color = CATEGORY_BASE_COLOR.get(cat_id, (150, 150, 150))
        cat_mask = np.isin(label_vol, frag_labels)

        # bounding box around this whole bone (small padding), for cheaper distance transforms
        slices = find_objects(cat_mask.astype(int))[0]
        pad = 4
        z0 = max(slices[0].start - pad, 0); z1 = min(slices[0].stop + pad, label_vol.shape[0])
        y0 = max(slices[1].start - pad, 0); y1 = min(slices[1].stop + pad, label_vol.shape[1])
        x0 = max(slices[2].start - pad, 0); x1 = min(slices[2].stop + pad, label_vol.shape[2])
        crop = label_vol[z0:z1, y0:y1, x0:x1]

        # centroid of the whole bone, for explode direction
        cat_coords = np.argwhere(cat_mask[z0:z1, y0:y1, x0:x1])
        cat_centroid = cat_coords.mean(axis=0)  # in cropped voxel space (z,y,x)

        multi_fragment = len(frag_labels) > 1

        for label in frag_labels:
            frag_id = decode_label(label)[1]
            frag_mask = (crop == label)
            if frag_mask.sum() < 20:
                print(f"  skipped label {label} ({cat_name}, fragment {frag_id}) -- too small to mesh")
                continue

            try:
                verts, faces, _, _ = smoothed_mesh(frag_mask, sigma=args.smooth)
            except (RuntimeError, ValueError):
                print(f"  skipped label {label} ({cat_name}, fragment {frag_id}) -- too thin to mesh")
                continue

            # --- fracture-proximity coloring ---
            if multi_fragment:
                other_mask = (crop != label) & (crop != 0) & np.isin(crop, frag_labels)
                if other_mask.any():
                    # sampling=voxel_spacing gives distances in mm directly (handles non-cubic voxels)
                    dist_mm = distance_transform_edt(~other_mask, sampling=voxel_spacing)
                    # sample distance at each vertex, nearest-neighbor
                    idx = np.round(verts).astype(int)
                    idx[:, 0] = np.clip(idx[:, 0], 0, dist_mm.shape[0] - 1)
                    idx[:, 1] = np.clip(idx[:, 1], 0, dist_mm.shape[1] - 1)
                    idx[:, 2] = np.clip(idx[:, 2], 0, dist_mm.shape[2] - 1)
                    d = dist_mm[idx[:, 0], idx[:, 1], idx[:, 2]]
                    intensity = np.clip(d / FRACTURE_HIGHLIGHT_RANGE_MM, 0, 1)
                else:
                    intensity = np.ones(verts.shape[0])
            else:
                intensity = np.ones(verts.shape[0])  # no fracture partner -> uniform bone color

            # convert voxel coords -> mm, add crop offset back to get position within full volume
            verts_mm = verts.copy()
            verts_mm[:, 0] = (verts[:, 0] + z0) * vz
            verts_mm[:, 1] = (verts[:, 1] + y0) * vy
            verts_mm[:, 2] = (verts[:, 2] + x0) * vx

            colorscale = make_colorscale(base_color)
            seen_categories.add(cat_name)
            trace_name = f"{cat_name} - fragment {frag_id}"

            common_kwargs = dict(
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=intensity, intensitymode="vertex",
                colorscale=colorscale, cmin=0, cmax=1, showscale=False,
                name=trace_name, legendgroup=cat_name, showlegend=True,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.75, specular=0.4, roughness=0.55, fresnel=0.15),
                lightposition=dict(x=100, y=200, z=300),
            )

            # assembled position
            traces_assembled.append(go.Mesh3d(
                x=verts_mm[:, 2], y=verts_mm[:, 1], z=verts_mm[:, 0], **common_kwargs
            ))

            # exploded position: push fragment away from bone centroid
            frag_centroid = np.argwhere(frag_mask).mean(axis=0)  # (z,y,x) in crop-voxel space
            direction = frag_centroid - cat_centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-6 and multi_fragment:
                direction = direction / norm
            else:
                direction = np.zeros(3)
            offset_mm = direction * args.explode  # (dz, dy, dx) in mm

            verts_exploded = verts_mm.copy()
            verts_exploded[:, 0] += offset_mm[0]
            verts_exploded[:, 1] += offset_mm[1]
            verts_exploded[:, 2] += offset_mm[2]

            traces_exploded.append(go.Mesh3d(
                x=verts_exploded[:, 2], y=verts_exploded[:, 1], z=verts_exploded[:, 0], **common_kwargs
            ))

            print(f"  fragment {label} ({cat_name}, frag {frag_id}): "
                  f"{verts.shape[0]} vertices, {faces.shape[0]} faces"
                  f"{' [fractured]' if multi_fragment else ' [intact]'}")

    n = len(traces_assembled)
    fig = go.Figure(data=traces_assembled)
    # add exploded traces too (hidden initially), so the button can toggle between them
    for t in traces_exploded:
        t.visible = False
        fig.add_trace(t)

    assembled_visibility = [True] * n + [False] * n
    exploded_visibility = [False] * n + [True] * n

    fig.update_layout(
        title=f"Case {args.case_id} - 3D bone fragments (bright seam = fracture surface)",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            aspectmode="data",
            bgcolor="rgb(245,245,248)",
        ),
        paper_bgcolor="white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.02, y=1.06, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Assembled", method="update", args=[{"visible": assembled_visibility}]),
                dict(label="Exploded", method="update", args=[{"visible": exploded_visibility}]),
            ],
        )],
    )

    out_path = Path(f"pengwin_{args.case_id}_3d.html")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nWrote interactive 3D view to {out_path.resolve()}")

    if args.png:
        png_path = Path(f"pengwin_{args.case_id}_3d.png")
        fig.write_image(str(png_path), width=2400, height=1800, scale=2)
        print(f"Wrote high-res static snapshot to {png_path.resolve()}")

    print("Opening in your default browser...")
    fig.show()


if __name__ == "__main__":
    main()
