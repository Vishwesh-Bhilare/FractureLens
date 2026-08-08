"""HTML report context construction and rendering."""

import base64
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fracturelens.core.io import CATEGORY_DISPLAY_NAMES, decode_label, get_case_paths, load_volume
from fracturelens.core.geometry import FragmentMesh
from fracturelens.core.metrics import CaseReport
from fracturelens.core.render2d import render_axial_slice_png
from fracturelens.core.render3d import build_fractured_bones_figure


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def build_report_context(case_report: CaseReport, root: Path, mesh_smooth_iterations: int = 6, fragment_meshes: dict[int, FragmentMesh] | None = None) -> dict[str, Any]:
    """Assemble CaseReport fields plus embedded image data for the HTML template."""
    image_path, label_path = get_case_paths(root, case_report.case_id)
    image_vol, _ = load_volume(image_path)
    label_vol, spacing = load_volume(label_path)
    by_bone = []
    slice_images = []
    fractured_ids = []
    for cid, name in CATEGORY_DISPLAY_NAMES.items():
        frags = [f for f in case_report.fragments if f.category_id == cid]
        by_bone.append({"category_id": cid, "name": name, "fragments": frags, "fragment_count": len(frags)})
        if len(frags) > 1:
            fractured_ids.append(cid)
            labels = [f.label for f in frags]
            mask = np.isin(label_vol, labels)
            counts = mask.sum(axis=(1, 2))
            z = int(counts.argmax())
            slice_images.append({"bone_name": name, "z_index": z, "data_uri": _data_uri(render_axial_slice_png(image_vol, label_vol, z))})
    render3d_note = None
    render3d_html = None
    if fractured_ids:
        try:
            fig = build_fractured_bones_figure(label_vol, spacing, fractured_ids, mesh_smooth_iterations, fragment_meshes)
            fig.update_layout(height=700)
            # include_plotlyjs=True inlines plotly.js for offline portability; use CDN only if report size becomes a problem.
            render3d_html = fig.to_html(
                full_html=False,
                include_plotlyjs=True,
                div_id="fracture-3d-render",
            )
        except Exception as exc:
            render3d_note = f"3D render skipped: {exc}"
    return {"report": case_report, "by_bone": by_bone, "slice_images": slice_images, "render3d_html": render3d_html, "render3d_note": render3d_note, "dataset_name": "PENGWIN CT", "mesh_smooth_iterations": mesh_smooth_iterations}


def write_html_report(case_report: CaseReport, root: Path, output_dir: Path, mesh_smooth_iterations: int = 6, fragment_meshes: dict[int, FragmentMesh] | None = None) -> Path:
    """Render and write the self-contained case HTML report."""
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape())
    html = env.get_template("case_report.html.j2").render(**build_report_context(case_report, root, mesh_smooth_iterations, fragment_meshes))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{case_report.case_id}_report.html"
    out.write_text(html, encoding="utf-8")
    return out
