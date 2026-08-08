"""FastAPI app for the local FractureLens web interface."""

import html
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fracturelens.core.io import CATEGORY_DISPLAY_NAMES, DEFAULT_ROOT, get_case_paths, list_available_case_ids, load_volume
from fracturelens.core.metrics import QuickCaseSummary, quick_case_summary
from fracturelens.core.render2d import render_slice_png
from fracturelens.core.render3d import build_fractured_bones_figure
from fracturelens.core.report_cache import CACHE_ROOT, get_report
from fracturelens.report.builder import write_html_report

WEB_ROOT = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"

app = FastAPI(title="FractureLens")
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")

templates = Environment(loader=FileSystemLoader(WEB_ROOT / "templates"), autoescape=select_autoescape())


def _root(request: Request) -> Path:
    return Path(getattr(request.app.state, "root", DEFAULT_ROOT))


def _mesh_smooth(request: Request) -> int:
    return int(getattr(request.app.state, "mesh_smooth", 6))


def _output_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "output_dir", DEFAULT_OUTPUT_DIR))


def _render(template_name: str, **context: Any) -> str:
    return templates.get_template(template_name).render(**context)


def _summary_from_cache_or_labels(case_id: str, root: Path) -> QuickCaseSummary:
    cache = CACHE_ROOT / f"{case_id}_metrics.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return QuickCaseSummary(
            case_id=case_id,
            fractured_bones=list(data.get("fractured_bones", [])),
            intact_bones=list(data.get("intact_bones", [])),
            missing_bones=list(data.get("missing_bones", [])),
            total_fragment_count=int(data.get("total_fragment_count", 0)),
        )
    return quick_case_summary(case_id, root)


def _shape_context(root: Path, case_id: str) -> dict[str, Any]:
    _, label_path = get_case_paths(root, case_id)
    label_vol, _ = load_volume(label_path)
    shape = tuple(int(v) for v in label_vol.shape)
    max_indices = {0: shape[0] - 1, 1: shape[1] - 1, 2: shape[2] - 1}
    middle_indices = {axis: max_index // 2 for axis, max_index in max_indices.items()}
    return {"shape": shape, "max_indices": max_indices, "middle_indices": middle_indices}


@app.get("/", response_class=HTMLResponse)
def case_list(request: Request) -> HTMLResponse:
    root = _root(request)
    cases = [_summary_from_cache_or_labels(case_id, root) for case_id in list_available_case_ids(root)]
    return HTMLResponse(_render("case_list.html.j2", request=request, cases=cases, root=root))


@app.get("/case/{case_id}", response_class=HTMLResponse)
def case_detail(case_id: str, request: Request) -> HTMLResponse:
    root = _root(request)
    context = _shape_context(root, case_id)
    return HTMLResponse(_render("case_detail.html.j2", request=request, case_id=case_id, **context))


@app.get("/case/{case_id}/slice")
def case_slice(case_id: str, request: Request, axis: int = 0, index: int = 0) -> Response:
    if axis not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="axis must be 0 (axial), 1 (coronal), or 2 (sagittal)")
    root = _root(request)
    image_path, label_path = get_case_paths(root, case_id)
    image_vol, _ = load_volume(image_path)
    label_vol, _ = load_volume(label_path)
    if index < 0 or index >= label_vol.shape[axis]:
        raise HTTPException(status_code=400, detail=f"index must be between 0 and {label_vol.shape[axis] - 1} for axis {axis}")
    return Response(content=render_slice_png(image_vol, label_vol, axis, index), media_type="image/png")


@app.get("/case/{case_id}/metrics-panel", response_class=HTMLResponse)
def metrics_panel(case_id: str, request: Request, no_cache: bool = False) -> HTMLResponse:
    root = _root(request)
    mesh_smooth = _mesh_smooth(request)
    try:
        case_report, fragment_meshes = get_report(case_id, root, mesh_smooth, no_cache, include_meshes=True)
        render3d_html = None
        if case_report.fractured_bones:
            _, label_path = get_case_paths(root, case_id)
            label_vol, spacing = load_volume(label_path)
            fractured_ids = [cid for cid, name in CATEGORY_DISPLAY_NAMES.items() if name in case_report.fractured_bones]
            fig = build_fractured_bones_figure(label_vol, spacing, fractured_ids, mesh_smooth, fragment_meshes)
            fig.update_layout(height=700)
            # Inline plotly.js deliberately for offline local reports/panels; CDN would be smaller but network-dependent.
            render3d_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id=f"fracture-3d-render-{case_id}")
        return HTMLResponse(_render(
            "metrics_panel.html.j2",
            report=case_report,
            categories=CATEGORY_DISPLAY_NAMES,
            render3d_html=render3d_html,
            mesh_smooth_iterations=mesh_smooth,
        ))
    except Exception as exc:
        return HTMLResponse(f'<div class="error"><strong>Unable to compute metrics:</strong> {html.escape(str(exc))}</div>')


@app.get("/case/{case_id}/report")
def download_report(case_id: str, request: Request) -> FileResponse:
    root = _root(request)
    mesh_smooth = _mesh_smooth(request)
    case_report, fragment_meshes = get_report(case_id, root, mesh_smooth, False, include_meshes=True)
    path = write_html_report(case_report, root, _output_dir(request), mesh_smooth, fragment_meshes)
    return FileResponse(path, media_type="text/html", filename=f"{case_id}_report.html")
