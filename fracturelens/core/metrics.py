"""Case- and fragment-level metric computation for PENGWIN labels."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from fracturelens.core.geometry import build_fragment_mesh, fragment_bounding_box
from fracturelens.core.io import CATEGORY_DISPLAY_NAMES, decode_label, get_case_paths, load_volume


@dataclass
class FragmentMetrics:
    """Measurements for one labeled bone fragment."""
    label: int
    category_id: int
    category_name: str
    fragment_id: int
    voxel_count: int
    volume_mm3: float
    surface_area_mm2: float | None
    centroid_mm: tuple[float, float, float]
    bbox_mm: tuple[float, float, float, float, float, float]
    nearest_neighbor_dist_mm: float | None
    nearest_neighbor_label: int | None


@dataclass
class CaseReport:
    """Aggregated report data for one PENGWIN case."""
    case_id: str
    fragments: list[FragmentMetrics]
    fractured_bones: list[str]
    intact_bones: list[str]
    missing_bones: list[str]
    total_fragment_count: int
    severity_score: float
    computed_at: str


def case_report_to_dict(report: CaseReport, mesh_smooth_iterations: int | None = None) -> dict[str, Any]:
    """Serialize a ``CaseReport`` to a JSON-compatible dictionary."""
    data = asdict(report)
    if mesh_smooth_iterations is not None:
        data["mesh_smooth_iterations"] = mesh_smooth_iterations
    return data


def case_report_from_dict(data: dict[str, Any]) -> CaseReport:
    """Deserialize a ``CaseReport`` from a dictionary."""
    fragments = [FragmentMetrics(**frag) for frag in data["fragments"]]
    return CaseReport(
        case_id=data["case_id"], fragments=fragments, fractured_bones=data["fractured_bones"],
        intact_bones=data["intact_bones"], missing_bones=data["missing_bones"],
        total_fragment_count=int(data["total_fragment_count"]), severity_score=float(data["severity_score"]),
        computed_at=data["computed_at"],
    )


def _surface_area_mm2(mask: np.ndarray, voxel_volume_mm3: float, voxel_spacing: tuple[float, float, float], iterations: int) -> float | None:
    try:
        mesh = build_fragment_mesh(mask, voxel_volume_mm3, iterations)
    except (RuntimeError, ValueError):
        return None
    try:
        import trimesh
        # Mesh vertices are (z,y,x) voxel indices. Scale each axis before area because
        # anisotropic spacing makes naive scalar area conversion incorrect.
        mesh_mm = trimesh.Trimesh(vertices=mesh.verts * np.asarray(voxel_spacing), faces=mesh.faces, process=False)
        return float(mesh_mm.area)
    except Exception:
        return None


def compute_case_report(case_id: str, root: Path, mesh_smooth_iterations: int = 6) -> CaseReport:
    """Load one label volume, compute fragment metrics, and aggregate a case report."""
    _, label_path = get_case_paths(root, case_id)
    label_vol, spacing_xyz = load_volume(label_path)
    voxel_spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    unique_labels = sorted(int(v) for v in np.unique(label_vol) if int(v) != 0)
    labels_by_cat: dict[int, list[int]] = {cid: [] for cid in CATEGORY_DISPLAY_NAMES}
    for label in unique_labels:
        cid, _ = decode_label(label)
        labels_by_cat.setdefault(cid, []).append(label)

    fragments: list[FragmentMetrics] = []
    for cid in sorted(CATEGORY_DISPLAY_NAMES):
        frag_labels = labels_by_cat.get(cid, [])
        if not frag_labels:
            continue
        bbox = fragment_bounding_box(np.isin(label_vol, frag_labels))
        crop = label_vol[bbox]
        for label in frag_labels:
            mask_full = label_vol == label
            mask_crop = crop == label
            coords = np.argwhere(mask_full)
            mins = coords.min(axis=0); maxs = coords.max(axis=0) + 1
            centroid_zyx = coords.mean(axis=0)
            nearest_dist = None; nearest_label = None
            if len(frag_labels) > 1:
                other = (crop != label) & (crop != 0) & np.isin(crop, frag_labels)
                dist_mm = distance_transform_edt(~other, sampling=voxel_spacing)
                nearest_dist = float(dist_mm[mask_crop].min())
                best = None
                for sibling in frag_labels:
                    if sibling == label:
                        continue
                    d = distance_transform_edt(crop != sibling, sampling=voxel_spacing)
                    val = float(d[mask_crop].min())
                    if best is None or val < best[0]:
                        best = (val, sibling)
                nearest_label = int(best[1]) if best else None
            voxel_count = int(mask_full.sum())
            _, frag_id = decode_label(label)
            fragments.append(FragmentMetrics(
                label=label, category_id=cid, category_name=CATEGORY_DISPLAY_NAMES.get(cid, f"Unknown ({cid})"),
                fragment_id=frag_id, voxel_count=voxel_count, volume_mm3=float(voxel_count * voxel_volume_mm3),
                surface_area_mm2=_surface_area_mm2(mask_crop, voxel_volume_mm3, voxel_spacing, mesh_smooth_iterations),
                centroid_mm=(float(centroid_zyx[2] * spacing_xyz[0]), float(centroid_zyx[1] * spacing_xyz[1]), float(centroid_zyx[0] * spacing_xyz[2])),
                bbox_mm=(float(mins[2]*spacing_xyz[0]), float(mins[1]*spacing_xyz[1]), float(mins[0]*spacing_xyz[2]), float(maxs[2]*spacing_xyz[0]), float(maxs[1]*spacing_xyz[1]), float(maxs[0]*spacing_xyz[2])),
                nearest_neighbor_dist_mm=nearest_dist, nearest_neighbor_label=nearest_label,
            ))
    counts = {cid: len(labels_by_cat.get(cid, [])) for cid in CATEGORY_DISPLAY_NAMES}
    fractured = [CATEGORY_DISPLAY_NAMES[cid] for cid, count in counts.items() if count > 1]
    intact = [CATEGORY_DISPLAY_NAMES[cid] for cid, count in counts.items() if count == 1]
    missing = [CATEGORY_DISPLAY_NAMES[cid] for cid, count in counts.items() if count == 0]
    excess = sum(max(0, count - 1) for count in counts.values() if count >= 1)
    fractured_fragments = [f for f in fragments if f.nearest_neighbor_dist_mm is not None]
    avg_gap = mean(f.nearest_neighbor_dist_mm for f in fractured_fragments) if fractured_fragments else 0.0
    # v1 illustrative formula, not a clinical severity measure -- see PROJECT_SCOPE.md §10.
    severity = excess * 10.0 + avg_gap
    return CaseReport(case_id, fragments, fractured, intact, missing, len(fragments), float(severity), datetime.now(timezone.utc).isoformat())
