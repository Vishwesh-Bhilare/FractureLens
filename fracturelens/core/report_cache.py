"""Shared report cache helpers for CLI and web views."""

import json
from pathlib import Path

from fracturelens.core.io import get_case_paths
from fracturelens.core.metrics import (
    CaseReport,
    case_report_from_dict,
    case_report_to_dict,
    compute_case_report,
    compute_case_report_with_meshes,
)
from fracturelens.core.geometry import FragmentMesh

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "cache"


def get_report(
    case_id: str,
    root: Path,
    mesh_smooth: int,
    no_cache: bool,
    include_meshes: bool = False,
) -> CaseReport | tuple[CaseReport, dict[int, FragmentMesh] | None]:
    """Load a cached report when valid, otherwise compute and cache it."""
    cache = CACHE_ROOT / f"{case_id}_metrics.json"
    _, label_path = get_case_paths(root, case_id)
    if not no_cache and cache.exists() and label_path.exists() and label_path.stat().st_mtime < cache.stat().st_mtime:
        data = json.loads(cache.read_text())
        if data.get("mesh_smooth_iterations") == mesh_smooth:
            report = case_report_from_dict(data)
            return (report, None) if include_meshes else report
    if include_meshes:
        report, fragment_meshes = compute_case_report_with_meshes(case_id, root, mesh_smooth)
    else:
        report = compute_case_report(case_id, root, mesh_smooth)
        fragment_meshes = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(case_report_to_dict(report, mesh_smooth), indent=2), encoding="utf-8")
    return (report, fragment_meshes) if include_meshes else report
