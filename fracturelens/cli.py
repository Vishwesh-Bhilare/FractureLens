"""Command-line interface for FractureLens Phase 0+1 reports and metrics."""

import argparse, json, sys, threading, time, webbrowser
from pathlib import Path

import uvicorn

from fracturelens.core.io import CATEGORIES, DEFAULT_ROOT, list_available_case_ids
from fracturelens.core.metrics import case_report_to_dict
from fracturelens.core.stl_export import export_bone_stl, export_fragment_stl
from fracturelens.core.report_cache import get_report
from fracturelens.report.builder import write_html_report

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def export_case_stls(case_id: str, root: Path, output_dir: Path, mesh_smooth: int = 6, no_cache: bool = False) -> Path:
    """Export per-fragment and per-bone STL files for a case report."""
    report, fragment_meshes = get_report(case_id, root, mesh_smooth, no_cache, include_meshes=True)
    if fragment_meshes is None:
        report, fragment_meshes = get_report(case_id, root, mesh_smooth, True, include_meshes=True)
    stl_dir = output_dir / f"{case_id}_stl"
    voxel_spacing = (1.0, 1.0, 1.0)
    from fracturelens.core.io import get_case_paths, load_volume
    _, label_path = get_case_paths(root, case_id)
    _, spacing_xyz = load_volume(label_path)
    voxel_spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    by_cat: dict[int, list] = {}
    for frag in report.fragments:
        mesh = fragment_meshes.get(frag.label)
        if mesh is None:
            continue
        code = CATEGORIES.get(frag.category_id, f"C{frag.category_id}")
        export_fragment_stl(mesh, voxel_spacing, stl_dir / f"{case_id}_{code}_frag{frag.fragment_id}.stl")
        by_cat.setdefault(frag.category_id, []).append(mesh)
    for cid, meshes in by_cat.items():
        code = CATEGORIES.get(cid, f"C{cid}")
        export_bone_stl(meshes, voxel_spacing, stl_dir / f"{case_id}_{code}_full.stl")
    return stl_dir


def main(argv: list[str] | None = None) -> int:
    """Run the FractureLens CLI."""
    p = argparse.ArgumentParser(prog="python -m fracturelens.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("report", "metrics"):
        sp = sub.add_parser(name); sp.add_argument("case_id"); sp.add_argument("--root", type=Path, default=DEFAULT_ROOT); sp.add_argument("--no-cache", action="store_true")
        sp.add_argument("--mesh-smooth", type=int, default=6)
        if name == "report":
            sp.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    bp = sub.add_parser("batch"); bp.add_argument("--root", type=Path, default=DEFAULT_ROOT); bp.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT); bp.add_argument("--mesh-smooth", type=int, default=6); bp.add_argument("--no-cache", action="store_true")
    ep = sub.add_parser("export-stl"); ep.add_argument("case_id"); ep.add_argument("--root", type=Path, default=DEFAULT_ROOT); ep.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT); ep.add_argument("--mesh-smooth", type=int, default=6); ep.add_argument("--no-cache", action="store_true")
    sp = sub.add_parser("serve"); sp.add_argument("--root", type=Path, default=DEFAULT_ROOT); sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=8765); sp.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "metrics":
        report = get_report(args.case_id, args.root, args.mesh_smooth, args.no_cache)
        print(json.dumps(case_report_to_dict(report), indent=2)); return 0
    if args.cmd == "report":
        report, fragment_meshes = get_report(args.case_id, args.root, args.mesh_smooth, args.no_cache, include_meshes=True)
        print(write_html_report(report, args.root, args.output_dir, args.mesh_smooth, fragment_meshes)); return 0
    if args.cmd == "export-stl":
        print(export_case_stls(args.case_id, args.root, args.output_dir, args.mesh_smooth, args.no_cache)); return 0
    if args.cmd == "serve":
        from fracturelens.web.app import app
        app.state.root = args.root
        app.state.mesh_smooth = 6
        app.state.output_dir = OUTPUT_ROOT
        url = f"http://{args.host}:{args.port}/"
        if not args.no_browser:
            # Delay slightly so uvicorn has time to bind before the browser loads.
            threading.Timer(0.75, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    failed=[]; ids=list_available_case_ids(args.root)
    for i, cid in enumerate(ids, 1):
        start=time.perf_counter()
        try:
            report, fragment_meshes = get_report(cid,args.root,args.mesh_smooth,args.no_cache,include_meshes=True); write_html_report(report,args.root,args.output_dir,args.mesh_smooth,fragment_meshes)
            print(f"[{i}/{len(ids)}] case {cid}: done ({time.perf_counter()-start:.1f}s)")
        except Exception as exc:
            failed.append(cid); print(f"[{i}/{len(ids)}] case {cid}: failed ({exc})", file=sys.stderr)
    if failed: print("Failed case_ids: " + ", ".join(failed), file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
