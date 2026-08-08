"""Command-line interface for FractureLens Phase 0+1 reports and metrics."""

import argparse, json, sys, time
from pathlib import Path

from fracturelens.core.io import DEFAULT_ROOT, get_case_paths, list_available_case_ids
from fracturelens.core.metrics import case_report_from_dict, case_report_to_dict, compute_case_report
from fracturelens.report.builder import write_html_report

CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def _get_report(case_id: str, root: Path, mesh_smooth: int, no_cache: bool):
    cache = CACHE_ROOT / f"{case_id}_metrics.json"
    _, label_path = get_case_paths(root, case_id)
    if not no_cache and cache.exists() and label_path.exists() and label_path.stat().st_mtime < cache.stat().st_mtime:
        data = json.loads(cache.read_text())
        if data.get("mesh_smooth_iterations") == mesh_smooth:
            return case_report_from_dict(data)
    report = compute_case_report(case_id, root, mesh_smooth)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(case_report_to_dict(report, mesh_smooth), indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the FractureLens CLI."""
    p = argparse.ArgumentParser(prog="python -m fracturelens.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("report", "metrics"):
        sp = sub.add_parser(name); sp.add_argument("case_id"); sp.add_argument("--root", type=Path, default=DEFAULT_ROOT); sp.add_argument("--no-cache", action="store_true")
        if name == "report":
            sp.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT); sp.add_argument("--mesh-smooth", type=int, default=6)
    bp = sub.add_parser("batch"); bp.add_argument("--root", type=Path, default=DEFAULT_ROOT); bp.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT); bp.add_argument("--mesh-smooth", type=int, default=6); bp.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "metrics":
        report = _get_report(args.case_id, args.root, 6, args.no_cache)
        print(json.dumps(case_report_to_dict(report), indent=2)); return 0
    if args.cmd == "report":
        report = _get_report(args.case_id, args.root, args.mesh_smooth, args.no_cache)
        print(write_html_report(report, args.root, args.output_dir, args.mesh_smooth)); return 0
    failed=[]; ids=list_available_case_ids(args.root)
    for i, cid in enumerate(ids, 1):
        start=time.perf_counter()
        try:
            report=_get_report(cid,args.root,args.mesh_smooth,args.no_cache); write_html_report(report,args.root,args.output_dir,args.mesh_smooth)
            print(f"[{i}/{len(ids)}] case {cid}: done ({time.perf_counter()-start:.1f}s)")
        except Exception as exc:
            failed.append(cid); print(f"[{i}/{len(ids)}] case {cid}: failed ({exc})", file=sys.stderr)
    if failed: print("Failed case_ids: " + ", ".join(failed), file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
