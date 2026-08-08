import json, subprocess, sys

def test_metrics_cli_smoke(synthetic_root):
    proc = subprocess.run([sys.executable, "-m", "fracturelens.cli", "metrics", "001", "--root", str(synthetic_root), "--no-cache"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["case_id"] == "001"
    assert data["total_fragment_count"] == 3
