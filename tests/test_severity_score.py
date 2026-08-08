from fracturelens.core.metrics import compute_case_report

def test_severity_score_formula(synthetic_root):
    report = compute_case_report("001", synthetic_root, mesh_smooth_iterations=0)
    gaps = [f.nearest_neighbor_dist_mm for f in report.fragments if f.nearest_neighbor_dist_mm is not None]
    assert report.severity_score == 10.0 + sum(gaps) / len(gaps)
