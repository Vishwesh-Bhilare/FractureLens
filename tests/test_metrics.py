from fracturelens.core.metrics import compute_case_report

def test_synthetic_metrics(synthetic_root):
    report = compute_case_report("001", synthetic_root, mesh_smooth_iterations=0)
    by_label = {f.label: f for f in report.fragments}
    assert by_label[21].voxel_count == 1000
    assert by_label[21].volume_mm3 == by_label[21].voxel_count * 1.0
    assert by_label[1].nearest_neighbor_dist_mm is None
    assert 0 <= by_label[21].nearest_neighbor_dist_mm <= 1.0
    assert 0 <= by_label[22].nearest_neighbor_dist_mm <= 1.0
