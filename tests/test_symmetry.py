import numpy as np

from fracturelens.core.io import get_case_paths, load_volume
from fracturelens.core.metrics import compute_case_report_with_meshes
from fracturelens.core.symmetry import compute_symmetry_displacement, find_midline_x_mm


def test_find_midline_prefers_sacrum_centroid(synthetic_symmetric_root):
    _, label_path = get_case_paths(synthetic_symmetric_root, "002")
    label_vol, spacing = load_volume(label_path)
    assert find_midline_x_mm(label_vol, spacing) == np.float64(31.5)


def test_compute_symmetry_displacement_for_one_sided_hip_fracture(synthetic_symmetric_root):
    report, meshes = compute_case_report_with_meshes("002", synthetic_symmetric_root, mesh_smooth_iterations=0)
    _, label_path = get_case_paths(synthetic_symmetric_root, "002")
    label_vol, spacing = load_volume(label_path)
    labels_by_cat = {cid: [f.label for f in report.fragments if f.category_id == cid] for cid in (1, 2, 3)}
    displacement = compute_symmetry_displacement(label_vol, spacing, meshes, labels_by_cat)
    assert displacement is not None
    assert set(displacement) == {21, 22}
    assert all(value >= 0 for value in displacement.values())


def test_case_report_uses_symmetry_when_applicable(synthetic_symmetric_root):
    report, _ = compute_case_report_with_meshes("002", synthetic_symmetric_root, mesh_smooth_iterations=0)
    assert report.severity_method == "symmetry"
    assert {f.label for f in report.fragments if f.symmetry_displacement_mm is not None} == {21, 22}


def test_original_synthetic_fixture_falls_back_to_fragment_count(synthetic_root):
    report, _ = compute_case_report_with_meshes("001", synthetic_root, mesh_smooth_iterations=0)
    assert report.severity_method == "fragment_count"
