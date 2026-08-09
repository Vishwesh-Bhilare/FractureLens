import trimesh

from fracturelens.core.metrics import compute_case_report_with_meshes
from fracturelens.core.stl_export import export_fragment_stl


def test_export_fragment_stl_round_trips_as_valid_mesh(synthetic_root, tmp_path):
    _, meshes = compute_case_report_with_meshes("001", synthetic_root, mesh_smooth_iterations=0)
    label = sorted(meshes)[0]
    out = export_fragment_stl(meshes[label], (1.0, 1.0, 1.0), tmp_path / "fragment.stl")
    loaded = trimesh.load(out)
    assert len(loaded.vertices) >= len(meshes[label].verts) * 0.8
    assert len(loaded.faces) >= len(meshes[label].faces) * 0.8
    assert abs(float(loaded.volume)) > 0
