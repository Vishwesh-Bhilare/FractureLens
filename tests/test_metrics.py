from fracturelens.core.metrics import compute_case_report

def test_synthetic_metrics(synthetic_root):
    report = compute_case_report("001", synthetic_root, mesh_smooth_iterations=0)
    by_label = {f.label: f for f in report.fragments}
    assert by_label[21].voxel_count == 1000
    assert by_label[21].volume_mm3 == by_label[21].voxel_count * 1.0
    assert by_label[1].nearest_neighbor_dist_mm is None
    assert 0 <= by_label[21].nearest_neighbor_dist_mm <= 1.0
    assert 0 <= by_label[22].nearest_neighbor_dist_mm <= 1.0


def _reference_nearest_labels(label_vol, frag_labels, voxel_spacing):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    from fracturelens.core.geometry import fragment_bounding_box

    bbox = fragment_bounding_box(np.isin(label_vol, frag_labels))
    crop = label_vol[bbox]
    nearest = {}
    for label in frag_labels:
        mask_crop = crop == label
        best = None
        for sibling in frag_labels:
            if sibling == label:
                continue
            d = distance_transform_edt(crop != sibling, sampling=voxel_spacing)
            val = float(d[mask_crop].min())
            if best is None or val < best[0]:
                best = (val, sibling)
        nearest[label] = int(best[1]) if best else None
    return nearest


def test_nearest_neighbor_label_matches_per_sibling_reference(synthetic_root):
    from fracturelens.core.io import get_case_paths, load_volume

    _, label_path = get_case_paths(synthetic_root, "001")
    label_vol, spacing_xyz = load_volume(label_path)
    voxel_spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    frag_labels = [21, 22]
    expected = _reference_nearest_labels(label_vol, frag_labels, voxel_spacing)

    report = compute_case_report("001", synthetic_root, mesh_smooth_iterations=0)
    actual = {f.label: f.nearest_neighbor_label for f in report.fragments if f.label in frag_labels}

    assert actual == expected
