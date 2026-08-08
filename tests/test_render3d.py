def test_case_figure_trace_count_and_buttons(synthetic_root):
    from fracturelens.core.io import get_case_paths, load_volume
    from fracturelens.core.metrics import compute_case_report_with_meshes
    from fracturelens.core.render3d import build_case_figure

    report, meshes = compute_case_report_with_meshes("001", synthetic_root, mesh_smooth_iterations=0)
    _, label_path = get_case_paths(synthetic_root, "001")
    label_vol, spacing = load_volume(label_path)
    fig = build_case_figure(label_vol, spacing, mesh_smooth_iterations=0, fragment_meshes=meshes)

    assert report.total_fragment_count == 3
    assert len(fig.data) == 6
    assert [trace.visible for trace in fig.data].count(True) == 3

    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 4

    fractured_assembled_button = next(b for b in buttons if b.label == "Fractured · Assembled")
    visible = fractured_assembled_button.args[0]["visible"]
    assert sum(visible) == 2
