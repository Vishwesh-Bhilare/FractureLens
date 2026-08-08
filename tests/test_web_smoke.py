from fastapi.testclient import TestClient

from fracturelens.web.app import app


def test_case_list_includes_synthetic_case(synthetic_root):
    app.state.root = synthetic_root
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "001" in response.text


def test_case_detail_shell_loads(synthetic_root):
    app.state.root = synthetic_root
    client = TestClient(app)
    response = client.get("/case/001")
    assert response.status_code == 200


def test_slice_png_loads(synthetic_root):
    app.state.root = synthetic_root
    client = TestClient(app)
    response = client.get("/case/001/slice?axis=0&index=20")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_slice_rejects_invalid_axis(synthetic_root):
    app.state.root = synthetic_root
    client = TestClient(app)
    response = client.get("/case/001/slice?axis=5&index=0")
    assert response.status_code == 400


def test_metrics_panel_smoke(synthetic_root):
    app.state.root = synthetic_root
    client = TestClient(app)
    response = client.get("/case/001/metrics-panel")
    assert response.status_code == 200
    assert "3" in response.text


def test_slice_uses_cached_volumes_on_repeat_request(synthetic_root, monkeypatch):
    import fracturelens.web.app as web_app

    web_app._load_volume_cached.cache_clear()
    call_count = 0
    original_load_volume = web_app.load_volume

    def counting_load_volume(path):
        nonlocal call_count
        call_count += 1
        return original_load_volume(path)

    monkeypatch.setattr(web_app, "load_volume", counting_load_volume)
    app.state.root = synthetic_root
    client = TestClient(app)

    first_response = client.get("/case/001/slice?axis=0&index=20")
    calls_after_first_request = call_count
    second_response = client.get("/case/001/slice?axis=0&index=20")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert calls_after_first_request == 2
    assert call_count == calls_after_first_request
