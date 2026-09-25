"""Test co ban cho Backend (mock AI Service qua monkeypatch)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app as backend_app  # noqa: E402


def test_health():
    client = backend_app.app.test_client()
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_missing_body_returns_400_or_502():
    client = backend_app.app.test_client()
    r = client.post("/api/predict", json={"features": {}})
    assert r.status_code in (400, 502)
