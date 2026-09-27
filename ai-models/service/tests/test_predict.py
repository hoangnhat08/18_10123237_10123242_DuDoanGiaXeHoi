"""Test co ban cho AI Service (chay bang pytest tu thu muc ai-models/service)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from service.app import app  # noqa: E402


def test_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_valid_returns_all_4_models():
    client = app.test_client()
    payload = {"features": {
        "brand": "Toyota", "model": "Camry 2.5Q", "series": "Sedan",
        "year": 2018, "driven_kms": 50000, "assemble_place": "Nhập khẩu",
        "engine_type": "Xăng", "transmission": "Số tự động",
        "num_of_door": 4, "num_of_seat": 5,
    }}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "prediction_million_vnd" in data  # tuong thich nguoc: ket qua cua best model
    assert "results" in data
    assert len(data["results"]) == 4  # phai co du 4 model
    keys = {item["key"] for item in data["results"]}
    assert keys == {"linear_regression", "decision_tree", "random_forest", "svr"}
    for item in data["results"]:
        assert "prediction_million_vnd" in item
        assert "model_name" in item
    assert "best_model" in data
    assert data["best_model"]["is_best"] is True


def test_predict_missing_field():
    client = app.test_client()
    r = client.post("/predict", json={"features": {"year": 2018}})
    assert r.status_code == 400
