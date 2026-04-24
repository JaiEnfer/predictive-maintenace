from fastapi.testclient import TestClient

from src.app import app


def test_index_page():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "maintenance intelligence" in r.text.lower()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_version" in data


def test_predict():
    with TestClient(app) as client:
        payload = {
            "temperature_c": 112,
            "vibration_mm_s": 18,
            "pressure_bar": 145,
            "runtime_hours": 8600,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "maintenance_required" in data
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
