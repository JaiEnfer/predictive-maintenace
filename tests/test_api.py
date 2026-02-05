from fastapi.testclient import TestClient
from pm.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid_payload():
    payload = {"sensor_values": [500.0] * 21}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "will_fail_soon" in data
    assert "failure_probability" in data
    assert data["will_fail_soon"] in (0, 1)
    assert 0.0 <= data["failure_probability"] <= 1.0

def test_predict_invalid_length():
    payload = {"sensor_values": [1.0] * 20}
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
