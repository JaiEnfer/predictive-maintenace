from pathlib import Path

from fastapi.testclient import TestClient

from src.app import app
from src.drift import DRIFT_REPORT_PATH


def test_drift_report_endpoint_generates_html():
    with TestClient(app) as client:
        # Create some data points
        for _ in range(5):
            payload = {
                "temperature_c": 118,
                "vibration_mm_s": 16,
                "pressure_bar": 138,
                "runtime_hours": 9100,
            }
            r = client.post("/predict", json=payload)
            assert r.status_code == 200

        # Now generate report
        r = client.get("/drift/report")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert Path(DRIFT_REPORT_PATH).exists()
