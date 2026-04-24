from fastapi.testclient import TestClient
from sqlalchemy import text

from src.app import app
from src.db import get_engine, init_db


def test_prediction_is_logged():
    init_db()
    engine = get_engine()

    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()

    with TestClient(app) as client:
        payload = {
            "temperature_c": 88,
            "vibration_mm_s": 9,
            "pressure_bar": 102,
            "runtime_hours": 4200,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    with engine.connect() as conn:
        after = conn.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()

    assert after == before + 1
