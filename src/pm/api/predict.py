from pathlib import Path
import joblib
import numpy as np

from pm.constants import SENSOR_FEATURES

MODEL_PATH = Path("models/rf_predictive_maintenance.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)


def predict_failure(sensor_values: list[float]) -> tuple[int, float]:
    arr = np.array(sensor_values, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(arr)[0, 1])
    label = int(prob >= 0.5)
    return label, prob


def predict_failure_from_row(values: dict) -> tuple[int, float]:
    # extract in correct order
    missing = [f for f in SENSOR_FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing required sensor fields: {missing}")

    sensor_values = [float(values[f]) for f in SENSOR_FEATURES]
    return predict_failure(sensor_values)
