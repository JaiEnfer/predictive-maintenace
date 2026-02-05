from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path("models/rf_predictive_maintenance.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)


def predict_failure(sensor_values: list[float]) -> tuple[int, float]:
    arr = np.array(sensor_values).reshape(1, -1)
    prob = model.predict_proba(arr)[0, 1]
    label = int(prob >= 0.5)
    return label, float(prob)
