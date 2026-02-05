from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("models/rf_predictive_maintenance.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
SENSOR_FEATURES = artifact["sensor_features"]



def predict_failure(sensor_values: list[float]) -> tuple[int, float]:
    if len(sensor_values) != len(SENSOR_FEATURES):
        raise ValueError(f"Expected {len(SENSOR_FEATURES)} sensor values")

    X = pd.DataFrame([sensor_values], columns=SENSOR_FEATURES)
    prob = float(pipeline.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    return label, prob


def predict_failure_from_row(values: dict) -> tuple[int, float]:
    missing = [f for f in SENSOR_FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing required sensor fields: {missing}")

    sensor_values = [float(values[f]) for f in SENSOR_FEATURES]
    return predict_failure(sensor_values)
