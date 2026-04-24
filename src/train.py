from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.settings import ARTIFACT_DIR

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "meta.json"
REFERENCE_PATH = ARTIFACT_DIR / "reference.csv"
VERSION_PATH = ARTIFACT_DIR / "version.json"



FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "runtime_hours"]
TARGET = "maintenance_required"

@dataclass
class TrainResult:
    auc: float

def make_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    temperature_c = rng.normal(72, 18, n).clip(20, 150)
    vibration_mm_s = rng.normal(8, 4, n).clip(0, 40)
    pressure_bar = rng.normal(95, 20, n).clip(30, 180)
    runtime_hours = rng.normal(4200, 1800, n).clip(100, 12000)

    # Higher temperature, vibration, pressure excursions, and long runtime
    # should increase the chance that the equipment needs maintenance.
    logits = (
        0.08 * (temperature_c - 75)
        + 0.22 * (vibration_mm_s - 8)
        + 0.03 * np.abs(pressure_bar - 95)
        + 0.00055 * (runtime_hours - 4000)
        - 3.1
    )
    prob = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "temperature_c": temperature_c,
            "vibration_mm_s": vibration_mm_s,
            "pressure_bar": pressure_bar,
            "runtime_hours": runtime_hours,
            "maintenance_required": y,
        }
    )
    return df

def next_version() -> str:
    if VERSION_PATH.exists():
        data = json.loads(VERSION_PATH.read_text(encoding="utf-8"))
        n = int(data.get("version", 1)) + 1
    else:
        n = 1
    VERSION_PATH.write_text(json.dumps({"version": n}), encoding="utf-8")
    return f"v{n}"



def train() -> TrainResult:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = make_synthetic_data()
    # Save a baseline (reference) dataset for drift monitoring
    df[FEATURES].sample(n=1000, random_state=42).to_csv(REFERENCE_PATH, index=False)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    version = next_version()


    joblib.dump(
        {"model": model, "features": FEATURES, "model_version": version},
        MODEL_PATH,
    )

    return TrainResult(auc=auc)


if __name__ == "__main__":
    result = train()
    print(f"Trained model saved to {MODEL_PATH}. AUC={result.auc:.4f}")
