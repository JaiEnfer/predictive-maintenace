from __future__ import annotations

import joblib
import pandas as pd

from src.settings import ARTIFACT_DIR
from src.train import FEATURES, train

MODEL_PATH = ARTIFACT_DIR / "model.joblib"


class ModelService:
    def __init__(self) -> None:
        self._bundle = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            train()
        self._bundle = joblib.load(MODEL_PATH)
        if self._bundle.get("features") != FEATURES:
            train()
            self._bundle = joblib.load(MODEL_PATH)
    def reload(self) -> None:
        # same as load, but nicer name for runtime updates
        self.load()

    @property
    def model_version(self) -> str:
        return self._bundle["model_version"]

    @property
    def features(self) -> list[str]:
        return self._bundle["features"]

    def predict_proba(self, row: dict) -> float:
        model = self._bundle["model"]
        X = pd.DataFrame([row], columns=self.features)
        proba = float(model.predict_proba(X)[0, 1])
        return proba
