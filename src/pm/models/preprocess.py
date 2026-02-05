from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from pm.constants import SENSOR_FEATURES


class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Stateless rolling-feature approximation for inference:
    If you only pass a single row, rolling mean=raw value and std=0.
    This keeps the API stateless and demo-friendly.
    """

    def __init__(self, window: int = 5):
        self.window = window

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        # X is expected to contain SENSOR_FEATURES columns
        X = X.copy()
        out = X[SENSOR_FEATURES].copy()

        # For single-row inference: roll_mean == value, roll_std == 0
        for col in SENSOR_FEATURES:
            out[f"{col}_roll_mean_{self.window}"] = out[col].astype(float)
            out[f"{col}_roll_std_{self.window}"] = 0.0

        return out
