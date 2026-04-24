from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.db import get_engine
from src.settings import ARTIFACT_DIR, REPORTS_DIR

REFERENCE_PATH = ARTIFACT_DIR / "reference.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"

FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "runtime_hours"]


@dataclass
class DriftStatus:
    drift_detected: bool
    share_drifted_features: float
    drifted_features: list[str]
    n_reference: int
    n_current: int


def load_current_data(limit: int = 500) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT {", ".join(FEATURES)}
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, engine)
    return df[FEATURES]


def _run_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def generate_drift_report(limit: int = 500) -> Path:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_PATH}. Run: python -m src.train"
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current = load_current_data(limit=limit)

    if current.empty:
        raise ValueError("No production data found yet. Call /predict a few times first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = _run_report(reference, current)
    report.save_html(str(DRIFT_REPORT_PATH))
    return DRIFT_REPORT_PATH


def get_drift_status(limit: int = 500) -> DriftStatus:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_PATH}. Run: python -m src.train"
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current = load_current_data(limit=limit)

    if current.empty:
        raise ValueError("No production data found yet. Call /predict a few times first.")

    report = _run_report(reference, current)
    result: dict[str, Any] = report.as_dict()

    metrics = result.get("metrics", [])
    preset_result = None

    # Find a metric whose result looks like drift output
    for m in metrics:
        r = m.get("result", {})
        if isinstance(r, dict) and ("dataset_drift" in r) and ("drift_by_columns" in r):
            preset_result = r
            break

    if preset_result is None:
        metric_names = [m.get("metric") for m in metrics]
        raise RuntimeError(
            f"Could not locate drift results in Evidently output. Metrics seen: {metric_names}"
        )

    # ---- Robust parsing starts here ----

    # drift_by_columns is usually a dict mapping feature -> info
    columns = preset_result.get("drift_by_columns", {})
    if not isinstance(columns, dict):
        columns = {}

    drifted_features: list[str] = []
    for col, info in columns.items():
        if isinstance(info, dict):
            if bool(info.get("drift_detected", False)):
                drifted_features.append(col)
        elif isinstance(info, bool):
            # some versions may store just True/False
            if info:
                drifted_features.append(col)

    drifted_features = sorted(set(drifted_features))

    dataset_drift_value = preset_result.get("dataset_drift", False)

    # Case A: dataset_drift is a dict (older format)
    if isinstance(dataset_drift_value, dict):
        drift_detected = bool(dataset_drift_value.get("dataset_drift", False))
        drift_share = float(dataset_drift_value.get("share_of_drifted_columns", 0.0))
    # Case B: dataset_drift is a boolean (your format)
    elif isinstance(dataset_drift_value, bool):
        drift_detected = dataset_drift_value
        # compute share ourselves
        drift_share = len(drifted_features) / max(len(FEATURES), 1)
    else:
        # fallback
        drift_detected = False
        drift_share = 0.0

    return DriftStatus(
        drift_detected=drift_detected,
        share_drifted_features=float(drift_share),
        drifted_features=drifted_features,
        n_reference=int(reference.shape[0]),
        n_current=int(current.shape[0]),
    )
