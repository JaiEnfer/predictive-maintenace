from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(tempfile.gettempdir()) / "ml-monitoring-system"


DATA_DIR = Path(os.getenv("ML_MONITORING_DATA_DIR", _default_data_dir()))
ARTIFACT_DIR = DATA_DIR / "artifacts"
REPORTS_DIR = DATA_DIR / "reports"
