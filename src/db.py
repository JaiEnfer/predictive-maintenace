from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from src.settings import ARTIFACT_DIR

DB_PATH = ARTIFACT_DIR / "predictions.db"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # features
    temperature_c = Column(Float, nullable=False)
    vibration_mm_s = Column(Float, nullable=False)
    pressure_bar = Column(Float, nullable=False)
    runtime_hours = Column(Float, nullable=False)

    # outputs
    probability = Column(Float, nullable=False)
    maintenance_required = Column(Integer, nullable=False)

    # metadata
    model_version = Column(String, nullable=False)


def get_engine():
    return create_engine(DB_URL, future=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

EXPECTED_COLUMNS = {
    "id",
    "timestamp_utc",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "runtime_hours",
    "probability",
    "maintenance_required",
    "model_version",
}


def _recreate_prediction_logs_if_needed(engine) -> None:
    inspector = inspect(engine)
    if "prediction_logs" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("prediction_logs")}
    if existing_columns == EXPECTED_COLUMNS:
        return

    PredictionLog.__table__.drop(bind=engine, checkfirst=True)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    _recreate_prediction_logs_if_needed(engine)
    Base.metadata.create_all(bind=engine)
