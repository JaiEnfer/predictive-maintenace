from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.db import PredictionLog, SessionLocal, init_db
from src.drift import generate_drift_report, get_drift_status
from src.model import ModelService
from src.retrain import retrain_safely
from src.schemas import PredictRequest, PredictResponse

model_service = ModelService()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_PATH = FRONTEND_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    model_service.load()
    yield

app = FastAPI(title="ML Monitoring System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_service.model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = req.model_dump()
    proba = model_service.predict_proba(row)
    maintenance_required = proba >= 0.5

    # Log to SQLite
    db = SessionLocal()
    try:
        log = PredictionLog(
            temperature_c=row["temperature_c"],
            vibration_mm_s=row["vibration_mm_s"],
            pressure_bar=row["pressure_bar"],
            runtime_hours=row["runtime_hours"],
            probability=proba,
            maintenance_required=int(maintenance_required),
            model_version=model_service.model_version,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return PredictResponse(
        maintenance_required=maintenance_required,
        probability=proba,
        model_version=model_service.model_version,
    )

@app.get("/drift/report")
def drift_report():
    path = generate_drift_report(limit=500)
    return FileResponse(path, media_type="text/html", filename=path.name)

@app.get("/drift/status")
def drift_status():
    try:
        status = get_drift_status(limit=500)
        return {
            "drift_detected": status.drift_detected,
            "share_drifted_features": status.share_drifted_features,
            "drifted_features": status.drifted_features,
            "n_reference": status.n_reference,
            "n_current": status.n_current,
        }
    except Exception as e:
        # Return readable error instead of generic 500
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift/alert")
def drift_alert(threshold: float = 0.5):
    """
    threshold: fraction of features drifted to raise an alert
    """
    status = get_drift_status(limit=500)
    level = "OK"
    if status.share_drifted_features >= threshold:
        level = "ALERT"

    return {
        "level": level,
        "threshold": threshold,
        "share_drifted_features": status.share_drifted_features,
        "drifted_features": status.drifted_features,
        "n_current": status.n_current,
        "n_reference": status.n_reference,
    }

@app.post("/retrain")
def retrain():
    ran = retrain_safely()
    if not ran:
        return {"status": "already_running"}

    # reload freshly trained model artifact
    model_service.reload()
    return {"status": "ok", "model_version": model_service.model_version}
