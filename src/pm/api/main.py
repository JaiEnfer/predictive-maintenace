import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request

from pm.api.metrics import metrics
from pm.api.predict import predict_failure, predict_failure_from_row
from pm.api.schemas import PredictionResponse, SensorData, TurbofanRow

logger = logging.getLogger("pm.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Predictive Maintenance API")


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    # request id: accept inbound or create one
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()

    # Count requests
    metrics.total_requests += 1

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id

    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return {
        "total_requests": metrics.total_requests,
        "total_predictions": metrics.total_predictions,
        "fail_soon_predictions": metrics.fail_soon_predictions,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    if len(data.sensor_values) != 21:
        raise HTTPException(status_code=400, detail="Expected 21 sensor values")

    label, prob = predict_failure(data.sensor_values)

    # metrics
    metrics.total_predictions += 1
    if label == 1:
        metrics.fail_soon_predictions += 1

    logger.info(
        "prediction made",
        extra={
            "endpoint": "/predict",
            "will_fail_soon": label,
            "failure_probability": round(prob, 6),
        },
    )

    return PredictionResponse(will_fail_soon=label, failure_probability=prob)


@app.post("/predict_row", response_model=PredictionResponse)
def predict_row(data: TurbofanRow):
    try:
        label, prob = predict_failure_from_row(data.values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # metrics
    metrics.total_predictions += 1
    if label == 1:
        metrics.fail_soon_predictions += 1

    logger.info(
        "prediction made",
        extra={
            "endpoint": "/predict_row",
            "will_fail_soon": label,
            "failure_probability": round(prob, 6),
        },
    )

    return PredictionResponse(will_fail_soon=label, failure_probability=prob)
