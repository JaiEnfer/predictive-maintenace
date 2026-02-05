from fastapi import FastAPI, HTTPException

from pm.api.schemas import SensorData, PredictionResponse, TurbofanRow
from pm.api.predict import predict_failure, predict_failure_from_row


app = FastAPI(title="Predictive Maintenance API")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    if len(data.sensor_values) != 21:
        raise HTTPException(status_code=400, detail="Expected 21 sensor values")

    label, prob = predict_failure(data.sensor_values)

    return PredictionResponse(
        will_fail_soon=label,
        failure_probability=prob
    )

@app.post("/predict_row", response_model=PredictionResponse)
def predict_row(data: TurbofanRow):
    try:
        label, prob = predict_failure_from_row(data.values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(will_fail_soon=label, failure_probability=prob)