# Predictive Maintenance Monitoring System

A FastAPI-based machine learning service that ingests equipment sensor readings and predicts whether a machine needs maintenance.

## What It Does

- Accepts live sensor measurements over an API
- Predicts whether maintenance is required
- Logs every prediction to SQLite
- Monitors incoming sensor drift with Evidently
- Generates drift reports for investigation
- Supports safe runtime retraining with model versioning

## Sensor Inputs

The prediction endpoint expects these fields:

- `temperature_c`
- `vibration_mm_s`
- `pressure_bar`
- `runtime_hours`

The response returns:

- `maintenance_required`
- `probability`
- `model_version`

## API

- `GET /` opens the browser dashboard
- `GET /health` returns service status and active model version
- `POST /predict` scores one sensor payload and logs the result
- `GET /drift/status` returns drift summary JSON
- `GET /drift/report` generates an HTML drift report
- `GET /drift/alert?threshold=0.5` returns `OK` or `ALERT`
- `POST /retrain` retrains the maintenance model and hot-reloads it

## Example Request

```json
{
  "temperature_c": 112,
  "vibration_mm_s": 18,
  "pressure_bar": 145,
  "runtime_hours": 8600
}
```

## Local Run

```sh
python -m pip install -r requirements.txt
uvicorn src.app:app --reload
```

Runtime artifacts are stored in a writable app data directory by default. You can override that location with `ML_MONITORING_DATA_DIR`.

Open `http://127.0.0.1:8000/` after starting the app to use the interactive operations dashboard with scenario presets, live risk visuals, drift monitoring, and retrain controls.

## Drift Demo

Baseline traffic:

```sh
python -m src.simulate_traffic --mode baseline --n 200
```

Shifted traffic:

```sh
python -m src.simulate_traffic --mode shifted --n 200
```

Then inspect:

```http
GET /drift/status
GET /drift/report
```

## Notes

- The app auto-retrains on startup if an old model artifact does not match the current sensor feature set.
- The SQLite logging table is recreated automatically if it still has the old project schema.
