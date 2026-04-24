![CI](https://github.com/JaiEnfer/ml-monitoring-system/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/Docker-Build%20Ready-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![MLOps](https://img.shields.io/badge/MLOps-Drift%20Monitoring-orange)
![GHCR](https://img.shields.io/badge/GHCR-Docker%20Image-success)
![Lifecycle](https://img.shields.io/badge/ML%20Lifecycle-Train%20→%20Monitor%20→%20Retrain-purple)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)

# Predictive Maintenance Monitoring System

A production-style predictive maintenance application that accepts machine sensor readings, predicts whether maintenance is needed, tracks incoming data drift, and exposes everything through both an API and a browser dashboard.

## What This Project Includes

- FastAPI backend for maintenance inference
- Interactive frontend dashboard at `GET /`
- Sensor-driven prediction endpoint
- SQLite prediction logging
- Drift monitoring with Evidently
- HTML drift report generation
- Safe runtime retraining with model versioning
- Docker support
- GitHub Actions CI for lint, tests, and container build/push

## User Experience

The dashboard is designed like a lightweight operations console:

- live service heartbeat and model version
- sensor input form with presets and random sample generation
- circular risk gauge for maintenance probability
- drift radar summary and HTML report access
- retrain control for hot model refresh
- recent prediction history panel

Open `http://127.0.0.1:8000/` after starting the service.

## API Endpoints

- `GET /` serves the frontend dashboard
- `GET /health` returns service status and active model version
- `POST /predict` scores one machine payload
- `GET /drift/status` returns drift summary JSON
- `GET /drift/report` generates an HTML drift report
- `GET /drift/alert?threshold=0.5` returns `OK` or `ALERT`
- `POST /retrain` retrains the model and hot-reloads it

## Prediction Schema

Request body:

```json
{
  "temperature_c": 112,
  "vibration_mm_s": 18,
  "pressure_bar": 145,
  "runtime_hours": 8600
}
```

Response body:

```json
{
  "maintenance_required": true,
  "probability": 0.87,
  "model_version": "v4"
}
```

## Project Structure

```text
frontend/              Browser dashboard assets
src/app.py             FastAPI app and route wiring
src/db.py              SQLite logging
src/drift.py           Drift detection and reporting
src/model.py           Model loading and inference
src/retrain.py         Safe retraining lock
src/schemas.py         Request and response schemas
src/settings.py        Writable runtime data locations
src/simulate_traffic.py Drift demo traffic generator
src/train.py           Synthetic training pipeline
tests/                 API and workflow tests
```

## Local Run

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

Run the app:

```sh
uvicorn src.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

## Development Checks

Run lint:

```sh
ruff check src tests --no-cache
```

Run tests:

```sh
pytest -q -p no:cacheprovider
```

## Drift Demo

Send baseline traffic:

```sh
python -m src.simulate_traffic --mode baseline --n 200
```

Send shifted traffic:

```sh
python -m src.simulate_traffic --mode shifted --n 200
```

Then inspect:

```text
http://127.0.0.1:8000/drift/status
http://127.0.0.1:8000/drift/report
```

## Runtime Data

Runtime artifacts are stored in a writable app-data directory by default so the service can run even when the repository itself is read-only.

Override the location with:

```text
ML_MONITORING_DATA_DIR
```

That directory contains generated items such as:

- model artifacts
- version metadata
- reference drift dataset
- SQLite prediction log
- generated drift reports

## Docker

Build locally:

```sh
docker build -t predictive-maintenance .
```

Run locally:

```sh
docker run --rm -p 8000:8000 predictive-maintenance
```

## CI/CD

The GitHub Actions workflow runs:

- Ruff lint checks
- pytest suite
- Docker build validation
- GHCR push on `main` and version tags

## Notes

- The app auto-trains on startup if the model artifact is missing or stale.
- The logging table is recreated automatically if an older schema is detected.
- The model currently uses synthetic sensor data for demonstration, which keeps the full workflow runnable without an external data dependency.
