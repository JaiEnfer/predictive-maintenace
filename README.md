# Predictive Maintenance (Industrial ML System)

Production-style predictive maintenance system using the NASA C-MAPSS turbofan dataset.
Includes feature engineering, baseline model training, FastAPI inference service, Docker, tests, and CI/CD.

## ✅ CI Status
![CI](../../actions/workflows/ci.yml/badge.svg)

## What this project demonstrates
- End-to-end ML workflow: data loading → feature engineering → model training → inference API
- Production-minded engineering: structured project layout, automated tests, linting, CI/CD
- Deployability: Dockerized API + image publishing to GHCR

## Tech Stack
Python, Pandas, scikit-learn, FastAPI, Docker, GitHub Actions, Ruff, Pytest

---

## Quickstart (Windows)

### 1) Setup
```powershell
.\scripts\setup.ps1
```

### 2) Train baseline model
```sh
.\scripts\train.ps1
```

### 3) Run API
```sh
.\scripts\run_api.ps1
```

[Open Swagger UI:](http://127.0.0.1:8000/docs)

---

## Run with Docker

Build locally

```sh
docker build -t predictive-maintenance:latest .
docker run -p 8000:8000 predictive-maintenance:latest
```

Pull from GHCR (published by CI/CD)

```sh
docker run -p 8000:8000 ghcr.io/JaiEnfer/predictive-maintenance:latest
```

---

## API Endpoints

- GET / health check
- POST /predict expects 21 sensor values
- POST /predict_row expects a dict containing sensor_measurement_1..21
- GET /metrics simple demo metrics

---

## Dataset

[Data Set link](https://zenodo.org/records/15346912?utm_source=chatgpt.com)

---


