[![CI](https://github.com/JaiEnfer/predictive-maintenace/actions/workflows/ci.yml/badge.svg)](https://github.com/JaiEnfer/predictive-maintenace/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
![GHCR](https://img.shields.io/badge/GHCR-published-blue)
![MLOps](https://img.shields.io/badge/MLOps-production--ready-brightgreen)



# 🏭 Predictive Maintenance ML System

Production-style machine learning system for **industrial predictive maintenance**, built using the NASA C-MAPSS turbofan engine dataset.  
This project demonstrates how to take an ML model from raw data to a **containerized, deployable API with CI/CD**.

---

## 🚀 What This Project Demonstrates

This is not just a notebook model. It showcases:

- 📊 Time-series feature engineering for equipment health prediction  
- 🧠 ML pipeline with preprocessing + model bundled together  
- 🏭 Leakage-free train/validation split by engine (real-world practice)  
- 🌐 FastAPI inference service for real-time predictions  
- 🐳 Dockerized deployment  
- 🧪 Automated tests (pytest)  
- 🧹 Linting with Ruff  
- 🔄 CI/CD pipeline using GitHub Actions  
- 📦 Docker image published to GitHub Container Registry (GHCR)

---

## 🧠 Business Context

In manufacturing and industrial environments, unexpected equipment failures are costly and dangerous.  
Predictive maintenance systems use machine learning to detect early signs of failure so maintenance can be scheduled **before breakdowns occur**.

This project simulates such a system end-to-end.

---

## 🏗️ System Architecture

**Training Pipeline**

```text
Raw Sensor Data → Feature Engineering → ML Pipeline (Preprocessing + Model) → Saved Artifact
```

**Inference Service**

```text
API Request → Feature Extraction → Loaded ML Pipeline → Failure Risk Prediction
```

---

## 🛠 Tech Stack

| Area | Tools |
|------|------|
| Language | Python |
| ML | scikit-learn, pandas, numpy |
| API | FastAPI |
| Testing | pytest |
| Linting | Ruff |
| Packaging | Docker |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (GHCR) |

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

|Endpoint | Description|
|---------|------------|
|GET /| health check|
|POST /predict| Predict failure risk from 21 sensor values|
|POST /predict_row| Predict from row-style sensor dictionary|
|GET /metrics| Simple request & prediction counters|

---

## Dataset

[Data Set link](https://zenodo.org/records/15346912?utm_source=chatgpt.com)

---

## 🎯 Key Engineering Practices

✔ Prevented training/serving skew by packaging preprocessing with the model
✔ Avoided data leakage using engine-level splits
✔ Built a stateless inference API while keeping feature logic consistent
✔ Implemented CI/CD for testing and Docker image publishing

---

___Thank You___
