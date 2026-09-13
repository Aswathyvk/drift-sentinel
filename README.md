# 🔍 ML Model Monitoring Service

**A production-shaped API that catches ML model drift before it silently tanks your accuracy.**

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/containerized-docker-2496ED?logo=docker&logoColor=white)](#deployment)
[![Tests](https://img.shields.io/badge/tests-10%20passing-success)](#testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**[Live Demo →](https://drift-detector-api.onrender.com/docs)** · **[API Docs →](https://drift-detector-api.onrender.com/docs)** · **[Report Bug](https://github.com/Aswathyvk/ml-model-monitoring-service/issues)**

*(free-tier hosting — first request after inactivity may take 30–60s to wake up)*

---

## 📌 The Problem

A deployed ML model's accuracy can silently degrade in production — long before you have new labels to measure it directly — because the data it sees drifts away from what it was trained on. Most teams find out only after something visibly breaks downstream.

This service is the early-warning layer: it statistically compares live production data against training-time data and flags exactly what shifted, and by how much.

---

## ✨ Key Features

| Capability | What it catches |
|---|---|
| **Data drift** (KS test + PSI) | A feature's distribution shifted — mean, spread, or shape |
| **Data quality checks** | Missing values or internal outliers in a live batch |
| **Verdict aggregation** | One `ok` / `warn` / `alert` per batch, worst-case across all features |
| **Typed error handling** | Every failure mode returns a clean, predictable JSON error |
| **Containerized + deployed** | Runs identically locally, in Docker, and on Render |

---

## 🧠 Core Concepts

### Kolmogorov-Smirnov (KS) Test
A nonparametric two-sample statistical test. Builds the empirical CDF for both baseline and live samples, finds the maximum vertical gap between them (the **D statistic**), and converts it, with sample sizes, into a **p-value** — the probability of seeing a gap this large by chance if the two samples came from the same distribution. p < 0.05 → distributions likely differ.

Makes no assumption about distribution shape, so it catches shifts a mean/variance comparison would miss. Limitation: continuous features only; can get hypersensitive at very large sample sizes.

### Population Stability Index (PSI)
Bins the baseline into deciles (**edges frozen from baseline**, reused on live data — critical, otherwise you're comparing a moving target to itself). Sums a weighted log-ratio of population share per bin:

PSI = Σ (live% − baseline%) × ln(live% / baseline%)

| PSI value | Meaning |
|:---:|---|
| **< 0.10** | ✅ Stable — no meaningful shift |
| **0.10 – 0.25** | ⚠️ Moderate shift — worth investigating |
| **≥ 0.25** | 🚨 Major shift — model likely needs review/retraining |

Works on categorical features too. Fixed, business-friendly thresholds — the industry-standard choice for dashboards and alerting.

### Why run both?
KS is statistically rigorous but continuous-only and oversensitive at scale. PSI is a business-friendly magnitude score with somewhat arbitrary bin boundaries. Running both covers each one's blind spot.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Validation | Pydantic / pydantic-settings |
| Statistics | NumPy, SciPy |
| Testing | pytest |
| Containerization | Docker |
| Deployment | Render |
| Language | Python 3.10 |

---

## 🏗️ Architecture

    app/
    ├── stats/
    │   ├── ks_test.py         KS drift statistic + p-value
    │   └── psi.py             PSI calculation (continuous + categorical)
    ├── monitors.py            missing-value / outlier ratio checks
    ├── drift_detector.py      orchestrates stats + monitors -> verdict
    ├── reference_store.py     baseline distribution storage
    ├── exceptions.py          typed domain errors -> HTTP status mapping
    └── api.py                 FastAPI routes + centralized error handling
    tests/                     10 passing unit + integration tests
    Dockerfile                 containerized, deployed on Render

**Design principle:** `stats/` and `monitors.py` are pure functions — numpy arrays in, dataclasses out, zero framework dependency — so the actual math is unit-testable in isolation. `drift_detector.py` is the only place business rules live ("PSI major → alert"). `api.py` is a thin HTTP shell on top. Storage backend, web framework, or thresholds can each change independently without touching the others.

---

## 🔌 API Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reference` | Register/overwrite the baseline for a `model_id` |
| `POST` | `/monitor` | Compare a live batch against the stored baseline |

Full interactive docs: **[/docs](https://drift-detector-api.onrender.com/docs)**

## 📥 Example Request / Response

**Register a baseline:**

    curl -X POST https://drift-detector-api.onrender.com/reference \
      -H "Content-Type: application/json" \
      -d '{
        "model_id": "churn_v1",
        "features": {"income": [50000, 52000, 48000]},
        "feature_types": {"income": "numeric"}
      }'

**Check a live batch:**

    curl -X POST https://drift-detector-api.onrender.com/monitor \
      -H "Content-Type: application/json" \
      -d '{
        "model_id": "churn_v1",
        "features": {"income": [70000, 72000, 68000]}
      }'

**Response:**

    {
      "model_id": "churn_v1",
      "overall_verdict": "alert",
      "feature_reports": [
        {
          "feature": "income",
          "psi": 11.52,
          "psi_severity": "major",
          "ks_drifted": true,
          "missing_ratio": 0.0,
          "verdict": "alert"
        }
      ]
    }

---

## ⚠️ Error Handling

1. **Schema validation** (Pydantic) rejects malformed requests at the edge.
2. **Typed domain exceptions** (`app/exceptions.py`) carry their own HTTP status + machine-readable `error_code`, mapped by one global FastAPI exception handler.
3. **Catch-all handler** logs the full traceback server-side, returns a generic 500 to the caller — no internals leaked.
4. Stats functions raise `InsufficientDataError` instead of crashing on tiny/degenerate batches.

---

## ✅ Testing

10 automated tests covering statistical correctness and API behavior:

    pytest -v

---

## 🚀 Running Locally

    git clone https://github.com/Aswathyvk/ml-model-monitoring-service.git
    cd ml-model-monitoring-service
    pip install -r requirements.txt
    uvicorn main:app --reload

Docs: http://127.0.0.1:8000/docs

---

## 📦 Deployment

Containerized via the included `Dockerfile`, deployed on [Render](https://render.com) (free tier) — Render builds the image directly from this repo on every push to `main`.

---

## 🔭 Known Limitations / Next Steps

- **In-memory storage** — baselines lost on restart; production version would use Redis/Postgres.
- **No concept drift detection** — needs ground-truth labels + a labeling-delay pipeline; out of scope here.
- **No alerting integration** — `/monitor` response is designed to wire into Slack/PagerDuty easily, doesn't include one.
- **No scheduling** — stateless check-on-demand API; a cron job would call `/monitor` periodically in a real deployment.
- **No history/dashboarding** — stores baselines, not a time series of past monitor runs.

---

## 👤 Author

**Aswathy VK**
[GitHub](https://github.com/Aswathyvk) · [LinkedIn](https://linkedin.com/in/aswathy-vk-034465280) · [Portfolio](https://aswathyvk.github.io)

---

*Built solo as a learning project to understand production ML monitoring concepts end-to-end — from statistical theory to deployed API.*
