# 🔍 ML Model Monitoring Service

**Detect data drift in production ML models before accuracy silently degrades.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://drift-detector-api.onrender.com/docs)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-10%20passing-success)](#testing)
[![Docker](https://img.shields.io/badge/containerized-docker-2496ED?logo=docker&logoColor=white)](#deployment)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

🔗 **[Try it live →](https://drift-detector-api.onrender.com/docs)**
*(free-tier hosting — first request after inactivity may take 30–60s to wake up)*

---

## Project Highlights

- Implements two industry-standard drift detection methods **from first principles** (not a wrapper around a library) — Kolmogorov-Smirnov test and Population Stability Index — with math validated against synthetic same/shifted distributions before integration
- Clean layered architecture: pure statistical functions, business-rule orchestrator, and HTTP layer are fully decoupled and independently testable
- Centralized, typed error handling — no unhandled exceptions leak to API consumers
- 10 automated tests (pytest) covering statistical correctness and API behavior
- Fully containerized (Docker) and deployed live with a public REST API and interactive Swagger documentation

---

## Table of Contents
- [Why this exists](#why-this-exists)
- [Core concepts](#core-concepts)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Example](#example-request--response)
- [Error handling](#error-handling)
- [Testing](#testing)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [Known limitations](#known-limitations--next-steps)
- [Author](#author)

---

## Why this exists

A model's accuracy is measured once, at training time. In production, the
real world keeps changing — user behavior shifts, upstream pipelines
change, seasons change. The model itself doesn't change, but the data it
sees does. This mismatch is called **drift**, and it's one of the most
common reasons deployed ML models silently degrade without anyone
noticing until business metrics tank.

This service compares live production traffic against a stored
training-time baseline on demand, and flags when drift crosses a
threshold — an early-warning system before accuracy visibly drops.

---

## Core concepts

### 📊 Kolmogorov-Smirnov (KS) Test
A nonparametric two-sample statistical test. Builds the empirical CDF for
both baseline and live samples, finds the maximum vertical gap between
them (the **D statistic**), and converts it, with sample sizes, into a
**p-value** — the probability of seeing a gap this large by chance if the
two samples came from the same distribution. p < 0.05 → distributions
likely differ.

Makes no assumption about distribution shape, so it catches shifts a
mean/variance comparison would miss. Limitation: continuous features
only; can get hypersensitive at very large sample sizes.

### 📈 Population Stability Index (PSI)
Bins the baseline into deciles (**edges frozen from baseline**, reused on
live data — critical, otherwise you're comparing a moving target to
itself). Sums a weighted log-ratio of population share per bin:



| PSI value | Meaning |
|:---:|---|
| **< 0.10** | ✅ Stable — no meaningful shift |
| **0.10 – 0.25** | ⚠️ Moderate shift — worth investigating |
| **≥ 0.25** | 🚨 Major shift — model likely needs review/retraining |

Works on categorical features too. Fixed, business-friendly thresholds —
the industry-standard choice for dashboards and alerting.

### Why run both?
KS is statistically rigorous but continuous-only and oversensitive at
scale. PSI is a business-friendly magnitude score with somewhat arbitrary
bin boundaries. Running both covers each one's blind spot.

---

## Tech stack

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

## Architecture
app/
├── stats/
│   ├── ks_test.py         # KS drift statistic + p-value
│   └── psi.py              # PSI calculation (continuous + categorical)
├── monitors.py              # missing-value / outlier ratio checks
├── drift_detector.py        # orchestrates stats + monitors → verdict
├── reference_store.py       # baseline distribution storage
├── exceptions.py             # typed domain errors → HTTP status mapping
└── api.py                   # FastAPI routes + centralized error handling
tests/                        # 10 passing unit + integration tests
Dockerfile                    # containerized, deployed on Render



**Design principle:** `stats/` and `monitors.py` are pure functions —
numpy arrays in, dataclasses out, zero framework dependency — so the
actual math is unit-testable in isolation. `drift_detector.py` is the
only place business rules live ("PSI major → alert"). `api.py` is a thin
HTTP shell on top. Storage backend, web framework, or thresholds can each
change independently without touching the others.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reference` | Register/overwrite the baseline for a `model_id` |
| `POST` | `/monitor` | Compare a live batch against the stored baseline |

Full interactive docs: **[/docs](https://drift-detector-api.onrender.com/docs)**

## Example request / response

**Register a baseline:**
```bash
curl -X POST https://drift-detector-api.onrender.com/reference \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn_v1",
    "features": {"income": [50000, 52000, 48000, ...]},
    "feature_types": {"income": "numeric"}
  }'
```

**Check a live batch:**
```bash
curl -X POST https://drift-detector-api.onrender.com/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn_v1",
    "features": {"income": [70000, 72000, 68000, ...]}
  }'
```

**Response:**
```json
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
```

---

## Error handling

1. **Schema validation** (Pydantic) rejects malformed requests at the edge.
2. **Typed domain exceptions** (`app/exceptions.py`) carry their own HTTP
   status + machine-readable `error_code`, mapped by one global FastAPI
   exception handler.
3. **Catch-all handler** logs the full traceback server-side, returns a
   generic 500 to the caller — no internals leaked.
4. Stats functions raise `InsufficientDataError` instead of crashing on
   tiny/degenerate batches.

---

## Testing

10 automated tests covering statistical correctness and API behavior:
```bash
pytest -v
```

---

## Running locally

```bash
git clone https://github.com/Aswathyvk/ml-model-monitoring-service.git
cd ml-model-monitoring-service
pip install -r requirements.txt
uvicorn main:app --reload
```
Docs: http://127.0.0.1:8000/docs

---

## Deployment

Containerized via the included `Dockerfile`, deployed on [Render](https://render.com)
(free tier) — Render builds the image directly from this repo on every
push to `main`.

---

## Known limitations / next steps

- **In-memory storage** — baselines lost on restart; production version
  would use Redis/Postgres.
- **No concept drift detection** — needs ground-truth labels + a
  labeling-delay pipeline; out of scope here.
- **No alerting integration** — `/monitor` response is designed to wire
  into Slack/PagerDuty easily, doesn't include one.
- **No scheduling** — stateless check-on-demand API; a cron job would
  call `/monitor` periodically in a real deployment.
- **No history/dashboarding** — stores baselines, not a time series of
  past monitor runs.

---

## Author

**Aswathy VK**
[GitHub](https://github.com/Aswathyvk) · [LinkedIn](https://linkedin.com/in/aswathy-vk-034465280) · [Portfolio](https://aswathyvk.github.io)

---

*Built solo as a learning project to understand production ML monitoring
concepts end-to-end — from statistical theory to deployed API.*
