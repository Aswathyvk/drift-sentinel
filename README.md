# ML Model Monitoring Service

A lightweight, self-contained service that detects **data drift** and
**data quality issues** in ML model inputs — built with FastAPI, deployable
as a standalone microservice with zero external dependencies.

## Why this exists

A model's accuracy is measured once, at training time. In production, the
real world keeps changing — user behavior shifts, upstream data pipelines
change, seasons change. The model itself doesn't change, but the data it
sees does. This mismatch is called **drift**, and it's one of the most
common reasons deployed ML models silently degrade without anyone
noticing until business metrics tank. This service compares live
production traffic against a stored training-time baseline on demand, and
flags when drift crosses a threshold — an early-warning system before
accuracy visibly drops.

## Core concepts

### Kolmogorov-Smirnov (KS) Test
A nonparametric two-sample statistical test. It builds the empirical CDF
(cumulative distribution function) for both the baseline and live samples,
then finds the maximum vertical gap between them — the **D statistic**.
That gap, combined with both sample sizes, converts into a **p-value**:
the probability of seeing a gap this large if the two samples truly came
from the same distribution. A p-value below 0.05 means the distributions
likely differ — flagged as drift.

KS makes no assumption about the shape of the distribution, so it catches
shifts a simple mean/variance comparison would miss. Limitation: only
works on continuous numeric features, and becomes hypersensitive
("everything looks significant") at very large sample sizes.

### Population Stability Index (PSI)
Bins the baseline feature into deciles (bin edges are frozen from the
baseline and reused for the live batch — critical detail, otherwise
you're comparing a moving target to itself). For each bin, computes the
share of the baseline vs. the live population, then sums:


Industry-standard interpretation:
| PSI value | Meaning |
|---|---|
| < 0.10 | Stable — no meaningful shift |
| 0.10 – 0.25 | Moderate shift — worth investigating |
| ≥ 0.25 | Major shift — model likely needs review/retraining |

PSI works for categorical features too (bins = category labels), and
gives one interpretable number with fixed, business-friendly thresholds —
which is why it's the standard choice for dashboards and alerting,
alongside the more statistically rigorous KS test.

### Why run both?
KS is a formal hypothesis test but is continuous-only and can be
oversensitive at scale. PSI is a magnitude-based heuristic that works on
any feature type but has somewhat arbitrary bin boundaries. Running both
covers each one's blind spot.

## Architecture

app/
├── stats/
│ ├── ks_test.py # KS drift statistic + p-value
│ └── psi.py # PSI calculation (continuous + categorical)
├── monitors.py # missing-value and outlier ratio checks
├── drift_detector.py # orchestrates stats + monitors → per-feature verdict
├── reference_store.py # baseline (reference) distribution storage
├── exceptions.py # typed domain errors → mapped to HTTP codes
└── api.py # FastAPI routes + centralized error handling
tests/ # unit + integration tests
main.py # uvicorn entrypoint


**Design principle:** `stats/` and `monitors.py` are pure functions —
numpy arrays in, dataclasses out — with zero HTTP or framework
dependency, so the actual math is unit-testable in isolation.
`drift_detector.py` is the only place business rules live (e.g. "PSI
major → alert"). `api.py` is a thin HTTP layer on top. This separation
means the storage backend, web framework, or verdict thresholds can each
change independently without touching the others.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reference` | Register/overwrite the baseline for a `model_id` |
| `POST` | `/monitor` | Compare a live batch against the stored baseline |

### Register a baseline
```bash
curl -X POST http://127.0.0.1:8000/reference \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn_v1",
    "features": {"income": [50000, 52000, ...], "region": ["north","south",...]},
    "feature_types": {"income": "numeric", "region": "categorical"}
  }'
```

### Check a live batch
```bash
curl -X POST http://127.0.0.1:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn_v1",
    "features": {"income": [61000, 59500, ...], "region": ["north","east",...]}
  }'
```
Response includes per-feature PSI/KS values, severity, and an
`overall_verdict` of `ok` / `warn` / `alert` — the worst verdict across
all monitored features.

## Error handling

1. **Schema validation** (Pydantic) rejects malformed requests at the edge.
2. **Typed domain exceptions** (`app/exceptions.py`) carry their own HTTP
   status code and machine-readable `error_code`, mapped by a single
   global FastAPI exception handler.
3. **Catch-all handler** logs the full traceback server-side but returns
   a generic 500 to the caller — no internals leaked.
4. Statistics functions raise `InsufficientDataError` rather than
   crashing on tiny/degenerate batches.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Interactive API docs: http://127.0.0.1:8000/docs

Run tests:
```bash
pytest
```

## What's out of scope (by design)

- **Concept drift** — needs ground-truth labels and a labeling-delay
  pipeline; this service only handles feature-level drift.
- **Alerting integration** (Slack/PagerDuty) — the `/monitor` response is
  designed to be easy to wire into one, not to include one.
- **Scheduling** — this is a stateless check-on-demand API; a cron job
  would call `/monitor` periodically in a real deployment.
- **History/dashboarding** — the reference store persists baselines, not
  a time series of past monitor runs.
