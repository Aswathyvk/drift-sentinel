import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.api import app, store

@pytest.fixture(autouse=True)
def clean_store():
    store._data.clear()
    yield
    store._data.clear()

client = TestClient(app)

def _baseline_payload(model_id="model_a"):
    rng = np.random.default_rng(0)
    return {
        "model_id": model_id,
        "features": {"income": rng.normal(50000, 10000, 200).tolist()},
        "feature_types": {"income": "numeric"},
    }

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_register_then_monitor_stable_batch():
    client.post("/reference", json=_baseline_payload())
    rng = np.random.default_rng(0)
    live = {"model_id": "model_a", "features": {"income": rng.normal(50000, 10000, 200).tolist()}}
    resp = client.post("/monitor", json=live)
    assert resp.status_code == 200
    assert resp.json()["overall_verdict"] in ("ok", "warn")

def test_monitor_without_reference_returns_404():
    resp = client.post("/monitor", json={"model_id": "ghost", "features": {"income": [1,2,3]}})
    assert resp.status_code == 404

def test_drifted_batch_flags_alert():
    client.post("/reference", json=_baseline_payload("model_c"))
    live = {"model_id": "model_c", "features": {"income": (np.random.default_rng(1).normal(200000, 10000, 200)).tolist()}}
    resp = client.post("/monitor", json=live)
    assert resp.json()["overall_verdict"] == "alert"