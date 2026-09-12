from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.reference_store import ReferenceStore
from app.drift_detector import analyze_feature

app = FastAPI(title="ml-model-monitor")
store = ReferenceStore()

class RegisterReferenceRequest(BaseModel):
    model_id: str
    features: dict[str, list]
    feature_types: dict[str, str]

class MonitorBatchRequest(BaseModel):
    model_id: str
    features: dict[str, list]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reference", status_code=201)
def register_reference(payload: RegisterReferenceRequest):
    store.set_reference(payload.model_id, payload.features, payload.feature_types)
    return {"model_id": payload.model_id, "status": "reference_registered"}

@app.post("/monitor")
def monitor_batch(payload: MonitorBatchRequest):
    ref = store.get_reference(payload.model_id)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"No reference for model_id={payload.model_id}")

    reports = []
    for name, live_values in payload.features.items():
        baseline_values = ref["features"][name]
        reports.append(analyze_feature(name, baseline_values, live_values))

    overall = "ok"
    order = {"ok": 0, "warn": 1, "alert": 2}
    for r in reports:
        if order[r["verdict"]] > order[overall]:
            overall = r["verdict"]

    return {"model_id": payload.model_id, "overall_verdict": overall, "feature_reports": reports}