from dataclasses import dataclass
import numpy as np
from app.stats.ks_test import InsufficientDataError

MIN_SAMPLES_FOR_PSI = 30
EPSILON = 1e-6

@dataclass
class PSIResult:
    feature: str
    psi: float
    severity: str  # "stable" | "moderate" | "major"

def _severity(psi, warn_threshold, alert_threshold):
    if psi >= alert_threshold:
        return "major"
    if psi >= warn_threshold:
        return "moderate"
    return "stable"

def compute_psi_continuous(feature, baseline, live, bin_count=10, warn_threshold=0.1, alert_threshold=0.25):
    baseline = _clean(baseline)
    live = _clean(live)

    if len(baseline) < MIN_SAMPLES_FOR_PSI or len(live) < MIN_SAMPLES_FOR_PSI:
        raise InsufficientDataError(f"{feature}: need >= {MIN_SAMPLES_FOR_PSI} samples")

    # bin edges frozen from BASELINE only — key correctness detail
    quantiles = np.linspace(0, 1, bin_count + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 3:
        edges = np.array([baseline.min() - EPSILON, baseline.max() + EPSILON])

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    baseline_pct = np.clip(baseline_counts / max(baseline_counts.sum(), 1), EPSILON, None)
    live_pct = np.clip(live_counts / max(live_counts.sum(), 1), EPSILON, None)

    per_bin_psi = (live_pct - baseline_pct) * np.log(live_pct / baseline_pct)
    total_psi = float(per_bin_psi.sum())

    return PSIResult(feature, total_psi, _severity(total_psi, warn_threshold, alert_threshold))

def _clean(arr):
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]