import numpy as np
from app.monitors import missing_ratio, outlier_ratio
from app.stats.ks_test import run_ks_test, InsufficientDataError
from app.stats.psi import compute_psi_continuous

def verdict_for_feature(psi_severity, ks_drifted, missing, max_missing_ratio=0.05):
    if missing > max_missing_ratio:
        return "warn"
    if psi_severity == "major":
        return "alert"
    if psi_severity == "moderate" or ks_drifted:
        return "warn"
    return "ok"

def analyze_feature(name, baseline_values, live_values):
    missing = missing_ratio(live_values)
    baseline_arr = np.array(baseline_values, dtype=float)
    live_arr = np.array(live_values, dtype=float)

    ks_drifted = None
    try:
        ks_result = run_ks_test(name, baseline_arr, live_arr)
        ks_drifted = ks_result.is_drifted
    except InsufficientDataError:
        pass

    psi_severity = None
    psi_value = None
    try:
        psi_result = compute_psi_continuous(name, baseline_arr, live_arr)
        psi_severity = psi_result.severity
        psi_value = psi_result.psi
    except InsufficientDataError:
        pass

    verdict = verdict_for_feature(psi_severity, ks_drifted, missing)

    return {
        "feature": name,
        "psi": psi_value,
        "psi_severity": psi_severity,
        "ks_drifted": ks_drifted,
        "missing_ratio": missing,
        "verdict": verdict,
    }