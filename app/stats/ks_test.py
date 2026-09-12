from dataclasses import dataclass
import numpy as np
from scipy import stats as scipy_stats

MIN_SAMPLES_FOR_KS = 20

class InsufficientDataError(Exception):
    pass

@dataclass
class KSResult:
    feature: str
    statistic: float
    p_value: float
    is_drifted: bool

def run_ks_test(feature, baseline, live, p_threshold=0.05):
    baseline = _clean(baseline)
    live = _clean(live)
    if len(baseline) < MIN_SAMPLES_FOR_KS or len(live) < MIN_SAMPLES_FOR_KS:
        raise InsufficientDataError(f"{feature}: need >= {MIN_SAMPLES_FOR_KS} samples")
    stat, p = scipy_stats.ks_2samp(baseline, live)
    return KSResult(feature, float(stat), float(p), bool(p < p_threshold))

def _clean(arr):
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]