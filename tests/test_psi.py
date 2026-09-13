import numpy as np
import pytest
from app.stats.psi import compute_psi_continuous
from app.stats.ks_test import InsufficientDataError

def test_identical_distribution_gives_near_zero_psi():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 1000)
    live = baseline.copy()
    result = compute_psi_continuous("score", baseline, live)
    assert result.psi < 0.01
    assert result.severity == "stable"

def test_major_shift_flags_alert():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 1000)
    live = rng.normal(3, 1, 1000)
    result = compute_psi_continuous("score", baseline, live)
    assert result.severity == "major"
    assert result.psi >= 0.25

def test_insufficient_samples_raises():
    baseline = np.array([1.0, 2.0])
    live = np.array([1.0, 2.0])
    with pytest.raises(InsufficientDataError):
        compute_psi_continuous("tiny", baseline, live)
        