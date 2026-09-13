import numpy as np
import pytest
from app.stats.ks_test import run_ks_test
from app.stats.ks_test import InsufficientDataError

def test_identical_distributions_not_drifted():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 500)
    live = rng.normal(0, 1, 500)
    result = run_ks_test("age", baseline, live)
    assert result.is_drifted is False

def test_shifted_distribution_is_drifted():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 500)
    live = rng.normal(5, 1, 500)
    result = run_ks_test("age", baseline, live)
    assert result.is_drifted is True
    assert result.statistic > 0.5

def test_insufficient_samples_raises():
    baseline = np.array([1.0, 2.0, 3.0])
    live = np.array([1.0, 2.0, 3.0])
    with pytest.raises(InsufficientDataError):
        run_ks_test("tiny", baseline, live)