import numpy as np
from app.stats.ks_test import run_ks_test

rng = np.random.default_rng(42)
baseline = rng.normal(0, 1, 500)
live_same = rng.normal(0, 1, 500)
live_shifted = rng.normal(5, 1, 500)

print("SAME distribution:", run_ks_test("age", baseline, live_same))
print("SHIFTED distribution:", run_ks_test("age", baseline, live_shifted))