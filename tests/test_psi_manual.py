import numpy as np
from app.stats.psi import compute_psi_continuous

rng = np.random.default_rng(1)
baseline = rng.normal(0, 1, 1000)
live_same = baseline.copy()
live_shifted = rng.normal(3, 1, 1000)

print("SAME:", compute_psi_continuous("score", baseline, live_same))
print("SHIFTED:", compute_psi_continuous("score", baseline, live_shifted))
