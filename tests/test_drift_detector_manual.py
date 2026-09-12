import numpy as np
from app.drift_detector import analyze_feature

rng = np.random.default_rng(7)
baseline = rng.normal(50, 10, 300).tolist()
live_stable = rng.normal(50, 10, 300).tolist()
live_drifted = rng.normal(80, 10, 300).tolist()

print("STABLE:", analyze_feature("income", baseline, live_stable))
print("DRIFTED:", analyze_feature("income", baseline, live_drifted))