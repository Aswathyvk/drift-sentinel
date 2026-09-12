import numpy as np
from app.monitors import missing_ratio, outlier_ratio

vals = [1, 2, None, 4, 5, float('nan'), 7]
print("missing_ratio:", missing_ratio(vals))

clean_vals = np.array([1,2,3,4,5,100])  # 100 = outlier
print("outlier_ratio:", outlier_ratio(clean_vals, z_threshold=2.0))