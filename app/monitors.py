import numpy as np

def missing_ratio(values):
    if len(values) == 0:
        return 0.0
    arr = np.array(values, dtype=object)
    missing = sum(1 for v in arr if v is None or (isinstance(v, float) and np.isnan(v)))
    return missing / len(arr)

def outlier_ratio(values, z_threshold=4.0):
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return 0.0
    std = clean.std()
    if std == 0:
        return 0.0
    z_scores = np.abs((clean - clean.mean()) / std)
    return float((z_scores > z_threshold).sum() / len(clean))