import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE
from .base import ISampler

class SMOTESampler(ISampler):
    name = "smote"
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, **params):
        k = params.get("k_neighbors", 5)
        min_count = y.value_counts().min()
        if k >= min_count:
            new_k = max(1, min_count - 1)
            warnings.warn(f"[SMOTE] k_neighbors {k} > minority {min_count} → adjusted to {new_k}")
            k = new_k
        strategy = params.get("sampling_strategy", "auto")
        random_state = params.get("random_state", 42)
        sm = SMOTE(k_neighbors=k, sampling_strategy=strategy, random_state=random_state)
        Xr, yr = sm.fit_resample(X, y)
        return pd.DataFrame(Xr, columns=X.columns), yr

def smote_default_grid():
    return {
        "k_neighbors": [3, 5, 7],
        "sampling_strategy": ["auto", 0.8, 1.0],
    }
