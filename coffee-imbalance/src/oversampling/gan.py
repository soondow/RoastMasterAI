import pandas as pd
from .base import ISampler

class WGANStubSampler(ISampler):
    name = "wgan_gp_stub"
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, **params):
        print("[WGAN-GP] Stub: returning original data (no-op).")
        return X.copy(), y.copy()

def wgan_default_grid():
    return {
        "epochs": [2000],
        "lr": [1e-4],
        "latent_dim": [32, 64],
        "hidden": [128],
    }
