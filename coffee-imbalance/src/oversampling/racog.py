import warnings, logging
import numpy as np, pandas as pd
from dataclasses import dataclass
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
pandas2ri.activate()

from .base import ISampler
logger = logging.getLogger(__name__)

@dataclass
class RACOGParams:
    burnin: int = 200
    lag: int = 50
    strategy: str = "equal"        # equal|ratio|multiplier
    target_ratio: float | None = None
    multiplier: float | None = None
    target_name: str = "target"
    positive: str | int | None = None

def _import_imbalance():
    try:
        return rpackages.importr('imbalance')
    except Exception as e:
        raise RuntimeError("R 패키지 'imbalance' 미설치. 설치 셀을 먼저 실행하세요.") from e

def _calc_num_instances(vc: pd.Series, p: RACOGParams, minority):
    n_min = int(vc.loc[minority]); n_maj = int(vc.max())
    if p.strategy == "equal":
        return max(n_maj - n_min, 0)
    if p.strategy == "ratio":
        r = float(p.target_ratio or 1.0)
        return max(int(round(r * n_maj - n_min)), 0)
    if p.strategy == "multiplier":
        m = float(p.multiplier or 1.0)
        return max(int(round(n_min * m) - n_min), 0)
    return max(n_maj - n_min, 0)

class RACOGSampler(ISampler):
    name = "racog"
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, **params):
        p = RACOGParams(**params)
        X = X.copy(); y = y.copy().astype("category")
        vc = y.value_counts()
        minority = vc.idxmin()
        positive = p.positive if p.positive is not None else minority
        num_instances = _calc_num_instances(vc, p, minority)
        if num_instances <= 0:
            warnings.warn("[RACOG] num_instances <= 0 → skipping oversampling.")
            return X, y

        df = X.copy(); df[p.target_name] = y.astype(str).values
        try:
            imbalance = _import_imbalance()
            r_df = pandas2ri.py2rpy(df)
            new_samples_r = imbalance.racog(
                r_df,
                classAttr=p.target_name,
                numInstances=ro.IntVector([num_instances]),
                burnin=ro.IntVector([p.burnin]),
                lag=ro.IntVector([p.lag]),
                positive=ro.StrVector([str(positive)])
            )
            new_samples = pandas2ri.rpy2py(new_samples_r)
            X_new = new_samples.drop(columns=[p.target_name])
            y_new = new_samples[p.target_name].astype(str)
            X_res = pd.concat([X, X_new], axis=0, ignore_index=True)
            y_res = pd.concat([y.astype(str), y_new], axis=0, ignore_index=True).astype("category")
            return X_res, y_res
        except Exception as e:
            msg = f"[RACOG] failed with params={p}: {e}"
            warnings.warn(msg)
            logger.error(msg)
            return None, None

def racog_default_grid():
    return {
        "burnin": [100, 200, 300],
        "lag": [20, 50, 80],
        "strategy": ["equal", "ratio", "multiplier"],
        "target_ratio": [1.0, 0.8],
        "multiplier": [1.5, 2.0],
    }
