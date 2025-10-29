from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from ..oversampling.smote_wrapper import get_smote
from ..oversampling.racog_wrapper import racog_fit_resample

def optimal_f1_threshold(y_true, prob):
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    idx = np.nanargmax(f1s)
    return max(0.0, min(1.0, thr[idx]))

@dataclass
class CVConfig:
    n_splits: int = 5
    random_state: int = 42
    repeats: int = 2
    sampler: Literal["none","smote","racog"] = "smote"
    model_name: Literal["logreg","rf","xgb"] = "xgb"
    params: Optional[Dict[str, Any]] = None

def build_model(name: str, params: Optional[Dict[str, Any]]):
    if name == "logreg":
        base = LogisticRegression(max_iter=2000, n_jobs=None)
    elif name == "rf":
        base = RandomForestClassifier()
    elif name == "xgb":
        base = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=0)
    else:
        raise ValueError(name)
    if params:
        base.set_params(**params)
    return base

def cv_run(X: np.ndarray, y: np.ndarray, cfg: CVConfig, log_fn=None):
    rng = np.random.RandomState(cfg.random_state)
    metrics = []
    for rep in range(cfg.repeats):
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=rng.randint(0, 1_000_000))
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]

            scaler = StandardScaler()
            X_trs = scaler.fit_transform(X_tr)
            X_vas = scaler.transform(X_va)

            # Sampling
            if cfg.sampler == "smote":
                sampler = get_smote(cfg.random_state)
                X_trs, y_tr = sampler.fit_resample(X_trs, y_tr)
            elif cfg.sampler == "racog":
                X_trs, y_tr = racog_fit_resample(X_trs, y_tr)

            clf = build_model(cfg.model_name, cfg.params)
            clf.fit(X_trs, y_tr)

            prob = clf.predict_proba(X_vas)[:,1]
            thr = optimal_f1_threshold(y_va, prob)
            y_pred = (prob >= thr).astype(int)

            f1 = f1_score(y_va, y_pred)
            pr_auc = average_precision_score(y_va, prob)
            roc = roc_auc_score(y_va, prob)

            row = {"rep": rep, "fold": fold, "thr": thr, "F1": f1, "PR_AUC": pr_auc, "ROC_AUC": roc, "model": cfg.model_name, "sampler": cfg.sampler}
            metrics.append(row)
            if log_fn:
                log_fn(row)
    return pd.DataFrame(metrics)
