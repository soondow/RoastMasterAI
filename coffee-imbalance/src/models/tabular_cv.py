import numpy as np, pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm

from ..config import N_SPLITS, N_REPEATS, SEED

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def get_models_small():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, n_jobs=-1, eval_metric="logloss"),
        "Logistic": LogisticRegression(max_iter=500, C=0.1, penalty="l2", n_jobs=-1, random_state=42),
    }

def _eval_binary(y_true, y_prob, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        "PR_AUC": average_precision_score(y_true, y_prob) if y_prob is not None else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def run_cv_once(X, y, sampler, sampler_params: dict, seed=SEED, n_splits=N_SPLITS):
    models = get_models_small()
    recs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()

        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=X.columns)
        Xte = pd.DataFrame(scaler.transform(Xte), columns=X.columns)

        if sampler is not None:
            X_os, y_os = sampler.fit_resample(Xtr, ytr, **(sampler_params or {}))
            if X_os is None or y_os is None:
                continue
        else:
            X_os, y_os = Xtr, ytr

        for name, clf in models.items():
            pipe = Pipeline([("clf", clf)])
            pipe.fit(X_os, y_os)
            prob = pipe.predict_proba(Xte)[:,1] if hasattr(pipe, "predict_proba") else None
            pred = pipe.predict(Xte)
            m = _eval_binary(yte, prob, pred)
            m.update({"fold":fold,"model":name})
            recs.append(m)
    return pd.DataFrame(recs)

def run_cv_repeated(X, y, sampler, sampler_params: dict, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS):
    recs=[]
    rng = np.random.RandomState(seed)
    for rep in range(repeats):
        df = run_cv_once(X, y, sampler, sampler_params, seed=rng.randint(0,10**9), n_splits=n_splits)
        df["rep"] = rep
        recs.append(df)
    out = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    return out

def grid_runner(X, y, sampler, param_grid: dict, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS, filter_fn=None):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = []
    for vtuple in __import__("itertools").product(*vals):
        params = dict(zip(keys, vtuple))
        if filter_fn and (not filter_fn(params)):
            continue
        combos.append(params)

    all_rows=[]
    for gid, params in enumerate(tqdm(combos, desc="Grid Search")):
        df = run_cv_repeated(X, y, sampler, params, repeats=repeats, seed=seed, n_splits=n_splits)
        if df is None or len(df)==0:
            continue
        for k,v in params.items(): df[k] = v
        df["grid_id"] = gid
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    full = pd.concat(all_rows, ignore_index=True)
    summary = (full.groupby(["model","grid_id"] + keys)
                    .agg(F1=("F1","mean"), PR_AUC=("PR_AUC","mean"), ROC_AUC=("ROC_AUC","mean"))
                    .reset_index()
                    .sort_values(["model","F1"], ascending=[True, False]))
    return full, summary
