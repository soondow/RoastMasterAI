import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def summarize(df):
    if df is None or len(df)==0:
        return pd.DataFrame()
    return (df.groupby("model")
            .agg(F1_mean=("F1","mean"), PR_AUC=("PR_AUC","mean"), ROC_AUC=("ROC_AUC","mean"))
            .sort_values("F1_mean", ascending=False))

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x>y for x in a for y in b)
    lt = sum(x<y for x in a for y in b)
    n1, n2 = len(a), len(b)
    return (gt - lt) / (n1*n2 + 1e-9)

def wilcoxon_corrected(df_a, df_b, metric="F1"):
    models = sorted(set(df_a.get("model",[])) & set(df_b.get("model",[])))
    p_vals, deltas, used_models = [], [], []
    for mdl in models:
        a = df_a[df_a["model"]==mdl][metric].values
        b = df_b[df_b["model"]==mdl][metric].values
        if len(a) != len(b) or len(a)==0:
            continue
        _, p = wilcoxon(a, b)
        p_vals.append(p); deltas.append(cliffs_delta(a,b)); used_models.append(mdl)
    if not p_vals:
        return pd.DataFrame(columns=["model","p","p_adj","cliffs_delta","significant"])
    reject, p_adj, _, _ = multipletests(p_vals, method="bonferroni")
    return pd.DataFrame({"model": used_models, "p": p_vals, "p_adj": p_adj, "cliffs_delta": deltas, "significant": reject})

def bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    xs = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        xs.append(np.mean(s))
    lo = np.percentile(xs, 100*alpha/2)
    hi = np.percentile(xs, 100*(1-alpha/2))
    return lo, hi
