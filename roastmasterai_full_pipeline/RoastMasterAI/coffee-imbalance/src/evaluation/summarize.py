import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def cliffs_delta(x, y):
    # x vs y effect size; return delta
    nx, ny = len(x), len(y)
    greater = 0; lesser = 0
    for xi in x:
        greater += (xi > y).sum()
        lesser  += (xi < y).sum()
    delta = (greater - lesser) / (nx * ny)
    return delta

def pairwise_wilcoxon(df: pd.DataFrame, metric: str, model_col: str='model'):
    models = df[model_col].unique().tolist()
    out = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            a, b = models[i], models[j]
            xa = df.loc[df[model_col]==a, metric].values
            xb = df.loc[df[model_col]==b, metric].values
            if len(xa)==len(xb) and len(xa)>=5:
                stat, p = wilcoxon(xa, xb, zero_method='wilcox', correction=True, alternative='two-sided', mode='auto')
            else:
                # Fallback to independent if lengths differ
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(xa, xb, alternative='two-sided')
            cd = cliffs_delta(pd.Series(xa), pd.Series(xb))
            out.append({"A": a, "B": b, "p_value": p, "cliffs_delta": cd})
    res = pd.DataFrame(out)
    # Bonferroni correction
    if not res.empty:
        m = len(res)
        res["p_adj_bonf"] = np.minimum(res["p_value"] * m, 1.0)
    return res
