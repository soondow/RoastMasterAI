import numpy as np, pandas as pd

def trim_and_resample(df, meta, start_event="CHARGE", end_event="DROP", resample_sec=1.0, cols=("BT","ET","RoR")):
    t0 = meta.get(start_event, df["seconds"].min())
    t1 = meta.get(end_event, df["seconds"].max())
    cut = df[(df["seconds"]>=t0) & (df["seconds"]<=t1)].copy()

    new_idx = np.arange(int(np.floor(t0)), int(np.ceil(t1))+1, resample_sec)
    out = pd.DataFrame(index=new_idx); out.index.name = "seconds"
    for c in cols:
        out[c] = np.interp(new_idx, cut["seconds"].values, cut[c].astype(float).values) if c in cut.columns else np.nan
    return out.reset_index()
