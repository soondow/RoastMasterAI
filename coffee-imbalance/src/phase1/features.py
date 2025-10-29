import numpy as np, pandas as pd

def summarize_features(df_ts, meta):
    return pd.DataFrame([{
        "bt_mean":  df_ts["BT"].mean(),
        "bt_max":   df_ts["BT"].max(),
        "bt_min":   df_ts["BT"].min(),
        "ror_mean": df_ts["RoR"].mean(),
        "ror_max":  df_ts["RoR"].max(),
        "ror_min":  df_ts["RoR"].min(),
        "et_mean":  df_ts["ET"].mean(),
        "et_max":   df_ts["ET"].max(),
        "duration": df_ts["seconds"].iloc[-1] - df_ts["seconds"].iloc[0],
        "fc_time":  meta.get("FCS", np.nan),
        "drop_time":meta.get("DROP", np.nan),
        "file_stem": meta.get("FILENAME", None)
    }])
