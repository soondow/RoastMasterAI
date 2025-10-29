import numpy as np
import pandas as pd

def summarize_features(ts: pd.DataFrame, meta: dict[str, str]|None=None) -> pd.Series:
    feats = {}
    for col in ["BT","ET","RoR"]:
        s = ts[col].astype(float)
        feats[f"{col.lower()}_mean"] = s.mean()
        feats[f"{col.lower()}_std"] = s.std(ddof=1)
        feats[f"{col.lower()}_min"] = s.min()
        feats[f"{col.lower()}_max"] = s.max()
        feats[f"{col.lower()}_p95"] = s.quantile(0.95)
    if meta and "file_stem" in meta:
        feats["file_stem"] = meta["file_stem"]
    return pd.Series(feats)

def features_from_csvs(csv_paths: list[str]) -> pd.DataFrame:
    from .parse_artisan import parse_artisan_csv
    from .preprocess import trim_and_resample
    rows = []
    for path in csv_paths:
        parsed = parse_artisan_csv(path)
        ts = trim_and_resample(parsed["data"], 0, None, 1.0)
        rows.append(summarize_features(ts, parsed["meta"]))
    return pd.DataFrame(rows)
