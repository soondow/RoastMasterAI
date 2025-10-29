from pathlib import Path
import numpy as np, pandas as pd
from src.config import RAW, PROCESSED, START_EVENT, END_EVENT, RESAMPLE_SEC, GAF_SIZE
from src.utils import ensure_dirs
from src.phase1.parse_artisan import parse_artisan_csv
from src.phase1.preprocess import trim_and_resample
from src.phase1.features import summarize_features
from src.phase1.gaf import build_gaf_3ch_from_df

ensure_dirs(RAW, PROCESSED)

rows=[]
for csv in RAW.glob("*.csv"):
    parsed = parse_artisan_csv(str(csv))
    meta, df_raw = parsed["meta"], parsed["data"]
    meta = {"FILENAME": csv.stem, **meta}
    df_ts = trim_and_resample(df_raw, meta, START_EVENT, END_EVENT, RESAMPLE_SEC, ("BT","ET","RoR"))
    rows.append(summarize_features(df_ts, meta))
    np.save(PROCESSED / f"{csv.stem}_gaf3ch_{GAF_SIZE}.npy", build_gaf_3ch_from_df(df_ts, image_size=GAF_SIZE))

df_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
df_all.to_csv(PROCESSED / "features_table.csv", index=False)
print("saved:", PROCESSED / "features_table.csv")
