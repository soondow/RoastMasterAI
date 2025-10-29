import sys
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
LABELS = PROCESSED / "labels_map.csv"
FEATS = PROCESSED / "features_table.csv"

def main():
    errs = 0
    if not LABELS.exists() or not FEATS.exists():
        print("[ERROR] expected files not found:", LABELS, FEATS)
        sys.exit(1)

    dfm = pd.read_csv(LABELS)
    dff = pd.read_csv(FEATS)
    for col in ["file_stem","label"]:
        if col not in dfm.columns:
            print(f"[ERROR] labels_map.csv missing column: {col}")
            errs += 1

    dups = dfm["file_stem"].duplicated(keep=False)
    if dups.any():
        print("[ERROR] duplicated file_stem:", dfm.loc[dups,"file_stem"].unique()[:10])
        errs += 1

    missing = set(dff["file_stem"]) - set(dfm["file_stem"])
    if missing:
        print(f"[ERROR] {len(missing)} files in features_table.csv not labeled. e.g., {list(missing)[:10]}")
        errs += 1

    bad = ~dfm["label"].astype(str).isin(["0","1"])
    if bad.any():
        print("[ERROR] invalid labels (must be 0/1):", dfm.loc[bad,"file_stem"].tolist()[:10])
        errs += 1

    if errs == 0:
        print("labels_map.csv OK.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
