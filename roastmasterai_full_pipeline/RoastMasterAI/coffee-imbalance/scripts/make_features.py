import argparse
from pathlib import Path
import pandas as pd
from src.config import RAW, PROCESSED
from src.phase1.features import features_from_csvs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(RAW / "*.csv"))
    ap.add_argument("--out", default=str(PROCESSED / "features_table.csv"))
    args = ap.parse_args()

    paths = [str(p) for p in Path().glob(args.glob)]
    if not paths:
        paths = [str(p) for p in (RAW).glob("*.csv")]
    if not paths:
        raise SystemExit("No CSV files found under data/raw")
    df = features_from_csvs(paths)
    df.to_csv(args.out, index=False)
    print("Saved", args.out, df.shape)

if __name__ == "__main__":
    main()
