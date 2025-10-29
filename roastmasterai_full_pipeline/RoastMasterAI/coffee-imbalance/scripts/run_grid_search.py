import argparse, itertools, json
import pandas as pd
from pathlib import Path
from src.config import PROCESSED, REPORTS, RANDOM_STATE
from src.models.tabular_cv import CVConfig, cv_run
from src.tracking.mlflow_utils import run_mlflow, log_metrics

XGB_GRID = {
  "n_estimators": [100, 300, 600],
  "max_depth": [3, 5, 7],
  "learning_rate": [0.05, 0.1],
  "subsample": [0.7, 0.9, 1.0],
  "colsample_bytree": [0.7, 0.9, 1.0],
  "reg_lambda": [1.0, 2.0, 5.0],
  "reg_alpha": [0.0, 0.5]
}

RF_GRID = {
  "n_estimators": [200, 500, 1000],
  "max_depth": [None, 5, 10],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4],
  "class_weight": [None, "balanced"]
}

LOGREG_GRID = {
  "C": [0.1, 1.0, 3.0],
  "penalty": ["l2"],
  "class_weight": [None, "balanced"]
}

def iter_grid(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(PROCESSED / "features_table.csv"))
    ap.add_argument("--labels", default=str(PROCESSED / "labels_map.csv"))
    ap.add_argument("--model", choices=["logreg","rf","xgb"], default="xgb")
    ap.add_argument("--sampler", choices=["none","smote","racog"], default="smote")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    Xy = pd.read_csv(args.features).merge(pd.read_csv(args.labels), on="file_stem")
    y = Xy["label"].astype(int).values
    X = Xy.drop(columns=["file_stem","label"]).values

    grid = {"xgb": XGB_GRID, "rf": RF_GRID, "logreg": LOGREG_GRID}[args.model]
    rows = []
    for i, params in enumerate(iter_grid(grid), 1):
        cfg = CVConfig(n_splits=args.splits, repeats=args.repeats, random_state=RANDOM_STATE,
                       sampler=args.sampler, model_name=args.model, params=params)
        with run_mlflow(run_name=f"grid_{args.model}", params={**cfg.__dict__}, tags={"stage":"grid"}):
            df = cv_run(X, y, cfg)
            df["config_id"] = i
            df["params_json"] = json.dumps(params)
            rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    out = PROCESSED / f"{args.model}_{args.sampler}_grid" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)
    print("Saved grid results:", out, all_df.shape)

if __name__ == "__main__":
    main()
