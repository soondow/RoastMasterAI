import argparse
import pandas as pd
from pathlib import Path
from src.config import PROCESSED, REPORTS, RANDOM_STATE, N_SPLITS, N_REPEATS
from src.models.tabular_cv import CVConfig, cv_run
from src.tracking.mlflow_utils import run_mlflow, log_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(PROCESSED / "features_table.csv"))
    ap.add_argument("--labels", default=str(PROCESSED / "labels_map.csv"))
    ap.add_argument("--model", choices=["logreg","rf","xgb"], default="xgb")
    ap.add_argument("--sampler", choices=["none","smote","racog"], default="smote")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    Xy = pd.read_csv(args.features).merge(pd.read_csv(args.labels), on="file_stem")
    y = Xy["label"].astype(int).values
    X = Xy.drop(columns=["file_stem","label"]).values

    cfg = CVConfig(n_splits=args.splits, repeats=args.repeats, random_state=RANDOM_STATE,
                   sampler=args.sampler, model_name=args.model, params=None)

    rows = []
    with run_mlflow(run_name=f"cv_{args.model}_{args.sampler}", params=cfg.__dict__, tags={"stage":"cv"}):
        def log(row): log_metrics({"F1": row["F1"], "PR_AUC": row["PR_AUC"], "ROC_AUC": row["ROC_AUC"]})
        df = cv_run(X, y, cfg, log_fn=log)
        out = PROCESSED / f"cv_{args.model}_{args.sampler}.csv"
        df.to_csv(out, index=False)
        print("Saved:", out, df.shape)

        # summary
        summ = df.groupby("model")[["F1","PR_AUC","ROC_AUC"]].agg(["mean","std","count"])
        REPORTS.mkdir(exist_ok=True); 
        summ.to_csv(REPORTS / f"summary_{args.model}_{args.sampler}.csv")
        print("Summary saved:", REPORTS / f"summary_{args.model}_{args.sampler}.csv")

if __name__ == "__main__":
    main()
