import argparse, json, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED, MODELS
from src.models.tabular_cv import build_model
from src.oversampling.smote_wrapper import get_smote
from src.oversampling.racog_wrapper import racog_fit_resample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(PROCESSED / "features_table.csv"))
    ap.add_argument("--labels", default=str(PROCESSED / "labels_map.csv"))
    ap.add_argument("--model", choices=["logreg","rf","xgb"], default="xgb")
    ap.add_argument("--sampler", choices=["none","smote","racog"], default="smote")
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    Xy = pd.read_csv(args.features).merge(pd.read_csv(args.labels), on="file_stem")
    y = Xy["label"].astype(int).values
    X = Xy.drop(columns=["file_stem","label"]).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if args.sampler == "smote":
        Xs, y = get_smote().fit_resample(Xs, y)
    elif args.sampler == "racog":
        Xs, y = racog_fit_resample(Xs, y)

    params = json.loads(args.params)
    clf = build_model(args.model, params)
    clf.fit(Xs, y)

    MODELS.mkdir(exist_ok=True, parents=True)
    joblib.dump({"scaler": scaler, "model": clf}, MODELS / f"final_{args.model}_{args.sampler}.joblib")
    print("Saved:", MODELS / f"final_{args.model}_{args.sampler}.joblib")

if __name__ == "__main__":
    main()
