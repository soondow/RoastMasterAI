import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from src.config import PROCESSED, MODELS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(MODELS / "final_xgb_smote.joblib"))
    ap.add_argument("--features", required=False, help="external features csv")
    ap.add_argument("--labels", required=False, help="external labels csv (optional)")
    args = ap.parse_args()

    pack = joblib.load(args.model_path)
    scaler, clf = pack["scaler"], pack["model"]

    if args.features:
        df = pd.read_csv(args.features)
    else:
        df = pd.read_csv(PROCESSED / "features_table.csv")
    X = df.drop(columns=[c for c in ["file_stem","label"] if c in df.columns], errors="ignore").values
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[:,1]
    pred = (prob >= 0.5).astype(int)  # default threshold for external quick check

    print("Pred head:", pred[:10], "Prob head:", prob[:10])
    if args.labels and Path(args.labels).exists():
        y = pd.read_csv(args.labels)["label"].astype(int).values
        ap = average_precision_score(y, prob)
        roc = roc_auc_score(y, prob)
        print("External AP:", ap, "ROC_AUC:", roc)
        print(classification_report(y, pred, digits=4))

if __name__ == "__main__":
    main()
