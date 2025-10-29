# Evaluate on external set with NO leakage (uses stored scaler/model).
import joblib, pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from src.config import PROCESSED

art = joblib.load(PROCESSED / "final_tabular.pkl")
scaler, clf = art["scaler"], art["model"]

df_ext = pd.read_csv(PROCESSED / "external_features.csv")
has_label = "label" in df_ext.columns
X_ext = df_ext.drop(columns=["label"], errors="ignore")
X_ext_t = scaler.transform(X_ext)  # transform ONLY

if has_label:
    y_ext = df_ext["label"].astype(int).values
    y_prob = clf.predict_proba(X_ext_t)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    print("[External Report]")
    print(classification_report(y_ext, y_pred, digits=4))
    print("ROC_AUC:", roc_auc_score(y_ext, y_prob))
    print("PR_AUC :", average_precision_score(y_ext, y_prob))
    print("F1     :", f1_score(y_ext, y_pred))
else:
    y_prob = clf.predict_proba(X_ext_t)[:,1]
    print("[External] No labels provided. Positive rate:", (y_prob>=0.5).mean())
