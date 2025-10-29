# Train the final tabular model with NO leakage and save artifacts.
import json, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.config import PROCESSED

df = pd.read_csv(PROCESSED / "features_table.csv")
assert "label" in df.columns, "features_table.csv에 label(0/1)이 필요합니다."
X, y = df.drop(columns=["label"]), df["label"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_params = dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)

clf = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
clf.fit(X_scaled, y)

artifacts = {"scaler": scaler, "model": clf}
joblib.dump(artifacts, PROCESSED / "final_tabular.pkl")

with open(PROCESSED / "final_config.json", "w") as f:
    json.dump({"model":"XGBClassifier", "params": clf.get_params()}, f, indent=2)

print("Saved:", PROCESSED / "final_tabular.pkl")
