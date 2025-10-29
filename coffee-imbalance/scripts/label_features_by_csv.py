import pandas as pd
from src.config import PROCESSED

FEATURES = PROCESSED / "features_table.csv"
LABELS = PROCESSED / "labels_map.csv"  # columns=[file_stem,label]

df_feat = pd.read_csv(FEATURES)
df_map = pd.read_csv(LABELS)

assert "file_stem" in df_feat.columns, "features_table.csv에 file_stem 컬럼이 필요합니다."
assert "file_stem" in df_map.columns and "label" in df_map.columns, "labels_map.csv에 file_stem,label 컬럼이 필요합니다."

df = df_feat.merge(df_map, on="file_stem", how="left")
assert "label" in df.columns, "merge 실패: label 없음"
df.to_csv(PROCESSED / "features_table.csv", index=False)
print("labeled:", PROCESSED / "features_table.csv")
