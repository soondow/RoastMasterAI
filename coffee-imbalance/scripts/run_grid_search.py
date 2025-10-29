import pandas as pd
from pathlib import Path
from src.config import PROCESSED, N_REPEATS, N_SPLITS, SEED
from src.models.tabular_cv import grid_runner
from src.oversampling.smote import SMOTESampler, smote_default_grid
from src.oversampling.racog import RACOGSampler, racog_default_grid

df = pd.read_csv(PROCESSED / "features_table.csv")
assert "label" in df.columns, "features_table.csv에 label 컬럼(0/1)을 추가하세요."
X, y = df.drop(columns=["label"]), df["label"].astype("category")

mode = "smote"   # ← "racog" 로 변경 가능

if mode == "smote":
    sampler = SMOTESampler()
    grid = smote_default_grid()
    def filt(p): return p["k_neighbors"] >= 1
else:
    sampler = RACOGSampler()
    grid = racog_default_grid()
    def filt(p):
        s = p["strategy"]
        if s == "ratio" and p.get("target_ratio") is None: return False
        if s == "multiplier" and p.get("multiplier") is None: return False
        return True

full, summary = grid_runner(X, y, sampler, grid, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS, filter_fn=filt)

out_dir = PROCESSED / f"{mode}_grid"
Path(out_dir).mkdir(parents=True, exist_ok=True)
full.to_csv(out_dir / "cv_full.csv", index=False)
summary.to_csv(out_dir / "summary.csv", index=False)
print(f"saved to {out_dir}")
print(summary.groupby("model").head(3))
