import pandas as pd
from src.config import PROCESSED, N_REPEATS, N_SPLITS, SEED
from src.models.tabular_cv import run_cv_repeated
from src.oversampling.smote import SMOTESampler
from src.oversampling.racog import RACOGSampler
from src.evaluation.summarize import summarize, wilcoxon_corrected

df = pd.read_csv(PROCESSED / "features_table.csv")
assert "label" in df.columns, "features_table.csv에 label 컬럼(0/1)을 추가하세요."
X, y = df.drop(columns=["label"]), df["label"].astype("category")

results = {}
results["baseline"] = run_cv_repeated(X, y, sampler=None, sampler_params={}, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS)
results["smote"]    = run_cv_repeated(X, y, sampler=SMOTESampler(), sampler_params={"k_neighbors":5,"sampling_strategy":"auto"}, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS)
results["racog"]    = run_cv_repeated(X, y, sampler=RACOGSampler(), sampler_params={"burnin":200,"lag":50,"strategy":"equal"}, repeats=N_REPEATS, seed=SEED, n_splits=N_SPLITS)

for k,v in results.items():
    print(f"\n== {k.upper()} ==")
    print(summarize(v))

print("\nWilcoxon+Bonferroni (F1): SMOTE vs BASE")
print(wilcoxon_corrected(results["smote"], results["baseline"], metric="F1"))
print("\nWilcoxon+Bonferroni (F1): RACOG vs BASE")
print(wilcoxon_corrected(results["racog"], results["baseline"], metric="F1"))
print("\nWilcoxon+Bonferroni (F1): RACOG vs SMOTE")
print(wilcoxon_corrected(results["racog"], results["smote"], metric="F1"))
