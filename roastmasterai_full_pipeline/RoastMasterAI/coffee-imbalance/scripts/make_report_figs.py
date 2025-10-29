import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("data/processed")

def load_summary_paths():
    # Prefer grid summaries, fallback to single summary
    paths = list(PROCESSED.glob("*_grid/summary.csv"))
    return paths or list(PROCESSED.glob("summary.csv"))

def plot_box(df, metric="F1", out="box_f1.png"):
    plt.figure()
    df.boxplot(column=metric, by="model")
    plt.suptitle("")
    plt.title(f"{metric} by model"); plt.xlabel("model"); plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(PROCESSED / out, dpi=150)

def main():
    paths = load_summary_paths()
    if not paths:
        print("No summary.csv found under data/processed/")
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "model" not in df.columns or "F1" not in df.columns:
        print("summary.csv must contain columns: model, F1, ...")
        return
    plot_box(df, "F1", "box_f1.png")
    summary = df.groupby("model")["F1"].agg(["mean","std","count"]).sort_values("mean", ascending=False)
    summary.to_csv(PROCESSED / "table_f1_summary.csv")
    print("Saved figures/tables under:", PROCESSED)

if __name__ == "__main__":
    main()
