from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

START_EVENT = "CHARGE"
END_EVENT = "DROP"
RESAMPLE_SEC = 1.0

GAF_SIZE = 128

N_SPLITS = 5
N_REPEATS = 10
SEED = 42

PRIMARY_METRIC = "F1"  # or "PR_AUC"
