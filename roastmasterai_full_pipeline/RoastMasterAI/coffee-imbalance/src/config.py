from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]  # .../coffee-imbalance/src -> repo root
DATA = ROOT / "coffee-imbalance" / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "coffee-imbalance" / "models"
REPORTS = DATA / "reports"

# Defaults
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 2

PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
