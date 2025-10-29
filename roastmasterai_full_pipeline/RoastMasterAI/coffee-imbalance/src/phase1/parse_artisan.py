from pathlib import Path
import pandas as pd

def parse_artisan_csv(path: str):
    """Minimal parser for Artisan-like CSV with columns: seconds, BT, ET, RoR.
    Returns dict(meta, data).
    """
    p = Path(path)
    df = pd.read_csv(p)
    # Normalize expected columns
    rename_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename_map)
    if "seconds" not in df.columns:
        # Attempt common alternatives
        for alt in ["time_sec", "time", "Time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "seconds"}); break
    required = {"seconds","BT","ET","RoR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df = df[["seconds","BT","ET","RoR"]].sort_values("seconds").reset_index(drop=True)
    meta = {"file_stem": p.stem, "path": str(p)}
    return {"meta": meta, "data": df}
