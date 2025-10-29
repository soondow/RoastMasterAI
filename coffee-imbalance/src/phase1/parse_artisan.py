import pandas as pd, numpy as np, re
from ..utils import hhmm_to_sec

def parse_artisan_csv(path: str) -> dict:
    raw = pd.read_csv(path, header=None, dtype=str, encoding_errors="ignore")
    if raw.shape[1] != 1:
        return {"meta": {}, "data": raw}

    lines = raw.iloc[:,0].tolist()
    meta_line, header_line, rows = lines[0], lines[1], lines[2:]

    tokens = re.split(r"\t| +", meta_line.strip())
    meta = {}
    for tok in tokens:
        if ":" in tok:
            k, v = tok.split(":", 1)
            meta[k.upper()] = v

    header = header_line.split("\t")
    df = pd.DataFrame([ln.split("\t") for ln in rows], columns=header).replace({"": np.nan})

    time_col = "Time1" if "Time1" in df.columns else next((c for c in df.columns if c.lower().startswith("time")), df.columns[0])
    df["seconds"] = df[time_col].apply(hhmm_to_sec)

    for c in ("ET","BT"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("seconds")
    if "BT" in df.columns:
        df["RoR"] = df["BT"].diff() / df["seconds"].diff()

    events_sec = {}
    for k,v in meta.items():
        if k in {"CHARGE","TP","DRYE","FCS","FCE","SCS","SCE","DROP"}:
            sec = hhmm_to_sec(v)
            if pd.notna(sec): events_sec[k] = sec

    return {"meta": events_sec, "data": df}
