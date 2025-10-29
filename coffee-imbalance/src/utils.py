import re, numpy as np, pandas as pd

_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?")

def hhmm_to_sec(s: str) -> float:
    if not isinstance(s, str) or not s.strip(): return np.nan
    m = _TIME_RE.match(s.strip())
    if not m: return np.nan
    h, m_, s_ = int(m.group("h")), int(m.group("m")), int(m.group("s") or 0)
    return h*60 + m_ if h < 60 else h*3600 + m_*60 + s_

def unit_norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

def safe_minority(y: pd.Series):
    vc = y.value_counts()
    return vc.idxmin(), vc.min(), vc.max()

def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
