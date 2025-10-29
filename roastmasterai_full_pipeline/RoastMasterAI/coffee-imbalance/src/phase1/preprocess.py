import numpy as np
import pandas as pd

def trim_and_resample(df: pd.DataFrame, start_sec: float|int=0, end_sec: float|int|None=None, step: float=1.0):
    """Trim to [start_sec, end_sec] and resample to step seconds using nearest.
    """
    if end_sec is None:
        end_sec = df['seconds'].max()
    mask = (df['seconds'] >= start_sec) & (df['seconds'] <= end_sec)
    df = df.loc[mask].copy()
    new_sec = np.arange(start_sec, end_sec+1e-9, step)
    out = pd.DataFrame({"seconds": new_sec})
    out = out.merge(df, on="seconds", how="left")
    # fill forward/backward
    out[["BT","ET","RoR"]] = out[["BT","ET","RoR"]].interpolate().ffill().bfill()
    return out
