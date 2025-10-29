import numpy as np
from ..utils import unit_norm
try:
    from pyts.image import GramianAngularField
except ImportError:
    GramianAngularField = None

def to_gaf(x_1d, image_size=128, method="summation"):
    if GramianAngularField is None:
        raise RuntimeError("pyts가 없습니다. pip install pyts")
    gaf = GramianAngularField(image_size=image_size, method=method)
    return gaf.fit_transform(x_1d.reshape(1,-1))[0][None, ...]

def build_gaf_3ch_from_df(df_ts, image_size=128, fill_ror=0.0):
    bt = unit_norm(df_ts["BT"].values)
    et = unit_norm(df_ts["ET"].values)
    ror= unit_norm(df_ts["RoR"].fillna(fill_ror).values)
    return np.concatenate([to_gaf(bt,image_size), to_gaf(et,image_size), to_gaf(ror,image_size)], axis=0)
