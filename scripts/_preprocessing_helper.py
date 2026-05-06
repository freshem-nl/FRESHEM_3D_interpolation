import numpy as np
import pandas as pd

def snap_to_center_grid(v, step):
    # returns the nearest grid-center at/below v
    phase = step / 2
    return phase + np.floor((v - phase) / step) * step

def snap_to_center_grid_up(v, step):
    # returns the nearest grid-center at/above v
    phase = step / 2
    return phase + np.ceil((v - phase) / step) * step

def snap_index_regular(values, x):
    """Return nearest index on a regular 1D grid."""
    values = values.to_numpy()
    x = x.values
    idx = np.rint((values - x[0]) / (x[1] - x[0])).astype(np.int32)
    return np.clip(idx, 0, len(x) - 1)


def interval_to_iz_centers_increasing(z_centers, z_top, z_bot):
    """
    Inclusive iz range such that z_centers[iz] lies within [min(top,bot), max(top,bot)].
    Assumes z_centers strictly increasing.
    """
    zc = np.asarray(z_centers.values, dtype=float)
    lo = np.minimum(z_top.to_numpy(), z_bot.to_numpy())
    hi = np.maximum(z_top.to_numpy(), z_bot.to_numpy())

    iz0 = np.searchsorted(zc, lo, side="left")          # first center >= lo
    iz1 = np.searchsorted(zc, hi, side="right") - 1     # last center <= hi

    # mark empty intervals (no center falls inside)
    empty = iz0 > iz1

    iz0 = np.clip(iz0, 0, len(zc) - 1).astype(np.int32)
    iz1 = np.clip(iz1, 0, len(zc) - 1).astype(np.int32)

    return iz0, iz1, empty

def expand_iz(df, iz_top="iz_top", iz_bot="iz_bot", cols_to_repeat=("ix","iy","RHO")):
    lo = df[iz_top].to_numpy(np.int32)
    hi = df[iz_bot].to_numpy(np.int32)

    counts = (hi - lo + 1).astype(np.int32)
    total = int(counts.sum())

    out = {c: np.repeat(df[c].to_numpy(), counts) for c in cols_to_repeat}

    starts = np.repeat(lo, counts)
    offsets = np.arange(total, dtype=np.int32) - np.repeat(np.cumsum(counts) - counts, counts)
    out["iz"] = starts + offsets

    return pd.DataFrame(out, copy=False)