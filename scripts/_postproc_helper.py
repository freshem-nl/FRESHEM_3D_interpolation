import numpy as np
import pandas as pd


def ind_probs_to_quantiles(
    df: pd.DataFrame,
    indicators,
    indicator_bounds=(0.0, 150.0),
    q_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    q_names=None,
    dtype=np.float32
):

    indicators = np.asarray(indicators, dtype=np.float32)
    q_levels = np.asarray(q_levels, dtype=np.float32)

    # Extend thresholds with hard bounds: t_ext = [lower, t1..tm, upper]
    lower, upper = map(float, indicator_bounds)
    t_ext = np.concatenate([[lower], indicators, [upper]]).astype(np.float32)
    m_ext = t_ext.size

    # Read CDF values: shape (n, m)
    F = df.to_numpy(dtype=np.float64, copy=False)

    # monotonicity enforcement (can be needed if CDF values are noisy and not perfectly increasing)
    F = np.maximum.accumulate(F, axis=1)

    # clip to [0,1] for safety
    F = np.clip(F, 0.0, 1.0)

    # extend F with 0 at the left and 1 at the right (n, m+2)
    n = F.shape[0]
    F_ext = np.empty((n, F.shape[1] + 2), dtype=dtype)
    F_ext[:, 0] = 0.0
    F_ext[:, 1:-1] = F
    F_ext[:, -1] = 1.0

    # Allocate output: (n, k)
    Q = np.empty((n, q_levels.size), dtype=np.float64)

    # Invert per quantile level p
    for j, p in enumerate(q_levels):

        # Find the first index hi where F_ext >= p (done via counting values < p)
        hi = np.sum(F_ext < p, axis=1).astype(np.int32)  # gives index of upper bracket
        hi = np.clip(hi, 0, m_ext - 1)  # safety clip
        lo = np.clip(hi - 1, 0, m_ext - 1)  # lower bracket is one index below hi, with safety clip at 0

        # Gather brackets
        t_lo = t_ext[lo]  # (n,) lower threshold
        t_hi = t_ext[hi]  #  (n,) upper threshold
        F_lo = F_ext[np.arange(n), lo]  # (n,) CDF value at lower threshold
        F_hi = F_ext[np.arange(n), hi]  #  (n,) CDF value at upper threshold

        # Linear interpolation weight; handle flat segments safely
        denom = F_hi - F_lo
        # w = np.where(denom > 0, (p - F_lo) / denom, 0.0)       
        w = np.zeros_like(denom, dtype=np.float64)
        mask = denom > 0
        w[mask] = (p - F_lo[mask]) / denom[mask]


        # linear interpolation between t_lo and t_hi
        Q[:, j] = t_lo + w * (t_hi - t_lo)

    if q_names is None:
        q_names = [f"q{int(p*100):02d}" for p in q_levels]

    df_quant = pd.DataFrame(Q.astype(dtype, copy=False), index=df.index, columns=q_names)

    return df_quant


def class_from_quantile(q, indicators, bounds):

    q_arr = np.asarray(q, float)
    ind = np.asarray(indicators, float)
    lo, hi = map(float, bounds)
    
    # Class index for intervals: [lo,t1], (t1,t2], ..., (t_last, hi]
    # side="left" ensures exact boundaries go to the lower (right-closed) class.
    codes = np.searchsorted(ind, q_arr, side="left")
    
    # Labels (ordered)
    labels = [f"[{lo:g}, {ind[0]:g}]"]
    for i in range(1, len(ind)):
        labels.append(f"({ind[i-1]:g}, {ind[i]:g}]")
    labels.append(f"({ind[-1]:g}, {hi:g}]")

    cat = pd.Categorical.from_codes(codes, categories=labels, ordered=True)   

    return pd.Series(cat, index=q.index, name=q.name)