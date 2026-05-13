import numpy as np
import pandas as pd

def indicator_probs_to_quantiles(
    df: pd.DataFrame,
    indicators,
    indicator_col_names,
    q_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    lower=0.0,
    upper=150.0,
    ensure_monotonic=True,
    dtype=np.float32,
    chunk_size=None,  # bv. 1_000_000 als je memory wil sparen
    prefix="Q",
):

    indicators = np.asarray(indicators, dtype=np.float32)
    q_levels = np.asarray(q_levels, dtype=np.float32)

    # Extend thresholds with hard bounds: t_ext = [lower, t1..tm, upper]
    t_ext = np.concatenate([[lower], indicators, [upper]]).astype(np.float32)
    m_ext = t_ext.size 

    # output column names: e.g. Q05, Q10, Q25, Q50, Q75, Q90, Q95
    out_cols = [f"{prefix}{int(round(q*100)):02d}" for q in q_levels]
    
    # Read CDF values: shape (n, m)
    F = df.loc[:, indicator_col_names].to_numpy(dtype=np.float64, copy=False)

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
        hi = np.sum(F_ext < p, axis=1).astype(np.int32) # gives index of upper bracket
        hi = np.clip(hi, 0, m_ext - 1) # safety clip
        lo = np.clip(hi - 1, 0, m_ext - 1) # lower bracket is one index below hi, with safety clip at 0

        # Gather brackets
        t_lo = t_ext[lo] # (n,) lower threshold
        t_hi = t_ext[hi] #  (n,) upper threshold
        F_lo = F_ext[np.arange(n), lo] # (n,) CDF value at lower threshold
        F_hi = F_ext[np.arange(n), hi] #  (n,) CDF value at upper threshold

        # Linear interpolation weight; handle flat segments safely
        denom = (F_hi - F_lo)
        w = np.where(denom > 0, (p - F_lo) / denom, 0.0)

        # linear interpolation between t_lo and t_hi
        Q[:, j] = t_lo + w * (t_hi - t_lo)

    df_quant = pd.DataFrame(Q.astype(dtype, copy=False), index=df.index, columns=out_cols)

    return df_quant