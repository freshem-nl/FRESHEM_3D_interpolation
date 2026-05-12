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

    # thresholds with lower and upper bounds included
    t_ext = np.concatenate([[lower], indicators, [upper]]).astype(np.float32)  # (m+2,)
    m_ext = t_ext.size  # L

    # output
    out_cols = [f"{prefix}{int(round(q*100)):02d}" for q in q_levels]
    out = pd.DataFrame(index=df.index, columns=out_cols, dtype=np.float32)

    def _process_block(df_block, out_block):
        # get CDF values as numpy array (n, m)
        F = df_block[indicator_col_names].to_numpy(dtype=dtype, copy=False)

        # optional monotonicity enforcement (can be needed if CDF values are noisy and not perfectly increasing)
        if ensure_monotonic:
            F = np.maximum.accumulate(F, axis=1)

        # clip to [0,1] for safety (in case of non-monotonicity or other issues)
        F = np.clip(F, 0.0, 1.0)

        # extend F with 0 at the left and 1 at the right (n, m+2)
        n = F.shape[0]
        F_ext = np.empty((n, F.shape[1] + 2), dtype=dtype)
        F_ext[:, 0] = 0.0
        F_ext[:, 1:-1] = F
        F_ext[:, -1] = 1.0

        # calculate quantiles
        for j, p in enumerate(q_levels):
            # indices of bounding indicators in F_ext
            idx = np.sum(F_ext < p, axis=1).astype(np.int32)

            # indices of bounding indicators in t_ext
            lo = np.clip(idx - 1, 0, m_ext - 1)
            hi = np.clip(idx,     0, m_ext - 1)

            # corresponding t-values and F-values
            t_lo = t_ext[lo]
            t_hi = t_ext[hi]
            F_lo = F_ext[np.arange(n), lo]
            F_hi = F_ext[np.arange(n), hi]

            # avoid division by zero (flat pieces)
            denom = (F_hi - F_lo)
            w = np.where(denom > 0, (p - F_lo) / denom, 0.0).astype(dtype)

            # linear interpolation between t_lo and t_hi
            qv = (t_lo + w * (t_hi - t_lo)).astype(dtype)

            # bounds neat: if hi==lo (at edge) then qv = t_lo remains
            out_block.iloc[:, j] = qv

        return out_block

    if chunk_size is None:
        out = _process_block(df, out)
    else:
        # chunk processing to limit peak memory usage
        idx = df.index
        for start in range(0, len(df), chunk_size):
            sl = slice(start, min(start + chunk_size, len(df)))
            out.iloc[sl, :] = _process_block(df.iloc[sl], out.iloc[sl].copy()).to_numpy()

    return out