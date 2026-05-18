import numpy as np
import pandas as pd



def rps_from_cdf(pred, obs, normalize=True):
    """
    Soft RPS for ordinal classes, directly from CDF values at class boundaries.
    RPS = sum_k (F_k - G_k)^2, where F_k and G_k are cumulative probabilities. 
    [1](https://www.lokad.com/continuous-ranked-probability-score)
    [2](https://link.springer.com/content/pdf/10.1007/s11336-014-9439-4.pdf)
    """
    pred = pred.to_numpy()
    obs = obs.to_numpy()

    # Sum over the provided boundaries (K-1 terms)
    rps = np.sum((pred - obs) ** 2, axis=1)

    if normalize:
        m = pred.shape[1]  # = K-1 boundaries
        rps =  rps / m if m > 0 else rps

    return rps

def rps_summary(df, rps_col, class_col, path):
    # overall stats
    overall_row = pd.Series(
        {
            "n": df[rps_col].notna().sum(),
            "mean": df[rps_col].mean(),
            "median": df[rps_col].median(),
        },
        name="overall",
    )

    # group by median class and compute summary stats
    rps_summary = (
        df
        .groupby(class_col, sort=False, observed=False)[rps_col]
        .agg(n="size", mean="mean", median="median")
        .sort_index()
    )

    rps_summary = pd.concat([overall_row.to_frame().T, rps_summary], axis=0)

    #save results
    rps_summary.to_csv(path, index=True)