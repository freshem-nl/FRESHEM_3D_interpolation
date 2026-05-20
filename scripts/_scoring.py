import numpy as np
import pandas as pd
import os



def rps_from_cdf(pred, obs, normalize=True):
    """
    Soft RPS for ordinal classes, directly from CDF values at class boundaries.
    RPS = sum_k (F_k - G_k)^2, where F_k and G_k are cumulative probabilities. 
    [1](https://www.lokad.com/continuous-ranked-probability-score)
    [2](https://link.springer.com/content/pdf/10.1007/s11336-014-9439-4.pdf)
    """
    index = obs.index
    pred = pred.to_numpy()
    obs = obs.to_numpy()

    # Sum over the provided boundaries (K-1 terms)
    rps = np.sum((pred - obs) ** 2, axis=1)

    if normalize:
        m = pred.shape[1]  # = K-1 boundaries
        rps =  rps / m if m > 0 else rps

    rps = pd.Series(rps, index=index, name="rps")

    return rps

def rps_summary(rps, classes, path):
    # overall stats
    overall_row = pd.Series(
        {
            "n": rps.notna().sum(),
            "mean": rps.mean(),
            "median": rps.median(),
        },
        name="overall",
    )

    # group by median class and compute summary stats
    rps_summary = (
        pd.DataFrame({"rps": rps, "class": classes})
        .groupby("class", sort=False, observed=False)["rps"]
        .agg(n="size", mean="mean", median="median")
        .sort_index()
    )

    rps_summary = pd.concat([overall_row.to_frame().T, rps_summary], axis=0)

    #save results
    os.makedirs(path.parent, exist_ok=True)
    rps_summary.to_csv(path, index=True)