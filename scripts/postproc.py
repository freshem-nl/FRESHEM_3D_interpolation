from datetime import datetime

import numpy as np
import xarray as xr

from scripts import _postproc_helper, read, write


def ds_ind_probs_to_quantiles(cfg):
    t0 = datetime.now()
    print("\nPOSTPROCESSING PREDICTIONS")
    print("calculate quantiles from indicator probabilities...", end=" ")

    # from config
    path_pred = cfg["path_prediction"]
    path_output = cfg["path_postproc"]
    indicators = cfg["indicators"]
    quantiles = cfg["quantiles"]
    bounds = cfg["indicator_bounds"]

    # read predictions
    ds_ind_probs = read.dataset(path_pred)

    indicator_col_names = [f"P({i})" for i in indicators]

    mask_all = xr.concat([ds_ind_probs[v].notnull() for v in indicator_col_names], dim="v").all("v")

    mask1d = mask_all.stack(cell=("Z", "Y", "X"))

    df_ind_probs = ds_ind_probs[indicator_col_names].stack(cell=("Z", "Y", "X")).where(mask1d, drop=True).to_dataframe()

    df_quant = _postproc_helper.ind_probs_to_quantiles(
        df_ind_probs,
        indicators=indicators,
        q_levels=quantiles,
        lower=bounds[0],
        upper=bounds[1],
        dtype=np.float32,
    )

    # init new dataset with same coords and attrs as original, but no data variables
    ds_quant = xr.Dataset(coords=ds_ind_probs.coords, attrs=ds_ind_probs.attrs)

    # convert quantiles back to xarray Dataset (dims constructed from index-levels)
    ds_newvars = xr.Dataset.from_dataframe(df_quant).reindex_like(ds_ind_probs)

    # add new quantile variables to dataset
    for v in ds_newvars:
        ds_quant[v] = ds_newvars[v]

    # save dataset
    write.dataset(ds_quant, path_output)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return ds_quant

def ensure_monotonicity(ds, cfg):

    cols = cfg["indicator_names"]

    ## ensure values between 0 and 1, and monotonicity (increasing with threshold)
    arr = np.stack([ds[v].values for v in cols], axis=0)

    # all values between 0 and 1
    arr = np.clip(arr, 0.0, 1.0)

    # ensure monotonicity (increasing with threshold)
    arr = np.maximum.accumulate(arr, axis=0)

    # write back to dataset
    for i, v in enumerate(cols):
        ds[v].data = arr[i]
    
    return ds