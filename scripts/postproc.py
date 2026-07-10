from datetime import datetime

import numpy as np
import xarray as xr

from scripts import _postproc_helper, read, write


def ds_ind_probs_to_quantiles(pred, cfg):
    t0 = datetime.now()
    print("\nPOSTPROCESSING PREDICTIONS")
    print("calculate quantiles from indicator probabilities...", end=" ")

    # from config
    indicators = cfg["indicators"]
    indicator_names = cfg["indicator_names"]
    quantiles = cfg["quantiles"]
    quantile_names = cfg["quantile_names"]
    bounds = cfg["indicator_bounds"]

    # create mask indicator variables: True if all indicator probabilities are not null, False otherwise
    mask_all = xr.concat([pred[v].notnull() for v in indicator_names], dim="v").all("v")

    # stack mask to 1D array for indexing
    mask1d = mask_all.stack(cell=("layer", "y", "x"))

    # create dataframe with indicator probabilities for all cells where mask is True
    df_ind_probs = pred[indicator_names].stack(cell=("layer", "y", "x")).where(mask1d, drop=True).to_dataframe()

    # convert indicator probabilities to quantiles
    df_quant = _postproc_helper.ind_probs_to_quantiles(
        df_ind_probs,
        indicators=indicators,
        indicator_bounds=bounds,
        q_levels=quantiles,
        q_names=quantile_names,
    )

    # init new dataset with same coords and attrs as original, but no data variables
    pred_quant = xr.Dataset(coords=pred.coords, attrs=pred.attrs)


    # Convert dataframe columns to variables and assign them onto the template dataset
    for var in df_quant.columns:
        # Create a DataArray from the dataframe column, with the same index as the dataset
        da = df_quant[var].to_xarray().reindex_like(pred_quant)
        # Assign data only, so template coordinates and their attrs remain untouched
        pred_quant[var] = (da.dims, da.data)

    for var in ("top", "bottom", cfg["doi_name"]):
        pred_quant[var] = pred[var]

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return pred_quant

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