from datetime import datetime

import numpy as np
import xarray as xr

from scripts import _postproc_helper, _read_and_write


def quantiles(cfg):
    t0 = datetime.now()
    print('\nPOSTPROCESSING PREDICTIONS')
    print("calculate quantiles from indicator probabilities...", end=" ")

    # from config
    path_pred = cfg["path_prediction"]
    path_output = cfg["path_postproc"]
    indicators = cfg["indicators"]
    quantiles = cfg["quantiles"]
    bounds = cfg["indicator_bounds"]

    # read predictions
    ds_ind_probs = _read_and_write.read_dataset(path_pred)

    indicator_col_names = [f"P({i})" for i in indicators]

    mask_all = xr.concat([ds_ind_probs[v].notnull() for v in indicator_col_names], dim="v").all("v")

    mask1d = mask_all.stack(cell=("z", "y", "x"))

    df_ind_probs = ds_ind_probs[indicator_col_names].stack(cell=("z", "y", "x")).where(mask1d, drop=True).to_dataframe()

    df_quant = _postproc_helper.indicator_probs_to_quantiles(
        df_ind_probs,
        indicators=indicators,
        indicator_col_names=indicator_col_names,
        q_levels=quantiles,
        lower=bounds[0],
        upper=bounds[1],
        ensure_monotonic=True,
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
    _read_and_write.write_dataset(ds_quant, path_output)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return ds_quant
