import os
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts import _read_and_write, _postproc_helper


def indicators_to_percentiles(cfg):
    t0 = datetime.now()
    print("Converting indicators to percentiles...", end=" ")

    path_pred = cfg["path_prediction"]
    indicators = cfg["indicators"]
    quantiles = cfg["quantiles"]
    bounds = cfg['indicator_bounds']

    # read predictions
    ds = _read_and_write.read_dataset(path_pred)

    indicator_col_names = [f"P({i})" for i in indicators]

    mask_all = xr.concat([ds[v].notnull() for v in indicator_col_names], dim="v").all("v")

    mask1d = mask_all.stack(cell=("z", "y", "x"))

    df = ds[indicator_col_names].stack(cell=("z", "y", "x")).where(mask1d, drop=True).to_dataframe()

    qdf = _postproc_helper.indicator_probs_to_quantiles(
        df,
        indicators=indicators,
        indicator_col_names=indicator_col_names,
        q_levels=quantiles,
        lower=bounds[0],
        upper=bounds[1],
        ensure_monotonic=True,
        dtype=np.float32,
    )

    # terug naar xarray Dataset (dims worden gereconstrueerd uit index-levels)
    ds_newvars = xr.Dataset.from_dataframe(qdf)

    # alignen met originele ds-grid (zorgt dat alle (z,y,x) bestaan, met NaN buiten mask)
    for var in ds_newvars.data_vars:
        ds[var] = ds_newvars[var].reindex_like(ds[indicator_col_names[0]])

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
    
    return ds


def plotting(ds, cfg):
    t0 = datetime.now()
    print("post-processing predictions...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]
    plotting_depths = cfg["plotting_depths"]

    os.makedirs(dir_plot, exist_ok=True)
    for var in ds.data_vars:

        # select target depths (exact match or nearest)
        target_depths = np.array(plotting_depths)
        depths = ds["z"].sel(z=target_depths, method="nearest").values

        for depth in depths:

            # select 2D slice at target depth
            da = ds[var].sel(z=depth)

            # log colorscale for quantiles, linear for indicators
            if var.startswith("Q"):
                vals = da.values
                vals = vals[np.isfinite(vals) & (vals > 0)]
                vmin, vmax = np.quantile(vals, [0.02, 0.98])
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = None

            # plot map
            da.plot(norm=norm)
            plt.title(f"{var} at z={depth}m")

            # save figure
            path = dir_plot / f"data prediction - {var} at z={depth}m.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
