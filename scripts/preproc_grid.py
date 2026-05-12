import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from scipy.ndimage import distance_transform_edt

import scripts._utils as _utils
from scripts import _preprocessing_helper, _read_and_write


def snap_data_to_grid(cfg):

    t0 = datetime.now()

    # from config
    path_data = cfg["path_preproc_data"]
    cellsize_xy = cfg["cellsize_xy"]
    cellsize_z = cfg["cellsize_z"]
    buffer_xy = cfg["buffer_xy"]
    buffer_z = cfg["buffer_z"]
    epsg = cfg["epsg"]
    path_flightlines_out = cfg["path_preproc_data_flightlines"]

    print(f"Snapping data to XY grid with cellsize {cellsize_xy}...", end=" ")

    # read data
    df = _read_and_write.read_table(path_data.with_suffix(".parquet"))

    prob_cols = [col for col in df.columns if col.startswith("P(")]

    # snap XY to grid centers
    df["X"] = np.floor(df["X"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2
    df["Y"] = np.floor(df["Y"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2

    # init dataset with grid coordinates
    ds = _utils.init_ds(df[["X", "Y", "Z"]], cellsize_xy, cellsize_z, epsg, buffer_xy, buffer_z)

    # mean indicator values per voxel
    g = df.groupby(["X", "Y", "Z"], sort=False)[prob_cols].mean()

    # snap measurements to grid and add to dataset
    ds = _utils.add_df_to_ds(
        ds,
        g.reset_index(),
        coord_map={"x": "X", "y": "Y", "z": "Z"},
        value_cols=prob_cols,
    )

    # flightlines per xy-cell
    df_flightlines = df[["X", "Y", "LINE_NO"]].drop_duplicates()
    df_flightlines.to_parquet(path_flightlines_out.with_suffix(".parquet"))

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return ds


def mask_xy(ds, cfg):

    # from config
    cellsize_xy = cfg["cellsize_xy"]
    buffer_xy = cfg["buffer_xy"]
    indicators = cfg["indicators"]

    t0 = datetime.now()
    print(f"Masking XY grid with {buffer_xy}m buffer to data...", end=" ")

    # take one variable to determine where data is present
    var = f"P({indicators[0]})"

    # check which XY cells have any data in Z direction
    da = ds[var]
    has_data_xy = da.notnull().any("z").values

    # calculate distance to nearest cell with data, and mask cells beyond buffer distance
    dist_m = distance_transform_edt(~has_data_xy, sampling=(cellsize_xy, cellsize_xy))
    mask_xy = dist_m <= buffer_xy

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return mask_xy


def mask_z(ds, cfg):

    # from config
    buffer_z = cfg["buffer_z"]
    indicators = cfg["indicators"]

    t0 = datetime.now()
    print(f"Masking Z grid with {buffer_z}m buffer to data...", end=" ")

    # take one variable to determine where data is present
    var = f"P({indicators[0]})"
    da = ds[var]

    # "data present" = non-NaN
    has = da.notnull()

    # for each (y,x) cell, find top and bottom z with data
    top_z = ds["z"].where(has).max("z").rename("top_z")  # (y,x)
    bot_z = ds["z"].where(has).min("z").rename("bot_z")  # (y,x)

    # stack -> 1D list of (y,x) cells, drop NaNs
    top_1d = top_z.stack(cell=("y", "x")).dropna("cell")
    bot_1d = bot_z.stack(cell=("y", "x")).dropna("cell")

    # coordinates of cell centers with data, and their top/bottom z values
    xp_top = top_1d["x"].values
    yp_top = top_1d["y"].values
    vp_top = top_1d.values.astype(np.float64)

    xp_bot = bot_1d["x"].values
    yp_bot = bot_1d["y"].values
    vp_bot = bot_1d.values.astype(np.float64)

    xg = ds["x"].values
    yg = ds["y"].values

    # interpolate top and bottom surfaces to grid using IDW
    top_grid = _preprocessing_helper.idw_to_grid(xp_top, yp_top, vp_top, xg, yg, k=12, p=2.0)
    bot_grid = _preprocessing_helper.idw_to_grid(xp_bot, yp_bot, vp_bot, xg, yg, k=12, p=2.0)

    # create DataArrays for top and bottom surfaces
    top_surf = xr.DataArray(top_grid, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="top_surf")
    bot_surf = xr.DataArray(bot_grid, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="bot_surf")

    # broadcasting top and bottom surfaces broadcasten to 3D for comparison with Z-coordinates
    Z3, TOP3 = xr.broadcast(ds["z"], top_surf)  # -> (z,y,x)
    _, BOT3 = xr.broadcast(ds["z"], bot_surf)

    # 3D z-coordinates
    Z3 = ds["z"].broadcast_like(ds[var])

    # mask where Z3 in between top_surf and bot_surf (with buffer)
    mask_z = ((Z3 <= top_surf) & (Z3 >= bot_surf)).rename("mask_z")

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return mask_z


def combine_masks(ds, mask_xy, mask_z, cfg):

    t0 = datetime.now()
    print("Combining XY and Z masks...", end=" ")

    # from config
    path_out = cfg["path_preproc_data_gridded"]

    mask = (mask_xy & mask_z).rename("mask")
    ds["mask"] = mask.astype(bool)

    _read_and_write.write_dataset(ds, path_out)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return ds


def plotting(ds, cfg):

    t0 = datetime.now()
    print("Plotting...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]
    plotting_depths = cfg["plotting_depths"]

    # sample data for histogram plotting
    n = 10000

    os.makedirs(dir_plot, exist_ok=True)
    for var in ds.data_vars:

        # length of dataset values (non-nan)
        n_ds = np.isfinite(ds[var].values).sum()
        data_plot = np.random.choice(
            (a := ds[var].values.ravel())[np.isfinite(a)], size=min(n, np.isfinite(a).sum()), replace=False
        )
        plt.figure()
        sns.histplot(data_plot, bins=20, kde=False)
        plt.title(f"gridded {var}, n={n_ds:,}")

        path = dir_plot / f"data gridded - {var}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        target_depths = np.array(plotting_depths)
        depths = ds["z"].sel(z=target_depths, method="nearest").values

        for depth in depths:
            ds[var].sel(z=depth).plot()
            plt.title(f"{var} at z={depth}m")

            path = dir_plot / f"data gridded - {var} at z={depth}m.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
