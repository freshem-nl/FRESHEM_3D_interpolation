import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd

from statistics import NormalDist

from ._preprocessing_helper import snap_to_center_grid, snap_to_center_grid_up, snap_index_regular, interval_to_iz_centers_increasing, expand_iz
import _utils

def initiate_dataset(df, cfg):
    print("Initiating gridded dataset with", end=" ")

    # from config
    cellsize_xy = cfg["cell_size_xy"]
    cellsize_z = cfg["cell_size_z"]

    # get bounds
    x_min = df['X'].min()
    x_max = df['X'].max()
    y_min = df['Y'].min()
    y_max = df['Y'].max()
    z_min = (df['ELEVATION'] - df["DOI_STANDARD"]).min()
    z_max = (df['ELEVATION'] - df["DEP_TOP_1"]).max()

    # snap bounds to grid centers
    x_min_cell_center = snap_to_center_grid(x_min, cellsize_xy)
    x_max_cell_center = snap_to_center_grid_up(x_max, cellsize_xy)
    y_min_cell_center = snap_to_center_grid(y_min, cellsize_xy)
    y_max_cell_center = snap_to_center_grid_up(y_max, cellsize_xy)
    z_min_cell_center = snap_to_center_grid(z_min, cellsize_z)
    z_max_cell_center = snap_to_center_grid_up(z_max, cellsize_z)


    # build centers
    x = np.arange(x_min_cell_center, x_max_cell_center + cellsize_xy, cellsize_xy)
    y = np.arange(y_min_cell_center, y_max_cell_center + cellsize_xy, cellsize_xy)
    z = np.arange(z_min_cell_center, z_max_cell_center + cellsize_z, cellsize_z)

    # build dataset
    ds = xr.Dataset(coords={"x": x, "y": y, "z": z})

    print(f"shape: {ds.sizes}, total voxels: {ds.sizes['x'] * ds.sizes['y'] * ds.sizes['z']}")

    return ds
    
def snap_measurements_to_grid(df, ds):

    t0 = datetime.now()
    print("Snapping measurements to grid...")

    n_measuements = 0
    n_dropped = 0
    meas_gridded_per_layer = []
    layer_numbers = [int(x.split('_')[1]) for x in df.columns if x.startswith("RHO") and not "STD" in x]
    for i in layer_numbers:
    # for i in [15]:
        print(f"\tLayer {i}:", end=" ")
        df_sel = df[['LINE_NO', 'X', 'Y', "ELEVATION", f"RHO_{i}",f"RHO_STD{i}", f"DEP_TOP_{i}", f"DEP_BOT_{i}", "DOI_STANDARD"]].copy()

        # rename to stable names
        df_sel = df_sel.rename(columns={f'RHO_{i}': 'rho', f'RHO_STD{i}': 'rho_std'})

        df_sel['Z_TOP'] = df_sel['ELEVATION'] - df_sel[f"DEP_TOP_{i}"]
        df_sel['Z_BOT'] = df_sel['ELEVATION'] - df_sel[f"DEP_BOT_{i}"]
        df_sel['DOI_Z'] = df_sel['ELEVATION'] - df_sel["DOI_STANDARD"]

        # ---- DOI handling: clip bottoms to DOI, then drop intervals fully below DO
        n0 = len(df_sel)
        df_sel['Z_BOT'] = np.maximum(df_sel['Z_BOT'], df_sel['DOI_Z'])
        df_sel = df_sel[df_sel['Z_TOP'] > df_sel['DOI_Z']]
        n_dropped += (n0 - len(df_sel))
        print(f"dropped {n0 - len(df_sel)} below DOI_STANDARD", end=", ")

        # XY -> indices on ds grid
        df_sel["ix"] = snap_index_regular(df_sel["X"], ds['x'])
        df_sel["iy"] = snap_index_regular(df_sel["Y"], ds['y'])

        # z coverage by centers-in-interval
        df_sel["iz_top"], df_sel["iz_bot"], df_sel["empty"] = interval_to_iz_centers_increasing(ds['z'], df_sel['Z_TOP'], df_sel['Z_BOT'])

        n1 = len(df_sel)
        df_sel = df_sel.loc[~df_sel["empty"]].copy()
        n_dropped += (n1 - len(df_sel))
        print(f"dropped {n1 - len(df_sel)} intervals not intersecting any z-cell-center")

        if len(df_sel) == 0:
            continue

        n_measuements += len(df_sel)

        # Expand interval rows -> voxel-hit rows
        vox_long = expand_iz(df_sel, cols_to_repeat=("LINE_NO","ix","iy","rho", "rho_std"))
        meas_gridded_per_layer.append(vox_long)

    # Combine all layers' voxel hits
    gridded_measurements = pd.concat(meas_gridded_per_layer, ignore_index=True)

    print(f"...Snapped {n_measuements} measurements to grid, dropped {n_dropped} measurements.\n...Results in {len(gridded_measurements.groupby(['ix','iy','iz']))} voxels containing {gridded_measurements.shape[0]} measurements.")

    return gridded_measurements

def quantiles_per_voxel(measurements_gridded, ds, cfg):
    t0 = datetime.now()
    print(f"Computing percentiles per voxel...", end=" ")

    # group per voxel (ix,iy,iz)
    gb = measurements_gridded.groupby(["ix","iy","iz"])

    # desired quantiles
    qs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    # get z-scores for the quantiles (z-scores are the quantiles of the standard normal distribution)
    z = np.array([NormalDist().inv_cdf(q) for q in qs])  # z-scores

    # column names for the quantiles to be stored in the output dataframe
    qcols = [f"rho_p{int(q*100):02d}" for q in qs]

    # mean of means
    mu = gb["rho"].mean()

    # variance of means
    var_between = gb["rho"].var(ddof=0).fillna(0.0)

    # mean of variances
    var_within  = gb["rho_std"].apply(lambda s: np.mean(np.square(s.to_numpy(float))))

    # total variance = variance of means + mean of variances
    sigma = np.sqrt(var_between + var_within)

    # compute quantiles using the mean and total stddev, assuming normal distribution of measurements within each voxel
    qdf_norm = pd.concat([(mu + sigma * zi).rename(col) for zi, col in zip(z, qcols)], axis=1)

    # also add count of measurements per voxel
    qdf_norm["n_measurements"] = gb.size()

    # add quantiles and count to dataset
    ds = _utils.add_to_dataset(qdf_norm, ds)

    # save dataset
    path = cfg['dir_output'] / "data_quantiles.nc"
    _utils.save_dataset(ds, path)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")


def flightlines_per_voxel(measurements_gridded, cfg):
    t0 = datetime.now()
    print(f"Computing flightlines per voxel...", end=" ")
    # counts of measurements per voxel per flightline
    line_counts = (
        measurements_gridded
        .groupby(["ix", "iy", "iz", "LINE_NO"], sort=False)
        .size()
        .rename("n_from_flightline")
        .reset_index()
    )

    # total contributions per voxel (across all flightlines)
    voxel_n = (
        line_counts
        .groupby(["ix", "iy", "iz"], sort=False)["n_from_flightline"]
        .sum()
        .rename("n_total")
        .reset_index()
    )

    # attach fraction per flightline (optional but handy)
    line_counts = line_counts.merge(voxel_n, on=["ix", "iy", "iz"], how="left")
    line_counts["flightline_fraction"] = line_counts["n_from_flightline"] / line_counts["n_total"]

    path = cfg["dir_output"] / "voxel_flightlines_contribution.parquet"
    line_counts.to_parquet(path, engine="fastparquet")

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
