import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from scipy.spatial import cKDTree


def initiate_grid(data, cfg):

    # from config
    cellsize_xy = cfg["cellsize_xy"]
    buffer_xy = cfg["buffer_xy"]
    epsg = cfg["epsg"]
    # path_flightlines_out = cfg["path_preproc_data_flightlines"]

    # snap XY to grid centers
    x_min = np.floor(data["x"].min() / cellsize_xy) * cellsize_xy + cellsize_xy / 2
    x_max = np.floor(data["x"].max() / cellsize_xy) * cellsize_xy + cellsize_xy / 2
    y_min = np.floor(data["y"].min() / cellsize_xy) * cellsize_xy + cellsize_xy / 2
    y_max = np.floor(data["y"].max() / cellsize_xy) * cellsize_xy + cellsize_xy / 2

    x = np.arange(x_min - buffer_xy, x_max + buffer_xy + cellsize_xy, cellsize_xy)

    y = np.arange(y_min - buffer_xy, y_max + buffer_xy + cellsize_xy, cellsize_xy)

    layers = np.sort(data["layer"].unique())

    ds = xr.Dataset(coords={"layer": layers, "x": x, "y": y})

    ds.attrs.update({"cellsize_x": cellsize_xy, "cellsize_y": cellsize_xy, "grid_mapping": "spatial_ref"})

    ds["x"].attrs.update({"standard_name": "X-coordinate", "units": "m"})
    ds["y"].attrs.update({"standard_name": "Y-coordinate", "units": "m"})

    if epsg is not None:
        ds.attrs["crs"] = f"EPSG:{int(epsg)}"
        ds = ds.rio.write_crs(f"EPSG:{epsg}")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return ds


# def snap_data_to_grid(df, cfg):

#     t0 = datetime.now()
#     print("\nPREPROCESSING GRIDDED DATA")

#     # from config
#     cellsize_xy = cfg["cellsize_xy"]
#     cellsize_z = cfg["cellsize_z"]
#     buffer_xy = cfg["buffer_xy"]
#     buffer_z = cfg["buffer_z"]
#     epsg = cfg["epsg"]
#     path_flightlines_out = cfg["path_preproc_data_flightlines"]

#     print(f"Snapping data to XY grid with cellsize {cellsize_xy}...", end=" ")

#     # ONLY probability columns can be averaged when snapping to grid
#     prob_cols = [col for col in df.columns if col.startswith("P(")]

#     # snap XY to grid centers
#     df["x"] = np.floor(df["x"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2
#     df["y"] = np.floor(df["y"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2

#     # init dataset with grid coordinates
#     ds = _utils.init_ds(df[["z", "y", "x"]], cellsize_xy, cellsize_z, epsg, buffer_xy, buffer_z)

#     # mean indicator values per voxel
#     g = df.groupby(["z", "y", "x"], sort=False)[prob_cols].mean()

#     # snap measurements to grid and add to dataset
#     ds = _utils.add_df_to_ds(
#         ds,
#         g.reset_index(),
#         value_cols=prob_cols,
#     )

#     ##FLIGHTLINES TO DATASET, FOR USE IN ANYSOTROPY ANALYSIS
#     # FIRST flightline per voxel
#     g = df.groupby(["x", "y", "z"], sort=False)[["line_no"]].first()

#     # snap flightline no to grid and add to dataset
#     ds = _utils.add_df_to_ds(
#         ds,
#         g.reset_index(),
#         value_cols=["line_no"],
#     )

#     ##FLIGHTLINES TO DATAFRAME, FOR USE IN CROSS-VALIDATION
#     # flightlines per xy-cell
#     df_flightlines = df[["x", "y", "line_no"]].drop_duplicates()
#     df_flightlines = _utils.df_to_gdf(df_flightlines, crs=df.crs)

#     df_flightlines.to_parquet(path_flightlines_out.with_suffix(".parquet"))

#     print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

#     return ds


def mask_overall(data, pred, cfg):
    buffer_xy = cfg["buffer_xy"]

    t0 = datetime.now()
    print("\nPREPROCESSING PREDICTION GRID")
    print(f"Masking XY grid with {buffer_xy}m buffer to data...", end=" ")

    # Extract coordinates
    x = pred.coords["x"].values
    y = pred.coords["y"].values

    # Create grid points
    grid_points = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)

    # Unique data points
    data_points = data[["x", "y"]].drop_duplicates().to_numpy()

    # KDTree query
    tree = cKDTree(data_points)
    dist, _ = tree.query(grid_points, distance_upper_bound=buffer_xy)

    # Create mask
    mask = np.isfinite(dist).reshape(len(y), len(x))

    pred["mask_overall"] = (("y", "x"), mask)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")
    return pred

def mask_per_layer(pred):

    t0 = datetime.now()
    print(f"Mask per layer...", end=" ")


    mask_overall = pred["mask_overall"].broadcast_like(pred["top"])

    pred["mask"] = (
        mask_overall
        & pred["top"].notnull()
        & pred["bottom"].notnull()
    )

    # remove overall mask
    pred = pred.drop_vars("mask_overall")

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")
    return pred

# def mask_xy(data, pred, cfg):

#     # from config
#     buffer_xy = cfg["buffer_xy"]

#     t0 = datetime.now()
#     print("\nPREPROCESSING PREDICTION GRID")
#     print(f"Masking XY grid with {buffer_xy}m buffer to data...", end=" ")

#     # Get grid-cell centre coordinates from dataset
#     xx, yy = np.meshgrid(pred.coords["x"].values, pred.coords["y"].values)

#     grid_points = np.column_stack(
#         [
#             xx.ravel(),
#             yy.ravel(),
#         ]
#     )


#     mask = np.full(
#         (pred.sizes["y"], pred.sizes["x"]),
#         False,
#         dtype=bool,
#     )

#     data_points = data[["x", "y"]].drop_duplicates().to_numpy()

#     # Find grid cells within buffer distance of any point
#     tree = cKDTree(data_points)
#     dist, _ = tree.query(grid_points, distance_upper_bound=buffer_xy)

#     # Reshape back to dataset grid
#     mask[:, :] = np.isfinite(dist).reshape(pred.sizes["y"], pred.sizes["x"])

#     pred["mask"] = (("y", "x"), mask)

#     print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

#     return pred


#     a=1


#     for i, layer in enumerate(pred.coords["layer"].values):
#         df_layer = data[data['layer'] == layer]


#         # Get data point coordinates from dataframe
#         data_points = df_layer[["x", "y"]].to_numpy()

#         # Find grid cells within buffer distance of any point
#         tree = cKDTree(data_points)
#         dist, _ = tree.query(grid_points, distance_upper_bound=buffer_xy)

#         # Reshape back to dataset grid
#         mask[i, :, :] = np.isfinite(dist).reshape(pred.sizes["y"], pred.sizes["x"])

#     pred["mask"] = (("layer", "y", "x"), mask)


#     print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

#     return pred


# def mask_z(ds, cfg):

#     # from config
#     indicator_names = cfg["indicator_names"]

#     t0 = datetime.now()
#     print("Masking Z grid ...", end=" ")

#     # take one variable to determine where data is present
#     var = indicator_names[0]
#     da = ds[var]

#     # "data present" = non-NaN
#     has = da.notnull()

#     # for each (y,x) cell, find top and bottom z with data
#     top_z = ds["z"].where(has).max("z").rename("top_z")  # (Y,X)
#     bot_z = ds["z"].where(has).min("z").rename("bot_z")  # (Y,X)

#     # stack -> 1D list of (Y,X) cells, drop NaNs
#     top_1d = top_z.stack(cell=("y", "x")).dropna("cell")
#     bot_1d = bot_z.stack(cell=("y", "x")).dropna("cell")

#     # coordinates of cell centers with data, and their top/bottom z values
#     xp_top = top_1d["x"].values
#     yp_top = top_1d["y"].values
#     vp_top = top_1d.values.astype(np.float64)

#     xp_bot = bot_1d["x"].values
#     yp_bot = bot_1d["y"].values
#     vp_bot = bot_1d.values.astype(np.float64)

#     xg = ds["x"].values
#     yg = ds["y"].values

#     # interpolate top and bottom surfaces to grid using IDW
#     top_grid = _preproc_helper.idw_to_grid(xp_top, yp_top, vp_top, xg, yg, k=12, p=2.0)
#     bot_grid = _preproc_helper.idw_to_grid(xp_bot, yp_bot, vp_bot, xg, yg, k=12, p=2.0)

#     # create DataArrays for top and bottom surfaces
#     top_surf = xr.DataArray(top_grid, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="top_surf")
#     bot_surf = xr.DataArray(bot_grid, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="bot_surf")

#     # broadcasting top and bottom surfaces broadcasten to 3D for comparison with Z-coordinates
#     Z3, TOP3 = xr.broadcast(ds["z"], top_surf)  # -> (Z,Y,X)
#     _, BOT3 = xr.broadcast(ds["z"], bot_surf)

#     # 3D z-coordinates
#     Z3 = ds["z"].broadcast_like(ds[var])

#     # mask where Z3 in between top_surf and bot_surf (with buffer)
#     mask_z = ((Z3 <= top_surf) & (Z3 >= bot_surf)).rename("mask_z")

#     print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

#     return mask_z


# def combine_masks(data_g, mask_xy, mask_z):

#     t0 = datetime.now()
#     print("Combining XY and Z masks...", end=" ")

#     mask = (mask_xy & mask_z).rename("mask")

#     pred_g = xr.Dataset(data_vars={"mask": mask}, coords=data_g.coords, attrs=data_g.attrs)

#     print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

#     return pred_g


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
    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")
