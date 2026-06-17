from datetime import datetime

import isatis as isa
import isatis.constants as cst
import numpy as np
import pandas as pd
import xarray as xr

isa.setLicenseString("52100@lic-isatis.tno.nl")


def kriging(data, pred, cfg, verbose=True):
    t0 = datetime.now()
    if verbose:
        print("\nSPATIAL INTERPOLATION")
        print("indicator kriging...", end=" ")

    # From config
    indicator_names = cfg["indicator_names"]
    range_xy = cfg["variogram_model_range_xy"]
    range_z = cfg["variogram_model_range_z"]
    neigh_dist_xy = cfg["neighbourhood_dist_xy"]
    neigh_dist_z = cfg["neighbourhood_dist_z"]
    neigh_n_sectors = cfg["neighbourhood_n_sectors"]
    neigh_max_neigh_per_sector = cfg["neighbourhood_max_neigh_per_sector"]

    # Create Isatis input database
    input_db = isa.DbPandas(data.to_dataframe().dropna().reset_index())

    # Use one fixed dimension order for everything sent to Isatis
    pred_xyz = pred.transpose("x", "y", "z")

    # Create Isatis output grid
    grid = isa.GridGeom(
        origin=[pred_xyz["x"].values.min(), pred_xyz["y"].values.min(), pred_xyz["z"].values.min()],
        cell_size=[pred_xyz.attrs["cellsize_x"], pred_xyz.attrs["cellsize_y"], pred_xyz.attrs["cellsize_z"]],
        nxyz=[pred_xyz.sizes["x"], pred_xyz.sizes["y"], pred_xyz.sizes["z"]],
        ndim=3,
    )

    # Build output dataframe in the exact same grid order
    mask = pred_xyz["mask"].values.ravel()
    cell_id = np.arange(mask.size, dtype=np.int64)

    df_out = pd.DataFrame({"mask": mask, "cell_id": cell_id})

    output_db = isa.DbPandas(df_out, grid=grid)

    # Make multivariate variogram model
    n_var = len(indicator_names)

    multi_vario = isa.VModel(nvar=n_var)
    sill_matrix = np.zeros((n_var, n_var))
    nugg_matrix = np.zeros((n_var, n_var))

    for i in range(n_var):
        sill_matrix[i, i] = 0.33
        nugg_matrix[i, i] = 0.001

    sph_struct = isa.VStruc(
        stype=cst.MOD.SPH,
        nvar=n_var,
        ranges=[range_xy, range_xy, range_z],
        sill=sill_matrix,
    )
    multi_vario.add_struct(sph_struct)
    multi_vario.set_nugget(nugg_matrix)

    # Define neighbourhood
    neigh = isa.Neigh(
        n_sectors=neigh_n_sectors,
        max_neigh_per_sector=neigh_max_neigh_per_sector,
        ellipsoid_size=[neigh_dist_xy, neigh_dist_xy, neigh_dist_z],
    )

    # Run kriging
    runner = isa.Kriging()
    runner.set_input_data(input_db, coords=["x", "y", "z"], invars=indicator_names)
    runner.set_output_data(output_db, sel="mask")
    output_db = runner.kriging(model=multi_vario, neigh=neigh)

    # Isatis database to dataframe
    output_df = output_db.df()
    output_df.columns = output_df.columns.str.removesuffix("_" + runner.kriging_suffix)

    # Sort back to the original cell order
    output_df = output_df.sort_values("cell_id").reset_index(drop=True)

    # Keep only prediction columns
    values = output_df[indicator_names].to_numpy(dtype=np.float32)

    # Write results back to dataset
    grid_shape = pred_xyz["mask"].shape

    for i, var in enumerate(indicator_names):
        arr_xyz = values[:, i].reshape(grid_shape)

        da = pred_xyz["mask"].copy(data=arr_xyz).astype(np.float32).rename(var)

        new_da = da.transpose(*pred["mask"].dims)

        if var in pred:
            pred[var] = xr.where(pred["mask"], new_da, pred[var])
        else:
            pred[var] = new_da.where(pred["mask"])

    if verbose:
        
        dt = datetime.now() - t0
        m, s = divmod(round(dt.total_seconds()), 60)
        print(f"({m}m{s:02d}s)")


    return pred