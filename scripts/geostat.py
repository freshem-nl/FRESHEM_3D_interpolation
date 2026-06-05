import isatis as isa
import isatis.constants as cst
import numpy as np
from datetime import datetime

isa.setLicenseString("52100@lic-isatis.tno.nl")


def kriging(data, pred, cfg):
    t0 = datetime.now()
    print("KRIGING")

    # From config
    indicator_names = cfg["indicator_names"]
    range_xy = cfg["variogram_model_range_xy"]
    range_z = cfg["variogram_model_range_z"]
    neigh_dist_xy = cfg["neighbourhood_dist_xy"]
    neigh_dist_z = cfg["neighbourhood_dist_z"]
    neigh_n_sectors = cfg["neighbourhood_n_sectors"]
    neigh_max_neigh_per_sector = cfg["neighbourhood_max_neigh_per_sector"]

    # Create Isatis input database
    input_db = isa.DbPandas(data.to_dataframe().reset_index())

    # Use one fixed dimension order for everything sent to Isatis
    pred_xyz = pred.transpose("X", "Y", "Z")

    # Create Isatis output grid
    grid = isa.GridGeom(origin=[
                            pred_xyz["X"].values.min(),
                            pred_xyz["Y"].values.min(),
                            pred_xyz["Z"].values.min()],
                        cell_size=[
                            pred_xyz.attrs["cellsize_x"],
                            pred_xyz.attrs["cellsize_y"],
                            pred_xyz.attrs["cellsize_z"]],
                        nxyz=[
                            pred_xyz.sizes["X"],
                            pred_xyz.sizes["Y"],
                            pred_xyz.sizes["Z"]],
                        ndim=3)

    # Build output dataframe in the exact same grid order

    base_df = pred_xyz["mask"].to_dataframe().reset_index()

    # Assign a unique cell id in the exact original flat order
    base_df["cell_id"] = np.arange(len(base_df), dtype=np.int64)

    # Keep only fields needed for the output database
    df_out = base_df[["mask", "cell_id"]].copy()


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
    runner.set_input_data(input_db, coords=["X", "Y", "Z"], invars=indicator_names)
    runner.set_output_data(output_db, sel="mask")
    output_db = runner.kriging(model=multi_vario, neigh=neigh)

    # Isatis database to dataframe
    output_df = output_db.df().copy()
    output_df.columns = output_df.columns.str.removesuffix("_" + runner.kriging_suffix)
    
    # Sort back to the original cell order
    output_df = output_df.sort_values("cell_id").reset_index(drop=True)

    # Keep only prediction columns
    values = output_df[indicator_names].to_numpy(dtype=np.float32)


    # Write results back to dataset
    pred_out = pred.copy()
    grid_shape = pred_xyz["mask"].shape
    n_cells = np.prod(grid_shape)


    for i, var in enumerate(indicator_names):
        arr_xyz = values[:, i].reshape(grid_shape)

        da = pred_xyz["mask"].copy(data=arr_xyz).astype(np.float32).rename(var)

        # Restore original dimension order
        pred_out[var] = da.transpose(*pred["mask"].dims)

    print(f"done {datetime.now() - t0}")

    return pred_out


# import numpy as np
# import os

# import isatis as isa
# import isatis.constants as cst
# import pandas as pd
# from scripts import config_loader, visualisation, _utils
# from pathlib import Path

# isa.setLicenseString('52100@lic-isatis.tno.nl')

# def kriging(data, pred, cfg):

#     # from config
#     # indicators = cfg["indicators"]
#     indicator_names = cfg["indicator_names"]
#     range_xy = cfg["variogram_model_range_xy"]
#     range_z = cfg["variogram_model_range_z"]
#     neigh_dist_xy = cfg['neighbourhood_dist_xy']
#     neigh_dist_z = cfg['neighbourhood_dist_z']
#     neigh_n_sectors = cfg['neighbourhood_n_sectors']
#     neigh_max_neigh_per_sector = cfg['neighbourhood_max_neigh_per_sector']

#     # Create isatis input databases from the input DataFrames:
#     input_db  = isa.DbPandas(data.to_dataframe().reset_index())

#     # Use one fixed dimension order for everything sent to Isatis
#     pred_xyz = pred.transpose("X", "Y", "Z")

#     # Create Isatis output database from full regular grid dataset
#     grid = isa.GridGeom(
#         origin=[pred_xyz["X"].values.min(), pred_xyz["Y"].values.min(), pred_xyz["Z"].values.min()],
#         cell_size=[pred_xyz.attrs["cellsize_x"], pred_xyz.attrs["cellsize_y"], pred_xyz.attrs["cellsize_z"]],
#         nxyz=[pred_xyz.sizes["X"], pred_xyz.sizes["Y"], pred_xyz.sizes["Z"]],
#         ndim=len(pred_xyz.sizes),
#     )

#     # dataframe of all pred variables, properly ordered
#     df_out = (
#         pred_xyz.to_dataframe()
#         .reset_index()
#         .drop(columns=["X", "Y", "Z"])
#     )


#     output_db = isa.DbPandas(df_out, grid=grid)

#     # Make multivariate variogram model
#     n_var = len(indicator_names)

#     # create the sill martrix
#     multi_vario = isa.VModel(nvar=n_var)
#     sill_matrix = np.zeros((n_var, n_var))
#     nugg_matrix = np.zeros((n_var, n_var))
#     for i in range(n_var):
#         sill_matrix[i, i] = 0.33
#         nugg_matrix[i, i] = 0.001

#     # add the structure of the multivariate model
#     sph_struct = isa.VStruc(stype=cst.MOD.SPH,
#                             nvar=n_var,
#                             ranges=[range_xy, range_xy, range_z],
#                             sill=sill_matrix)
#     multi_vario.add_struct(sph_struct)
#     multi_vario.set_nugget(nugg_matrix)

#     # define neighbourhood
#     neigh = isa.Neigh(n_sectors=neigh_n_sectors, max_neigh_per_sector=neigh_max_neigh_per_sector, ellipsoid_size=[neigh_dist_xy, neigh_dist_xy, neigh_dist_z])

#     # Initialize a calculator to do the Kriging:
#     runner = isa.Kriging()

#     # Tell the calculator what the input data is.
#     runner.set_input_data(input_db, coords = ["X", "Y", "Z"], invars = indicator_names)

#     # Tell the calculator what the output data is.
#     runner.set_output_data(output_db, sel='mask')

#     # Run the Kriging
#     output_db = runner.kriging(model = multi_vario, neigh=neigh)

#     # isatis database to dataframe
#     output_df = output_db.df().copy()
#     output_df.columns = output_df.columns.str.removesuffix('_' + runner.kriging_suffix)
#     output_df = output_df[indicator_names]

#     # Write results back to dataset
#     pred_out = pred.copy()
#     grid_shape = (pred.sizes["X"], pred.sizes["Y"], pred.sizes["Z"])
#     dims = ("X", "Y", "Z")

#     values = output_df.to_numpy(dtype=np.float32)

#     for i, var in enumerate(indicator_names):
#         arr = values[:, i].reshape(grid_shape)
#         pred_out[var] = (dims, arr)

#     return pred_out
