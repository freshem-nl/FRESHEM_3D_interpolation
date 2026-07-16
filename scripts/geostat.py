from datetime import datetime

import isatis as isa
import isatis.constants as cst
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


def kriging(data, pred, cfg, verbose=True):
    t0 = datetime.now()

    # Set Isatis license
    isa.setLicenseString("52100@lic-isatis.tno.nl")

    if verbose:
        print("\nSPATIAL INTERPOLATION")
        tqdm_leave = True
        tqdm_position = 0
    else:
        tqdm_leave = False
        tqdm_position = 1

    # From config
    indicator_names = cfg["indicator_names"]
    range_xy = cfg["variogram_model_range_xy"]
    neigh_dist_xy = cfg["neighbourhood_dist_xy"]
    neigh_dist_z = cfg["neighbourhood_dist_z"]
    neigh_n_sectors = cfg["neighbourhood_n_sectors"]
    neigh_max_neigh_per_sector = cfg["neighbourhood_max_neigh_per_sector"]
    use_anisotropy = cfg["use_anisotropy"]

    layers = data["layer"].unique()
    txt = "interpolation per layer"
    for layer in tqdm(layers, desc=txt, unit="layer", leave=tqdm_leave, position=tqdm_position):

        # Create Isatis input database
        input_db = isa.DbPandas(data.loc[data["layer"] == layer].reset_index())

        # Use one fixed dimension order for everything sent to Isatis
        pred_xy = pred.sel(layer=layer).transpose("x", "y")

        # Create Isatis output grid
        grid = isa.GridGeom(
            origin=[pred_xy["x"].values.min(), pred_xy["y"].values.min()],
            cell_size=[pred_xy.attrs["cellsize_x"], pred_xy.attrs["cellsize_y"]],
            nxyz=[pred_xy.sizes["x"], pred_xy.sizes["y"]],
            ndim=2,
        )

        # Build output dataframe in the exact same grid order
        df_out = pd.DataFrame(
            {
                "mask": pred_xy["mask"].values.ravel(),
                "cell_id": np.arange(pred_xy["mask"].size),
            }
        )

        # Add anisotropy variables to output dataframe if needed
        if use_anisotropy:
            df_out = df_out.assign(
                laf_major_angle=pred_xy["laf_major_angle"].values.ravel(),
                laf_factor_minor=pred_xy["laf_ratio"].values.ravel(),
                laf_factor_major=1.0,
            )

        # Create Isatis output database
        output_db = isa.DbPandas(df_out, grid=grid)

        # Make multivariate variogram model
        n_var = len(indicator_names)

        multi_vario = isa.VModel(nvar=n_var)
        sill_matrix = np.zeros((n_var, n_var))
        nugg_matrix = np.zeros((n_var, n_var))

        for i in range(n_var):
            sill_matrix[i, i] = 0.33
            nugg_matrix[i, i] = 0.001

        range_dummy = range_xy
        sph_struct = isa.VStruc(
            stype=cst.MOD.SPH,
            nvar=n_var,
            ndim=2,
            ranges=[range_xy, range_xy, range_dummy],
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

        # Create Local Geostatistics Set structure
        if use_anisotropy:
            aniso_rot_name = ["laf_major_angle"]
            names_factors = ["laf_factor_major", "laf_factor_minor"]
            lgs_vars = isa.LgsNames(conv_id=isa.CONV.MATH, names_rotation=aniso_rot_name, names_factors=names_factors)

        # Run kriging
        runner = isa.Kriging()
        runner.set_input_data(input_db, coords=["x", "y"], invars=indicator_names)
        
        if use_anisotropy:
            # runner.set_output_data(output_db, sel="mask", lgs_model_vars=lgs_vars)
            # runner.set_output_data(output_db, sel="mask", lgs_neigh_vars=lgs_vars)
            runner.set_output_data(output_db, sel="mask", lgs_model_vars=lgs_vars,lgs_neigh_vars=lgs_vars)
        else:
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
        grid_shape = pred_xy["mask"].shape

        for i, var in enumerate(indicator_names):
            arr_xy = values[:, i].reshape(grid_shape)

            da_xy = pred_xy["mask"].copy(data=arr_xy).astype(np.float32).rename(var)

            da_yx = da_xy.transpose("y", "x")

            # Initialise output variable if it does not exist yet
            if var not in pred:
                pred[var] = xr.full_like(
                    pred["mask"].transpose("layer", "y", "x"),
                    fill_value=np.nan,
                    dtype=np.float32,
                )

            # overwrite with new values only within mask
            mask = pred["mask"].sel(layer=layer)

            old = pred[var].sel(layer=layer)

            new = old.where(~mask, da_yx)

            pred[var].loc[{"layer": layer}] = new

    pred = pred.drop_vars("mask")

    if verbose:

        dt = datetime.now() - t0
        m, s = divmod(round(dt.total_seconds()), 60)
        print(f"({m}m{s:02d}s)")

    return pred
