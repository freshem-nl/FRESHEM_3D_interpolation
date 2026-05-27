from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from scripts import _preprocessing_helper, _read_and_write, _utils


def drop_below_doi_and_resample_layers_to_z(df, cfg):

    t0 = datetime.now()

    # from config
    cellsize_z = cfg["cellsize_z"]
    col_doi = cfg["doi_name"]
    epsg = cfg["epsg"]

    n_layers = 0
    n_dropped = 0

    dfs = []
    zs = []

    layer_numbers = [int(x.split("_")[1]) for x in df.columns if x.startswith("RHO") and "STD" not in x]
    print(f"Drop measurements below {col_doi}, resampling layers to voxel centers with cellsize in z: {cellsize_z}")
    for i in layer_numbers:
        print(f"\tLayer {i}:", end=" ")

        # get relevant data for this layer
        cols = ["LINE_NO", "X", "Y", "ELEVATION", f"RHO_{i}", f"RHO_STD{i}", f"DEP_TOP_{i}", f"DEP_BOT_{i}", col_doi]
        df_layer = df[cols].copy()

        # rename to stable names
        df_layer = df_layer.rename(columns={f"RHO_{i}": "RHO", f"RHO_STD{i}": "RHO_STD"})
        df_layer["Z_TOP"] = df_layer["ELEVATION"] - df_layer[f"DEP_TOP_{i}"]
        df_layer["Z_BOT"] = df_layer["ELEVATION"] - df_layer[f"DEP_BOT_{i}"]
        df_layer["Z_DOI"] = df_layer["ELEVATION"] - df_layer[col_doi]
        df_layer["LAYER"] = i

        n_layers += len(df_layer)

        # ---- DOI handling: clip bottoms to DOI, then drop intervals fully below DO
        n0 = len(df_layer)
        df_layer["Z_BOT"] = np.maximum(df_layer["Z_BOT"], df_layer["Z_DOI"])
        df_layer = df_layer[df_layer["Z_TOP"] > df_layer["Z_DOI"]]
        n_dropped += n0 - len(df_layer)
        print(f"{n0 - len(df_layer):,} layers below DOI", end="")

        if df_layer.empty:
            print(" (no remaining layers)")
        else:

            z_min = df_layer["Z_BOT"].min()
            z_max = df_layer["Z_TOP"].max()

            # Get the z-centers of the grid cells based on the min and max z values and the cell size
            z_centers = _preprocessing_helper.min_max_to_cell_centers(z_min, z_max, cellsize_z)

            # Create a 2D boolean mask of shape (n_rows, n_z_centers)
            # Each row indicates which z_centers fall within [Z_BOT, Z_TOP] of that layer
            mask = (z_centers >= df_layer["Z_BOT"].values[:, None]) & (z_centers <= df_layer["Z_TOP"].values[:, None])

            # Count number of voxel centers per row (i.e., how many True values per row)
            n = mask.sum(axis=1)
            n_dropped += (n == 0).sum()  # count rows with no matching z-centers
            print(
                f", {len(df_layer):,} resampled to {n.sum():,} points ({(n == 0).sum():,} without a voxel center)",
                end="\n",
            )

            # drop columns not needed for resampling
            drop_cols = ["ELEVATION", "Z_DOI", col_doi, "DEP_TOP_" + str(i), "DEP_BOT_" + str(i), "Z_TOP", "Z_BOT"]
            df_layer = df_layer.drop(columns=drop_cols)

            # Repeat each row n times so that we can assign one z_center per voxel later
            # Rows with n=0 are automatically dropped (no matching voxels)
            df_rep = df_layer.loc[df_layer.index.repeat(n)].reset_index(drop=True)

            # Extract the matching z_centers:
            # np.where(mask) returns (row_indices, col_indices)
            # We take the column indices → positions in z_centers
            z_rep = z_centers[np.where(mask)[1]]

            # Collect repeated rows and matching z-centers separately for each layer, to combine later
            dfs.append(df_rep)
            zs.append(z_rep)

    # Concatenate all repeated rows from all layers into one DataFrame
    df_z = pd.concat(dfs, ignore_index=True)

    # Concatenate all z_center arrays and assign them to the DataFrame
    # Length matches df_z by construction (1 z_center per repeated row)
    df_z["Z"] = np.concatenate(zs)

    # Reorder columns to have X, Y, Z first
    df_z = df_z[["X", "Y", "Z"] + [c for c in df_z.columns if c not in ["X", "Y", "Z"]]]

    # Convert to geodataframe
    df_z = _utils.df_to_gdf(df_z, epsg=epsg)

    txt = f"...resampled {n_layers:,} layers to {df_z.shape[0]:,} measurements, dropped {n_dropped:,} layers ({(datetime.now() - t0).total_seconds():.2f}s)."
    print(txt)

    return df_z

def quantiles_and_indicator_probs(df, cfg):

    from scipy.stats import norm

    # from config
    variable_name = cfg["variable_name"]
    inds = np.array(cfg["indicators"])
    quantiles = np.array(cfg["quantiles"])
    path_df_out = cfg["path_preproc_data"]

    t0 = datetime.now()
    print(f"Calculating quantiles and indicator probabilities for {variable_name}...", end=" ")

    # --- 1) find unique (mu, std) combos
    mu_col = variable_name
    sd_col = f"{variable_name}_STD"

    unique = df[[mu_col, sd_col]].drop_duplicates()
    mu = unique[mu_col].values[:, None]
    sd = unique[sd_col].values[:, None]

    # --- 2) compute probabilities
    z = (inds[None, :] - mu) / sd
    p = norm.cdf(z)

    # handle sd == 0, in which case p should be 0 if inds < mu, 1 if inds > mu
    mask_zero = (sd.squeeze() == 0)
    if np.any(mask_zero):
        p[mask_zero, :] = (inds[None, :] >= mu[mask_zero, 0][:, None]).astype(np.float32)


    # --- 3) quantiles using norm.ppf
    # Q(p) = mu + sd * ppf(p)
    zq = norm.ppf(quantiles[None, :]).astype(np.float32)          # (1, K)
    qv = (mu * 1.0 + sd * zq).astype(np.float32)                 # (U, K)

    # Handle sd == 0: all quantiles collapse to mu
    if np.any(mask_zero):
        qv[mask_zero, :] = mu[mask_zero, :]


    # --- 4) attach results to uniq
    prob_cols = [f"P({i})" for i in inds]
    q_cols = [f"Q{int(round(q * 100)):02d}" for q in quantiles]

    unique_out = pd.concat(
        [unique.reset_index(drop=True),
         pd.DataFrame(p, columns=prob_cols),
         pd.DataFrame(qv, columns=q_cols)],
        axis=1
    )

    # --- 4) merge back
    df_out = df.merge(unique_out, on=[mu_col, sd_col], how="left", sort=False)

    # save
    _read_and_write.write_table(df_out, path_df_out.with_suffix(".parquet"))

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return df_out
