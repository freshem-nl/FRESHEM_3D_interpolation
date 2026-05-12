import os
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts import _preprocessing_helper, _read_and_write


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
    df_z = gpd.GeoDataFrame(df_z, geometry=gpd.points_from_xy(df_z["X"], df_z["Y"]), crs=f"EPSG:{epsg}")

    txt = f"...resampled {n_layers:,} layers to {df_z.shape[0]:,} measurements, dropped {n_dropped:,} layers ({(datetime.now() - t0).total_seconds():.2f}s)."
    print(txt)

    return df_z


def calc_indicators(df, cfg):
    from scipy.stats import norm

    # from config
    variable_name = cfg["variable_name"]
    inds = np.array(cfg["indicators"])
    path_df_out = cfg["path_preproc_data"]

    t0 = datetime.now()
    print(f"Calculating indicators for {variable_name}...", end=" ")

    # Calculate z-scores and probabilities for each indicator
    z = (inds[None, :] - df[f"{variable_name}"].values[:, None]) / df[f"{variable_name}_STD"].values[:, None]
    p = norm.cdf(z)

    # create dataframe with indicator columns
    df_probs = pd.DataFrame(p, columns=[f"P({i})" for i in inds])
    df_out = pd.concat([df, df_probs], axis=1)

    # save
    _read_and_write.write_table(df_out, path_df_out.with_suffix(".parquet"))

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return df_out


def plotting(df, cfg):

    t0 = datetime.now()
    print("Plotting...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]

    # sample data for histogram plotting
    n = 10000

    # in case it's a geodataframe, to avoid geometry column issues
    df = df.drop(columns="geometry", errors="ignore")

    os.makedirs(dir_plot, exist_ok=True)
    for var in df.columns:

        n_df = df[var].notna().sum()
        df_plot = df[var].dropna().sample(n=min(n, n_df), random_state=42)

        plt.figure()
        sns.histplot(df_plot, bins=20, kde=False)
        plt.title(f"{var}, n={n_df:,}")

        path = dir_plot / f"data - {var}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
