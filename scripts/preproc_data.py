from datetime import datetime

import numpy as np
import pandas as pd

from scripts import _preproc_helper, _utils


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

    layer_numbers = [int(x.split("_")[1]) for x in df.columns if x.startswith("rho_") and "std" not in x]
    print(f"Drop measurements below {col_doi}, resampling layers to voxel centers with cellsize in z: {cellsize_z}")
    for i in layer_numbers:
        print(f"\tLayer {i}:", end=" ")

        # get relevant data for this layer
        cols = ["line_no", "x", "y", "elevation", f"rho_{i}", f"rho_std{i}", f"dep_top_{i}", f"dep_bot_{i}", col_doi]
        df_layer = df[cols].copy()

        # rename to stable names
        df_layer = df_layer.rename(columns={f"rho_{i}": "rho", f"rho_std{i}": "rho_std"})
        df_layer["z_top"] = df_layer["elevation"] - df_layer[f"dep_top_{i}"]
        df_layer["z_bot"] = df_layer["elevation"] - df_layer[f"dep_bot_{i}"]
        df_layer["z_doi"] = df_layer["elevation"] - df_layer[col_doi]
        df_layer["layer"] = i

        n_layers += len(df_layer)

        # ---- DOI handling: clip bottoms to DOI, then drop intervals fully below DO
        n0 = len(df_layer)
        df_layer["z_bot"] = np.maximum(df_layer["z_bot"], df_layer["z_doi"])
        df_layer = df_layer[df_layer["z_top"] > df_layer["z_doi"]]
        n_dropped += n0 - len(df_layer)
        print(f"{n0 - len(df_layer):,} layers below DOI", end="")

        if df_layer.empty:
            print(" (no remaining layers)")
        else:

            z_min = df_layer["z_bot"].min()
            z_max = df_layer["z_top"].max()

            # Get the z-centers of the grid cells based on the min and max z values and the cell size
            z_centers = _preproc_helper.min_max_to_cell_centers(z_min, z_max, cellsize_z)

            # Create a 2D boolean mask of shape (n_rows, n_z_centers)
            # Each row indicates which z_centers fall within [Z_BOT, Z_TOP] of that layer
            mask = (z_centers >= df_layer["z_bot"].values[:, None]) & (z_centers <= df_layer["z_top"].values[:, None])

            # Count number of voxel centers per row (i.e., how many True values per row)
            n = mask.sum(axis=1)
            n_dropped += (n == 0).sum()  # count rows with no matching z-centers
            print(
                f", {len(df_layer):,} resampled to {n.sum():,} points ({(n == 0).sum():,} without a voxel center)",
                end="\n",
            )

            # drop columns not needed for resampling
            drop_cols = ["elevation", "z_doi", col_doi, "dep_top_" + str(i), "dep_bot_" + str(i), "z_top", "z_bot"]
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
    df_z["z"] = np.concatenate(zs)

    # Reorder columns to have X, Y, Z first
    df_z = df_z[["x", "y", "z"] + [c for c in df_z.columns if c not in ["x", "y", "z"]]]

    # Convert to geodataframe
    df_z = _utils.df_to_gdf(df_z, epsg=epsg)

    txt = f"...resampled {n_layers:,} layers to {df_z.shape[0]:,} measurements, dropped {n_dropped:,} layers ({(datetime.now() - t0).total_seconds():.2f}s)"
    print(txt)

    return df_z


def quantiles_and_indicator_probs(df, cfg):

    from scipy.stats import norm

    # from config
    variable_name = cfg["variable_name"]
    inds = np.array(cfg["indicators"])
    quantiles = np.array(cfg["quantiles"])
    indicator_names = cfg["indicator_names"]
    quantile_names = cfg["quantile_names"]

    t0 = datetime.now()
    print(f"Calculating quantiles and indicator probabilities for {variable_name}...", end=" ")

    # --- 1) find unique (mu, std) combos
    mu_col = variable_name
    sd_col = f"{variable_name}_std"

    unique = df[[mu_col, sd_col]].drop_duplicates()
    mu = unique[mu_col].values[:, None]
    sd = unique[sd_col].values[:, None]

    # --- 2) compute threshold probabilities
    z = (inds[None, :] - mu) / sd
    p = norm.cdf(z)

    # handle sd == 0, in which case p should be 0 if inds < mu, 1 if inds > mu
    mask_zero = sd.squeeze() == 0
    if np.any(mask_zero):
        p[mask_zero, :] = (inds[None, :] >= mu[mask_zero, 0][:, None]).astype(np.float32)

    # --- 3) quantiles using norm.ppf
    # Q(p) = mu + sd * ppf(p)
    zq = norm.ppf(quantiles[None, :]).astype(np.float32)  # (1, K)
    qv = (mu * 1.0 + sd * zq).astype(np.float32)  # (U, K)

    # Handle sd == 0: all quantiles collapse to mu
    if np.any(mask_zero):
        qv[mask_zero, :] = mu[mask_zero, :]

    # --- 4) attach results to uniq
    # prob_cols = [f"P({i})" for i in inds]
    # q_cols = [f"Q{int(round(q * 100)):02d}" for q in quantiles]

    unique_out = pd.concat(
        [
            unique.reset_index(drop=True),
            pd.DataFrame(p, columns=indicator_names),
            pd.DataFrame(qv, columns=quantile_names),
        ],
        axis=1,
    )

    # --- 4) merge back
    df_out = df.merge(unique_out, on=[mu_col, sd_col], how="left", sort=False)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return df_out


def percentiles_to_indicators(data, cfg=None):

    variable_name = cfg["variable_name"]
    indicators = cfg["indicators"]
    indicator_names = cfg["indicator_names"]
    bounds = cfg["indicator_bounds"]

    # get percentile columns from data
    dtype = np.float32
    percentiles = [int(x.split("_")[1][1:]) for x in data.columns if x.startswith(variable_name.lower())]
    percentiles = np.asarray(percentiles, dtype=dtype)
    percentiles

    pcols = [f"cl_p{int(p)}" for p in percentiles]
    q = data[pcols].to_numpy(dtype=np.float32, copy=False)
    q = np.clip(q, *bounds)

    z = np.column_stack(
        [
            np.full(len(data), bounds[0], dtype=np.float32),
            q,
            np.full(len(data), bounds[1], dtype=np.float32),
        ]
    )
    p = np.r_[0.0, np.asarray(percentiles, dtype=np.float32) / 100.0, 1.0]

    out = np.empty((len(data), len(indicators)), dtype=np.float32)
    rows = np.arange(len(data))

    for j, t in enumerate(indicators):
        t = np.clip(t, *bounds)
        i = np.clip((z <= t).sum(axis=1) - 1, 0, z.shape[1] - 2)

        z0, z1 = z[rows, i], z[rows, i + 1]
        p0, p1 = p[i], p[i + 1]

        f = np.divide(
            t - z0,
            z1 - z0,
            out=np.zeros_like(z0),
            where=(z1 > z0),
        )
        out[:, j] = p0 + f * (p1 - p0)

    cols = indicator_names
    df_ind = pd.DataFrame(out, columns=cols, index=data.index)

    data[df_ind.columns] = df_ind

    return data


def resample_layers_to_z(data, cfg):

    t0 = datetime.now()

    # from config
    cellsize_z = cfg["cellsize_z"]
    z_centers = _preproc_helper.min_max_to_cell_centers(data["bottom"].min(), data["top"].max(), cellsize_z)

    z_centers = np.asarray(z_centers)

    top = data["top"].to_numpy()
    bottom = data["bottom"].to_numpy()

    # Indices in z_centers that fall within each row interval
    i0 = np.searchsorted(z_centers, bottom, side="right")
    i1 = np.searchsorted(z_centers, top, side="right")

    n = i1 - i0
    valid = n > 0

    # Repeat original row indices
    row_idx = np.repeat(np.flatnonzero(valid), n[valid])

    # Build z-center indices
    starts = np.repeat(i0[valid], n[valid])
    offsets = np.arange(n[valid].sum()) - np.repeat(np.cumsum(n[valid]) - n[valid], n[valid])
    z_idx = starts + offsets

    # Create expanded dataframe
    out = data.iloc[row_idx].copy()
    out["Z"] = z_centers[z_idx]
    out = out.reset_index(drop=True)

    out = out.sort_values(["X", "Y", "Z"], ascending=[True, True, False]).reset_index(drop=True)

    return out
