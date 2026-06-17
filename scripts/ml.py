from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor


def rf_train(df, cfg, verbose=True):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    indicator_names = cfg["indicator_names"]
    n_trees = cfg["rf_n_trees"]

    # output_names = [f"P({x})" for x in indicators]

    df = df.sample(frac=0.05, random_state=42)

    # train model

    model = RandomForestRegressor(
        n_estimators=n_trees, n_jobs=-1, max_depth=20, min_samples_leaf=2, max_features="sqrt", random_state=42
    )
    X = df[features]
    y = df[indicator_names]
    if verbose:
        print("\nSPATIAL INTERPOLATION")
        print(f"Training random forest on {len(X)} samples...", end=" ")
    model.fit(X, y)

    # save model
    # path = dir_output / "rf_model.joblib"
    # dump(model, path)

    if verbose:
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return model


def rf_predict(model, output_names, pred, cfg, xval=False, verbose=True):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    indicator_names = cfg["indicator_names"]
    
    feature_arrays = []
    for var in features:

        # Get feature from data_vars or coords
        if var in pred.data_vars:
            da = pred[var]
        elif var in pred.coords:
            da = pred.coords[var]

        # Broadcast feature to full 3D grid
        if not set(da.dims).issubset(pred["mask"].dims):
            raise ValueError(f"Feature '{var}' has incompatible dims {da.dims}")

        da = da.broadcast_like(pred["mask"]).transpose("Z", "Y", "X")

        # Add feature dimension while preserving the original feature name
        da = da.expand_dims(feature=[var])

        feature_arrays.append(da)

    # Combine all features into one DataArray: (Z, Y, X, feature)
    X_da = xr.concat(feature_arrays, dim="feature").transpose("Z", "Y", "X", "feature")

    # Stack spatial dims
    X_da = X_da.stack(cell=("Z", "Y", "X"), create_index=False).transpose("cell", "feature")

    # Stack mask in exactly the same way: (cell,)
    mask_1d = pred["mask"].transpose("Z", "Y", "X").stack(cell=("Z", "Y", "X"), create_index=False)

    # Build prediction dataframe
    X_pred = pd.DataFrame(X_da.values[mask_1d].astype(np.float32, copy=False), columns=X_da["feature"].values)

    # # if needed: coordinates to features
    # for var in features:
    #     # Create a 3D feature (z,y,x) that matches mask
    #     da = pred[var].broadcast_like(pred["mask"]).rename(var)
    #     pred = pred.assign({var: da})

    # # Stack spatial dims for features
    # X_da = (
    #     pred[features]
    #     .copy()
    #     .to_array("feature")  # (feature, z, y, x)
    #     .transpose("Z", "Y", "X", "feature")  # (z, y, x, feature)
    #     .stack(cell=("Z", "Y", "X"), create_index=False)  # (cell, feature)
    #     .transpose("cell", "feature")  # (cell, feature)
    # )

    # # Apply mask
    # mask_1d = pred["mask"].stack(cell=("Z", "Y", "X"), create_index=False)  # (cell,)

    # # valid cell = inside mask AND all features finite
    # valid = mask_1d.values & np.isfinite(X_da).all("feature").values

    # # to dataframe for sklearn
    # X_pred = pd.DataFrame(X_da.values[valid].astype(np.float32, copy=False), columns=features)

    if verbose:
        print(f"Predicting on {len(X_pred)} voxels...", end=" ")
    y_pred = model.predict(X_pred)


    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]


    # for i, out_name in enumerate(output_names):
    #     full = np.full(mask_1d.size, np.nan, dtype=np.float32)
    #     full[mask_1d] = y_pred[:, i]

    #     pred[out_name] = xr.DataArray(
    #         full.reshape(pred.sizes["Z"], pred.sizes["Y"], pred.sizes["X"]),
    #         dims=("Z", "Y", "X"),
    #         coords=pred.coords,
    #     )


    # Make a (cell, output) array filled with NaN
    full = np.full((mask_1d.size, len(indicator_names)), np.nan, dtype=np.float32)
    full[mask_1d.values, :] = y_pred.astype(np.float32, copy=False)


    # Reshape back to grid: (Z, Y, X, output) -> (output, Z, Y, X)
    pred_grid = xr.DataArray(
        full.reshape(pred.sizes["Z"], pred.sizes["Y"], pred.sizes["X"], len(indicator_names)),
        coords={"Z": pred["Z"], "Y": pred["Y"], "X": pred["X"], "output": indicator_names},
        dims=("Z", "Y", "X", "output"),
    ).transpose("output", "Z", "Y", "X")

    # # Create DataArray with same coords as mask
    # pred_cell = xr.DataArray(
    #     full,
    #     coords={"cell": mask_1d["cell"], "output": output_names},
    #     dims=("cell", "output"),
    # )

    # # unstack back to (z,y,x) and separate outputs into different DataArrays
    # pred_grid = pred_cell.unstack("cell").transpose("output", "Z", "Y", "X")  # (z,y,x,output)  # (output,z,y,x)

    # Add predicted outputs to dataset, replacing values where prediction is not NaN
    for name in indicator_names:
        new = pred_grid.sel(output=name).drop_vars("output")

        if name not in pred:
            pred[name] = new
        else:
            old = pred[name]
            pred[name] = xr.where(new.notnull(), new, old)


    if verbose:
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return pred
