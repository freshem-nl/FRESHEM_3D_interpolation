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

    if verbose:
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return model


def rf_predict(model, pred, cfg, xval=False, verbose=True):

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

        da = da.broadcast_like(pred["mask"]).transpose("layer", "y", "x")

        # Add feature dimension while preserving the original feature name
        da = da.expand_dims(feature=[var])

        feature_arrays.append(da)

    # Combine all features into one DataArray: (layer, y, x, feature)
    X_da = xr.concat(feature_arrays, dim="feature").transpose("layer", "y", "x", "feature")

    # Stack spatial dims
    X_da = X_da.stack(cell=("layer", "y", "x"), create_index=False).transpose("cell", "feature")

    # Stack mask in exactly the same way: (cell,)
    mask_1d = pred["mask"].transpose("layer", "y", "x").stack(cell=("layer", "y", "x"), create_index=False)

    # Build prediction dataframe
    X_pred = pd.DataFrame(X_da.values[mask_1d].astype(np.float32, copy=False), columns=X_da["feature"].values)

    if verbose:
        print(f"Predicting on {len(X_pred)} voxels...", end=" ")
    y_pred = model.predict(X_pred)

    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # Make a (cell, output) array filled with NaN
    full = np.full((mask_1d.size, len(indicator_names)), np.nan, dtype=np.float32)
    full[mask_1d.values, :] = y_pred.astype(np.float32, copy=False)

    # Reshape back to grid: (layer, y, x, output) -> (output, layer, y, x)
    pred_grid = xr.DataArray(
        full.reshape(pred.sizes["layer"], pred.sizes["y"], pred.sizes["x"], len(indicator_names)),
        coords={"layer": pred["layer"], "y": pred["y"], "x": pred["x"], "output": indicator_names},
        dims=("layer", "y", "x", "output"),
    ).transpose("output", "layer", "y", "x")

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
