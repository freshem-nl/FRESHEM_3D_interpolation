from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from scripts import _read_and_write


def rf_train(df, cfg, verbose=True):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    indicators = cfg["indicators"]
    dir_output = cfg["dir_output"]
    n_trees = cfg["rf_n_trees"]

    output_names = [f"P({x})" for x in indicators]

    df = df.sample(frac=0.05, random_state=42)

    # train model

    model = RandomForestRegressor(
        n_estimators=n_trees, n_jobs=-1, max_depth=20, min_samples_leaf=2, max_features="sqrt", random_state=42
    )
    X = df[features]
    y = df[output_names]
    if verbose:
        print(f"Training random forest on {len(X)} samples...", end=" ")
    model.fit(X, y)

    # save model
    path = dir_output / "rf_model.joblib"
    dump(model, path)

    if verbose:
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return model, output_names


def rf_predict(model, output_names, ds_feat, cfg, ds_pred=None, xval=False, verbose=True):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    if xval:
        path_output = cfg["path_prediction_xval"]
    else:
        path_output = cfg["path_prediction"]

    # if needed: coordinates to features
    for var in ["X", "Y", "Z"]:
        if var in features:
            # Create a 3D feature (z,y,x) that matches mask
            da = ds_feat[var.lower()].broadcast_like(ds_feat["mask"]).rename(var)
            ds_feat = ds_feat.assign({var: da})

    # Stack spatial dims for features
    X_da = (
        ds_feat[features]
        .copy()
        .to_array("feature")  # (feature, z, y, x)
        .transpose("z", "y", "x", "feature")  # (z, y, x, feature)
        .stack(cell=("z", "y", "x"))
        .transpose("cell", "feature")  # (cell, feature)
    )

    # Apply mask
    mask_1d = ds_feat["mask"].stack(cell=("z", "y", "x"))

    # valid cell = inside mask AND all features finite
    valid = mask_1d.values & np.isfinite(X_da).all("feature").values

    # to dataframe for sklearn
    X_pred = pd.DataFrame(X_da.values[valid].astype(np.float32, copy=False), columns=features)

    if verbose:
        print(f"Predicting on {len(X_pred)} voxels...", end=" ")
    y_pred = model.predict(X_pred)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    elif y_pred.ndim == 2:
        pass

    # Make a (cell, output) array filled with NaN
    full = np.full((mask_1d.size, len(output_names)), np.nan, dtype=np.float32)
    full[valid, :] = y_pred.astype(np.float32, copy=False)

    # Create DataArray with same coords as mask
    pred_cell = xr.DataArray(
        full,
        coords={"cell": mask_1d["cell"], "output": output_names},
        dims=("cell", "output"),
    )

    # unstack back to (z,y,x) and separate outputs into different DataArrays
    pred_grid = pred_cell.unstack("cell").transpose("output", "z", "y", "x")  # (z,y,x,output)  # (output,z,y,x)

    if ds_pred is None:
        # Create new dataset with predictions as separate DataArrays (if multiple outputs)
        ds_pred = xr.Dataset(coords=ds_feat.coords, attrs=ds_feat.attrs)

    # Add predicted outputs to dataset, replacing values where prediction is not NaN
    for name in output_names:
        new = pred_grid.sel(output=name).drop_vars("output")

        if name not in ds_pred:
            ds_pred[name] = new
        else:
            old = ds_pred[name]
            ds_pred[name] = xr.where(new.notnull(), new, old)

    # save predictions
    _read_and_write.write_dataset(ds_pred, path_output)

    if verbose:
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return ds_pred
