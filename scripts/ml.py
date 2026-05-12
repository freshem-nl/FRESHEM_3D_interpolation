from datetime import datetime
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import xarray as xr
from scripts import _read_and_write

def rf_train(df, cfg):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    indicators = cfg["indicators"]
    dir_output = cfg["dir_output"]
    n_trees = cfg["rf_n_trees"]

    output_names = [f'P({x})' for x in indicators]

    df = df.sample(frac=0.05, random_state=42)

    # train model

    model = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, max_depth=20,min_samples_leaf=2, max_features="sqrt", random_state=42)
    X = df[features]
    y = df[output_names]
    print(f"Training random forest on {len(X)} samples...", end=" ")
    model.fit(X, y)

    # save model
    path = dir_output / "rf_model.joblib"
    dump(model, path)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return model, output_names

def rf_predict(model, output_names, ds, cfg):

    t0 = datetime.now()

    # from config
    features = cfg["features"]
    path_output = cfg["path_prediction"]

    # make features
    if 'Z' in features:
        # Create a 3D Z feature (z,y,x) that matches mask
        Z3 = ds["z"].broadcast_like(ds["mask"]).rename("Z")
        feat_ds = ds.assign(Z=Z3)


    # Stack spatial dims
    X_da = (
        feat_ds[features].copy()
        .to_array("feature")                    # (feature, z, y, x)
        .transpose("z", "y", "x", "feature")    # (z, y, x, feature)
        .stack(cell=("z", "y", "x")) 
        .transpose("cell", "feature")            # (cell, feature)
    )

    # Apply mask
    mask_1d = ds["mask"].stack(cell=("z", "y", "x"))

    # valid cell = inside mask AND all features finite
    valid = mask_1d.values & np.isfinite(X_da).all("feature").values

    # ---- extract numpy matrix for sklearn ----
    X_pred = pd.DataFrame(X_da.values[valid].astype(np.float32, copy=False), columns=features)

    print(f"Predicting on {len(X_pred)} voxels...", end=" ")
    y_pred = model.predict(X_pred)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    elif y_pred.ndim == 2:
        pass

    # Make a (cell, output) array filled with NaN
    full = np.full((mask_1d.size, len(output_names)), np.nan, dtype=np.float32)
    full[valid, :] = y_pred.astype(np.float32, copy=False)

    # unstack back to (z,y,x) and separate outputs into different DataArrays
    pred_cell = xr.DataArray(
        full,
        coords={"cell": mask_1d["cell"], "output": output_names},
        dims=("cell", "output"),
    )

    # unstack back to (z,y,x) and separate outputs into different DataArrays
    pred_grid = (
        pred_cell
        .unstack("cell")                      # (z,y,x,output)
        .transpose("output", "z", "y", "x")   # (output,z,y,x)
    )

    # Create new dataset with predictions as separate DataArrays
    ds_pred = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    for name in output_names:
        ds_pred[name] = pred_grid.sel(output=name).drop_vars("output")

    # save predictions
    _read_and_write.write_dataset(ds_pred, path_output)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")