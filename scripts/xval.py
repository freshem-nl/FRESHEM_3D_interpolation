import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime

from scripts import _read_and_write, _scoring, _postproc_helper, visualisation

def xval_lines(cfg):

    #TODO DEZE NOG VERSIMPELEN: NEEM 50% LANGSTE LIJNEN EN TREK DAARUIT

    # from config
    path_flightlines = cfg["path_preproc_data_flightlines"]
    n_lines = cfg["xval_n_lines"]

    # read flightlines
    xy_lines = _read_and_write.read_table(path_flightlines)

    # Count number of points per line
    counts = xy_lines.groupby("LINE_NO").size().sort_values(ascending=False)    

    # Candidate pool: top-K longest lines (but never smaller than n_lines if possible)
    k = min(int(75), len(counts))
    k = max(k, min(n_lines, len(counts)))
    pool_lines = counts.index[:k].to_numpy()

    # Sample exactly n_lines (or fewer if not enough lines exist)
    n_select = min(n_lines, len(pool_lines))
    rng = np.random.default_rng(42)
    selected_lines = rng.choice(pool_lines, size=n_select, replace=False).tolist()

    xy_selected = xy_lines[xy_lines["LINE_NO"].isin(selected_lines)].copy()
    return selected_lines, xy_selected


def mask_line(df, mask_overall, line_no):

    # get relevant XY for the line
    df = df.copy()
    df = df.loc[df["LINE_NO"] == line_no, ["X", "Y"]].drop_duplicates()

    # 2) coord -> index (exact match)
    x_index = pd.Index(mask_overall["x"].values)
    y_index = pd.Index(mask_overall["y"].values)

    ix = x_index.get_indexer(df["X"].to_numpy())
    iy = y_index.get_indexer(df["Y"].to_numpy())

    # 3) 2D mask met *paired* indexing
    mask_xy_np = np.zeros((mask_overall.sizes["y"], mask_overall.sizes["x"]), dtype=bool)
    mask_xy_np[iy, ix] = True

    mask_xy = xr.DataArray(
        mask_xy_np,
        coords={"y": mask_overall["y"], "x": mask_overall["x"]},
        dims=("y", "x"),
        name="mask_xy_line",
    )

    # broadcast to z and combine with old mask
    new_mask = mask_overall & mask_xy.broadcast_like(mask_overall)

    return new_mask

def validation(cfg):

    t0 = datetime.now()
    print("\nCROSS-VALIDATION SCORING")

    # from config
    path_obs = cfg["path_preproc_data_gridded"]
    path_pred = cfg["path_prediction_xval"]
    inds = np.array(cfg["indicators"])
    ind_bounds = cfg["indicator_bounds"]
    dir_output = cfg["dir_output"]
    dir_plot = cfg["dir_plot"]

    ind_cols = [f"P({b:g})" for b in inds]

    # read datasets
    ds_obs = _read_and_write.read_dataset(path_obs)
    ds_pred = _read_and_write.read_dataset(path_pred)

    # convert to dataframes and drop non-data variables and NaNs
    df_obs = ds_obs.to_dataframe().drop(columns=['spatial_ref', 'mask']).dropna()
    df_pred = ds_pred.to_dataframe().drop(columns=['spatial_ref']).dropna()

    #keep only df_obs with index in df_pred
    df_obs = df_obs[df_obs.index.isin(df_pred.index)]

    # calculate median quantiles and convert to class labels
    df_obs["median"] = _postproc_helper.ind_probs_to_quantiles(df_obs[ind_cols], inds, (0.5,), ind_bounds[0], ind_bounds[1])
    df_obs["median class"] = _postproc_helper.class_from_quantile(df_obs["median"], inds, ind_bounds)

    df_pred["median"] = _postproc_helper.ind_probs_to_quantiles(df_pred[ind_cols], inds, (0.5,), ind_bounds[0], ind_bounds[1])
    df_pred["median class"] = _postproc_helper.class_from_quantile(df_pred["median"], inds, ind_bounds)

    # calculate RPS (ranked probability score) for each cell, and put in dataframe
    print("...ranked probability score (RPS)")
    df_obs["rps"] = _scoring.rps_from_cdf(df_pred[ind_cols], df_obs[ind_cols], normalize=True)

    # summarize RPS per class, and save to csv
    path = dir_output / "ranked probability score per class.csv"
    _scoring.rps_summary(df_obs, rps_col="rps", class_col="median class", path=path)

     # confusion matrix for median class
    print("...confusion matrix")
    y_true = df_obs["median class"]
    y_pred = df_pred["median class"]
    labels = df_obs["median class"].cat.categories

    cms = [
        (confusion_matrix(y_true, y_pred, labels=labels),               "counts",               "d"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="true"), "row-normalized", ".2f"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="pred"), "col-normalized", ".2f"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="all"),  "normalized",      ".2f"),
    ]

    for cm, title, fmt in cms:
        path = dir_plot / f"pred - xval - confusion matrix - {title.replace(' ', '_')}.png"
        visualisation.plot_confusion_matrix(cm, labels, title, fmt, path)
    
    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)

   # classification report for median class
    print("...classification report")
    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)
    path = dir_output / "classification report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(cr)

    print(f"...done ({(datetime.now() - t0).total_seconds():.2f}s).")