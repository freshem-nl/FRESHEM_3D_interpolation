from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, confusion_matrix

from scripts import _postproc_helper, _read_and_write, _scoring, visualisation


def xval_lines(cfg):

    t0 = datetime.now()

    # from config
    path_flightlines = cfg["path_preproc_data_flightlines"]
    n_lines = cfg["xval_n_lines"]

    print(f"\nselect {n_lines} lines", end="... ")

    # read flightlines
    xy_lines = _read_and_write.read_table(path_flightlines)

    # Count number of points per line
    counts = xy_lines.groupby("LINE_NO").size().sort_values(ascending=False)

    # select top_lines from the longest lines, at least 50% of the lines
    n_lines_total = xy_lines["LINE_NO"].nunique()
    fraction_to_select = n_lines/n_lines_total
    fraction_to_select_from = max(0.5, fraction_to_select)  # at least top 50% lines
    n_top = int(np.ceil(len(counts) * fraction_to_select_from))
    top_lines = counts.index[:n_top].tolist()

    # random sample n_lines from top_lines without replacement, with fixed seed for reproducibility
    rng = np.random.default_rng(cfg.get("seed", 42))  # or cfg["seed"]
    n_pick = min(n_lines, len(top_lines))  # safety if n_lines > available
    selected_lines = rng.choice(top_lines, size=n_pick, replace=False).tolist()

    # filter xy_lines to selected lines
    xy_lines_selected = xy_lines[xy_lines["LINE_NO"].isin(selected_lines)].copy()

    print(f'done ({(datetime.now() - t0).total_seconds():.2f}s).')

    return xy_lines_selected


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
    dir_data = cfg["dir_data"]
    dir_xval = cfg["dir_xval"]

    ind_cols = [f"P({b:g})" for b in inds]

    # read datasets
    ds_obs = _read_and_write.read_dataset(path_obs)
    ds_pred = _read_and_write.read_dataset(path_pred)

    # convert to dataframes and drop non-data variables and NaNs
    df_obs = ds_obs.to_dataframe().drop(columns=["spatial_ref", "mask"]).dropna()
    df_pred = ds_pred.to_dataframe().drop(columns=["spatial_ref"]).dropna()

    # keep only df_obs with index in df_pred
    df_obs = df_obs[df_obs.index.isin(df_pred.index)]

    # calculate median quantiles and convert to class labels
    df_obs["median"] = _postproc_helper.ind_probs_to_quantiles(
        df_obs[ind_cols], inds, (0.5,), ind_bounds[0], ind_bounds[1]
    )
    df_obs["median class"] = _postproc_helper.class_from_quantile(df_obs["median"], inds, ind_bounds)

    df_pred["median"] = _postproc_helper.ind_probs_to_quantiles(
        df_pred[ind_cols], inds, (0.5,), ind_bounds[0], ind_bounds[1]
    )
    df_pred["median class"] = _postproc_helper.class_from_quantile(df_pred["median"], inds, ind_bounds)

    # calculate RPS (ranked probability score) for each cell, and put in dataframe
    print("...ranked probability score (RPS)")
    rps = _scoring.rps_from_cdf(df_pred[ind_cols], df_obs[ind_cols], normalize=True)

    # summarize RPS overall and per class, and save to csv
    path = dir_xval / "xval - ranked probability score.csv"
    _scoring.rps_summary(rps, df_obs["median class"], path=path)

    # boxplot of overall RPS
    path = dir_xval / "xval - RPS.png"
    visualisation.boxplot(df_obs.assign(RPS=rps), y="RPS", path=path, showfliers=False)

    # boxplot of RPS by true class
    path = dir_xval / "xval - RPS by true class.png"
    visualisation.boxplot(df_obs.assign(RPS=rps), x="median class", y="RPS", path=path, showfliers=False)

    # confusion matrix for median class
    print("...confusion matrix")
    y_true = df_obs["median class"]
    y_pred = df_pred["median class"]
    labels = df_obs["median class"].cat.categories

    cms = [
        (confusion_matrix(y_true, y_pred, labels=labels), "counts", "d"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="true"), "row-normalized", ".2f"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="pred"), "col-normalized", ".2f"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="all"), "normalized", ".2f"),
    ]

    for cm, title, fmt in cms:
        path = dir_xval / f"xval - confusion matrix - {title.replace(' ', '_')}.png"
        visualisation.plot_confusion_matrix(cm, labels, title, fmt, path)

    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)

    # classification report for median class
    print("...classification report")
    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)
    path = dir_xval / "xval - classification report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(cr)

    # save results
    path = dir_data / "xval - rps.parquet"
    _read_and_write.write_table(rps, path)

    print(f"...done ({(datetime.now() - t0).total_seconds():.2f}s).")
