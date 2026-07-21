from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report, confusion_matrix

from scripts import _postproc_helper, _scoring, visualisation, write


def xval_lines(data, cfg):

    t0 = datetime.now()
    print("\nSPATIAL INTERPOLATION CROSS-VALIDATION")

    # from config
    n_lines = cfg["xval_n_lines"]

    print(f"select {n_lines} lines", end="... ")

    # Count number of points per line
    counts = data.groupby("line_no").size().sort_values(ascending=False)

    # select top_lines from the longest lines, at least 50% of the lines
    n_lines_total = data["line_no"].nunique()
    fraction_to_select = n_lines / n_lines_total
    fraction_to_select_from = max(0.5, fraction_to_select)  # at least top 50% lines
    n_top = int(np.ceil(len(counts) * fraction_to_select_from))
    top_lines = counts.index[:n_top].tolist()

    # random sample n_lines from top_lines without replacement, with fixed seed for reproducibility
    rng = np.random.default_rng(cfg.get("seed", 42))  # or cfg["seed"]
    n_pick = min(n_lines, len(top_lines))  # safety if n_lines > available
    selected_lines = rng.choice(top_lines, size=n_pick, replace=False).tolist()

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return selected_lines


def mask_line(data, mask_overall, lines):

    lines = np.atleast_1d(lines)

    data_line = data.loc[data["line_no"].isin(lines), ["x", "y", "layer"]]

    mask_line = xr.zeros_like(mask_overall, dtype=bool)

    x = mask_overall.x.values
    y = mask_overall.y.values

    dx = np.diff(x).mean()
    dy = np.diff(y).mean()

    x0 = x[0] - dx / 2
    y0 = y[0] - dy / 2

    ix = np.floor((data_line["x"] - x0) / dx).astype(int)
    iy = np.floor((data_line["y"] - y0) / dy).astype(int)

    valid = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))

    layers = data_line.loc[valid, "layer"].values

    mask_line.values[layers - 1, iy[valid], ix[valid]] = True  # assuming layers are 1..30

    new_mask = mask_line & mask_overall  # only keep voxels that are also in the overall mask

    return new_mask


def validation(data, pred_grid, cfg):

    t0 = datetime.now()
    print("\nCROSS-VALIDATION SCORING")

    # from config
    inds = np.array(cfg["indicators"])
    ind_cols = cfg["indicator_names"]
    ind_bounds = cfg["indicator_bounds"]
    dir_data = cfg["dir_data"]
    dir_xval = cfg["dir_xval"]

    def sample(df, ds, ind_cols):

        df_sampled = pd.DataFrame(index=df.index)

        x = xr.DataArray(df["x"].values, dims="p")
        y = xr.DataArray(df["y"].values, dims="p")
        layer = xr.DataArray(df["layer"].values, dims="p")

        for var in ind_cols:

            da = ds[var]

            # sample the data at the specified coordinates
            df_sampled[var] = da.sel(layer=layer).sel(x=x, y=y, method="nearest").values

        return df_sampled

    # sample relevant data at sample locations from prediction grid
    if cfg["method"] == "geostat":
        cols = ind_cols + ["laf_ratio"]
    else:
        cols = ind_cols
    pred = sample(data, pred_grid, cols).dropna()

    # true indicator probabilities from data, only keep rows with xval predition
    true = data.copy()
    true = true[true.index.isin(pred.index)]

    # calculate median quantiles and convert to class labels
    true["median"] = _postproc_helper.ind_probs_to_quantiles(true[ind_cols], inds, ind_bounds, (0.5,))
    true["median class"] = _postproc_helper.class_from_quantile(true["median"], inds, ind_bounds)

    pred["median"] = _postproc_helper.ind_probs_to_quantiles(pred[ind_cols], inds, ind_bounds, (0.5,))
    pred["median class"] = _postproc_helper.class_from_quantile(pred["median"], inds, ind_bounds)

    # create plots of performance by median class
    path = dir_xval / "xval - performance by true median class.png"
    visualisation.class_performance(true=true, pred=pred, group_col="median class", group_source="true", path=path)

    # calculate RPS (ranked probability score) for each cell, and put in dataframe
    print("...ranked probability score (RPS)")
    rps = _scoring.rps_from_cdf(pred[ind_cols], true[ind_cols], normalize=True)
    true = true.assign(RPS=rps)

    # summarize RPS overall and per class, and save to csv
    path = dir_xval / "xval - ranked probability score - by true median class.csv"
    _scoring.rps_summary(rps, true["median class"], path=path)

    # summarize RPS overall and per layer, and save to csv
    path = dir_xval / "xval - ranked probability score - by layer.csv"
    _scoring.rps_summary(rps, true["layer"], path=path)

    # boxplot of overall RPS
    path = dir_xval / "xval - RPS.png"
    visualisation.boxplot(true, y="RPS", path=path, showfliers=False)

    # boxplot of RPS by true median class
    path = dir_xval / "xval - RPS by true median class.png"
    visualisation.boxplot(true, x="median class", y="RPS", path=path, showfliers=False)

    # boxplot of RPS by layer
    path = dir_xval / "xval - RPS by layer.png"
    visualisation.boxplot(true, x="layer", y="RPS", path=path, showfliers=False)

    # laf ratio class plots
    if "laf_ratio" in pred.columns:
        bins = np.arange(0, 1.1, 0.1)
        labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]]

        pred["laf_ratio"] = pd.cut(pred["laf_ratio"], bins=bins, labels=labels, include_lowest=True)
        true["LAF ratio"] = pred["laf_ratio"]

        # plot of performance by LAF ratio class
        path = dir_xval / "xval - performance by LAF ratio.png"
        visualisation.class_performance(true=true, pred=pred, group_col="laf_ratio", group_source="pred", path=path)

        # summarize RPS overall and per layer, and save to csv
        path = dir_xval / "xval - ranked probability score - by LAF ratio.csv"
        _scoring.rps_summary(rps, true["LAF ratio"], path=path)

        # boxplot of RPS by LAF ratio
        path = dir_xval / "xval - RPS vs LAF ratio.png"
        visualisation.boxplot(true, x="LAF ratio", y="RPS", path=path, showfliers=False)

    # confusion matrix for median class
    print("...confusion matrix")
    y_true = true["median class"]
    y_pred = pred["median class"]
    labels = true["median class"].cat.categories

    cms = [
        (confusion_matrix(y_true, y_pred, labels=labels), "counts", "d"),
        (confusion_matrix(y_true, y_pred, labels=labels, normalize="true"), "row-normalized", ".2f"),
    ]

    for cm, title, fmt in cms:
        path = dir_xval / f"xval - confusion matrix - {title.replace(' ', '_')}.png"
        visualisation.plot_confusion_matrix(cm, labels, title, fmt, path)

    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)

    # classification report for median class
    print("...classification report")
    cr = classification_report(y_true, y_pred, labels=labels, target_names=[str(c) for c in labels], zero_division=0)
    path = dir_xval / "xval - median class - classification report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(cr)

    # save results
    path = dir_data / "xval - rps.parquet"
    write.table(rps, path)

    print(f"...({(datetime.now() - t0).total_seconds():.2f}s)")
