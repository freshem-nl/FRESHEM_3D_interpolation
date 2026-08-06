import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Ellipse
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.auto import tqdm


def plot_df(df, name, cfg):

    t0 = datetime.now()

    # drop geometry column in case it's a geodataframe
    df = df.drop(columns="geometry", errors="ignore")

    # from config
    dir_plot = cfg["dir_plot"]

    os.makedirs(dir_plot, exist_ok=True)

    txt = f"Plotting histograms for dataframe {df.columns.to_list()}..."
    for var in tqdm(df.columns, desc=txt, unit="var"):

        path = dir_plot / f"{name} - {var}.png"

        histogram(df[var], path, cfg)

        # per-layer hists for the main variable and its stddev column
        if var in (cfg["variable_name"], f"{cfg['variable_name']}_std") and "layer" in df.columns:
            for layer in df["layer"].unique():
                path = dir_plot / f"{name} - {var} - layer {layer}.png"
                histogram(df[df["layer"] == layer][var], path, cfg)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")


def plot_ds(ds, name, cfg, do_not_plot=None):
    t0 = datetime.now()

    # from config
    dir_plot = cfg["dir_plot"]
    indicator_names = cfg["indicator_names"]
    quantile_names = cfg["quantile_names"]
    plotting_layers = cfg["plotting_layers"]


    def get_norm(da, var, indicator_names, quantile_names):
        if var in quantile_names:
            vals = da.values
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                return None
            vmin, vmax = np.quantile(vals, [0.02, 0.98])
            return LogNorm(vmin=vmin, vmax=vmax)

        elif var in indicator_names:
            return Normalize(vmin=0, vmax=1)

        return None

    # remove variables that should not be plotted
    if do_not_plot is not None:
        ds = ds.drop_vars(do_not_plot, errors="ignore")

    os.makedirs(dir_plot, exist_ok=True)

    txt = f"plotting dataset {list(ds.data_vars)} for layers {cfg['plotting_layers']}"
   
    for var in tqdm(ds.data_vars, desc=txt, unit="var"):

        # for var in ds.data_vars:
        da = ds[var]

        # histogram
        series = da.stack(cell=da.dims).dropna("cell").to_series().rename(var)
        path = dir_plot / f"{name} - {var}.png"
        histogram(series, path, cfg)

        slices = (
            [(layer, da.sel(layer=layer)) for layer in plotting_layers if layer in da.layer.values]
            if "layer" in da.dims
            else [(None, da)]
        )

        for layer, da_slice in slices:
            norm = get_norm(da_slice, var, indicator_names, quantile_names)
            da_slice.plot(norm=norm)

            title = f"layer {layer} - {var}" if layer is not None else var
            filename = f"{name} - layer {layer} - {var}.png" if layer is not None else f"{name} - {var}.png"

            plt.title(title)

            path = dir_plot / "grid" / filename
            os.makedirs(path.parent, exist_ok=True)

            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")


def boxplot(df, x=None, y=None, path=None, hue=None, showfliers=True):

    df = df.copy().reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue, showfliers=showfliers)

    title = f"{y} by {x}" if x is not None else y
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(path.parent, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def histogram(series, path, cfg):

    # from config
    n_bins = cfg["histogram_bins"]
    sample_size = cfg["histogram_sample_size"]
    variable_name = cfg["variable_name"]
    indicator_bounds = cfg["indicator_bounds"]
    quantile_names = cfg["quantile_names"]

    name = series.name

    n_data = series.notna().sum()
    df_plot = series.dropna().sample(n=min(sample_size, n_data), random_state=42)

    # If values are (almost) discrete, don't use more bins than unique values
    nunique = df_plot.nunique(dropna=True)
    n_bins_eff = min(n_bins, int(nunique)) if nunique > 0 else bin

    hist_kws = dict(kde=False, color="C0", edgecolor="black", linewidth=0.5)

    plt.figure()
    # check if plotting  main variable or an indicator (starts with "P("), then use log scale for histogram
    if (name == variable_name) or (name in quantile_names):  # log scale for density and quantiles
        x = df_plot.to_numpy()
        x = x[np.isfinite(x) & (x > 0)]  # log needs positive values
        xmin, xmax = indicator_bounds
        edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins_eff + 1)
        sns.histplot(x, bins=edges, **hist_kws)
        plt.xscale("log")

        ax = plt.gca()
        # Major ticks: 1, 10, 100, ...
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        # Minor ticks: 2..9 within each decade (2,3,4,...,9, 20,30,...
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    else:
        sns.histplot(df_plot, bins=n_bins_eff, **hist_kws)

    # probabilities x-axis from 0 to 1
    if name.startswith("P("):
        plt.xlim(0, 1)

    plt.title(f"{name}, n={n_data:,}")
    plt.xlabel(name)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, labels, title, fmt, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=fmt)
    ax.set_title(f"Confusion matrix ({title})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def anisotropy(
    sl,
    bg,
    *,
    max_angle,
    max_dist,
    min_dist,
    stride=10,
    scale=0.25,
    alpha=0.35,
    lw=0.6,
    edgecolor="k",
    variable_unit="m",
    variable_value=None,
    facecolor="none",
    use_half_long=True,
    use_half_short=False,
    figsize=(14, 10),
    dpi=150,
    path=None,
):
    """
    Plot categorical background (-1/0/1) with anisotropy ellipses on top.

    Angle convention input:
      - 0° = North, positive clockwise (GIS)
    Matplotlib Ellipse angle:
      - degrees CCW from +x (East)
      => angle_mpl = 90 - angle_input
    """

    # Coords
    y = sl["y"].values
    x = sl["x"].values

    # Ensure bg is a plain ndarray
    bg = np.asarray(bg)

    # --- Background colormap for classes -1/0/1 ---
    cmap = ListedColormap(["lightgray", "orange", "dodgerblue"])  # -1, 0, 1
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.pcolormesh(x, y, bg, cmap=cmap, norm=norm, shading="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["no data", f"<={variable_value} {variable_unit}", f">{variable_value} {variable_unit}"])

    # --- Ellipse semi-axes in meters ---
    a = (0.5 * max_dist) if use_half_long else max_dist
    b = (0.5 * min_dist) if use_half_short else min_dist

    # Shrink ellipses for visualization
    a = a * scale
    b = b * scale

    # Only draw where everything is valid
    ok = np.isfinite(max_angle) & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)

    # Sample grid to reduce clutter
    iy = np.arange(0, ok.shape[0], stride)
    ix = np.arange(0, ok.shape[1], stride)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    sample = ok[IY, IX]

    ys = y[IY[sample]]
    xs = x[IX[sample]]
    ang = max_angle[IY[sample], IX[sample]]
    a_s = a[IY[sample], IX[sample]]
    b_s = b[IY[sample], IX[sample]]

    # Convert to Matplotlib angle convention
    ang_mpl = 90.0 - ang

    patches = [
        Ellipse((x0, y0), width=2 * a0, height=2 * b0, angle=float(am))
        for x0, y0, a0, b0, am in zip(xs, ys, a_s, b_s, ang_mpl)
    ]

    ax.add_collection(PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha))

    ax.set_title("Background classes with anisotropy ellipses")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    # plt.show()
    plt.savefig(path)
    plt.close()


def plot_laf(
    ds,
    cfg,
    bbox=None,
    step=10,
    ellipse_scale=1.0,
    cmap="viridis",
    path=None,
    suffix="",
):

    # from config
    plotting_layers = cfg["plotting_layers"]
    indicators = cfg["indicators"]
    indicator_names = cfg["indicator_names"]
    aniso_indicator = cfg["aniso_indicator"]
    dir_plot = cfg["dir_plot"]

    # variable to use for anisotropy estimation
    var = indicator_names[indicators.index(aniso_indicator)]

    p_var = f"{var}_obs"
    angle_var = f"laf_major_angle{suffix}"
    ratio_var = f"laf_ratio{suffix}"

    x = ds.x.values
    y = ds.y.values

    if bbox is None:
        x_mask = np.ones_like(x, dtype=bool)
        y_mask = np.ones_like(y, dtype=bool)
    else:
        xmin, xmax, ymin, ymax = bbox
        x_mask = (x >= xmin) & (x <= xmax)
        y_mask = (y >= ymin) & (y <= ymax)

    xs = x[x_mask]
    ys = y[y_mask]
    xx, yy = np.meshgrid(xs, ys)

    slices = (
        [(layer, ds.sel(layer=layer)) for layer in plotting_layers if layer in ds.layer.values]
        if "layer" in ds.dims
        else [(None, ds)]
    )

    txt = f"plotting laf for layers {cfg['plotting_layers']}"
    for layer, da in tqdm(slices, desc=txt, unit="layer"):

        p = da[p_var].values[np.ix_(y_mask, x_mask)]
        angle = da[angle_var].values[np.ix_(y_mask, x_mask)]
        ratio = da[ratio_var].values[np.ix_(y_mask, x_mask)]

        # Choose ellipse size from plotting density and an explicit scale factor.
        dx = np.nanmedian(np.abs(np.diff(xs)))
        dy = np.nanmedian(np.abs(np.diff(ys)))
        ellipse_major = ellipse_scale * step * min(dx, dy)

        # Keep visual ellipse ratios in a valid range.
        ratio = np.clip(ratio, 0.0, 1.0)

        rows, cols = np.indices(angle.shape)
        valid = np.isfinite(angle) & np.isfinite(ratio) & (rows % step == 0) & (cols % step == 0)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.pcolormesh(xs, ys, p, shading="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(p_var)

        for xi, yi, ai, ri in zip(xx[valid], yy[valid], angle[valid], ratio[valid]):
            ell = Ellipse(
                (xi, yi),
                width=ellipse_major,
                height=ellipse_major * ri,
                angle=ai,
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,
                alpha=0.8,
            )
            ax.add_patch(ell)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{p_var} with LAF ellipses | layer {layer}")

        if bbox is not None:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        plt.tight_layout()
        if suffix == "":
            filename = f"anisotropy - layer_{layer}.png"
        else:
            filename = f"anisotropy - layer_{layer} - {suffix.replace('_', '')}.png"
        path = dir_plot / "grid" / filename
        os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()


import pandas as pd


def class_performance(
    true,
    pred,
    group_col,
    true_class_col="median class",
    pred_class_col="median class",
    group_source="pred",
    include_within_1=True,
    title=None,
    path=None,
    figsize=(9, 6),
):
    """
    Plot classification performance per existing category/group.

    Parameters
    ----------
    true, pred : pandas.DataFrame
        Dataframes with true and predicted class labels.
    group_col : str
        Existing categorical/group column, e.g. 'laf_ratio_class' or 'median class'.
    true_class_col, pred_class_col : str
        Columns with true and predicted classes.
    group_source : {'true', 'pred'}
        Dataframe from which group_col is taken.
    include_within_1 : bool
        Also plot fraction of predictions within +/- 1 class.
    title : str, optional
        Plot title.
    path : pathlib.Path or str, optional
        If given, save figure to this path.

    Returns
    -------
    stats : pandas.DataFrame
        Performance statistics per group.
    fig : matplotlib.figure.Figure
        Figure object.
    """

    y_true = true[true_class_col]
    y_pred = pred[pred_class_col]

    if isinstance(y_true.dtype, pd.CategoricalDtype):
        categories = y_true.cat.categories
        y_true_code = y_true.cat.codes
        y_pred_code = y_pred.astype(pd.CategoricalDtype(categories=categories, ordered=True)).cat.codes
    else:
        y_true_code = y_true
        y_pred_code = y_pred

    group = true[group_col] if group_source == "true" else pred[group_col]

    df = pd.DataFrame(
        {
            "group": group,
            "true": y_true_code,
            "pred": y_pred_code,
        }
    ).dropna()

    df["abs_error"] = np.abs(df["true"] - df["pred"])
    df["correct"] = df["true"] == df["pred"]
    df["within_1"] = df["abs_error"] <= 1

    stats = (
        df.groupby("group", observed=True, sort=True)
        .agg(
            n=("correct", "size"),
            mae=("abs_error", "mean"),
            accuracy=("correct", "mean"),
            within_1=("within_1", "mean"),
        )
        .reset_index()
    )

    stats["group_label"] = stats["group"].astype(str)

    metrics = ["mae", "accuracy"]
    if include_within_1:
        metrics.append("within_1")

    stats_long = stats.melt(
        id_vars=["group", "group_label", "n"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    sns.lineplot(
        data=stats_long,
        x="group_label",
        y="value",
        hue="metric",
        marker="o",
        ax=ax1,
    )

    ax1.set_ylabel("Performance")
    ax1.set_xlabel("")
    ax1.grid(axis="y", alpha=0.3)

    if title is None:
        title = f"Classification performance by {group_col}"

    ax1.set_title(title)

    sns.barplot(
        data=stats,
        x="group_label",
        y="n",
        color="lightgrey",
        ax=ax2,
    )

    ax2.set_ylabel("n")
    ax2.set_xlabel(group_col)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if path is not None:
        os.makedirs(path.parent, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        stats.to_csv(path.with_suffix(".csv"), index=False)
    plt.close(fig)


def feature_importance(model, cfg):

    # from config
    dir_plot = cfg["dir_plot"]

    path = dir_plot / "prediction - feature importance.png"

    imp = pd.DataFrame(
        {
            "feature": cfg["features"],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    imp.to_csv(path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.3)))

    sns.barplot(
        data=imp,
        x="importance",
        y="feature",
        color="steelblue",
        ax=ax,
    )

    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig, ax, imp
