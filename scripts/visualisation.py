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


def plot_df(df, name, cfg):

    t0 = datetime.now()

    # drop geometry column in case it's a geodataframe
    df = df.drop(columns="geometry", errors="ignore")

    print(f"Plotting histograms for dataframe {df.columns.to_list()}...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]

    # replace "rho" with "ρ" in column names for better plot labels
    df.columns = [col.replace("rho", "ρ") for col in df.columns]

    os.makedirs(dir_plot, exist_ok=True)
    for var in df.columns:
        path = dir_plot / f"{name} - {var}.png"

        histogram(df[var], path, cfg)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")


def plot_ds(ds, name, cfg):
    t0 = datetime.now()
    print(f"plotting dataset {list(ds.data_vars)} at depths {cfg['plotting_depths']}...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]
    plotting_depths = cfg["plotting_depths"]
    indicator_names = cfg["indicator_names"]
    quantile_names = cfg["quantile_names"]

    # select target depths (exact match or nearest)
    depths = ds["z"].sel(z=np.array(plotting_depths), method="nearest").values

    os.makedirs(dir_plot, exist_ok=True)

    #replace rho with "ρ" for all data vars in ds
    ds = ds.rename({var: var.replace("rho", "ρ") for var in ds.data_vars})

    for var in ds.data_vars:

        # histogram
        da = ds[var]
        series = da.stack(cell=da.dims).dropna("cell").to_series().rename(var)
        path = dir_plot / f"{name} - {var} histogram.png"
        histogram(series, path, cfg)

        # If the variable has a Z dimension: make one plot per target depth
        if "z" in da.dims:
            plot_items = [(depth, da.sel(z=depth)) for depth in depths]
        # If the variable has no Z dimension: still make one 2D plot
        else:
            plot_items = [(None, da)]

        for depth, da_plot in plot_items:

            cmap = None
            # log colorscale for quantiles, linear for indicators
            if var in quantile_names:
                lo, hi = cfg["indicator_bounds"]
                norm = LogNorm(vmin=lo, vmax=hi)
                if cfg["variable_name"].lower() == "rho":
                    cmap = "RdYlBu_r"
            elif var in indicator_names:
                norm = Normalize(vmin=0, vmax=1)
                if cfg["variable_name"].lower() == "rho":
                    cmap = "Blues"
            else:
                norm = None

            # plot map
            if cmap is not None:
                da_plot.plot(norm=norm, cmap=cmap)
            else:
                da_plot.plot(norm=norm)

            # set title and path based on whether depth is available
            if depth is not None:
                plt.title(f"{var} at z={depth}m")
                path = dir_plot / "depth_slices" / f"{name} - {var} at z={depth}m.png"
            else:
                plt.title(var)
                path = dir_plot / "depth_slices" / f"{name} - {var}.png"

            # save figure
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

    name = series.name

    n_data = series.notna().sum()
    df_plot = series.dropna().sample(n=min(sample_size, n_data), random_state=42)

    # If values are (almost) discrete, don't use more bins than unique values
    nunique = df_plot.nunique(dropna=True)
    n_bins_eff = min(n_bins, int(nunique)) if nunique > 0 else bin

    hist_kws = dict(kde=False, color="C0", edgecolor="black", linewidth=0.5)

    plt.figure()
    # check if plotting  main variable or an indicator (starts with "P("), then use log scale for histogram
    if (name == variable_name) or (name.startswith("Q")):  # log scale for density and quantiles
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
