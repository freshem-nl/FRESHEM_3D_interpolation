import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm


def plot_df(df, name, cfg):

    t0 = datetime.now()

    # drop geometry column in case it's a geodataframe
    df = df.drop(columns="geometry", errors="ignore")

    print(f"Plotting histograms for dataframe {df.columns.to_list()}...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]

    os.makedirs(dir_plot, exist_ok=True)
    for var in df.columns:
        path = dir_plot / f"{name} - {var}.png"

        histogram(df[var], path, cfg)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")


def plot_ds(ds, name, cfg):
    t0 = datetime.now()
    print(f"plotting dataset {list(ds.data_vars)} at depths {cfg['plotting_depths']}...", end=" ")

    # from config
    dir_plot = cfg["dir_plot"]
    plotting_depths = cfg["plotting_depths"]

    os.makedirs(dir_plot, exist_ok=True)
    for var in ds.data_vars:

        # histogram
        series = ds[var].stack(cell=("z", "y", "x")).dropna("cell").to_series().rename(var)
        path = dir_plot / f"{name} - {var} histogram.png"
        histogram(series, path, cfg)

        # select target depths (exact match or nearest)
        target_depths = np.array(plotting_depths)
        depths = ds["z"].sel(z=target_depths, method="nearest").values

        for depth in depths:

            # select 2D slice at target depth
            da = ds[var].sel(z=depth)

            # log colorscale for quantiles, linear for indicators
            if var.startswith("Q"):
                vals = da.values
                vals = vals[np.isfinite(vals) & (vals > 0)]
                vmin, vmax = np.quantile(vals, [0.02, 0.98])
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = None

            # plot map
            da.plot(norm=norm)
            plt.title(f"{var} at z={depth}m")

            # save figure
            path = dir_plot / f"{name} - {var} at z={depth}m.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

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
