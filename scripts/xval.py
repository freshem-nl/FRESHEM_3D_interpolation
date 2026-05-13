import numpy as np
import xarray as xr
import pandas as pd

from scripts import _read_and_write

def xval_lines(cfg):

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