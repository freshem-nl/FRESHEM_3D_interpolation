from datetime import datetime

import numpy as np
import xarray as xr

from scripts import _anisotropy_helper, visualisation


def main(cfg):

    t0 = datetime.now()
    print("Anisotropy estimation...", end=" ")

    path_gridded_data = cfg["path_preproc_data_gridded"]
    cellsize_xy = cfg["cellsize_xy"]
    cellsize_z = cfg["cellsize_z"]

    indicator = cfg["aniso_indicator"]
    indicator_z_range = cfg["aniso_indicator_z_range"]
    long_axis_min = cfg["aniso_long_axis_min"]
    long_axis_max = cfg["aniso_long_axis_max"]
    short_axis_max = cfg["aniso_short_axis_max"]
    min_long_short_angle_diff = cfg["aniso_long_short_angle_min_diff"]
    plotting_depths = cfg["plotting_depths"]

    dir_plot = cfg["dir_plot"]
    variable_unit = cfg["variable_unit"]

    # read data
    ds = read_and_write.read_dataset(path_gridded_data)

    # Create boolean mask of gridcells containing data
    ds["data_mask"] = ds[f"P({indicator})"].notnull()

    # Create boolean mask of gridcells containing median value greather than anisotropy_indicator
    ds["fresh_mask"] = (ds[f"P({indicator})"] < 0.5).astype(bool)
    # cond = (ds[f"P({anisotropy_indicator})"] < 0.5)

    # Seed mask for anisotropy estimation:
    # True only where ALL cells within +/- anisotropy_indicator_z_range are FRESH.
    n = int(np.ceil(indicator_z_range / abs(cellsize_z)))  # Convert meters to number of vertical cells
    aniso_mask = ds["fresh_mask"].copy()
    for k in range(1, n + 1):
        aniso_mask = (
            aniso_mask & ds["fresh_mask"].shift(z=+k, fill_value=False) & ds["fresh_mask"].shift(z=-k, fill_value=False)
        )

    ds["aniso_mask"] = aniso_mask.astype(bool)

    # Precompute offsets once per heading
    angle_step = 5
    thetas = list(range(0, 181, angle_step))  # 0..180 inclusive
    angles_needed = sorted({t % 360 for t in thetas} | {(t + 180) % 360 for t in thetas})
    dict_offsets = {
        a: _anisotropy_helper.ray_offsets(a, cellsize=cellsize_xy, maxdist=long_axis_max) for a in angles_needed
    }

    # Preallocate outputs (z,y,x)
    nz, ny, nx = ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]

    short_dist = np.full((nz, ny, nx), np.nan, np.float32)
    short_angle = np.full((nz, ny, nx), np.nan, np.float32)
    long_dist = np.full((nz, ny, nx), np.nan, np.float32)
    long_angle = np.full((nz, ny, nx), np.nan, np.float32)

    for iz in range(ds.sizes["z"]):
        sl = ds.isel(z=iz)
        # for iz,z in enumerate([-7.5]):  # just one slice for testing
        #     sl = ds.sel(z=z)

        # print(f"Processing z={z}...")
        # select 2D slice
        aniso_mask2d = sl["aniso_mask"].values
        fresh_mask2d = sl["fresh_mask"].values
        data_mask2d = sl["data_mask"].values
        line2d = sl["LINE_NO"].values

        # Skip slices with no active cells
        if not aniso_mask2d.any():
            continue

        # Initialize outputs (NaN everywhere)
        short_dist2d = np.full(aniso_mask2d.shape, np.nan, np.float32)
        short_angle2d = np.full(aniso_mask2d.shape, np.nan, np.float32)
        long_dist2d = np.full(aniso_mask2d.shape, np.nan, np.float32)  # long-axis length (fwd+bwd)
        long_angle2d = np.full(aniso_mask2d.shape, np.nan, np.float32)  # theta for which max occur

        # Precompute seed indices once per slice
        y0, x0 = np.nonzero(aniso_mask2d)
        if x0.size == 0:
            continue

        for theta in thetas:
            # print(f"Processing z={z}, theta={theta}°...")
            # for theta in [0,45,90]:  # just three angles for testing

            # forward direction
            ix, iy, dist_off = dict_offsets[theta]  # use precomputed offsets
            dist_salt_fwd, dist_fresh_fwd, bool_3_lines_fwd = _anisotropy_helper.ray_dist_to_aniso0_with_offsets(
                y0, x0, fresh_mask2d, data_mask2d, line2d, ix, iy, dist_off
            )

            # backward direction
            theta_opp = (theta + 180) % 360
            ix, iy, dist_off = dict_offsets[theta_opp]  # use precomputed offsets
            dist_salt_bwd, dist_fresh_bwd, bool_3_lines_bwd = _anisotropy_helper.ray_dist_to_aniso0_with_offsets(
                y0, x0, fresh_mask2d, data_mask2d, line2d, ix, iy, dist_off
            )

            # UPDATE SHORT AXIS
            # minimum of forward and backward direction
            smallest_dist_fwd_bwd = np.fmin(dist_salt_fwd, dist_salt_bwd)

            # short-axis validity checks:
            # - distance must be smaller than short_axis_max ()
            cand = np.where(smallest_dist_fwd_bwd <= short_axis_max, smallest_dist_fwd_bwd, np.nan)

            # Update per-cell minimum (NaN-safe)
            cur = short_dist2d[y0, x0]
            upd = np.isfinite(cand) & (np.isnan(cur) | (cand < cur))
            if upd.any():
                short_dist2d[y0[upd], x0[upd]] = cand[upd]
                short_angle2d[y0[upd], x0[upd]] = theta

            # UPDATE LONG AXIS
            # long-axis validity checks:
            # -crossing 3 lines
            # -have a minimum length of long_axis_min
            valid_fwd = bool_3_lines_fwd & np.isfinite(dist_fresh_fwd) & (dist_fresh_fwd >= long_axis_min)
            valid_bwd = bool_3_lines_bwd & np.isfinite(dist_fresh_bwd) & (dist_fresh_bwd >= long_axis_min)
            valid = valid_fwd & valid_bwd

            # Candidate long-axis length only if BOTH halves are valid
            cand = np.where(valid, dist_fresh_fwd + dist_fresh_bwd, np.nan).astype(np.float32)

            # Update per-cell maximum (NaN-safe)
            cur = long_dist2d[y0, x0]
            upd = np.isfinite(cand) & (np.isnan(cur) | (cand > cur))
            if upd.any():
                long_dist2d[y0[upd], x0[upd]] = cand[upd]
                long_angle2d[y0[upd], x0[upd]] = theta

        # final validity check: long and short axis must differ in angle by at least min_long_short_angle_diff (30 degrees)
        ang_diff = np.abs((short_angle2d - long_angle2d + 90) % 180 - 90)  # [0..90]
        valid = ang_diff >= min_long_short_angle_diff

        short_dist2d = np.where(valid, short_dist2d, np.nan).astype(np.float32)
        short_angle2d = np.where(valid, short_angle2d, np.nan).astype(np.float32)
        long_dist2d = np.where(valid, long_dist2d, np.nan).astype(np.float32)
        long_angle2d = np.where(valid, long_angle2d, np.nan).astype(np.float32)

        # update 3D arrays with this slice's results
        short_dist[iz] = short_dist2d
        short_angle[iz] = short_angle2d
        long_dist[iz] = long_dist2d
        long_angle[iz] = long_angle2d

    # Write back into ds
    ds["short_dist"] = (("z", "y", "x"), short_dist)
    ds["short_angle"] = (("z", "y", "x"), short_angle)
    ds["long_dist"] = (("z", "y", "x"), long_dist)
    ds["long_angle"] = (("z", "y", "x"), long_angle)

    read_and_write.write_dataset(ds, cfg["path_data_anisotropy"])

    # VISUALISATION
    # select target depths (exact match or nearest)
    target_depths = np.array(plotting_depths)
    depths = ds["z"].sel(z=target_depths, method="nearest").values

    for z in depths:
        sl = ds.sel(z=z)

        data_mask = sl["data_mask"]  # bool
        fresh_mask = sl["fresh_mask"]

        bg = xr.where(~data_mask, -1, xr.where(fresh_mask, 1, 0)).astype(np.int8)

        # max_angle/max_dist/min_dist are your numpy 2D arrays from the loop
        path = dir_plot / "depth_slices" / f"data - anisotropy at z={z}m.png"
        visualisation.anisotropy(
            sl,
            bg,
            max_angle=sl["long_angle"].values,
            max_dist=sl["long_dist"].values,
            min_dist=sl["short_dist"].values,
            stride=2,  # tune: 8..20 typically
            scale=0.2,
            alpha=0.5,
            lw=0.6,
            edgecolor="black",
            variable_value=indicator,
            variable_unit=variable_unit,
            use_half_long=True,  # often correct if max_dist is fwd+bwd total
            use_half_short=False,  # min_dist is single-direction -> treat as radius
            path=path,
        )

    print(f"done ({datetime.now() - t0})")

    return ds
