import numpy as np

def ray_offsets(heading_deg, *, cellsize, maxdist=2000.0):

    """
    Compute integer grid offsets (ix, iy) along a ray using a 2D DDA
    (Amanatides & Woo style) traversal from the center of the origin cell.

    The function returns the visited grid cells in the exact order in which
    the ray crosses cell boundaries, excluding the origin cell (0,0).

    Angle convention:
      - heading_deg = 0° points to North
      - angles increase clockwise (90° = East, 180° = South, 270° = West)

    Axis convention:
      - ix is +East (increasing x)
      - iy is +North (increasing y; matches your dataset where y increases northward)

    Returns
    -------
    ix : (N,) int32
        X-offsets in grid cells from the origin.
    iy : (N,) int32
        Y-offsets in grid cells from the origin.
    dist : (N,) float32
        Euclidean distance in meters from the origin cell center to the
        center of each visited offset cell.
    """

    # Normalize heading to [0, 360)
    heading_deg %= 360
    h = np.deg2rad(heading_deg)

    # Unit direction of the ray in grid coordinates:
    # dx corresponds to East component, dy to North component.
    dx = np.sin(h)   # East component (0° -> 0, 90° -> 1)
    dy = np.cos(h)   # North component (0° -> 1, 180° -> -1)


    # Step direction per axis: -1, 0, or +1 depending on ray direction.
    # (If dx/dy is exactly 0, the ray never crosses boundaries in that axis.)
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)


    # Inverse absolute direction components (in "cell units"):
    # used to increment parametric distances to the next grid boundary.
    # If dx/dy is 0, set to inf so that axis is never chosen.
    inv_dx = np.inf if dx == 0 else 1.0 / abs(dx)
    inv_dy = np.inf if dy == 0 else 1.0 / abs(dy)


    # Starting at the origin cell CENTER:
    # distance (in parametric t, measured in cell-width units) to the first
    # vertical/horizontal boundary is half a cell along that axis.
    tMaxX = 0.5 * inv_dx
    tMaxY = 0.5 * inv_dy

    # How far we must advance in parametric t to cross one additional grid line
    # in x or y direction.
    tDeltaX = inv_dx
    tDeltaY = inv_dy

    # Current integer offsets from the origin cell.
    ix = 0
    iy = 0

    # Accumulate offsets and corresponding metric distances.
    ix_list, iy_list, dist_list = [], [], []

    # Safe upper bound on number of boundary crossings.
    # (Overshoots a bit to stay robust for shallow angles and diagonal paths.)
    max_steps = int(np.ceil(maxdist / cellsize)) * 4

    for _ in range(max_steps):
        # Advance to the next cell by crossing whichever boundary is reached first.
        if tMaxX < tMaxY:
            ix += step_x
            tMaxX += tDeltaX
        else:
            iy += step_y
            tMaxY += tDeltaY

        # Never include the origin cell itself.
        if ix == 0 and iy == 0:
            continue

        # Convert the integer offset to a metric distance from the origin center
        d = np.sqrt((ix * cellsize)**2 + (iy * cellsize)**2)

        # Stop once we exceed the requested maximum distance.
        if d > maxdist + 1e-9:
            break

        ix_list.append(ix)
        iy_list.append(iy)
        dist_list.append(d)

    # Return compact, typed arrays for fast downstream use.
    return (np.asarray(ix_list, np.int32),
            np.asarray(iy_list, np.int32),
            np.asarray(dist_list, np.float32))

def add_unique_lines(l1, l2, l3, idx, lid):
    """
    Append (potentially) new line IDs to per-start-cell containers (l1, l2, l3).

    Each start cell (identified by positions in idx) tracks up to THREE unique line IDs.
    The arrays l1/l2/l3 store these IDs; an empty slot is indicated by -1.

    Parameters
    ----------
    l1, l2, l3 : 1D int arrays
        Per-start-cell storage for up to three unique line IDs (initialized with -1).
    idx : 1D int array
        Indices of the start cells we want to update (refers to positions in l1/l2/l3).
    lid : 1D int array
        Candidate line IDs for those start cells (same length as idx).
    """

    # Keep only ids that are not already present
    new = (lid != l1[idx]) & (lid != l2[idx]) & (lid != l3[idx])
    if not np.any(new):
        return
    
    # Filter to only the truly new (idx, lid) pairs
    idx = idx[new]
    lid = lid[new]


    # Fill the first available slot in order: l1 -> l2 -> l3
    # Note: We keep shrinking (idx, lid) to those still not inserted.
    m = (l1[idx] == -1)
    if np.any(m):
        l1[idx[m]] = lid[m]
    idx = idx[~m]; lid = lid[~m]
    if idx.size == 0:
        return

    m = (l2[idx] == -1)
    if np.any(m):
        l2[idx[m]] = lid[m]
    idx = idx[~m]; lid = lid[~m]
    if idx.size == 0:
        return

    m = (l3[idx] == -1)
    if np.any(m):
        l3[idx[m]] = lid[m]

def ray_dist_to_aniso0_with_offsets(y0, x0, fresh_mask, data_mask, line2d, ix, iy, dist_off):
    """
    For each start cell (seed) in `aniso_mask`, walk along a precomputed ray (ix, iy)
    and return distances to:
      1) the FIRST non-fresh (SALT) cell encountered (with data), and
      2) the LAST fresh cell encountered (with data) before that first SALT.

    Notes
    -----
    - The traversal uses `fresh_mask` as the class/traversal mask:
        True  -> still inside FRESH area (ray continues)
        False -> boundary reached (ray stops for that start cell)
    - Cells where `data_mask` is False are treated as transparent and ignored:
      they do NOT stop the ray and do NOT count as boundary hits.
    - Additionally, we track up to 3 unique flightline IDs encountered along the ray
      (excluding the start cell's own line). `ok3` is True if 3 unique line IDs were
      seen before the boundary was reached.

    Parameters
    ----------
    aniso_mask : 2D bool array
        Seed cells where distances should be computed (start points).
    fresh_mask : 2D bool array
        Class mask defining "inside fresh". Boundary is first cell where this is False.
    data_mask : 2D bool array
        Valid-data mask; False cells are skipped (transparent).
    line2d : 2D array (float/int)
        Flightline identifier per cell; NaN indicates missing.
    ix, iy : 1D int arrays
        Ray offsets (in cell units) in the correct traversal order (excluding (0,0)).
    dist_off : 1D float array
        Metric distance (meters) for each offset step (same length as ix/iy).

    Returns
    -------
    salt2d : 2D float32 array
        Distance (m) from start cell to the FIRST SALT cell (first fresh_mask=False with data).
    fresh2d : 2D float32 array
        Distance (m) from start cell to the LAST FRESH cell before the first SALT.
        (NaN if no fresh cell was encountered along the ray before the boundary.)
    ok3_2d : 2D bool array
        True if at least 3 unique flightline IDs were seen along the ray before boundary.
    """

    # --- 1) Identify start cells (seeds) ---
    n0 = x0.size

    # Per-start-cell outputs stored as 1D vectors (mapped back to 2D at the end)
    out_salt  = np.full(n0, np.nan, np.float32)   # distance to FIRST SALT (boundary cell)
    out_fresh = np.full(n0, np.nan, np.float32)   # distance to LAST FRESH before boundary
    out_ok3   = np.zeros(n0, dtype=bool)          # True if 3 unique lines seen before boundary

    # Early exit: no seeds in this slice
    if n0 == 0:
        return out_salt, out_fresh, out_ok3


    ny, nx = fresh_mask.shape

    # Track which start cells are still "active" (not yet terminated by boundary/out-of-bounds)
    active = np.ones(n0, bool)

    # Track the last distance where we were still inside FRESH for each start cell
    last_fresh_dist = np.full(n0, np.nan, np.float32)

    # --- 2) Track up to 3 unique line IDs per start cell (excluding the start cell's own line) ---
    l1 = np.full(n0, -1, np.int32)
    l2 = np.full(n0, -1, np.int32)
    l3 = np.full(n0, -1, np.int32)

    # Convert line2d to int IDs with -1 for NaN (faster comparisons / storage)
    line_int = np.full(line2d.shape, -1, np.int32)
    m = np.isfinite(line2d)
    line_int[m] = line2d[m].astype(np.int32)

    # Line id at the seed cell (used to ignore the same line when counting "unique lines")
    start_line = line_int[y0, x0]

    # --- 3) Step along the ray offsets until each start cell terminates ---
    for k in range(ix.size):
        if not active.any():
            break  # all seeds have terminated

        idx = np.flatnonzero(active)  # active seed indices in [0..n0)

        # Compute target positions for the current ray step (vectorized over active seeds)
        xa = x0[idx] + ix[k]
        ya = y0[idx] + iy[k]

        # Drop seeds that would step outside the grid at this k
        # inside = (xa >= 0) & (xa < nx) & (ya >= 0) & (ya < ny)
        # active[idx[~inside]] = False
        # if not inside.any():
        #     continue


        inside = (xa >= 0) & (xa < nx) & (ya >= 0) & (ya < ny)

        # Rays that leave the grid without hitting salt:
        # keep the last fresh distance found so far
        oob_idx = idx[~inside]
        if oob_idx.size:
            out_fresh[oob_idx] = last_fresh_dist[oob_idx]
            out_ok3[oob_idx] = (l3[oob_idx] != -1)
            active[oob_idx] = False

        if not inside.any():
            continue


        xa = xa[inside]
        ya = ya[inside]
        idx = idx[inside]

        # Skip NoData cells (transparent): they neither update line tracking nor trigger boundaries
        ok = data_mask[ya, xa]
        if not ok.any():
            continue

        xa = xa[ok]
        ya = ya[ok]
        idx = idx[ok]

        # True while still inside FRESH; False means boundary (SALT) has been reached
        same_class = fresh_mask[ya, xa]

        # Remember last FRESH distance for those still inside the class at this step
        if same_class.any():
            last_fresh_dist[idx[same_class]] = dist_off[k]

        # Update unique line IDs only while still in FRESH and while we still have free slots
        upd_lines = same_class & (l3[idx] == -1)
        if upd_lines.any():
            idx_u = idx[upd_lines]
            lid_u = line_int[ya[upd_lines], xa[upd_lines]]

            # Only count real line IDs, and ignore the start cell's own line ID
            keep = (lid_u >= 0) & (lid_u != start_line[idx_u])
            if keep.any():
                add_unique_lines(l1, l2, l3, idx_u[keep], lid_u[keep])

        # Boundary hit: first valid-data cell that is NOT fresh -> stop those seeds
        hit = ~same_class
        if hit.any():
            hit_idx = idx[hit]

            # Distance to the boundary cell (FIRST SALT)
            out_salt[hit_idx] = dist_off[k]

            # Distance to the last FRESH cell before boundary (may be NaN if none)
            out_fresh[hit_idx] = last_fresh_dist[hit_idx]

            # ok3 is True if we have filled l3 (three unique lines encountered)
            out_ok3[hit_idx] = (l3[hit_idx] != -1)

            # Deactivate seeds that have reached the boundary
            active[hit_idx] = False
    
    # Rays that never hit a boundary: keep the last fresh distance found
    if active.any():
        out_fresh[active] = last_fresh_dist[active]
        out_ok3[active] = (l3[active] != -1)


    return out_salt, out_fresh, out_ok3