import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

def ratio_from_d_transition(d_trasition, major_range, minor_range_min):

    minor_obs = d_trasition.clip(min=minor_range_min, max=major_range)
    ratio = minor_obs / major_range

    return ratio


def axial_angle_diff(a, b):
    """Return the smallest difference between two axial angles in degrees.

    The angles represent axes, not vectors. Therefore, 0 and 180 degrees are
    equivalent, and the maximum meaningful difference is 90 degrees.
    """
    diff = np.abs(a - b) % 180
    return np.minimum(diff, 180 - diff)


def pca_angle_eig_ratio(xy, w=None):
    """Compute PCA major-axis angle and eigenvalue ratio for 2D coordinates.

    Parameters
    ----------
    xy : ndarray of shape (n_points, 2)
        Coordinates used for PCA.
    w : ndarray or None
        Optional weights for weighted PCA.

    Returns
    -------
    angle : float
        Major-axis angle in degrees, measured counter-clockwise from +x.
        The angle is axial, so it is returned modulo 180 degrees.
    eig_ratio : float
        Ratio between the largest and smallest eigenvalue.
        Higher values indicate a more elongated point cloud.
    """
    # Compute the weighted or unweighted covariance matrix of the coordinates.
    # This 2x2 covariance matrix describes the spatial spread of the points.
    cov = np.cov(xy.T, aweights=w, bias=True)

    # Compute eigenvalues and eigenvectors of the covariance matrix.
    # The eigenvector with the largest eigenvalue is the major PCA axis.
    eigval, eigvec = np.linalg.eigh(cov)

    # Sort eigenvalues/eigenvectors from largest to smallest eigenvalue.
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    # If the minor eigenvalue is zero or negative, the ratio is not reliable.
    if eigval[1] <= 0:
        return np.nan, np.nan

    # Convert the major eigenvector to an angle.
    # arctan2 gives an angle counter-clockwise from the positive x-axis.
    # Modulo 180 is used because anisotropy direction is axial.
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0])) % 180

    # Compute elongation as the ratio between major and minor eigenvalues.
    eig_ratio = eigval[0] / eigval[1]

    return angle, eig_ratio


def local_pca_laf(
    ds,
    var="P(rho≤5)",
    radius=1000.0,
    fresh_min=0.5,
    salt_max=0.5,
    n_min=10,
    eig_ratio_min=2.0,
    sampling_eig_ratio_min=2.0,
    min_angle_diff=15.0,
):
    """Estimate observed LAF angles from gridded fresh-water cells.

    The method estimates sparse local anisotropy field observations using PCA
    on nearby fresh-water cells. It also compares the resulting fresh-water
    direction with the local data-support direction to suppress directions that
    are likely caused by AEM flightline geometry.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimensions layer, y, x.
    var : str
        Name of the gridded salt-probability variable.
        Fresh probability is computed as 1 - ds[var].
    radius : float
        Search radius for local PCA, in coordinate units.
    fresh_min : float
        Minimum fresh probability used to classify a cell as fresh.
    salt_max : float
        Maximum fresh probability used to classify a cell as salt.
    n_min : int
        Minimum number of fresh cells required for local PCA.
    eig_ratio_min : float
        Minimum PCA eigenvalue ratio required for a valid fresh-water direction.
    sampling_eig_ratio_min : float
        Minimum eigenvalue ratio of the local data geometry before applying the
        sampling-direction rejection filter.
    min_angle_diff : float
        Minimum required axial angle difference between the fresh-water PCA
        direction and local data-support PCA direction.

    Returns
    -------
    ds : xarray.Dataset
        Input dataset with observed LAF variables added.
    """
    # Extract grid coordinates.
    x = ds.x.values
    y = ds.y.values

    # Create 2D coordinate arrays with the same shape as each layer.
    xx, yy = np.meshgrid(x, y)

    # Lists for collecting layer-wise outputs.
    major_angles, d_transitions = [], []

    # Process each layer independently.
    layers = ds.layer.values
    txt = f"determine anisotropy from observations for {len(layers)} layers"
    for layer in tqdm(layers, desc=txt, unit="layer"):
   
        # Read salt probability for this layer.
        p_salt = ds[var].sel(layer=layer).values

        # Convert salt probability to fresh probability.
        p_fresh = 1.0 - p_salt

        # Define valid cells: cells where data are present.
        valid = np.isfinite(p_fresh)

        # Define fresh cells used for the geological PCA.
        fresh = valid & (p_fresh >= fresh_min)

        # Define salt cells used to estimate distance to nearest salt.
        salt = valid & (p_fresh <= salt_max)

        # Convert fresh, salt, and valid grid cells to coordinate arrays.
        fresh_xy = np.c_[xx[fresh], yy[fresh]]
        salt_xy = np.c_[xx[salt], yy[salt]]
        valid_xy = np.c_[xx[valid], yy[valid]]

        # Use fresh probability as weight in the fresh-water PCA.
        # Cells with higher fresh probability contribute more strongly.
        fresh_w = p_fresh[fresh]

        # Build spatial search trees for fast neighbourhood queries.
        tree_fresh = cKDTree(fresh_xy)
        tree_salt = cKDTree(salt_xy)
        tree_valid = cKDTree(valid_xy)

        # The target cells are only the fresh cells.
        # LAF observations are estimated only where fresh water is present.
        targets = fresh_xy

        # Store positions of fresh cells in the flattened full grid.
        target_idx = np.flatnonzero(fresh.ravel())

        # Initialize flattened output arrays for this layer.
        major_angle_out = np.full(p_fresh.size, np.nan, dtype=np.float32)
        d_transition_out = np.full(p_fresh.size, np.nan, dtype=np.float32)

        # Compute distance from each fresh target cell to the nearest salt cell.
        # This is used as a proxy for proximity to the fresh-salt transition.
        d_transition, _ = tree_salt.query(targets)

        # Only evaluate fresh target cells within one radius of salt.
        # Fresh cells far from salt are not considered relevant for LAF observations.
        candidates = np.flatnonzero(d_transition <= radius)

        # Find neighbouring fresh cells within the PCA radius for each candidate.
        neighbours = tree_fresh.query_ball_point(targets[candidates], r=radius)

        # Loop over candidate target cells and their local fresh neighbourhoods.
        for pos, idx in zip(candidates, neighbours):
            # Skip locations with too few fresh cells for a stable PCA.
            if len(idx) < n_min:
                continue

            # Compute weighted PCA on local fresh-water cells.
            # This estimates the local major axis of the fresh-water body.
            major_angle, fresh_eig_ratio = pca_angle_eig_ratio(
                fresh_xy[idx],
                w=fresh_w[idx],
            )

            # Skip invalid PCA results.
            if not np.isfinite(fresh_eig_ratio):
                continue

            # Skip weakly elongated point clouds.
            # A low eigenvalue ratio means no clear anisotropy direction.
            if fresh_eig_ratio < eig_ratio_min:
                continue

            # Compute local PCA of all valid data cells only after the
            # fresh-water PCA passed the main filters. This keeps the script faster.
            valid_idx = tree_valid.query_ball_point(targets[pos], r=radius)
            data_angle, data_eig_ratio = pca_angle_eig_ratio(valid_xy[valid_idx])

            # If the local data geometry is strongly elongated, compare its
            # direction with the fresh-water PCA direction.
            if np.isfinite(data_eig_ratio) and data_eig_ratio >= sampling_eig_ratio_min:
                angle_diff = axial_angle_diff(major_angle, data_angle)

                # Reject if the fresh-water direction is too similar to the data
                # geometry direction. This suppresses flightline-driven artefacts.
                if angle_diff <= min_angle_diff:
                    continue

            # Convert candidate position back to flattened full-grid index.
            i = target_idx[pos]

            # Store accepted observed LAF result.
            major_angle_out[i] = major_angle
            d_transition_out[i] = d_transition[pos]

        # Reshape flattened arrays back to 2D layer grids.
        shape = p_fresh.shape
        major_angles.append(major_angle_out.reshape(shape))
        d_transitions.append(d_transition_out.reshape(shape))

    # Add observed/sparse LAF results to the dataset.
    ds["laf_major_angle_obs"] = (("layer", "y", "x"), np.stack(major_angles))
    ds["laf_d_transition_obs"] = (("layer", "y", "x"), np.stack(d_transitions))

    # Add metadata for clarity.
    ds["laf_major_angle_obs"].attrs["units"] = "degrees"
    ds["laf_major_angle_obs"].attrs[
        "description"
    ] = "Observed-grid LAF major-axis angle from PCA on fresh-water cells"

    ds["laf_d_transition_obs"].attrs["units"] = "m"
    ds["laf_d_transition_obs"].attrs[
        "description"
    ] = "Observed-grid distance to nearest salt-water cell"

    return ds

import numpy as np
from scipy.spatial import cKDTree


def idw(xy, values, targets, k=16, power=2.0):
    """Interpolate values at target locations using inverse distance weighting."""
    tree = cKDTree(xy)

    kk = min(k, len(xy))
    d, idx = tree.query(targets, k=kk)

    if kk == 1:
        d = d[:, None]
        idx = idx[:, None]

    w = 1.0 / np.maximum(d, 1e-12) ** power
    w = w / w.sum(axis=1, keepdims=True)

    return np.sum(w * values[idx], axis=1), d[:, 0]


def fill_laf_grid(
    ds,
    angle_obs="laf_major_angle_obs",
    dist_obs="laf_d_transition_obs",
    k=16,
    power=2.0,
):
    x = ds.x.values
    y = ds.y.values
    xx, yy = np.meshgrid(x, y)
    targets = np.c_[xx.ravel(), yy.ravel()]

    filled_angles = []
    d_transitions = []

    layers = ds.layer.values
    txt = f"interpolate observation anisotropy to locally varying anisotropy fields for {len(layers)} layers"
    for layer in tqdm(layers, desc=txt, unit="layer"):

        major_angle_obs = ds[angle_obs].sel(layer=layer).values
        d_transition_obs = ds[dist_obs].sel(layer=layer).values

        # Use only cells where the local anisotropy analysis produced a result.
        anchor = np.isfinite(major_angle_obs) & np.isfinite(d_transition_obs)

        major_angle_out = np.full(major_angle_obs.size, np.nan, dtype=np.float32)
        d_transition_out = np.full(major_angle_obs.size, np.nan, dtype=np.float32)

        if anchor.any():
            anchor_xy = np.c_[xx[anchor], yy[anchor]]

            # Convert axial angles to doubled-angle unit vectors.
            # This is needed because 0 and 180 degrees describe the same axis.
            theta = np.deg2rad(major_angle_obs[anchor])
            u = np.cos(2 * theta)
            v = np.sin(2 * theta)

            # Use distance to nearest transition as local transition distance at anchor cells.
            d_transition_anchor = d_transition_obs[anchor]

            # IDW interpolate the axial direction components.
            u_idw, d_anchor = idw(anchor_xy, u, targets, k=k, power=power)
            v_idw, _ = idw(anchor_xy, v, targets, k=k, power=power)

            # Convert interpolated doubled-angle vectors back to axial angles.
            theta_idw = 0.5 * np.arctan2(v_idw, u_idw)
            major_angle_out[:] = (np.rad2deg(theta_idw) % 180).astype(np.float32)

            # IDW interpolate transition distance from LAV anchor cells.
            d_transition_idw, _ = idw(anchor_xy, d_transition_anchor, targets, k=k, power=power)

            # Increase transition distance linearly with distance from nearest LAV anchor.
            d_transition_out[:] = (d_transition_idw + d_anchor).astype(np.float32)

        filled_angles.append(major_angle_out.reshape(major_angle_obs.shape))
        d_transitions.append(d_transition_out.reshape(major_angle_obs.shape))

    ds["laf_major_angle"] = (("layer", "y", "x"), np.stack(filled_angles))
    ds["laf_d_transition"] = (("layer", "y", "x"), np.stack(d_transitions))

    ds["laf_major_angle"].attrs["units"] = "degrees"
    ds["laf_major_angle"].attrs["description"] = "Interpolated local anisotropy major-axis direction"

    ds["laf_d_transition"].attrs["units"] = "m"
    ds["laf_d_transition"].attrs["description"] = "Interpolated distance to resistivity transition"

    return ds