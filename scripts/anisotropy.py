from datetime import datetime

import numpy as np

from scripts import _anisotropy_helper


def anisotropy_of_observations(data, pred, cfg):

    # from config
    aniso_indicator = cfg["aniso_indicator"]
    indicators = cfg["indicators"]
    indicator_names = cfg["indicator_names"]
    cellsize_xy = cfg["cellsize_xy"]
    radius = cfg["aniso_pca_radius"]
    nmin = cfg["aniso_pca_nmin"]
    eig_ratio_min = cfg["aniso_pca_eig_ratio_min"]
    sampling_eig_min = cfg["aniso_pca_sampling_eig_min"]
    angle_diff_max = cfg["aniso_pca_angle_diff_max"]
    major_range = cfg["variogram_model_range_xy"]
    minor_range_min = cfg["variogram_minor_range_xy_min"]

    # variable to use for anisotropy estimation
    var = indicator_names[indicators.index(aniso_indicator)]

    # map data coordinates to grid and compute mean indicator values for each grid cell
    g = (
        data.assign(
            x=np.floor(data["x"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2,
            y=np.floor(data["y"] / cellsize_xy) * cellsize_xy + cellsize_xy / 2,
        )
        .groupby(["layer", "y", "x"])[var]
        .mean()
        .to_xarray()
    )

    # averaged data value per grid cell to the prediction grid
    pred[f"{var}_obs"] = g.reindex(
        layer=pred.layer,
        y=pred.y,
        x=pred.x,
    )

    pred = _anisotropy_helper.local_pca_laf(
        pred,
        var=f"{var}_obs",
        radius=radius,
        n_min=nmin,
        eig_ratio_min=eig_ratio_min,
        sampling_eig_ratio_min=sampling_eig_min,
        min_angle_diff=angle_diff_max,
    )

    # compute anisotropy ratio from d_transition and major range
    d_trasition = pred["laf_d_transition_obs"]
    pred["laf_ratio_obs"] = _anisotropy_helper.ratio_from_d_transition(d_trasition, major_range, minor_range_min)

    return pred

def interpolate_to_laf(pred, cfg):
    
    # from config
    major_range = cfg["variogram_model_range_xy"]
    minor_range_min = cfg["variogram_minor_range_xy_min"]

    pred = _anisotropy_helper.fill_laf_grid(
    pred,
    angle_obs="laf_major_angle_obs",
    dist_obs="laf_d_transition_obs",
    k=16,
    power=2.0,
)
    
    # compute anisotropy ratio from d_transition and major range
    d_trasition = pred["laf_d_transition"]
    pred["laf_ratio"] = _anisotropy_helper.ratio_from_d_transition(d_trasition, major_range, minor_range_min)

    return pred