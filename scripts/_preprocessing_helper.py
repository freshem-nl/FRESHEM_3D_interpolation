import numpy as np
from scipy.spatial import cKDTree
import pandas as pd


def min_max_to_cell_centers(vmin, vmax, step):

    # k values for cell edges are from floor(vmin/step) to ceil(vmax/step)
    kmin = np.floor(vmin / step)
    kmax = np.ceil(vmax / step) - 1

    # k values for cell centers are from kmin to kmax (inclusive)
    k = np.arange(kmin, kmax + 1)

    # cell centers are at k*step + step/2
    centers = k * step + step / 2

    return centers


def idw_to_grid(xp, yp, vp, xg, yg, k=12, p=2.0, eps=1e-12):
    """
    Interpoleer punten (xp,yp,vp) naar grid (xg,yg) met kNN-IDW.
    xg, yg zijn 1D arrays (ds['x'], ds['y']).
    """
    tree = cKDTree(np.c_[xp, yp])

    Xg, Yg = np.meshgrid(xg, yg)  # (ny, nx)
    q = np.c_[Xg.ravel(), Yg.ravel()]  # (ncell, 2)

    dist, idx = tree.query(q, k=k, workers=-1)
    dist = np.maximum(dist, eps)

    w = 1.0 / (dist**p)
    vv = vp[idx]

    out = (w * vv).sum(axis=1) / w.sum(axis=1)
    return out.reshape(Xg.shape)


def calc_oblique_geographic_coordinates(x_crds, y_crds, theta_deg):

    # calculate oblique geographic coordinate (OGC) based on x and y coordinates and angle theta
    # following the method described in:
    # Møller et al (2020) "Oblique geographic coordinates as  covariates for digital soil mapping"
    # https://doi.org/10.5194/soil-6-269-2020

    # theta to radians
    theta = np.radians(theta_deg)

    # see Møller for formulas
    c = np.hypot(x_crds, y_crds)
    angle_a2 = theta - np.arctan2(y_crds, x_crds)
    OGC = c * np.cos(angle_a2)

    # Indien x_crds een Series is, index overnemen
    if isinstance(x_crds, pd.Series):
        OGC = pd.Series(OGC, index=x_crds.index)

    return OGC
