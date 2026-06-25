import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from datetime import datetime


def model_top_bottom(data, pred, cfg):

    t0 = datetime.now()
    print("model top and bottom layers...", end=" ")

    # from config
    doi_col = cfg["doi_name"]

    idw_power = 2
    idw_k_nearest = 32


    # --- initialise top layer ---
    pred["top"] = xr.full_like(
        pred["mask_overall"].expand_dims(layer=pred["layer"]).transpose("layer", "y", "x"),
        np.nan,
        dtype=np.float32,
    )


    for type in ['model_top', 'model_bottom']:

        # select input data
        if type == 'model_top':
            depth_col = 'z_top'
            df = data.loc[data["layer"] == 1, ["x", "y", depth_col]].dropna()
        elif type == 'model_bottom':
            depth_col = 'z_' + doi_col
            df = data[["x", "y", depth_col]].drop_duplicates()

        xy_data = df[["x", "y"]].to_numpy()
        z_data = df[depth_col].to_numpy()

        tree = cKDTree(xy_data)

        # grid coords
        x = pred.coords["x"].values
        y = pred.coords["y"].values
        xx, yy = np.meshgrid(x, y, indexing="xy")

        # get mask
        mask = pred["mask_overall"].values

        # target points (only inside mask)
        xy_target = np.column_stack([xx[mask], yy[mask]])

        # query
        dist, idx = tree.query(xy_target, k=idw_k_nearest)

        # weights
        valid = np.isfinite(dist)
        w = np.zeros_like(dist, dtype=float)
        w[valid] = 1.0 / dist[valid] ** idw_power

        depth_pred = np.full(len(xy_target), np.nan, dtype=np.float32)

        # handle exact hits
        exact = (dist == 0) & valid
        has_exact = exact.any(axis=1)

        if has_exact.any():
            first = exact.argmax(axis=1)
            depth_pred[has_exact] = z_data[idx[has_exact, first[has_exact]]]

        # normal IDW
        todo = ~has_exact & valid.any(axis=1)
        depth_pred[todo] = np.sum(w[todo] * z_data[idx[todo]], axis=1) / np.sum(w[todo], axis=1)

        # create 2D output
        depth_pred_2d = xr.full_like(pred["mask_overall"], np.nan, dtype=np.float32)
        depth_pred_2d.values[pred["mask_overall"].values] = depth_pred


        if type == 'model_top':
            pred["top"].loc[{"layer": 1}] = depth_pred_2d
        elif type == 'model_bottom':
            pred[doi_col] = depth_pred_2d

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return pred

def layers_top_bottom(data, pred, cfg):

    t0 = datetime.now()
    print("model top and bottom layers per layer...", end=" ")

    # from config
    doi_col = cfg["doi_name"]


    # --- check constant thickness per layer ---
    tol = 1e-9
    is_constant = data.groupby("layer")["thickness"].apply(
        lambda s: np.allclose(s, s.iloc[0], atol=tol, rtol=0)
    )
    if not is_constant.all():
        invalid = is_constant[~is_constant].index.tolist()
        raise ValueError(f"Thickness not constant in layers: {invalid}")

    # create dictionary of thickness for each layer
    thickness_by_layer = (
        data.groupby("layer")["thickness"]
        .first()
        .to_dict()
    )

    # initialize bottom layer
    pred["bottom"] = xr.full_like(pred["top"], np.nan, dtype=np.float32)
    for layer, thickness in thickness_by_layer.items():

        if layer != 1:
            pred["top"].loc[{"layer": layer}] = (
                pred["top"].sel(layer=layer - 1) - thickness
            )

        # start met "normale" bottom

        top0 = pred["top"].sel(layer=layer)
        bottom0 = top0 - thickness
        doi = pred[doi_col]

        mask_valid = doi <= top0

        top = top0.where(mask_valid)
        bottom = bottom0.where(mask_valid)

        bottom = xr.where((doi > bottom0) & mask_valid, doi, bottom)


        # write immediately correct values
        pred["top"].loc[{"layer": layer}] = top
        pred["bottom"].loc[{"layer": layer}] = bottom

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return pred

    # # check if thickness is constant within each layer
    # is_constant = data.groupby("layer")["thickness"].apply(
    #     lambda s: np.allclose(s, s.iloc[0], atol=tol, rtol=0, equal_nan=False)
    # )

    # if not is_constant.all():
    #     invalid_layers = is_constant[~is_constant].index.tolist()
    #     raise ValueError(f"Thickness is not constant within layer(s): {invalid_layers}")

    # # create dictionary of thickness for each layer
    # thickness_by_layer = (
    #     data.groupby("layer")["thickness"]
    #     .first()
    #     .to_dict()
    # )

    # # initialize bottom layer
    # pred["bottom"] = xr.full_like(pred["top"], np.nan, dtype=np.float32)

    # # assign top and bottom for each layer
    # for layer, thickness in thickness_by_layer.items():
    #     if not layer == 1:
    #         pred["top"].loc[{"layer": layer}] = pred["top"].sel(layer=layer - 1) - thickness
    #     pred["bottom"].loc[{"layer": layer}] = pred["top"].sel(layer=layer) - thickness

    # return pred