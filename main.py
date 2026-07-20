import os
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from scripts import (
    anisotropy,
    config_loader,
    depth,
    geostat,
    ml,
    postproc,
    preproc_data,
    preproc_grid,
    preproc_ml,
    read,
    visualisation,
    write,
    xval,
)


def main(cfg):
    t = datetime.now()

    # copy config file
    path = cfg["dir_output"] / "config.yaml"
    write.txt_to_yaml(cfg, path)

    method = cfg["method"]

    def preprocessing_data():
        data = read.skytem_xyz(cfg)
        data = preproc_data.restructure(data, cfg)
        data = preproc_data.quantiles_and_indicator_probs(data, cfg)
        if method == "ml":
            data = preproc_ml.OGC(data, cfg)
        ###TEMP
        # cond = (data["x"] > 39700) & (data["x"] < 43900) & (data["y"] > 391400) & (data["y"] < 397600)
        # data = data.loc[cond]
        ### END TEMP
        write.table(data, cfg["path_preproc_data"])
        write.table(data, cfg["path_preproc_data"].with_suffix(".csv"))
        visualisation.plot_df(data, "preproc - data", cfg)

    def preprocessing_prediction_grid():
        data = read.table(cfg["path_preproc_data"])
        pred = preproc_grid.initiate_grid(data, cfg)
        pred = preproc_grid.mask_overall(data, pred, cfg)
        pred = depth.model_top_bottom(data, pred, cfg)
        pred = depth.layers_top_bottom(data, pred, cfg)
        pred = preproc_grid.mask_per_layer(pred)
        if method == "ml":
            pred = preproc_ml.OGC(pred, cfg)
        if method == "geostat":
            pred = anisotropy.anisotropy_of_observations(data, pred, cfg)
            visualisation.plot_laf(pred, cfg, suffix="_obs", step=1, ellipse_scale=5.0)
            pred = anisotropy.interpolate_to_laf(pred, cfg)
            visualisation.plot_laf(pred, cfg)
        write.dataset(pred, cfg["path_preproc_prediction_grid"])
        visualisation.plot_ds(pred, "preproc - prediction grid", cfg)

    def interpolation():
        data = read.table(cfg["path_preproc_data"])
        pred = read.dataset(cfg["path_preproc_prediction_grid"])
        if method == "ml":
            model, output_names = ml.rf_train(data, cfg)
            pred = ml.rf_predict(model, output_names, pred, cfg)
        elif method == "geostat":
            pred = geostat.kriging(data, pred, cfg)
        pred = postproc.ensure_monotonicity(pred, cfg)
        write.dataset(pred, cfg["path_prediction"])
        write.ds_to_tiff(pred, cfg["dir_rasters"], "pred")
        visualisation.plot_ds(pred, "prediction", cfg)

    def postprocessing():
        pred = read.dataset(cfg["path_prediction"])
        pred_quant = postproc.ds_ind_probs_to_quantiles(pred, cfg)
        write.dataset(pred_quant, cfg["path_postproc"])
        visualisation.plot_ds(pred_quant, "postproc", cfg)
        write.ds_to_tiff(pred_quant, cfg["dir_rasters"], "postproc")

    def interpolation_xval():
        data = read.table(cfg["path_preproc_data"])
        pred = read.dataset(cfg["path_preproc_prediction_grid"])

        lines = xval.xval_lines(data, cfg)
        model_mask = pred["mask"].copy()  # overall mask for reuse
        txt = "crossvalidation: train without data of line, predict on line"
        for line in tqdm(lines, desc=txt, unit="line", leave=True, position=0):
            data_fold = data[data["line_no"] != line].copy()  # exclude data from line
            pred["mask"] = xval.mask_line(data, model_mask, line)  # only voxels in model_mask and line
            if method == "ml":
                model = ml.rf_train(data_fold, cfg, verbose=False)
                pred = ml.rf_predict(model, pred, cfg, verbose=False)
            elif method == "geostat":
                pred = geostat.kriging(data_fold, pred, cfg, verbose=False)
        pred = postproc.ensure_monotonicity(pred, cfg)
        write.dataset(pred, cfg["path_prediction_xval"])
        write.ds_to_tiff(pred, cfg["dir_rasters"], "xval")
        visualisation.plot_ds(pred, "xval", cfg)

    def xval_scoring():
        data = read.table(cfg["path_preproc_data"])
        pred_xval = read.dataset(cfg["path_prediction_xval"])
        xval.validation(data, pred_xval, cfg)

    preprocessing_data()
    preprocessing_prediction_grid()
    interpolation()
    postprocessing()
    interpolation_xval()
    xval_scoring()

    # total runtime
    print(f"\nTotal runtime: {(datetime.now() - t)}.\n\n")


if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
