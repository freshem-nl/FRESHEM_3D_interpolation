import os
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from scripts import (
    anisotropy,
    config_loader,
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
        if cfg["variable_name"].lower() == "rho":
            data = read.skytem_xyz(cfg)
            data = preproc_data.drop_below_doi_and_resample_layers_to_z(data, cfg)
            data = preproc_data.quantiles_and_indicator_probs(data, cfg)
        elif cfg["variable_name"].lower() == "cl":
            data = read.deltares_cl(cfg)
            data = preproc_data.percentiles_to_indicators(data, cfg)
            data = preproc_data.resample_layers_to_z(data, cfg)
        if method == "ml":
            data = preproc_ml.OGC(data, cfg)
        ###TEMP
        cond = (data["x"] > 39700) & (data["x"] < 43900) & (data["y"] > 391400) & (data["y"] < 397600)
        data = data.loc[cond]
        ### END TEMP
        write.table(data, cfg["path_preproc_data"])
        visualisation.plot_df(data, "preproc - data", cfg)

    def preprocessing_data_gridded():
        data = read.table(cfg["path_preproc_data"])
        data_g = preproc_grid.snap_data_to_grid(data, cfg)
        if method == "geostat":
            data_g = anisotropy.from_data(data_g, cfg)
            data_g['magnitude'] = 1000/data_g['short_dist']
            write.ds_anisotropy_to_tif(data_g, "preproc - data anisotropy", cfg)
        write.dataset(data_g, cfg["path_preproc_data_gridded"])
        visualisation.plot_ds(data_g, "preproc - gridded data", cfg)

    def preprocessing_prediction_grid():
        data_g = read.dataset(cfg["path_preproc_data_gridded"])
        mask_xy = preproc_grid.mask_xy(data_g, cfg)
        mask_z = preproc_grid.mask_z(data_g, cfg)
        pred = preproc_grid.combine_masks(data_g, mask_xy, mask_z)
        if method == "ml":
            pred = preproc_ml.OGC(pred, cfg)
        if method == "geostat":
            # interpoleer data_aniso parameters naar relevante ellips parameters (hoek, ratio) in pred
            pass
        write.dataset(pred, cfg["path_preproc_prediction_grid"])
        visualisation.plot_ds(pred, "preproc - prediction grid", cfg)

    def interpolation():
        pred = read.dataset(cfg["path_preproc_prediction_grid"])
        if method == "ml":
            data = read.table(cfg["path_preproc_data"])
            model, output_names = ml.rf_train(data, cfg)
            pred = ml.rf_predict(model, output_names, pred, cfg)
        elif method == "geostat":
            data_g = read.dataset(cfg["path_preproc_data_gridded"])
            pred = geostat.kriging(data_g, pred, cfg)
        pred = postproc.ensure_monotonicity(pred, cfg)
        write.dataset(pred, cfg["path_prediction"])
        write.ds_to_tiff(pred, cfg["dir_rasters"], "pred")
        visualisation.plot_ds(pred, "pred", cfg)

    def postprocessing():
        ds = postproc.ds_ind_probs_to_quantiles(cfg)
        visualisation.plot_ds(ds, "postproc", cfg)
        write.ds_to_tiff(ds, cfg["dir_rasters"], "postproc")

    def interpolation_xval():
        pred = read.dataset(cfg["path_preproc_prediction_grid"])
        data = read.table(cfg["path_preproc_data"])
        data_g = read.dataset(cfg["path_preproc_data_gridded"])

        lines = xval.xval_lines(cfg)
        model_mask = pred["mask"].copy()  # overall mask for reuse
        txt = "crossvalidation: train without data of line, predict on line"
        for line in tqdm(lines["line_no"].unique(), desc=txt, unit="line", leave=True):
            pred["mask"] = xval.mask_line(lines, model_mask, line)  # only voxels in model_mask and line
            if method == "ml":
                data_fold = data[data["line_no"] != line].copy()  # exclude data from line
                model = ml.rf_train(data_fold, cfg, verbose=False)
                pred = ml.rf_predict(model, pred, cfg, verbose=False)
            elif method == "geostat":
                data_g_fold = data_g.copy().where(~pred["mask"])  # exclude voxels of line
                pred = geostat.kriging(data_g_fold, pred, cfg, verbose=False)
        pred = postproc.ensure_monotonicity(pred, cfg)
        write.dataset(pred, cfg["path_prediction_xval"])
        write.ds_to_tiff(pred, cfg["dir_rasters"], "xval")
        visualisation.plot_ds(pred, "xval", cfg)

    def xval_scoring():
        xval.validation(cfg)

    # preprocessing_data()
    # preprocessing_data_gridded()
    # preprocessing_prediction_grid()
    interpolation()
    postprocessing()
    # interpolation_xval()
    # xval_scoring()

    # total runtime
    print(f"\nTotal runtime: {(datetime.now() - t)}.\n\n")


if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
