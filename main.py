import os
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from scripts import (
    _utils,
    anisotropy,
    config_loader,
    ml,
    postproc,
    preproc_data,
    preproc_grid,
    preproc_ml,
    read,
    write,
    visualisation,
    xval,
    geostat
)


def main(cfg):
    t = datetime.now()

    # copy config file
    path = cfg["dir_output"] / "config.yaml"
    write.txt_to_yaml(cfg, path)

    method = cfg["method"]

    def preprocessing_data():
        data = read.skytem_xyz(cfg)
        data = preproc_data.drop_below_doi_and_resample_layers_to_z(data, cfg)
        data = preproc_data.quantiles_and_indicator_probs(data, cfg)
        if method == "ml":
            data = preproc_ml.OGC(data, cfg)
        write.table(data, cfg["path_preproc_data"])
        visualisation.plot_df(data, "preproc - data", cfg)

    def preprocessing_data_gridded():
        data = read.table(cfg["path_preproc_data"])
        data_g = preproc_grid.snap_data_to_grid(data, cfg)
        write.dataset(data_g, cfg["path_preproc_data_gridded"])
        visualisation.plot_ds(data_g, "preproc - gridded data", cfg)

    def preprocessing_prediction_grid():
        data_g = read.dataset(cfg["path_preproc_data_gridded"])
        mask_xy = preproc_grid.mask_xy(data_g,cfg)
        mask_z = preproc_grid.mask_z(data_g, cfg)
        pred = preproc_grid.combine_masks(data_g, mask_xy, mask_z)
        if method == "ml":
            pred = preproc_ml.OGC(pred, cfg)
        if method == "geostat":
            pass #add aniso to pred_g
        write.dataset(pred, cfg["path_preproc_prediction_grid"])
        visualisation.plot_ds(pred, "preproc - prediction grid", cfg)

    def interpolation():
        pred = read.dataset(cfg["path_preproc_prediction_grid"])
        if method == "ml":
            data = read.table(cfg["path_preproc_data"])
            model, output_names = ml.rf_train(data, cfg)
            pred = ml.rf_predict(model, output_names, pred, cfg)  # hier ALTIJD ds_pred meegeven, in deze functie NIET wegschrijven naar schijf
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
        read_and_write.ds_to_tiff(ds, cfg["dir_rasters"], "postproc")

    def xval_machine_learning():
        # TODO: invoegen aparte functie voor inlezen van df_data, ds_data en ds_pred
        df, ds_feat = preproc_ml.OGC(cfg)
        lines = xval.xval_lines(cfg)
        model_mask = ds_feat["mask"].copy()  # overall mask for reuse
        ds_pred = None
        txt = "crossvalidation: train without data of line, predict on line"
        for line in tqdm(lines["LINE_NO"].unique(), desc=txt, unit="line", leave=True):
            # TODO: dan hier aparte df_data_fold, ds_data_fold, ds_pred_fold maken en input in interoplatie functies

            # exclude line from training data, exclude outside mask from prediction grid
            df_train = df[df["LINE_NO"] != line].copy()  # exclude line
            ds_feat["mask"] = xval.mask_line(lines, model_mask, line)  # include line only
            # train and predict
            model, output_names = ml.rf_train(df_train, cfg, verbose=False)  # train
            ds_pred = ml.rf_predict(model, output_names, ds_feat, cfg, ds_pred=ds_pred, xval=True, verbose=False)
        visualisation.plot_ds(ds_pred, "xval", cfg)

    def geostat_anisotropy():
        anisotropy.main(cfg)

    def xval_scoring():
        xval.validation(cfg)

    # preprocessing_data()
    # preprocessing_data_gridded()
    # preprocessing_prediction_grid()
    interpolation()
    # postprocessing()
    # xval_machine_learning()
    # xval_scoring()
    # geostat_anisotropy()

    # total runtime
    print(f"\nTotal runtime: {(datetime.now() - t)}.")


if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
