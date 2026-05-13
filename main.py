import os
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from scripts import (
    _read_and_write,
    config_loader,
    ml,
    postproc,
    preproc_data,
    preproc_grid,
    preproc_ml,
    visualisation,
    xval,
)


def main(cfg):
    t = datetime.now()

    def preprocessing_data():
        df = _read_and_write.read_skytem_xyz(cfg)
        df = preproc_data.drop_below_doi_and_resample_layers_to_z(df, cfg)
        df = preproc_data.quantiles_and_indicator_probs(df, cfg)
        visualisation.plot_df(df, "preproc - data", cfg)

    def preprocessing_grid():
        ds = preproc_grid.snap_indicator_probs_to_grid(cfg)
        mask_xy = preproc_grid.mask_xy(ds, cfg)
        mask_z = preproc_grid.mask_z(ds, cfg)
        ds = preproc_grid.combine_masks(ds, mask_xy, mask_z, cfg)
        visualisation.plot_ds(ds, "preproc - gridded data", cfg)

    def machine_learning():
        df, ds_feat = preproc_ml.OGC(cfg)
        model, output_names = ml.rf_train(df, cfg)
        ds_pred = ml.rf_predict(model, output_names, ds_feat, cfg)
        visualisation.plot_ds(ds_pred, "pred", cfg)

    def postprocessing():
        ds = postproc.quantiles(cfg)
        visualisation.plot_ds(ds, "postproc - grid", cfg)
        _read_and_write.ds_to_tiff(ds, cfg["dir_rasters"], "postproc")

    def machine_learning_xval():
        lines_name, lines_xy = xval.xval_lines(cfg)
        ds_pred = None
        txt = "crossvalidation: train without data of line, predict on line"
        df, ds_feat = preproc_ml.OGC(cfg)
        model_mask = ds_feat["mask"].copy()  # overall mask for reuse
        for line in tqdm(lines_name, desc=txt, unit="line", leave=True):
            # exclude line from training data, exclude outside mask from prediction grid
            df_train = df[df["LINE_NO"] != line].copy()  # exclude line
            ds_feat["mask"] = xval.mask_line(lines_xy, model_mask, line)  # include line only
            # train and predict
            model, output_names = ml.rf_train(df_train, cfg, verbose=False)  # train
            ds_pred = ml.rf_predict(model, output_names, ds_feat, cfg, ds_pred=ds_pred, xval=True, verbose=False)
        visualisation.plot_ds(ds_pred, "pred - xval", cfg)

    def xval_validation():
        # voor later: eenvoudig quantiiles berekenen ook van gridded data. Data en prediction quantiles omzetten naar klassen, en vergelijken via confusion matrix, of bekijk verschil in klassen tussen p25 en p75
        pass

    preprocessing_data()
    preprocessing_grid()
    machine_learning()
    postprocessing()
    machine_learning_xval()

    # total runtime
    print(f"\nTotal runtime: {(datetime.now() - t)}.")

if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
