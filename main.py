import os
from pathlib import Path

from scripts import _read_and_write, config_loader, ml, preproc_data, preproc_grid, preproc_ml, postproc


def main(cfg):

    def preprocessing_data():
        df = _read_and_write.read_skytem_xyz(cfg)
        df = preproc_data.drop_below_doi_and_resample_layers_to_z(df, cfg)
        df = preproc_data.calc_indicators(df, cfg)
        preproc_data.plotting(df, cfg)

    def preprocessing_grid():
        ds = preproc_grid.snap_data_to_grid(cfg)
        mask_xy = preproc_grid.mask_xy(ds, cfg)
        mask_z = preproc_grid.mask_z(ds, cfg)
        ds = preproc_grid.combine_masks(ds, mask_xy, mask_z, cfg)
        preproc_grid.plotting(ds, cfg)

    def machine_learning():
        df, ds = preproc_ml.OGC(cfg)
        model, output_names = ml.rf_train(df, cfg)
        ml.rf_predict(model, output_names, ds, cfg)

    def postprocessing():
        ds = postproc.indicators_to_percentiles(cfg)
        postproc.plotting(ds, cfg)

    preprocessing_data()
    preprocessing_grid()
    machine_learning()
    postprocessing()


if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
