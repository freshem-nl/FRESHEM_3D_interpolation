from datetime import datetime

import numpy as np

from scripts import _preprocessing_helper, _read_and_write


def OGC(cfg):

    t0 = datetime.now()
    print("Calculating oblique geographic coordinates (OGC)...", end=" ")

    # from config
    thetas = cfg["OGC_angles"]
    path_df_in = cfg["path_preproc_data"]
    path_ds_in = cfg["path_preproc_data_gridded"]

    # read data
    df = _read_and_write.read_table(path_df_in.with_suffix(".parquet"))
    ds_data = _read_and_write.read_dataset(path_ds_in)

    # only keep mask and coordinates in dataset, data values not needed for RF prediction
    ds = ds_data.copy()
    ds = ds[["mask"]]

    for theta in thetas:

        # calculate OGC for data points and add to dataframe
        df[f"OGC_{theta}"] = _preprocessing_helper.calc_oblique_geographic_coordinates(df["X"], df["Y"], theta)

        # calculate OGC for grid cell centers and add to dataset
        x = ds["x"].values
        y = ds["y"].values
        x_grid, y_grid = np.meshgrid(x, y)
        OGC = _preprocessing_helper.calc_oblique_geographic_coordinates(x_grid.flatten(), y_grid.flatten(), theta)
        ds[f"OGC_{theta}"] = (("y", "x"), OGC.reshape(x_grid.shape))

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return df, ds
