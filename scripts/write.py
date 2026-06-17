import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def table(data, path):

    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        data.to_parquet(path)
        # df.to_parquet(path, engine="fastparquet")


def dataset(ds, path):

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}

    path = path.with_suffix(".nc")
    path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def ds_to_tiff(ds, dir_output, name):

    os.makedirs(dir_output, exist_ok=True)

    for var in ds.data_vars:
        da = ds[var]


        # Only export 3D variables with Z, Y, X dimensions
        if set(da.dims) != {"z", "y", "x"}:
            continue

        da = da.transpose("z", "y", "x")

        # da = da.rio.set_spatial_dims(x_dim="X", y_dim="Y")

        # da.rio.to_raster(path.with_suffix(f"_{var}.tif"))
        da = da.astype("float32")
        da = da.fillna(-9999)
        da = da.rio.write_nodata(-9999)

        z_vals = ds.z.values

        da.attrs["long_name"] = [f"z={z:.1f} m" for z in z_vals]

        # Explicitly define spatial dimensions for this DataArray
        # da = da.rio.set_spatial_dims(x_dim="X", y_dim="Y")


        path = dir_output / f"{name} - {var}.tif"
        da.rio.to_raster(path)


def txt_to_yaml(data, path):

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    serializable_data = convert(data)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(serializable_data, f)
