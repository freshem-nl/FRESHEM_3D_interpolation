import numpy as np
import xarray as xr
import pandas as pd

def add_to_dataset(data, ds):

    if isinstance(data, pd.DataFrame):
        if 'ix' in data.index.names and 'iy' in data.index.names and 'iz' in data.index.names:
            ix = data.index.get_level_values("ix").to_numpy(np.int32)
            iy = data.index.get_level_values("iy").to_numpy(np.int32)
            iz = data.index.get_level_values("iz").to_numpy(np.int32)

        for col in data.columns:
            dtype=data[col].dtype
            arr = np.full((ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]), np.nan, dtype=dtype)
            arr[iz, iy, ix] = data[col].to_numpy(dtype=dtype)
            ds[f"{col}"] = (("z", "y", "x"), arr)

    return ds

def save_dataset(ds, path):
    path = path.with_suffix(".nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)