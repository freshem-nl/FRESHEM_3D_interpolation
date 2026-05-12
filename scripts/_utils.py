import numpy as np
import xarray as xr
import pandas as pd

# def add_to_dataset(data, ds):

#     if isinstance(data, pd.DataFrame):
#         if 'ix' in data.index.names and 'iy' in data.index.names and 'iz' in data.index.names:
#             ix = data.index.get_level_values("ix").to_numpy(np.int32)
#             iy = data.index.get_level_values("iy").to_numpy(np.int32)
#             iz = data.index.get_level_values("iz").to_numpy(np.int32)

#         for col in data.columns:
#             dtype=data[col].dtype
#             if np.issubdtype(dtype, np.floating):
#                 nodata_value = np.float32(np.nan)
#             elif np.issubdtype(dtype, np.integer):
#                 nodata_value = np.int32(-9999)
#             arr = np.full((ds.sizes["z"], ds.sizes["y"], ds.sizes["x"]), nodata_value, dtype=dtype)
#             arr[iz, iy, ix] = data[col].to_numpy(dtype=dtype)
#             ds[f"{col}"] = (("z", "y", "x"), arr)

#     return ds

import numpy as np
import xarray as xr
import rioxarray


def init_ds(df, cellsize_xy=None, cellsize_z=None, epsg=None, buffer_xy=0, buffer_z=0):

    x = np.arange(
        df["X"].min() - buffer_xy,
        df["X"].max() + buffer_xy + cellsize_xy,
        cellsize_xy
    )

    y = np.arange(
        df["Y"].min() - buffer_xy,
        df["Y"].max() + buffer_xy + cellsize_xy,
        cellsize_xy
    )

    z = np.arange(
        df["Z"].min() - buffer_z,
        df["Z"].max() + buffer_z + cellsize_z,
        cellsize_z
    )

    ds = xr.Dataset(coords={"x": x, "y": y, "z": z})

    if epsg is not None:
        ds = ds.rio.write_crs(f"EPSG:{epsg}")

    return ds


import numpy as np

def add_df_to_ds(ds, df, coord_map=None, value_cols=None, dtype="float32"):
    """
    Add one or more value columns from a DataFrame to an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have coordinates ds.x, ds.y, ds.z (cell centers).
    df : pandas.DataFrame
        Must contain coordinate columns (e.g. X,Y,Z) and one or more value columns.
    coord_map : dict
        Mapping from ds coord name -> df column name, default {"x":"X","y":"Y","z":"Z"}.
    value_cols : list or None
        Which df columns to write. If None: all columns except coord columns.
    dtype : str
        dtype for new variables (default float32).

    Returns
    -------
    ds : xarray.Dataset (modified)
    """
    if coord_map is None:
        coord_map = {"x": "X", "y": "Y", "z": "Z"}

    # Identify coordinate columns in df
    df_coord_cols = [coord_map["x"], coord_map["y"], coord_map["z"]]

    # Auto-detect value columns if not provided
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in df_coord_cols]

    # Grid coords
    x = ds.x.values
    y = ds.y.values
    z = ds.z.values

    # Map centers -> indices (assumes df coords match ds coords)
    ix = np.searchsorted(x, df[coord_map["x"]].to_numpy())
    iy = np.searchsorted(y, df[coord_map["y"]].to_numpy())
    iz = np.searchsorted(z, df[coord_map["z"]].to_numpy())

    #value
    dfv = df.loc[:, value_cols]

    # Create variables on-the-fly (if missing), then fill
    shape = (len(z), len(y), len(x))
    for c in value_cols:
        if c not in ds:
            ds[c] = (("z", "y", "x"), np.full(shape, np.nan, dtype=dtype))
        ds[c].values[iz, iy, ix] = dfv[c].to_numpy()

    return ds



