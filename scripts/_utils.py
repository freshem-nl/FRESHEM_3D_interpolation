import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rio


def df_to_gdf(df, epsg=None, crs=None):

    if epsg is not None:
        crs = f"EPSG:{epsg}"
    # Convert to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)
    return gdf


def init_ds(df, cellsize_xy=None, cellsize_z=None, epsg=None, buffer_xy=0, buffer_z=0):

    x = np.arange(df["x"].min() - buffer_xy, df["x"].max() + buffer_xy + cellsize_xy, cellsize_xy)

    y = np.arange(df["y"].min() - buffer_xy, df["y"].max() + buffer_xy + cellsize_xy, cellsize_xy)

    z = np.arange(df["z"].min() - buffer_z, df["z"].max() + buffer_z + cellsize_z, cellsize_z)

    ds = xr.Dataset(coords={"z": z, "y": y, "x": x})


    ds.attrs.update({
        "cellsize_x": cellsize_xy,
        "cellsize_y": cellsize_xy,
        "cellsize_z": cellsize_z,
        "grid_mapping": "spatial_ref"
    })


    ds["x"].attrs.update({"standard_name": "X-coordinate", "units": "m"})
    ds["y"].attrs.update({"standard_name": "Y-coordinate", "units": "m"})
    ds["z"].attrs.update({"standard_name": "Z-coordinate", "positive": "up", "units": "m"})


    if epsg is not None:
        ds.attrs["crs"] = f"EPSG:{int(epsg)}"
        ds = ds.rio.write_crs(f"EPSG:{epsg}")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")


    return ds


def add_df_to_ds(ds, df, value_cols=None, dtype="float32"):
    """
    Add one or more value columns from a DataFrame to an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have coordinates ds.x, ds.y, ds.z (cell centers).
    df : pandas.DataFrame
        Must contain coordinate columns (e.g. X,Y,Z) and one or more value columns.
    value_cols : list or None
        Which df columns to write. If None: all columns except coord columns.
    dtype : str
        dtype for new variables (default float32).

    Returns
    -------
    ds : xarray.Dataset (modified)
    """
    # Identify coordinate columns in df
    df_coord_cols = ["x", "y", "z"]

    # Auto-detect value columns if not provided
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in df_coord_cols]

    # Grid coords
    x = ds.x.values
    y = ds.y.values
    z = ds.z.values

    # Map centers -> indices (assumes df coords match ds coords)
    ix = np.searchsorted(x, df['x'].to_numpy())
    iy = np.searchsorted(y, df['y'].to_numpy())
    iz = np.searchsorted(z, df['z'].to_numpy())

    # value
    dfv = df.loc[:, value_cols]

    # Create variables on-the-fly (if missing), then fill
    shape = (len(z), len(y), len(x))
    for c in value_cols:
        if c not in ds:
            ds[c] = (("z", "y", "x"), np.full(shape, np.nan, dtype=dtype))
        ds[c].values[iz, iy, ix] = dfv[c].to_numpy()

    return ds