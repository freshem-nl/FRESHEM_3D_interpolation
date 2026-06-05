import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray as rio


def df_to_gdf(df, epsg=None, crs=None):

    if epsg is not None:
        crs = f"EPSG:{epsg}"
    # Convert to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs=crs)
    return gdf


def init_ds(df, cellsize_xy=None, cellsize_z=None, epsg=None, buffer_xy=0, buffer_z=0):

    x = np.arange(df["X"].min() - buffer_xy, df["X"].max() + buffer_xy + cellsize_xy, cellsize_xy)

    y = np.arange(df["Y"].min() - buffer_xy, df["Y"].max() + buffer_xy + cellsize_xy, cellsize_xy)

    z = np.arange(df["Z"].min() - buffer_z, df["Z"].max() + buffer_z + cellsize_z, cellsize_z)

    ds = xr.Dataset(coords={"Z": z, "Y": y, "X": x})


    ds.attrs.update({
        "cellsize_x": cellsize_xy,
        "cellsize_y": cellsize_xy,
        "cellsize_z": cellsize_z,
        "grid_mapping": "spatial_ref"
    })


    ds["X"].attrs.update({"standard_name": "X", "units": "m"})
    ds["Y"].attrs.update({"standard_name": "Y", "units": "m"})
    ds["Z"].attrs.update({"standard_name": "Z", "positive": "up", "units": "m"})


    if epsg is not None:
        ds.attrs["crs"] = f"EPSG:{int(epsg)}"
        ds = ds.rio.write_crs(f"EPSG:{epsg}")
        ds = ds.rio.set_spatial_dims(x_dim="X", y_dim="Y")


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
    df_coord_cols = ["X", "Y", "Z"]

    # Auto-detect value columns if not provided
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in df_coord_cols]

    # Grid coords
    x = ds.X.values
    y = ds.Y.values
    z = ds.Z.values

    # Map centers -> indices (assumes df coords match ds coords)
    ix = np.searchsorted(x, df['X'].to_numpy())
    iy = np.searchsorted(y, df['Y'].to_numpy())
    iz = np.searchsorted(z, df['Z'].to_numpy())

    # value
    dfv = df.loc[:, value_cols]

    # Create variables on-the-fly (if missing), then fill
    shape = (len(z), len(y), len(x))
    for c in value_cols:
        if c not in ds:
            ds[c] = (("Z", "Y", "X"), np.full(shape, np.nan, dtype=dtype))
        ds[c].values[iz, iy, ix] = dfv[c].to_numpy()

    return ds