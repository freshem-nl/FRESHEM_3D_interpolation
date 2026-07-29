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
    elif path.suffix == ".csv":
        data.to_csv(path)

def dataset(ds, path):
    """Write NetCDF with CF spatial metadata so GDAL/QGIS pick up CRS and geotransform."""
    ds = ds.copy(deep=False)

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}

    # Coordinate axes must not carry _FillValue (GDAL ignores axes that do)
    for name, standard_name, axis in (
        ("x", "projection_x_coordinate", "X"),
        ("y", "projection_y_coordinate", "Y"),
    ):
        if name not in ds.coords:
            continue
        ds[name].attrs.setdefault("units", "m")
        ds[name].attrs.setdefault("standard_name", standard_name)
        ds[name].attrs.setdefault("axis", axis)
        encoding[name] = {"_FillValue": None}

    if "layer" in ds.coords:
        encoding["layer"] = {"_FillValue": None}

    if "z" in ds.coords:
        ds["z"].attrs.setdefault("units", "m")
        ds["z"].attrs.setdefault("positive", "up")
        encoding["z"] = {"_FillValue": None}

    # Link variables to CRS; drop rioxarray's misleading coordinates=spatial_ref
    if "spatial_ref" in ds.coords or "spatial_ref" in ds.variables:
        for var in ds.data_vars:
            ds[var].attrs["grid_mapping"] = "spatial_ref"
            if ds[var].attrs.get("coordinates") == "spatial_ref":
                del ds[var].attrs["coordinates"]
        ds.attrs.setdefault("grid_mapping", "spatial_ref")

    path = path.with_suffix(".nc")
    path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def ds_to_tiff(ds, dir_output, name):

    os.makedirs(dir_output, exist_ok=True)

    for var in ds.data_vars:
        da = ds[var]

        # if set(da.dims) == {"layer", "y", "x"}:

        if set(da.dims) == {"z", "y", "x"}:
            z_vals = ds.z.values

            da = da.transpose("z", "y", "x")
            da.attrs["long_name"] = [f"z={z:.1f} m" for z in z_vals]


        if set(da.dims) == {"layer", "y", "x"}:
            da = da.rename({"layer": "z"}).transpose("z", "y", "x")
            da.attrs["long_name"] = [f"layer {int(l)}" for l in da["z"].values]


        da = da.astype("float32")
        da = da.fillna(-9999)
        da = da.rio.write_nodata(-9999)

        path = dir_output / f"{name} - {var}.tif"
        da.rio.to_raster(path)

def ds_anisotropy_to_tif(
    ds,
    name,
    cfg
):
    """Export one z-slice from a 3D xarray Dataset as a 2-band GeoTIFF."""

    dir_output = cfg["dir_rasters"]
    plotting_depths = cfg["plotting_depths"]

    os.makedirs(dir_output, exist_ok=True)

    # select target depths (exact match or nearest)
    depths = ds["z"].sel(z=np.array(plotting_depths), method="nearest").values

    for depth in depths:
        # Select a 2D slice by z value or index
        ds_2d = ds.sel(z=depth, method="nearest")

        # Build a 2-band DataArray: band 1 = magnitude, band 2 = direction
        da_out = ds_2d[['magnitude', 'long_angle']].to_array(dim="band")

        # Use numeric band coordinates
        da_out = da_out.assign_coords(band=[1, 2])

        # Optional metadata for band interpretation
        da_out.attrs["long_name"] = ['magnitude', 'direction']

        # Set nodata
        da_out = da_out.rio.write_nodata(np.nan)

        # Export as multiband GeoTIFF
        path = dir_output / f"{name} - at z={depth}m.tif"
        da_out.rio.to_raster(path)



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
