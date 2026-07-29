"""Tests for CF spatial metadata on NetCDF write."""

import netCDF4 as nc
import numpy as np
import rioxarray  # noqa: F401  — registers .rio accessor
import xarray as xr

from scripts import write


def test_dataset_writes_cf_spatial_metadata(tmp_path):
    ds = xr.Dataset(
        coords={"layer": [1, 2], "x": [120175.0, 120225.0], "y": [411625.0, 411675.0]},
        data_vars={
            "Q(0.5)": (
                ("layer", "y", "x"),
                np.ones((2, 2, 2), dtype=np.float32),
            ),
        },
    )
    ds = ds.rio.write_crs("EPSG:28992")
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    # Mimic the misleading attr rioxarray can leave on variables
    ds["Q(0.5)"].attrs["coordinates"] = "spatial_ref"

    out = tmp_path / "postproc.nc"
    write.dataset(ds, out)

    with nc.Dataset(out) as root:
        assert root.variables["x"].ncattrs() == () or "_FillValue" not in root.variables["x"].ncattrs()
        assert "_FillValue" not in root.variables["y"].ncattrs()
        assert "_FillValue" not in root.variables["layer"].ncattrs()
        assert root.variables["x"].getncattr("standard_name") == "projection_x_coordinate"
        assert root.variables["y"].getncattr("standard_name") == "projection_y_coordinate"
        assert root.variables["Q(0.5)"].getncattr("grid_mapping") == "spatial_ref"
        assert "coordinates" not in root.variables["Q(0.5)"].ncattrs()
        assert "spatial_ref" in root.variables


def test_dataset_writes_cf_for_voxel_z(tmp_path):
    ds = xr.Dataset(
        coords={"z": [-10.0, -9.5], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars={
            "Q(0.5)": (("z", "y", "x"), np.ones((2, 2, 2), dtype=np.float32)),
        },
    )
    ds = ds.rio.write_crs("EPSG:28992")

    out = tmp_path / "voxel.nc"
    write.dataset(ds, out)

    with nc.Dataset(out) as root:
        assert "_FillValue" not in root.variables["z"].ncattrs()
        assert root.variables["z"].getncattr("positive") == "up"
        assert root.variables["Q(0.5)"].getncattr("grid_mapping") == "spatial_ref"
