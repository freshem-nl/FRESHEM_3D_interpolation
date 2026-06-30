"""Tests for NetCDF -> IDF export."""

import numpy as np
import xarray as xr

import imod

from scripts.idf_export import export_netcdf, var_folder


def test_var_folder():
    assert var_folder("P(rho≤5)") == "P_rho_le_5"


def test_export_layer_model(tmp_path):
    ds = xr.Dataset(
        coords={"layer": [1, 2], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars={
            "P(rho≤1)": (
                ("layer", "y", "x"),
                np.ones((2, 2, 2), dtype=np.float32),
            ),
        },
    )
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_netcdf(nc_path, dst_dir, ["P(rho≤1)"], vertical_dim="layer")

    path = dst_dir / "P_rho_le_1" / "idx_000_layer_01.idf"
    assert path.is_file()
    assert imod.idf.open(path).shape == (2, 2)


def test_export_voxel_model(tmp_path):
    data = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    ds = xr.Dataset(
        coords={"z": [-50.0], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars={"P(rho≤5)": (("z", "y", "x"), data)},
    )
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_netcdf(nc_path, dst_dir, ["P(rho≤5)"], vertical_dim="z", z_offset=-0.25)

    path = dst_dir / "P_rho_le_5" / "idx_000_NAP_-50_25.idf"
    assert path.is_file()
    back = imod.idf.open(path)
    np.testing.assert_allclose(back.sortby("y").values, data[0], rtol=1e-5)
