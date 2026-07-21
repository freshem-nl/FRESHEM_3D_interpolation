"""Tests for per-area NetCDF combine."""

import numpy as np
import xarray as xr

from scripts.netcdf_combine import combine


def _postproc_dataset(x, y, prop_value):
    layers = [1, 2]
    shape = (len(layers), len(y), len(x))
    top = np.full(shape, -10.0, dtype=np.float32)
    return xr.Dataset(
        coords={"layer": layers, "x": x, "y": y},
        data_vars={
            "top": (("layer", "y", "x"), top),
            "bottom": (("layer", "y", "x"), top - 5.0),
            "Q(0.5)": (("layer", "y", "x"), np.full(shape, prop_value, dtype=np.float32)),
        },
    )


def test_combine_adjacent_tiles(tmp_path, monkeypatch):
    ds_a = _postproc_dataset([25.0, 75.0], [200025.0, 200075.0], 1.0)
    ds_b = _postproc_dataset([75.0, 125.0], [200075.0, 200125.0], 2.0)
    path_a = tmp_path / "a.nc"
    path_b = tmp_path / "b.nc"
    ds_a.to_netcdf(path_a)
    ds_b.to_netcdf(path_b)

    written = {}

    def fake_write(ds, path):
        written["ds"] = ds
        written["path"] = path

    monkeypatch.setattr("scripts.netcdf_combine.write.dataset", fake_write)

    combined = combine([path_a, path_b], tmp_path / "combined.nc")

    np.testing.assert_allclose(
        combined["Q(0.5)"].sel(layer=1, x=25, y=200025).item(), 1.0
    )
    np.testing.assert_allclose(
        combined["Q(0.5)"].sel(layer=1, x=125, y=200125).item(), 2.0
    )
    assert np.isnan(combined["Q(0.5)"].sel(layer=1, x=25, y=200125).item())
    assert written["path"] == tmp_path / "combined.nc"


def test_combine_overlap_list_order(tmp_path, monkeypatch):
    ds_a = _postproc_dataset([25.0, 75.0], [200025.0, 200075.0], 1.0)
    ds_b = _postproc_dataset([25.0, 75.0], [200025.0, 200075.0], 9.0)
    path_a = tmp_path / "a.nc"
    path_b = tmp_path / "b.nc"
    ds_a.to_netcdf(path_a)
    ds_b.to_netcdf(path_b)

    monkeypatch.setattr("scripts.netcdf_combine.write.dataset", lambda ds, path: None)

    combined = combine([path_a, path_b], tmp_path / "combined.nc")
    np.testing.assert_allclose(combined["Q(0.5)"].values, 9.0)
