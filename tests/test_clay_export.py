"""Tests for clay-from-resistivity export."""

import numpy as np
import xarray as xr

import imod

from scripts.clay_export import build_clay_dataset, clay_from_rho, export_clay


def test_clay_from_rho_mask_and_clip():
    rho = xr.DataArray([3.0, 5.0, 10.0, 20.0, 100.0])
    clay = clay_from_rho(rho, a=1.17, b=-0.0163, rho_min=5.0, clip_to_unit=True)

    assert np.isnan(clay.values[0])
    assert np.isnan(clay.values[1])
    np.testing.assert_allclose(clay.values[2], 1.0)  # raw ~1.007 -> clipped
    np.testing.assert_allclose(clay.values[3], 1.17 - 0.0163 * 20.0)
    np.testing.assert_allclose(clay.values[4], 0.0)  # raw negative -> clipped


def test_clay_from_rho_no_clip():
    rho = xr.DataArray([10.0, 100.0])
    clay = clay_from_rho(rho, clip_to_unit=False)
    np.testing.assert_allclose(clay.values[0], 1.17 - 0.0163 * 10.0)
    assert clay.values[1] < 0.0


def _postproc_dataset():
    rho = np.array(
        [
            [[3.0, 10.0], [20.0, 100.0]],
            [[8.0, 50.0], [5.0, 30.0]],
        ],
        dtype=np.float32,
    )
    top = np.full_like(rho, -10.0)
    return xr.Dataset(
        coords={"layer": [1, 2], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars={
            "top": (("layer", "y", "x"), top),
            "bottom": (("layer", "y", "x"), top - 5.0),
            "Q(0.5)": (("layer", "y", "x"), rho),
        },
    )


def test_build_clay_dataset():
    ds = build_clay_dataset(_postproc_dataset(), clip_to_unit=True)
    assert "clay" in ds
    assert "top" in ds and "bottom" in ds
    assert "Q(0.5)" in ds
    assert ds["clay"].attrs["rho_min_ohm_m"] == 5.0
    assert np.isnan(ds["clay"].values[0, 0, 0])
    np.testing.assert_allclose(ds["clay"].values[0, 0, 1], 1.0)


def test_export_clay_idf_and_nc(tmp_path):
    ds = _postproc_dataset()
    nc_path = tmp_path / "Zuid_A1 - postproc.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "out"
    result = export_clay(nc_path, dst_dir, write_idf=True, write_nc=True)

    idf_dir = result["idf_dir"]
    assert idf_dir == dst_dir / "clay"
    expected = [
        "001_layer01_top.idf",
        "002_layer01_clay.idf",
        "003_layer01_bottom.idf",
        "004_layer02_top.idf",
        "005_layer02_clay.idf",
        "006_layer02_bottom.idf",
    ]
    for fname in expected:
        assert (idf_dir / fname).is_file()
    assert (idf_dir / "imod_load_order.txt").is_file()

    back = imod.idf.open(idf_dir / "002_layer01_clay.idf")
    # layer 1: [[3, 10], [20, 100]] -> nan, 1.0, ~0.844, 0.0
    vals = back.sortby("y").values
    assert np.isnan(vals[0, 0])
    np.testing.assert_allclose(vals[0, 1], 1.0, rtol=1e-5)

    nc_out = result["nc_out"]
    assert nc_out.is_file()
    assert nc_out.name == "Zuid_A1 - postproc - clay.nc"
    written = xr.open_dataset(nc_out)
    try:
        assert "clay" in written
        assert bool(written.attrs["clay_clipped_to_0_1"])
    finally:
        written.close()


def test_export_clay_requires_output(tmp_path):
    ds = _postproc_dataset()
    nc_path = tmp_path / "postproc.nc"
    ds.to_netcdf(nc_path)
    try:
        export_clay(nc_path, tmp_path / "out", write_idf=False, write_nc=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "write_idf" in str(exc)
