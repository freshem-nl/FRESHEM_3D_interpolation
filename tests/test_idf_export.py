"""Tests for NetCDF -> IDF export."""

import numpy as np
import xarray as xr

import imod

from scripts.idf_export import export_layer_coloured, export_voxel_bulk, var_folder

BULK_EXPORT = {
    "filename_template": "idx_{idx:03d}_{var}_NAP_{z}.idf",
    "z_format": ".2f",
    "z_offset": -0.25,
}


def test_var_folder():
    assert var_folder("P(rho≤5)") == "P_rho_le_5"


def _layer_dataset(properties):
    top = np.array(
        [
            [[-10.0, -10.0], [-10.0, -10.0]],
            [[-20.0, -20.0], [-20.0, -20.0]],
        ],
        dtype=np.float32,
    )
    bottom = top - 5.0
    data_vars = {
        "top": (("layer", "y", "x"), top),
        "bottom": (("layer", "y", "x"), bottom),
    }
    data_vars.update(
        {
            name: (("layer", "y", "x"), values)
            for name, values in properties.items()
        }
    )
    return xr.Dataset(
        coords={"layer": [1, 2], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars=data_vars,
    )


def test_export_coloured_3d_model(tmp_path):
    prop = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    ds = _layer_dataset({"P(rho≤5)": prop})
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_layer_coloured(nc_path, dst_dir, ["P(rho≤5)"])

    out_dir = dst_dir / "P_rho_le_5"
    expected = [
        "001_layer01_top.idf",
        "002_layer01_P_rho_le_5.idf",
        "003_layer01_bottom.idf",
        "004_layer02_top.idf",
        "005_layer02_P_rho_le_5.idf",
        "006_layer02_bottom.idf",
    ]
    for fname in expected:
        assert (out_dir / fname).is_file()

    manifest = (out_dir / "imod_load_order.txt").read_text(encoding="utf-8")
    for fname in expected:
        assert fname in manifest

    back = imod.idf.open(out_dir / "002_layer01_P_rho_le_5.idf")
    np.testing.assert_allclose(back.sortby("y").values, prop[0], rtol=1e-5)


def test_export_coloured_3d_model_multiple_properties(tmp_path):
    prop5 = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    prop1 = prop5 + 0.1
    ds = _layer_dataset({"P(rho≤5)": prop5, "P(rho≤1)": prop1})
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_layer_coloured(nc_path, dst_dir, ["P(rho≤5)", "P(rho≤1)"])

    for folder, prop in (("P_rho_le_5", prop5), ("P_rho_le_1", prop1)):
        out_dir = dst_dir / folder
        fname = f"002_layer01_{folder}.idf"
        assert (out_dir / fname).is_file()
        assert (out_dir / "imod_load_order.txt").is_file()
        back = imod.idf.open(out_dir / fname)
        np.testing.assert_allclose(back.sortby("y").values, prop[0], rtol=1e-5)


def test_export_voxel_model_bulk(tmp_path):
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    ds = xr.Dataset(
        coords={"z": [-50.0, -49.5], "x": [100.0, 150.0], "y": [200.0, 250.0]},
        data_vars={"P(rho≤5)": (("z", "y", "x"), data)},
    )
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_voxel_bulk(
        nc_path,
        dst_dir,
        ["P(rho≤5)"],
        export_cfg=BULK_EXPORT,
    )

    path = dst_dir / "P_rho_le_5" / "idx_000_P_rho_le_5_NAP_-50_25.idf"
    assert path.is_file()
    back = imod.idf.open(path)
    np.testing.assert_allclose(back.sortby("y").values, data[0], rtol=1e-5)


def test_export_uppercase_dims_with_mapping(tmp_path):
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    ds = xr.Dataset(
        coords={"Z": [-50.0, -49.5], "X": [100.0, 150.0], "Y": [200.0, 250.0]},
        data_vars={"P(150)": (("Z", "Y", "X"), data)},
    )
    nc_path = tmp_path / "pred.nc"
    ds.to_netcdf(nc_path)

    dst_dir = tmp_path / "idf"
    export_voxel_bulk(
        nc_path,
        dst_dir,
        ["P(150)"],
        dim_mapping={"Z": "z", "Y": "y", "X": "x"},
        export_cfg=BULK_EXPORT,
    )

    path = dst_dir / "P_150" / "idx_000_P_150_NAP_-50_25.idf"
    assert path.is_file()
