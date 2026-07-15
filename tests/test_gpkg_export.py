"""Tests for SkyTEM xyz -> GeoPackage export."""

import geopandas as gpd

from scripts.gpkg_export import clip_bbox, export_rho_xyz, flightlines_from_xyz
from scripts.read import parse_skytem_xyz


XYZ_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 DOI_STANDARD DOI_CONSERVATIVE
1 100.0 200.0 5.0 10.5 1.2 0.0 5.0 5.0 50.0 40.0
1 150.0 250.0 5.0 20.0 2.0 5.0 15.0 10.0 50.0 40.0
2 300.0 400.0 4.0 30.0 3.0 0.0 8.0 8.0 45.0 35.0
2 350.0 450.0 4.0 35.0 3.5 8.0 18.0 10.0 45.0 35.0
"""


def _write_sample_xyz(path):
    path.write_text(XYZ_SAMPLE, encoding="utf-8")


def test_parse_skytem_xyz(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    _write_sample_xyz(xyz_path)

    df = parse_skytem_xyz(xyz_path)

    assert len(df) == 4
    assert "rho_1" in df.columns
    assert df.loc[0, "x"] == 100.0


def test_flightlines_from_xyz(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    _write_sample_xyz(xyz_path)
    df = parse_skytem_xyz(xyz_path)

    gdf = flightlines_from_xyz(df, epsg=28992)

    assert len(gdf) == 2
    assert set(gdf["line_no"]) == {1, 2}


def test_export_rho_xyz(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    gpkg_path = tmp_path / "sample.gpkg"
    _write_sample_xyz(xyz_path)

    export_rho_xyz(xyz_path, gpkg_path, epsg=28992)

    points = gpd.read_file(gpkg_path, layer="rho_points")
    lines = gpd.read_file(gpkg_path, layer="flightlines")

    assert len(points) == 4
    assert {"rho", "z_top", "z_bottom", "layer"}.issubset(points.columns)
    assert len(lines) == 2


def test_clip_bbox(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    _write_sample_xyz(xyz_path)
    df = parse_skytem_xyz(xyz_path)

    clipped = clip_bbox(df, [95, 105, 195, 205])

    assert len(clipped) == 1
    assert clipped.iloc[0]["line_no"] == 1
    assert clipped.iloc[0]["x"] == 100.0


def test_export_rho_xyz_with_bbox(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    gpkg_path = tmp_path / "sample_clipped.gpkg"
    _write_sample_xyz(xyz_path)

    export_rho_xyz(xyz_path, gpkg_path, epsg=28992, bbox=[290, 360, 390, 460])

    points = gpd.read_file(gpkg_path, layer="rho_points")
    lines = gpd.read_file(gpkg_path, layer="flightlines")

    assert len(points) == 2
    assert set(points["line_no"]) == {2}
    assert len(lines) == 1
