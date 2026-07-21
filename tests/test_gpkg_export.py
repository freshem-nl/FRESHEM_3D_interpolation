"""Tests for SkyTEM xyz -> GeoPackage export."""

import geopandas as gpd

from scripts.gpkg_export import (
    clip_bbox,
    clip_doi,
    export_rho_xyz,
    flightlines_from_xyz,
    parse_skytem_xyz,
)


XYZ_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 DOI_STANDARD DOI_CONSERVATIVE
1 100.0 200.0 5.0 10.5 1.2 0.0 5.0 5.0 50.0 40.0
1 150.0 250.0 5.0 20.0 2.0 5.0 15.0 10.0 50.0 40.0
2 300.0 400.0 4.0 30.0 3.0 0.0 8.0 8.0 45.0 35.0
2 350.0 450.0 4.0 35.0 3.5 8.0 18.0 10.0 45.0 35.0
"""

XYZ_DOI_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 RHO_2 RHO_STD2 DEP_TOP_2 DEP_BOT_2 THK_2 DOI_STANDARD DOI_CONSERVATIVE
1 100.0 200.0 5.0 10.5 1.2 0.0 5.0 5.0 8.0 1.0 5.0 15.0 10.0 4.0 3.0
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


def test_clip_doi():
    import pandas as pd

    df = pd.DataFrame(
        {
            "z_top": [5.0, 0.0, -2.0],
            "z_bottom": [0.0, -10.0, -5.0],
            "z_doi_standard": [1.0, 1.0, 1.0],
        }
    )

    clipped = clip_doi(df)

    assert len(clipped) == 1
    assert clipped.iloc[0]["z_top"] == 5.0
    assert clipped.iloc[0]["z_bottom"] == 1.0


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


def test_export_rho_xyz_with_doi_clip(tmp_path):
    xyz_path = tmp_path / "sample_doi.xyz"
    gpkg_path = tmp_path / "sample_doi.gpkg"
    xyz_path.write_text(XYZ_DOI_SAMPLE, encoding="utf-8")

    export_rho_xyz(xyz_path, gpkg_path, epsg=28992, apply_doi_clip=True, include_flightlines=False)

    points = gpd.read_file(gpkg_path, layer="rho_points")

    assert len(points) == 1
    assert points.iloc[0]["layer"] == 1


def test_export_rho_xyz_without_doi_clip(tmp_path):
    xyz_path = tmp_path / "sample_doi.xyz"
    gpkg_path = tmp_path / "sample_doi_full.gpkg"
    xyz_path.write_text(XYZ_DOI_SAMPLE, encoding="utf-8")

    export_rho_xyz(xyz_path, gpkg_path, epsg=28992, apply_doi_clip=False, include_flightlines=False)

    points = gpd.read_file(gpkg_path, layer="rho_points")

    assert len(points) == 2
