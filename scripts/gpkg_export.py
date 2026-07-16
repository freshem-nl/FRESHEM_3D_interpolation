from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from scripts import preproc_data, read


def z_doi_column(doi_name: str) -> str:
    return doi_name if doi_name.startswith("z_") else f"z_{doi_name}"


def clip_doi(df: pd.DataFrame, doi_name: str = "doi_standard") -> pd.DataFrame:
    """Drop layers below DOI and clip layer bottoms to the DOI surface."""
    z_doi = z_doi_column(doi_name)
    if z_doi not in df.columns:
        raise ValueError(f"Column {z_doi!r} not found; expected doi field from restructure().")

    out = df.loc[df["z_top"] > df[z_doi]].copy()
    out["z_bottom"] = np.maximum(out["z_bottom"], out[z_doi])
    return out.reset_index(drop=True)


def clip_bbox(df: pd.DataFrame, bbox) -> pd.DataFrame:
    """Filter rows to an XY bounding box [xmin, xmax, ymin, ymax]."""
    if not bbox:
        return df

    xmin, xmax, ymin, ymax = bbox
    mask = (df["x"] >= xmin) & (df["x"] <= xmax) & (df["y"] >= ymin) & (df["y"] <= ymax)
    return df.loc[mask].reset_index(drop=True)


def flightlines_from_xyz(df: pd.DataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Build one LineString per flight line, preserving row order in the xyz file."""
    rows = []
    for line_no, group in df.groupby("line_no", sort=False):
        if len(group) < 2:
            continue
        rows.append(
            {
                "line_no": int(line_no),
                "geometry": LineString(zip(group["x"], group["y"])),
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["line_no", "geometry"], crs=f"EPSG:{epsg}")
    return gpd.GeoDataFrame(rows, crs=f"EPSG:{epsg}")


def export_rho_xyz(
    xyz_path,
    gpkg_path,
    *,
    epsg=28992,
    include_flightlines=True,
    points_layer="rho_points",
    flightlines_layer="flightlines",
    bbox=None,
    apply_doi_clip=True,
    doi_name="doi_standard",
):
    """Parse a SkyTEM rho xyz file and write point + optional flightline layers to GeoPackage."""
    xyz_path = Path(xyz_path)
    gpkg_path = Path(gpkg_path)

    if not xyz_path.is_file():
        raise FileNotFoundError(f"Input file not found: {xyz_path}")

    print(f"Reading {xyz_path}...", end=" ")
    df = read.parse_skytem_xyz(xyz_path)
    print(f"{len(df):,} rows")

    if bbox:
        n_before = len(df)
        df = clip_bbox(df, bbox)
        print(f"clip bbox {bbox}: {n_before:,} -> {len(df):,} rows")

    gdf_points = preproc_data.restructure(df, {"epsg": epsg})

    if apply_doi_clip:
        n_before = len(gdf_points)
        gdf_points = clip_doi(gdf_points, doi_name)
        print(f"clip DOI ({doi_name}): {n_before:,} -> {len(gdf_points):,} rows")

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()

    print(f"Writing {gpkg_path} layer {points_layer!r}...", end=" ")
    gdf_points.to_file(gpkg_path, layer=points_layer, driver="GPKG")
    print("done")

    if include_flightlines:
        gdf_lines = flightlines_from_xyz(df, epsg)
        print(f"Writing {gpkg_path} layer {flightlines_layer!r}...", end=" ")
        gdf_lines.to_file(gpkg_path, layer=flightlines_layer, driver="GPKG", mode="a")
        print(f"done ({len(gdf_lines):,} lines)")

    return gpkg_path
