from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from scripts import preproc_data


def parse_skytem_xyz(path_input):
    """Parse a SkyTEM inversion xyz export, handling AGS (/ LINE_NO) and #HEADERS styles."""
    path_input = Path(path_input)
    lines = path_input.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_line = None
    data_lines: list[str] = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("/ LINE_NO"):
            header_line = line[1:].strip()
            for tail in lines[idx + 1 :]:
                stripped = tail.strip()
                if stripped and not stripped.startswith("/") and not stripped.startswith("#"):
                    data_lines.append(stripped)
            break

        if line.upper().startswith("#HEADERS"):
            header_line = line.replace("#HEADERS", "").strip()
            continue

        if header_line and line.upper().startswith("#DATA"):
            data_lines.append(line.replace("#DATA", "").strip())
            continue

        if header_line and not line.startswith("/") and not line.startswith("#"):
            data_lines.append(line)

    if header_line is None:
        for idx, raw_line in enumerate(lines):
            stripped = raw_line.strip().lstrip("/").lstrip("#").strip()
            if "LINE_NO" in stripped and ("RHO_" in stripped or "SIGMA_" in stripped):
                header_line = stripped
                for tail in lines[idx + 1 :]:
                    entry = tail.strip()
                    if entry and not entry.startswith("/") and not entry.startswith("#"):
                        data_lines.append(entry)
                break

    if header_line is None:
        raise ValueError("Unable to find column headers containing LINE_NO and RHO_/SIGMA_ information.")

    columns = [col.strip() for col in header_line.split() if col.strip()]
    rows: list[list[str]] = []
    for entry in data_lines:
        values = entry.split()
        if len(values) == len(columns):
            rows.append(values)

    if not rows:
        raise ValueError(f"No data rows found in {path_input}")

    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([-9999, 9999], np.nan)

    if "UTMX" in df.columns and "X" not in df.columns:
        df["X"] = df["UTMX"]
    if "UTMY" in df.columns and "Y" not in df.columns:
        df["Y"] = df["UTMY"]

    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("Input file must contain X/Y or UTMX/UTMY coordinates.")

    df.columns = [x.lower() for x in df.columns]
    return df


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
    df = parse_skytem_xyz(xyz_path)
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
