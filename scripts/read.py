from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from scripts import _utils, write


def table(path):

    if path.suffix == ".parquet":
        try:
            # First try reading with geopandas, which can handle geospatial metadata if present
            data = gpd.read_parquet(path)
        except Exception:
            data = pd.read_parquet(path)
    if path.suffix == ".csv":
        data = pd.read_csv(path)

    return data


def dataset(path):
    path = path.with_suffix(".nc")
    ds = xr.open_dataset(path)
    return ds

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


def skytem_xyz(cfg):
    """Read SkyTEM xyz for the main pipeline, using a parquet cache when present."""
    t0 = datetime.now()
    print("\nPREPROCESSING DATA")

    path_input = cfg["path_input"]
    dir_data = cfg["dir_data"]
    path_output = (dir_data / path_input.stem).with_suffix(".parquet")

    if path_output.exists():
        print(f"Reading {path_output}...", end=" ")
        df = table(path_output)
        print(f"({(datetime.now() - t0).total_seconds():.2f}s)")
        return df

    print(f"Reading {path_input}...", end=" ")
    df = parse_skytem_xyz(path_input)

    path_output.parent.mkdir(parents=True, exist_ok=True)
    write.table(df, path_output)
    write.table(df, path_output.with_suffix(".csv"))

    txt = (
        f"({(datetime.now() - t0).total_seconds():.2f}s). Read {len(df)} rows with {len(df.columns)} columns"
    )
    print(txt)

    return df

def deltares_cl(cfg):

    # from config
    path_in = cfg["path_input"]
    epsg = cfg["epsg"]

    data = pd.read_feather(path_in)
    data = data.dropna()

    data = _utils.df_to_gdf(data, epsg=epsg)

    return data
