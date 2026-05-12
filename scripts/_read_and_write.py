from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump, load

def write_dataset(ds, path):

    path = path.with_suffix(".nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)

# def save_dataset_for_arcgis(ds, path):
#     ds = ds.transpose("z", "y", "x")
    
#     path = path.with_suffix(".nc")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     ds.to_netcdf(path)

def read_dataset(path):
    path = path.with_suffix(".nc")
    return xr.open_dataset(path)

def write_table(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        data.to_parquet(path)
        # df.to_parquet(path, engine="fastparquet")

def read_table(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)    


def read_skytem_xyz(cfg):
    """Parse the SkyTEM inversion export, handling AGS (/ LINE_NO) and #HEADERS styles."""
    t0 = datetime.now()

    path_input = cfg["path_input"]
    dir_output = cfg["dir_output"]
    path_output = (dir_output / path_input.stem).with_suffix(".parquet")

    if path_output.exists():
        print(f"Reading {path_output}...", end=" ")
        df = read_table(path_output)
        print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")
        return df
    else:

        print(f"Reading {path_input}...", end=" ")

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
                    if (
                        stripped
                        and not stripped.startswith("/")
                        and not stripped.startswith("#")
                    ):
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
                if "LINE_NO" in stripped and (
                    "RHO_" in stripped or "SIGMA_" in stripped
                ):
                    header_line = stripped
                    for tail in lines[idx + 1 :]:
                        entry = tail.strip()
                        if (
                            entry
                            and not entry.startswith("/")
                            and not entry.startswith("#")
                        ):
                            data_lines.append(entry)
                    break

        if header_line is None:
            raise ValueError(
                "Unable to find column headers containing LINE_NO and RHO_/SIGMA_ information."
            )

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

        path_output.parent.mkdir(parents=True, exist_ok=True)
        write_table(df, path_output)

        txt = f"done ({(datetime.now() - t0).total_seconds():.2f}s). Read {len(df)} rows with {len(df.columns)} columns."
        print(txt)

    return df
