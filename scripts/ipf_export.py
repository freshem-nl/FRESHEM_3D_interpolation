"""SkyTEM xyz -> iMOD IPF (point + associated 1D logs) + optional DLF legend."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gpkg_export import clip_bbox, clip_doi, parse_skytem_xyz, z_doi_column

_QML_DIR = Path(__file__).resolve().parent.parent / "export_gpkg" / "qml"
if str(_QML_DIR) not in sys.path:
    sys.path.insert(0, str(_QML_DIR))

from rho_colormap import rho_freshem_classes, rho_to_class_index, write_rho_freshem_dlf  # noqa: E402

_SOUNDING_TXT_RE = re.compile(r"^L.+_\d{4}\.txt$")


def thin_min_spacing(df: pd.DataFrame, min_spacing_m) -> pd.DataFrame:
    """Keep soundings at least min_spacing_m apart along each flight line (file order)."""
    if not min_spacing_m or min_spacing_m <= 0:
        return df

    keep_idx = []
    for _, group in df.groupby("line_no", sort=False):
        last_x = last_y = None
        for idx, row in group.iterrows():
            if last_x is None:
                keep_idx.append(idx)
                last_x, last_y = float(row["x"]), float(row["y"])
                continue
            dist = float(np.hypot(row["x"] - last_x, row["y"] - last_y))
            if dist >= min_spacing_m:
                keep_idx.append(idx)
                last_x, last_y = float(row["x"]), float(row["y"])

    return df.loc[keep_idx].reset_index(drop=True)


def _line_tag(line_no) -> str:
    if pd.isna(line_no):
        return "NA"
    value = float(line_no)
    if value == int(value):
        return str(int(value))
    return str(line_no)


def assign_sounding_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign L{line}_{nnnn} IDs sequential within each flight line (file order)."""
    out = df.copy()
    counters: dict[str, int] = {}
    ids = []
    for line_no in out["line_no"]:
        tag = _line_tag(line_no)
        counters[tag] = counters.get(tag, 0) + 1
        ids.append(f"L{tag}_{counters[tag]:04d}")
    out["sounding_id"] = ids
    return out


def resolve_package_ipf_path(ipf_path: Path) -> Path:
    """Place IPF in a same-named package folder so TXT can sit beside it without a path prefix."""
    ipf_path = Path(ipf_path)
    if ipf_path.parent.name == ipf_path.stem:
        return ipf_path
    return ipf_path.parent / ipf_path.stem / ipf_path.name


def clear_sounding_txt(package_dir: Path) -> None:
    """Remove previously exported sounding TXT files in the package folder."""
    if not package_dir.is_dir():
        return
    for path in package_dir.iterdir():
        if path.is_file() and _SOUNDING_TXT_RE.match(path.name):
            path.unlink()


def layers_long(df: pd.DataFrame) -> pd.DataFrame:
    """Expand wide SkyTEM soundings to one row per layer, keeping sounding_id."""
    layer_numbers = [
        int(col.split("_")[1])
        for col in df.columns
        if col.startswith("rho_") and "std" not in col
    ]
    if not layer_numbers:
        raise ValueError("No RHO_* layer columns found in SkyTEM table.")

    frames = []
    for i in layer_numbers:
        part = df.copy()
        part = part.rename(
            columns={
                f"rho_{i}": "rho",
                f"rho_std{i}": "rho_std",
                f"thk_{i}": "thickness",
            }
        )
        part["z_top"] = part["elevation"] - part[f"dep_top_{i}"]
        part["z_bottom"] = part["elevation"] - part[f"dep_bot_{i}"]
        part["z_doi_standard"] = part["elevation"] - part["doi_standard"]
        part["z_doi_conservative"] = part["elevation"] - part["doi_conservative"]
        part["layer"] = i
        frames.append(
            part[
                [
                    "sounding_id",
                    "line_no",
                    "x",
                    "y",
                    "elevation",
                    "layer",
                    "rho",
                    "rho_std",
                    "z_top",
                    "z_bottom",
                    "thickness",
                    "z_doi_standard",
                    "z_doi_conservative",
                    "doi_standard",
                    "doi_conservative",
                ]
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["sounding_id", "layer"]).reset_index(drop=True)


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "-999.99"
        return f"{float(value):.6g}"
    return str(value)


def write_associated_txt(path: Path, layers: pd.DataFrame, classes) -> None:
    """Write one iMOD associated TXT (itype=2) for a sounding."""
    rows = []
    for _, row in layers.iterrows():
        rho = row["rho"]
        if pd.isna(rho):
            continue
        cls = rho_to_class_index(float(rho), classes)
        rows.append((_fmt(row["z_top"]), _fmt(rho), str(cls)))

    if not rows:
        raise ValueError(f"No valid rho layers for {path.name}")

    z_end = layers["z_bottom"].iloc[-1]
    rows.append((_fmt(z_end), "end", "-"))

    lines = [
        str(len(rows)),
        "3,2",
        '"topnap",-999.99',
        '"rho",-999.99',
        '"rho_class",-999.99',
    ]
    lines.extend(f"{top},{rho},{cls}" for top, rho, cls in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ipf(path: Path, points: pd.DataFrame, assoc_col: int = 3) -> None:
    """Write the main IPF pointing at associated TXT files."""
    headers = [
        '"X_CRD (m)"',
        '"Y_CRD (m)"',
        '"ID"',
        '"ELEVATION (m+nap)"',
        '"DOI (m-MV)"',
        '"LINE_NO"',
    ]
    lines = [
        str(len(points)),
        str(len(headers)),
        *headers,
        f"{assoc_col},txt",
    ]
    for _, row in points.iterrows():
        lines.append(
            ",".join(
                [
                    _fmt(row["x"]),
                    _fmt(row["y"]),
                    str(row["id"]),
                    _fmt(row["elevation"]),
                    _fmt(row["doi"]),
                    _fmt(row["line_no"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_rho_ipf(
    xyz_path,
    ipf_path,
    *,
    bbox=None,
    min_spacing_m=None,
    apply_doi_clip=True,
    doi_name="doi_standard",
    write_dlf=True,
    dlf_name="rho_freshem.dlf",
):
    """Parse SkyTEM rho xyz and write an iMOD IPF + associated TXT logs.

    Layout: ``{name}/{name}.ipf`` with ``L{line}_{nnnn}.txt`` beside the IPF
    so iMOD IDs are bare sounding names (no folder prefix).
    """
    xyz_path = Path(xyz_path)
    ipf_path = resolve_package_ipf_path(Path(ipf_path))
    package_dir = ipf_path.parent

    if not xyz_path.is_file():
        raise FileNotFoundError(f"Input file not found: {xyz_path}")

    print(f"Reading {xyz_path}...", end=" ")
    df = parse_skytem_xyz(xyz_path)
    print(f"{len(df):,} soundings")

    if bbox:
        n_before = len(df)
        df = clip_bbox(df, bbox)
        print(f"clip bbox {bbox}: {n_before:,} -> {len(df):,} soundings")

    if min_spacing_m:
        n_before = len(df)
        df = thin_min_spacing(df, min_spacing_m)
        print(
            f"thin min_spacing {min_spacing_m:g} m: {n_before:,} -> {len(df):,} soundings"
        )

    if df.empty:
        raise ValueError("No soundings left after clipping.")

    df = assign_sounding_ids(df.reset_index(drop=True))
    sounding_order = list(dict.fromkeys(df["sounding_id"]))

    long = layers_long(df)
    if apply_doi_clip:
        n_before = len(long)
        long = clip_doi(long, doi_name)
        print(f"clip DOI ({doi_name}): {n_before:,} -> {len(long):,} layer rows")

    if long.empty:
        raise ValueError("No layers left after DOI clipping.")

    package_dir.mkdir(parents=True, exist_ok=True)
    clear_sounding_txt(package_dir)

    classes = rho_freshem_classes()
    z_doi = z_doi_column(doi_name)
    doi_depth_col = doi_name if doi_name in df.columns else doi_name.replace("z_", "")

    point_rows = []
    n_written = 0
    for sounding_id in sounding_order:
        group = long.loc[long["sounding_id"] == sounding_id]
        if group.empty:
            continue
        group = group.sort_values("layer")
        txt_path = package_dir / f"{sounding_id}.txt"
        write_associated_txt(txt_path, group, classes)
        n_written += 1

        first = group.iloc[0]
        if doi_depth_col in group.columns:
            doi_mv = first[doi_depth_col]
        else:
            doi_mv = first["elevation"] - first[z_doi]

        point_rows.append(
            {
                "x": first["x"],
                "y": first["y"],
                "id": sounding_id,
                "elevation": first["elevation"],
                "doi": doi_mv,
                "line_no": first["line_no"],
            }
        )

    points = pd.DataFrame(point_rows)
    write_ipf(ipf_path, points)
    print(f"Wrote {ipf_path} ({len(points):,} points, {n_written:,} associated TXT)")

    dlf_path = None
    if write_dlf:
        dlf_path = ipf_path.with_name(dlf_name)
        write_rho_freshem_dlf(dlf_path)
        print(f"Wrote {dlf_path}")

    return {"ipf": ipf_path, "associated_dir": package_dir, "dlf": dlf_path}
