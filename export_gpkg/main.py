import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpkg_export

CONFIG_DIR = Path(__file__).parent


def load_config(config_name):

    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg["paths"].get("xyz_file") or not cfg["paths"].get("gpkg_file"):
        raise FileNotFoundError(
            f"paths.xyz_file and paths.gpkg_file must be set in {path}"
        )

    return cfg


def main(cfg):

    xyz_path = Path(cfg["paths"]["xyz_file"])
    gpkg_path = Path(cfg["paths"]["gpkg_file"])

    print(f"\nexport GeoPackage from {xyz_path}...")
    gpkg_export.export_rho_xyz(
        xyz_path,
        gpkg_path,
        epsg=cfg.get("epsg", 28992),
        include_flightlines=cfg.get("include_flightlines", True),
        points_layer=cfg.get("points_layer", "rho_points"),
        flightlines_layer=cfg.get("flightlines_layer", "flightlines"),
        bbox=cfg.get("clip_bbox") or None,
        apply_doi_clip=cfg.get("apply_doi_clip", True),
        doi_name=cfg.get("doi_name", "doi_standard"),
    )
    print(f"done.\n...output in {gpkg_path}\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Export SkyTEM rho xyz files to GeoPackage for QGIS inspection."
    )
    ap.add_argument(
        "--config",
        default="config.rho.yaml",
        help="Config file in export_gpkg/ (default: config.rho.yaml)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
