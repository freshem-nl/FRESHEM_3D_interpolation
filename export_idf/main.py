import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import idf_export

CONFIG_DIR = Path(__file__).parent


def load_config(config_name):

    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg["paths"].get("nc_file") or not cfg["paths"].get("dst_dir"):
        raise FileNotFoundError(
            f"paths.nc_file and paths.dst_dir must be set in {path}"
        )

    return cfg


def main(cfg):

    nc_path = Path(cfg["paths"]["nc_file"])
    dst_dir = Path(cfg["paths"]["dst_dir"])
    kind = cfg.get("kind")

    print(f"\nexport IDF from {nc_path}...", end=" ")
    if kind == "layer-coloured":
        idf_export.export_layer_coloured(
            nc_path,
            dst_dir,
            cfg["properties"],
            cfg.get("layers"),
        )
    elif kind == "voxel-bulk":
        idf_export.export_voxel_bulk(
            nc_path,
            dst_dir,
            cfg["variables"],
            cfg.get("dim_mapping", {}),
            cfg.get("export", {}),
        )
    else:
        raise ValueError("Config kind must be 'layer-coloured' or 'voxel-bulk'")
    print(f"done.\n...output in {dst_dir}\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Export NetCDF variables to iMOD IDF files.")
    ap.add_argument(
        "--config",
        default="config.layer.yaml",
        help="Config file in export_idf/ (config.voxel.yaml or config.layer.yaml)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
