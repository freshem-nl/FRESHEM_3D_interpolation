import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import netcdf_combine

CONFIG_DIR = Path(__file__).parent


def load_config(config_name):

    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("sources"):
        raise ValueError(f"sources must be set in {path}")
    if not cfg.get("paths", {}).get("output_nc"):
        raise ValueError(f"paths.output_nc must be set in {path}")

    return cfg


def main(cfg):

    source_paths = [Path(p) for p in cfg["sources"]]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source NetCDF not found: {path}")

    output_path = Path(cfg["paths"]["output_nc"])

    print(f"\ncombine {len(source_paths)} NetCDF files...", end=" ")
    netcdf_combine.combine(source_paths, output_path)
    print("done.")
    print(f"...output in {output_path.with_suffix('.nc')}\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Combine per-area FRESHEM postproc NetCDF files."
    )
    ap.add_argument(
        "--config",
        default="config.yaml",
        help="Config file in combine_nc/ (default: config.yaml)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
