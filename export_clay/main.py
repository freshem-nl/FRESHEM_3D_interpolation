import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import clay_export

CONFIG_DIR = Path(__file__).parent


def load_config(config_name):

    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("paths") or not cfg["paths"].get("nc_file") or not cfg["paths"].get("dst_dir"):
        raise FileNotFoundError(
            f"paths.nc_file and paths.dst_dir must be set in {path}"
        )

    return cfg


def main(cfg):

    paths = cfg["paths"]
    formula = cfg.get("formula") or {}
    nc_path = Path(paths["nc_file"])
    dst_dir = Path(paths["dst_dir"])

    print(f"\nderive clay from {nc_path}...")
    result = clay_export.export_clay(
        nc_path,
        dst_dir,
        rho_var=cfg.get("rho_var", clay_export.DEFAULT_RHO_VAR),
        a=float(formula.get("a", clay_export.DEFAULT_A)),
        b=float(formula.get("b", clay_export.DEFAULT_B)),
        rho_min=float(cfg.get("rho_min", clay_export.DEFAULT_RHO_MIN)),
        clip_to_unit=bool(cfg.get("clip_to_unit", True)),
        property_name=cfg.get("property_name", clay_export.DEFAULT_PROPERTY),
        write_idf=bool(cfg.get("write_idf", True)),
        write_nc=bool(cfg.get("write_nc", True)),
        nc_out=paths.get("nc_out"),
        layers=cfg.get("layers"),
    )
    if result["idf_dir"] is not None:
        print(f"IDF: {result['idf_dir']}")
    if result["nc_out"] is not None:
        print(f"NetCDF: {result['nc_out']}")
    print("done.\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Derive clay from resistivity postproc and export IDF / NetCDF."
    )
    ap.add_argument(
        "--config",
        default="config.zuid_a1.yaml",
        help="Config file in export_clay/ (default: config.zuid_a1.yaml)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
