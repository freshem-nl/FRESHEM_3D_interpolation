import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ipf_export

CONFIG_DIR = Path(__file__).parent


def load_config(config_name):

    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg["paths"].get("xyz_file") or not cfg["paths"].get("ipf_file"):
        raise FileNotFoundError(
            f"paths.xyz_file and paths.ipf_file must be set in {path}"
        )

    return cfg


def main(cfg):

    xyz_path = Path(cfg["paths"]["xyz_file"])
    ipf_path = Path(cfg["paths"]["ipf_file"])

    print(f"\nexport IPF from {xyz_path}...")
    result = ipf_export.export_rho_ipf(
        xyz_path,
        ipf_path,
        bbox=cfg.get("clip_bbox") or None,
        min_spacing_m=cfg.get("min_spacing_m"),
        apply_doi_clip=cfg.get("apply_doi_clip", True),
        doi_name=cfg.get("doi_name", "doi_standard"),
        associated_dirname=cfg.get("associated_dirname"),
        write_dlf=cfg.get("write_dlf", False),
        dlf_name=cfg.get("dlf_name", "rho_freshem.dlf"),
    )
    print(f"done.\n...output in {result['ipf']}\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Export SkyTEM rho xyz files to iMOD IPF for borehole-style inspection."
    )
    ap.add_argument(
        "--config",
        default="config.rho.yaml",
        help="Config file in export_ipf/ (default: config.rho.yaml)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
