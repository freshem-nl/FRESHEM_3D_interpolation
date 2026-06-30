import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import idf_export


def load_config():

    config_dir = Path(__file__).parent
    path = config_dir / "config.yaml"
    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_path = config_dir / "config.local.yaml"
    if local_path.exists():
        print(f"loading local overrides from {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})

    if not cfg["paths"].get("nc_file") or not cfg["paths"].get("dst_dir"):
        raise FileNotFoundError(
            f"paths.nc_file and paths.dst_dir must be set in {local_path} "
            "(copy from config.local.yaml.example)"
        )

    return cfg


def main(cfg):

    nc_path = Path(cfg["paths"]["nc_file"])
    dst_dir = Path(cfg["paths"]["dst_dir"])

    print(f"\nexport IDF from {nc_path}...", end=" ")
    idf_export.export_netcdf(
        nc_path,
        dst_dir,
        cfg["variables"],
        cfg["vertical_dim"],
        cfg.get("dim_mapping", {}),
    )
    print(f"done.\n...output in {dst_dir}\n")


if __name__ == "__main__":

    cfg = load_config()
    main(cfg)
