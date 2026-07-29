from datetime import datetime
from pathlib import Path

import yaml


def load_config(path="config.yaml", data_input=None):

    path = Path(path)
    print(f"\nloading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_path = path.parent / "config.local.yaml"
    if local_path.exists():
        print(f"loading local overrides from {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg.update(local_cfg)

    if data_input is not None:
        cfg["data_input"] = data_input
        cfg["name"] = f'{cfg["name"]} - {Path(data_input).stem}'
    elif isinstance(cfg.get("data_input"), list):
        raise ValueError("data_input is a list; use load_configs()")

    if not cfg.get("dir_base") or not cfg.get("dir_input"):
        raise FileNotFoundError(
            f"dir_base and dir_input must be set in {local_path} " "(copy from config.local.yaml.example)"
        )

    t0 = datetime.today()

    # vars
    cfg["indicator_names"] = [f'P({cfg["variable_name"]}≤{x})' for x in cfg["indicators"]]
    cfg["quantile_names"] = [f"Q({q})" for q in cfg["quantiles"]]

    # paths
    cfg["dir_base"] = Path(cfg["dir_base"])
    cfg["dir_input"] = Path(cfg["dir_input"])
    run_name = (
        f'{t0.strftime("%Y%m%d")} - '
        f'{cfg["method"]}'
        f'{" with laf" if (cfg["method"] == "geostat" and cfg["use_anisotropy"]) else ""}'
        f'{" isotropic" if (cfg["method"] == "geostat" and not cfg["use_anisotropy"]) else ""}'
        f' - {cfg["variable_name"]} - {cfg["name"]}'
    )
    # run_name = f'{t0.strftime("%Y%m%d")} - {cfg["method"]} - {cfg["variable_name"]} - {cfg["name"]}'
    output_base = Path(cfg["dir_output_base"]) if cfg.get("dir_output_base") else cfg["dir_base"] / "output"
    cfg["dir_output"] = output_base / run_name
    cfg["dir_data"] = cfg["dir_output"] / "data"
    cfg["dir_plot"] = cfg["dir_output"] / "plots"
    cfg["dir_rasters"] = cfg["dir_output"] / "rasters"
    cfg["dir_xval"] = cfg["dir_output"] / "cross-validation"
    cfg["dir_idf"] = cfg["dir_output"] / "idf"
    cfg["dir_mdf"] = cfg["dir_output"] / "mdf"
    cfg["dir_leg"] = Path(__file__).resolve().parent.parent / "data" / "leg"
    if cfg.get("dir_imod"):
        cfg["dir_imod"] = Path(cfg["dir_imod"])
    cfg["path_input"] = cfg["dir_input"] / cfg["data_input"]

    # temp files
    input_name = Path(cfg["data_input"]).stem
    cfg["path_preproc_data"] = cfg["dir_data"] / f"{input_name} - preproc - data.parquet"
    cfg["path_preproc_data_gridded"] = cfg["dir_data"] / f"{input_name} - preproc - data gridded.nc"
    cfg["path_preproc_prediction_grid"] = cfg["dir_data"] / f"{input_name} - preproc - prediction grid.nc"
    cfg["path_preproc_data_flightlines"] = cfg["dir_data"] / f"{input_name} - preproc - data - flightlines.parquet"
    cfg["path_depths"] = cfg["dir_data"] / f"{input_name} - layer depths.nc"
    cfg["path_prediction"] = cfg["dir_data"] / f"{input_name} - {cfg['method']} - {cfg['variable_name']}.nc"
    cfg["path_prediction_xval"] = cfg["dir_data"] / f"{input_name} - xval.nc"
    cfg["path_postproc"] = cfg["dir_data"] / f"{input_name} - postproc.nc"
    cfg["path_data_anisotropy"] = cfg["dir_data"] / f"{input_name} - preproc - data - anisotropy.nc"

    return cfg


def load_configs(path="config.yaml"):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data_inputs = yaml.safe_load(f)["data_input"]

    return [load_config(path, data_input) for data_input in data_inputs]
