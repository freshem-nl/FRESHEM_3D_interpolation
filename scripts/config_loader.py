from datetime import datetime
from pathlib import Path

import yaml


def load_config(path="config.yaml"):

    print(f"\nloading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    t0 = datetime.today()

    # paths
    cfg["dir_base"] = Path(cfg["dir_base"])
    cfg["dir_input"] = cfg["dir_base"] / cfg["dir_input"]
    cfg["dir_output"] = cfg["dir_base"] / "output" / f'{t0.strftime("%Y%m%d")} - {cfg["name"]}'
    cfg["dir_data"] = cfg["dir_output"] / "data"
    cfg["dir_plot"] = cfg["dir_output"] / "plots"
    cfg["dir_rasters"] = cfg["dir_output"] / "rasters"
    cfg["dir_xval"] = cfg["dir_output"] / "cross-validation"
    cfg["path_input"] = cfg["dir_input"] / cfg["data_input"]

    # temp files
    cfg["path_preproc_data"] = cfg["dir_data"] / "preproc - data.nc"
    cfg["path_preproc_data_gridded"] = cfg["dir_data"] / "preproc - data gridded.nc"
    cfg["path_preproc_data_flightlines"] = cfg["dir_data"] / "preproc - data - flightlines.parquet"
    cfg["path_prediction"] = cfg["dir_data"] / "pred.nc"
    cfg["path_prediction_xval"] = cfg["dir_data"] / "xval.nc"
    cfg["path_postproc"] = cfg["dir_data"] / "postproc.nc"
    
    return cfg