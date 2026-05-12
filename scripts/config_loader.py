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
    cfg["dir_plot"] = cfg["dir_output"] / "plots"

    cfg["path_input"] = cfg["dir_input"] / cfg["data_input"]

    # temp files
    cfg["path_preproc_data"] = cfg["dir_output"] / "preproc - data.nc"
    cfg["path_preproc_data_gridded"] = cfg["dir_output"] / "preproc - data gridded.nc"
    cfg["path_preproc_data_flightlines"] = cfg["dir_output"] / "preproc - data - flightlines.parquet"
    cfg["path_prediction"] = cfg["dir_output"] / "prediction.nc"

    return cfg