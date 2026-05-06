import yaml
from pathlib import Path
from datetime import datetime

def load_config(path="config.yaml"):

    print(f"\nloading config from {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    t0 = datetime.today()

    #paths
    config["dir_base"] = Path(config["dir_base"])
    config["dir_input"] = config["dir_base"] / config["dir_input"]
    config["dir_output"] = config["dir_base"] / "output" / f'{t0.strftime("%Y%m%d")} - {config["name"]}'

    config["path_input"] = config["dir_input"] / config["data_input"]

    return config