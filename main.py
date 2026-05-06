from scripts import config_loader, readers
from pathlib import Path
import os
from scripts import readers, preprocessing

def main(cfg):

    def preproc():
        df = readers.read_skytem_xyz(cfg["path_input"])
        ds = preprocessing.initiate_dataset(df, cfg)
        measurements_gridded = preprocessing.snap_measurements_to_grid(df, ds)
        preprocessing.quantiles_per_voxel(measurements_gridded, ds, cfg)
        preprocessing.flightlines_per_voxel(measurements_gridded, cfg)

    def anisotropy():
        pass

    preproc()
    anisotropy()

if __name__ == "__main__":

    cfg = config_loader.load_config(Path(os.getcwd()) / "config.yaml")
    main(cfg)
