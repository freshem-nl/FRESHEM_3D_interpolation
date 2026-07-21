"""Combine per-area FRESHEM postproc NetCDF files onto one RD-aligned grid."""

from pathlib import Path

import numpy as np
import xarray as xr

from scripts import write


def combine(source_paths, output_path):
    """Combine NetCDFs; later paths overwrite earlier ones where both have data."""
    datasets = [xr.open_dataset(path) for path in source_paths]
    try:
        x = np.sort(np.unique(np.concatenate([ds["x"].values for ds in datasets])))
        y = np.sort(np.unique(np.concatenate([ds["y"].values for ds in datasets])))

        out = None
        for ds in datasets:
            piece = ds.reindex(x=x, y=y)
            out = piece if out is None else piece.combine_first(out)

        write.dataset(out, Path(output_path))
        return out
    finally:
        for ds in datasets:
            ds.close()
