import re
from pathlib import Path

import imod
import xarray as xr


def var_folder(name):
    m = re.match(r"P\((\w+)≤(\d+(?:\.\d+)?)\)", name)
    if m:
        return f"P_{m.group(1)}_le_{m.group(2).replace('.', '_')}"
    return re.sub(r"[^\w]+", "_", name).strip("_")


def export_netcdf(nc_path, dst_dir, variables, vertical_dim, dim_mapping=None, z_offset=0.0):

    ds = xr.open_dataset(nc_path)
    try:
        for var_name in variables:
            da = ds[var_name]
            dim_mapping_var = {old: new for old, new in (dim_mapping or {}).items() if old in da.dims}
            if dim_mapping_var:
                da = da.rename(dim_mapping_var)

            out_dir = Path(dst_dir) / var_folder(var_name)
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, value in enumerate(da[vertical_dim].values):
                slab = da.sel({vertical_dim: value}).squeeze(drop=True).astype("float32")
                if vertical_dim in slab.coords:
                    slab = slab.drop_vars(vertical_dim)

                if vertical_dim == "layer":
                    fname = f"idx_{idx:03d}_layer_{int(value):02d}.idf"
                else:
                    z = format(float(value) + z_offset, ".2f").replace(".", "_")
                    fname = f"idx_{idx:03d}_NAP_{z}.idf"

                imod.idf.save(out_dir / fname, slab)
    finally:
        ds.close()
