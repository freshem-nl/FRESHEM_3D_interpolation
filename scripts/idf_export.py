import re
import shutil
import tempfile
from pathlib import Path

import imod
import xarray as xr


def var_folder(name):
    m = re.match(r"P\((\w+)≤(\d+(?:\.\d+)?)\)", name)
    if m:
        return f"P_{m.group(1)}_le_{m.group(2).replace('.', '_')}"
    return re.sub(r"[^\w]+", "_", name).strip("_")


def var_token(name):
    return var_folder(name)


def _format_z(z, z_format, z_offset):
    raw_z = float(z) + z_offset
    fmt_z = format(raw_z, z_format)
    return fmt_z.replace(".", "_")


def export_bulk(da, out_dir, var_name, filename_template, z_format, z_offset):
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp.idf"
        imod.idf.save(temp_path, da)

        generated_files = sorted(Path(temp_dir).glob("*.idf"))
        z_values = da["z"].values
        var = var_token(var_name)

        for idx, generated_file in enumerate(generated_files):
            if idx >= len(z_values):
                break
            z = _format_z(z_values[idx], z_format, z_offset)
            fname = filename_template.format(var=var, z=z, idx=idx)
            shutil.copy2(generated_file, out_dir / fname)


def export_per_layer(da, out_dir, var_name, vertical_dim, filename_template, z_format, z_offset):
    out_dir = Path(out_dir)
    var = var_token(var_name)

    for idx, value in enumerate(da[vertical_dim].values):
        slab = da.sel({vertical_dim: value}).squeeze(drop=True).astype("float32")
        if vertical_dim in slab.coords:
            slab = slab.drop_vars(vertical_dim)

        if vertical_dim == "layer":
            fname = f"idx_{idx:03d}_layer_{int(value):02d}.idf"
        else:
            z = _format_z(value, z_format, z_offset)
            fname = filename_template.format(var=var, z=z, idx=idx)

        imod.idf.save(out_dir / fname, slab)


def export_netcdf(
    nc_path,
    dst_dir,
    variables,
    vertical_dim,
    dim_mapping=None,
    export_cfg=None,
):
    export_cfg = export_cfg or {}
    mode = export_cfg.get("mode", "bulk")
    filename_template = export_cfg.get(
        "filename_template", "idx_{idx:03d}_{var}_NAP_{z}.idf"
    )
    z_format = export_cfg.get("z_format", ".2f")
    z_offset = float(export_cfg.get("z_offset", -0.25))

    ds = xr.open_dataset(nc_path)
    try:
        for var_name in variables:
            da = ds[var_name]
            dim_mapping_var = {
                old: new for old, new in (dim_mapping or {}).items() if old in da.dims
            }
            if dim_mapping_var:
                da = da.rename(dim_mapping_var)

            out_dir = Path(dst_dir) / var_folder(var_name)
            out_dir.mkdir(parents=True, exist_ok=True)
            da = da.astype("float32")

            if mode == "bulk" and vertical_dim == "z":
                export_bulk(da, out_dir, var_name, filename_template, z_format, z_offset)
            else:
                export_per_layer(
                    da, out_dir, var_name, vertical_dim, filename_template, z_format, z_offset
                )
    finally:
        ds.close()
