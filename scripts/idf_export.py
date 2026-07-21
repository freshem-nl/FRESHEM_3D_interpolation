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


def _coloured_3d_filename(seq, layer_value, role, property_token):
    layer = int(layer_value)
    if role == "top":
        return f"{seq:03d}_layer{layer:02d}_top.idf"
    if role == "bottom":
        return f"{seq:03d}_layer{layer:02d}_bottom.idf"
    return f"{seq:03d}_layer{layer:02d}_{property_token}.idf"


def export_coloured_3d_model(ds, dst_dir, property_name, layers=None):
    """Export top, property, bottom IDFs per layer in iMOD Coloured 3-D Model load order."""
    out_dir = Path(dst_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for var_name in ("top", property_name, "bottom"):
        if var_name not in ds:
            raise KeyError(f"Variable {var_name!r} not found in dataset")

    layer_values = ds["layer"].values
    if layers is not None:
        layer_set = {int(layer) for layer in layers}
        layer_values = [value for value in layer_values if int(value) in layer_set]

    property_token = var_token(property_name)
    manifest_lines = [
        "# iMOD Coloured 3-D Model — load IDFs in this order",
        "# Per layer: top, property, bottom",
        "",
    ]
    seq = 1

    for layer_value in layer_values:
        for role, var_name in (
            ("top", "top"),
            ("property", property_name),
            ("bottom", "bottom"),
        ):
            slab = (
                ds[var_name]
                .sel({"layer": layer_value})
                .squeeze(drop=True)
                .astype("float32")
            )
            if "layer" in slab.coords:
                slab = slab.drop_vars("layer")

            fname = _coloured_3d_filename(seq, layer_value, role, property_token)
            imod.idf.save(out_dir / fname, slab)
            manifest_lines.append(fname)
            seq += 1

    manifest_path = out_dir / "imod_load_order.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def export_layer_coloured(nc_path, dst_dir, properties, layers=None):
    """Export one iMOD Coloured 3-D Model package per layer-model property."""
    if not properties:
        raise ValueError("properties must contain at least one variable name")

    ds = xr.open_dataset(nc_path)
    try:
        for property_name in properties:
            property_dir = Path(dst_dir) / var_token(property_name)
            export_coloured_3d_model(ds, property_dir, property_name, layers)
    finally:
        ds.close()


def export_voxel_bulk(nc_path, dst_dir, variables, dim_mapping=None, export_cfg=None):
    """Export voxel-model variables to one IDF stack per variable."""
    if not variables:
        raise ValueError("variables must contain at least one variable name")

    export_cfg = export_cfg or {}
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

            export_bulk(da, out_dir, var_name, filename_template, z_format, z_offset)
    finally:
        ds.close()
