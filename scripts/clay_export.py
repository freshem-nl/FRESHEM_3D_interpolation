"""Derive clay fraction from resistivity postproc and export IDF / NetCDF."""

from pathlib import Path

import xarray as xr

from scripts import idf_export

DEFAULT_A = 1.17
DEFAULT_B = -0.0163
DEFAULT_RHO_MIN = 5.0
DEFAULT_RHO_VAR = "Q(0.5)"
DEFAULT_PROPERTY = "clay"


def clay_from_rho(rho, a=DEFAULT_A, b=DEFAULT_B, rho_min=DEFAULT_RHO_MIN, clip_to_unit=True):
    """Linear clay regression; mask rho <= rho_min; optionally clip to [0, 1]."""
    clay = a + b * rho
    clay = clay.where(rho > rho_min)
    if clip_to_unit:
        clay = clay.clip(min=0.0, max=1.0)
    return clay


def build_clay_dataset(
    ds,
    rho_var=DEFAULT_RHO_VAR,
    a=DEFAULT_A,
    b=DEFAULT_B,
    rho_min=DEFAULT_RHO_MIN,
    clip_to_unit=True,
    property_name=DEFAULT_PROPERTY,
):
    """Return Dataset with clay, top, bottom (and source rho) for export."""
    for name in (rho_var, "top", "bottom"):
        if name not in ds:
            raise KeyError(f"Variable {name!r} not found in dataset")

    rho = ds[rho_var]
    clay = clay_from_rho(rho, a=a, b=b, rho_min=rho_min, clip_to_unit=clip_to_unit)
    clay.name = property_name
    clay.attrs.update(
        {
            "long_name": f"clay fraction from {rho_var} resistivity",
            "units": "1",
            "formula": f"clay = {a} + ({b}) * rho",
            "rho_var": rho_var,
            "rho_min_ohm_m": float(rho_min),
            "clipped_to_0_1": int(bool(clip_to_unit)),
        }
    )

    out = xr.Dataset(
        coords=ds.coords,
        data_vars={
            property_name: clay,
            "top": ds["top"],
            "bottom": ds["bottom"],
            rho_var: rho,
        },
        attrs=dict(ds.attrs),
    )
    out.attrs["clay_formula"] = clay.attrs["formula"]
    out.attrs["clay_rho_min_ohm_m"] = float(rho_min)
    out.attrs["clay_clipped_to_0_1"] = int(bool(clip_to_unit))
    return out


def export_clay(
    nc_path,
    dst_dir,
    *,
    rho_var=DEFAULT_RHO_VAR,
    a=DEFAULT_A,
    b=DEFAULT_B,
    rho_min=DEFAULT_RHO_MIN,
    clip_to_unit=True,
    property_name=DEFAULT_PROPERTY,
    write_idf=True,
    write_nc=True,
    nc_out=None,
    layers=None,
):
    """Derive clay from postproc NetCDF; write layer-coloured IDFs and/or clay NetCDF."""
    if not write_idf and not write_nc:
        raise ValueError("At least one of write_idf or write_nc must be true")

    nc_path = Path(nc_path)
    dst_dir = Path(dst_dir)

    ds = xr.open_dataset(nc_path)
    try:
        clay_ds = build_clay_dataset(
            ds,
            rho_var=rho_var,
            a=a,
            b=b,
            rho_min=rho_min,
            clip_to_unit=clip_to_unit,
            property_name=property_name,
        )
    finally:
        ds.close()

    result = {"dataset": clay_ds, "idf_dir": None, "nc_out": None}

    if write_idf:
        idf_dir = dst_dir / idf_export.var_token(property_name)
        idf_export.export_coloured_3d_model(clay_ds, idf_dir, property_name, layers)
        result["idf_dir"] = idf_dir

    if write_nc:
        if nc_out is None:
            nc_out = dst_dir / f"{nc_path.stem} - {property_name}.nc"
        else:
            nc_out = Path(nc_out)
        nc_out.parent.mkdir(parents=True, exist_ok=True)
        clay_ds.to_netcdf(nc_out)
        result["nc_out"] = nc_out

    return result
