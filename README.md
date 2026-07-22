# FRESHEM 3D interpolation

## NetCDF to iMOD IDF export

Standalone exporter for FRESHEM prediction NetCDF files.

1. Copy `export_idf/config.local.yaml.example` to `export_idf/config.local.yaml`
2. Set `paths.nc_file`, `paths.dst_dir`, `variables`, and `vertical_dim` (`z` or `layer`)
3. Run `export_idf/main.py`

Output: `dst_dir/{var_folder}/idx_000_{var}_NAP_-50_00.idf` (bulk voxel export, default)

## SkyTEM xyz to GeoPackage export

Standalone exporter for raw SkyTEM rho inversion xyz files (pre-pipeline inspection in QGIS).

1. Set `paths.xyz_file` and `paths.gpkg_file` in `export_gpkg/config.rho.yaml`
2. Run `python export_gpkg/main.py --config config.rho.yaml`

Output: GeoPackage with `rho_points` (one row per layer per measurement, with `z_top`/`z_bottom`) and `flightlines`.

## SkyTEM xyz to iMOD IPF export

Standalone exporter for raw SkyTEM rho inversion xyz files (borehole-style inspection in iMOD).

1. Set `paths.xyz_file` and `paths.ipf_file` in `export_ipf/config.rho.yaml`
2. Optionally set `clip_bbox` and/or `min_spacing_m` (along-line thinning)
3. Run `python export_ipf/main.py --config config.rho.yaml`

Output:
- `{name}.ipf` — sounding points; ID is `soundings\L{line}_{nnnn}` (Windows-style, like borehole IPFs)
- `soundings/L{line}_{nnnn}.txt` — associated 1D logs (`topnap`, `rho`, `rho_class`)
- `rho_freshem.dlf` — discrete Freshem rho legend for colouring `rho_class`
