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
