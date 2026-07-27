# FRESHEM 3D interpolation

## Modelling pipeline

1. Copy `config.local.yaml.example` to `config.local.yaml` and set `dir_base`, `dir_input`, and `dir_imod`
2. Run `python main.py` from the repo root

With `export.idf.enabled: true`, postprocessing writes layer-coloured IDFs under `{dir_output}/idf/{property}/` (from `postproc.nc`). With `export.mdf.enabled: true`, MDFs follow under `{dir_output}/mdf/{property}_{input}.mdf`, embedding the chosen legend from `data/leg/`. MDF paths point to `{dir_imod}/idf/{property}/`, so copy `idf/` and `mdf/` from the run output into the iMOD project with the same relative layout.

## NetCDF to iMOD IDF export (standalone)

Standalone exporter for FRESHEM prediction NetCDF files (re-export without re-running the model).

1. Set `paths.nc_file` and `paths.dst_dir` in `export_idf/config.layer.yaml` (or `config.voxel.yaml`)
2. Run `python export_idf/main.py --config config.layer.yaml`

Output (layer-coloured): `dst_dir/{property}/NNN_layerLL_{top|prop|bottom}.idf`

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
- `{name}.ipf` — sounding points; ID is `{name}\L{line}_{nnnn}` (Windows-style, like borehole IPFs)
- `{name}/L{line}_{nnnn}.txt` — associated 1D logs (`topnap`, `rho`, `rho_ohm`)

`rho_ohm` is a 1 Ohm.m colour code (`round(rho)`, clipped to 1..150) for iMOD DLF matching.
Generate matching DLFs with `python snippets_chris/scripts/write_rho_1ohm_dlf.py`
(`snippets_chris/example_imod/dlf/rho_zoet_1ohm.dlf`, `rho_zout_1ohm.dlf`). Colour the
associated log on column `rho_ohm`.
