# Freshem rho colour scale (iMOD + QGIS)

Shared palette for resistivity: log scale 0.01–150 Ωm, `RdYlBu_r`, gamma 1.5 (more blue at low rho).

## QGIS — vector points (SkyTEM xyz gpkg)

Use with GeoPackages from `export_gpkg/` (`rho_points` layer, field `rho`).

1. Load the gpkg in QGIS
2. Right-click `rho_points` → Properties → Symbology → Style → **Load Style…**
3. Choose `rho_freshem.qml`

## QGIS — raster (postproc quantile GeoTIFFs)

Use with multiband GeoTIFFs from `write.ds_to_tiff()` (e.g. `postproc - Q(0.5).tif`).

1. Load the GeoTIFF in QGIS
2. Right-click layer → Properties → Symbology → Style → **Load Style…**
3. Choose `rho_freshem_raster.qml`
4. Pick the **band** for the depth/layer you want (default is band 1)
5. Set nodata **-9999** under Transparency if masked pixels show up coloured

The raster QML uses **Singleband pseudocolor** with discrete Freshem classes — not multiband RGB.

## Regenerate

```bash
python export_gpkg/qml/gen_rho_qml.py
```

## Source

- `rho_colormap.py` — constants, log/gamma mapping, `rho_freshem_classes()`
- `rho_qml.py` — vector + raster QML builders
- `gen_rho_qml.py` — writes both `.qml` files

iMOD `.leg` files live in `snippets_chris/example_imod/leg/`.
iMOD `.dlf` for IPF associated-file colouring is written by `export_ipf/` (`rho_freshem.dlf`).
