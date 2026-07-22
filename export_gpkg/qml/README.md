# Freshem rho colour scale (iMOD + QGIS)

Shared palettes for resistivity:

- **freshem** — log 0.01–150 Ωm, `RdYlBu_r`, gamma 1.5
- **zoet** — linear 0–100 Ωm, custom RGB ramp, round bins, open ends
- **zout** — log 1–100 Ωm, same custom RGB ramp, round bins, open ends

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
python snippets_chris/example_imod/leg/_gen_prob_legs.py
```

## Source

- `rho_colormap.py` — constants, log/gamma mapping, freshem/zoet/zout classes
- `rho_qml.py` — vector + raster QML builders
- `gen_rho_qml.py` — writes both `.qml` files

iMOD `.leg` files live in `snippets_chris/example_imod/leg/` (`rho_freshem`, `rho_zoet`, `rho_zout`).
iMOD `.dlf` files live in `snippets_chris/example_imod/dlf/` (`rho_zoet`, `rho_zout`); IPF export also writes `rho_freshem.dlf` next to the IPF.
