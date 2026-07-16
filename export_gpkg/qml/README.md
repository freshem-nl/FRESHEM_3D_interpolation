# Freshem rho colour scale (iMOD + QGIS)

Shared palette for resistivity: log scale 0.01–150 Ωm, `RdYlBu_r`, gamma 1.5 (more blue at low rho).

## iMOD

- `rho_freshem.leg` — discrete legend for iMOD Coloured 3-D Model
- Regenerate all `.leg` files: `python _gen_prob_legs.py`

## QGIS

Use with GeoPackages from `export_gpkg/` (`rho_points` layer, field `rho`).

1. Load the gpkg in QGIS
2. Right-click `rho_points` → Properties → Symbology → Style → **Load Style…**
3. Choose `rho_freshem.qml`

Regenerate after palette changes: `python gen_rho_qml.py`

## Source

- `rho_colormap.py` — constants, log/gamma mapping, `rho_freshem_classes()`
- `rho_qml.py` — QML builder
- `gen_rho_qml.py` — writes `rho_freshem.qml`
