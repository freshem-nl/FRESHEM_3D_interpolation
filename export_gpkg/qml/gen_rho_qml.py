"""Generate QGIS QML styles for Freshem rho symbology."""

from pathlib import Path

from rho_qml import write_rho_freshem_qml, write_rho_freshem_raster_qml

OUT_DIR = Path(__file__).parent

if __name__ == "__main__":
    write_rho_freshem_qml(OUT_DIR / "rho_freshem.qml")
    write_rho_freshem_raster_qml(OUT_DIR / "rho_freshem_raster.qml")
    print(f"wrote {OUT_DIR / 'rho_freshem.qml'}")
    print(f"wrote {OUT_DIR / 'rho_freshem_raster.qml'}")
