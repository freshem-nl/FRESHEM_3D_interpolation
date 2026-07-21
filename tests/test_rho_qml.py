import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QML_DIR = ROOT / "export_gpkg" / "qml"
sys.path.insert(0, str(QML_DIR))

from rho_colormap import RHO_INDICATORS, rho_freshem_classes
from rho_qml import rho_freshem_qml, rho_freshem_raster_qml, write_rho_freshem_raster_qml


def test_rho_freshem_class_count():
    assert len(rho_freshem_classes()) == len(RHO_INDICATORS) + 1


def test_rho_freshem_raster_qml_is_valid_xml():
    ET.fromstring(rho_freshem_raster_qml())


def test_rho_freshem_raster_qml_renderer(tmp_path):
    path = write_rho_freshem_raster_qml(tmp_path / "rho_freshem_raster.qml")
    text = path.read_text(encoding="utf-8")
    assert 'type="singlebandpseudocolor"' in text
    assert 'colorRampType="DISCRETE"' in text
    assert 'prop k="stops"' in text
    assert 'minimumValue="0.01"' in text
    assert 'band="1"' in text
    assert 'label="&lt; 1"' in text
    assert 'label="&gt; 100"' in text
    assert 'color="#4b7cb7"' in text


def test_rho_freshem_vector_qml_is_valid_xml():
    ET.fromstring(rho_freshem_qml())
