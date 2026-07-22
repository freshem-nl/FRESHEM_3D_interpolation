import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QML_DIR = ROOT / "export_gpkg" / "qml"
sys.path.insert(0, str(QML_DIR))

from rho_colormap import (
    RHO_ZOET_INDICATORS,
    RHO_ZOUT_INDICATORS,
    rho_zoet_classes,
    rho_zout_classes,
    write_rho_zoet_dlf,
    write_rho_zoet_leg,
    write_rho_zout_dlf,
    write_rho_zout_leg,
)


def test_rho_zoet_class_count_and_open_ends():
    classes = rho_zoet_classes()
    assert len(classes) == len(RHO_ZOET_INDICATORS) + 1
    assert classes[0]["label"] == "< 5"
    assert classes[-1]["label"] == "> 100"
    assert classes[0]["upper"] == 5
    assert classes[-1]["lower"] == 100
    assert classes[1]["label"] == "5 - 10"


def test_rho_zout_class_count_and_open_ends():
    classes = rho_zout_classes()
    assert len(classes) == len(RHO_ZOUT_INDICATORS) + 1
    assert classes[0]["label"] == "< 1"
    assert classes[-1]["label"] == "> 100"
    assert classes[0]["upper"] == 1
    assert classes[-1]["lower"] == 100
    assert classes[1]["label"] == "1 - 2"
    assert any(c["label"] == "5 - 7" for c in classes)


def test_rho_zoet_leg_and_dlf_roundtrip(tmp_path):
    n = len(RHO_ZOET_INDICATORS) + 1
    leg = write_rho_zoet_leg(tmp_path / "rho_zoet.leg").read_text(encoding="utf-8")
    dlf = write_rho_zoet_dlf(tmp_path / "rho_zoet.dlf").read_text(encoding="utf-8")
    assert leg.startswith(f"{n},1,1,1,1,1,1,1")
    assert '"> 100"' in leg
    assert '"< 5"' in leg
    assert "0.1000000E+21" in leg
    assert "0.000000" in leg
    assert dlf.startswith("Label,Ired,Igreen,Iblue,Label-text")
    assert '"0",' in dlf
    assert f'"{n - 1}",' in dlf


def test_rho_zout_leg_and_dlf_roundtrip(tmp_path):
    n = len(RHO_ZOUT_INDICATORS) + 1
    leg = write_rho_zout_leg(tmp_path / "rho_zout.leg").read_text(encoding="utf-8")
    dlf = write_rho_zout_dlf(tmp_path / "rho_zout.dlf").read_text(encoding="utf-8")
    assert leg.startswith(f"{n},1,1,1,1,1,1,1")
    assert '"> 100"' in leg
    assert '"< 1"' in leg
    assert "0.1000000E+21" in leg
    assert dlf.startswith("Label,Ired,Igreen,Iblue,Label-text")
