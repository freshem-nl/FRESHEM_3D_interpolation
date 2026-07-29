"""Tests for IDF folder -> iMOD MDF export."""

from pathlib import Path

from scripts.mdf_export import export_mdf, export_mdfs


def _write_fake_idf_package(idf_dir: Path, property_token: str = "Q_0_5"):
    idf_dir.mkdir(parents=True, exist_ok=True)
    names = [
        f"001_layer01_top.idf",
        f"002_layer01_{property_token}.idf",
        f"003_layer01_bottom.idf",
    ]
    for name in names:
        (idf_dir / name).write_text("idf-placeholder\n", encoding="utf-8")
    manifest = [
        "# iMOD Coloured 3-D Model — load IDFs in this order",
        "# Per layer: top, property, bottom",
        "",
        *names,
        "",
    ]
    (idf_dir / "imod_load_order.txt").write_text("\n".join(manifest), encoding="utf-8")
    return names


def test_export_mdf_embeds_leg_and_styles(tmp_path):
    idf_dir = tmp_path / "idf" / "Q_0_5"
    names = _write_fake_idf_package(idf_dir)

    leg_path = tmp_path / "rho_zout.leg"
    leg_path.write_text(
        "13,1,1,1,1,1,1,1\n"
        "UPPERBND,LOWERBND,IRED,IGREEN,IBLUE,DOMAIN\n"
        '1.000000,0.000000,0,0,190,"< 1"\n',
        encoding="utf-8",
    )

    mdf_path = tmp_path / "mdf" / "Q_0_5.mdf"
    export_mdf(idf_dir, mdf_path, leg_path)

    text = mdf_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].strip() == "3"
    assert "UPPERBND,LOWERBND,IRED,IGREEN,IBLUE,DOMAIN" in text
    assert text.count('"< 1"') == 3

    # top / property / bottom style flags
    assert f',1,0' in text
    assert f',81,0' in text

    for name in names:
        assert f'"{name}"' in text
        assert name.replace("/", "\\") in text or name in text


def test_export_mdfs_writes_one_per_property(tmp_path):
    dir_idf = tmp_path / "idf"
    _write_fake_idf_package(dir_idf / "Q_0_5", "Q_0_5")

    leg_path = tmp_path / "leg.leg"
    leg_path.write_text(
        "2,1,1,1,1,1,1,1\nUPPERBND,LOWERBND,IRED,IGREEN,IBLUE,DOMAIN\n",
        encoding="utf-8",
    )

    imod_idf_root = Path(
        "C:/Temp_Geomodelling/FreshEM/idf/lagenmodel/LCI_Sharp_MOD_inv/Rho"
    )
    written = export_mdfs(
        dir_idf,
        tmp_path / "mdf",
        ["Q(0.5)"],
        leg_path,
        imod_idf_root,
        name_suffix="LCI_Sharp_MOD_inv",
    )
    assert len(written) == 1
    assert written[0].name == "Q_0_5_LCI_Sharp_MOD_inv.mdf"
    assert written[0].is_file()

    text = written[0].read_text(encoding="utf-8")
    assert (
        "C:\\Temp_Geomodelling\\FreshEM\\idf\\lagenmodel\\"
        "LCI_Sharp_MOD_inv\\Rho\\Q_0_5" in text
    )
    assert str(dir_idf) not in text
