"""Tests for SkyTEM xyz -> iMOD IPF export."""

from scripts.gpkg_export import parse_skytem_xyz
from scripts.ipf_export import (
    assign_sounding_ids,
    export_rho_ipf,
    layers_long,
    rho_to_ohm_code,
    thin_min_spacing,
)


XYZ_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 DOI_STANDARD DOI_CONSERVATIVE
1 100.0 200.0 5.0 10.5 1.2 0.0 5.0 5.0 50.0 40.0
1 150.0 250.0 5.0 20.0 2.0 5.0 15.0 10.0 50.0 40.0
2 300.0 400.0 4.0 30.0 3.0 0.0 8.0 8.0 45.0 35.0
2 350.0 450.0 4.0 35.0 3.5 8.0 18.0 10.0 45.0 35.0
"""

XYZ_SPACING_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 DOI_STANDARD DOI_CONSERVATIVE
1 0.0 0.0 5.0 10.0 1.0 0.0 5.0 5.0 50.0 40.0
1 40.0 0.0 5.0 11.0 1.0 0.0 5.0 5.0 50.0 40.0
1 80.0 0.0 5.0 12.0 1.0 0.0 5.0 5.0 50.0 40.0
1 160.0 0.0 5.0 13.0 1.0 0.0 5.0 5.0 50.0 40.0
2 0.0 100.0 4.0 20.0 1.0 0.0 5.0 5.0 45.0 35.0
2 30.0 100.0 4.0 21.0 1.0 0.0 5.0 5.0 45.0 35.0
"""

XYZ_DOI_SAMPLE = """\
/ LINE_NO X Y ELEVATION RHO_1 RHO_STD1 DEP_TOP_1 DEP_BOT_1 THK_1 RHO_2 RHO_STD2 DEP_TOP_2 DEP_BOT_2 THK_2 DOI_STANDARD DOI_CONSERVATIVE
1 100.0 200.0 5.0 10.5 1.2 0.0 5.0 5.0 8.0 1.0 5.0 15.0 10.0 4.0 3.0
"""


def _write_sample_xyz(path, text=XYZ_SAMPLE):
    path.write_text(text, encoding="utf-8")


def test_assign_sounding_ids(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    _write_sample_xyz(xyz_path)
    df = assign_sounding_ids(parse_skytem_xyz(xyz_path))

    assert list(df["sounding_id"]) == ["L1_0001", "L1_0002", "L2_0001", "L2_0002"]


def test_layers_long(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    _write_sample_xyz(xyz_path)
    df = assign_sounding_ids(parse_skytem_xyz(xyz_path))

    long = layers_long(df)

    assert len(long) == 4
    assert set(long["sounding_id"]) == {"L1_0001", "L1_0002", "L2_0001", "L2_0002"}
    assert {"z_top", "z_bottom", "rho", "elevation"}.issubset(long.columns)


def test_rho_to_ohm_code():
    assert rho_to_ohm_code(10.5) == 10
    assert rho_to_ohm_code(10.6) == 11
    assert rho_to_ohm_code(0.3) == 1
    assert rho_to_ohm_code(200.0) == 150
    assert rho_to_ohm_code(float("nan")) is None


def test_export_rho_ipf(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    ipf_path = tmp_path / "sample.ipf"
    _write_sample_xyz(xyz_path)

    result = export_rho_ipf(xyz_path, ipf_path, apply_doi_clip=False)

    assert result["ipf"] == ipf_path
    assert result["ipf"].is_file()
    assert result["associated_dir"] == tmp_path / "sample"
    assert "dlf" not in result

    ipf_text = result["ipf"].read_text(encoding="utf-8").splitlines()
    assert ipf_text[0] == "4"
    assert ipf_text[1] == "6"
    assert ipf_text[8].startswith("3,txt")
    assert ipf_text[9].split(",")[2] == r"sample\L1_0001"

    txt = (result["associated_dir"] / "L1_0001.txt").read_text(encoding="utf-8").splitlines()
    assert txt[0] == "2"
    assert txt[1] == "3,2"
    assert '"topnap",-999.99' in txt
    assert '"rho",-999.99' in txt
    assert '"rho_ohm",-999.99' in txt
    # RHO_1 = 10.5 -> ohm code 10
    assert txt[5] == "5,10.5,10"
    assert txt[-1].endswith(",end,-")


def test_export_rho_ipf_bbox(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    ipf_path = tmp_path / "clipped.ipf"
    _write_sample_xyz(xyz_path)

    result = export_rho_ipf(
        xyz_path,
        ipf_path,
        bbox=[290, 360, 390, 460],
        apply_doi_clip=False,
    )

    ipf_text = result["ipf"].read_text(encoding="utf-8").splitlines()
    assert ipf_text[0] == "2"
    assert len(list(result["associated_dir"].glob("L*_*.txt"))) == 2
    assert ipf_text[9].split(",")[2] == r"clipped\L2_0001"


def test_export_rho_ipf_doi_clip(tmp_path):
    xyz_path = tmp_path / "sample_doi.xyz"
    ipf_path = tmp_path / "sample_doi.ipf"
    _write_sample_xyz(xyz_path, XYZ_DOI_SAMPLE)

    result = export_rho_ipf(xyz_path, ipf_path, apply_doi_clip=True)

    txt = (result["associated_dir"] / "L1_0001.txt").read_text(encoding="utf-8").splitlines()
    # one layer + end row
    assert txt[0] == "2"
    data_row = txt[5]
    assert data_row.startswith("5,")


def test_thin_min_spacing(tmp_path):
    xyz_path = tmp_path / "spacing.xyz"
    _write_sample_xyz(xyz_path, XYZ_SPACING_SAMPLE)
    df = parse_skytem_xyz(xyz_path)

    thinned = thin_min_spacing(df, 75)

    # line 1: keep 0, then 80 (from 0), then 160; line 2: keep 0 only (30 m < 75)
    assert len(thinned) == 4
    assert list(thinned["x"]) == [0.0, 80.0, 160.0, 0.0]
    assert list(thinned["line_no"]) == [1, 1, 1, 2]


def test_export_rho_ipf_min_spacing(tmp_path):
    xyz_path = tmp_path / "spacing.xyz"
    ipf_path = tmp_path / "spacing.ipf"
    _write_sample_xyz(xyz_path, XYZ_SPACING_SAMPLE)

    result = export_rho_ipf(
        xyz_path,
        ipf_path,
        min_spacing_m=75,
        apply_doi_clip=False,
    )

    assert result["ipf"].read_text(encoding="utf-8").splitlines()[0] == "4"
    assert len(list(result["associated_dir"].glob("L*_*.txt"))) == 4
    ids = [line.split(",")[2] for line in result["ipf"].read_text(encoding="utf-8").splitlines()[9:]]
    assert ids == [
        r"spacing\L1_0001",
        r"spacing\L1_0002",
        r"spacing\L1_0003",
        r"spacing\L2_0001",
    ]
