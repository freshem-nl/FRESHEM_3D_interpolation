import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "snippets_chris" / "example_imod" / "leg"))

from _gen_prob_legs import RHO_INDICATORS, rho_freshem_rows


def test_rho_freshem_row_count():
    assert len(rho_freshem_rows()) == len(RHO_INDICATORS) + 1


def test_rho_freshem_top_class():
    top = rho_freshem_rows()[0]
    assert "> 100" in top
    assert top.startswith("0.1000000E+21,100,")
