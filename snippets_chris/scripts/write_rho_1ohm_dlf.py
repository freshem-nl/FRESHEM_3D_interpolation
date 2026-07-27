"""Write 1 Ohm.m iMOD DLF legends for IPF rho_ohm colouring.

Matches associated-file codes from scripts/ipf_export.py:
  rho_ohm = round(rho), clipped to 1..150

Run from repo root:
  python snippets_chris/scripts/write_rho_1ohm_dlf.py

Writes:
  snippets_chris/example_imod/dlf/rho_zoet_1ohm.dlf
  snippets_chris/example_imod/dlf/rho_zout_1ohm.dlf
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QML_DIR = ROOT / "export_gpkg" / "qml"
OUT_DIR = ROOT / "snippets_chris" / "example_imod" / "dlf"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QML_DIR))

from rho_colormap import (  # noqa: E402
    RHO_ZOET_BOUNDS,
    RHO_ZOUT_BOUNDS,
    _rho_custom_fraction,
    rho_custom_cmap,
)
from scripts.ipf_export import RHO_OHM_CODE_MAX, RHO_OHM_CODE_MIN  # noqa: E402

SCALES = {
    "zoet": {"bounds": RHO_ZOET_BOUNDS, "mode": "linear"},
    "zout": {"bounds": RHO_ZOUT_BOUNDS, "mode": "log"},
}


def rgb_at(value: float, scale: str) -> tuple[int, int, int]:
    cfg = SCALES[scale]
    cmap = rho_custom_cmap()
    t = _rho_custom_fraction(value, cfg["bounds"], scale=cfg["mode"])
    r, g, b, _ = cmap(t)
    return int(r * 255), int(g * 255), int(b * 255)


def format_1ohm_dlf(scale: str) -> str:
    lines = ["Label,Ired,Igreen,Iblue,Label-text"]
    for code in range(RHO_OHM_CODE_MIN, RHO_OHM_CODE_MAX + 1):
        r, g, b = rgb_at(float(code), scale)
        lines.append(f'"{code}",{r},{g},{b},"{code}",0.5')
    return "\n".join(lines) + "\n"


def write_1ohm_dlfs(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scale in SCALES:
        path = out_dir / f"rho_{scale}_1ohm.dlf"
        path.write_text(format_1ohm_dlf(scale), encoding="utf-8")
        written.append(path)
        print(f"Wrote {path} ({RHO_OHM_CODE_MAX - RHO_OHM_CODE_MIN + 1} classes)")
    return written


if __name__ == "__main__":
    write_1ohm_dlfs()
