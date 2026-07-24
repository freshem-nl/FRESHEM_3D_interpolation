"""Build iMOD MDF recipes from layer-coloured IDF folders + embedded .leg palettes."""

from pathlib import Path
import zlib

from scripts.idf_export import var_token

# iMOD Coloured 3-D Model: property slabs use style 81; top/bottom use 1.
_STYLE_PROPERTY = 81
_STYLE_TOP_BOTTOM = 1


def _idf_role(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem.endswith("_top"):
        return "top"
    if stem.endswith("_bottom"):
        return "bottom"
    return "property"


def _style_for_role(role: str) -> int:
    if role == "property":
        return _STYLE_PROPERTY
    return _STYLE_TOP_BOTTOM


def _load_order_filenames(idf_dir: Path) -> list[str]:
    manifest = idf_dir / "imod_load_order.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing IDF load order: {manifest}")
    names = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    if not names:
        raise ValueError(f"No IDF filenames in {manifest}")
    return names


def _read_leg(leg_path: Path) -> str:
    if not leg_path.is_file():
        raise FileNotFoundError(f"Legend not found: {leg_path}")
    text = leg_path.read_text(encoding="utf-8")
    return text if text.endswith("\n") else text + "\n"


def export_mdf(idf_dir: Path, mdf_path: Path, leg_path: Path) -> Path:
    """Write one MDF for an IDF property folder, embedding the given .leg palette."""
    idf_dir = Path(idf_dir).resolve()
    mdf_path = Path(mdf_path)
    leg_block = _read_leg(Path(leg_path))
    filenames = _load_order_filenames(idf_dir)

    blocks = []
    for fname in filenames:
        idf_path = (idf_dir / fname).resolve()
        if not idf_path.is_file():
            raise FileNotFoundError(f"IDF listed in load order not found: {idf_path}")
        role = _idf_role(fname)
        style = _style_for_role(role)
        entry_id = zlib.crc32(fname.encode("utf-8")) & 0x7FFFFFFF
        # iMOD on Windows expects backslash paths in MDF entries.
        abs_path = str(idf_path).replace("/", "\\")
        blocks.append(
            f'"{abs_path}","{fname}",{entry_id},{style},0\n{leg_block}'
        )

    mdf_path.parent.mkdir(parents=True, exist_ok=True)
    body = f"{len(filenames):4d}\n" + "".join(blocks)
    mdf_path.write_text(body, encoding="utf-8")
    return mdf_path


def export_mdfs(dir_idf: Path, dir_mdf: Path, properties: list[str], leg_path: Path) -> list[Path]:
    """Write one MDF per property under dir_mdf, from matching IDF subfolders."""
    if not properties:
        raise ValueError("properties must contain at least one variable name")

    written = []
    for property_name in properties:
        token = var_token(property_name)
        idf_dir = Path(dir_idf) / token
        mdf_path = Path(dir_mdf) / f"{token}.mdf"
        written.append(export_mdf(idf_dir, mdf_path, leg_path))
    return written
