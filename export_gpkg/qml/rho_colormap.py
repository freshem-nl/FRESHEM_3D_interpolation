"""Freshem resistivity colour scale for iMOD legends, plots, and QGIS styles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

RHO_LOWER = 0.01
RHO_UPPER = 150.0
RHO_BOUNDS = (RHO_LOWER, RHO_UPPER)
RHO_CMAP = "RdYlBu_r"
RHO_GAMMA = 1.5
RHO_INDICATORS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
PROBABILITY_CMAP = "Blues"
RASTER_NODATA = -9999

# Custom RGB palette (UI order: high lavender → low navy). Flipped for mapping.
RHO_CUSTOM_RGB_HI_TO_LO = [
    (230, 155, 240),
    (195, 130, 240),
    (165, 70, 220),
    (140, 40, 180),
    (255, 0, 120),
    (255, 0, 0),
    (255, 115, 0),
    (255, 195, 0),
    (255, 255, 0),
    (180, 255, 30),
    (30, 210, 0),
    (80, 240, 255),
    (0, 200, 255),
    (0, 150, 235),
    (0, 75, 220),
    (0, 0, 190),
]
RHO_CUSTOM_RGB_LO_TO_HI = list(reversed(RHO_CUSTOM_RGB_HI_TO_LO))

# Zoet: linear scale, round thresholds. Zout: log scale, round thresholds.
# Colour domain is 0–100 / 1–100; UPPER is only for sampling the open last class.
RHO_ZOET_BOUNDS = (0.0, 100.0)
RHO_ZOET_LOWER = 0.0
RHO_ZOET_UPPER = 150.0
RHO_ZOET_INDICATORS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RHO_ZOUT_BOUNDS = (1.0, 100.0)
RHO_ZOUT_LOWER = 0.5
RHO_ZOUT_UPPER = 150.0
RHO_ZOUT_INDICATORS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]


def rho_log_fraction(value):
    """Map rho (Ohm.m) to [0, 1] on a log scale, with gamma bias toward blue."""
    log_lo = np.log10(RHO_LOWER)
    log_hi = np.log10(RHO_UPPER)
    log_v = np.log10(np.clip(value, RHO_LOWER, RHO_UPPER))
    t = (log_v - log_lo) / (log_hi - log_lo)
    return t ** RHO_GAMMA


class RhoNorm(LogNorm):
    """LogNorm for resistivity with gamma remap to extend the blue range."""

    def __init__(self, vmin=None, vmax=None, gamma=RHO_GAMMA, clip=False):
        super().__init__(vmin=vmin or RHO_LOWER, vmax=vmax or RHO_UPPER, clip=clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip=clip)
        if isinstance(result, np.ma.MaskedArray):
            return np.ma.power(result, self.gamma)
        return np.power(result, self.gamma)


def rho_freshem_classes():
    """Discrete rho classes (low to high) matching the iMOD rho_freshem.leg palette."""
    thresholds = [RHO_LOWER, *RHO_INDICATORS, RHO_UPPER]
    cmap = plt.get_cmap(RHO_CMAP)
    classes = []

    for i in range(len(thresholds) - 1):
        t_lo = thresholds[i]
        t_hi = thresholds[i + 1]
        mid_val = np.sqrt(t_lo * t_hi)
        t_norm = float(rho_log_fraction(mid_val))
        r, g, b, _ = cmap(t_norm)
        rgb = (int(r * 255), int(g * 255), int(b * 255))

        if i == 0:
            lower = 0.0
            label = f"< {t_hi:g}"
        elif i == len(thresholds) - 2:
            lower = t_lo
            t_hi = 1.0e9
            label = f"> {t_lo:g}"
        else:
            lower = t_lo
            label = f"{t_lo:g} - {thresholds[i + 1]:g}"

        classes.append(
            {
                "index": i,
                "lower": lower,
                "upper": t_hi,
                "label": label,
                "rgb": rgb,
            }
        )

    return classes


def rho_to_class_index(rho, classes=None):
    """Map a resistivity value to the Freshem discrete class index."""
    if classes is None:
        classes = rho_freshem_classes()
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return None
    for cls in classes:
        if cls["lower"] <= rho < cls["upper"]:
            return cls["index"]
    return classes[-1]["index"]


def format_rho_freshem_dlf(classes=None):
    """Return iMOD DLF text for Freshem rho classes (IPF associated-file colouring)."""
    if classes is None:
        classes = rho_freshem_classes()
    lines = ["Label,Ired,Igreen,Iblue,Label-text"]
    for cls in classes:
        r, g, b = cls["rgb"]
        lines.append(f'"{cls["index"]}",{r},{g},{b},"{cls["label"]}",0.5')
    return "\n".join(lines) + "\n"


def write_rho_freshem_dlf(path):
    """Write an iMOD DLF legend for Freshem resistivity classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_rho_freshem_dlf(), encoding="utf-8")
    return path


def _format_bound(value):
    """Compact numeric string for .leg bounds (enough precision to avoid gaps)."""
    return f"{float(value):.6g}"


def _format_label_bound(value):
    """Shorter numeric string for legend labels."""
    return f"{float(value):g}"


def rho_custom_cmap():
    """Continuous colormap from the custom RGB stops (low blue → high lavender)."""
    colors = [(r / 255, g / 255, b / 255) for r, g, b in RHO_CUSTOM_RGB_LO_TO_HI]
    return LinearSegmentedColormap.from_list("rho_custom", colors, N=256)


def _rho_custom_fraction(value, bounds, scale):
    lo, hi = bounds
    if scale == "log":
        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        log_v = np.log10(np.clip(value, lo, hi))
        return float((log_v - log_lo) / (log_hi - log_lo))
    return float((np.clip(value, lo, hi) - lo) / (hi - lo))


def _rho_custom_classes(thresholds, bounds, scale):
    """Open-ended classes at round thresholds; colours sampled from the custom ramp."""
    cmap = rho_custom_cmap()
    classes = []
    n_intervals = len(thresholds) - 1

    for i in range(n_intervals):
        t_lo = float(thresholds[i])
        t_hi = float(thresholds[i + 1])
        if scale == "log":
            mid_val = float(np.sqrt(max(t_lo, 1e-12) * t_hi))
        else:
            mid_val = 0.5 * (t_lo + t_hi)
        t_norm = _rho_custom_fraction(mid_val, bounds, scale)
        r, g, b, _ = cmap(t_norm)
        rgb = (int(r * 255), int(g * 255), int(b * 255))

        if i == 0:
            lower = 0.0
            upper = t_hi
            label = f"< {_format_label_bound(t_hi)}"
        elif i == n_intervals - 1:
            lower = t_lo
            upper = 1.0e9
            label = f"> {_format_label_bound(t_lo)}"
        else:
            lower = t_lo
            upper = t_hi
            label = f"{_format_label_bound(t_lo)} - {_format_label_bound(t_hi)}"

        classes.append(
            {
                "index": i,
                "lower": lower,
                "upper": upper,
                "label": label,
                "rgb": rgb,
            }
        )
    return classes


def rho_zoet_classes():
    """Discrete rho classes: linear zoet scale, round bins, custom RGB palette."""
    thresholds = [RHO_ZOET_LOWER, *RHO_ZOET_INDICATORS, RHO_ZOET_UPPER]
    return _rho_custom_classes(thresholds, RHO_ZOET_BOUNDS, scale="linear")


def rho_zout_classes():
    """Discrete rho classes: log zout scale, round bins, custom RGB palette."""
    thresholds = [RHO_ZOUT_LOWER, *RHO_ZOUT_INDICATORS, RHO_ZOUT_UPPER]
    return _rho_custom_classes(thresholds, RHO_ZOUT_BOUNDS, scale="log")

def format_rho_dlf(classes):
    """Return iMOD DLF text for discrete rho classes."""
    lines = ["Label,Ired,Igreen,Iblue,Label-text"]
    for cls in classes:
        r, g, b = cls["rgb"]
        lines.append(f'"{cls["index"]}",{r},{g},{b},"{cls["label"]}",0.5')
    return "\n".join(lines) + "\n"


def format_rho_leg(classes):
    """Return iMOD .leg text for discrete rho classes (high→low rows)."""
    rows = []
    n = len(classes)
    for cls in reversed(classes):
        r, g, b = cls["rgb"]
        if cls["index"] == n - 1:
            upper = "0.1000000E+21"
            lower = _format_bound(cls["lower"])
        elif cls["index"] == 0:
            upper = _format_bound(cls["upper"])
            lower = "0.000000"
        else:
            upper = _format_bound(cls["upper"])
            lower = _format_bound(cls["lower"])
        rows.append(f'{upper},{lower},{r},{g},{b},"{cls["label"]}"')
    lines = [f"{n},1,1,1,1,1,1,1", "UPPERBND,LOWERBND,IRED,IGREEN,IBLUE,DOMAIN", *rows]
    return "\n".join(lines) + "\n"


def write_rho_dlf(path, classes):
    """Write an iMOD DLF legend for the given rho classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_rho_dlf(classes), encoding="utf-8")
    return path


def write_rho_leg(path, classes):
    """Write an iMOD .leg legend for the given rho classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_rho_leg(classes), encoding="utf-8")
    return path


def write_rho_zoet_dlf(path):
    return write_rho_dlf(path, rho_zoet_classes())


def write_rho_zout_dlf(path):
    return write_rho_dlf(path, rho_zout_classes())


def write_rho_zoet_leg(path):
    return write_rho_leg(path, rho_zoet_classes())


def write_rho_zout_leg(path):
    return write_rho_leg(path, rho_zout_classes())
