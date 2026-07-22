"""Freshem resistivity colour scale for iMOD legends, plots, and QGIS styles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RHO_LOWER = 0.01
RHO_UPPER = 150.0
RHO_BOUNDS = (RHO_LOWER, RHO_UPPER)
RHO_CMAP = "RdYlBu_r"
RHO_GAMMA = 1.5
RHO_INDICATORS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
PROBABILITY_CMAP = "Blues"
RASTER_NODATA = -9999


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
