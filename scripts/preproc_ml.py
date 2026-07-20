import re
from datetime import datetime

from scripts import _preproc_helper

def OGC(data, cfg):

    t0 = datetime.now()
    print("Calculating oblique geographic coordinates (OGC)...", end=" ")

    # from config
    features = cfg["features"]

    # get OGC angles from features
    thetas = []
    OGC_names = []
    for feature in features:
        # match OGC features of the form "OGC(theta)" or "OGC(theta°)"
        match = re.fullmatch(r"OGC\(\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*\)", feature)
        if match:
            thetas.append(float(match.group(1)))
            OGC_names.append(feature)

    for theta, ogc_name in zip(thetas, OGC_names):

        # calculate OGC
        data[ogc_name] = _preproc_helper.oblique_geographic_coordinates(data["x"], data["y"], theta)

    print(f"({(datetime.now() - t0).total_seconds():.2f}s)")

    return data
