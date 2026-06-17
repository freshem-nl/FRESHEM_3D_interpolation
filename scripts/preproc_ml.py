import re
from datetime import datetime

from scripts import _preproc_helper

def OGC(data, cfg):

    t0 = datetime.now()
    print("Calculating oblique geographic coordinates (OGC)...", end=" ")

    features = cfg["features"]

    # from config
    # thetas = cfg["OGC_angles"]

    # get OGC angles from features
    thetas = []
    OGC_names = []
    for feature in features:
        match = re.fullmatch(r"OGC\(\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*\)", feature)
        if match:
            thetas.append(float(match.group(1)))
            OGC_names.append(feature)

    for theta, ogc_name in zip(thetas, OGC_names):

        # calculate OGC for data points
        data[ogc_name] = _preproc_helper.oblique_geographic_coordinates(data["x"], data["y"], theta)

    print(f"done ({(datetime.now() - t0).total_seconds():.2f}s).")

    return data
