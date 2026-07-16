"""Generate QGIS QML layer styles for Freshem rho symbology."""

import html
from pathlib import Path

from rho_colormap import rho_freshem_classes


def _marker_symbol(symbol_id: int, rgb: tuple[int, int, int, int]) -> str:
    r, g, b, a = rgb
    color = f"{r},{g},{b},{a}"
    return f"""      <symbol type="marker" alpha="1" clip_to_extent="1" force_rhr="0" name="{symbol_id}">
        <data_defined_properties>
          <Option type="Map"/>
        </data_defined_properties>
        <layer pass="0" locked="0" class="SimpleMarker" enabled="1">
          <Option type="Map">
            <Option type="QString" name="angle" value="0"/>
            <Option type="QString" name="cap_style" value="square"/>
            <Option type="QString" name="color" value="{color}"/>
            <Option type="QString" name="horizontal_anchor_point" value="1"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="name" value="circle"/>
            <Option type="QString" name="offset" value="0,0"/>
            <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_unit" value="MM"/>
            <Option type="QString" name="outline_color" value="35,35,35,255"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0"/>
            <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="scale_method" value="diameter"/>
            <Option type="QString" name="size" value="2"/>
            <Option type="QString" name="size_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="size_unit" value="MM"/>
            <Option type="QString" name="vertical_anchor_point" value="1"/>
          </Option>
        </layer>
      </symbol>"""


def rho_freshem_qml(attribute: str = "rho") -> str:
    """Build a graduated-symbol QML string for rho_points layers."""
    classes = rho_freshem_classes()

    range_lines = []
    symbol_lines = []
    for i, cls in enumerate(classes):
        label = html.escape(cls["label"], quote=True)
        range_lines.append(
            f'      <range upper="{cls["upper"]}" lower="{cls["lower"]}" '
            f'symbol="{i}" label="{label}" render="true"/>'
        )
        symbol_lines.append(_marker_symbol(i, (*cls["rgb"], 255)))

    ranges = "\n".join(range_lines)
    symbols = "\n".join(symbol_lines)
    source_symbol = _marker_symbol(0, (*classes[0]["rgb"], 255))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis styleCategories="Symbology" version="3.34.0-Prizren">
  <renderer-v2 type="graduatedSymbol" forceraster="0" attr="{attribute}" symbollevels="0" enableorderby="0" graduatedMethod="GraduatedColor">
    <ranges>
{ranges}
    </ranges>
    <symbols>
{symbols}
    </symbols>
    <source-symbol>
{source_symbol}
    </source-symbol>
    <colorramp type="gradient" name="[source]">
      <Option type="Map">
        <Option type="QString" name="color1" value="0,0,255,255"/>
        <Option type="QString" name="color2" value="255,0,0,255"/>
        <Option type="QString" name="discrete" value="0"/>
        <Option type="QString" name="rampType" value="gradient"/>
      </Option>
    </colorramp>
    <mode name="custom"/>
    <symmetricMode enabled="0" astride="0" symmetryPoint="0"/>
    <roundPrecision value="2"/>
    <classificationMethod id="Custom"/>
    <symbol type="marker" alpha="1" clip_to_extent="1" force_rhr="0" name="symbol">
      <data_defined_properties>
        <Option type="Map"/>
      </data_defined_properties>
      <layer pass="0" locked="0" class="SimpleMarker" enabled="1">
        <Option type="Map">
          <Option type="QString" name="color" value="227,26,28,255"/>
          <Option type="QString" name="name" value="circle"/>
          <Option type="QString" name="size" value="2"/>
        </Option>
      </layer>
    </symbol>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>0</layerGeometryType>
</qgis>
"""


def write_rho_freshem_qml(path, attribute: str = "rho") -> Path:
    """Write rho_freshem.qml for use with export_gpkg rho_points layers."""
    path = Path(path)
    path.write_text(rho_freshem_qml(attribute), encoding="utf-8")
    return path
