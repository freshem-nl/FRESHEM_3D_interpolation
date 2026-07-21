<?xml version="1.0" encoding="UTF-8"?>
<qgis styleCategories="Symbology" version="3.34.0-Prizren">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" band="1" classificationMin="0.01" classificationMax="150.0" type="singlebandpseudocolor" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <stat>MinMax</stat>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader clip="0" colorRampType="DISCRETE" classificationMode="1" labelPrecision="6" maximumValue="150.0" minimumValue="0.01">
          <colorramp type="gradient" name="[source]">
            <prop k="color1" v="75,124,183,255"/>
            <prop k="color2" v="180,15,38,255"/>
            <prop k="discrete" v="1"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.3314;75,124,183,255:0.4090;207,234,243,255:0.4568;233,246,230,255:0.5196;250,253,198,255:0.6089;254,234,160,255:0.7028;253,196,118,255:0.7598;250,152,87,255:0.8336;244,111,68,255:0.9374;218,56,42,255:1.0000;180,15,38,255"/>
          </colorramp>
          <item label="&lt; 1" color="#4b7cb7" alpha="255" value="1"/>
          <item label="1 - 2" color="#cfeaf3" alpha="255" value="2"/>
          <item label="2 - 3" color="#e9f6e6" alpha="255" value="3"/>
          <item label="3 - 5" color="#fafdc6" alpha="255" value="5"/>
          <item label="5 - 10" color="#feeaa0" alpha="255" value="10"/>
          <item label="10 - 20" color="#fdc476" alpha="255" value="20"/>
          <item label="20 - 30" color="#fa9857" alpha="255" value="30"/>
          <item label="30 - 50" color="#f46f44" alpha="255" value="50"/>
          <item label="50 - 100" color="#da382a" alpha="255" value="100"/>
          <item label="&gt; 100" color="#b40f26" alpha="255" value="inf"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
  <layerTransparency enabled="0"/>
  <customproperties>
    <Option type="Map">
      <Option name="freshem_nodata" type="double" value="-9999"/>
    </Option>
  </customproperties>
</qgis>
