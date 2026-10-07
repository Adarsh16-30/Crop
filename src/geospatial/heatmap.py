"""
Interactive Disease Risk Heatmap Generator
============================================
Creates Folium-based interactive maps with heatmap overlays,
risk markers, and zone boundaries for crop disease visualization.
"""

import os

import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
import pandas as pd


def generate_risk_heatmap(
    gdf: gpd.GeoDataFrame,
    output_path: str = 'reports/disease_risk_map.html',
    title: str = 'Crop Disease Risk Map'
) -> folium.Map:
    """
    Generate an interactive Folium heatmap showing disease risk distribution.

    Features:
    - Base heatmap layer (green → yellow → red gradient)
    - Individual markers for high-risk zones (risk > 0.7)
    - Popup details per marker
    - Layer controls for toggling overlays
    - Custom tile layers

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with lat, lon, risk_score, predicted_label, confidence.
    output_path : str
        Path to save the HTML map.
    title : str
        Map title.

    Returns
    -------
    folium.Map
        The generated Folium map object.
    """
    # Center map on data centroid
    center_lat = gdf['lat'].mean()
    center_lon = gdf['lon'].mean()

    # Create base map with multiple tile options
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None  # We'll add tiles manually for layer control
    )

    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street').add_to(m)

    # ── Heatmap Layer ──────────────────────────────────────────
    heat_data = [
        [row['lat'], row['lon'], row['risk_score']]
        for _, row in gdf.iterrows()
    ]

    heatmap_layer = folium.FeatureGroup(name='Disease Risk Heatmap')
    HeatMap(
        heat_data,
        radius=20,
        blur=15,
        max_zoom=15,
        gradient={
            '0.0': '#00ff00',   # Green — low risk
            '0.3': '#7fff00',   # Yellow-green
            '0.5': '#ffff00',   # Yellow — moderate
            '0.7': '#ff7f00',   # Orange — high
            '1.0': '#ff0000',   # Red — critical
        }
    ).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    # ── High-Risk Markers ──────────────────────────────────────
    label_names = {0: 'Healthy', 1: 'Early Stress', 2: 'Active Disease'}
    label_colors = {0: 'green', 1: 'orange', 2: 'red'}

    high_risk = gdf[gdf['risk_score'] > 0.7]
    if len(high_risk) > 0:
        marker_cluster = MarkerCluster(name='High-Risk Points')

        for _, row in high_risk.iterrows():
            label = int(row['predicted_label'])
            popup_html = f"""
            <div style='width:200px;font-family:Arial;'>
                <h4 style='margin:0;color:{label_colors[label]}'>
                    ⚠️ {label_names[label]}
                </h4>
                <hr style='margin:5px 0;'>
                <b>Risk Score:</b> {row['risk_score']:.3f}<br>
                <b>Confidence:</b> {row['confidence']:.3f}<br>
                <b>Location:</b> ({row['lat']:.4f}, {row['lon']:.4f})<br>
            </div>
            """
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=label_colors[label],
                fill=True,
                fill_color=label_colors[label],
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Risk: {row['risk_score']:.2f}"
            ).add_to(marker_cluster)

        marker_cluster.add_to(m)

    # ── All Sample Markers (small, semi-transparent) ───────────
    all_markers = folium.FeatureGroup(name='All Samples', show=False)
    for _, row in gdf.iterrows():
        label = int(row['predicted_label'])
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            color=label_colors[label],
            fill=True,
            fill_color=label_colors[label],
            fill_opacity=0.3,
            weight=1,
            tooltip=f"{label_names[label]} | Risk: {row['risk_score']:.2f}"
        ).add_to(all_markers)
    all_markers.add_to(m)

    # ── Legend ─────────────────────────────────────────────────
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background-color: white; border: 2px solid grey; border-radius: 8px;
         padding: 12px; font-family: Arial; font-size: 13px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b style="font-size:14px;">🌱 Disease Risk Legend</b><br>
        <i style="background:#00ff00;width:12px;height:12px;display:inline-block;
           border-radius:2px;margin-right:5px;"></i> Low Risk (0.0 - 0.2)<br>
        <i style="background:#ffff00;width:12px;height:12px;display:inline-block;
           border-radius:2px;margin-right:5px;"></i> Moderate (0.2 - 0.5)<br>
        <i style="background:#ff7f00;width:12px;height:12px;display:inline-block;
           border-radius:2px;margin-right:5px;"></i> High (0.5 - 0.7)<br>
        <i style="background:#ff0000;width:12px;height:12px;display:inline-block;
           border-radius:2px;margin-right:5px;"></i> Critical (0.7 - 1.0)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background-color: rgba(255,255,255,0.9);
         border: 2px solid #333; border-radius: 8px; padding: 8px 20px;
         font-family: Arial; font-size: 16px; font-weight: bold;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Interactive risk heatmap saved: {output_path}")

    return m


def generate_zone_map(
    zone_gdf: gpd.GeoDataFrame,
    output_path: str = 'reports/zone_risk_map.html'
) -> folium.Map:
    """
    Generate a choropleth-style map showing risk aggregated by zone.

    Parameters
    ----------
    zone_gdf : gpd.GeoDataFrame
        Zone GeoDataFrame from aggregate_zone_risk().
    output_path : str
        Path to save HTML.

    Returns
    -------
    folium.Map
        Generated Folium map.
    """
    center_lat = zone_gdf['center_lat'].mean()
    center_lon = zone_gdf['center_lon'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    def risk_color(risk: float) -> str:
        """Map risk score to color."""
        if risk < 0.2:
            return '#27ae60'  # Green
        elif risk < 0.5:
            return '#f1c40f'  # Yellow
        elif risk < 0.7:
            return '#e67e22'  # Orange
        else:
            return '#e74c3c'  # Red

    for _, zone in zone_gdf.iterrows():
        color = risk_color(zone['mean_risk'])
        popup_html = f"""
        <b>Zone Risk: {zone['mean_risk']:.3f}</b><br>
        Max Risk: {zone['max_risk']:.3f}<br>
        Samples: {zone['sample_count']}<br>
        """

        # Convert Shapely geometry to GeoJSON
        geo_json = folium.GeoJson(
            zone['geometry'].__geo_interface__,
            style_function=lambda feature, c=color: {
                'fillColor': c,
                'color': '#333',
                'weight': 1,
                'fillOpacity': 0.5,
            },
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"Mean Risk: {zone['mean_risk']:.2f}"
        )
        geo_json.add_to(m)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Zone risk map saved: {output_path}")

    return m
