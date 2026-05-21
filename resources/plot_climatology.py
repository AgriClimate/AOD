# File: plot_climatology.py
# Description:
# This script reads the single combined monthly climatology NetCDF file and generates 
# 3-panel geospatial plots showing Omega, Score, and Score * Omega side-by-side.
# It supports dynamic configurations for the active sink, output directories,
# and incorporates the active wind type ('wind10m' or 'wind850hp') in its naming.
#
# How to run:
# python resources/plot_climatology.py
#
# Dependencies:
# - Python 3.10+
# - numpy, xarray, matplotlib, geopandas
#
# Expected inputs:
# - Combined climatology NetCDF file in data/results/climatology/
# - config.json defining sink coordinates, target plots directory, and active_wind_type
#
# Expected outputs:
# - Climatology maps as PNGs saved in the plots directory (e.g. plots/climatology_wwte_score_{wind_type}_{MM}.png)

import json
import os
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Tuple


EARTH_RADIUS_KM = 6371.0
DECAY_LENGTH_KM = 800.0


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    """
    Calculates great-circle distance (km) from every grid pixel to a target point 
    using the Haversine formula.

    Args:
        lon1 (np.ndarray): 2D array of longitudes.
        lat1 (np.ndarray): 2D array of latitudes.
        lon2 (float): Target longitude.
        lat2 (float): Target latitude.

    Returns:
        np.ndarray: 2D array of distances in kilometers.
    """
    rlat1 = np.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(rlat1) * math.cos(rlat2) * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def bearing_unit_vector(lon2d: np.ndarray, lat2d: np.ndarray, target_lon: float, target_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates bearing unit vector components pointing from each grid cell toward the target sink.

    Args:
        lon2d (np.ndarray): 2D array of longitudes.
        lat2d (np.ndarray): 2D array of latitudes.
        target_lon (float): Target longitude.
        target_lat (float): Target latitude.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The (dx_hat, dy_hat) components of the unit vector.
    """
    cos_lat = np.cos(np.radians((lat2d + target_lat) / 2.0))
    dlon = (target_lon - lon2d) * cos_lat
    dlat = target_lat - lat2d
    mag = np.sqrt(dlon ** 2 + dlat ** 2)
    mag = np.where(mag == 0, np.nan, mag)
    return (dlon / mag).astype("float32"), (dlat / mag).astype("float32")


def compute_components(aod: np.ndarray, u: np.ndarray, v: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, 
                       mask: np.ndarray, target_lon: float, target_lat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates intermediate transport parameters: Omega (wind alignment and magnitude), 
    raw Score, and Score * Omega.

    Args:
        aod (np.ndarray): 2D array of AOD values.
        u (np.ndarray): 2D array of U wind component.
        v (np.ndarray): 2D array of V wind component.
        lon2d (np.ndarray): 2D array of grid longitudes.
        lat2d (np.ndarray): 2D array of grid latitudes.
        mask (np.ndarray): 2D boolean mask of AOD source region.
        target_lon (float): Target sink longitude.
        target_lat (float): Target sink latitude.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Omega, Score, and Score * Omega 2D arrays.
    """
    dx_hat, dy_hat = bearing_unit_vector(lon2d, lat2d, target_lon, target_lat)
    wind_mag = np.sqrt(u ** 2 + v ** 2)

    valid = mask & np.isfinite(aod) & np.isfinite(u) & np.isfinite(v) & (wind_mag > 0)

    u_hat = np.where(wind_mag > 0, u / wind_mag, np.nan)
    v_hat = np.where(wind_mag > 0, v / wind_mag, np.nan)

    cosang = u_hat * dx_hat + v_hat * dy_hat
    toward = valid & (cosang > 0)

    speed_max = float(np.nanmax(wind_mag[toward])) if np.any(toward) else 1.0
    speed_norm = wind_mag / max(speed_max, 1e-6)

    dist_km = haversine_km(lon2d, lat2d, target_lon, target_lat)
    dist_decay = np.exp(-dist_km / DECAY_LENGTH_KM)

    omega = np.full(aod.shape, np.nan, dtype="float32")
    omega[toward] = np.maximum(cosang[toward], 0.0) * speed_norm[toward]

    score = np.full(aod.shape, np.nan, dtype="float32")
    score[toward] = aod[toward] * omega[toward] * dist_decay[toward]

    score_omega = np.full(aod.shape, np.nan, dtype="float32")
    score_omega[toward] = score[toward] * omega[toward]

    return omega, score, score_omega


def main() -> None:
    """
    Loads consolidated NetCDF climatology layers and plots monthly maps side-by-side using dynamic configurations.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'resources', 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dirs = config['directories']
    climatology_dir = os.path.join(base_dir, dirs['output'], 'climatology')
    
    plots_subdir = dirs.get('plots', 'plots')
    plots_dir = os.path.join(base_dir, plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    
    wind_type = config.get("active_wind_type", "wind850mb")
    
    # Enforce strict naming conventions requested: 'wind10m', 'wind850mb', or 'wind850hp'
    if wind_type in ["10m", "wind10m"]:
        wind_file_suffix = "wind10m"
        wind_label = "10m"
    elif "850mb" in wind_type or "850" in wind_type:
        wind_file_suffix = "wind850mb"
        wind_label = "850mb"
    else:
        wind_file_suffix = "wind850hp"
        wind_label = "850hp"
    
    nc_path = os.path.join(climatology_dir, f"wwte_climatology_{wind_file_suffix}_combined.nc")
    if not os.path.exists(nc_path):
        print(f"Error: Combined climatology file not found: {nc_path}. Run climatology script first.")
        return
        
    try:
        world = gpd.read_file("https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip")
    except Exception as e:
        print(f"Failed to load world map: {e}")
        world = None

    # Load Iran provinces boundaries GeoJSON
    iran_provinces_path = os.path.join(base_dir, 'data', 'iran_provinces.geojson')
    iran_provinces = None
    if os.path.exists(iran_provinces_path):
        try:
            print(f"Loading Iran provinces boundaries locally: {iran_provinces_path}")
            iran_provinces = gpd.read_file(iran_provinces_path)
        except Exception as e:
            print(f"Failed to load local Iran provinces: {e}")

    if iran_provinces is None:
        try:
            print("Attempting to load Iran provinces boundaries from public GeoJSON repository...")
            iran_provinces = gpd.read_file("https://raw.githubusercontent.com/mrunderline/iran-geojson/master/ir_states_boundaries_coordinates.geojson")
        except Exception as e:
            print(f"Failed to fetch online Iran provinces map: {e}")
            iran_provinces = None

    sink_name = config['sink_location']['name']
    if 'lon' in config['sink_location'] and 'lat' in config['sink_location']:
        s_lon = config['sink_location']['lon']
        s_lat = config['sink_location']['lat']
    else:
        from geopy.geocoders import Nominatim
        print(f"[GEOCODE] Looking up coordinates for '{sink_name}' in plotting script...")
        geolocator = Nominatim(user_agent="wwte_pipeline_plot")
        location = geolocator.geocode(sink_name)
        if location is None:
            raise ValueError(f"Could not geocode '{sink_name}' in plotting script.")
        s_lon = location.longitude
        s_lat = location.latitude
        print(f"[GEOCODE] Resolved '{sink_name}' -> lon={s_lon:.4f}, lat={s_lat:.4f}")
    
    print(f"Opening combined climatology dataset: {os.path.basename(nc_path)}")
    ds = xr.open_dataset(nc_path)
    
    x_min, x_max = float(ds.x.min()), float(ds.x.max())
    y_min, y_max = float(ds.y.min()), float(ds.y.max())
    lons = ds.x.values
    lats = ds.y.values
    lon2d, lat2d = np.meshgrid(lons, lats)
    extent = [x_min, x_max, y_min, y_max]
    
    for month in range(1, 13):
        mm = f"{month:02d}"
        
        # Check if the month index exists in the coordinate
        if month not in ds.month.values:
            continue
            
        print(f"Plotting climatology for month {mm} using active wind '{wind_label}'...")
        ds_month = ds.sel(month=month)
        
        aod = ds_month["AOD"].values
        u = ds_month["U"].values
        v = ds_month["V"].values
        hotspot_mask = ds_month["Hotspot_Mask"].values > 0
        
        omega, score, score_omega = compute_components(aod, u, v, lon2d, lat2d, hotspot_mask, s_lon, s_lat)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        components = [
            ("Score", score),
            ("Omega", omega),
            ("Score $\\times$ Omega", score_omega)
        ]
        
        for j, (title, data) in enumerate(components):
            ax = axes[j]
            valid_data = data[np.isfinite(data)]
            
            if len(valid_data) > 0 and np.max(valid_data) > 0:
                q25 = np.percentile(valid_data, 25)
                q50 = np.percentile(valid_data, 50)
                q75 = np.percentile(valid_data, 75)
                vmax = np.max(valid_data)
                
                # Create dynamic bounds to ensure categories are visible even for small values
                bounds = [0, max(0.0001, q25), max(0.0002, q50), max(0.0003, q75), vmax + 0.0001]
                bounds = sorted(list(set(bounds)))  # remove duplicates if any
                
                if len(bounds) < 5:
                    bounds = list(np.linspace(0, vmax + 0.0001, 5))
                    
                colors = ['#ffcc00', '#ff6600', '#cc0000', '#660000']
                cmap_discrete = ListedColormap(colors)
                norm_discrete = BoundaryNorm(bounds, cmap_discrete.N)
                
                im = ax.imshow(data, extent=extent, cmap=cmap_discrete, norm=norm_discrete, 
                               origin='upper' if lats[0] > lats[-1] else 'lower')
                
                ticks = [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
                cbar.ax.set_yticklabels(['Low', 'Medium', 'High', 'Extreme'])
                cbar.ax.tick_params(labelsize=14)
            else:
                ax.text(0.5, 0.5, 'No valid transport', horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontsize=16)

            if world is not None:
                world.boundary.plot(ax=ax, color='black', linewidth=0.8)
                
            if iran_provinces is not None:
                # Plot Iran province boundaries with very thin, elegant dashed lines in dark gray
                iran_provinces.boundary.plot(ax=ax, color='dimgray', linewidth=0.4, linestyle='--')
                
            ax.plot(s_lon, s_lat, 'b*', markersize=18, label=sink_name)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # Enforce title references '10m' or '850hp'
            ax.set_title(f'{title} ({wind_label} Month {mm})', fontsize=20)
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)
            ax.legend(fontsize=14, loc='upper right')
            
        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"climatology_wwte_score_{wind_file_suffix}_{mm}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    ds.close()


if __name__ == "__main__":
    main()
