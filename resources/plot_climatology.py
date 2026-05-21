# File: plot_climatology_final.py
# Description:
# An advanced, object-oriented, and production-grade visualization engine for the 
# Wind-Weighted AOD Transport Efficiency (WWTE) pipeline climatology.
# Uses modular design, strong type hinting, and premium Matplotlib styling.
#
# Author: Hossein Lotfi — Research Scientist
#
# How to run:
# python resources/plot_climatology_final.py
#

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from geopy.geocoders import Nominatim

# --- CONSTANTS ---
EARTH_RADIUS_KM: float = 6371.0
DEFAULT_DECAY_LENGTH_KM: float = 800.0


@dataclass
class VisualizerConfig:
    """
    Data container for configuration parameters, styling options, and directory paths.
    """
    sink_name: str
    target_lon: float
    target_lat: float
    active_wind_type: str
    wind_file_suffix: str
    wind_label: str
    output_dir: str
    plots_dir: str
    decay_length_km: float = DEFAULT_DECAY_LENGTH_KM
    extent: List[float] = field(default_factory=lambda: [30.0, 75.0, 11.0, 53.0])
    color_palette: List[str] = field(default_factory=lambda: ['#ffcc00', '#ff6600', '#cc0000', '#660000'])
    categories: List[str] = field(default_factory=lambda: ['Low', 'Medium', 'High', 'Extreme'])


class GeospatialBoundaryManager:
    """
    Manages fetching and loading of cultural and political vector boundaries (countries, provinces).
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _load_boundaries(
        self,
        dataset_name: str,
        local_candidates: List[str],
        remote_candidates: List[str]
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Loads boundaries using a local-first strategy, with remote fallback.
        """
        for rel_path in local_candidates:
            local_path = os.path.join(self.base_dir, rel_path)
            if os.path.exists(local_path):
                try:
                    print(f"[BOUNDARIES] Loading {dataset_name} from local file: {local_path}")
                    return gpd.read_file(local_path)
                except Exception as e:
                    print(f"[BOUNDARIES WARNING] Failed reading local {dataset_name} ({local_path}): {e}")

        for url in remote_candidates:
            try:
                print(f"[BOUNDARIES] Fetching {dataset_name} from remote source: {url}")
                return gpd.read_file(url)
            except Exception as e:
                print(f"[BOUNDARIES WARNING] Failed downloading {dataset_name} from {url}: {e}")

        print(f"[BOUNDARIES WARNING] No usable {dataset_name} boundaries found.")
        return None

    def load_world_boundaries(self) -> Optional[gpd.GeoDataFrame]:
        """
        Loads world country boundaries from local files, with online fallback.
        """
        return self._load_boundaries(
            dataset_name="world countries",
            local_candidates=[
                "inputs/Shpefile/world_countries.geojson",
                "inputs/Shpefile/countries.geojson"
            ],
            remote_candidates=[
                "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
                "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
            ]
        )

    def load_iran_provinces(self) -> Optional[gpd.GeoDataFrame]:
        """
        Loads Iran provinces boundaries from local files, with online fallback.
        """
        return self._load_boundaries(
            dataset_name="Iran provinces",
            local_candidates=[
                "inputs/Shpefile/iran_provinces.geojson",
                "inputs/Shpefile/provinces.geojson"
            ],
            remote_candidates=[
                "https://raw.githubusercontent.com/mrunderline/iran-geojson/master/ir_states_boundaries_coordinates.geojson"
            ]
        )


class WWTEGeospatialEngine:
    """
    Provides mathematical utilities for calculating bearing vectors, distance decay, and score components.
    """
    @staticmethod
    def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
        """
        Calculates great-circle distance in kilometers from grid points to target coordinates.
        """
        rlat1 = np.radians(lat1)
        rlat2 = math.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(rlat1) * math.cos(rlat2) * np.sin(dlon / 2) ** 2)
        return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def bearing_unit_vectors(lon2d: np.ndarray, lat2d: np.ndarray, target_lon: float, target_lat: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates bearing unit vectors pointing toward the sink, correcting for latitude convergence.
        """
        cos_lat = np.cos(np.radians((lat2d + target_lat) / 2.0))
        dlon = (target_lon - lon2d) * cos_lat
        dlat = target_lat - lat2d
        mag = np.sqrt(dlon ** 2 + dlat ** 2)
        mag = np.where(mag == 0, np.nan, mag)
        return (dlon / mag).astype("float32"), (dlat / mag).astype("float32")

    @classmethod
    def compute_transport_layers(
        cls, 
        aod: np.ndarray, 
        u: np.ndarray, 
        v: np.ndarray, 
        lon2d: np.ndarray, 
        lat2d: np.ndarray, 
        mask: np.ndarray, 
        config: VisualizerConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes 2D grids of:
          - Omega (speed-normalized wind direction alignment weight)
          - Score (AOD weighted by wind speed, direction, and distance decay)
          - Score * Omega (intensity weighted score highlighting hotspots)
        """
        dx_hat, dy_hat = cls.bearing_unit_vectors(lon2d, lat2d, config.target_lon, config.target_lat)
        wind_mag = np.sqrt(u ** 2 + v ** 2)

        valid = mask & np.isfinite(aod) & np.isfinite(u) & np.isfinite(v) & (wind_mag > 0)

        u_hat = np.where(wind_mag > 0, u / wind_mag, np.nan)
        v_hat = np.where(wind_mag > 0, v / wind_mag, np.nan)

        cosang = u_hat * dx_hat + v_hat * dy_hat
        toward = valid & (cosang > 0)

        speed_max = float(np.nanmax(wind_mag[toward])) if np.any(toward) else 1.0
        speed_norm = wind_mag / max(speed_max, 1e-6)

        dist_km = cls.haversine_km(lon2d, lat2d, config.target_lon, config.target_lat)
        dist_decay = np.exp(-dist_km / config.decay_length_km)

        omega = np.full(aod.shape, np.nan, dtype="float32")
        omega[toward] = np.maximum(cosang[toward], 0.0) * speed_norm[toward]

        score = np.full(aod.shape, np.nan, dtype="float32")
        score[toward] = aod[toward] * omega[toward] * dist_decay[toward]

        score_omega = np.full(aod.shape, np.nan, dtype="float32")
        score_omega[toward] = score[toward] * omega[toward]

        return omega, score, score_omega


class PremiumPlotter:
    """
    Handles plotting of highly detailed, multi-panel visual maps with custom colormaps.
    """
    def __init__(self, config: VisualizerConfig, boundaries: GeospatialBoundaryManager):
        self.config = config
        self.boundaries = boundaries
        self.world_gdf = boundaries.load_world_boundaries()
        self.iran_gdf = boundaries.load_iran_provinces()

    def generate_discrete_colormap(self, data: np.ndarray) -> Tuple[ListedColormap, BoundaryNorm, List[float]]:
        """
        Creates custom high-contrast color steps based on the empirical quantiles of the active data.
        """
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0 and np.max(valid_data) > 0:
            q25 = np.percentile(valid_data, 25)
            q50 = np.percentile(valid_data, 50)
            q75 = np.percentile(valid_data, 75)
            vmax = np.max(valid_data)
            
            bounds = [0.0, max(0.0001, q25), max(0.0002, q50), max(0.0003, q75), vmax + 0.0001]
            bounds = sorted(list(set(bounds)))
            if len(bounds) < 5:
                bounds = list(np.linspace(0.0, vmax + 0.0001, 5))
        else:
            bounds = [0.0, 0.25, 0.5, 0.75, 1.0]

        cmap = ListedColormap(self.config.color_palette)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, bounds

    def plot_month_climatology(self, month: int, ds_month: xr.Dataset, lon2d: np.ndarray, lat2d: np.ndarray) -> str:
        """
        Plots a high-fidelity 3-panel figure representing Score, Omega, and Score*Omega for a single month.
        """
        mm = f"{month:02d}"
        aod = ds_month["AOD"].values
        u = ds_month["U"].values
        v = ds_month["V"].values
        mask = ds_month["Hotspot_Mask"].values > 0

        omega, score, score_omega = WWTEGeospatialEngine.compute_transport_layers(aod, u, v, lon2d, lat2d, mask, self.config)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

        panels = [
            ("Score", score, "Geospatial transport score factoring AOD, wind alignment and distance decay"),
            ("Omega", omega, "Wind dynamics factor (alignment cosine * normalized speed)"),
            ("Score $\\times$ Omega", score_omega, "Intensity weighted transport hotspots")
        ]

        for i, (title, data, subtitle) in enumerate(panels):
            ax = axes[i]
            
            # Draw base vector outlines
            if self.world_gdf is not None:
                self.world_gdf.boundary.plot(ax=ax, color='#111111', linewidth=0.8, zorder=2)
            if self.iran_gdf is not None:
                self.iran_gdf.boundary.plot(ax=ax, color='dimgray', linewidth=0.4, linestyle='--', zorder=2)

            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0 and np.max(valid_data) > 0:
                cmap, norm, bounds = self.generate_discrete_colormap(data)
                
                # Image display
                im = ax.imshow(
                    data, 
                    extent=self.config.extent, 
                    cmap=cmap, 
                    norm=norm, 
                    origin='upper',
                    zorder=1
                )
                
                # Custom colorbar
                ticks = [(bounds[k] + bounds[k+1])/2 for k in range(len(bounds)-1)]
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
                cbar.ax.set_yticklabels(self.config.categories, fontweight='bold')
                cbar.ax.tick_params(labelsize=12)
            else:
                ax.text(
                    0.5, 0.5, 'No active dust transport', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=16, color='gray', fontstyle='italic'
                )

            # Mark Sink Location
            ax.plot(
                self.config.target_lon, 
                self.config.target_lat, 
                marker='*', 
                color='#1e90ff', 
                markersize=18, 
                markeredgecolor='black', 
                markeredgewidth=1.5,
                zorder=5,
                label=f"Sink: {self.config.sink_name}"
            )

            # Titles & Axes
            ax.set_xlim(self.config.extent[0], self.config.extent[1])
            ax.set_ylim(self.config.extent[2], self.config.extent[3])
            ax.set_title(f"{title} ({self.config.wind_label} - Month {mm})", fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel('Longitude (°E)', fontsize=12, labelpad=10)
            ax.set_ylabel('Latitude (°N)', fontsize=12, labelpad=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

        plt.suptitle(
            f"WWTE Climatological Diagnostics for {self.config.sink_name} (Wind: {self.config.wind_label}) — Month {mm}", 
            fontsize=22, 
            fontweight='bold', 
            y=0.98
        )
        
        plt.tight_layout()
        out_filename = f"climatology_wwte_score_{self.config.wind_file_suffix}_{mm}.png"
        out_path = os.path.join(self.config.plots_dir, out_filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[PLOT SUCCESS] Saved advanced plot: {out_path}")
        return out_path


class WWTEVisualizerPipeline:
    """
    Coordinator class managing loading configurations, loading dataset, and initiating plotting engine.
    """
    def __init__(self, config_rel_path: str):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, config_rel_path)
        self.config_data: Optional[Dict[str, Any]] = None
        self.visualizer_config: Optional[VisualizerConfig] = None
        self.boundaries: Optional[GeospatialBoundaryManager] = None
        self.plotter: Optional[PremiumPlotter] = None

    def _resolve_wind_config(self) -> Tuple[str, str, str]:
        """Resolves active_wind_type to canonical suffix and display label."""
        wind_type_raw = str(self.config_data.get("active_wind_type", "wind10m")).strip()
        wind_type = wind_type_raw.lower()
        if wind_type in {"10m", "wind10m"}:
            return wind_type_raw, "wind10m", "10m"
        if wind_type in {"850", "850mb", "wind850mb"}:
            return wind_type_raw, "wind850mb", "850mb"
        raise ValueError(
            f"Unsupported active_wind_type: {wind_type_raw}. "
            "Use 'wind10m' or 'wind850mb'."
        )

    def initialize(self) -> None:
        """
        Parses configuration and coordinates geocoding if necessary.
        """
        print("="*60)
        print(" INITIALIZING ADVANCED WWTE GEOSPATIAL VISUALIZATION PIPELINE")
        print("="*60)
        
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)
            
        dirs = self.config_data['directories']
        params = self.config_data['parameters']
        sink_cfg = self.config_data['sink_location']
        sink_name = sink_cfg['name']
        
        # Geocoding Lookup if explicit lat/lon is omitted
        if 'lon' in sink_cfg and 'lat' in sink_cfg:
            lon = float(sink_cfg['lon'])
            lat = float(sink_cfg['lat'])
        else:
            print(f"[GEOCODE] Coordinates missing in config. Looking up '{sink_name}'...")
            geolocator = Nominatim(user_agent="wwte_pipeline_plot_final")
            location = geolocator.geocode(sink_name)
            if location is None:
                raise ValueError(f"Could not resolve coordinates for '{sink_name}'.")
            lon = location.longitude
            lat = location.latitude
            print(f"[GEOCODE] Resolved '{sink_name}' -> lon={lon:.4f}, lat={lat:.4f}")

        # Active wind selection
        wind_type, wind_file_suffix, wind_label = self._resolve_wind_config()

        bbox = params['bounding_box']
        extent = [bbox['min_lon'], bbox['max_lon'], bbox['min_lat'], bbox['max_lat']]
        decay_length = params.get('decay_length_km', DEFAULT_DECAY_LENGTH_KM)

        plots_subdir = dirs.get('plots', 'outputs/plots')
        plots_dir = os.path.join(self.base_dir, plots_subdir)
        os.makedirs(plots_dir, exist_ok=True)
        
        output_dir = os.path.join(self.base_dir, dirs['output'])

        self.visualizer_config = VisualizerConfig(
            sink_name=sink_name.split(',')[0].strip(),
            target_lon=lon,
            target_lat=lat,
            active_wind_type=wind_type,
            wind_file_suffix=wind_file_suffix,
            wind_label=wind_label,
            output_dir=output_dir,
            plots_dir=plots_dir,
            decay_length_km=decay_length,
            extent=extent
        )

        self.boundaries = GeospatialBoundaryManager(self.base_dir)
        self.plotter = PremiumPlotter(self.visualizer_config, self.boundaries)
        print("[INITIALIZATION COMPLETE] Visualization framework prepared successfully.\n")

    def run(self) -> None:
        """
        Loads the combined climatology NetCDF file and generates advanced plots for all months.
        """
        if self.visualizer_config is None or self.plotter is None:
            raise RuntimeError("Pipeline must be initialized before running.")

        climatology_dir = os.path.join(self.visualizer_config.output_dir, 'climatology')
        nc_filename = f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_combined.nc"
        nc_path = os.path.join(climatology_dir, nc_filename)

        if not os.path.exists(nc_path):
            # Try loading from the root of the results directory as a fallback
            fallback_nc_path = os.path.join(self.visualizer_config.output_dir, f"wwte_{self.visualizer_config.wind_file_suffix}_combined.nc")
            if os.path.exists(fallback_nc_path):
                nc_path = fallback_nc_path
            else:
                raise FileNotFoundError(f"Climatology NetCDF file not found at: {nc_path} or {fallback_nc_path}")

        print(f"[NETCDF LOAD] Reading combined monthly averages from: {nc_path}")
        ds = xr.open_dataset(nc_path)
        
        # Grid parameters (support both lat/lon and legacy y/x naming)
        lons = ds['lon'].values if 'lon' in ds.coords else ds['x'].values
        lats = ds['lat'].values if 'lat' in ds.coords else ds['y'].values
        lon2d, lat2d = np.meshgrid(lons, lats)

        # Plot all months present in dataset
        plotted_files = []
        for month in range(1, 13):
            # Select month dimension (depending on coordinate name 'month' or 'time.month')
            month_coord = 'month' if 'month' in ds.coords else 'time'
            
            if month_coord == 'month':
                if month not in ds.month.values:
                    continue
                ds_month = ds.sel(month=month)
            else:
                # time coordinates
                matching_times = [t for t in ds.time.values if pd.to_datetime(t).month == month]
                if not matching_times:
                    continue
                ds_month = ds.sel(time=matching_times[0])

            plot_path = self.plotter.plot_month_climatology(month, ds_month, lon2d, lat2d)
            plotted_files.append(plot_path)

        ds.close()
        print(f"\n[PIPELINE SUCCESS] Visualizer pipeline completed. Plotted {len(plotted_files)} month(s).")
        print("="*60)


def main() -> None:
    """
    Main visualizer script entry point.
    """
    try:
        pipeline = WWTEVisualizerPipeline("config/config.json")
        pipeline.initialize()
        pipeline.run()
    except Exception as e:
        print("\n" + "#"*40)
        print(" [FATAL ERROR] Visualizer pipeline crashed!")
        print("#"*40 + "\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
