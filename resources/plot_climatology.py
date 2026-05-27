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
import logging
import rasterio
from rasterio.transform import from_origin
import glob

# --- LOGGING SETUP ---
logging.basicConfig(
    filename="pipeline.log",
    filemode="a",  # Append to log file
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("wwte.plot_climatology")

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
    show_sink_state_icon: bool = True


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
                    print(f"[BOUNDARIES] Loading {dataset_name} from local file: {os.path.relpath(local_path, self.base_dir)}")
                    logger.info(f"[BOUNDARIES] Loading {dataset_name} from local file: {os.path.relpath(local_path, self.base_dir)}")
                    return gpd.read_file(local_path)
                except Exception as e:
                    print(f"[BOUNDARIES WARNING] Failed reading local {dataset_name} ({local_path}): {e}")
                    logger.warning(f"[BOUNDARIES WARNING] Failed reading local {dataset_name} ({os.path.relpath(local_path, self.base_dir)}): {e}")

        for url in remote_candidates:
            try:
                print(f"[BOUNDARIES] Fetching {dataset_name} from remote source: {url}")
                logger.info(f"[BOUNDARIES] Fetching {dataset_name} from remote source: {url}")
                return gpd.read_file(url)
            except Exception as e:
                print(f"[BOUNDARIES WARNING] Failed downloading {dataset_name} from {url}: {e}")
                logger.warning(f"[BOUNDARIES WARNING] Failed downloading {dataset_name} from {url}: {e}")

        print(f"[BOUNDARIES WARNING] No usable {dataset_name} boundaries found.")
        logger.warning(f"[BOUNDARIES WARNING] No usable {dataset_name} boundaries found.")
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

        # If an extent/bounding box is provided, clip vector boundaries to that box
        try:
            if hasattr(self.config, 'extent') and self.config.extent is not None:
                xmin, xmax, ymin, ymax = self.config.extent[0], self.config.extent[1], self.config.extent[2], self.config.extent[3]
                from shapely.geometry import box
                bbox_geom = box(xmin, ymin, xmax, ymax)
                if self.world_gdf is not None and not self.world_gdf.empty:
                    try:
                        self.world_gdf = gpd.clip(self.world_gdf, bbox_geom)
                    except Exception:
                        # Fallback to .cx selection where available
                        try:
                            self.world_gdf = self.world_gdf.cx[xmin:xmax, ymin:ymax]
                        except Exception:
                            pass
                if self.iran_gdf is not None and not self.iran_gdf.empty:
                    try:
                        self.iran_gdf = gpd.clip(self.iran_gdf, bbox_geom)
                    except Exception:
                        try:
                            self.iran_gdf = self.iran_gdf.cx[xmin:xmax, ymin:ymax]
                        except Exception:
                            pass
        except Exception:
            # Be resilient: if clipping fails, continue with full boundaries
            logger.exception("Failed to clip boundaries to extent; proceeding with full boundaries.")

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
                # Compute plotting extent from the lon/lat grid to ensure
                # the plotted domain matches the underlying data (not only config)
                try:
                    xmin, xmax = float(lon2d.min()), float(lon2d.max())
                    ymin, ymax = float(lat2d.min()), float(lat2d.max())
                    data_extent = [xmin, xmax, ymin, ymax]
                except Exception:
                    data_extent = self.config.extent

                im = ax.imshow(
                    data,
                    extent=data_extent,
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

            # Mark Sink Location (optional star icon controlled by config)
            if getattr(self.config, 'show_sink_state_icon', True):
                ax.plot(
                    self.config.target_lon,
                    self.config.target_lat,
                    marker='*',
                    color='#1e90ff',
                    markersize=12,
                    markeredgecolor='black',
                    markeredgewidth=1.5,
                    zorder=5,
                    label=f"Sink: {self.config.sink_name}"
                )

            # Titles & Axes
            # Ensure axis limits match data grid if available
            try:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            except Exception:
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
        import re
        sink_slug = re.sub(r'[^A-Za-z0-9_-]+', '_', self.config.sink_name.strip().lower())
        out_path = os.path.join(self.config.plots_dir, f"climatology_wwte_score_{self.config.wind_file_suffix}_{sink_slug}_{mm}.png")
        # Ensure directory exists and save the figure
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            fig.savefig(out_path, dpi=300)
        finally:
            plt.close(fig)

        rel_out_path = os.path.relpath(out_path, self.config.plots_dir)
        print(f"[PLOT SUCCESS] Saved advanced plot: {os.path.join('outputs/plots', rel_out_path)}")
        logger.info(f"[PLOT SUCCESS] Saved advanced plot: {os.path.join('outputs/plots', rel_out_path)}")
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
        banner = "\n" + "="*60 + "\n INITIALIZING ADVANCED WWTE GEOSPATIAL VISUALIZATION PIPELINE\n" + "="*60
        print(banner)
        logger.info(banner)
        
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

        # Plot options
        show_icon = bool(self.config_data.get('plot_options', {}).get('show_sink_state_icon', True))

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
            ,show_sink_state_icon=show_icon
        )

        self.boundaries = GeospatialBoundaryManager(self.base_dir)
        self.plotter = PremiumPlotter(self.visualizer_config, self.boundaries)
        print("[INITIALIZATION COMPLETE] Visualization framework prepared successfully.\n")
        logger.info("[INITIALIZATION COMPLETE] Visualization framework prepared successfully.")

    def run(self) -> None:
        """
        Loads the combined climatology NetCDF file and generates advanced plots for all months.
        """
        if self.visualizer_config is None or self.plotter is None:
            raise RuntimeError("Pipeline must be initialized before running.")

        climatology_dir = os.path.join(self.visualizer_config.output_dir, 'climatology')
        plotted_files = []
        out_format = str(self.config_data.get('climatology_format', 'nc')).lower()

        import re
        sink_slug = re.sub(r'[^A-Za-z0-9_-]+', '_', self.visualizer_config.sink_name.strip().lower())

        if out_format in ('nc', 'netcdf'):
            # Try sink-specific filename first, then fallback to legacy name
            nc_filename = f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_{sink_slug}_combined.nc"
            nc_path = os.path.join(climatology_dir, nc_filename)

            if not os.path.exists(nc_path):
                legacy_nc = os.path.join(climatology_dir, f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_combined.nc")
                fallback_nc_path = os.path.join(self.visualizer_config.output_dir, f"wwte_{self.visualizer_config.wind_file_suffix}_combined.nc")
                if os.path.exists(legacy_nc):
                    nc_path = legacy_nc
                elif os.path.exists(fallback_nc_path):
                    nc_path = fallback_nc_path
                else:
                    raise FileNotFoundError(f"Climatology NetCDF file not found at: {nc_path} or legacy/fallback locations")

            rel_nc_path = os.path.relpath(nc_path, self.base_dir)
            print(f"[NETCDF LOAD] Reading combined monthly averages from: {rel_nc_path}")
            logger.info(f"[NETCDF LOAD] Reading combined monthly averages from: {rel_nc_path}")

            ds = xr.open_dataset(nc_path)
            lons = ds['lon'].values if 'lon' in ds.coords else ds['x'].values
            lats = ds['lat'].values if 'lat' in ds.coords else ds['y'].values
            lon2d, lat2d = np.meshgrid(lons, lats)

            # Plot all months present in dataset
            for month in range(1, 13):
                month_coord = 'month' if 'month' in ds.coords else 'time'
                if month_coord == 'month':
                    if month not in ds.month.values:
                        continue
                    ds_month = ds.sel(month=month)
                else:
                    matching_times = [t for t in ds.time.values if pd.to_datetime(t).month == month]
                    if not matching_times:
                        continue
                    ds_month = ds.sel(time=matching_times[0])

                plot_path = self.plotter.plot_month_climatology(month, ds_month, lon2d, lat2d)
                plotted_files.append(plot_path)
            ds.close()
        elif out_format in ('tif', 'tiff'):
            # Load per-month GeoTIFF layers and plot
            clim_dir = os.path.join(self.visualizer_config.output_dir, 'climatology')
            for month in range(1, 13):
                mm = f"{month:02d}"
                # Prefer a single multi-band GeoTIFF per month
                # Try sink-specific multi-band TIFF, then legacy name
                multi_path = os.path.join(clim_dir, f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_{sink_slug}_{mm}.tif")
                if not os.path.exists(multi_path):
                    multi_path = os.path.join(clim_dir, f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_{mm}.tif")
                if os.path.exists(multi_path):
                    with rasterio.open(multi_path) as src:
                        count = src.count
                        width = src.width
                        height = src.height
                        transform = src.transform
                        xs = transform.c + np.arange(width) * transform.a
                        ys = transform.f - np.arange(height) * abs(transform.e)
                        lons = xs
                        lats = ys

                        data_vars = {}
                        band_descriptions = src.descriptions if src.descriptions is not None else [f"band{i+1}" for i in range(count)]
                        for i in range(count):
                            arr = src.read(i+1).astype('float32')
                            if src.nodata is not None:
                                arr = np.where(arr == src.nodata, np.nan, arr)
                            varname = band_descriptions[i] if band_descriptions[i] is not None else f"band{i+1}"
                            data_vars[varname] = (('lat', 'lon'), arr)

                    ds_month = xr.Dataset(
                        {k: xr.DataArray(v[1], dims=('lat', 'lon')) for k, v in data_vars.items()},
                        coords={'lon': lons, 'lat': lats}
                    )

                    lon2d, lat2d = np.meshgrid(lons, lats)
                    plot_path = self.plotter.plot_month_climatology(month, ds_month, lon2d, lat2d)
                    plotted_files.append(plot_path)
                    continue

                # Fallback: read multiple single-band TIFFs for this month
                pattern = os.path.join(clim_dir, f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_*_{mm}.tif")
                tif_files = sorted([p for p in glob.glob(pattern)])
                if not tif_files:
                    continue

                # Read all TIFFs for the month and build an xarray.Dataset
                data_vars = {}
                lons = None
                lats = None
                for tif in tif_files:
                    varname = os.path.basename(tif).replace(f"wwte_climatology_{self.visualizer_config.wind_file_suffix}_", "")
                    varname = varname.rsplit(f"_{mm}.tif", 1)[0]
                    with rasterio.open(tif) as src:
                        arr = src.read(1).astype('float32')
                        # mask nodata
                        if src.nodata is not None:
                            arr = np.where(arr == src.nodata, np.nan, arr)
                        # compute lon/lat grid from transform
                        transform = src.transform
                        width = src.width
                        height = src.height
                        x0 = transform.c
                        y0 = transform.f
                        xres = transform.a
                        yres = -transform.e if transform.e < 0 else transform.e
                        xs = x0 + np.arange(width) * xres
                        ys = y0 - np.arange(height) * yres
                        # ensure lons/lats set once
                        if lons is None:
                            lons = xs
                        if lats is None:
                            lats = ys

                        data_vars[varname] = (('lat', 'lon'), arr)

                if lons is None or lats is None:
                    continue

                ds_month = xr.Dataset(
                    {k: xr.DataArray(v[1], dims=('lat', 'lon')) for k, v in data_vars.items()},
                    coords={'lon': lons, 'lat': lats}
                )

                lon2d, lat2d = np.meshgrid(lons, lats)
                plot_path = self.plotter.plot_month_climatology(month, ds_month, lon2d, lat2d)
                plotted_files.append(plot_path)
        else:
            raise ValueError(f"Unsupported climatology_format: {out_format}")

        print(f"\n[PIPELINE SUCCESS] Visualizer pipeline completed. Plotted {len(plotted_files)} month(s).")
        logger.info(f"[PIPELINE SUCCESS] Visualizer pipeline completed. Plotted {len(plotted_files)} month(s).")
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
