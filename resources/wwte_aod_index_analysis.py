# File: wwte_aod_index_analysis_final.py
# Description:
# This is an advanced, object-oriented, and highly modularized implementation of the 
# Wind-Weighted AOD Transport Efficiency (WWTE) pipeline.
#
# Author: Hossein Lotfi — Research Scientist
#
# How to run:
# python resources/wwte_aod_index_analysis_final.py
#
# Expected inputs:
# - config.json
# - Satellite AOD combined grids, hotspot masks, and wind datasets.

import glob
import json
import math
import os
import sys
import traceback
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from geopy.geocoders import Nominatim
from shapely.geometry import mapping


# --- CONSTANTS ---
EARTH_RADIUS_KM: float = 6371.0



def profile_stage(func):
    """
    A decorator designed to log and profile execution stages.
    """
    def wrapper(*args, **kwargs):
        print(f"\n[PIPELINE STAGE] Running: {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            print(f"[PIPELINE STAGE] Finished: {func.__name__} successfully.")
            return result
        except Exception as e:
            print(f"[PIPELINE ERROR] Failed in stage {func.__name__}: {e}")
            raise e
    return wrapper


class SinkLocation:
    def __init__(self, name: str, lon: float, lat: float, buffer_deg: float) -> None:
        self.name = name
        self.lon = lon
        self.lat = lat
        self.buffer_deg = buffer_deg


class WWTEGeospatialEngine:
    """
    Core engine handling spatial coordinate mathematics and physical transport scoring.
    """
    
    @staticmethod
    def haversine_distance_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
        """
        Calculates great-circle distance in kilometers from every grid cell to the sink.
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
        Calculates the unit vector pointing from each cell toward the sink, accounting for lat convergence.
        """
        cos_lat = np.cos(np.radians((lat2d + target_lat) / 2.0))
        dlon = (target_lon - lon2d) * cos_lat
        dlat = target_lat - lat2d
        mag = np.sqrt(dlon ** 2 + dlat ** 2)
        mag = np.where(mag == 0, np.nan, mag)
        return (dlon / mag).astype("float32"), (dlat / mag).astype("float32")


class ConfigManager:
    """
    Manages loading and validation of pipeline parameters.
    """
    
    @staticmethod
    @profile_stage
    def load_parameters(config_path: str) -> Dict[str, Any]:
        """
        Loads parameters from the central config file.
        """
        with open(config_path, 'r') as f:
            return json.load(f)

    @classmethod
    def parse_sink_location(cls, config: Dict[str, Any]) -> SinkLocation:
        """
        Parses and validates sink coordinates.
        Supports two modes:
          1. Explicit: {"name": "Zabol", "lon": 61.49, "lat": 31.03}
          2. Geocode: {"name": "Zabol, Iran"} — lat/lon looked up automatically
        """
        sink_cfg = config['sink_location']
        params = config['parameters']
        name = sink_cfg['name']

        if 'lon' in sink_cfg and 'lat' in sink_cfg:
            lon = float(sink_cfg['lon'])
            lat = float(sink_cfg['lat'])
        else:
            # Geocode the city name using Nominatim (OpenStreetMap)
            print(f"[GEOCODE] Looking up coordinates for '{name}'...")
            geolocator = Nominatim(user_agent="wwte_pipeline")
            location = geolocator.geocode(name)
            if location is None:
                raise ValueError(
                    f"Could not geocode '{name}'. Provide explicit lon/lat in config "
                    f"or use a more specific name (e.g., 'Zabol, Iran')."
                )
            lon = location.longitude
            lat = location.latitude
            print(f"[GEOCODE] Resolved '{name}' → lon={lon:.4f}, lat={lat:.4f}")

        buffer_val = params.get(f"{name.split(',')[0].strip().lower()}_buffer_deg", 0.3)
        return SinkLocation(name=name.split(',')[0].strip(), lon=lon, lat=lat, buffer_deg=buffer_val)


class WWTEPipeline:
    """
    The main coordinator class orchestrating dataset loading, subsetting, and calculation.
    """
    
    def __init__(self, config_path: str):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, config_path)
        self.config: Optional[Dict[str, Any]] = None
        self.sink: Optional[SinkLocation] = None
        self.monthly_datasets: List[xr.Dataset] = []
        self.records: List[Dict[str, Any]] = []

    @profile_stage
    def initialize(self) -> None:
        """Loads configuration and initializes targets."""
        self.config = ConfigManager.load_parameters(self.config_path)
        self.sink = ConfigManager.parse_sink_location(self.config)

    def _resolve_wind_suffix(self) -> str:
        """Maps active_wind_type from config to the canonical output/input suffix."""
        cfg = self.config or {}
        wind_type = str(cfg.get("active_wind_type", "wind10m")).strip().lower()
        if wind_type in {"10m", "wind10m"}:
            return "wind10m"
        if wind_type in {"850", "850mb", "wind850mb"}:
            return "wind850mb"
        raise ValueError(
            f"Unsupported active_wind_type: {cfg.get('active_wind_type')}. "
            "Use 'wind10m' or 'wind850mb'."
        )

    def _resolve_aod_source_mode(self) -> str:
        """Resolves how AOD pixels are selected before transport calculations."""
        cfg = self.config or {}
        mode = str(cfg.get("aod_source_mode", "hotspot")).strip().lower()
        if mode in {"hotspot", "hotspots"}:
            return "hotspot"
        if mode in {"all", "all_pixels", "allpixels"}:
            return "all"
        raise ValueError(
            f"Unsupported aod_source_mode: {cfg.get('aod_source_mode')}. "
            "Use 'hotspot' or 'all'."
        )

    def _resolve_aod_threshold(self) -> Optional[float]:
        """Returns optional AOD threshold; None means disabled."""
        cfg = self.config or {}
        raw = cfg.get("aod_threshold", None)
        if raw is None:
            return None
        if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
            return None
        threshold = float(raw)
        if threshold < 0:
            raise ValueError("aod_threshold must be >= 0 when provided.")
        return threshold

    @staticmethod
    def _hotspot_path_for_month(dir_hotspot: str, month: str) -> str:
        """Builds the canonical hotspot mask path for a given month (01..12)."""
        mm = f"{int(month):02d}"
        return os.path.join(dir_hotspot, f"binary_Avg_AOD_24yr_Month{mm}_2001_2024.tif")

    def _get_country_mask(self, country_name: str, lons: np.ndarray, lats: np.ndarray, ref_ds: xr.Dataset) -> np.ndarray:
        """
        Creates a boolean mask for the specified country using Natural Earth boundaries.
        """
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except Exception:
            # For newer geopandas versions where datasets is deprecated
            import geodatasets
            world = gpd.read_file(geodatasets.data.naturalearth.land110)

        country = world[world['name'].str.lower() == country_name.lower()]
        if country.empty:
            # Try partial match
            country = world[world['name'].str.lower().str.contains(country_name.lower())]
        if country.empty:
            print(f"[WARNING] Country '{country_name}' not found. Using full domain.")
            return np.ones((len(lats), len(lons)), dtype=bool)

        # Create a dummy DataArray on the reference grid and clip to country geometry
        dummy = xr.DataArray(
            np.ones((len(lats), len(lons)), dtype=np.float32),
            dims=('y', 'x'),
            coords={'y': lats, 'x': lons}
        )
        dummy.rio.write_crs("EPSG:4326", inplace=True)
        try:
            clipped = dummy.rio.clip(country.geometry.apply(mapping), all_touched=True)
            mask = np.isfinite(clipped.values)
        except Exception:
            print(f"[WARNING] Failed to clip to '{country_name}'. Using full domain.")
            mask = np.ones((len(lats), len(lons)), dtype=bool)
        return mask

    @profile_stage
    def run_spatial_analysis(self) -> None:
        """Executes the core calculations on all monthly inputs."""
        dirs = self.config['directories']
        params = self.config['parameters']
        
        dir_hotspot = os.path.join(self.base_dir, dirs['hotspot_binary'])
        dir_aod = os.path.join(self.base_dir, dirs['aod_combined'])
        dir_wind = os.path.join(self.base_dir, dirs['wind'])
        
        wind_file_suffix = self._resolve_wind_suffix()
        
        bbox = params['bounding_box']
        res = params['resolution_deg']
        decay_length = params['decay_length_km']
        q_high = params['af_high_aod_quantile']
        aod_source_mode = self._resolve_aod_source_mode()
        aod_threshold = self._resolve_aod_threshold()
        
        # Reference grids
        lons = np.arange(bbox['min_lon'], bbox['max_lon'] + res/2, res)
        lats = np.arange(bbox['max_lat'], bbox['min_lat'] - res/2, -res)
        lon2d, lat2d = np.meshgrid(lons, lats)
        ref_ds = xr.Dataset(coords={'y': lats, 'x': lons})
        ref_ds.rio.write_crs("EPSG:4326", inplace=True)
        
        # Load wind global database
        wind_path = os.path.join(
            dir_wind, 
            "era5_monthly_wind10m_combined.nc" if wind_file_suffix == "wind10m" else "era5_monthly_wind850mb_combined.nc"
        )
        combined_wind = xr.open_dataset(wind_path)
        

        aod_files = sorted(glob.glob(os.path.join(dir_aod, 'MCDAL2_M_AER_OD_*-*.FLOAT.TIFF')))
        
        for aod_path in aod_files:
            filename = os.path.basename(aod_path)
            year_month = filename.split('_')[4].split('.')[0]
            year, month = year_month.split('-')
            
            time_str = f"{year}-{month}-01"
            ds_wind_month = None
            for coord in ["valid_time", "time"]:
                if coord in combined_wind.coords:
                    try:
                        ds_wind_month = combined_wind.sel({coord: time_str}, method='nearest')
                        break
                    except Exception:
                        pass
                        
            if ds_wind_month is None:
                continue
                
            hotspot_path = self._hotspot_path_for_month(dir_hotspot, month)
            if not os.path.exists(hotspot_path):
                continue
                
            # Process single slice
            try:
                da_aod_raw = rioxarray.open_rasterio(aod_path).isel(band=0)
                da_hotspot_raw = rioxarray.open_rasterio(hotspot_path).isel(band=0)
                
                # Crop to box with 1deg safety padding
                pad = 1.0
                da_aod_crop = da_aod_raw.rio.clip_box(
                    minx=max(float(da_aod_raw.x.min()), bbox['min_lon'] - pad),
                    miny=max(float(da_aod_raw.y.min()), bbox['min_lat'] - pad),
                    maxx=min(float(da_aod_raw.x.max()), bbox['max_lon'] + pad),
                    maxy=min(float(da_aod_raw.y.max()), bbox['max_lat'] + pad)
                )
                da_hotspot_crop = da_hotspot_raw.rio.clip_box(
                    minx=max(float(da_hotspot_raw.x.min()), bbox['min_lon'] - pad),
                    miny=max(float(da_hotspot_raw.y.min()), bbox['min_lat'] - pad),
                    maxx=min(float(da_hotspot_raw.x.max()), bbox['max_lon'] + pad),
                    maxy=min(float(da_hotspot_raw.y.max()), bbox['max_lat'] + pad)
                )
                
                u_var = 'u10' if 'u10' in ds_wind_month else 'u'
                v_var = 'v10' if 'v10' in ds_wind_month else 'v'
                
                da_u_raw = ds_wind_month[u_var].squeeze(drop=True)
                da_v_raw = ds_wind_month[v_var].squeeze(drop=True)
                
                rename_dict = {}
                for dim_name, std_name in [('latitude', 'y'), ('lat', 'y'), ('longitude', 'x'), ('lon', 'x')]:
                    if dim_name in da_u_raw.dims:
                        rename_dict[dim_name] = std_name
                if rename_dict:
                    da_u_raw = da_u_raw.rename(rename_dict)
                    da_v_raw = da_v_raw.rename(rename_dict)
                    
                da_u_raw.rio.write_crs("EPSG:4326", inplace=True)
                da_v_raw.rio.write_crs("EPSG:4326", inplace=True)
                
                da_u_crop = da_u_raw.rio.clip_box(
                    minx=max(float(da_u_raw.x.min()), bbox['min_lon'] - pad),
                    miny=max(float(da_u_raw.y.min()), bbox['min_lat'] - pad),
                    maxx=min(float(da_u_raw.x.max()), bbox['max_lon'] + pad),
                    maxy=min(float(da_u_raw.y.max()), bbox['max_lat'] + pad)
                )
                da_v_crop = da_v_raw.rio.clip_box(
                    minx=max(float(da_v_raw.x.min()), bbox['min_lon'] - pad),
                    miny=max(float(da_v_raw.y.min()), bbox['min_lat'] - pad),
                    maxx=min(float(da_v_raw.x.max()), bbox['max_lon'] + pad),
                    maxy=min(float(da_v_raw.y.max()), bbox['max_lat'] + pad)
                )
                
                # Match grids
                da_aod_ref = da_aod_crop.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
                da_hotspot_ref = da_hotspot_crop.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.nearest)
                da_u_ref = da_u_crop.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
                da_v_ref = da_v_crop.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
                
                aod_arr = da_aod_ref.values.astype("float32")
                aod_arr[aod_arr > 10.0] = np.nan
                aod_arr[aod_arr < 0.0] = np.nan
                
                hotspot_arr = da_hotspot_ref.values
                u_arr = da_u_ref.values.astype("float32")
                v_arr = da_v_ref.values.astype("float32")
                
                # Proximity analysis
                dist_deg = np.sqrt((lon2d - self.sink.lon) ** 2 + (lat2d - self.sink.lat) ** 2)
                buf_mask = dist_deg <= self.sink.buffer_deg
                sink_aod_vals = aod_arr[buf_mask & np.isfinite(aod_arr)]
                sink_aod_val = float(np.nanmean(sink_aod_vals)) if len(sink_aod_vals) > 0 else np.nan
                
                # Transport analysis: apply source country mask, then AOD selection mask from config.
                source_country = self.config.get('source_country', 'full_domain')
                if source_country.lower() == 'full_domain':
                    country_mask = np.ones_like(aod_arr, dtype=bool)
                else:
                    country_mask = self._get_country_mask(source_country, lons, lats, ref_ds)

                hotspot_mask = np.isfinite(hotspot_arr) & (hotspot_arr > 0)
                if aod_source_mode == "hotspot":
                    aod_selection_mask = hotspot_mask.copy()
                else:
                    aod_selection_mask = np.isfinite(aod_arr)

                if aod_threshold is not None:
                    aod_selection_mask &= np.isfinite(aod_arr) & (aod_arr > aod_threshold)

                source_mask = country_mask & aod_selection_mask

                dx_hat, dy_hat = WWTEGeospatialEngine.bearing_unit_vectors(lon2d, lat2d, self.sink.lon, self.sink.lat)
                wind_mag = np.sqrt(u_arr**2 + v_arr**2)
                
                valid = source_mask & np.isfinite(aod_arr) & np.isfinite(u_arr) & np.isfinite(v_arr) & (wind_mag > 0)
                u_hat = np.where(wind_mag > 0, u_arr / wind_mag, np.nan)
                v_hat = np.where(wind_mag > 0, v_arr / wind_mag, np.nan)
                
                cosang = u_hat * dx_hat + v_hat * dy_hat
                toward = valid & (cosang > 0)
                
                speed_max = float(np.nanmax(wind_mag[toward])) if np.any(toward) else 1.0
                speed_norm = wind_mag / max(speed_max, 1e-6)
                
                dist_km = WWTEGeospatialEngine.haversine_distance_km(lon2d, lat2d, self.sink.lon, self.sink.lat)
                dist_decay = np.exp(-dist_km / decay_length)
                
                score = np.full(aod_arr.shape, np.nan, dtype="float32")
                score[toward] = (
                    aod_arr[toward]
                    * np.maximum(cosang[toward], 0.0)
                    * speed_norm[toward]
                    * dist_decay[toward]
                )
                
                weight = np.zeros_like(aod_arr, dtype="float32")
                weight[toward] = np.maximum(cosang[toward], 0.0) * speed_norm[toward]
                
                w_sum = float(np.nansum(weight[toward]))
                wwte_index = float(np.nansum(score[toward] * weight[toward]) / w_sum) if w_sum > 0 else np.nan
                
                # Diagnostics
                if np.any(toward):
                    q = np.nanquantile(aod_arr[toward], q_high)
                    high_toward = toward & (aod_arr >= q)
                    high_toward_aod = float(np.nanmean(aod_arr[high_toward])) if np.any(high_toward) else np.nan
                    high_toward_frac = float(np.sum(high_toward) / np.sum(toward))
                else:
                    high_toward_aod = np.nan
                    high_toward_frac = np.nan
                    
                self.records.append({
                    "year_month": year_month,
                    "year": year,
                    "month": month,
                    "aod_source_mode": aod_source_mode,
                    "aod_threshold": aod_threshold,
                    f"{self.sink.name.lower()}_aod": sink_aod_val,
                    "wwte_index": wwte_index,
                    "toward_pixel_count": int(np.sum(toward)),
                    "source_valid_pixel_count": int(np.sum(valid)),
                    "source_aod_mean": float(np.nanmean(aod_arr[valid])) if np.any(valid) else np.nan,
                    "source_aod_toward_mean": float(np.nanmean(aod_arr[toward])) if np.any(toward) else np.nan,
                    "mean_dist_decay": float(np.nanmean(dist_decay[toward])) if np.any(toward) else np.nan,
                    "mean_cosang_toward": float(np.nanmean(cosang[toward])) if np.any(toward) else np.nan,
                    "high_toward_aod_mean": high_toward_aod,
                    "high_toward_fraction": high_toward_frac,
                })
                
                ds_out = xr.Dataset(
                    {
                        "AOD": (("lat", "lon"), aod_arr),
                        "U": (("lat", "lon"), u_arr),
                        "V": (("lat", "lon"), v_arr),
                        "wind_speed": (("lat", "lon"), wind_mag),
                        "wind_speed_norm": (("lat", "lon"), speed_norm.astype("float32")),
                        "cos_angle": (("lat", "lon"), cosang.astype("float32")),
                        "dist_decay": (("lat", "lon"), dist_decay.astype("float32")),
                        "toward_mask": (("lat", "lon"), toward.astype(np.int8)),
                        "source_mask": (("lat", "lon"), source_mask.astype(np.int8)),
                        "country_mask": (("lat", "lon"), country_mask.astype(np.int8)),
                        "aod_selection_mask": (("lat", "lon"), aod_selection_mask.astype(np.int8)),
                        "Hotspot_Mask": (("lat", "lon"), hotspot_mask.astype(np.int32)),
                        "WWTE_Score": (("lat", "lon"), score),
                        "WWTE_Weight": (("lat", "lon"), weight),
                    },
                    coords={
                        "lon": lons,
                        "lat": lats,
                        "time": pd.to_datetime(f"{year}-{month}-01")
                    }
                )
                ds_out.rio.write_crs("EPSG:4326", inplace=True)
                self.monthly_datasets.append(ds_out)
                
                # Cleanup
                da_aod_raw.close()
                da_hotspot_raw.close()
                da_aod_crop.close()
                da_hotspot_crop.close()
                
            except Exception as e:
                print(f"Error executing year-month {year_month}: {e}")
                
        combined_wind.close()

    @profile_stage
    def export_results(self) -> None:
        """Consolidates and writes datasets to disk."""
        dirs = self.config['directories']
        dir_out = os.path.join(self.base_dir, dirs['output'])
        os.makedirs(dir_out, exist_ok=True)
        
        wind_file_suffix = self._resolve_wind_suffix()
        
        # Write unified multi-year NetCDF (monthly climatology: average Jan, Feb, ... Dec across all years)
        if self.monthly_datasets:
            print("\nComputing monthly climatology (mean across years per month)...")
            
            combined_ds = xr.concat(self.monthly_datasets, dim='time')
            combined_ds = combined_ds.sortby('time')
            
            # Group by month and take the mean across years
            climatology_ds = combined_ds.groupby('time.month').mean(dim='time')
            
            # Ensure coordinates are named lat/lon for compatibility
            rename_map = {}
            if 'x' in climatology_ds.dims and 'lon' not in climatology_ds.dims:
                rename_map['x'] = 'lon'
            if 'y' in climatology_ds.dims and 'lat' not in climatology_ds.dims:
                rename_map['y'] = 'lat'
            if rename_map:
                climatology_ds = climatology_ds.rename(rename_map)
                combined_ds = combined_ds.rename(rename_map)

            out_nc = os.path.join(dir_out, f"wwte_{wind_file_suffix}_combined.nc")
            climatology_ds.to_netcdf(out_nc)
            print(f"Monthly climatology NetCDF exported: {out_nc}")
            
            # Also export to climatology folder for direct use in plot_climatology.py
            climatology_dir = os.path.join(dir_out, 'climatology')
            os.makedirs(climatology_dir, exist_ok=True)
            climatology_out_nc = os.path.join(climatology_dir, f"wwte_climatology_{wind_file_suffix}_combined.nc")
            climatology_ds.to_netcdf(climatology_out_nc)
            print(f"Climatology NetCDF exported to: {climatology_out_nc}")
            
            combined_ds.close()
            climatology_ds.close()
            
            # Close all cached monthly datasets to prevent shutdown errors
            for ds in self.monthly_datasets:
                ds.close()
            self.monthly_datasets.clear()
            
        # Write tabular CSV summary
        if self.records:
            df = pd.DataFrame(self.records)
            idx_min = df["wwte_index"].min()
            idx_max = df["wwte_index"].max()
            if idx_max > idx_min:
                df["wwte_index_norm"] = (df["wwte_index"] - idx_min) / (idx_max - idx_min)
            else:
                df["wwte_index_norm"] = 0.5
                
            out_csv = os.path.join(dir_out, f"wwte_summary_{wind_file_suffix}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Tabular summary exported: {out_csv}")
        else:
            print("\n[PIPELINE WARNING] No records found. Verify that input AOD files exist and extensions match.")


# --- MAIN TRIGGER ---
def main() -> None:
    """
    Pipeline execution entry point.
    """
    print("="*60)
    print("  WWTE ADVANCED GEOSPATIAL TRANSPORT ANALYSIS PIPELINE  ")
    print("="*60)
    
    try:
        # Initialize pipeline using central config
        pipeline = WWTEPipeline("config/config.json")
        pipeline.initialize()
        pipeline.run_spatial_analysis()
        pipeline.export_results()
        print("\nPipeline execution sequence completed.")
        
        # Force garbage collection before shutdown to prevent rioxarray/rasterio
        # file handle errors during Python interpreter teardown
        del pipeline
        import gc
        gc.collect()
        
    except Exception as e:
        print("\n" + "#"*40)
        print(" [FATAL ERROR] Pipeline crashed during execution!")
        print("#"*40 + "\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # Suppress rasterio/rioxarray shutdown errors (harmless file handle cleanup)
    sys.excepthook = lambda *_: None
