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
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from shapely.geometry import mapping

# --- LOGGING SETUP ---
logging.basicConfig(
    filename="pipeline.log",
    filemode="w",  # Overwrite log file on each run
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)


# --- CONSTANTS ---
EARTH_RADIUS_KM: float = 6371.0



def profile_stage(func):
    """
    A decorator designed to log and profile execution stages.
    """
    def wrapper(*args, **kwargs):
        print(f"\n[PIPELINE STAGE] Running: {func.__name__}...")
        logging.info(f"[PIPELINE STAGE] Running: {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            print(f"[PIPELINE STAGE] Finished: {func.__name__} successfully.")
            logging.info(f"[PIPELINE STAGE] Finished: {func.__name__} successfully.")
            return result
        except Exception as e:
            print(f"[PIPELINE ERROR] Failed in stage {func.__name__}: {e}")
            logging.error(f"[PIPELINE ERROR] Failed in stage {func.__name__}: {e}")
            # Redact absolute paths from traceback
            tb = traceback.format_exc()
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tb_redacted = tb.replace(base_dir + os.sep, "")
            logging.error(tb_redacted)
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
        Only uses explicit coordinates from config (offline safe).
        """
        sink_cfg = config['sink_location']
        params = config['parameters']
        name = sink_cfg['name']

        if 'lon' not in sink_cfg or 'lat' not in sink_cfg:
            raise ValueError(
                f"sink_location must include explicit 'lon' and 'lat' fields in config.json for offline use."
            )
        lon = float(sink_cfg['lon'])
        lat = float(sink_cfg['lat'])

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
        # Store paths to per-month output files (written to disk immediately)
        self.monthly_files: List[str] = []
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
        """Executes the core calculations on all monthly inputs. Detailed logging added."""
        import traceback
        dirs = self.config['directories']
        params = self.config['parameters']
        logger = logging.getLogger("wwte.run_spatial_analysis")

        dir_hotspot = os.path.join(self.base_dir, dirs['hotspot_binary'])
        dir_aod = os.path.join(self.base_dir, dirs['aod_combined'])
        dir_wind = os.path.join(self.base_dir, dirs['wind'])
        # Log only local (relative) paths for input directories
        rel_hotspot = os.path.relpath(dir_hotspot, self.base_dir)
        rel_aod = os.path.relpath(dir_aod, self.base_dir)
        rel_wind = os.path.relpath(dir_wind, self.base_dir)
        logger.info(f"Input directories: hotspot={rel_hotspot}, aod={rel_aod}, wind={rel_wind}")

        wind_file_suffix = self._resolve_wind_suffix()
        logger.info(f"Wind file suffix: {wind_file_suffix}")

        bbox = params['bounding_box']
        res = params['resolution_deg']
        decay_length = params['decay_length_km']
        q_high = params['af_high_aod_quantile']
        aod_source_mode = self._resolve_aod_source_mode()
        aod_threshold = self._resolve_aod_threshold()
        logger.info(f"Params: bbox={bbox}, res={res}, decay_length={decay_length}, q_high={q_high}, aod_source_mode={aod_source_mode}, aod_threshold={aod_threshold}")

        # Reference grids
        lons = np.arange(bbox['min_lon'], bbox['max_lon'] + res/2, res)
        lats = np.arange(bbox['max_lat'], bbox['min_lat'] - res/2, -res)
        lon2d, lat2d = np.meshgrid(lons, lats)
        ref_ds = xr.Dataset(coords={'y': lats, 'x': lons})
        ref_ds.rio.write_crs("EPSG:4326", inplace=True)
        logger.info(f"Reference grid shapes: lons={lons.shape}, lats={lats.shape}")

        # Load wind global database
        wind_path = os.path.join(
            dir_wind, 
            "era5_monthly_wind10m_combined.nc" if wind_file_suffix == "wind10m" else "era5_monthly_wind850mb_combined.nc"
        )
        rel_wind_path = os.path.relpath(wind_path, self.base_dir)
        logger.info(f"Loading wind dataset: {rel_wind_path}")
        try:
            combined_wind = xr.open_dataset(wind_path)
        except Exception as e:
            logger.error(f"Failed to load wind dataset: {wind_path}: {e}")
            logger.error(traceback.format_exc())
            raise

        aod_files = sorted(glob.glob(os.path.join(dir_aod, 'MCDAL2_M_AER_OD_*-*.FLOAT.TIFF')))
        logger.info(f"Found {len(aod_files)} AOD files.")

        for aod_path in aod_files:
            filename = os.path.basename(aod_path)
            year_month = filename.split('_')[4].split('.')[0]
            year, month = year_month.split('-')
            rel_aod_path = os.path.relpath(aod_path, self.base_dir)
            logger.info(f"Processing {year_month}: {rel_aod_path}")

            time_str = f"{year}-{month}-01"
            ds_wind_month = None
            for coord in ["valid_time", "time"]:
                if coord in combined_wind.coords:
                    try:
                        ds_wind_month = combined_wind.sel({coord: time_str}, method='nearest')
                        logger.info(f"Selected wind month for {coord}={time_str}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to select wind month for {coord}={time_str}: {e}")
            if ds_wind_month is None:
                logger.warning(f"No wind data for {year_month}, skipping.")
                continue


            hotspot_path = self._hotspot_path_for_month(dir_hotspot, month)
            rel_hotspot_path = os.path.relpath(hotspot_path, self.base_dir)
            if not os.path.exists(hotspot_path):
                logger.warning(f"Missing hotspot mask for {year_month}: {rel_hotspot_path}, skipping.")
                continue

            try:
                logger.info("--- Start: Load and crop datasets ---")
                logger.info(f"Loading AOD: {rel_aod_path}")
                da_aod_raw = rioxarray.open_rasterio(aod_path).isel(band=0)
                logger.info(f"Loading Hotspot: {rel_hotspot_path}")
                da_hotspot_raw = rioxarray.open_rasterio(hotspot_path).isel(band=0)

                pad = 1.0
                logger.info(f"Cropping AOD and Hotspot to bounding box with pad={pad}")
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
                logger.info(f"Wind variable names: u={u_var}, v={v_var}")

                da_u_raw = ds_wind_month[u_var].squeeze(drop=True)
                da_v_raw = ds_wind_month[v_var].squeeze(drop=True)

                rename_dict = {}
                for dim_name, std_name in [('latitude', 'y'), ('lat', 'y'), ('longitude', 'x'), ('lon', 'x')]:
                    if dim_name in da_u_raw.dims:
                        rename_dict[dim_name] = std_name
                if rename_dict:
                    da_u_raw = da_u_raw.rename(rename_dict)
                    da_v_raw = da_v_raw.rename(rename_dict)
                    logger.info(f"Renamed wind dimensions: {rename_dict}")

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

                logger.info(f"Reprojecting/cropping all arrays to reference grid")
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

                logger.info(f"Shapes: aod={aod_arr.shape}, hotspot={hotspot_arr.shape}, u={u_arr.shape}, v={v_arr.shape}")

                logger.info("--- Start: Proximity analysis ---")
                dist_deg = np.sqrt((lon2d - self.sink.lon) ** 2 + (lat2d - self.sink.lat) ** 2)
                buf_mask = dist_deg <= self.sink.buffer_deg
                sink_aod_vals = aod_arr[buf_mask & np.isfinite(aod_arr)]
                sink_aod_val = float(np.nanmean(sink_aod_vals)) if len(sink_aod_vals) > 0 else np.nan

                
                logger.info("--- Start: Transport analysis ---")
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

                
                logger.info("--- Start: Diagnostics ---")
                if np.any(toward):
                    q = np.nanquantile(aod_arr[toward], q_high)
                    high_toward = toward & (aod_arr >= q)
                    high_toward_aod = float(np.nanmean(aod_arr[high_toward])) if np.any(high_toward) else np.nan
                    high_toward_frac = float(np.sum(high_toward) / np.sum(toward))
                else:
                    high_toward_aod = np.nan
                    high_toward_frac = np.nan

                
                logger.info("--- Start: Output and cleanup ---")
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
                # Write per-month dataset to disk immediately to reduce memory usage
                out_dir_monthly = os.path.join(self.base_dir, dirs['output'], 'monthly')
                os.makedirs(out_dir_monthly, exist_ok=True)
                out_monthly_path = os.path.join(out_dir_monthly, f"wwte_{wind_file_suffix}_{year_month}.nc")
                ds_out.to_netcdf(out_monthly_path)
                self.monthly_files.append(out_monthly_path)

                # Close dataset and free large arrays to reduce memory footprint
                ds_out.close()
                da_aod_raw.close()
                da_hotspot_raw.close()
                da_aod_crop.close()
                da_hotspot_crop.close()
                # Explicitly delete large numpy arrays and collect garbage
                try:
                    del aod_arr, hotspot_arr, u_arr, v_arr, wind_mag, score, weight, dist_decay, cosang
                except Exception:
                    pass
                import gc
                gc.collect()
                logger.info(f"Finished processing {year_month} successfully. Saved: {os.path.relpath(out_monthly_path, self.base_dir)}")

            except Exception as e:
                logger.error(f"Error executing year-month {year_month}: {e}")
                logger.error(traceback.format_exc())
                print(f"Error executing year-month {year_month}: {e}")

        combined_wind.close()
        logger.info("Closed wind dataset.")

    @profile_stage
    def export_results(self) -> None:
        """Consolidates and writes datasets to disk."""
        dirs = self.config['directories']
        dir_out = os.path.join(self.base_dir, dirs['output'])
        os.makedirs(dir_out, exist_ok=True)
        
        wind_file_suffix = self._resolve_wind_suffix()
        
        # Write unified multi-year NetCDF (monthly climatology: average Jan, Feb, ... Dec across all years)
        if self.monthly_files:
            print("\nComputing monthly climatology (mean across years per month) from on-disk monthly files...")

            # Accumulate per-month sums and counts to avoid loading all months at once
            month_sums: Dict[int, xr.Dataset] = {}
            month_counts: Dict[int, int] = {}

            for mf in sorted(self.monthly_files):
                try:
                    ds = xr.open_dataset(mf)
                except Exception as e:
                    logging.warning(f"Failed to open monthly file {mf}: {e}")
                    continue

                # Normalize dimension names if needed
                rename_map = {}
                if 'x' in ds.dims and 'lon' not in ds.dims:
                    rename_map['x'] = 'lon'
                if 'y' in ds.dims and 'lat' not in ds.dims:
                    rename_map['y'] = 'lat'
                if rename_map:
                    ds = ds.rename(rename_map)

                # Extract month integer from time coordinate
                try:
                    m = int(pd.to_datetime(ds['time'].values).month)
                except Exception:
                    # Fallback: try parsing from filename
                    basename = os.path.basename(mf)
                    parts = basename.split('_')
                    mm = parts[-1].split('.')[0] if parts else '01'
                    m = int(mm.split('-')[-1].split('.')[0]) if '-' in mm else int(mm[:2])

                # Convert all data variables to float64 for safe accumulation
                ds_float = ds.copy(deep=True)
                for vn in list(ds.data_vars):
                    try:
                        ds_float[vn] = ds[vn].astype('float64')
                    except Exception:
                        ds_float[vn] = ds[vn].astype('float64', copy=False)

                if m not in month_sums:
                    month_sums[m] = ds_float
                    month_counts[m] = 1
                else:
                    month_sums[m] = month_sums[m] + ds_float
                    month_counts[m] += 1

                ds.close()
                try:
                    del ds, ds_float
                except Exception:
                    pass
                import gc
                gc.collect()

            # Build climatology dataset by averaging accumulated sums per month
            climatology_parts: List[xr.Dataset] = []
            for month in sorted(month_sums.keys()):
                s = month_sums[month]
                cnt = max(1, month_counts.get(month, 1))
                mean_ds = s / cnt
                mean_ds = mean_ds.expand_dims({'month': [month]})
                climatology_parts.append(mean_ds)

            if climatology_parts:
                climatology_ds = xr.concat(climatology_parts, dim='month')

                # Export climatology to the dedicated climatology folder for downstream use
                climatology_dir = os.path.join(dir_out, 'climatology')
                os.makedirs(climatology_dir, exist_ok=True)
                climatology_out_nc = os.path.join(climatology_dir, f"wwte_climatology_{wind_file_suffix}_combined.nc")
                climatology_ds.to_netcdf(climatology_out_nc)
                rel_climatology_out_nc = os.path.relpath(climatology_out_nc, self.base_dir)
                print(f"Climatology NetCDF exported to: {rel_climatology_out_nc}")
                logging.info(f"Climatology NetCDF exported to: {rel_climatology_out_nc}")

                climatology_ds.close()

            # Cleanup temporary monthly file list to release references
            month_sums.clear()
            month_counts.clear()
            self.monthly_files.clear()
            
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
            rel_out_csv = os.path.relpath(out_csv, self.base_dir)
            print(f"Tabular summary exported: {rel_out_csv}")
            logging.info(f"Tabular summary exported: {rel_out_csv}")
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
        logging.info("Pipeline execution sequence completed.")
        
        # Force garbage collection before shutdown to prevent rioxarray/rasterio
        # file handle errors during Python interpreter teardown
        del pipeline
        import gc
        gc.collect()
        
    except Exception as e:
        print("\n" + "#"*40)
        print(" [FATAL ERROR] Pipeline crashed during execution!")
        print("#"*40 + "\n")
        logging.error("[FATAL ERROR] Pipeline crashed during execution!")
        logging.error(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # Suppress rasterio/rioxarray shutdown errors (harmless file handle cleanup)
    sys.excepthook = lambda *_: None

# NOTE: All major steps, errors, and exceptions are logged to pipeline.log for traceability.
