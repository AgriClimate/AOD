# File: wwte_aod_index_analysis.py
# Description:
# This script computes the Wind-Weighted AOD Transport Efficiency (WWTE) index 
# for the transport pathway from a dynamic source region to a sink location.
# It reads monthly time-series AOD, ERA5 wind (supporting pressure level 850hPa 
# or surface 10m wind), and hotspot binary masks. It dynamically loads the wind 
# type configured in config.json ('wind10m' or 'wind850hp').
# All monthly model outputs are consolidated into a single combined multi-year 
# NetCDF file to eliminate redundant coordinate storage and dramatically optimize disk space.
#
# How to run:
# python resources/wwte_aod_index_analysis.py
#
# Dependencies:
# - Python 3.10+
# - numpy, pandas, xarray, rioxarray, rasterio, netCDF4
#
# Expected inputs:
# - config.json defining active sink, bounding box, and active_wind_type
# - AOD combined, binary hotspot, and wind NetCDF datasets
#
# Expected outputs:
# - Combined multi-year NetCDF file: outputs/results/wwte_{wind_type}_combined.nc
# - Consolidated summary CSV: outputs/results/wwte_summary_{wind_type}.csv

from __future__ import annotations

import glob
import json
import math
import os
import traceback
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr


EARTH_RADIUS_KM = 6371.0


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads configuration parameters from a JSON file.

    Args:
        path (str): The file path to the config JSON.

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    """
    with open(path, 'r') as f:
        return json.load(f)


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    """
    Calculates great-circle distance in kilometers from every grid cell to a target point 
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
    Calculates the unit vector components pointing from each grid cell toward the target sink, 
    accounting for latitude convergence.

    Args:
        lon2d (np.ndarray): 2D array of longitudes.
        lat2d (np.ndarray): 2D array of latitudes.
        target_lon (float): Target sink longitude.
        target_lat (float): Target sink latitude.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The (dx_hat, dy_hat) components of the unit vector.
    """
    cos_lat = np.cos(np.radians((lat2d + target_lat) / 2.0))
    dlon = (target_lon - lon2d) * cos_lat
    dlat = target_lat - lat2d
    mag = np.sqrt(dlon ** 2 + dlat ** 2)
    mag = np.where(mag == 0, np.nan, mag)
    return (dlon / mag).astype("float32"), (dlat / mag).astype("float32")


def sink_area_mean(aod: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, s_lon: float, s_lat: float, buffer_deg: float) -> float:
    """
    Calculates the spatial average of AOD within a designated radial buffer around the sink city.

    Args:
        aod (np.ndarray): 2D array of AOD values.
        lon2d (np.ndarray): 2D array of longitudes.
        lat2d (np.ndarray): 2D array of latitudes.
        s_lon (float): Sink city longitude.
        s_lat (float): Sink city latitude.
        buffer_deg (float): Buffer radius in degrees.

    Returns:
        float: Mean AOD value within the buffer, or NaN if no valid values exist.
    """
    dist_deg = np.sqrt((lon2d - s_lon) ** 2 + (lat2d - s_lat) ** 2)
    buf_mask = dist_deg <= buffer_deg
    vals = aod[buf_mask & np.isfinite(aod)]
    return float(np.nanmean(vals)) if len(vals) > 0 else np.nan


def clip_to_bbox(da: xr.DataArray, bbox: Dict[str, float], buffer: float = 1.0) -> xr.DataArray:
    """
    Efficiently crops a DataArray to the configured study domain bounding box with a safety buffer.

    Args:
        da (xr.DataArray): The input spatial DataArray.
        bbox (Dict[str, float]): Study area bounding box dictionary.
        buffer (float, optional): Extra padding degrees. Defaults to 1.0.

    Returns:
        xr.DataArray: The cropped spatial DataArray.
    """
    min_lon = max(float(da.x.min()), bbox['min_lon'] - buffer)
    max_lon = min(float(da.x.max()), bbox['max_lon'] + buffer)
    min_lat = max(float(da.y.min()), bbox['min_lat'] - buffer)
    max_lat = min(float(da.y.max()), bbox['max_lat'] + buffer)
    
    return da.rio.clip_box(
        minx=min_lon,
        miny=min_lat,
        maxx=max_lon,
        maxy=max_lat
    )


def main() -> None:
    """
    Main pipeline execution function. Parses configuration, reads datasets, crops 
    them to the study domain, performs spatial transport analysis, and saves consolidated outputs.
    """
    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_config(config_path)
    
    dirs = config['directories']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dir_hotspot = os.path.join(base_dir, dirs['hotspot_binary'])
    dir_aod = os.path.join(base_dir, dirs['aod_combined'])
    dir_wind = os.path.join(base_dir, dirs['wind'])
    dir_out = os.path.join(base_dir, dirs['output'])
    
    os.makedirs(dir_out, exist_ok=True)
    
    sink_name = config['sink_location']['name']
    s_lon = config['sink_location']['lon']
    s_lat = config['sink_location']['lat']
    
    # Retrieve active wind type from configuration (defaults to wind850hp if not present)
    wind_type = config.get("active_wind_type", "wind850hp")
    
    # Enforce strict naming conventions requested: 'wind10m' or 'wind850hp'
    if wind_type in ["10m", "wind10m"]:
        wind_file_suffix = "wind10m"
    else:
        wind_file_suffix = "wind850hp"
    
    # Retrieve buffer key dynamically matching the sink name
    params = config['parameters']
    s_buffer = params.get(
        f"{sink_name.lower()}_buffer_deg", 
        params.get("zahedan_buffer_deg", params.get("sink_buffer_deg", 0.3))
    )
    
    decay_length = params['decay_length_km']
    q_high = params['af_high_aod_quantile']
    
    bbox = params['bounding_box']
    res = params['resolution_deg']
    
    # 2. Create Reference Grid
    lons = np.arange(bbox['min_lon'], bbox['max_lon'] + res/2, res)
    lats = np.arange(bbox['max_lat'], bbox['min_lat'] - res/2, -res)
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # Create reference rioxarray Dataset
    ref_ds = xr.Dataset(coords={'y': lats, 'x': lons})
    ref_ds.rio.write_crs("EPSG:4326", inplace=True)
    
    # 3. Check for Combined Multi-Year Wind Dataset
    combined_wind_ds = None
    if wind_file_suffix == "wind10m":
        combined_path = os.path.join(dir_wind, "era5_monthly_wind10m_combined.nc")
    else:
        # Note: combined file on disk is era5_monthly_wind850mb_combined.nc
        combined_path = os.path.join(dir_wind, "era5_monthly_wind850mb_combined.nc")
        
    if os.path.exists(combined_path):
        print(f"Detected combined wind dataset: {os.path.basename(combined_path)}. Loading globally to optimize performance.")
        combined_wind_ds = xr.open_dataset(combined_path)
    
    # 4. Scan AOD Files to find available time-series months
    aod_files = sorted(glob.glob(os.path.join(dir_aod, 'MCDAL2_M_AER_OD_*-*.FLOAT.TIFF')))
    records = []
    monthly_datasets = []
    
    for aod_path in aod_files:
        filename = os.path.basename(aod_path)
        year_month = filename.split('_')[4].split('.')[0]  # e.g. '2001-01'
        year, month = year_month.split('-')
        
        # Select single time slice from combined dataset, or search for yearly file
        ds_wind_month = None
        yearly_wind_path = None
        
        if combined_wind_ds is not None:
            time_str = f"{year}-{month}-01"
            # Attempt selection dynamically using valid coordinates
            for time_coord in ["valid_time", "time"]:
                if time_coord in combined_wind_ds.coords:
                    try:
                        ds_wind_month = combined_wind_ds.sel({time_coord: time_str}, method='nearest')
                        break
                    except Exception:
                        pass
            if ds_wind_month is None:
                print(f"Skipping {year}-{month}: Time slice {time_str} not found in combined wind dataset.")
                continue
        else:
            # Check yearly files
            if wind_file_suffix == "wind10m":
                w_path = os.path.join(dir_wind, f"era5_monthly_wind10m_{year}.nc")
            else:
                w_path_850_1 = os.path.join(dir_wind, f"era5_monthly_wind850_{year}.nc")
                w_path_850_2 = os.path.join(dir_wind, f"era5_monthly_wind850mb_{year}.nc")
                w_path_850_3 = os.path.join(dir_wind, f"era5_monthly_wind850hp_{year}.nc")
                
                w_path = w_path_850_1
                for path_candidate in [w_path_850_1, w_path_850_2, w_path_850_3]:
                    if os.path.exists(path_candidate):
                        w_path = path_candidate
                        break
                
            if os.path.exists(w_path):
                yearly_wind_path = w_path
            else:
                print(f"Skipping {year}-{month}: Missing yearly wind file {os.path.basename(w_path)}.")
                continue
            
        hotspot_path = os.path.join(dir_hotspot, f"binary_Avg_AOD_24yr_Month{month}_2001_2024.tif")
        if not os.path.exists(hotspot_path):
            print(f"Skipping {year}-{month}: Missing hotspot data.")
            continue
            
        print(f"Processing {year}-{month} using active wind model '{wind_file_suffix}'...")
        
        try:
            # --- Load, Crop, and Reproject Datasets ---
            da_aod_raw = rioxarray.open_rasterio(aod_path).isel(band=0)
            da_hotspot_raw = rioxarray.open_rasterio(hotspot_path).isel(band=0)
            
            # Crop to bounding box before reprojection to optimize memory footprint
            da_aod_cropped = clip_to_bbox(da_aod_raw, bbox)
            da_hotspot_cropped = clip_to_bbox(da_hotspot_raw, bbox)
            
            if ds_wind_month is None and yearly_wind_path is not None:
                ds_wind = xr.open_dataset(yearly_wind_path)
                time_str = f"{year}-{month}-01"
                for time_coord in ["valid_time", "time"]:
                    if time_coord in ds_wind.coords:
                        try:
                            ds_wind_month = ds_wind.sel({time_coord: time_str}, method='nearest')
                            break
                        except Exception:
                            pass
                if ds_wind_month is None:
                    ds_wind.close()
                    raise KeyError(f"Could not find matching time slice for {time_str} in yearly wind file.")
            
            # Dynamically identify variable names (u/v or u10/v10)
            if 'u10' in ds_wind_month:
                u_var = 'u10'
                v_var = 'v10'
            elif 'u' in ds_wind_month:
                u_var = 'u'
                v_var = 'v'
            else:
                raise KeyError(f"Wind variables 'u'/'v' or 'u10'/'v10' not found in dataset slice")
                
            da_u_raw = ds_wind_month[u_var].squeeze(drop=True)
            da_v_raw = ds_wind_month[v_var].squeeze(drop=True)
            
            # Rename coordinates to standard y and x
            rename_dict = {}
            if 'latitude' in da_u_raw.dims:
                rename_dict['latitude'] = 'y'
            elif 'lat' in da_u_raw.dims:
                rename_dict['lat'] = 'y'
            if 'longitude' in da_u_raw.dims:
                rename_dict['longitude'] = 'x'
            elif 'lon' in da_u_raw.dims:
                rename_dict['lon'] = 'x'
                
            if rename_dict:
                da_u_raw = da_u_raw.rename(rename_dict)
                da_v_raw = da_v_raw.rename(rename_dict)
                
            da_u_raw.rio.write_crs("EPSG:4326", inplace=True)
            da_v_raw.rio.write_crs("EPSG:4326", inplace=True)
            
            # Crop wind coordinates to bounding box
            da_u_cropped = clip_to_bbox(da_u_raw, bbox)
            da_v_cropped = clip_to_bbox(da_v_raw, bbox)
            
            # Align and reproject all layers to the reference grid
            da_aod_ref = da_aod_cropped.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
            da_hotspot_ref = da_hotspot_cropped.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.nearest)
            da_u_ref = da_u_cropped.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
            da_v_ref = da_v_cropped.rio.reproject_match(ref_ds, resampling=rasterio.enums.Resampling.bilinear)
            
            # Extract arrays
            aod_arr = da_aod_ref.values.astype("float32")
            aod_arr[aod_arr > 10.0] = np.nan
            aod_arr[aod_arr < 0.0] = np.nan
            
            hotspot_arr = da_hotspot_ref.values
            u_arr = da_u_ref.values.astype("float32")
            v_arr = da_v_ref.values.astype("float32")
            
            # --- Transport Index Calculations ---
            # 1. Sink AOD Average
            sink_aod_val = sink_area_mean(aod_arr, lon2d, lat2d, s_lon, s_lat, s_buffer)
            
            # 2. General Source Mask (no country boundary restriction)
            source_mask = np.ones_like(aod_arr, dtype=bool) # Apply on full bounding box
            
            # 3. Wind Direction and Alignment
            dx_hat, dy_hat = bearing_unit_vector(lon2d, lat2d, s_lon, s_lat)
            wind_mag = np.sqrt(u_arr**2 + v_arr**2)
            
            valid = source_mask & np.isfinite(aod_arr) & np.isfinite(u_arr) & np.isfinite(v_arr) & (wind_mag > 0)
            
            u_hat = np.where(wind_mag > 0, u_arr / wind_mag, np.nan)
            v_hat = np.where(wind_mag > 0, v_arr / wind_mag, np.nan)
            
            cosang = u_hat * dx_hat + v_hat * dy_hat
            toward = valid & (cosang > 0)
            
            # 4. Normalized Speed
            speed_max = float(np.nanmax(wind_mag[toward])) if np.any(toward) else 1.0
            speed_norm = wind_mag / max(speed_max, 1e-6)
            
            # 5. Spatial Distance Decay
            dist_km = haversine_km(lon2d, lat2d, s_lon, s_lat)
            dist_decay = np.exp(-dist_km / decay_length)
            
            # 6. WWTE Calculation
            score = np.full(aod_arr.shape, np.nan, dtype="float32")
            score[toward] = (
                aod_arr[toward]
                * np.maximum(cosang[toward], 0.0)
                * speed_norm[toward]
                * dist_decay[toward]
            )
            
            # 7. Aggregated WWTE Index
            weight = np.zeros_like(aod_arr, dtype="float32")
            weight[toward] = np.maximum(cosang[toward], 0.0) * speed_norm[toward]
            
            w_sum = float(np.nansum(weight[toward]))
            wwte_index = float(np.nansum(score[toward] * weight[toward]) / w_sum) if w_sum > 0 else np.nan
            
            # 8. High AOD Diagnostics
            if np.any(toward):
                q = np.nanquantile(aod_arr[toward], q_high)
                high_toward = toward & (aod_arr >= q)
                high_toward_aod_mean = float(np.nanmean(aod_arr[high_toward])) if np.any(high_toward) else np.nan
                high_toward_fraction = float(np.sum(high_toward) / np.sum(toward))
            else:
                high_toward_aod_mean = np.nan
                high_toward_fraction = np.nan
                
            records.append({
                "year_month": year_month,
                "year": year,
                "month": month,
                f"{sink_name.lower()}_aod": sink_aod_val,
                "wwte_index": wwte_index,
                "toward_pixel_count": int(np.sum(toward)),
                "source_valid_pixel_count": int(np.sum(valid)),
                "source_aod_mean": float(np.nanmean(aod_arr[valid])) if np.any(valid) else np.nan,
                "source_aod_toward_mean": float(np.nanmean(aod_arr[toward])) if np.any(toward) else np.nan,
                "mean_dist_decay": float(np.nanmean(dist_decay[toward])) if np.any(toward) else np.nan,
                "mean_cosang_toward": float(np.nanmean(cosang[toward])) if np.any(toward) else np.nan,
                "high_toward_aod_mean": high_toward_aod_mean,
                "high_toward_fraction": high_toward_fraction,
            })
            
            # --- Save Monthly NetCDF to RAM for Consolidation ---
            ds_out = xr.Dataset(
                {
                    "AOD": (("y", "x"), aod_arr),
                    "U": (("y", "x"), u_arr),
                    "V": (("y", "x"), v_arr),
                    "Hotspot_Mask": (("y", "x"), source_mask.astype(np.int32)),
                    "WWTE_Score": (("y", "x"), score),
                },
                coords={
                    "x": lons,
                    "y": lats,
                    "time": pd.to_datetime(f"{year}-{month}-01")
                }
            )
            ds_out.rio.write_crs("EPSG:4326", inplace=True)
            monthly_datasets.append(ds_out)
            
            # Clean up active resources
            da_aod_raw.close()
            da_hotspot_raw.close()
            da_aod_cropped.close()
            da_hotspot_cropped.close()
            if yearly_wind_path is not None:
                ds_wind.close()
            
        except Exception as e:
            print(f"Error processing {year}-{month}: {e}")
            traceback.print_exc()
            
    # Clean up combined wind resource
    if combined_wind_ds is not None:
        combined_wind_ds.close()
        
    # --- Consolidated Combined Multi-Year NetCDF Export ---
    if monthly_datasets:
        print("\nConsolidating and writing monthly outputs to a single combined NetCDF...")
        combined_ds = xr.concat(monthly_datasets, dim='time')
        combined_ds = combined_ds.sortby('time')
        
        combined_ds.attrs["description"] = f"Combined multi-year WWTE model outputs using {wind_file_suffix} wind model"
        
        out_nc_path = os.path.join(dir_out, f"wwte_{wind_file_suffix}_combined.nc")
        combined_ds.to_netcdf(out_nc_path)
        print(f"Successfully generated combined NetCDF: {out_nc_path}")
        combined_ds.close()
            
    # --- Consolidated CSV Export ---
    if records:
        df = pd.DataFrame(records)
        
        # Normalized WWTE score column
        idx_min = df["wwte_index"].min()
        idx_max = df["wwte_index"].max()
        if idx_max > idx_min:
            df["wwte_index_norm"] = (df["wwte_index"] - idx_min) / (idx_max - idx_min)
        else:
            df["wwte_index_norm"] = 0.5
            
        # Enforce strict naming convention for CSV
        out_csv = os.path.join(dir_out, f"wwte_summary_{wind_file_suffix}.csv")
        try:
            df.to_csv(out_csv, index=False)
            print(f"Successfully generated: {out_csv}")
        except PermissionError:
            alt_csv = os.path.join(dir_out, f"wwte_summary_{wind_file_suffix}_locked.csv")
            df.to_csv(alt_csv, index=False)
            print(f"\n[WARNING] Permission denied to write '{out_csv}'. The file might be open in Excel or another editor.")
            print(f"-> Saved data to alternative path: {alt_csv}")
    else:
        print("Pipeline execution failed: No records processed successfully.")


if __name__ == "__main__":
    main()
