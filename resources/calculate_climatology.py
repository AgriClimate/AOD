# File: calculate_climatology.py
# Description:
# This script calculates the long-term monthly averages (climatology) 
# from the combined WWTE NetCDF file. It aggregates data across all available
# years for the active wind type configured in config.json ('wind10m' or 'wind850hp') 
# and saves the resulting averages to a single consolidated climatology NetCDF file.
#
# How to run:
# python resources/calculate_climatology.py
#
# Dependencies:
# - Python 3.10+
# - xarray, netCDF4, dask
#
# Expected inputs:
# - Combined NetCDF file: data/results/wwte_{wind_type}_combined.nc
# - config.json defining active_wind_type
#
# Expected outputs:
# - A single consolidated climatology NetCDF file: 
#   data/results/climatology/wwte_climatology_{wind_type}_combined.nc

import json
import os
import glob
import xarray as xr
import numpy as np
import warnings


warnings.filterwarnings("ignore", message="Mean of empty slice")
try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:
    rasterio = None


def main() -> None:
    """
    Main execution function. Loads the combined multi-year NetCDF, calculates 
    long-term climatological averages for each month (1-12) in a single operation, 
    and saves the consolidated climatology dataset to disk.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dirs = config['directories']
    wind_type = config.get("active_wind_type", "wind850mb")
    
    # Enforce strict naming conventions requested: 'wind10m', 'wind850mb', or 'wind850hp'
    if wind_type in ["10m", "wind10m"]:
        wind_file_suffix = "wind10m"
    elif "850mb" in wind_type or "850" in wind_type:
        wind_file_suffix = "wind850mb"
    else:
        wind_file_suffix = "wind850hp"
    
    in_file = os.path.join(base_dir, dirs['output'], f"wwte_{wind_file_suffix}_combined.nc")
    out_dir = os.path.join(base_dir, dirs['output'], 'climatology')
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(in_file):
        print(f"Error: Combined model output file not found: {in_file}. Run analysis script first.")
        return
        
    print(f"Loading combined dataset: {os.path.basename(in_file)}")
    
    try:
        # Prefer reading per-month files with open_mfdataset (more efficient for many files)
        monthly_dir = os.path.join(base_dir, dirs['output'], 'monthly')
        pattern = os.path.join(monthly_dir, f"wwte_{wind_file_suffix}_*.nc")
        monthly_files = sorted(glob.glob(pattern))
        if monthly_files:
            print(f"Opening {len(monthly_files)} monthly files with xarray.open_mfdataset...")
            # Use nested combine along the time dimension when files lack reliable
            # coordinate ordering information.
            ds = xr.open_mfdataset(
                monthly_files,
                combine='nested',
                concat_dim='time',
                coords='minimal',
                compat='override',
                parallel=False,
            )
        else:
            print("Monthly files not found; falling back to combined file.")
            ds = xr.open_dataset(in_file)

        # Calculate the long-term monthly climatology across the time dimension
        print(f"Calculating climatology for all 12 months using xarray groupby...")
        climatology = ds.groupby('time.month').mean(dim='time', skipna=True)

        # Determine output format: 'nc' (NetCDF) or 'tif'/'tiff' (GeoTIFFs)
        out_format = str(config.get('climatology_format', 'nc')).lower()
        
        # Retain coordinate reference system if present
        if "spatial_ref" in ds.coords:
            climatology = climatology.assign_coords(spatial_ref=ds.coords["spatial_ref"])
            
        climatology.attrs["description"] = f"Long-term climatology averages (1-12) using {wind_file_suffix} wind model"
        
        if out_format in ('nc', 'netcdf'):
            # Save to a single combined climatology NetCDF file
            out_file = os.path.join(out_dir, f"wwte_climatology_{wind_file_suffix}_combined.nc")
            climatology.to_netcdf(out_file)
            print(f"Successfully saved combined climatology (NetCDF): {out_file}")
        elif out_format in ('tif', 'tiff'):
            # Export per-month GeoTIFFs for each variable
            if rasterio is None:
                raise RuntimeError("rasterio is required to export GeoTIFFs but is not available in the environment")

            # Helper to get lon/lat grid
            def _get_lonlat(ds_obj):
                if 'lon' in ds_obj.coords and 'lat' in ds_obj.coords:
                    lons = ds_obj['lon'].values
                    lats = ds_obj['lat'].values
                    return lons, lats
                if 'x' in ds_obj.coords and 'y' in ds_obj.coords:
                    lons = ds_obj['x'].values
                    lats = ds_obj['y'].values
                    return lons, lats
                raise RuntimeError('No lon/lat or x/y coordinates found for GeoTIFF export')

            lons, lats = _get_lonlat(climatology)
            nlat = len(lats)
            nlon = len(lons)

            # compute pixel size (assume regular grid)
            xres = float((lons.max() - lons.min()) / max(nlon - 1, 1))
            yres = float((lats.max() - lats.min()) / max(nlat - 1, 1))

            for month in climatology['month'].values:
                ds_month = climatology.sel(month=month)
                mm = f"{int(month):02d}"
                for var in ds_month.data_vars:
                    arr = ds_month[var].values
                    # Ensure 2D (lat, lon)
                    if arr.ndim != 2:
                        print(f"Skipping variable '{var}' for month {mm}: unsupported dims {arr.shape}")
                        continue

                    # Rasterio expects data in (bands, rows, cols); we'll write single-band TIFF
                    out_tif = os.path.join(out_dir, f"wwte_climatology_{wind_file_suffix}_{var}_{mm}.tif")
                    transform = from_origin(lons.min() - xres / 2.0, lats.max() + yres / 2.0, xres, yres)

                    # Determine dtype
                    dtype = arr.dtype
                    # Replace NaNs with a nodata value and cast
                    nodata = -9999
                    write_arr = arr.copy()
                    write_arr = write_arr.astype('float32')
                    write_arr[np.isnan(write_arr)] = nodata

                    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
                    with rasterio.Env():
                        with rasterio.open(
                            out_tif,
                            'w',
                            driver='GTiff',
                            height=write_arr.shape[0],
                            width=write_arr.shape[1],
                            count=1,
                            dtype='float32',
                            crs='EPSG:4326',
                            transform=transform,
                            nodata=nodata,
                            compress='deflate'
                        ) as dst:
                            dst.write(write_arr, 1)

                    print(f"Saved GeoTIFF: {out_tif}")
        else:
            raise ValueError(f"Unsupported climatology_format: {out_format}. Use 'nc' or 'tif'.")
        
        ds.close()
        climatology.close()
        
    except Exception as e:
        print(f"Error calculating climatology: {e}")


if __name__ == "__main__":
    main()
