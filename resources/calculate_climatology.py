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
import xarray as xr
import warnings


warnings.filterwarnings("ignore", message="Mean of empty slice")


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
        ds = xr.open_dataset(in_file)
        
        # Calculate the long-term monthly climatology across the time dimension
        print(f"Calculating climatology for all 12 months using xarray groupby...")
        climatology = ds.groupby('time.month').mean(dim='time', skipna=True)
        
        # Retain coordinate reference system if present
        if "spatial_ref" in ds.coords:
            climatology = climatology.assign_coords(spatial_ref=ds.coords["spatial_ref"])
            
        climatology.attrs["description"] = f"Long-term climatology averages (1-12) using {wind_file_suffix} wind model"
        
        # Save to a single combined climatology file
        out_file = os.path.join(out_dir, f"wwte_climatology_{wind_file_suffix}_combined.nc")
        climatology.to_netcdf(out_file)
        print(f"Successfully saved combined climatology: {out_file}")
        
        ds.close()
        climatology.close()
        
    except Exception as e:
        print(f"Error calculating climatology: {e}")


if __name__ == "__main__":
    main()
