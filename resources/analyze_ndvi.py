# File: analyze_ndvi.py
# Description:
# This script processes monthly satellite Normalized Difference Vegetation Index (NDVI) 
# GeoTIFF files. It crops them to the configured regional bounding box to optimize 
# storage footprint, scales them strictly to the scientific range [-1.0, 1.0], 
# and saves the resulting float32 rasters.
# It supports NASA NEO uint8 (0-255) rasters, MODIS int16 (*0.0001) datasets, 
# and pre-scaled float32 grids. NoData values are converted to NaNs.
#
# How to run:
# python resources/analyze_ndvi.py --process --input-dir inputs/NEO_MOD_NDVI_M/unscaled --output-dir inputs/NEO_MOD_NDVI_M/scaled
#
# Dependencies:
# - Python 3.10+
# - numpy, rasterio
#
# Expected inputs:
# - Raw monthly NDVI GeoTIFF files (*.tif or *.TIFF)
#
# Expected outputs:
# - Cropped, scientifically scaled float32 GeoTIFF files in data/NDVI/scaled/

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds


def analyze_sample(filepath: str) -> None:
    """
    Reads a single NDVI GeoTIFF and reports raw statistics (min, max, mean) 
    along with scientifically rescaled values.

    Args:
        filepath (str): The absolute or relative path to the sample NDVI file.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata
        dtype = src.dtypes[0]
        
        print(f"--- Analysis for {os.path.basename(filepath)} ---")
        print(f"Data type: {dtype}")
        print(f"NoData value: {nodata}")
        
        # Filter out NoData and NaNs
        if np.issubdtype(data.dtype, np.floating):
            mask = ~np.isnan(data)
            if nodata is not None:
                mask &= (data != nodata)
            valid_data = data[mask]
        else:
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data
            
        if valid_data.size == 0:
            print("No valid data found in this file.")
            return

        min_val = valid_data.min()
        max_val = valid_data.max()
        print(f"Min value: {min_val}")
        print(f"Max value: {max_val}")
        print(f"Mean value: {valid_data.mean():.4f}")
        
        if dtype == 'uint8':
            print("\nInterpretation: 8-bit Raster (0-255).")
            scaled_neg1_1_min = (min_val / 255.0) * 2.0 - 1.0
            scaled_neg1_1_max = (max_val / 255.0) * 2.0 - 1.0
            print(f"Rescaled to [-1, 1]: Min={scaled_neg1_1_min:.4f}, Max={scaled_neg1_1_max:.4f}, Mean={(valid_data.mean() / 255.0) * 2.0 - 1.0:.4f}")
        elif dtype == 'int16':
            print("\nInterpretation: 16-bit Integer (MODIS scale factor 0.0001).")
            print(f"Rescaled Min: {min_val * 0.0001}")
            print(f"Rescaled Max: {max_val * 0.0001}")
        elif dtype == 'float32':
            print("\nInterpretation: 32-bit Floating Point.")
            if max_val > 1.0:
                print(f"Values > 1.0 found. Likely needs scaling.")
                if max_val <= 255.0:
                    print(f"Rescaled to [-1, 1]: Min={(min_val / 255.0) * 2.0 - 1.0:.4f}, Max={(max_val / 255.0) * 2.0 - 1.0:.4f}")
                else:
                    print(f"Applying MODIS scale factor (0.0001):")
                    print(f"Rescaled Min: {min_val * 0.0001}")
                    print(f"Rescaled Max: {max_val * 0.0001}")
            else:
                print("Values are already in range -1.0 to 1.0.")


def scale_data(data: np.ndarray, dtype: str, nodata: float | None) -> np.ndarray:
    """
    Scales raw input NDVI values strictly to the scientific range [-1.0, 1.0] 
    as float32 arrays, converting NoData values to NaNs.

    Args:
        data (np.ndarray): The raw NDVI spatial array.
        dtype (str): Raster data type name (e.g. 'uint8', 'int16', 'float32').
        nodata (float | None): The designated NoData value.

    Returns:
        np.ndarray: The scientifically scaled float32 array with NaN values.
    """
    data = data.astype(np.float32)
    
    # Mark nodata as NaN
    if nodata is not None:
        data[data == nodata] = np.nan
        
    if dtype == 'uint8':
        # Scale 8-bit [0, 255] strictly to [-1.0, 1.0]
        data = (data / 255.0) * 2.0 - 1.0
    elif dtype == 'int16':
        # Apply standard MODIS 16-bit scaling factor
        data = data * 0.0001
    elif dtype == 'float32':
        # Handle float32 values stored as raw 0-255 integers
        valid = data[~np.isnan(data)]
        if valid.size > 0 and np.nanmax(valid) > 1.0 and np.nanmax(valid) <= 255.0:
            data = (data / 255.0) * 2.0 - 1.0
            
    return data


def process_files(input_dir: str, output_dir: str, 
                  lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    """
    Batch processes, crops, and scales all NDVI TIFF files inside the input directory,
    saving the output float32 files to the output directory.

    Args:
        input_dir (str): Input directory containing raw GeoTIFF files.
        output_dir (str): Target directory for processed outputs.
        lon_min (float): Minimum longitude.
        lon_max (float): Maximum longitude.
        lat_min (float): Minimum latitude.
        lat_max (float): Maximum latitude.
    """
    os.makedirs(output_dir, exist_ok=True)

    tif_files = sorted(
        glob.glob(os.path.join(input_dir, "*.TIFF"))
        + glob.glob(os.path.join(input_dir, "*.tif"))
    )
    if not tif_files:
        print(f"No TIFF files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} files. Subsetting to "
          f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]\n")

    for filepath in tif_files:
        basename = os.path.basename(filepath)
        with rasterio.open(filepath) as src:
            # Check if the requested bbox overlaps with the file extent
            b = src.bounds
            if lon_min >= b.right or lon_max <= b.left or lat_min >= b.top or lat_max <= b.bottom:
                print(f"  {basename} -> SKIPPED (no overlap with subset bbox)")
                continue

            # Clamp bbox to file extent
            eff_lon_min = max(lon_min, b.left)
            eff_lon_max = min(lon_max, b.right)
            eff_lat_min = max(lat_min, b.bottom)
            eff_lat_max = min(lat_max, b.top)

            # Compute the pixel window for the bounding box
            window = window_from_bounds(
                eff_lon_min, eff_lat_min, eff_lon_max, eff_lat_max, src.transform
            )
            
            # Read the subset
            data = src.read(1, window=window)
            dtype = src.dtypes[0]
            nodata = src.nodata

            # Apply scaling strictly in range [-1.0, 1.0]
            scaled = scale_data(data, dtype, nodata)

            # Build the transform for the subset
            subset_transform = src.window_transform(window)
            height, width = scaled.shape

            # Write output
            out_name = os.path.splitext(basename)[0] + "_scaled.tif"
            out_path = os.path.join(output_dir, out_name)

            profile = src.profile.copy()
            profile.update(
                dtype='float32',
                height=height,
                width=width,
                transform=subset_transform,
                nodata=np.nan,
                compress='lzw',
            )

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(scaled, 1)

        print(f"  {basename} -> {out_name}  "
              f"(shape={height}x{width}, "
              f"min={np.nanmin(scaled):.4f}, max={np.nanmax(scaled):.4f})")

    print(f"\nDone. Scaled files saved to {output_dir}/")


def main() -> None:
    """
    Main parsing and orchestrating function. Handles command-line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--process" or "--process" in sys.argv:
        parser = argparse.ArgumentParser(description="NDVI Processing Tool")
        parser.add_argument("--process", action="store_true")
        parser.add_argument("--input-dir", default="inputs/NEO_MOD_NDVI_M/unscaled")
        parser.add_argument("--output-dir", default="inputs/NEO_MOD_NDVI_M/scaled")
        parser.add_argument("--lon-min", type=float, default=30.0)
        parser.add_argument("--lon-max", type=float, default=75.0)
        parser.add_argument("--lat-min", type=float, default=11.0)
        parser.add_argument("--lat-max", type=float, default=53.0)
        
        # Strip --process from argv if it's the first argument to avoid parser failure
        args_list = sys.argv[1:]
        if args_list and args_list[0] == "--process":
            args_list = args_list[1:]
            
        args = parser.parse_args(args_list)
        process_files(args.input_dir, args.output_dir,
                      args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    else:
        sample_file = sys.argv[1] if len(sys.argv) > 1 else "inputs/NEO_MOD_NDVI_M/unscaled/MOD_NDVI_M_2000-02.TIFF"
        analyze_sample(sample_file)


if __name__ == "__main__":
    main()
