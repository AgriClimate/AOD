# File: download_aod_neo.py
# Description:
# This script downloads monthly Terra (MODAL2) and Aqua (MYDAL2) Aerosol Optical Depth (AOD) 
# geotiff files from NASA NEO, crops them to the exact study area spatial extent defined 
# in config.json using rasterio windowed reads, and computes their nan-average 
# to save a combined, cropped monthly AOD dataset.
#
# How to run:
# python resources/download_aod_neo.py
#
# Dependencies:
# - Python 3.10+
# - requests, rasterio, numpy, urllib3
#
# Expected inputs:
# - config.json defining study area bounding box
# - NASA NEO archive index pages for Terra and Aqua AOD
#
# Expected outputs:
# - Combined, cropped geotiff files saved under data/AOD/combined/
#   matching the name: MCDAL2_M_AER_OD_YYYY-MM.FLOAT.TIFF

import os
import re
import warnings
import zipfile
import io
import json
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds, transform as window_transform


# Suppress rasterio warnings about None nodata
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

TERRA_URL = "https://neo.gsfc.nasa.gov/archive/geotiff.float/MODAL2_M_AER_OD/"
AQUA_URL = "https://neo.gsfc.nasa.gov/archive/geotiff.float/MYDAL2_M_AER_OD/"

DATA_DIR = os.path.abspath("inputs/NEO_MCDAL2_M")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

START_YEAR = 2000
END_YEAR = 2025


def load_bounding_box() -> dict[str, float]:
    """
    Loads spatial extent bounding box from config.json.
    
    Returns:
        dict[str, float]: Bounding box coordinates with keys 'min_lon', 'max_lon', 'min_lat', 'max_lat'.
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['parameters']['bounding_box']
    except Exception:
        # Fallback to default study bounding box
        return {
            'min_lon': 30.0,
            'max_lon': 75.0,
            'min_lat': 11.0,
            'max_lat': 53.0
        }


def get_file_list(url: str, pattern: str) -> list[tuple[str, str, str]]:
    """
    Fetches the file index list from NASA NEO repository and filters files by date range.

    Args:
        url (str): NASA NEO archive directory URL.
        pattern (str): Regular expression pattern to search for AOD TIFF filenames.

    Returns:
        list[tuple[str, str, str]]: A list of matching filename tuples (filename, year, month).
    """
    print(f"Fetching file list from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    matches = re.findall(pattern, response.text)
    
    filtered_files = []
    for filename, year_str, month_str in matches:
        year = int(year_str)
        if START_YEAR <= year <= END_YEAR:
            filtered_files.append((filename, year_str, month_str))
            
    return filtered_files


def download_file(item: tuple[str, str]) -> str | None:
    """
    Downloads a single AOD geotiff file from NASA NEO directory.

    Args:
        item (tuple[str, str]): A tuple containing (url_base, filename).

    Returns:
        str | None: Path to the downloaded file, or None if download failed.
    """
    url_base, filename = item
    url = urljoin(url_base, filename)
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    try:
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filepath
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return None


def process_month(year_str: str, month_str: str, terra_file: str | None, aqua_file: str | None) -> None:
    """
    Crops Terra and Aqua monthly files to the study area extent and saves their nan-average.

    Args:
        year_str (str): The processing year.
        month_str (str): The processing month.
        terra_file (str | None): Path to the Terra TIFF file.
        aqua_file (str | None): Path to the Aqua TIFF file.
    """
    out_filename = f"MCDAL2_M_AER_OD_{year_str}-{month_str}.FLOAT.TIFF"
    out_filepath = os.path.join(COMBINED_DIR, out_filename)
    
    if os.path.exists(out_filepath):
        return
    
    arrays = []
    meta = None
    
    bbox = load_bounding_box()
    min_lon = bbox['min_lon']
    max_lon = bbox['max_lon']
    min_lat = bbox['min_lat']
    max_lat = bbox['max_lat']
    
    for f in [terra_file, aqua_file]:
        if f and os.path.exists(f):
            with rasterio.open(f) as src:
                # Use rasterio.windows.from_bounds to crop on read
                window = from_bounds(
                    min_lon, min_lat, max_lon, max_lat, transform=src.transform
                )
                arr = src.read(1, window=window).astype("float32")
                
                # NEO float files use 99999.0 as NoData
                arr[arr >= 99999.0] = np.nan
                arrays.append(arr)
                
                if meta is None:
                    # Update metadata according to cropped spatial dimensions
                    meta = src.meta.copy()
                    new_transform = window_transform(window, src.transform)
                    meta.update({
                        'height': arr.shape[0],
                        'width': arr.shape[1],
                        'transform': new_transform
                    })
                    
    if not arrays:
        return
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_arr = np.nanmean(np.array(arrays), axis=0)
    
    # Restore NODATA
    avg_arr[np.isnan(avg_arr)] = 99999.0
    
    # Update meta to explicitly set nodata
    if meta is not None:
        meta.update({'nodata': 99999.0})
        
        print(f"Saving combined cropped file {out_filename}...")
        with rasterio.open(out_filepath, 'w', **meta) as dst:
            dst.write(avg_arr, 1)


def main() -> None:
    """
    Main execution loop. Fetches file indices, downloads missing files, crops them,
    and averages Aqua/Terra datasets.
    """
    terra_pattern = r'href="(MODAL2_M_AER_OD_(\d{4})-(\d{2})\.FLOAT\.TIFF)"'
    aqua_pattern  = r'href="(MYDAL2_M_AER_OD_(\d{4})-(\d{2})\.FLOAT\.TIFF)"'
    
    terra_files = get_file_list(TERRA_URL, terra_pattern)
    aqua_files = get_file_list(AQUA_URL, aqua_pattern)
    
    all_downloads = []
    for f, y, m in terra_files:
        all_downloads.append((TERRA_URL, f))
    for f, y, m in aqua_files:
        all_downloads.append((AQUA_URL, f))
        
    print(f"Found {len(all_downloads)} total files to download.")
    
    # Download in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(download_file, all_downloads))
        
    # Group by month
    months = set()
    for f, y, m in terra_files + aqua_files:
        months.add((y, m))
        
    print(f"\nProcessing {len(months)} unique months...")
    
    for y, m in sorted(months):
        terra_f = os.path.join(DATA_DIR, f"MODAL2_M_AER_OD_{y}-{m}.FLOAT.TIFF")
        aqua_f = os.path.join(DATA_DIR, f"MYDAL2_M_AER_OD_{y}-{m}.FLOAT.TIFF")
        
        terra_exists = os.path.exists(terra_f)
        aqua_exists = os.path.exists(aqua_f)
        
        process_month(y, m, terra_f if terra_exists else None, aqua_f if aqua_exists else None)

    print("Download, crop, and combination process completed.")


if __name__ == "__main__":
    main()
