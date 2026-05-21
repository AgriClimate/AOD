import os
import requests
import re
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Configuration
# Switching to the Floating Point archive for scientific data (-1.0 to 1.0)
BASE_URL = "https://neo.gsfc.nasa.gov/archive/geotiff.float/MOD_NDVI_M/"
DATA_DIR = os.path.abspath("inputs/NEO_MOD_NDVI_M")
START_YEAR = 2000
END_YEAR = 2025

def get_file_list(url):
    """Fetches the directory listing and extracts .tif links for scientific data."""
    print(f"Fetching file list from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Pattern for MOD13C2 (Monthly) Floating Point files
    # Format: MOD13C2.AYYYYDDD.061.TIMESTAMP.tif
    pattern = r'href="([^"]+MOD13C2\.A(\d{4})\d{3}\.061\.\d+\.tif)"'
    matches = re.findall(pattern, response.text)
    
    filtered_files = []
    for filename, year_str in matches:
        year = int(year_str)
        if START_YEAR <= year <= END_YEAR:
            filtered_files.append(filename)
            
    # Remove duplicates and sort
    return sorted(list(set(filtered_files)))

def download_file(filename):
    """Downloads a single file if it doesn't already exist."""
    url = urljoin(BASE_URL, filename)
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        return
    
    try:
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    files = get_file_list(BASE_URL)
    print(f"Found {len(files)} floating-point files to download for years {START_YEAR}-{END_YEAR}.")

    if not files:
        print("No files found matching the criteria.")
        return

    # Using 5 parallel threads for faster downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_file, files)

    print("Download process completed.")

if __name__ == "__main__":
    main()
