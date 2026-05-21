# File: download_all_wind_850.py
# Description:
# This script downloads a single consolidated NetCDF file containing monthly 
# ERA5 850 hPa pressure level wind components (u and v) for the period 2001 to 2024.
# This single combined request optimizes Copernicus CDS queuing and download times.
#
# How to run:
# python resources/download_all_wind_850.py
#
# Dependencies:
# - Python 3.10+
# - cdsapi
#
# Expected inputs:
# - CDS API key configured in ~/.cdsapirc
#
# Expected outputs:
# - Combined NetCDF file: data/Wind/era5_monthly_wind850mb_combined.nc

import os
import io
import zipfile
from pathlib import Path
import cdsapi


# Bounding box for the region requested by the user: lon 30 to 75, lat 11 to 53
# CDS area format is [North, West, South, East]
AREA = [53, 30, 11, 75]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "inputs" / "ERA5_Wind"


def _is_zip(path: Path) -> bool:
    """
    Checks if the downloaded file is a ZIP archive by reading its magic bytes.

    Args:
        path (Path): The path to the downloaded file.

    Returns:
        bool: True if the file starts with ZIP magic bytes, False otherwise.
    """
    try:
        return path.read_bytes()[:4] == b"PK\x03\x04"
    except OSError:
        return False


def _extract_zip_inplace(path: Path) -> None:
    """
    Extracts the inner NetCDF file from a downloaded ZIP archive and saves it 
    directly in place of the original ZIP file.

    Args:
        path (Path): Path to the downloaded ZIP archive.

    Raises:
        RuntimeError: If the ZIP archive is empty or does not contain a file.
    """
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError("ZIP payload did not contain a file")
        inner_bytes = zf.read(names[0])
    path.write_bytes(inner_bytes)


def main() -> None:
    """
    Submits a single retrieval request for ERA5 monthly pressure-level 850hPa wind variables 
    across all years 2001-2024, saving the combined NetCDF output to disk.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "era5_monthly_wind850mb_combined.nc"

    client = cdsapi.Client()

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["850"],
        "year": [str(y) for y in range(2001, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
    }

    print("Submitting combined CDS request for years 2001-2024:")
    print(f"  dataset=reanalysis-era5-pressure-levels-monthly-means")
    print(f"  years=2001-2024, months=01-12")
    print(f"  variables=['u_component_of_wind', 'v_component_of_wind']")
    print(f"  pressure_level=850")
    print(f"  area(N,W,S,E)={AREA}")

    client.retrieve("reanalysis-era5-pressure-levels-monthly-means", request, str(out_path))

    if _is_zip(out_path):
        print("CDS returned a ZIP payload. Extracting in place...")
        _extract_zip_inplace(out_path)

    print(f"Successfully downloaded combined 850hPa wind: {out_path}")


if __name__ == "__main__":
    main()
