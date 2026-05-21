# File: download_all_wind_10m.py
# Description:
# This script downloads a single consolidated NetCDF file containing monthly 
# ERA5 10-meter surface wind components (u and v) for the period 2001 to 2024.
# This single combined request optimizes Copernicus CDS queuing and download times.
#
# How to run:
# python resources/download_all_wind_10m.py
#
# Dependencies:
# - Python 3.10+
# - cdsapi
#
# Expected inputs:
# - CDS API key configured in ~/.cdsapirc
#
# Expected outputs:
# - Combined NetCDF file: data/Wind/era5_monthly_wind10m_combined.nc

import os
from pathlib import Path
import cdsapi


# Bounding box for the region requested by the user: lon 30 to 75, lat 11 to 53
# CDS area format is [North, West, South, East]
AREA = [53, 30, 11, 75]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "inputs" / "ERA5_Wind"


def main() -> None:
    """
    Submits a single retrieval request for ERA5 monthly single-level 10m wind variables 
    across all years 2001-2024, saving the combined NetCDF output to disk.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "era5_monthly_wind10m_combined.nc"

    client = cdsapi.Client()

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": [str(y) for y in range(2001, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
    }

    print("Submitting combined CDS request for years 2001-2024:")
    print(f"  dataset=reanalysis-era5-single-levels-monthly-means")
    print(f"  years=2001-2024, months=01-12")
    print(f"  variables=['10m_u_component_of_wind', '10m_v_component_of_wind']")
    print(f"  area(N,W,S,E)={AREA}")

    client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(out_path))
    print(f"Successfully downloaded combined 10m wind: {out_path}")


if __name__ == "__main__":
    main()
