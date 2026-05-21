# File: download_era5_wind.py
# Description:
# This script downloads monthly ERA5 wind data (u and v components) either at 
# the 850 hPa pressure level or at the 10-meter surface level for a specified region.
#
# How to run:
# python resources/download_era5_wind.py --start_year 2001 --end_year 2024 --type 850hPa
# python resources/download_era5_wind.py --start_year 2001 --end_year 2024 --type 10m
#
# Dependencies:
# - Python 3.10+
# - cdsapi
# - netCDF4
#
# Expected inputs:
# - CDS API key configured on the system
#
# Expected outputs:
# - NetCDF files in data/wind/ directory:
#   - era5_monthly_wind850_YYYY.nc
#   - era5_monthly_wind10m_YYYY.nc

import argparse
import io
import os
from pathlib import Path
import zipfile
from typing import List

import cdsapi
import netCDF4 as nc


# Bounding box for the region requested by the user: lon 30 to 75, lat 11 to 53
# CDS area format is [North, West, South, East]
AREA: List[int] = [53, 30, 11, 75]

SCRIPT_DIR: Path = Path(__file__).resolve().parent
OUT_DIR: Path = SCRIPT_DIR.parent / "inputs" / "ERA5_Wind"


def build_output_path(year: int, wind_type: str) -> Path:
    """
    Constructs the target NetCDF file path for the requested year and wind type.

    Args:
        year (int): The year to download.
        wind_type (str): Either '850hPa' or '10m'.

    Returns:
        Path: The absolute path of the output NetCDF file.
    """
    suffix = "wind850" if wind_type == "850hPa" else "wind10m"
    return OUT_DIR / f"era5_monthly_wind{suffix[4:]}_{year}.nc"


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


def download_year_12months(year: int, output_path: Path, wind_type: str) -> None:
    """
    Submits a CDS retrieval request for ERA5 monthly averages for a specific year, 
    and saves the resulting NetCDF file to disk.

    Args:
        year (int): The year to download.
        output_path (Path): Path where the downloaded file should be saved.
        wind_type (str): The type of wind to download ('850hPa' or '10m').

    Raises:
        Exception: If the CDS client retrieval fails.
    """
    client = cdsapi.Client()

    if wind_type == "850hPa":
        dataset = "reanalysis-era5-pressure-levels-monthly-means"
        variables = ["u_component_of_wind", "v_component_of_wind"]
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "pressure_level": ["850"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": AREA,
            "data_format": "netcdf",
        }
    else:
        dataset = "reanalysis-era5-single-levels-monthly-means"
        variables = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": AREA,
            "data_format": "netcdf",
        }

    print("Submitting CDS request:")
    print(f"  dataset={dataset}")
    print(f"  year={year}, months=01-12")
    print(f"  variables={variables}")
    if wind_type == "850hPa":
        print(f"  pressure_level=850")
    print(f"  area(N,W,S,E)={AREA}")

    client.retrieve(dataset, request, str(output_path))

    if _is_zip(output_path):
        print("CDS returned a ZIP payload. Extracting in place...")
        _extract_zip_inplace(output_path)

    print(f"Saved: {output_path}")


def inspect_sample(path: Path) -> None:
    """
    Inspects and prints dimensions and variables inside the downloaded NetCDF file.

    Args:
        path (Path): Path to the NetCDF file to inspect.
    """
    with nc.Dataset(path) as ds:
        print("\nSample file check")
        print(f"  Path: {path}")
        print(f"  Dimensions: {list(ds.dimensions.keys())}")
        print(f"  Variables: {list(ds.variables.keys())}")
        
        nc_wind_vars = ["u", "v", "u10", "v10"]
        wind_vars_found = [var for var in nc_wind_vars if var in ds.variables]
        if wind_vars_found:
            for var_name in wind_vars_found:
                var = ds.variables[var_name]
                print(f"  Wind variable {var_name} shape: {var.shape}")
                print(f"  {var_name} units: {getattr(var, 'units', 'n/a')}")
        else:
            print("  No expected wind variables found in dataset.")

        if "pressure_level" in ds.dimensions:
            print(f"  Pressure levels count: {len(ds.dimensions['pressure_level'])}")
        elif "lev" in ds.dimensions:
            print(f"  Pressure levels count: {len(ds.dimensions['lev'])}")
            
        if "valid_time" in ds.dimensions:
            print(f"  Time dimension length: {len(ds.dimensions['valid_time'])}")
        elif "time" in ds.dimensions:
            print(f"  Time dimension length: {len(ds.dimensions['time'])}")


def main() -> None:
    """
    Main entry point for command-line parsing and executing wind data downloads.
    """
    parser = argparse.ArgumentParser(
        description="Download yearly ERA5 monthly averages for pressure level or 10-meter surface level wind components."
    )
    parser.add_argument("--start_year", type=int, default=2001, help="Start year (inclusive)")
    parser.add_argument("--end_year", type=int, default=2024, help="End year (inclusive)")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["850hPa", "10m"], 
        default="850hPa", 
        help="Wind dataset type: '850hPa' pressure level or '10m' surface level wind."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start_year must be <= --end_year")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        out_path = build_output_path(year, args.type)
        if out_path.exists() and not args.force:
            print(f"Skip existing: {out_path}")
            continue

        download_year_12months(year, out_path, args.type)
        inspect_sample(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
