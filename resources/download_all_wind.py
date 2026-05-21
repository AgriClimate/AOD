# File: download_all_wind.py
# Description:
# Unified ERA5 downloader for monthly combined wind datasets.
# Users can select either 10m winds or 850mb winds via --level.
#
# How to run:
# python resources/download_all_wind.py --level 10m
# python resources/download_all_wind.py --level 850mb

from __future__ import annotations

import argparse
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
    """Checks whether the downloaded payload is a ZIP archive."""
    try:
        return path.read_bytes()[:4] == b"PK\x03\x04"
    except OSError:
        return False


def _extract_zip_inplace(path: Path) -> None:
    """Extracts the first file from a ZIP payload and replaces the ZIP file in place."""
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError("ZIP payload did not contain a file")
        inner_bytes = zf.read(names[0])
    path.write_bytes(inner_bytes)


def _build_request(level: str) -> tuple[str, dict, Path]:
    """Builds CDS dataset/request/output path for a requested wind level."""
    if level == "10m":
        dataset = "reanalysis-era5-single-levels-monthly-means"
        out_path = OUT_DIR / "era5_monthly_wind10m_combined.nc"
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [str(y) for y in range(2001, 2025)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": AREA,
            "data_format": "netcdf",
        }
        return dataset, request, out_path

    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    out_path = OUT_DIR / "era5_monthly_wind850mb_combined.nc"
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
    return dataset, request, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download combined ERA5 monthly wind dataset for 10m or 850mb."
    )
    parser.add_argument(
        "--level",
        choices=["10m", "850mb"],
        required=True,
        help="Wind level to download.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, request, out_path = _build_request(args.level)

    print("Submitting combined CDS request for years 2001-2024:")
    print(f"  dataset={dataset}")
    print("  years=2001-2024, months=01-12")
    print(f"  level={args.level}")
    print(f"  area(N,W,S,E)={AREA}")

    client = cdsapi.Client()
    client.retrieve(dataset, request, str(out_path))

    if _is_zip(out_path):
        print("CDS returned a ZIP payload. Extracting in place...")
        _extract_zip_inplace(out_path)

    print(f"Successfully downloaded combined {args.level} wind: {out_path}")


if __name__ == "__main__":
    main()