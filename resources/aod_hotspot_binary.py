"""
Create binary hotspot TIFF files from monthly averaged AOD TIFF inputs.

Rule:
- NaN is treated as 0
- Pixel >= 1 becomes 1, otherwise 0

How to run:
python resources/aod_hotspot_binary.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import rasterio


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_dir = os.path.join(
        base_dir,
        "inputs",
        "Hotspot_binary",
        "results_24year_averages",
    )

    output_dir = os.path.join(input_dir, "binary_maps")
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))

    if not tiff_files:
        print(f"No TIFF files found in: {input_dir}")
        return

    for tif_path in tiff_files:
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # Treat NaN as zero before thresholding.
            data = np.nan_to_num(data, nan=0.0)

            # Convert to binary hotspot map.
            binary = np.where(data >= 1, 1, 0).astype(np.uint8)

            profile.update(
                dtype=rasterio.uint8,
                count=1,
                nodata=0,
                compress="lzw",
            )

            out_path = os.path.join(output_dir, f"binary_{os.path.basename(tif_path)}")

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(binary, 1)

        print(f"Saved: {out_path}")

    print("DONE")


if __name__ == "__main__":
    main()
