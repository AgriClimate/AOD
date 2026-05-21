# WWTE – Wind-Weighted AOD Transport Efficiency Model

**Version: v1.3** | **Author: Hossein Lotfi** — Research Scientist

> A geospatial analysis pipeline that quantifies the contribution of regional dust sources to aerosol loading at a designated sink location, using satellite-derived AOD, ERA5 reanalysis winds, and hotspot classification masks.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Pipeline Architecture](#pipeline-architecture)
- [Version History](#version-history)
- [License](#license)

---

## Overview

The **WWTE (Wind-Weighted Transport Efficiency)** model evaluates how aerosol optical depth (AOD) from regional dust sources is transported toward a target location (sink). It integrates three data streams:

| Data Source | Variable | Resolution |
|---|---|---|
| NASA NEO (MODIS Deep Blue) | Monthly AOD | ~0.1° |
| ERA5 Reanalysis | U/V wind components (10 m or 850 hPa) | 0.25° → 0.1° |
| Hotspot Binary Masks | Pre-classified source regions | ~0.1° |

For every grid cell, the pipeline computes a transport score that factors in:

1. **AOD magnitude** – How much aerosol is present at the source.
2. **Wind alignment** – Cosine of the angle between wind direction and the bearing toward the sink.
3. **Wind speed** – Normalized against the domain maximum.
4. **Distance decay** – Exponential decay from the source to the sink (configurable decay length).

The final WWTE index is the weighted spatial mean of these scores, summarizing total dust transport potential toward the sink for each month.

---

## How It Works

```
┌──────────────────────────────────┐
│  Satellite AOD (MODIS/NEO)       │──┐
├──────────────────────────────────┤  │
│  ERA5 Wind (10m or 850 hPa)     │──┤──▶  WWTE Scoring Engine  ──▶  NetCDF + CSV + Maps
├──────────────────────────────────┤  │
│  Hotspot Binary Masks            │──┘
└──────────────────────────────────┘
```

**Scoring formula per grid cell:**

```
Score(i,j) = AOD(i,j) × max(cos θ, 0) × (|V| / |V_max|) × exp(-d / L)
```

Where:
- `θ` = angle between wind vector and bearing toward the sink
- `|V|` = wind speed magnitude
- `|V_max|` = domain-maximum wind speed among toward-blowing cells
- `d` = great-circle distance from source cell to sink (km)
- `L` = decay length scale (default: 800 km)

---

## Project Structure

```
AOD/
├── main.py                          # Pipeline entry point (runs analysis + plotting)
├── README.md                        # This file
│
├── config/
│   └── config.json                  # Central pipeline configuration
│
├── resources/                       # Core scripts
│   ├── wwte_aod_index_analysis.py   # WWTE scoring engine (OOP)
│   ├── plot_climatology.py          # Advanced 3-panel visualizer (OOP)
│   ├── calculate_climatology.py     # Climatology calculator utility
│   ├── download_aod_neo.py          # NASA NEO AOD downloader
│   ├── download_era5_wind.py        # ERA5 wind downloader (general)
│   ├── download_all_wind_10m.py     # ERA5 10m wind batch downloader
│   ├── download_all_wind_850.py     # ERA5 850 hPa wind batch downloader
│   ├── download_ndvi_neo.py         # NDVI data downloader
│   └── analyze_ndvi.py              # NDVI analysis utility
│
├── inputs/                          # Input datasets (not tracked in git)
│   ├── ERA5_Wind/                   # ERA5 monthly wind NetCDF files
│   ├── Hotspot_binary/              # Binary hotspot classification TIFFs
│   ├── NEO_MCDAL2_M/               # MODIS AOD GeoTIFF monthly composites
│   ├── NEO_MOD_NDVI_M/             # NDVI rasters (optional)
│   └── Shpefile/                    # Auxiliary shapefiles
│
├── outputs/                         # Generated outputs
│   ├── results/                     # NetCDF and CSV files
│   │   ├── wwte_wind{active_wind_type}_combined.nc
│   │   ├── wwte_summary_wind{active_wind_type}.csv
│   │   └── climatology/
│   │       └── wwte_climatology_wind{active_wind_type}_combined.nc
│   └── plots/                       # Monthly climatology maps (PNG)
│       └── climatology_wwte_score_wind{active_wind_type}_*.png
│
├── licence/                         # Proprietary license
│   └── LICENSE.txt
│
└── docs/                            # Documentation and supplementary materials
```

---

## Requirements

### Python ≥ 3.9

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array computation |
| `pandas` | Tabular data handling |
| `xarray` | NetCDF I/O and labeled array operations |
| `rioxarray` | CRS-aware raster reprojection |
| `rasterio` | GeoTIFF I/O |
| `geopandas` | Vector geometry operations |
| `geopy` | Geocoding sink location names |
| `matplotlib` | Publication-quality visualizations |
| `shapely` | Geometry primitives |
| `cdsapi` | ERA5 data downloads (Copernicus CDS) |

### Install

```bash
pip install numpy pandas xarray rioxarray rasterio geopandas geopy matplotlib shapely cdsapi geodatasets
```

---

## Configuration

All pipeline parameters are centralized in [`config/config.json`](config/config.json):

```json
{
    "directories": {
        "hotspot_binary": "inputs/Hotspot_binary",
        "aod_combined": "inputs/NEO_MCDAL2_M",
        "wind": "inputs/ERA5_Wind",
        "ndvi": "inputs/NEO_MOD_NDVI_M",
        "shapefiles": "inputs/Shpefile",
        "output": "outputs/results",
        "plots": "outputs/plots"
    },
    "source_country": "full_domain",
    "aod_source_mode": "hotspot",
    "aod_threshold": null,
    "active_wind_type": "wind850mb",
    "sink_location": {
        "name": "Zabol, Iran"
    },
    "parameters": {
        "zabol_buffer_deg": 0.3,
        "decay_length_km": 800.0,
        "af_high_aod_quantile": 0.75,
        "resolution_deg": 0.1,
        "bounding_box": {
            "min_lon": 30.0,
            "max_lon": 75.0,
            "min_lat": 11.0,
            "max_lat": 53.0
        }
    }
}
```

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `active_wind_type` | Wind level to use: `"wind10m"` or `"wind850mb"` | `"wind850mb"` |
| `aod_source_mode` | Select AOD source pixels: `"hotspot"` (only hotspot mask=1) or `"all"` | `"hotspot"` |
| `aod_threshold` | Optional AOD threshold applied on top of source mode (use `null` to disable) | `null` |
| `sink_location.name` | Target city name (auto-geocoded via Nominatim) | `"Zabol, Iran"` |
| `source_country` | Restrict sources to a country, or `"full_domain"` | `"full_domain"` |
| `decay_length_km` | Distance decay e-folding length (km) | `800.0` |
| `resolution_deg` | Output grid resolution (degrees) | `0.1` |
| `af_high_aod_quantile` | Quantile threshold for high-AOD diagnostics | `0.75` |

> **Note:** If `lon` and `lat` are omitted from `sink_location`, the pipeline automatically geocodes the city name using OpenStreetMap's Nominatim service.

---

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This executes two stages sequentially:

1. **Stage 1 — Spatial Analysis** (`wwte_aod_index_analysis.py`)
   - Loads AOD, wind, and hotspot data for each month
   - Computes WWTE scores on a uniform grid
   - Exports monthly and climatological NetCDF files + CSV summary

2. **Stage 2 — Visualization** (`plot_climatology.py`)
   - Reads the climatology NetCDF
   - Generates 12 monthly 3-panel maps (Score, Omega, Score×Omega)

### Run Individual Components

```bash
# Analysis only
python resources/wwte_aod_index_analysis.py

# Plotting only (requires NetCDF outputs from analysis)
python resources/plot_climatology.py
```

### Download Input Data

```bash
# Download MODIS AOD from NASA NEO
python resources/download_aod_neo.py

# Download ERA5 850 hPa winds (requires CDS API key)
python resources/download_all_wind_850.py

# Download ERA5 10m winds
python resources/download_all_wind_10m.py
```

---

## Outputs

### NetCDF Variables

The output NetCDF files (`outputs/results/`) contain the following variables on a `(lat, lon)` grid:

| Variable | Description |
|---|---|
| `AOD` | Aerosol Optical Depth (MODIS Deep Blue) |
| `U` | Zonal wind component (m/s) |
| `V` | Meridional wind component (m/s) |
| `wind_speed` | Wind speed magnitude (m/s) |
| `wind_speed_norm` | Wind speed normalized by domain maximum |
| `cos_angle` | Cosine of angle between wind and sink bearing |
| `dist_decay` | Exponential distance decay factor |
| `toward_mask` | Binary: wind blows toward the sink (1/0) |
| `country_mask` | Binary: cell is inside selected source country/domain (1/0) |
| `aod_selection_mask` | Binary: cell passed AOD source mode + threshold filter (1/0) |
| `source_mask` | Binary: final analysis mask = country_mask AND aod_selection_mask (1/0) |
| `Hotspot_Mask` | Binary hotspot raster mask from input TIFF (1/0) |
| `WWTE_Score` | Transport efficiency score |
| `WWTE_Weight` | Directional-speed weight (Omega) |

### CSV Summary

The CSV file (`outputs/results/wwte_summary_wind850mb.csv`) provides per-month aggregate statistics:

| Column | Description |
|---|---|
| `year_month` | YYYY-MM identifier |
| `aod_source_mode` | AOD source filter used for that run (`hotspot` or `all`) |
| `aod_threshold` | Applied AOD threshold (null/NaN means disabled) |
| `zabol_aod` | Mean AOD within the sink buffer zone |
| `wwte_index` | Weighted transport efficiency index |
| `wwte_index_norm` | Min-max normalized WWTE index (0–1) |
| `toward_pixel_count` | Number of cells with wind blowing toward the sink |
| `source_aod_mean` | Domain-mean AOD across valid source cells |
| `high_toward_aod_mean` | Mean AOD in the top quantile of toward-blowing cells |

### Maps

Monthly climatology maps are saved as 300 DPI PNG files in `outputs/plots/`:

```
climatology_wwte_score_wind{active_wind_type}_01.png  (January)
climatology_wwte_score_wind{active_wind_type}_02.png  (February)
...
climatology_wwte_score_wind{active_wind_type}_12.png  (December)
```

Each map contains three panels:
- **Score** — Geospatial transport score factoring AOD, wind alignment, and distance decay
- **Omega** — Wind dynamics factor (alignment cosine × normalized speed)
- **Score × Omega** — Intensity-weighted transport hotspots

---

## Pipeline Architecture

```
main.py
  │
  ├──▶ resources/wwte_aod_index_analysis.py
    │     ├── ConfigManager          — Loads config/config.json, geocodes sink location
  │     ├── WWTEGeospatialEngine   — Haversine distance, bearing vectors
  │     └── WWTEPipeline           — Orchestrates loading, scoring, and export
  │           ├── initialize()
  │           ├── run_spatial_analysis()
  │           └── export_results()
  │
  └──▶ resources/plot_climatology.py
        ├── VisualizerConfig            — Typed configuration container
        ├── GeospatialBoundaryManager   — Loads country/province boundaries
        ├── WWTEGeospatialEngine        — Recomputes Score, Omega, Score×Omega
        ├── PremiumPlotter              — Custom colormaps, 3-panel rendering
        └── WWTEVisualizerPipeline      — Coordinates loading and plotting
              ├── initialize()
              └── run()
```

---

## Version History

| Version | Date | Changes |
|---|---|---|
| **v1.4** | 2026-05-21 | Moved central config to `config/config.json`; strict wind-type resolution (`wind10m`/`wind850mb`) for input/output/plots; added `aod_source_mode` (`hotspot` or `all`) and optional `aod_threshold`; fixed monthly hotspot usage to always map AOD month MM to `binary_Avg_AOD_24yr_MonthMM_2001_2024.tif`; corrected exported masks (`Hotspot_Mask`, `country_mask`, `aod_selection_mask`, `source_mask`) |
| **v1.3** | 2025-05-21 | Output NetCDF uses `lat`/`lon` dimensions (CF-compliant); added proprietary licence; restructured folders to `inputs/`, `outputs/`, `resources/`; added `main.py` pipeline coordinator |
| **v1.2** | — | Advanced OOP-based plotting engine (`plot_climatology.py`); 3-panel Score/Omega/Score×Omega maps; discrete quantile colormaps |
| **v1.1** | — | Added 850 hPa wind support; geocoding for sink location; source country masking; distance decay scoring |
| **v1.0** | — | Initial WWTE pipeline with 10 m wind, AOD loading, basic transport scoring, and CSV export |

---

## License

This project is under a **proprietary license**. See [`licence/LICENSE.txt`](licence/LICENSE.txt).

> **⚠️ No research or publication use is permitted.** Only authorized developers may use, modify, or distribute this code. Written consent is required for any external use.

---

## Author

**Hossein Lotfi** — Research Scientist

© 2025 Hossein Lotfi, Development Team. All rights reserved.
