import math
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
import rasterio.plot

DATA_DIR = "data/gdrive"
OUT_DIR  = os.path.join(DATA_DIR, "analysis_out")
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = ["06", "07"]

ZAHEDAN_LON        = 60.86
ZAHEDAN_LAT        = 29.50
ZAHEDAN_BUFFER_DEG = 0.3
DECAY_LENGTH_KM    = 800.0
EARTH_RADIUS_KM    = 6371.0

def read_array(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr    = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, {
            "transform": src.transform,
            "crs":       src.crs,
            "width":     src.width,
            "height":    src.height,
            "bounds":    src.bounds,
        }

def resample_to_reference(src_path: str, ref_meta: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32")
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan

        out = np.full(
            (ref_meta["height"], ref_meta["width"]), np.nan, dtype="float32"
        )
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

def build_af_mask(transform, shape) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
        "ne_110m_admin_0_countries.zip"
    )
    world = world.rename(columns={c: c.lower() for c in world.columns})
    if "iso_a3" not in world.columns:
        for candidate in ("adm0_a3", "sov_a3", "gu_a3"):
            if candidate in world.columns:
                world["iso_a3"] = world[candidate]
                break

    af = world[world["iso_a3"] == "AFG"].to_crs("EPSG:4326")
    mask = geometry_mask(
        list(af.geometry),
        out_shape=shape,
        transform=transform,
        invert=True,
    )
    return mask, af

def pixel_lon_lat(transform, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(width)
    rows = np.arange(height)
    xs   = transform.c + (cols + 0.5) * transform.a
    ys   = transform.f + (rows + 0.5) * transform.e
    lon2d, lat2d = np.meshgrid(xs, ys)
    return lon2d.astype("float32"), lat2d.astype("float32")

def haversine_km(lon1, lat1, lon2, lat2):
    rlat1 = np.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat  = np.radians(lat2 - lat1)
    dlon  = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(rlat1) * math.cos(rlat2) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

def bearing_unit_vector(lon2d, lat2d):
    cos_lat = np.cos(np.radians((lat2d + ZAHEDAN_LAT) / 2.0))
    dlon    = (ZAHEDAN_LON - lon2d) * cos_lat
    dlat    = ZAHEDAN_LAT - lat2d

    mag  = np.sqrt(dlon ** 2 + dlat ** 2)
    mag  = np.where(mag == 0, np.nan, mag)
    return (dlon / mag).astype("float32"), (dlat / mag).astype("float32")

def compute_score_omega(aod, u, v, lon2d, lat2d, af_mask):
    dx_hat, dy_hat = bearing_unit_vector(lon2d, lat2d)
    wind_mag = np.sqrt(u ** 2 + v ** 2)

    valid = (
        af_mask
        & np.isfinite(aod)
        & np.isfinite(u)
        & np.isfinite(v)
        & (wind_mag > 0)
    )

    u_hat = np.where(wind_mag > 0, u / wind_mag, np.nan)
    v_hat = np.where(wind_mag > 0, v / wind_mag, np.nan)

    cosang = u_hat * dx_hat + v_hat * dy_hat
    toward = valid & (cosang > 0)

    speed_max  = float(np.nanmax(wind_mag[toward])) if np.any(toward) else 1.0
    speed_norm = wind_mag / max(speed_max, 1e-6)

    dist_km    = haversine_km(lon2d, lat2d, ZAHEDAN_LON, ZAHEDAN_LAT)
    dist_decay = np.exp(-dist_km / DECAY_LENGTH_KM)

    score = np.full(aod.shape, np.nan, dtype="float32")
    score[toward] = (
        aod[toward]
        * np.maximum(cosang[toward], 0.0)
        * speed_norm[toward]
        * dist_decay[toward]
    )

    weight = np.zeros_like(aod, dtype="float32")
    weight[toward] = np.maximum(cosang[toward], 0.0) * speed_norm[toward]
    
    score_omega = np.full(aod.shape, np.nan, dtype="float32")
    score_omega[toward] = score[toward] * weight[toward]
    
    return score_omega

def main():
    speed_06, meta_06 = read_array(os.path.join(DATA_DIR, "speed_06.tif"))
    af_mask, af_poly = build_af_mask(meta_06["transform"], (meta_06["height"], meta_06["width"]))
    lon2d, lat2d = pixel_lon_lat(meta_06["transform"], meta_06["width"], meta_06["height"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, m in enumerate(MONTHS):
        speed, meta = read_array(os.path.join(DATA_DIR, f"speed_{m}.tif"))
        u, _        = read_array(os.path.join(DATA_DIR, f"u_avg_{m}.tif"))
        v, _        = read_array(os.path.join(DATA_DIR, f"v_avg_{m}.tif"))

        aod_path = os.path.join(DATA_DIR, f"Avg_AOD_24yr_Month{m}_2001_2024.tif")
        aod      = resample_to_reference(aod_path, meta)

        score_omega = compute_score_omega(aod, u, v, lon2d, lat2d, af_mask)
        
        ax = axes[i]
        
        extent = [meta["bounds"].left, meta["bounds"].right, meta["bounds"].bottom, meta["bounds"].top]
        
        im = ax.imshow(score_omega, extent=extent, cmap='viridis', origin='upper')
        af_poly.boundary.plot(ax=ax, color='black', linewidth=1)
        
        # Plot Zahedan
        ax.plot(ZAHEDAN_LON, ZAHEDAN_LAT, 'r*', markersize=10, label='Zahedan')
        
        # Trim map to study area
        minx, miny, maxx, maxy = af_poly.total_bounds
        minx = min(minx, ZAHEDAN_LON) - 1.0
        maxx = max(maxx, ZAHEDAN_LON) + 1.0
        miny = min(miny, ZAHEDAN_LAT) - 1.0
        maxy = max(maxy, ZAHEDAN_LAT) + 1.0
        
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        
        ax.set_title(f'score * omega for Month {m}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(im, ax=ax, label='score * omega')
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "score_omega_map.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {out_path}")

if __name__ == "__main__":
    main()
