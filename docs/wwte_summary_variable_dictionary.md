# WWTE Summary CSV Variable Dictionary

This document serves as the master variable dictionary and metadata reference for the consolidated Wind-Weighted Aerosol Optical Depth (AOD) Transport Efficiency (WWTE) summary CSV files:
*   `data/results/wwte_summary_wind10m.csv`
*   `data/results/wwte_summary_wind850hp.csv`

---

## Variable Definitions & Metadata Reference

### 1. `year_month`
*   **Data Type**: `string (format: YYYY-MM)`
*   **Description**: The unique key representing the calendar year and month of the observation. 
*   **Example**: `2001-06` represents June 2001.

### 2. `year`
*   **Data Type**: `integer`
*   **Description**: The four-digit Gregorian calendar year corresponding to the observation.
*   **Example**: `2001`

### 3. `month`
*   **Data Type**: `integer (zero-padded string: MM)`
*   **Description**: The two-digit calendar month corresponding to the observation.
*   **Values**: `01` (January) through `12` (December).

### 4. `zabol_aod`
*   **Data Type**: `float32 (dimensionless, typical range: 0.0 to 5.0)`
*   **Description**: The spatial average of Aerosol Optical Depth (AOD) calculated within the designated radial buffer zone surrounding the sink receptor city (Zabol). The buffer radius is defined by `"zabol_buffer_deg"` in `config.json` (typically `0.3` degrees, approximately 33 km).
*   **Formula**:
    $$\text{Sink AOD} = \frac{1}{N_{buf}} \sum_{i \in \text{buffer}} \text{AOD}_i$$

### 5. `wwte_index`
*   **Data Type**: `float32 (dimensionless, typical range: 0.0 to 1.0)`
*   **Description**: The primary Wind-Weighted Aerosol Optical Depth (AOD) Transport Efficiency Index. It integrates source aerosol loading, wind direction alignment, wind magnitude, and spatial distance decay to quantify the total potential transport of dust along active trajectories toward the sink.
*   **Formula**:
    $$\text{WWTE Index} = \frac{\sum_{i \in \text{toward}} (\text{Score}_i \times \text{Weight}_i)}{\sum_{i \in \text{toward}} \text{Weight}_i}$$
    where:
    *   $\text{Score}_i = \text{AOD}_i \times \cos(\theta_i) \times \left(\frac{U_i}{U_{max}}\right) \times e^{-d_i / L_d}$
    *   $\text{Weight}_i = \cos(\theta_i) \times \left(\frac{U_i}{U_{max}}\right)$
    *   $\theta_i$ is the angle between the wind vector and the direct bearing vector pointing from grid cell $i$ to the sink.
    *   $d_i$ is the distance in kilometers from grid cell $i$ to the sink.
    *   $L_d$ is the characteristic spatial decay length (typically 800 km).

### 6. `toward_pixel_count`
*   **Data Type**: `integer`
*   **Description**: The total count of valid grid pixels within the dynamic source hotspot region where the wind vector points downwind toward the sink receptor area (meaning the alignment angle $\theta_i$ is acute, and the cosine term $\cos(\theta_i) > 0$).

### 7. `source_valid_pixel_count`
*   **Data Type**: `integer`
*   **Description**: The total count of all valid grid pixels within the active source hotspot region that possess both clean satellite AOD observations and complete ERA5 wind vector components.

### 8. `source_aod_mean`
*   **Data Type**: `float32 (dimensionless)`
*   **Description**: The unweighted spatial average of Aerosol Optical Depth (AOD) calculated over all valid pixels within the active source hotspot region, representing the baseline atmospheric dust loading.
*   **Formula**:
    $$\text{Source AOD Mean} = \frac{1}{N_{valid}} \sum_{i \in \text{valid}} \text{AOD}_i$$

### 9. `source_aod_toward_mean`
*   **Data Type**: `float32 (dimensionless)`
*   **Description**: The unweighted spatial average of Aerosol Optical Depth (AOD) calculated strictly over the subset of source hotspot pixels where wind vectors are blowing *toward* the sink area (i.e., $\cos(\theta_i) > 0$).
*   **Formula**:
    $$\text{Source AOD Toward Mean} = \frac{1}{N_{toward}} \sum_{i \in \text{toward}} \text{AOD}_i$$

### 10. `mean_dist_decay`
*   **Data Type**: `float32 (dimensionless, range: 0.0 to 1.0)`
*   **Description**: The spatial average of the exponential distance decay factor calculated over the subset of pixels blowing toward the sink. It indicates the proximity of active dust source regions to the receptor area. Values closer to 1.0 indicate that active source regions are very close to Zabol.
*   **Formula**:
    $$\text{Mean Distance Decay} = \frac{1}{N_{toward}} \sum_{i \in \text{toward}} e^{-d_i / L_d}$$

### 11. `mean_cosang_toward`
*   **Data Type**: `float32 (dimensionless, range: 0.0 to 1.0)`
*   **Description**: The spatial average of the wind alignment cosine term ($\cos(\theta_i)$) over all grid cells blowing toward the sink. It quantifies how directly the wind vectors align with the straight-line bearing trajectories leading to Zabol.
*   **Formula**:
    $$\text{Mean Cosine Angle} = \frac{1}{N_{toward}} \sum_{i \in \text{toward}} \cos(\theta_i)$$

### 12. `high_toward_aod_mean`
*   **Data Type**: `float32 (dimensionless)`
*   **Description**: The average Aerosol Optical Depth (AOD) calculated strictly over the subset of source pixels blowing toward the sink that exceed the high-aerosol threshold. The high-aerosol threshold is dynamically calculated as the upper quartile (75th percentile) of AOD values for that specific month.
*   **Formula**:
    $$\text{High Toward AOD Mean} = \frac{1}{N_{high}} \sum_{i \in \text{high}} \text{AOD}_i \quad \text{for } i \in \{j \in \text{toward} \mid \text{AOD}_j \ge P_{75}\}$$

### 13. `high_toward_fraction`
*   **Data Type**: `float32 (range: 0.0 to 1.0)`
*   **Description**: The proportion of high-aerosol source pixels relative to the total number of pixels blowing toward the sink.
*   **Formula**:
    $$\text{High Toward Fraction} = \frac{N_{high}}{N_{toward}}$$

### 14. `wwte_index_norm`
*   **Data Type**: `float32 (range: 0.0 to 1.0)`
*   **Description**: The normalized WWTE Index, rescaled linearly between the absolute historical minimum monthly index value (mapped to `0.0`) and the historical maximum monthly index value (mapped to `1.0`) across the entire 24-year timeseries. This normalization facilitates standard comparative and trending assessments.
*   **Formula**:
    $$\text{WWTE Index Norm} = \frac{\text{WWTE} - \text{WWTE}_{min}}{\text{WWTE}_{max} - \text{WWTE}_{min}}$$

---

## Configuration Reference (`config.json`)

### `source_country`
*   **Data Type**: `string`
*   **Description**: Defines the spatial domain used as the source region for the transport analysis. Controls whether the pipeline considers the entire bounding box or restricts source pixels to a specific country boundary.
*   **Options**:
    | Value | Behavior |
    |-------|----------|
    | `"full_domain"` | Uses the entire bounding box as the source region (no country masking). |
    | `"Afghanistan"` | Restricts source pixels to Afghanistan's boundary. |
    | `"Iran"` | Restricts source pixels to Iran's boundary. |
    | `"Pakistan"` | Restricts source pixels to Pakistan's boundary. |
    | `"Iraq"` | Restricts source pixels to Iraq's boundary. |
    | `"Saudi Arabia"` | Restricts source pixels to Saudi Arabia's boundary. |
    | Any other country name | Any country available in the Natural Earth dataset can be used. Partial name matching is supported. |
*   **Default**: `"full_domain"`
*   **Example**:
    ```json
    "source_country": "full_domain"
    "source_country": "Afghanistan"
    ```
*   **Note**: Country boundaries are sourced from the Natural Earth low-resolution dataset via `geopandas`. If the specified country name is not found, the pipeline falls back to full domain with a warning.
