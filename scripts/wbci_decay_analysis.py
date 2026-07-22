"""
WBCI Exponential Decay Analysis
Cities: Hyderabad | Ahmedabad | Pune | Surat
Change LST raster paths and water body coordinates
per city before running. 
Ahmedabad (Sabarmati River): longitude = 72.57567, latitude = 23.01789
Pune (Mula-Mutha River Confluence): longitude = 73.860443, latitude = 18.531343
Surat (Tapi River): longitude = 72.797502, latitude = 21.239055
"""

import rasterio
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

# ============================================================
# INPUT UTM RASTERS
# ============================================================

LST_2015 = "LST_2015_UTM.tif"
LST_2020 = "LST_2020_UTM.tif"
LST_2025 = "LST_2025_UTM.tif"

# ============================================================
# HUSSAIN SAGAR CENTER (WGS84)
# ============================================================

lake_lon = 78.47343
lake_lat = 17.4230

# ============================================================
# CONVERT POINT TO UTM
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32644",
    always_xy=True
)

x_utm, y_utm = transformer.transform(
    lake_lon,
    lake_lat
)

center_point = Point(x_utm, y_utm)

# ============================================================
# DISTANCE RINGS (METERS)
# ============================================================

DISTANCES = np.unique(np.concatenate([
    np.arange(0, 300, 60),
    np.arange(300, 900, 120),
    np.arange(900, 1800, 180),
    np.arange(1800, 2850, 210),
    np.arange(2850, 4050, 240),
    np.arange(4050, 5400, 270),
    np.arange(5400, 6901, 300)
]))
# ============================================================
# PCI (WBCI) MODEL
# ============================================================

def pci_model(d, A, B, C):

    return C - A * np.exp(-B * d)

# ============================================================
# EXTRACT RADIAL PROFILE
# ============================================================

def extract_radial_profile(lst_raster_path):

    with rasterio.open(lst_raster_path) as src:

        raster_crs = src.crs

        # ====================================================
        # CREATE RING BUFFERS
        # ====================================================

        rings = []

        for i in range(len(DISTANCES)-1):

            inner_d = DISTANCES[i]
            outer_d = DISTANCES[i+1]

            outer_buffer = center_point.buffer(
                outer_d
            )

            if inner_d == 0:

                ring = outer_buffer

            else:

                inner_buffer = center_point.buffer(
                    inner_d
                )

                ring = outer_buffer.difference(
                    inner_buffer
                )

            rings.append({
                'geometry': ring,
                'distance_mid_m': (
                    inner_d + outer_d
                ) / 2
            })

        gdf = gpd.GeoDataFrame(
            rings,
            crs=raster_crs
        )

        # ====================================================
        # EXTRACT MEAN LST
        # ====================================================

        lst_values = []

        for idx, row in gdf.iterrows():

            try:

                masked, _ = mask(
                    src,
                    [row.geometry],
                    crop=True
                )

                arr = masked[0].astype(float)

                # Remove nodata / invalid pixels
                arr[arr <= 0] = np.nan

                mean_val = np.nanmean(arr)

                lst_values.append(mean_val)

            except:

                lst_values.append(np.nan)

        distances_km = (
            gdf['distance_mid_m'].values / 1000
        )

    return {
        'distances_km': distances_km,
        'lst': np.array(lst_values)
    }

# ============================================================
# FIT PCI (WBCI) MODEL
# ============================================================

def fit_pci_model(distances_km, lst_values):

    valid = ~np.isnan(lst_values)

    d = distances_km[valid]

    lst = lst_values[valid]

    # Need enough valid points
    if len(lst) < 5:

        print("Too few valid observations")

        return None

    try:

        # Initial guesses
        A0 = np.max(lst) - np.min(lst)

        B0 = 1

        C0 = np.min(lst)

        popt, pcov = curve_fit(

            pci_model,

            d,
            lst,

            p0=[A0, B0, C0],

            bounds=(
                [0, 0, 0],
                [20, 20, 100]
            ),

            maxfev=30000
        )

        A, B, C = popt

        perr = np.sqrt(np.diag(pcov))

        predicted = pci_model(
            d,
            *popt
        )

        residuals = lst - predicted

        ss_res = np.sum(
            residuals**2
        )

        ss_tot = np.sum(
            (lst - np.mean(lst))**2
        )

        r_squared = 1 - (
            ss_res / ss_tot
        )

        rmse = np.sqrt(
            np.mean(residuals**2)
        )

        return {

            'A': A,
            'B': B,
            'C': C,

            'A_err': perr[0],
            'B_err': perr[1],
            'C_err': perr[2],

            'r_squared': r_squared,
            'rmse': rmse
        }

    except Exception as e:

        print(f"Fit failed: {e}")

        return None

# ============================================================
# RUN ANALYSIS
# ============================================================

print("=" * 70)
print("HYDERABAD WBCI DECAY ANALYSIS")
print("=" * 70)

results = {}
profiles = {}

for year, path in zip(

    [2015, 2020, 2025],

    [LST_2015, LST_2020, LST_2025]
):

    print(f"\nProcessing {year}...")

    profile = extract_radial_profile(
        path
    )

    profiles[year] = profile

    fit = fit_pci_model(

        profile['distances_km'],
        profile['lst']
    )

    results[year] = fit

    if fit:

        print(f"\n{year} RESULTS")
        print("-" * 40)

        print(
            f"A (Cooling Magnitude): "
            f"{fit['A']:.3f} ± {fit['A_err']:.3f} °C"
        )

        print(
            f"B (Decay Rate): "
            f"{fit['B']:.3f} ± {fit['B_err']:.3f} km⁻¹"
        )

        print(
            f"C (Background Temperature): "
            f"{fit['C']:.3f} ± {fit['C_err']:.3f} °C"
        )

        print(
            f"R²: {fit['r_squared']:.4f}"
        )

        print(
            f"RMSE: {fit['rmse']:.4f}"
        )

# ============================================================
# TREND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("COOLING TREND")
print("=" * 70)

A_vals = []

for year in [2015, 2020, 2025]:

    if results[year]:

        A_vals.append(results[year]['A'])

        print(
            f"{year}: "
            f"{results[year]['A']:.3f} °C"
        )

# ============================================================
# PLOTS
# ============================================================

plt.style.use('default')

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
    facecolor='white'
)

# ============================================================
# PCI CURVES
# ============================================================

ax = axes[0]

# Pure white background
ax.set_facecolor('white')

colors = ['blue', 'orange', 'red']

for year, color in zip(
    [2015, 2020, 2025],
    colors
):

    d = profiles[year]['distances_km']
    lst = profiles[year]['lst']

    ax.scatter(
        d,
        lst,
        s=40,
        alpha=0.85,
        color=color,
        edgecolors='none',
        label=str(year)
    )

    if results[year]:

        d_smooth = np.linspace(
            np.min(d),
            np.max(d),
            300
        )

        fit_curve = pci_model(
            d_smooth,
            results[year]['A'],
            results[year]['B'],
            results[year]['C']
        )

        ax.plot(
            d_smooth,
            fit_curve,
            linewidth=2,
            color=color
        )

ax.set_xlabel(
    "Distance from lake (km)",
    fontsize=11
)

ax.set_ylabel(
    "LST (°C)",
    fontsize=11
)

ax.set_title(
    "PCI decay curves",
    fontsize=13,
    weight='bold'
)

# REMOVE GRID
ax.grid(False)

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(
    frameon=False,
    loc='lower right'
)

# ============================================================
# COOLING TREND
# ============================================================

ax2 = axes[1]

ax2.set_facecolor('white')

years_arr = np.array([
    2015,
    2020,
    2025
])

ax2.plot(
    years_arr,
    A_vals,
    marker='o',
    linewidth=2.2,
    markersize=7
)

ax2.set_xlabel(
    "Year",
    fontsize=11
)

ax2.set_ylabel(
    "Cooling magnitude A (°C)",
    fontsize=11
)

ax2.set_title(
    "Lake cooling trend",
    fontsize=13,
    weight='bold'
)

# REMOVE GRID
ax2.grid(False)

# Clean axes
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ============================================================
# FINAL LAYOUT
# ============================================================

plt.tight_layout()

plt.savefig(
    "WBCI_Decay_Hyderabad.png",
    dpi=600,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

# ============================================================
# EXPORT RESULTS
# ============================================================

table_data = []

for year in [2015, 2020, 2025]:

    fit = results[year]

    if fit:

        table_data.append({

            'City': 'Hyderabad',

            'Year': year,

            'A_cooling_C': round(
                fit['A'], 3
            ),

            'B_decay_km_inv': round(
                fit['B'], 3
            ),

            'C_baseline_C': round(
                fit['C'], 3
            ),

            'R_squared': round(
                fit['r_squared'], 4
            ),

            'RMSE': round(
                fit['rmse'], 4
            )
        })

df_results = pd.DataFrame(
    table_data
)

df_results.to_csv(
    "WBCI_Results_Hyderabad.csv",
    index=False
)

print("\nSaved: WBCI_Results_Hyderabad.csv")

print("\nFINAL RESULTS")
print(df_results.to_string(index=False))
