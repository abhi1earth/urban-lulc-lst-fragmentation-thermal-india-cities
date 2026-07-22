# Urban LULC, LST and Landscape Fragmentation Analysis
# Indian Million-Plus Cities (2015–2025)

---

## Study

Urban morphology and divergent thermal trajectories: An integrated 
multi-city diagnostic framework for Indian million-plus cities (2015–2025)

**Submitted to:** Remote Sensing Applications: Society and Environment (RSASE)  
**Status:** Under revision

---

## Platform

Google Earth Engine (GEE) | Python (Google Colab)

---

## Data Source

Landsat 8 Collection 2 Level-2 (OLI/TIRS) imagery — USGS EarthExplorer  
https://earthexplorer.usgs.gov

---

## Cities

Hyderabad | Ahmedabad | Pune | Surat

---

## Analysis Period

2015 | 2020 | 2025

---

## Key Methods

- LULC Classification (Random Forest, 18-feature stack,
  spatial block cross-validation, SEED=42)
- SHAP-based class-specific feature importance analysis
- LST retrieval (Landsat 8 C2L2 single-channel algorithm)
- LST–Spectral Index Pearson correlation analysis
- LST-based thermal anomaly classification
  (Global SD-threshold: μ ± nσ)
- Surface Urban Heat Island (SUHI) quantification
- Thermal Intensity Ratio (TIR) — novel normalised metric
- Landscape fragmentation analysis (PyLandStats)
- Water Body Cool Island (WBCI) Exponential Decay Modelling
  (scipy.optimize.curve_fit)

---

## Repository Structure

```
/scripts/
    shap_beeswarm_final.py          — SHAP beeswarm plot generation
                                      (4 cities × 3 years + averaged)
    wbci_decay_analysis.py          — WBCI exponential decay modelling
                                      (PCI curve fitting, R², RMSE)
    correlation_with_normality.py   — LST vs spectral indices
                                      Pearson correlation with
                                      Shapiro-Wilk normality assessment
                                      and Q-Q plot generation

/sample_points/
    Sample training and validation point coordinates
    for all 12 city-year combinations (CSV format)
    Columns: City | Year | Longitude | Latitude |
             Class_Label | Class_Name | Split

/supplementary/
    Landsat_Scene_Metadata.csv
    — Scene IDs, acquisition dates, Path/Row,
      cloud cover (%) for all 12 city-year scenes
```

---

## Code Availability

Python analysis scripts are available in `/scripts/`.

GEE JavaScript scripts for image preprocessing,
spectral index computation, and RF classification
will be uploaded upon manuscript acceptance.

Full Google Colab notebooks (.ipynb) will be uploaded
upon manuscript acceptance.

---

## Data Availability

Training and validation sample point coordinates for
all 12 city-year combinations are available in `/sample_points/`.

Landsat 8 OLI/TIRS C2L2 imagery is freely accessible
via USGS EarthExplorer: https://earthexplorer.usgs.gov

Scene-specific metadata (Scene IDs, acquisition dates,
cloud cover) are provided in `/supplementary/Landsat_Scene_Metadata.csv`
and in Supplementary Table S2 of the manuscript.

For data or code requests during peer review, contact:
abhishekchakraborty392@gmail.com

---

## Author

**Abhishek Chakraborty**  
MSc Geography (Advances in Remote Sensing and Urban GIS)  
Department of Geography, School of Earth Sciences  
Central University of Tamil Nadu

---

## Supervisor

**Prof. Sulochana Shekhar**  
Professor, Department of Geography, School of Earth Sciences  
Central University of Tamil Nadu

---

## Citation

To be updated upon acceptance.
