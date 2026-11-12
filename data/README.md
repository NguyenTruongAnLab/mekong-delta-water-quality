# Data availability and directory structure

This directory contains the publicly redistributable agricultural statistics and documents the external data portals and restricted Zenodo archive for:

> Nguyen, A. T., Némery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T.-M.-T., Nguyen-Dam, Q.-H. & Garnier, J. River regulation and tidal amplification drive organic pollution across the Mekong Delta (submitted to *Communications Earth & Environment*, 2026).

## Derived datasets included in this repository

The harmonized station-year caches and precomputed output tables listed below are included directly in this repository so that all panel fixed-effects models, multi-buffer sensitivity tests, isometric log-ratio checks, LOYO/LOSO cross-validations, and TreeSHAP attributions run without external downloads:

| Path | Description | Coverage |
|---|---|---|
| `analysis/cache/station_year_master_latest.csv` | Harmonized station-year panel combining water quality, 16 km satellite LULC metrics, population density, rainfall, discharge, and tidal range | 23 stations (4 MRC, 19 Ben Tre), 1985--2022 (411 station-years, 151 variables) |
| `analysis/cache/bentre_station_year_bod5_halide_supplement.csv` | Station-year five-day biochemical oxygen demand (`BOD5_annual`) and mercuric-sulfate-masked dichromate COD (`COD_Sal_le_3p6_annual`) | 19 Ben Tre stations, 2005--2022 |
| `analysis/cache/lulc_metrics_corrected_{2,5,10,15,16}km.csv` | Annual buffer land-cover composition (%) and PyLandStats configuration metrics (patch density, edge density) across 2, 5, 10, 15, and 16 km radii from JAXA VLUCD | 23 stations, 1990--2020 |
| `analysis/wrtds_egret/outputs/*.csv` | Precomputed WRTDS flow-normalized concentration and flux decomposition tables | 4 MRC mainstem stations, 1985--2023 |
| `analysis/analysis_outputs/*.csv` | Precomputed statistical tables (variance partitioning, tidal range by year, LOSO/LOYO skill, TreeSHAP, 1--30 km buffer correlations, salinity mixing) | Full study period |
| `scripts/99_publish/cee_verification/metrics/*.csv` | Precomputed verification tables for Garcia--Gordon AOU partitioning, SMK trends, Livezey--Chen field significance, halide-masked COD and BOD5 mixing, ILR concordance, and 2--16 km panel fixed effects | Full study period |
| `scripts/99_publish/cee_figures/metrics/*.csv` | Source data summaries exported alongside Main Text Figs. 1--7 and Supplementary Figs. 1--12 | Full study period |
| `data/Fertilizers/FAO/Vietnam_FAOSTAT_data.csv` | FAOSTAT national agricultural use and cropland application rates for nitrogen (N), phosphate (P2O5), and potash (K2O) | Viet Nam, 1961--2023 |
| `data/Fertilizers/GSO_Vietnam/*.csv` | General Statistics Office of Viet Nam provincial paddy planted area and aquaculture production series | Viet Nam provinces, 1995--2024 |
| `data/Fertilizers/Worldbank/Vietnam_Fertilizer_consumption.csv` | World Bank national fertilizer consumption per hectare of arable land | Viet Nam, 1961--2022 |

## External public data portals and restricted Zenodo archive

Raw monitoring records from provincial and regional agencies are governed by institutional data-sharing agreements and archived under restricted access on Zenodo for non-commercial research and peer review (concept DOI https://doi.org/10.5281/zenodo.20024193; version record https://doi.org/10.5281/zenodo.20024194). Place the raw files from `data.zip` at the paths below to run the observation-level monthly and hourly scripts (`cee_verify_oxygen_and_trends.py`, `cee_verify_halide_and_bod5_mixing.py`, `03_08_cq_regime_shift.py`, `03_11_mass_balance.py`, and `cee_figures/cee_run_all.py`):

| Dataset | Expected local path | Source and access |
|---|---|---|
| MRC mainstem and Ben Tre DONRE water-quality records | `data/mekong_delta_water_quality_1985_2023.csv` | MRC Data Portal (https://portal.mrcmekong.org/) for mainstem stations; Ben Tre Department of Natural Resources and Environment (DONRE) via restricted Zenodo archive (https://doi.org/10.5281/zenodo.20024193) |
| Daily discharge and hourly water-level records (MRC & SIWRP) | `data/hydrology/outputs/final_datasets/Mekong_Delta_Gold_Standard_Master_1985_2025.csv`<br>`data/hydrology/outputs/final_datasets/water_level_unified_subset_1985_2022.csv` | MRC Data Portal and Southern Institute for Water Resources Planning (SIWRP) via restricted Zenodo archive (https://doi.org/10.5281/zenodo.20024193) |
| Viet Nam Land Use and Land Cover Dataset (VLUCD v21.09, 30 m) | `data/lulc_raw/` | JAXA Earth Observation Research Center (https://www.eorc.jaxa.jp/ALOS/en/dataset/lulc/lulc_vnm_v2109_e.htm) |
| Gridded population density (GlobPOP 30 arc-second) | `data/population_raw/` | Zenodo open repository (https://doi.org/10.5281/zenodo.11179644) |
| Precipitation (ERA5-Land and CHIRPS v3.0) | `data/climate_raw/` | Copernicus Climate Data Store (https://cds.climate.copernicus.eu) and UCSB Climate Hazards Center (https://www.chc.ucsb.edu/data/chirps3) |
