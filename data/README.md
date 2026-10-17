# Data availability and directory structure

This directory contains the publicly redistributable agricultural statistics and documents the external portals and restricted Zenodo archive for raw hydro-environmental monitoring records used in:

> Nguyen, A. T., Némery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T.-M.-T., Nguyen-Dam, Q.-H. & Garnier, J. Hydrodynamic shift controls organic pollution in the Mekong Delta. *Nature Water* (2026).

## Datasets included in this repository

The following harmonized and derived datasets are included directly in this repository so that all panel regressions, LOSO/LOYO cross-validations, TreeSHAP attributions, supplementary tables, and manuscript verification checks can be executed without external downloads:

| Path | Description | Coverage |
|---|---|---|
| `analysis/cache/station_year_master_latest.csv` | Harmonized station-year panel integrating annual and seasonal water quality, 16 km satellite LULC composition and configuration metrics, population density, ERA5-Land rainfall, discharge, and tidal range | 23 stations (4 MRC, 19 Ben Tre), 1985 to 2022 (411 station-years, 151 variables) |
| `analysis/cache/lulc_metrics_corrected_16km.csv` | Annual 16 km radial buffer land-use composition (%) and PyLandStats configuration metrics (patch density, edge density) derived from JAXA VLUCD | 23 stations, 1990 to 2020 |
| `analysis/wrtds_egret/wrtds_*.csv` | Precomputed Weighted Regressions on Time, Discharge, and Season (WRTDS) flow-normalized concentration and flux decomposition tables and bootstrap confidence intervals | 4 MRC mainstream stations, 1985 to 2022 |
| `analysis/analysis_outputs/*.csv` | Complete statistical outputs (variance partitioning, LOSO/LOYO skill, out-of-fold TreeSHAP, freshwater subsample validation, buffer sensitivity across 1 to 30 km, salinity mixing model, lag effects) | Full study period |
| `data/Fertilizers/FAO/Vietnam_FAOSTAT_data.csv` | FAOSTAT national agricultural use and cropland application rates for nitrogen ($\text{N}$), phosphate ($\text{P}_2\text{O}_5$), and potash ($\text{K}_2\text{O}$) | Viet Nam, 1961 to 2023 |
| `data/Fertilizers/GSO_Vietnam/*.csv` | General Statistics Office of Viet Nam provincial paddy planted area and agricultural indicators for the Mekong River Delta and Ben Tre province | Viet Nam provinces, 1995 to 2024 |
| `data/Fertilizers/Worldbank/Vietnam_Fertilizer_consumption.csv` | World Bank national fertilizer consumption per hectare of arable land | Viet Nam, 1961 to 2022 |

## External public data portals and restricted Zenodo archive

Raw monitoring records from provincial and national agencies are governed by institutional data-sharing policies and are archived in a restricted-access Zenodo repository for non-commercial research use (`https://doi.org/10.5281/zenodo.20024194`). When obtained from Zenodo, place the raw files at the paths listed below to run the raw monthly aggregation scripts (`03_08_cq_regime_shift.py` and `03_11_mass_balance.py`):

| Dataset | Expected local path | Source and access |
|---|---|---|
| MRC mainstream and Ben Tre DONRE water quality records | `data/mekong_delta_water_quality_1985_2023.csv` | MRC Data Portal (https://portal.mrcmekong.org/) for mainstream stations; Ben Tre Department of Natural Resources and Environment (DONRE) via restricted Zenodo archive (https://doi.org/10.5281/zenodo.20024194) |
| Daily discharge and hourly water-level records (MRC & SIWRP) | `data/hydrology/outputs/final_datasets/Mekong_Delta_Gold_Standard_Master_1985_2025.csv` | MRC Data Portal and Southern Institute for Water Resources Planning (SIWRP) via restricted Zenodo archive (https://doi.org/10.5281/zenodo.20024194) |
| Viet Nam Land Use / Land Cover Dataset (VLUCD v21.09, 30 m) | `data/lulc_raw/` | JAXA Earth Observation Research Center (https://www.eorc.jaxa.jp/ALOS/en/dataset/lulc/lulc_vnm_v2109_e.htm) |
| Gridded population density (GlobPOP 30 arc-second) | `data/population_raw/` | Zenodo open repository (https://doi.org/10.5281/zenodo.11179644) |
| Precipitation (ERA5-Land and CHIRPS v3.0) | `data/climate_raw/` | Copernicus Climate Data Store (https://cds.climate.copernicus.eu) and UCSB Climate Hazards Center (https://www.chc.ucsb.edu/data/chirps3) |
