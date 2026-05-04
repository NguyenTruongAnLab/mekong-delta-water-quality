# Data directory

This folder is intentionally empty in the public GitHub repository.

## What goes here

The complete dataset required to reproduce all analyses includes:

```
data/
├── mekong_delta_water_quality_1985_2023.csv   # MRC (1985–2022) + Ben Tre DONRE (2005–2022)
├── hydrology/
│   └── outputs/
│       └── final_datasets/
│           ├── Mekong_Delta_Gold_Standard_Master_1985_2025.csv   # SIWRP daily discharge
│           └── water_level_unified_subset_1985_2022.csv          # SIWRP hourly water levels
├── Fertilizers/
│   ├── FAO/Vietnam_FAOSTAT_data.csv         # FAOSTAT nitrogen fertiliser
│   ├── Worldbank/Vietnam_Fertilizer_consumption.csv
│   └── GSO_Vietnam/Planted area of paddy by province.csv
└── Population/
    └── outputs/reconstruction/
        └── vietnam_adm1_population_1985_2025_mekong_delta.gpkg
```

## How to obtain restricted data

Ben Tre provincial monitoring data (DONRE) and SIWRP discharge/water-level data are
subject to third-party organisational access policies and are **not** included in
this public repository.

To request access:
1. Visit the restricted Zenodo record: **https://doi.org/10.5281/zenodo.20024194**
2. Fill out the access request form stating your non-commercial research purpose
3. After approval, place the files in this `data/` folder following the structure above

## Public data sources

The following data can be freely downloaded from their original sources:

| Dataset | Source |
|---------|--------|
| MRC water quality (1985–2022) | https://portal.mrcmekong.org/ |
| JAXA VLUCD land-use/land-cover (1990–2020) | https://www.eorc.jaxa.jp/ALOS/en/dataset/lulc/lulc_vnm_v2109_e.htm |
| GlobPOP population density | https://doi.org/10.5281/zenodo.11179644 |
| ERA5-Land precipitation | https://cds.climate.copernicus.eu |
| CHIRPS v3.0 precipitation | https://www.chc.ucsb.edu/data/chirps3 |
| FAOSTAT fertiliser | https://www.fao.org/faostat/ |
| GSO paddy statistics | https://www.gso.gov.vn |

## Using your own data

To apply this workflow to **your own river system**, replace the files above with
your datasets following the same schema. The analysis scripts are designed to be
transferable — the same statistical framework (trend detection, changepoint
analysis, C–Q scaling, WRTDS flow-normalisation, fixed-effects regression,
SHAP attribution, variance decomposition, mass balance) can be applied to any
river-delta system with comparable monitoring data.
