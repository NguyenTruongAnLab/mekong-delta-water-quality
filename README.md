# Hydrodynamic shift controls organic pollution in the Mekong Delta

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20024193.svg)](https://doi.org/10.5281/zenodo.20024193)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)

This repository contains the data-processing, statistical-modeling, flow-normalization, table-generation, and verification code for the study:

> Nguyen, A. T., Némery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T.-M.-T., Nguyen-Dam, Q.-H. & Garnier, J. Hydrodynamic shift controls organic pollution in the Mekong Delta (manuscript submitted for publication, 2026).

## Overview

The analysis combines 38 years of mainstream observations at four Mekong River Commission stations (1985 to 2022) with 18 years of estuarine records across 19 Ben Tre stations (2005 to 2022). These water-quality records are paired with daily discharge reconstructions, tidal range records, and 30 m satellite land-use metrics across 1 to 30 km station buffers. The scripts in this repository reproduce the concentration-discharge scaling models, reach mass balances, two-way fixed-effects panel regressions, cross-validated machine-learning attributions, and manuscript verification checks.

## Repository structure

```text
.
├── analysis/
│   ├── analyzers/                      # Spatial gradient, lag effect, and model comparison modules
│   ├── generators/                     # Harmonized station-year master and CV split generators
│   ├── lib/                            # Data ingestion, spatial buffer, and statistical utilities
│   ├── validation/                     # Buffer scale sensitivity (1-30 km), VIF, and robustness modules
│   ├── wrtds_egret/                    # R script (run_wrtds.R) and WRTDS flow-normalization outputs
│   ├── cache/
│   │   ├── station_year_master_latest.csv        # Harmonized 411 station-year x 151 variable panel
│   │   ├── station_year_master_latest.schema.json
│   │   └── lulc_metrics_corrected_16km.csv       # 16 km satellite LULC composition/configuration cache
│   ├── analysis_outputs/               # Precomputed model outputs, SHAP tables, and sensitivity CSVs
│   ├── prepare_egret_data.py           # Formats daily discharge and water quality for EGRET/WRTDS
│   └── run_buffer_analysis_by_network.py
├── data/
│   ├── Fertilizers/                    # Public FAOSTAT, GSO Viet Nam, and World Bank agricultural CSVs
│   └── README.md                       # Data access instructions and file inventory
├── manuscript/
│   └── tables/                         # Generated CSV and Markdown tables for main text and SI
├── scripts/
│   ├── 02_methods/
│   │   └── 02_02_generate_lulc_metrics_station_buffers.py
│   ├── 03_results/
│   │   ├── 03_05b_primary_model_augmented.py     # LOSO/LOYO RF & Ridge CV + out-of-fold TreeSHAP
│   │   ├── 03_08_cq_regime_shift.py              # C-Q power-law scaling and chemostatic index
│   │   ├── 03_10_panel_fixed_effects.py          # Two-way station and year fixed-effects regressions
│   │   ├── 03_11_mass_balance.py                 # Reach mass balance and 10,000-draw Vam Nao Monte Carlo
│   │   └── 03_12_panel_and_cq_diagnostics.py     # CV_C/CV_Q decomposition, PELT, and FE delta-R2
│   ├── 05_appendix/
│   │   └── 05_06_fertilizer_vs_wq.py             # Fertilizer, paddy area, and water quality correlations
│   └── 99_publish/
│       ├── nw_tables/                            # Generators for Supplementary Tables S1-S5 + metrics/
│       └── nw_verification/                      # Manuscript audit (27 checks) and verification (76 checks)
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Installation

Create a Python 3.10+ environment and install the required packages:

```bash
pip install -r requirements.txt
```

To run the Weighted Regressions on Time, Discharge, and Season (WRTDS) script (`analysis/wrtds_egret/run_wrtds.R`), install R (v4.2+) and the `EGRET` package. Precomputed WRTDS summary tables are provided in `analysis/wrtds_egret/outputs/`.

## Reproducing the analysis

The harmonized station-year master table (`analysis/cache/station_year_master_latest.csv`) and derived outputs are included in this repository. Run the following commands from the repository root to execute the statistical models, regenerate the supplementary tables, and verify all manuscript claims:

```bash
# 1. Run the 27-point and 76-point manuscript verification suites
python scripts/99_publish/nw_verification/nw_audit_manuscript.py
python scripts/99_publish/nw_verification/nw_verify_manuscript.py

# 2. Run the two-way fixed-effects panel regressions and incremental R2 decomposition
python scripts/03_results/03_10_panel_fixed_effects.py
python scripts/03_results/03_12_panel_and_cq_diagnostics.py

# 3. Run the LOSO/LOYO Random Forest and Ridge cross-validation and TreeSHAP attribution
python scripts/03_results/03_05b_primary_model_augmented.py

# 4. Run the multivariate fertilizer and paddy area analysis
python scripts/05_appendix/05_06_fertilizer_vs_wq.py

# 5. Regenerate Supplementary Tables S1 to S5 (Supplementary Tables 6 to 10 in the SI)
python scripts/99_publish/nw_tables/nw_run_all.py
```

Two scripts aggregate raw monthly monitoring observations and daily discharge records (`scripts/03_results/03_08_cq_regime_shift.py` and `scripts/03_results/03_11_mass_balance.py`). These scripts require the restricted raw monitoring files archived on Zenodo (https://doi.org/10.5281/zenodo.20024193; this DOI represents all versions and always resolves to the latest dataset). Their precomputed output tables and Monte Carlo sensitivity metrics are included in `manuscript/tables/` and `scripts/99_publish/nw_verification/metrics/`.

## Data availability

Public datasets, derived panel tables, and access instructions for the latest restricted Zenodo dataset (`10.5281/zenodo.20024193`) are documented in [`data/README.md`](data/README.md).

## License

Code in this repository is released under the [MIT License](LICENSE).
