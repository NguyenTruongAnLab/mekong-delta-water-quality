# River regulation and tidal amplification drive organic pollution across the Mekong Delta

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20024193.svg)](https://doi.org/10.5281/zenodo.20024193)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)

This repository contains the data-processing, statistical-modeling, flow-normalization, figure-generation, and verification code for:

> Nguyen, A. T., Némery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T.-M.-T., Nguyen-Dam, Q.-H. & Garnier, J. River regulation and tidal amplification drive organic pollution across the Mekong Delta (submitted to *Communications Earth & Environment*, 2026).

## Overview

The study integrates 38 years of monthly water-quality observations at four Mekong River Commission (MRC) mainstem stations (1985--2022) with 18 years of coastal distributary monitoring across 19 stations in Ben Tre province (2005--2022). These records are paired with daily discharge series, hourly tidal records, and annual 30 m satellite land-use and land-cover (LULC) metrics extracted across 1 to 30 km station buffers.

The scripts and derived station-year caches in this repository reproduce:

- Seasonal Mann--Kendall trend tests, Livezey--Chen field-significance permutations, and Garcia--Gordon apparent oxygen utilization (AOU) solubility partitioning.
- Concentration--discharge (*C*--*Q*) power-law scaling, Weighted Regressions on Time, Discharge, and Season (WRTDS) flow normalization, and Tan Chau--My Thuan reach mass-balance budgets (including 10,000-draw Vam Nao flow-split Monte Carlo tests).
- Estuarine conservative-mixing budgets across all salinities, mercuric-sulfate-masked salinities (*S* <= 3.6), and five-day biochemical oxygen demand (BOD5).
- Two-way variance partitioning, Leave-One-Station-Out (LOSO) and Leave-One-Year-Out (LOYO) machine-learning cross-validation, out-of-fold TreeSHAP rankings, isometric log-ratio (ILR) compositional transformations, and multi-buffer (2--16 km) panel fixed-effects models across four covariance estimators (HC3, station-clustered, year-clustered, and two-way clustered, plus Driscoll--Kraay on the 38-year mainstem panel).

## Repository structure

```text
.
├── analysis/
│   ├── analyzers/                      # Spatial gradient, lag-effect, and model-comparison modules
│   ├── generators/                     # Harmonized station-year panel and CV split generators
│   ├── lib/                            # Data ingestion, spatial buffer, and statistical utilities
│   ├── validation/                     # Buffer-scale sensitivity (1-30 km), VIF, and robustness checks
│   ├── wrtds_egret/                    # R script (run_wrtds.R) and WRTDS flow-normalization outputs
│   ├── cache/
│   │   ├── station_year_master_latest.csv                    # 411 station-year x 151 variable panel
│   │   ├── bentre_station_year_bod5_halide_supplement.csv    # Station-year BOD5 and S <= 3.6 COD cache
│   │   └── lulc_metrics_corrected_{2,5,10,15,16}km.csv       # Multi-buffer 30 m satellite LULC caches
│   ├── analysis_outputs/               # Precomputed model outputs, tidal ranges, and sensitivity CSVs
│   ├── prepare_egret_data.py
│   └── run_buffer_analysis_by_network.py
├── data/
│   ├── Fertilizers/                    # Public FAOSTAT, GSO Viet Nam, and World Bank agricultural CSVs
│   └── README.md                       # Data access instructions and file inventory
├── manuscript/
│   └── tables/                         # Derived CSV and Markdown tables for main text and SI
├── scripts/
│   ├── 02_methods/
│   │   └── 02_02_generate_lulc_metrics_station_buffers.py
│   ├── 03_results/
│   │   ├── 03_05b_primary_model_augmented.py     # LOSO/LOYO RF & Ridge CV + out-of-fold TreeSHAP
│   │   ├── 03_08_cq_regime_shift.py              # C-Q power-law scaling and chemostatic index
│   │   ├── 03_10_panel_fixed_effects.py          # Station and year fixed-effects regressions
│   │   ├── 03_11_mass_balance.py                 # Reach mass balance and 10,000-draw Vam Nao Monte Carlo
│   │   └── 03_12_panel_and_cq_diagnostics.py     # CV_C/CV_Q decomposition, PELT, and FE delta-R2
│   ├── 05_appendix/
│   │   └── 05_06_fertilizer_vs_wq.py             # Fertilizer, paddy area, and nitrate correlations
│   └── 99_publish/
│       ├── cee_figures/                          # Publication figure scripts (Figs. 1-7, SFigs. 1-12) + metrics/
│       ├── cee_verification/                     # Statistical verification scripts and precomputed metrics/
│       └── nw_tables/                            # Generators for Supplementary Tables 6-10
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Installation

Create a Python 3.10+ environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Running the WRTDS flow-normalization script (`analysis/wrtds_egret/run_wrtds.R`) additionally requires R (v4.2+) with the `EGRET` package (v3.0.9). Precomputed WRTDS tables are provided in `analysis/wrtds_egret/outputs/`.

## Reproducing the statistical models from the included derived caches

All station-year panel caches (`station_year_master_latest.csv`, `bentre_station_year_bod5_halide_supplement.csv`, and `lulc_metrics_corrected_{2,5,10,15,16}km.csv`) are included in this repository. Run the commands below from the repository root without external downloads:

```bash
# 1. Variance partitioning (with BOD5), LOYO vs. LOSO CV, ILR concordance, 2-16 km FE, and tidal-range FE
python scripts/99_publish/cee_verification/cee_verify_loyo_loso_ilr_fe.py

# 2. Panel fixed-effects regressions, PELT sensitivity, and incremental R2 decomposition
python scripts/03_results/03_10_panel_fixed_effects.py
python scripts/03_results/03_12_panel_and_cq_diagnostics.py

# 3. LOSO/LOYO Random Forest and Ridge cross-validation and out-of-fold TreeSHAP attribution
python scripts/03_results/03_05b_primary_model_augmented.py

# 4. First-difference fertilizer and paddy area analysis
python scripts/05_appendix/05_06_fertilizer_vs_wq.py

# 5. Regenerate Supplementary Tables 6 to 10
python scripts/99_publish/nw_tables/nw_run_all.py
```

Scripts that aggregate observation-level monthly surveys, daily discharge series, or hourly water levels (`cee_verify_oxygen_and_trends.py`, `cee_verify_halide_and_bod5_mixing.py`, `03_08_cq_regime_shift.py`, `03_11_mass_balance.py`, and `cee_figures/cee_run_all.py`) require the restricted raw monitoring files archived on Zenodo (https://doi.org/10.5281/zenodo.20024193). Their verified output tables and figure source metrics are included in `scripts/99_publish/cee_verification/metrics/`, `scripts/99_publish/cee_figures/metrics/`, and `manuscript/tables/`.

## Data availability

Public datasets, derived panel caches, and access instructions for the Zenodo archive (`10.5281/zenodo.20024193`, version record `10.5281/zenodo.20024194`) are documented in [`data/README.md`](data/README.md).

## License

Code in this repository is released under the [MIT License](LICENSE).
