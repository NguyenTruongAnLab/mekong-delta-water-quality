# Hydrodynamic shift controls organic pollution in the Mekong Delta

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20024194.svg)](https://doi.org/10.5281/zenodo.20024194)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)

This repository contains the data-processing, statistical-modeling, flow-normalization, table-generation, and verification code accompanying the article:

> Nguyen, A. T., Némery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T.-M.-T., Nguyen-Dam, Q.-H. & Garnier, J. Hydrodynamic shift controls organic pollution in the Mekong Delta. *Nature Water* (2026).

## Study overview

Using a 38-year observational synthesis (1985 to 2022) across the Vietnamese Mekong Delta that integrates four long-term Mekong River Commission (MRC) mainstream boundary stations (1985 to 2022) and 19 provincial estuarine monitoring stations across Ben Tre province (2005 to 2022), paired with annual 30 m satellite land-use trajectories and multi-decadal discharge and tidal records, this repository reproduces three primary findings:

1. Around 2004 to 2005, the Mekong Delta shifted from a transport-dominated flushing conduit to a retention-dominated biogeochemical reactor. At the Tan Chau upstream boundary gauge, the chemical oxygen demand (COD) chemostatic index ($\text{CI} = \text{CV}_C / \text{CV}_Q$) declined from 0.84 to 0.49, driven by a 45.8% reduction in concentration variability ($\text{CV}_C$) versus an 8.0% reduction in discharge variability ($\text{CV}_Q$). Weighted Regressions on Time, Discharge, and Season (WRTDS) flow-normalization confirms that changes in the concentration-discharge relationship account for 98.8% to 105.6% ($\ge 99\%$) of the COD increase across all four MRC mainstream stations. Although upstream reservoir releases increased dry-season discharge by 23% to 67%, sand-mining riverbed incision (2 to 3 m) and dry-season tidal range amplification (+27% at My Thuan) expanded lower-estuary cross-sectional area (+43%) faster than baseflow increased, reducing mean dry-season advective velocity by 13% and doubling estuarine mixing-zone residence times to 4.5 to 7.6 days (and exceeding 20 days behind closed salinity-control sluices).
2. Along the 120 km Tan Chau to My Thuan mainstream reach, discharge-normalized mass balance shows that net $\text{NO}_3^-$ generation doubled (+24.3% to +46.1%), net COD consumption weakened from -10.9% removal toward near-neutral transport (-1.4%, Theil-Sen trend +0.49% yr$^{-1}$, $p = 0.011$), and total nitrogen shifted from 0.0% to +16.3% net generation ($p = 0.028$). In the downstream Ben Tre distributaries, conservative salinity mixing models show that non-conservative COD residuals shifted from net removal (-1.60 mg L$^{-1}$ in 2005 to 2012) to net internal generation (+4.45 mg L$^{-1}$ in 2013 to 2022).
3. Across the 19-station Ben Tre network, two-way variance decomposition ($N = 259$ station-years) shows that COD variability is dominated by shared interannual forcing (58% temporal versus 11% spatial). In a two-way station and year fixed-effects panel regression ($N = 229$, $R^2 = 0.649$), adding all 16 km buffer land-use and land-cover (LULC) predictors adds $\Delta R^2 = +0.005$ (nested $F = 0.46$, $p = 0.841$). In leave-one-station-out (LOSO) cross-validation, LULC-only models fail to transfer across stations (Ridge LOSO $R^2 = 0.07$), whereas adding regional hydroclimatic covariates increases cross-station skill to $R^2 = 0.23$ (Ridge) and $R^2 = 0.423$ (Random Forest). Out-of-fold TreeSHAP attribution identifies dry-season tidal range ($\overline{|\text{SHAP}|} = 1.32$ mg L$^{-1}$) and annual My Thuan discharge ($0.81$ mg L$^{-1}$) as the top predictors of COD across both the full estuarine panel and the strictly freshwater subsample (salinity $< 1$, $N = 117$, $\overline{|\text{SHAP}|} = 1.17$ mg L$^{-1}$).

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

For the Weighted Regressions on Time, Discharge, and Season (WRTDS) flow-normalization script (`analysis/wrtds_egret/run_wrtds.R`), R (v4.2+) with the `EGRET` package is used; precomputed WRTDS summary tables are provided in `analysis/wrtds_egret/outputs/`.

## Reproducing the analysis and manuscript verification

All panel regressions, Leave-One-Station-Out (LOSO) and Leave-One-Year-Out (LOYO) machine-learning cross-validations, TreeSHAP feature attributions, fertilizer trajectory analyses, Supplementary Tables S1 to S5, and the manuscript statistical verification suites run directly out of the box using the harmonized station-year master table (`analysis/cache/station_year_master_latest.csv`) and derived outputs included in this repository:

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

# 5. Regenerate all Supplementary Tables (Tables S1 to S5) and sample-size ledgers
python scripts/99_publish/nw_tables/nw_run_all.py
```

Scripts that aggregate raw monthly monitoring observations and daily discharge records (`scripts/03_results/03_08_cq_regime_shift.py` and `scripts/03_results/03_11_mass_balance.py`) require the restricted raw monitoring files archived on Zenodo (https://doi.org/10.5281/zenodo.20024194); their complete output tables and Monte Carlo sensitivity metrics are included in `manuscript/tables/` and `scripts/99_publish/nw_verification/metrics/`.

## Data availability

Public datasets, derived panel tables included in this repository, and access instructions for the restricted Zenodo archive (`10.5281/zenodo.20024194`) are documented in [`data/README.md`](data/README.md).

## License

Code in this repository is released under the [MIT License](LICENSE).
