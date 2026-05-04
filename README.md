# Mekong Delta Water Quality — Analysis Code & Derived Outputs

This is the **public code repository** accompanying:

> Nguyen, A. T., Nemery, J., Tran, K. T., Tran, T. T., Truong, T. V., Le, T. M. T., Nguyen-Dam, Q. H., & Garnier, J.: **Weakened hydrodynamic flushing decouples organic pollution from discharge in the Mekong Delta**, (in prep.).

---

## What this repository contains

This repository provides **all Python analysis scripts, pre-computed derived outputs, and publication figures** needed to reproduce the results and to apply the same analytical workflow to other river-delta systems. It does **not** include the restricted raw monitoring data (see [Data access](#data-access) below).

```
├── scripts/              Python figure & table generation scripts
│   └── 99_publish/
│       ├── nw_figures/   Main-text & Extended Data figure scripts
│       └── nw_tables/    Supplementary Table generation scripts
├── analysis/             Pre-computed derived outputs
│   ├── cache/            Station-year master table, LULC metrics
│   ├── analysis_outputs/ Fixed-effects, SHAP, variance decomposition,
│   │                     changepoint stats, mass balance, WRTDS
│   └── wrtds_egret/      WRTDS flow-normalisation results
├── manuscript/
│   ├── figures/          Publication figures (PDF + PNG, 600 DPI)
│   └── tables/           Supplementary tables (CSV)
├── data/                 [EMPTY — see data/README.md]
├── requirements.txt      Python dependencies
└── README.md
```

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/nguyentruonganlab/mekong-delta-water-quality.git
cd mekong-delta-water-quality

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Check data availability
python scripts/99_publish/nw_figures/_nw_common.py

# 4. Generate a figure that uses only public derived data
python scripts/99_publish/nw_figures/nw_fig4_landscape.py

# 5. Generate all figures that can run without restricted data
python scripts/99_publish/nw_figures/nw_run_all.py
```

---

## Which scripts can run out of the box?

The following scripts use **only pre-computed derived outputs** included in this repository and will run immediately:

### Fully runnable (no restricted data needed)

| Script | Output | Data source |
|--------|--------|-------------|
| `nw_fig4_landscape.py` | Figure 4 — Landscape Trajectories | `analysis/cache/` |
| `nw_fig5_attribution.py` | Figure 5 — Driver Attribution | `analysis/analysis_outputs/` |
| `nw_fig6_illusion.py` | Figure 6 — Attribution Illusion | `analysis/analysis_outputs/` |
| `nw_conceptual_mechanism.py` | Conceptual Mechanism figure | (pure drawing) |
| `nw_discussion_concept.py` | Discussion Concept figure | `analysis/analysis_outputs/` |
| `nw_graphical_abstract.py` | Graphical Abstract | (pure drawing) |
| `nw_sfig7_fertilizer_gradient.py` | Fig S7 — Fertilizer Gradient | `data/Fertilizers/` (FAOSTAT — public) |
| `nw_sfig7b_colocated_diagnostic.py` | Fig S7b — Co-located Diagnostic | `data/Fertilizers/` + `analysis/cache/` |
| `nw_sfig8_wrtds_flow_normalization.py` | Fig S8 — WRTDS Trends | `analysis/wrtds_egret/` |

### Require restricted data (will print clear guidance)

| Script | Missing data | Source |
|--------|-------------|--------|
| `nw_fig1_regime_shift.py` | WQ CSV (Ben Tre DONRE) | Restricted Zenodo |
| `nw_fig2_reactor.py` | WQ CSV (Ben Tre DONRE) | Restricted Zenodo |
| `nw_fig3_decoupling.py` | WQ + Hydrology (SIWRP) | Restricted Zenodo |
| `nw_ed_figs_1_3_validation.py` | WQ CSV | Restricted Zenodo |
| `nw_ed_figs_2_4_hydrodynamics.py` | WQ CSV | Restricted Zenodo |
| `nw_ed_figs_5_6_lulc_attribution.py` | WQ CSV | Restricted Zenodo |
| `nw_ed_fig7_hysteresis_decoupling.py` | WQ + Hydrology | Restricted Zenodo |
| `nw_ed_fig8_reactor_all_parameters.py` | WQ CSV | Restricted Zenodo |
| `nw_table_s1–s5` | WQ CSV | Restricted Zenodo |

When you run a script that needs restricted data, it will print a clear message telling you exactly which file is missing, where to put it, and how to request access.

---

## Data access

### Restricted data (Ben Tre DONRE / SIWRP)

Ben Tre provincial monitoring data and SIWRP discharge/water-level records are subject to third-party organisational access policies. They can be provided upon request for non-commercial use from a restricted-access Zenodo repository:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20024194.svg)](https://doi.org/10.5281/zenodo.20024194)

After approval, place the data files in the `data/` folder following the structure described in [`data/README.md`](data/README.md).

MRC water-quality data are publicly available from the [Mekong River Commission Data Portal](https://portal.mrcmekong.org/).

### Public data sources

| Dataset | Source |
|---------|--------|
| MRC water quality (1985–2022) | https://portal.mrcmekong.org/ |
| JAXA VLUCD land-use (1990–2020) | https://www.eorc.jaxa.jp/ALOS/en/dataset/lulc/lulc_vnm_v2109_e.htm |
| GlobPOP population density | https://doi.org/10.5281/zenodo.11179644 |
| ERA5-Land precipitation | https://cds.climate.copernicus.eu |
| CHIRPS v3.0 precipitation | https://www.chc.ucsb.edu/data/chirps3 |
| FAOSTAT fertiliser | https://www.fao.org/faostat/ |
| GSO paddy statistics | https://www.gso.gov.vn |

---

## Using this workflow for your own river system

This repository is designed to be **transferable** — the analytical framework can be applied to any river-delta system with comparable monitoring data. To adapt it:

1. **Replace the data**: put your water-quality, discharge, water-level, and land-use data in the `data/` folder following the same schema
2. **Update station names**: edit `MRC_STATIONS` and station lists in `_nw_common.py` and individual scripts
3. **Adjust the break year**: change `BREAK_YEAR` in `_nw_common.py` to match your system's regime shift
4. **Run the pipeline**: the same statistical methods (trend detection, C–Q scaling, WRTDS, fixed-effects regression, SHAP, variance decomposition, mass balance) will work on your data

The core analytical methods are:
- **Trend detection** — Seasonal Kendall, Theil–Sen slopes
- **Changepoint analysis** — PELT algorithm with bootstrapped confidence intervals
- **C–Q scaling** — Power-law concentration–discharge relationships
- **WRTDS flow-normalisation** — USGS EGRET/WRTDSKalman with structural breakpoints
- **Fixed-effects regression** — Two-way panel models with cluster-robust standard errors
- **SHAP attribution** — Tree-based models with LOSO cross-validation
- **Variance decomposition** — Spatial vs. temporal vs. residual partitioning
- **Conservative mixing analysis** — End-member mixing with Monte Carlo uncertainty
- **Mass balance decomposition** — C–Q-based load partitioning

---

## Citation

When using materials from this repository, please cite both:

1. Nguyen, A. T. et al. (2026). Mekong Delta water quality and hydrodynamic analysis package (1985–2022). Zenodo. https://doi.org/10.5281/zenodo.20024194
2. Nguyen, A. T. et al. (2026). Weakened hydrodynamic flushing decouples organic pollution from discharge in the Mekong Delta. (in prep.).

---

## License

Analysis code, derived outputs, and figures: **CC BY-NC 4.0**

Restricted data (Ben Tre DONRE / SIWRP): governed by the data providers' organisational policies.

---

## Contact

For questions about the code or methodology, please contact the corresponding author as listed in Nguyen et al. (2026).

For data access requests, use the Zenodo restricted-access form linked above.
