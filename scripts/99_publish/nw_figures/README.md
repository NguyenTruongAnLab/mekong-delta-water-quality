# Nature Water Figure Pipeline

## Overview

Self-contained pipeline that regenerates all 14 publication figures
(6 main text + 8 extended data) from source data and pre-computed
analysis CSVs.  Every figure exports a companion **metrics CSV** with
the exact numerical values plotted, enabling automated manuscript
verification without visually inspecting figures.

## Quick Start

```bash
# Generate all figures + run audit
python scripts/99_publish/nw_figures/nw_run_all.py

# Or run individual figures
python scripts/99_publish/nw_figures/nw_fig1_regime_shift.py
python scripts/99_publish/nw_figures/nw_fig5_attribution.py
```

## File Structure

| File | Content | Lines |
|------|---------|-------|
| `_nw_common.py` | Shared paths, style, palettes, data loaders | ~230 |
| `nw_fig1_regime_shift.py` | Figure 1: Spatiotemporal regime shift | ~110 |
| `nw_fig2_reactor.py` | Figure 2: Biogeochemical reactor | ~150 |
| `nw_fig3_decoupling.py` | Figure 3: Hydrodynamic decoupling | ~160 |
| `nw_fig4_landscape.py` | Figure 4: Landscape trajectories | ~120 |
| `nw_fig5_attribution.py` | Figure 5: Driver attribution | ~190 |
| `nw_fig6_illusion.py` | Figure 6: Attribution illusion | ~180 |
| `nw_ed_figs_1_3_validation.py` | ED Figs 1-3: Validation & trends | ~200 |
| `nw_ed_figs_4_6_hydrodynamics.py` | ED Figs 4-6: Hydrodynamics | ~200 |
| `nw_ed_figs_7_8_lulc_attribution.py` | ED Figs 7-8: LULC & attribution | ~180 |
| `nw_audit_manuscript.py` | Automated manuscript claim verification | ~220 |
| `nw_run_all.py` | Entry point for complete pipeline | ~100 |

## Output

### Figures (PDF + PNG at 600 DPI)

Saved to `manuscript/figures/nw_composites/`:

| # | Figure | Panels |
|---|--------|--------|
| 1 | Regime Shift | (a) map, (b-e) COD/DO/pH/NH₄ breakpoints |
| 2 | Biogeochemical Reactor | (a) TN/TP gradient, (b) mass balance |
| 3 | Hydrodynamic Decoupling | (a) CI bars, (b-e) C-Q scatter |
| 4 | Landscape Trajectories | (a) stacked LULC, (b) key trajectories |
| 5 | Driver Attribution | (a) variance decomp, (b) LOSO R², (c) SHAP |
| 6 | Attribution Illusion | (a) naïve scatters, (b) FE coefficients |
| ED1 | Network cross-validation | (a-d) MRC vs BenTre scatter |
| ED2 | Buffer scale sensitivity | (a-c) ρ vs radius |
| ED3 | Extended 8-param trends | (a-h) all WQ time series |
| ED4 | Mixing diagnostics | (a) excess, (b-c) C vs S |
| ED5 | HFI (TR/Q) | (a) time series, (b) decomposed |
| ED6 | Tidal range | (a) annual, (b) dry-season |
| ED7 | LULC clustering | (a-d) cluster trajectories |
| ED8 | SHAP comparison | (a) LULC vs HC, (b) R² gap |
| ED9 | Monthly hysteresis decoupling | 2×2 loops (COD/DO/NH4/NO3; Phase 1 vs Phase 2) |

### Companion Metrics CSVs

Saved to `scripts/99_publish/nw_figures/metrics/` — one per figure,
containing the exact numerical values used in each panel.

### Audit Report

`metrics/audit_results.csv` lists PASS/FAIL for each specific claim
made in the manuscript text, cross-checked against computed values.

## Design Principles

1. **No hardcoded values** — everything is read from data or CSVs
2. **Reproducible** — deterministic outputs from source data
3. **Auditable** — companion CSVs enable number verification
4. **Nature Water style** — 180 mm, Arial, 7–10 pt, colorblind-safe
5. **Modular** — each script is self-contained and under 200 lines
