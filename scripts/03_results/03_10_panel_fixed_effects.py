#!/usr/bin/env python3
"""
Panel Fixed Effects Model for Nature Water
==========================================

Implements two-way fixed effects (Station FE + Year FE) panel regression
to isolate within-station LULC change effects on WQ, absorbing all shared
temporal forcing (hydro-climate) via year dummies.

Model: C_{i,t} = μ_i + λ_t + θ' X_{i,t} + ε_{i,t}
  - μ_i: station fixed effects (removes time-invariant station differences)
  - λ_t: year fixed effects (absorbs ALL shared temporal forcing)
  - θ:   within-station LULC change effects
  - X:   LULC metrics (composition + configuration)

Using statsmodels OLS with station + year dummies (equivalent to PanelOLS).

Output:
  - manuscript/tables/Table_NW_Panel_FE_Results.csv
  - Console report with coefficient estimates, SEs, significance
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
TBL_DIR = ROOT / "manuscript" / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

# LULC predictors (annual aggregates from station_year_master)
LULC_COMP = [
    "Residential_Urban_composition_pct",
    "Aquaculture_composition_pct",
    "Rice_Cultivation_composition_pct",
]
LULC_CONFIG = [
    "Residential_Urban_config_PD",
    "Aquaculture_config_ED",
    "Rice_Cultivation_config_PD",
]
POP = ["pop_density"]

# WQ response variables
WQ_PARAMS = {
    "COD (mg/L)_annual": "COD",
    "DO (mg/L)_annual": "DO",
    "NH4-N (mg/L)_annual": "NH₄⁺",
    "NO3-N (mg/L)_annual": "NO₃⁻",
    "TSS (mg/L)_annual": "TSS",
}

MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]


def main():
    master = pd.read_csv(MASTER_PATH)
    # Ben Tre only (spatial attribution network)
    df = master[~master["stationName"].isin(MRC_STATIONS)].copy()

    all_predictors = LULC_COMP + LULC_CONFIG + POP
    all_results = []

    print("=" * 80)
    print("TWO-WAY FIXED EFFECTS PANEL MODEL")
    print("Station FE + Year FE: isolating within-station LULC change effects")
    print("=" * 80)

    for wq_col, wq_label in WQ_PARAMS.items():
        print(f"\n{'━' * 60}")
        print(f"  Response: {wq_label}")
        print(f"{'━' * 60}")

        # Available predictors for this response
        avail = [p for p in all_predictors if p in df.columns]
        cols_needed = ["stationName", "year", wq_col] + avail
        sub = df[cols_needed].dropna().copy()

        if len(sub) < 30:
            print(f"  Skipping: only {len(sub)} complete cases")
            continue

        n_stations = sub["stationName"].nunique()
        n_years = sub["year"].nunique()
        print(f"  N = {len(sub)} station-years, {n_stations} stations, {n_years} years")

        # ── Model 1: Station FE + Year FE + LULC ──────────────────
        # Create dummies
        stn_dummies = pd.get_dummies(sub["stationName"], prefix="stn", drop_first=True, dtype=float)
        year_dummies = pd.get_dummies(sub["year"], prefix="yr", drop_first=True, dtype=float)

        # Standardize LULC predictors for interpretability
        X_lulc = sub[avail].copy()
        means = X_lulc.mean()
        stds = X_lulc.std()
        stds = stds.replace(0, 1)  # avoid division by zero
        X_lulc_std = (X_lulc - means) / stds

        X = pd.concat([X_lulc_std, stn_dummies, year_dummies], axis=1)
        X = sm.add_constant(X)
        y = sub[wq_col].values

        model = sm.OLS(y, X).fit(cov_type="HC1")  # heteroskedasticity-robust SE

        print(f"\n  Model R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}")
        print(f"  F-stat = {model.fvalue:.2f}, p = {model.f_pvalue:.4e}")

        # Extract LULC coefficients only
        print(f"\n  {'Predictor':<40} {'Coef':>8} {'SE':>8} {'t':>8} {'p':>8} {'Sig':>5}")
        print(f"  {'-'*80}")

        for predictor in avail:
            if predictor in model.params.index:
                coef = model.params[predictor]
                se = model.bse[predictor]
                t = model.tvalues[predictor]
                p = model.pvalues[predictor]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""

                # Short label
                short = predictor.replace("_composition_pct", " %").replace("_config_", " ").replace("Residential_Urban", "Urban").replace("Rice_Cultivation", "Rice").replace("Aquaculture", "Aqua")

                print(f"  {short:<40} {coef:>+8.3f} {se:>8.3f} {t:>8.2f} {p:>8.4f} {sig:>5}")

                all_results.append({
                    "wq_parameter": wq_label,
                    "predictor": predictor,
                    "predictor_short": short,
                    "coef_std": coef,
                    "se": se,
                    "t": t,
                    "p": p,
                    "sig": sig,
                    "model_r2": model.rsquared,
                    "model_adj_r2": model.rsquared_adj,
                    "n": len(sub),
                    "n_stations": n_stations,
                    "n_years": n_years,
                })

        # -- Model 2: Station FE only (no year FE) comparison ----
        X_noyr = pd.concat([X_lulc_std, stn_dummies], axis=1)
        X_noyr = sm.add_constant(X_noyr)
        model_noyr = sm.OLS(y, X_noyr).fit(cov_type="HC1")

        # -- Model 3: Year FE only (no station FE) comparison ----
        X_nostn = pd.concat([X_lulc_std, year_dummies], axis=1)
        X_nostn = sm.add_constant(X_nostn)
        model_nostn = sm.OLS(y, X_nostn).fit(cov_type="HC1")

        # -- Model 4: LULC only (no FE) naive model --------------
        X_naive = sm.add_constant(X_lulc_std)
        model_naive = sm.OLS(y, X_naive).fit(cov_type="HC1")

        print(f"\n  Model comparison:")
        print(f"    LULC only (naive):        R² = {model_naive.rsquared:.3f}, Adj R² = {model_naive.rsquared_adj:.3f}")
        print(f"    Station FE + LULC:        R² = {model_noyr.rsquared:.3f}, Adj R² = {model_noyr.rsquared_adj:.3f}")
        print(f"    Year FE + LULC:           R² = {model_nostn.rsquared:.3f}, Adj R² = {model_nostn.rsquared_adj:.3f}")
        print(f"    Station FE + Year FE + LULC: R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}")

        # Key insight: how much does adding LULC improve over FE-only?
        X_fe_only = pd.concat([stn_dummies, year_dummies], axis=1)
        X_fe_only = sm.add_constant(X_fe_only)
        model_fe_only = sm.OLS(y, X_fe_only).fit()

        delta_r2 = model.rsquared - model_fe_only.rsquared
        print(f"    FE-only (no LULC):        R² = {model_fe_only.rsquared:.3f}")
        print(f"    ΔR² from adding LULC:     {delta_r2:+.3f}")

    # Save results
    if all_results:
        df_out = pd.DataFrame(all_results)
        out_path = TBL_DIR / "Table_NW_Panel_FE_Results.csv"
        df_out.to_csv(out_path, index=False, float_format="%.4f")
        print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
