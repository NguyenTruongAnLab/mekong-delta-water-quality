#!/usr/bin/env python3
"""
Panel Fixed-Effects, Changepoint, and C-Q Sensitivity Diagnostics
=================================================================

Computes sensitivity and robustness diagnostics across both monitoring networks:
1. Chemostatic index decomposition: CV_C and CV_Q by phase across MRC stations
2. Multi-station C-Q scaling exponents (beta) and chemostatic indices
3. Cross-network nitrate (NO3-N) and COD validation at co-located stations
4. Two-way fixed-effects incremental R2 decomposition (station + year FE)
5. PELT changepoint sensitivity across penalty functions (MBIC, BIC, AIC)
6. Dry-season discharge-concentration trend comparison

Outputs:
- manuscript/tables/Table_NW_CI_Decomposition.csv
- manuscript/tables/Table_NW_NO3_CrossValidation.csv
- manuscript/tables/Table_NW_Panel_FE_DeltaR2.csv
- manuscript/tables/Table_NW_PELT_Sensitivity.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
HYDRO_PATH = (
    ROOT
    / "data"
    / "hydrology"
    / "outputs"
    / "final_datasets"
    / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
)
MASTER_PATH = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
TBL_DIR = ROOT / "manuscript" / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

BREAK_YEAR = 2004
MRC = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
PARAMS = {
    "COD (mg/L)": "COD",
    "DO (mg/L)": "DO",
    "NH4-N (mg/L)": "NH₄⁺",
    "NO3-N (mg/L)": "NO₃⁻",
    "TSS (mg/L)": "TSS",
}
STN_Q = {
    "Tan Chau": "TanChau",
    "Can Tho": "Total_Inflow",
    "My Thuan": "Total_Inflow",
    "My Tho": "Total_Inflow",
}


def load_wq_hydro() -> tuple[pd.DataFrame, pd.DataFrame]:
    wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
    wq = wq[(wq["Data source"] == "MRC") & (wq["datetime"].dt.year <= 2022)].copy()
    wq["year"] = wq["datetime"].dt.year
    wq["month"] = wq["datetime"].dt.month
    wq["ym"] = wq["datetime"].dt.to_period("M")

    hydro = pd.read_csv(HYDRO_PATH)
    hydro["date"] = pd.to_datetime(hydro["Timestamp"], format="mixed", dayfirst=False)
    hydro["year"] = hydro["date"].dt.year
    hydro["month"] = hydro["date"].dt.month
    q_m = (
        hydro.groupby(["year", "month"])
        .agg(
            TanChau=("TanChau", "mean"),
            Total_Inflow=("Total_Inflow", "mean"),
            MyThuan=("MyThuan", "mean"),
        )
        .reset_index()
    )
    return wq, q_m


def ci_decomposition(wq: pd.DataFrame, q_m: pd.DataFrame) -> pd.DataFrame:
    print("=" * 80)
    print("DIAGNOSTIC 1: CHEMOSTATIC INDEX DECOMPOSITION (CV_C vs CV_Q)")
    print("=" * 80)
    rows = []
    for stn in MRC:
        q_col = STN_Q[stn]
        sw = wq[wq["stationName"] == stn].copy()
        merged = sw.merge(q_m, on=["year", "month"], how="inner")
        for pcol, plabel in PARAMS.items():
            if pcol not in merged.columns:
                continue
            for phase, mask_fn in [
                ("Phase 1 (≤2004)", lambda df: df["year"] <= BREAK_YEAR),
                ("Phase 2 (>2004)", lambda df: df["year"] > BREAK_YEAR),
            ]:
                sub = merged[mask_fn(merged)].dropna(subset=[pcol, q_col])
                sub = sub[(sub[q_col] > 0) & (sub[pcol] > 0)]
                if len(sub) < 10:
                    continue
                c = sub[pcol].values
                q = sub[q_col].values
                cv_c = np.std(c) / np.mean(c)
                cv_q = np.std(q) / np.mean(q)
                ci = cv_c / cv_q if cv_q > 0 else np.nan
                lc, lq = np.log10(c), np.log10(q)
                valid = np.isfinite(lc) & np.isfinite(lq)
                if valid.sum() < 10:
                    continue
                slope, _, r, p, _ = stats.linregress(lq[valid], lc[valid])
                rows.append(
                    {
                        "Station": stn,
                        "Parameter": plabel,
                        "Phase": phase,
                        "N": len(sub),
                        "Mean_C": np.mean(c),
                        "Std_C": np.std(c),
                        "CV_C": cv_c,
                        "Mean_Q": np.mean(q),
                        "Std_Q": np.std(q),
                        "CV_Q": cv_q,
                        "CI": ci,
                        "beta": slope,
                        "beta_p": p,
                        "r2": r**2,
                    }
                )

    df = pd.DataFrame(rows)
    out = TBL_DIR / "Table_NW_CI_Decomposition.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved {out}")
    return df


def cross_validate_no3() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: CROSS-NETWORK VALIDATION AT CO-LOCATED STATIONS")
    print("=" * 80)
    wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
    wq = wq[wq["datetime"].dt.year <= 2022].copy()
    wq["year"] = wq["datetime"].dt.year

    pairs = [
        ("My Thuan", "MRC", "CC01", "BenTre"),
        ("My Tho", "MRC", "MT01", "BenTre"),
    ]

    rows = []
    for mrc_stn, _, bt_stn, _ in pairs:
        for pcol, plabel in [
            ("NO3-N (mg/L)", "NO₃⁻"),
            ("COD (mg/L)", "COD"),
            ("DO (mg/L)", "DO"),
            ("NH4-N (mg/L)", "NH₄⁺"),
        ]:
            mrc_data = wq[wq["stationName"] == mrc_stn]
            bt_data = wq[wq["stationName"] == bt_stn]

            mrc_annual = mrc_data.groupby("year")[pcol].median().dropna()
            bt_annual = bt_data.groupby("year")[pcol].median().dropna()

            common_years = mrc_annual.index.intersection(bt_annual.index)
            if len(common_years) < 5:
                continue

            mc = mrc_annual.loc[common_years].values
            bc = bt_annual.loc[common_years].values

            r, p = stats.pearsonr(mc, bc)
            diff_mean = np.mean(bc - mc)
            diff_std = np.std(bc - mc)

            if len(common_years) >= 8:
                mrc_slope = stats.theilslopes(mc, common_years)[0]
                bt_slope = stats.theilslopes(bc, common_years)[0]
                trend_agree = (
                    "Same" if np.sign(mrc_slope) == np.sign(bt_slope) else "OPPOSITE"
                )
            else:
                mrc_slope = bt_slope = np.nan
                trend_agree = "N/A"

            rows.append(
                {
                    "MRC Station": mrc_stn,
                    "BenTre Station": bt_stn,
                    "Parameter": plabel,
                    "N_years": len(common_years),
                    "r": r,
                    "p": p,
                    "Mean_Diff (BT-MRC)": diff_mean,
                    "Std_Diff": diff_std,
                    "MRC_trend_slope": mrc_slope,
                    "BT_trend_slope": bt_slope,
                    "Trend_Agreement": trend_agree,
                }
            )

    df = pd.DataFrame(rows)
    out = TBL_DIR / "Table_NW_NO3_CrossValidation.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved {out}")
    return df


def panel_fe_delta_r2() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: TWO-WAY PANEL FIXED-EFFECTS INCREMENTAL R2")
    print("=" * 80)
    import statsmodels.api as sm

    master = pd.read_csv(MASTER_PATH)
    df = master[~master["stationName"].isin(MRC)].copy()

    lulc = [
        "Residential_Urban_composition_pct",
        "Aquaculture_composition_pct",
        "Rice_Cultivation_composition_pct",
        "Residential_Urban_config_PD",
        "Aquaculture_config_ED",
        "Rice_Cultivation_config_PD",
        "pop_density",
    ]
    hydro = [
        "Q_branch_AnnualMean_z",
        "Q_branch_DrySeasonMean_z",
        "Rain_Annual_z",
        "Rain_Dry_z",
        "TR_CanTho_m",
        "TR_MyThuan_m",
    ]
    wq_cols = {
        "COD (mg/L)_annual": "COD",
        "DO (mg/L)_annual": "DO",
        "NH4-N (mg/L)_annual": "NH₄⁺",
        "NO3-N (mg/L)_annual": "NO₃⁻",
    }

    rows = []
    for wq_col, wq_lbl in wq_cols.items():
        avail_lulc = [c for c in lulc if c in df.columns]
        avail_hydro = [c for c in hydro if c in df.columns]
        avail_all = avail_lulc + avail_hydro

        cols = ["stationName", "year", wq_col] + avail_all
        sub = df[[c for c in cols if c in df.columns]].dropna().copy()
        if len(sub) < 30:
            continue

        y = sub[wq_col].values
        stn_d = pd.get_dummies(
            sub["stationName"], prefix="stn", drop_first=True, dtype=float
        )
        yr_d = pd.get_dummies(sub["year"], prefix="yr", drop_first=True, dtype=float)

        def fit(X_extra):
            X = (
                pd.concat([X_extra, stn_d, yr_d], axis=1)
                if X_extra is not None
                else pd.concat([stn_d, yr_d], axis=1)
            )
            X = sm.add_constant(X)
            return sm.OLS(y, X).fit()

        def std(cols_list):
            x = sub[cols_list].copy()
            return (x - x.mean()) / x.std().replace(0, 1)

        m_fe_only = fit(None)
        m_fe_lulc = fit(std(avail_lulc))
        m_fe_hydro = fit(std(avail_hydro))
        m_fe_all = fit(std(avail_all))

        r = {
            "Parameter": wq_lbl,
            "N": len(sub),
            "R2_FE_only": m_fe_only.rsquared,
            "R2_FE_LULC": m_fe_lulc.rsquared,
            "R2_FE_Hydro": m_fe_hydro.rsquared,
            "R2_FE_All": m_fe_all.rsquared,
            "dR2_LULC": m_fe_lulc.rsquared - m_fe_only.rsquared,
            "dR2_Hydro": m_fe_hydro.rsquared - m_fe_only.rsquared,
            "dR2_All": m_fe_all.rsquared - m_fe_only.rsquared,
        }
        rows.append(r)
        print(
            f"  {wq_lbl} (N={len(sub)}): R2(FE)={r['R2_FE_only']:.4f}, "
            f"R2(FE+LULC)={r['R2_FE_LULC']:.4f} (dR2={r['dR2_LULC']:+.4f})"
        )

    df_out = pd.DataFrame(rows)
    out = TBL_DIR / "Table_NW_Panel_FE_DeltaR2.csv"
    df_out.to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved {out}")
    return df_out


def _exact_rbf_pelt(signal_1d: np.ndarray, pen_val: float, min_size: int = 5) -> list[int]:
    """Exact optimal partitioning under RBF kernel cost (identical optimum to ruptures.Pelt(model='rbf'))."""
    n = len(signal_1d)
    if n < 2 * min_size:
        return [n]
    diffs = np.subtract.outer(signal_1d, signal_1d) ** 2
    tri = diffs[np.triu_indices(n, k=1)]
    pos = tri[tri > 0]
    gamma = 1.0 / float(np.median(pos)) if len(pos) > 0 else 1.0
    K = np.exp(-gamma * diffs)
    # Prefix sums of 2D kernel matrix for O(1) segment cost queries
    P = np.zeros((n + 1, n + 1), dtype=float)
    P[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def seg_cost(s: int, e: int) -> float:
        m = e - s
        ksum = P[e, e] - P[s, e] - P[e, s] + P[s, s]
        return float(m - ksum / m)

    dp = np.full(n + 1, np.inf)
    prev = np.zeros(n + 1, dtype=int)
    dp[0] = -pen_val
    for t in range(min_size, n + 1):
        for s in range(0, t - min_size + 1):
            if s > 0 and s < min_size:
                continue
            c = dp[s] + seg_cost(s, t) + pen_val
            if c < dp[t]:
                dp[t] = c
                prev[t] = s
    bkps = []
    cur = n
    while cur > 0:
        bkps.append(int(cur))
        cur = int(prev[cur])
    bkps.reverse()
    return bkps


def pelt_sensitivity() -> pd.DataFrame | None:
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: PELT CHANGEPOINT PENALTY SENSITIVITY")
    print("=" * 80)
    try:
        import ruptures as rpt
    except ImportError:
        rpt = None

    wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
    wq = wq[(wq["Data source"] == "MRC") & (wq["datetime"].dt.year <= 2022)].copy()
    wq["year"] = wq["datetime"].dt.year

    rows = []
    for stn in MRC:
        for pcol, plabel in [
            ("COD (mg/L)", "COD"),
            ("DO (mg/L)", "DO"),
            ("NO3-N (mg/L)", "NO₃⁻"),
            ("NH4-N (mg/L)", "NH₄⁺"),
        ]:
            ann = wq[wq["stationName"] == stn].groupby("year")[pcol].median().dropna()
            if len(ann) < 15:
                continue
            signal = ann.values.reshape(-1, 1)
            years = ann.index.values

            for pen_name in ["MBIC", "BIC", "AIC"]:
                if pen_name == "MBIC":
                    n = len(signal)
                    pen_val = np.log(n) * np.log(np.log(n)) * signal.var()
                elif pen_name == "BIC":
                    pen_val = np.log(len(signal)) * signal.var()
                else:
                    pen_val = 2.0 * signal.var()

                if rpt is not None:
                    algo = rpt.Pelt(model="rbf", min_size=5).fit(signal)
                    bkps = algo.predict(pen=pen_val)
                else:
                    bkps = _exact_rbf_pelt(ann.values.astype(float), float(pen_val), min_size=5)
                bp_years = [int(years[b - 1]) for b in bkps if b < len(years)]
                rows.append(
                    {
                        "Station": stn,
                        "Parameter": plabel,
                        "Penalty": pen_name,
                        "Penalty_Value": pen_val,
                        "N_breakpoints": len(bp_years),
                        "Breakpoint_Years": str(bp_years) if bp_years else "None",
                    }
                )

    df = pd.DataFrame(rows)
    out = TBL_DIR / "Table_NW_PELT_Sensitivity.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"  Saved {out}")
    return df


if __name__ == "__main__":
    panel_fe_delta_r2()
    if WQ_PATH.exists() and HYDRO_PATH.exists():
        wq_df, q_m_df = load_wq_hydro()
        ci_decomposition(wq_df, q_m_df)
        cross_validate_no3()
        pelt_sensitivity()
    else:
        print(
            "\nNote: Raw monthly water quality and hydrology files are archived in "
            "Zenodo (https://doi.org/10.5281/zenodo.20024194). Precomputed "
            "diagnostic tables are available in manuscript/tables/."
        )
