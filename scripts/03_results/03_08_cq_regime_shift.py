#!/usr/bin/env python3
"""
Concentration-discharge (C-Q) power-law scaling analysis
========================================================

Computes concentration-discharge (C-Q) relationships at Mekong River Commission
(MRC) mainstream stations, evaluating structural shifts across the 2004
hydrodynamic transition.

Analyses:
1. Power-law C-Q scaling: log10(C) = alpha + beta * log10(Q), pre- and post-2004
2. Chemostatic index: CI = CV_C / CV_Q for each phase
3. Summary statistics table

Data:
- Water quality: MRC mainstream stations monthly observations (1985-2022)
- Discharge: Tan Chau gauge and system-level Total_Inflow

Output:
- manuscript/tables/Table_NW_CQ_Scaling.csv
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
TBL_DIR = ROOT / "manuscript" / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

BREAK_YEAR = 2004
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
PARAMS = [
    ("COD (mg/L)", "COD"),
    ("DO (mg/L)", "DO"),
    ("NO3-N (mg/L)", "NO₃⁻"),
    ("NH4-N (mg/L)", "NH₄⁺"),
    ("TSS (mg/L)", "TSS"),
]

STATION_Q_MAP = {
    "Tan Chau": "TanChau",
    "Can Tho": "Total_Inflow",
    "My Thuan": "Total_Inflow",
    "My Tho": "Total_Inflow",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare water quality and hydrology data."""
    if not WQ_PATH.exists() or not HYDRO_PATH.exists():
        raise FileNotFoundError(
            "Monthly C-Q scaling from raw records requires "
            f"{WQ_PATH.name} and {HYDRO_PATH.name} (archived on Zenodo: "
            "https://doi.org/10.5281/zenodo.20024194). Precomputed results are "
            "available in manuscript/tables/Table_NW_CQ_Scaling.csv."
        )
    wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
    wq = wq[wq["Data source"] == "MRC"].copy()
    wq["year"] = wq["datetime"].dt.year
    wq["month"] = wq["datetime"].dt.month
    wq["ym"] = wq["datetime"].dt.to_period("M")
    wq = wq[wq["year"] <= 2022]

    hydro = pd.read_csv(HYDRO_PATH)
    hydro["date"] = pd.to_datetime(hydro["Timestamp"], format="mixed", dayfirst=False)
    hydro["year"] = hydro["date"].dt.year
    hydro["month"] = hydro["date"].dt.month
    hydro["ym"] = hydro["date"].dt.to_period("M")
    hydro = hydro[hydro["year"] <= 2022]

    q_monthly = (
        hydro.groupby("ym")
        .agg(
            TanChau=("TanChau", "mean"),
            Total_Inflow=("Total_Inflow", "mean"),
        )
        .reset_index()
    )

    return wq, q_monthly


def compute_cq_metrics(c: np.ndarray, q: np.ndarray, label: str) -> dict:
    """Compute C-Q scaling metrics for a paired C, Q array."""
    mask = np.isfinite(c) & np.isfinite(q) & (c > 0) & (q > 0)
    c, q = c[mask], q[mask]
    n = len(c)
    if n < 10:
        return {
            "label": label,
            "n": n,
            "beta": np.nan,
            "beta_p": np.nan,
            "r": np.nan,
            "CV_C": np.nan,
            "CV_Q": np.nan,
            "CI": np.nan,
        }

    logc, logq = np.log10(c), np.log10(q)
    slope, intercept, r, p, se = stats.linregress(logq, logc)

    cv_c = np.std(c) / np.mean(c)
    cv_q = np.std(q) / np.mean(q)
    ci = cv_c / cv_q if cv_q > 0 else np.nan

    return {
        "label": label,
        "n": n,
        "beta": slope,
        "beta_se": se,
        "beta_p": p,
        "r": r,
        "r2": r**2,
        "CV_C": cv_c,
        "CV_Q": cv_q,
        "CI": ci,
        "intercept": intercept,
        "mean_C": np.mean(c),
        "mean_Q": np.mean(q),
    }


def main() -> None:
    """Run the C-Q scaling analysis."""
    wq, q_monthly = load_data()
    results = []

    for stn in MRC_STATIONS:
        q_col = STATION_Q_MAP[stn]
        stn_wq = wq[wq["stationName"] == stn].copy()
        merged = stn_wq.merge(q_monthly, on="ym", how="inner")

        for param_col, param_label in PARAMS:
            if param_col not in merged.columns:
                continue

            c_all = merged[param_col].values
            q_all = merged[q_col].values
            years = merged["year"].values

            mask1 = years <= BREAK_YEAR
            r1 = compute_cq_metrics(
                c_all[mask1],
                q_all[mask1],
                f"{stn} | {param_label} | Phase 1 (<={BREAK_YEAR})",
            )
            r1["station"] = stn
            r1["parameter"] = param_label
            r1["phase"] = "Phase 1"
            results.append(r1)

            mask2 = years > BREAK_YEAR
            r2 = compute_cq_metrics(
                c_all[mask2],
                q_all[mask2],
                f"{stn} | {param_label} | Phase 2 (>{BREAK_YEAR})",
            )
            r2["station"] = stn
            r2["parameter"] = param_label
            r2["phase"] = "Phase 2"
            results.append(r2)

    df_results = pd.DataFrame(results)

    print("=" * 80)
    print("C-Q POWER-LAW SCALING ANALYSIS")
    print("=" * 80)

    for _, param_label in PARAMS:
        print(f"\n{'-' * 60}")
        print(f"  {param_label}")
        print(f"{'-' * 60}")
        sub = df_results[df_results["parameter"] == param_label]
        for _, row in sub.iterrows():
            sig = (
                "***"
                if row["beta_p"] < 0.001
                else "**"
                if row["beta_p"] < 0.01
                else "*"
                if row["beta_p"] < 0.05
                else "ns"
            )
            beta_str = (
                f"beta={row['beta']:+.3f}+/-{row.get('beta_se', 0):.3f}"
                if np.isfinite(row["beta"])
                else "beta=N/A"
            )
            ci_str = f"CI={row['CI']:.2f}" if np.isfinite(row["CI"]) else "CI=N/A"
            print(f"  {row['label']}")
            print(f"    n={row['n']}, {beta_str} ({sig}), r={row['r']:.3f}, {ci_str}")

    cols_out = [
        "station",
        "parameter",
        "phase",
        "n",
        "beta",
        "beta_se",
        "beta_p",
        "r",
        "r2",
        "CV_C",
        "CV_Q",
        "CI",
    ]
    df_out = df_results[cols_out].copy()
    df_out.to_csv(
        TBL_DIR / "Table_NW_CQ_Scaling.csv", index=False, float_format="%.4f"
    )
    print(f"\nSaved {TBL_DIR / 'Table_NW_CQ_Scaling.csv'}")


if __name__ == "__main__":
    main()
