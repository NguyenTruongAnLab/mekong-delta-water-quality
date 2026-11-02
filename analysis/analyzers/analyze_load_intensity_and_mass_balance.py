#!/usr/bin/env python3
"""analyze_load_intensity_and_mass_balance.py

Diagnostic analyses supporting mechanistic interpretation of observational
land-use attribution:

1) Load intensity vs. urban fragmentation (Ben Tre branches with discharge proxies)
   - Computes annual COD loads (tons/day) using branch discharge proxies
   - Converts loads to intensities (kg/ha/day) per buffer area and per urban area
   - Summarizes intensities across Urban Patch Density (PD) quantiles

2) Simple load-dilution decomposition (selected key stations)
   - Uses annual mean concentration C and annual mean discharge Q
   - Uses identity: L = C * Q * 0.0864  (tons/day)
   - Decomposes concentration change between two endpoints into:
       dC_total = dC_load + dC_dilution + residual

Outputs (written to analysis/analysis_outputs):
- cod_load_intensity_by_urban_fragmentation_latest.csv
- cod_load_intensity_station_year_latest.csv
- cod_mass_balance_decomposition_latest.csv

Notes
-----
- These diagnostics use reconstructed/branch-level discharge estimates and are
  therefore intended as supporting evidence (not calibrated mass-balance).
- The load intensity per *urban* area becomes unstable when urban % is very
  small; a minimum urban composition threshold is enforced.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis.lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard


OUTPUT_DIR = str(ANALYSIS_DIR / "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WQ_FILE = str(REPO_ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv")
LULC_FILE = str(ANALYSIS_DIR / "cache" / "lulc_metrics_corrected_16km.csv")
Q_FILE = str(
    REPO_ROOT
    / "data"
    / "hydrology"
    / "outputs"
    / "final_datasets"
    / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
)

TARGET = "COD (mg/L)"


@dataclass(frozen=True)
class BranchSpec:
    name: str
    q_col: str


BRANCH_SPECS: dict[str, BranchSpec] = {
    "Ham Luong": BranchSpec("Ham Luong", "HamLuong"),
    "Co Chien": BranchSpec("Co Chien", "CoChien"),
    "My Tho": BranchSpec("My Tho", "MyTho"),
    "Can Tho": BranchSpec("Can Tho", "CanTho"),
    "Tien": BranchSpec("Tien", "TanChau"),
    "Hau": BranchSpec("Hau", "ChauDoc"),
}


def _buffer_area_ha(buffer_km: float) -> float:
    # circle area: π r^2 (km^2); 1 km^2 = 100 ha
    return float(np.pi * (buffer_km**2) * 100.0)


def _load_tons_per_day(c_mg_l: float, q_m3_s: float) -> float:
    # L (tons/day) = C (mg/L) * Q (m3/s) * 0.0864
    return float(c_mg_l * q_m3_s * 0.0864)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for fp in [WQ_FILE, LULC_FILE, Q_FILE]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing required input file: {fp}")

    df_wq = pd.read_csv(WQ_FILE)
    if "datetime" not in df_wq.columns or "stationName" not in df_wq.columns:
        raise ValueError("Water quality file missing required columns: datetime, stationName")
    df_wq["year"] = pd.to_datetime(df_wq["datetime"], errors="coerce").dt.year

    df_lulc = pd.read_csv(LULC_FILE)
    required_lulc_cols = {
        "year",
        "stationName",
        "River",
        "buffer_km",
        "Residential_Urban_composition_pct",
        "Residential_Urban_config_PD",
    }
    missing = required_lulc_cols - set(df_lulc.columns)
    if missing:
        raise ValueError(f"LULC metrics file missing required columns: {sorted(missing)}")

    # Gold Standard hydrology master (daily)
    df_q_daily = load_gold_standard(Q_FILE).reset_index().rename(columns={"Timestamp": "Date"})
    df_q_daily["year"] = pd.to_datetime(df_q_daily["Date"], errors="coerce").dt.year

    return df_wq, df_lulc, df_q_daily


def compute_bentre_load_intensity(df_wq: pd.DataFrame, df_lulc: pd.DataFrame, df_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute COD load intensity vs urban fragmentation for Ben Tre tributaries."""

    # Annual mean concentration per station-year
    wq_annual = (
        df_wq.groupby(["stationName", "year"], as_index=False)
        .mean(numeric_only=True)
    )

    if TARGET not in wq_annual.columns:
        raise ValueError(f"Water quality file does not contain target column: {TARGET}")

    # Merge with LULC metrics (same station-year)
    merged = pd.merge(wq_annual, df_lulc, on=["stationName", "year"], how="inner")

    # Keep years where LULC exists (typically 1990–2020); focus on Ben Tre monitoring era
    merged = merged[(merged["year"] >= 2005) & (merged["year"] <= 2020)].copy()

    # Ben Tre-coded stations generally have short IDs (HLxx/CCxx/MTxx/BLxx)
    merged = merged[merged["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")].copy()

    # Attach annual mean discharge by branch
    q_cols = ["year"] + [spec.q_col for spec in BRANCH_SPECS.values() if spec.q_col in df_q.columns]
    q_annual = df_q[q_cols].groupby("year", as_index=False).mean(numeric_only=True)

    def _assign_q(row: pd.Series) -> float:
        river = str(row.get("River", ""))
        # direct river-name mapping
        if river in BRANCH_SPECS and BRANCH_SPECS[river].q_col in q_annual.columns:
            q_col = BRANCH_SPECS[river].q_col
        else:
            # fallback by station prefix
            st = str(row.get("stationName", ""))
            if st.startswith("HL") and "HamLuong" in q_annual.columns:
                q_col = "HamLuong"
            elif st.startswith("CC") and "CoChien" in q_annual.columns:
                q_col = "CoChien"
            elif st.startswith("MT") and "MyTho" in q_annual.columns:
                q_col = "MyTho"
            else:
                return np.nan

        q_series = pd.Series(q_annual.loc[q_annual["year"] == row["year"], q_col])
        vals = q_series.to_numpy()
        if vals.size == 0:
            return np.nan
        return float(vals[0])

    merged["Q_branch_m3s"] = merged.apply(_assign_q, axis=1)

    # Drop where discharge missing (e.g., Ba Lai)
    merged = merged.dropna(subset=["Q_branch_m3s", TARGET, "buffer_km"]).copy()

    # Compute loads and intensities
    merged["COD_Load_tons_day"] = merged.apply(lambda r: _load_tons_per_day(r[TARGET], r["Q_branch_m3s"]), axis=1)

    merged["Buffer_area_ha"] = merged["buffer_km"].apply(_buffer_area_ha)
    merged["Urban_area_ha"] = merged["Buffer_area_ha"] * (merged["Residential_Urban_composition_pct"] / 100.0)

    merged["COD_Load_kg_ha_buffer_day"] = (merged["COD_Load_tons_day"] * 1000.0) / merged["Buffer_area_ha"]

    # Avoid unstable division where urban area is tiny
    min_urban_pct = 0.5
    merged["COD_Load_kg_ha_urban_day"] = np.where(
        merged["Residential_Urban_composition_pct"] >= min_urban_pct,
        (merged["COD_Load_tons_day"] * 1000.0) / merged["Urban_area_ha"],
        np.nan,
    )

    # Quantile groups by Urban PD within the analyzed subset
    merged["Urban_PD_quantile"] = pd.qcut(
        merged["Residential_Urban_config_PD"],
        q=4,
        labels=["Q1 (lowest PD)", "Q2", "Q3", "Q4 (highest PD)"],
        duplicates="drop",
    )

    summary = (
        merged.groupby("Urban_PD_quantile", as_index=False, observed=True)
        .agg(
            N=("stationName", "count"),
            Urban_PD_median=("Residential_Urban_config_PD", "median"),
            Urban_pct_median=("Residential_Urban_composition_pct", "median"),
            COD_Load_tons_day_mean=("COD_Load_tons_day", "mean"),
            COD_Load_tons_day_sd=("COD_Load_tons_day", "std"),
            COD_kg_ha_buffer_day_mean=("COD_Load_kg_ha_buffer_day", "mean"),
            COD_kg_ha_buffer_day_sd=("COD_Load_kg_ha_buffer_day", "std"),
            COD_kg_ha_urban_day_mean=("COD_Load_kg_ha_urban_day", "mean"),
            COD_kg_ha_urban_day_sd=("COD_Load_kg_ha_urban_day", "std"),
        )
        .sort_values("Urban_PD_median")
    )

    return merged, summary


def compute_mass_balance_decomposition(df_wq: pd.DataFrame, df_q: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple ΔC decomposition for selected key stations."""

    key_stations = ["Tan Chau", "My Thuan", "My Tho", "Can Tho"]

    # Annual mean concentrations
    wq_annual = (
        df_wq[df_wq["stationName"].isin(key_stations)]
        .groupby(["stationName", "year"], as_index=False)
        .mean(numeric_only=True)
    )

    if TARGET not in wq_annual.columns:
        raise ValueError(f"Water quality file does not contain target column: {TARGET}")

    # Annual mean discharges (station-specific)
    q_annual = df_q.groupby("year", as_index=False).mean(numeric_only=True)

    station_q_col = {
        "Tan Chau": GoldStandardColumns().q_tanchau,
        "My Thuan": GoldStandardColumns().q_mythuan,
        "My Tho": GoldStandardColumns().q_mytho,
        "Can Tho": GoldStandardColumns().q_cantho,
    }

    rows: list[dict] = []
    for st in key_stations:
        q_col = station_q_col.get(st)
        if q_col is None or q_col not in q_annual.columns:
            continue

        st_ann = wq_annual[wq_annual["stationName"] == st].copy()
        st_ann = st_ann[(st_ann["year"] >= 2005) & (st_ann["year"] <= 2022)].copy()
        if st_ann.empty:
            continue

        # merge in Q by year
        st_ann = pd.merge(st_ann, q_annual[["year", q_col]], on="year", how="left")
        st_ann = st_ann.dropna(subset=[TARGET, q_col])
        if st_ann.shape[0] < 2:
            continue

        # Endpoints: first and last available year in window
        y0 = int(st_ann["year"].min())
        y1 = int(st_ann["year"].max())

        r0 = st_ann.loc[st_ann["year"] == y0].iloc[0]
        r1 = st_ann.loc[st_ann["year"] == y1].iloc[0]

        C0 = float(r0[TARGET])
        C1 = float(r1[TARGET])
        Q0 = float(r0[q_col])
        Q1 = float(r1[q_col])

        L0 = _load_tons_per_day(C0, Q0)
        L1 = _load_tons_per_day(C1, Q1)

        Q_mean = float(np.mean([Q0, Q1]))
        L_mean = float(np.mean([L0, L1]))

        # Decomposition in concentration units (mg/L)
        dC_total = C1 - C0
        dC_load = (L1 - L0) / (Q_mean * 0.0864)
        dC_dilution = (L_mean / 0.0864) * (1.0 / Q1 - 1.0 / Q0)
        dC_resid = dC_total - (dC_load + dC_dilution)

        def _pct(x: float) -> float:
            if dC_total == 0 or not np.isfinite(dC_total):
                return np.nan
            return float(100.0 * x / dC_total)

        rows.append(
            {
                "Station": st,
                "Year0": y0,
                "Year1": y1,
                "C0_mgL": C0,
                "C1_mgL": C1,
                "Q0_m3s": Q0,
                "Q1_m3s": Q1,
                "L0_tons_day": L0,
                "L1_tons_day": L1,
                "dC_total_mgL": dC_total,
                "dC_load_mgL": dC_load,
                "dC_dilution_mgL": dC_dilution,
                "dC_residual_mgL": dC_resid,
                "pct_load": _pct(dC_load),
                "pct_dilution": _pct(dC_dilution),
                "pct_residual": _pct(dC_resid),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df_wq, df_lulc, df_q = load_inputs()

    # 1) Ben Tre mechanistic-style load intensity diagnostic
    station_year, summary = compute_bentre_load_intensity(df_wq, df_lulc, df_q)

    out_station_year = os.path.join(OUTPUT_DIR, "cod_load_intensity_station_year_latest.csv")
    out_summary = os.path.join(OUTPUT_DIR, "cod_load_intensity_by_urban_fragmentation_latest.csv")
    station_year.to_csv(out_station_year, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"Wrote station-year load intensity table: {out_station_year}")
    print(f"Wrote fragmentation-quantile summary: {out_summary}")

    # 2) Simple ΔC decomposition for key stations
    df_mb = compute_mass_balance_decomposition(df_wq, df_q)
    out_mb = os.path.join(OUTPUT_DIR, "cod_mass_balance_decomposition_latest.csv")
    df_mb.to_csv(out_mb, index=False)
    print(f"Wrote mass-balance decomposition table: {out_mb}")


if __name__ == "__main__":
    main()
