"""
Shared data loading, path management, and table-writing utilities
for the Supplementary Table generation pipeline.
=================================================================

All table scripts import from here to guarantee:
  • Identical data loading logic (raw WQ CSV + LULC CSV + hydro Gold Standard)
  • Consistent Ben Tre / MRC station filtering
  • Reproducible merge and analysis-window subsetting
  • Uniform Markdown table output with Quarto cross-reference labels
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# PATHS (relative to repository root = scripts/ parent)
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[3]  # repo root

WQ_PATH     = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
LULC_PATH   = ROOT / "analysis" / "cache" / "lulc_metrics_corrected_16km.csv"
MASTER_PATH = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
HYDRO_PATH  = (ROOT / "data" / "hydrology" / "outputs" / "final_datasets"
               / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv")
Q_FILE      = ROOT / "analysis" / "hydrology_MRC" / "VMD_Total_Discharge_Estimated_1985_2022.csv"
TIDAL_FILE  = ROOT / "analysis" / "analysis_outputs" / "Table_Flushing_Capacity.csv"
TIDAL_YEAR  = ROOT / "analysis" / "analysis_outputs" / "Table_Tidal_Range_By_Year.csv"

# Output directories
MANU_TABLES = ROOT / "manuscript" / "tables"
METRICS_OUT = Path(__file__).resolve().parent / "metrics"
MANU_TABLES.mkdir(parents=True, exist_ok=True)
METRICS_OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
YEAR_MIN     = 2013
YEAR_MAX     = 2020
PRIMARY_YEAR_MIN = 2005
PRIMARY_YEAR_MAX = 2020

# 5 WQ parameters used in spatial attribution (TP excluded: 100% NaN in Ben Tre)
WQ_VARS = ["COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)", "TSS (mg/L)"]

# Corresponding column names in station_year_master_latest.csv
WQ_ANNUAL_VARS = [
    "COD (mg/L)_annual",
    "DO (mg/L)_annual",
    "NH4-N (mg/L)_annual",
    "NO3-N (mg/L)_annual",
    "TSS (mg/L)_annual",
]

LABEL_MAP_WQ_ANNUAL = {
    "COD (mg/L)_annual":   "COD",
    "DO (mg/L)_annual":    "DO",
    "NH4-N (mg/L)_annual": "NH4-N",
    "NO3-N (mg/L)_annual": "NO3-N",
    "TSS (mg/L)_annual":   "TSS",
}

# 6 baseline LULC predictors
LULC_VARS = [
    "Residential_Urban_composition_pct",
    "Residential_Urban_config_PD",
    "Aquaculture_composition_pct",
    "Aquaculture_config_ED",
    "Rice_Cultivation_composition_pct",
    "Rice_Cultivation_config_PD",
]

# Primary augmented predictors (6 LULC + 1 Pop + 4 Hydro-climate = 11 features)
PRIMARY_POP_VARS = ["pop_density"]
PRIMARY_HYDRO_VARS = [
    "Rain_Annual_z",
    "Rain_Dry_z",
    "Q_MyThuan_AnnualMean_z",
    "TR_MyThuan_m",
]
PRIMARY_AUGMENTED_VARS = LULC_VARS + PRIMARY_POP_VARS + PRIMARY_HYDRO_VARS

# Display labels for predictors
LABEL_MAP_X = {
    "Residential_Urban_composition_pct": "Urban %",
    "Residential_Urban_config_PD":      "Urban PD",
    "Aquaculture_composition_pct":      "Aquaculture %",
    "Aquaculture_config_ED":            "Aquaculture ED",
    "Rice_Cultivation_composition_pct": "Rice %",
    "Rice_Cultivation_config_PD":       "Rice PD",
    "pop_density":                      "Population Density",
    "Rain_Annual_z":                    "Annual Rainfall (z)",
    "Rain_Dry_z":                       "Dry-Season Rainfall (z)",
    "Q_MyThuan_AnnualMean_z":           "Annual Discharge (z)",
    "TR_MyThuan_m":                     "Tidal Range (m)",
}

LABEL_MAP_Y = {
    "COD (mg/L)":   "COD",
    "DO (mg/L)":    "DO",
    "NH4-N (mg/L)": "NH4-N",
    "NO3-N (mg/L)": "NO3-N",
    "TSS (mg/L)":   "TSS",
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def require_file(path: Path, label: str = "") -> None:
    """Fail-fast if a required data file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {label or path.name}\n  path: {path}"
        )


def load_master_bentre(
    year_min: int = PRIMARY_YEAR_MIN,
    year_max: int = PRIMARY_YEAR_MAX,
) -> pd.DataFrame:
    """Load station_year_master_latest.csv filtered to Ben Tre stations and year window."""
    require_file(MASTER_PATH, "Station-year master CSV")
    df = pd.read_csv(MASTER_PATH)
    df = df[df["network"] == "Ben Tre"].copy()
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    return df


def shap_direction_and_corr(
    X: pd.DataFrame,
    shap_values: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Compute SHAP feature effect direction from sign of Pearson r(X_j, phi_j).

    Because E[phi_j] = 0 by additive expectation, the raw arithmetic mean of SHAP
    values fluctuates near zero due to sampling and out-of-fold test splits.
    The physically meaningful direction of a feature's effect is the sign of
    corr(X_j, phi_j) across observations.
    """
    X_arr = np.asarray(X, dtype=float)
    sv_arr = np.asarray(shap_values, dtype=float)
    n_features = X_arr.shape[1]
    corrs = np.zeros(n_features, dtype=float)
    directions: list[str] = []
    for j in range(n_features):
        x_j = X_arr[:, j]
        sv_j = sv_arr[:, j]
        valid_mask = np.isfinite(x_j) & np.isfinite(sv_j)
        if valid_mask.sum() > 2 and np.std(x_j[valid_mask]) > 0 and np.std(sv_j[valid_mask]) > 0:
            corr_j = float(np.corrcoef(x_j[valid_mask], sv_j[valid_mask])[0, 1])
        else:
            corr_j = 0.0
        corrs[j] = corr_j
        directions.append("Positive" if corr_j >= 0 else "Negative")
    return directions, corrs


def load_wq_raw() -> pd.DataFrame:
    """Load the raw water quality CSV and add a year column."""
    require_file(WQ_PATH, "Water quality CSV")
    df = pd.read_csv(WQ_PATH)
    df["year"] = pd.to_datetime(df["datetime"], errors="coerce").dt.year
    return df


def load_wq_annual(include_mrc: bool = False) -> pd.DataFrame:
    """Load annual station water quality means.

    Uses the harmonized station-year master table when available and falls back
    to aggregating raw monitoring observations if the master cache is absent.
    """
    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH, float_precision="round_trip")
        if not include_mrc:
            master = master[~master["stationName"].isin(MRC_STATIONS)].copy()
        rename_map = {f"{v}_annual": v for v in WQ_VARS if f"{v}_annual" in master.columns}
        if "Salinity_annual" in master.columns:
            rename_map["Salinity_annual"] = "Salinity"
        keep_cols = ["stationName", "year"] + list(rename_map.keys())
        wq_annual = master[keep_cols].rename(columns=rename_map).copy()
        return wq_annual

    require_file(WQ_PATH, "Water quality CSV")
    df = load_wq_raw()
    if not include_mrc:
        df = df[~df["stationName"].isin(MRC_STATIONS)]
    wq_annual = df.groupby(["stationName", "year"]).mean(numeric_only=True).reset_index()
    return wq_annual


def load_lulc() -> pd.DataFrame:
    """Load the corrected 16 km LULC metrics."""
    require_file(LULC_PATH, "LULC metrics CSV")
    return pd.read_csv(LULC_PATH)


def merge_wq_lulc(
    include_mrc: bool = False,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    bentre_only: bool = True,
) -> pd.DataFrame:
    """
    Merge WQ annual means with LULC metrics and subset to the
    analysis window (default 2013–2020, Ben Tre stations only).
    """
    wq = load_wq_annual(include_mrc=include_mrc)
    lulc = load_lulc()
    merged = pd.merge(wq, lulc, on=["stationName", "year"], how="inner")
    merged = merged[(merged["year"] >= year_min) & (merged["year"] <= year_max)].copy()

    if bentre_only:
        # Ben Tre station codes: HL01..HL10, CC01.., MT01.., BL01..
        merged = merged[
            merged["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")
        ].copy()

    return merged


def prepare_xy(
    df: pd.DataFrame,
    use_display_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    From merged DataFrame, extract:
      - Y: WQ response matrix (available parameters only)
      - X: LULC predictor matrix (6 baseline predictors)
      - groups: station names for LOSO CV
    """
    wq_vars = [c for c in WQ_VARS if c in df.columns]
    lulc_vars = [c for c in LULC_VARS if c in df.columns]

    df_clean = df.dropna(subset=wq_vars + lulc_vars).copy()
    Y = df_clean[wq_vars].copy()
    X = df_clean[lulc_vars].copy()
    groups = df_clean["stationName"]

    if use_display_labels:
        X.rename(columns={k: v for k, v in LABEL_MAP_X.items() if k in LULC_VARS}, inplace=True)
        Y.rename(columns=LABEL_MAP_Y, inplace=True)

    return Y, X, groups


# ═══════════════════════════════════════════════════════════════════════
# TABLE WRITING
# ═══════════════════════════════════════════════════════════════════════

def write_table_md(
    content: str,
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    """Write a Markdown table file with Quarto caption + cross-reference label.

    NOTE: Pandoc requires the caption line (starting with ':') to immediately
    follow the table with NO blank line. A blank line would cause the caption
    to render as a standalone paragraph instead of a table caption.
    """
    full = f"{content}\n: {caption} {{{label}}}\n"
    out_path.write_text(full, encoding="utf-8")
    print(f"  ✓ {out_path.name}")


def write_metrics_csv(df: pd.DataFrame, name: str) -> None:
    """Write companion metrics CSV alongside the table."""
    out = METRICS_OUT / f"{name}_metrics.csv"
    df.to_csv(out, index=False)


def ensure_hydro_ingest_importable() -> None:
    """Add analysis/ to sys.path so we can import analysis.lib.hydro_ingest."""
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)

