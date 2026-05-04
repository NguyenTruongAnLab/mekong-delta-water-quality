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
# ZENODO DOI  (replace with real DOI after upload)
# ═══════════════════════════════════════════════════════════════════════
ZENODO_DOI = "10.5281/zenodo.20024194"

_RESTRICTED_MSG = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  RESTRICTED DATA REQUIRED                                               ║
║                                                                         ║
║  This script requires the full water-quality dataset that includes      ║
║  Ben Tre provincial monitoring data (DONRE) and/or SIWRP hydrology.     ║
║  These are subject to third-party organizational access policies and    ║
║  are NOT included in this public GitHub repository.                     ║
║                                                                         ║
║  To obtain the complete dataset:                                        ║
║    1. Request access via the restricted Zenodo record:                  ║
║       {ZENODO_DOI}                                                      ║
║    2. After approval, place the data files in the `data/` folder        ║
║       at the repository root (see data/README.md for structure).        ║
║                                                                         ║
║  For applying this workflow to YOUR OWN river system, replace the       ║
║  data files with your own datasets following the same schema.           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _fail_restricted(file_label: str, path: Path) -> None:
    """Print clear guidance about restricted data and exit."""
    print(f"\n  Missing: {file_label}")
    print(f"  Expected at: {path}")
    print(_RESTRICTED_MSG)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# PATHS (relative to repository root = scripts/ parent)
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[3]  # repo root

WQ_PATH     = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
LULC_PATH   = ROOT / "analysis" / "cache" / "lulc_metrics_corrected_16km.csv"
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

# 5 WQ parameters used in spatial attribution (TP excluded: 100% NaN in Ben Tre)
WQ_VARS = ["COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)", "TSS (mg/L)"]

# 6 baseline LULC predictors
LULC_VARS = [
    "Residential_Urban_composition_pct",
    "Residential_Urban_config_PD",
    "Aquaculture_composition_pct",
    "Aquaculture_config_ED",
    "Rice_Cultivation_composition_pct",
    "Rice_Cultivation_config_PD",
]

# Display labels for LULC predictors
LABEL_MAP_X = {
    "Residential_Urban_composition_pct": "Urban %",
    "Residential_Urban_config_PD":      "Urban PD",
    "Aquaculture_composition_pct":      "Aquaculture %",
    "Aquaculture_config_ED":            "Aquaculture ED",
    "Rice_Cultivation_composition_pct": "Rice %",
    "Rice_Cultivation_config_PD":       "Rice PD",
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


def load_wq_raw() -> pd.DataFrame:
    """Load the raw water quality CSV and add a year column.

    Requires the full dataset including Ben Tre DONRE restricted data.
    If missing, prints clear guidance about Zenodo restricted-access record.
    """
    if not WQ_PATH.exists():
        _fail_restricted("Water quality CSV (MRC + Ben Tre DONRE)", WQ_PATH)
    df = pd.read_csv(WQ_PATH)
    df["year"] = pd.to_datetime(df["datetime"], errors="coerce").dt.year
    return df


def load_wq_annual(include_mrc: bool = False) -> pd.DataFrame:
    """WQ → annual station means, optionally excluding MRC stations."""
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
        X.rename(columns=LABEL_MAP_X, inplace=True)
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
