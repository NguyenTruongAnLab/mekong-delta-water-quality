"""
Shared utilities, paths, styles, and data loaders for Nature Water figures.
===========================================================================

All figure scripts import from here to guarantee identical:
  • file paths and data loading logic
  • matplotlib style (Nature Water: Arial, 7–10 pt, 180 mm, 600 DPI)
  • colour palettes (colorblind-friendly)
  • panel-label helper
  • save helper (PDF + PNG + companion metrics CSV)

This module is the single source of truth for the entire figure pipeline.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes, kendalltau

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# ZENODO DOI  (replace with real DOI after upload)
# ═══════════════════════════════════════════════════════════════════════
ZENODO_DOI = "10.5281/zenodo.20024194"


def placeholders_allowed() -> bool:
    """Whether the figure pipeline may use placeholder/fallback data.

    Default is strict (False) to avoid silently generating submission figures
    from missing inputs. Set NW_ALLOW_PLACEHOLDERS=1 to enable fallbacks.
    """
    v = os.environ.get("NW_ALLOW_PLACEHOLDERS", "0").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def require_file(path: Path, *, label: str | None = None) -> None:
    """Fail-fast if a required input file is missing (unless placeholders are allowed)."""
    if path.exists():
        return
    name = label or path.name
    msg = (
        f"Required input missing: {name}\n"
        f"  path: {path}\n"
        "Either regenerate upstream artifacts or (for development only) set "
        "NW_ALLOW_PLACEHOLDERS=1 to allow placeholder outputs."
    )
    if placeholders_allowed():
        print(f"WARNING: {msg}")
        return
    raise FileNotFoundError(msg)


# ═══════════════════════════════════════════════════════════════════════
# RESTRICTED DATA  (Ben Tre DONRE / SIWRP — available only via Zenodo)
# ═══════════════════════════════════════════════════════════════════════
_RESTRICTED_MSG = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  RESTRICTED DATA REQUIRED                                               ║
║                                                                         ║
║  This script requires the full water-quality and/or hydrology dataset   ║
║  that includes Ben Tre provincial monitoring data (DONRE) and SIWRP     ║
║  discharge/water-level records. These are subject to third-party        ║
║  organizational access policies and are NOT included in this public     ║
║  GitHub repository.                                                     ║
║                                                                         ║
║  To obtain the complete dataset:                                        ║
║    1. Request access via the restricted Zenodo record:                  ║
║       {ZENODO_DOI}                                                      ║
║    2. After approval, place the data files in the `data/` folder        ║
║       at the repository root (see data/README.md for structure).        ║
║                                                                         ║
║  MRC water-quality data are publicly available from:                    ║
║    https://portal.mrcmekong.org/                                        ║
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
# PATHS  (relative to repository root)
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[3]          # repo root

WQ_PATH      = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
HYDRO_PATH   = (ROOT / "data" / "hydrology" / "outputs" / "final_datasets"
                / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv")
WL_PATH      = (ROOT / "data" / "hydrology" / "outputs" / "final_datasets"
                / "water_level_unified_subset_1985_2022.csv")
MASTER_PATH  = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
LULC_PATH    = ROOT / "analysis" / "cache" / "lulc_metrics_corrected_16km.csv"

# Pre-computed analysis outputs
AO          = ROOT / "analysis" / "analysis_outputs"
FE_CSV      = AO / "cod_confirmatory_fixed_effects_latest.csv"
SHAP_CSV    = AO / "primary_model_shap_latest.csv"
SHAP_HC_CSV = AO / "shap_with_hydro_climate_latest.csv"
VP_CSV      = AO / "primary_variance_partitioning_latest.csv"
BUF_BT_CSV  = AO / "Buffer_Correlations_BenTre.csv"
HFI_CSV     = AO / "hfi_tr_over_q_dryseason_latest.csv"
TIDAL_CSV   = AO / "Table_Tidal_Range_By_Year.csv"
MIX_CSV     = AO / "mixing_excess_summary_latest.csv"
LTG_CSV     = AO / "lulc_trajectory_groups_16km_1990_2020_latest.csv"
CP_CSV      = AO / "changepoint_effect_sizes_latest.csv"
ATTR_HC_CSV = AO / "attribution_with_hydroclimate_comparison_latest.csv"
MAP_IMG     = ROOT / "manuscript" / "figures" / "Figure_2.1_Study_Area_Map.png"
MB_CSV      = ROOT / "manuscript" / "tables" / "Table_NW_Mass_Balance.csv"

# Output directories
FIG_OUT     = ROOT / "manuscript" / "figures" / "nw_composites"
METRICS_OUT = Path(__file__).resolve().parent / "metrics"
FIG_OUT.mkdir(parents=True, exist_ok=True)
METRICS_OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
BREAK_YEAR = 2004
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]

# Core 5 parameters for main-text figures (manuscript narrative consistency)
CORE_PARAMS = [
    ("COD (mg/L)", "COD", r"COD (mg $\mathrm{L^{-1}}$)"),
    ("DO (mg/L)",  "DO",  r"DO (mg $\mathrm{L^{-1}}$)"),
    ("NH4-N (mg/L)", r"NH$_4^+$", r"NH$_4$-N (mg $\mathrm{L^{-1}}$)"),
    ("TN (mg/L)", "TN", r"TN (mg $\mathrm{L^{-1}}$)"),
    ("TSS (mg/L)", "TSS", r"TSS (mg $\mathrm{L^{-1}}$)"),
]

# Station → discharge column mapping
# Only Tan Chau has a real independent gauge; downstream stations use
# Total_Inflow (= TanChau + ChauDoc) as the system-level proxy, because
# CanTho Q is synthetic (0.907 × MyThuan) and My Tho has no gauge.
STN_Q_MAP = {
    "Tan Chau": "TanChau",
    "Can Tho":  "Total_Inflow",
    "My Thuan": "Total_Inflow",
    "My Tho":   "Total_Inflow",
}

# mm → inches conversion
MM = 1 / 25.4

# ═══════════════════════════════════════════════════════════════════════
# NATURE WATER MATPLOTLIB STYLE
# ═══════════════════════════════════════════════════════════════════════
NW_STYLE: dict = {
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           7,
    "axes.labelsize":      8,
    "axes.titlesize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     6.5,
    "legend.title_fontsize": 7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "axes.grid":           False,
    "grid.linewidth":      0.4,
    "grid.alpha":          0.3,
    "figure.dpi":          150,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "savefig.facecolor":   "white",
    "lines.linewidth":     1.0,
    "lines.markersize":    3,
}


def apply_style() -> None:
    """Apply Nature Water publication style globally."""
    plt.rcParams.update(NW_STYLE)


# ═══════════════════════════════════════════════════════════════════════
# COLOUR PALETTES  (all colorblind-friendly)
# ═══════════════════════════════════════════════════════════════════════
PHASE1_CLR = "#2166ac"          # blue  – Phase 1 (≤2004)
PHASE2_CLR = "#b2182b"          # red   – Phase 2 (>2004)

STATION_CLR = {
    "Tan Chau": "#2166ac",
    "Can Tho":  "#4393c3",
    "My Thuan": "#f4a582",
    "My Tho":   "#d6604d",
}

PARAM_CLR = {
    # Keep PHASE1/PHASE2 semantics exclusive to blue/red.
    "COD": "#e66101", "DO":    "#5ab4ac", "pH":    "#2ca02c",
    "NH₄⁺": "#fdb863", "NO₃⁻": "#8073ac", "TN":   "#018571",
    "TP":  "#e377c2",  "TSS":  "#8c564b", "Sal":  "#bcbd22",
}

LULC_CLR = {
    "Rice":         "#66c2a5",
    "Upland Crops": "#e6ab02",
    "Aquaculture":  "#3288bd",
    "Residential":  "#d53e4f",
    "Forest":       "#5e4fa2",
    "Wetland":      "#bf812d",
    "Water":        "#80b1d3",
}

CLR_SPATIAL  = "#3288bd"        # variance decomposition
CLR_TEMPORAL = "#d53e4f"
CLR_RESIDUAL = "#d9d9d9"


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def panel_label(ax, text: str, x: float = -0.06, y: float = 1.06,
                fontsize: float = 10) -> None:
    """Add bold lowercase panel label (Nature style)."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            fontfamily="sans-serif")


def save_fig(fig: plt.Figure, name: str, metrics: pd.DataFrame | None = None) -> None:
    """Save figure as PDF + PNG (600 DPI) and optional companion metrics CSV."""
    fig.savefig(FIG_OUT / f"{name}.pdf", format="pdf")
    fig.savefig(FIG_OUT / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  ✓ {name}  →  {FIG_OUT / name}.{{pdf,png}}")
    if metrics is not None:
        csv_path = METRICS_OUT / f"{name}_metrics.csv"
        metrics.to_csv(csv_path, index=False)
        print(f"    metrics → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING  (lazy-cached singletons)
# ═══════════════════════════════════════════════════════════════════════
_cache: dict = {}


def load_wq() -> pd.DataFrame:
    """Load water-quality data (1985–2022).

    Requires the full dataset including Ben Tre DONRE restricted data.
    If missing, prints clear guidance about Zenodo restricted-access record.
    """
    if "wq" not in _cache:
        if not WQ_PATH.exists():
            _fail_restricted("Water quality CSV (MRC + Ben Tre DONRE)", WQ_PATH)
        wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
        wq["year"]  = wq["datetime"].dt.year
        wq["month"] = wq["datetime"].dt.month
        wq["ym"]    = wq["datetime"].dt.to_period("M")
        wq = wq[wq["year"] <= 2022].copy()
        _cache["wq"] = wq
    return _cache["wq"]


def load_hydro_monthly() -> pd.DataFrame:
    """Monthly-mean discharge at key stations.

    Requires the SIWRP Gold Standard hydrology dataset.
    If missing, prints clear guidance about Zenodo restricted-access record.
    """
    if "q_monthly" not in _cache:
        if not HYDRO_PATH.exists():
            _fail_restricted("Hydrology Gold Standard master (SIWRP)", HYDRO_PATH)
        h = pd.read_csv(HYDRO_PATH)
        h["date"]  = pd.to_datetime(h["Timestamp"], format="mixed", dayfirst=False)
        h["year"]  = h["date"].dt.year
        h["month"] = h["date"].dt.month
        h["ym"]    = h["date"].dt.to_period("M")
        h = h[h["year"] <= 2022]
        q_m = h.groupby("ym").agg(
            TanChau=("TanChau", "mean"),
            MyThuan=("MyThuan", "mean"),
            CanTho=("CanTho", "mean"),
            Total_Inflow=("Total_Inflow", "mean"),
        ).reset_index()
        _cache["q_monthly"] = q_m
    return _cache["q_monthly"]


def load_master() -> pd.DataFrame:
    """Station-year master table (WQ + LULC + Pop + Hydro)."""
    if "master" not in _cache:
        _cache["master"] = pd.read_csv(MASTER_PATH)
    return _cache["master"]


def load_lulc() -> pd.DataFrame:
    """LULC metrics at 16 km buffer scale."""
    if "lulc" not in _cache:
        _cache["lulc"] = pd.read_csv(LULC_PATH)
    return _cache["lulc"]


def load_bt(master: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return Ben-Tre-only rows from the master table."""
    if master is None:
        master = load_master()
    return master[~master["stationName"].isin(MRC_STATIONS)].copy()


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════
def cq_metrics(c: np.ndarray, q: np.ndarray):
    """C–Q power-law exponent β, standard error, chemostatic index, p-value."""
    mask = np.isfinite(c) & np.isfinite(q) & (c > 0) & (q > 0)
    c, q = c[mask], q[mask]
    if len(c) < 10:
        return np.nan, np.nan, np.nan, np.nan
    lc, lq = np.log10(c), np.log10(q)
    sl, ic, r, p, se = stats.linregress(lq, lc)
    cv_c = np.std(c) / np.mean(c)
    cv_q = np.std(q) / np.mean(q)
    ci = cv_c / cv_q if cv_q > 0 else np.nan
    return sl, se, ci, p


# ═══════════════════════════════════════════════════════════════════════
# DATA AVAILABILITY CHECK
# ═══════════════════════════════════════════════════════════════════════

# Which restricted raw-data files are present?
WQ_AVAILABLE = WQ_PATH.exists()
HYDRO_AVAILABLE = HYDRO_PATH.exists()
WL_AVAILABLE = WL_PATH.exists()
ALL_RESTRICTED_AVAILABLE = WQ_AVAILABLE and HYDRO_AVAILABLE and WL_AVAILABLE


def print_data_status() -> None:
    """Print a summary of which data sources are available / missing."""
    print("=" * 70)
    print("  DATA AVAILABILITY STATUS")
    print("=" * 70)

    def _status(available: bool, label: str) -> str:
        return "✓ AVAILABLE" if available else "✗ MISSING (restricted)"

    print(f"\n  Raw datasets (restricted — require Zenodo access):")
    print(f"    {_status(WQ_AVAILABLE, 'WQ'):30s}  Water quality (MRC + Ben Tre)")
    print(f"    {_status(HYDRO_AVAILABLE, 'HYDRO'):30s}  Discharge (SIWRP Gold Standard)")
    print(f"    {_status(WL_AVAILABLE, 'WL'):30s}  Water levels (SIWRP)")

    print(f"\n  Pre-computed outputs (included in this repo):")
    for label, p in [
        ("Master table", MASTER_PATH),
        ("LULC metrics", LULC_PATH),
        ("Fixed effects", FE_CSV),
        ("SHAP", SHAP_CSV),
        ("Variance partitioning", VP_CSV),
        ("HFI (TR/Q)", HFI_CSV),
        ("Mass balance", MB_CSV),
        ("LULC trajectory groups", LTG_CSV),
    ]:
        print(f"    {'✓' if p.exists() else '✗'}  {label}")

    if not ALL_RESTRICTED_AVAILABLE:
        print(f"\n  ═══ RESTRICTED DATA REQUIRED ═══")
        print(f"  Some scripts need Ben Tre DONRE / SIWRP data.")
        print(f"  Request access via Zenodo: {ZENODO_DOI}")
        print(f"  Then place files in the data/ folder (see data/README.md).")
        print(f"  ═══════════════════════════════")

    print()

# Print status on import (can be silenced with NW_QUIET=1)
if os.environ.get("NW_QUIET", "0").strip() not in {"1", "true", "yes"}:
    print_data_status()
