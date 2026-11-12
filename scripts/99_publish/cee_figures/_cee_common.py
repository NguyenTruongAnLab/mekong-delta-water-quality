"""
Shared utilities, paths, styles, and data loaders for CEE manuscript figures.
=============================================================================
Cloned directly from `scripts/99_publish/nw_figures/_nw_common.py` to preserve
100% of the author's tuned typography, palettes, layout proportions, and dynamic
panel-label alignment, outputting to `manuscript/figures/cee_composites/`.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes, kendalltau

warnings.filterwarnings("ignore")


def placeholders_allowed() -> bool:
    v = os.environ.get("NW_ALLOW_PLACEHOLDERS", "0").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def require_file(path: Path, *, label: str | None = None) -> None:
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
# PATHS  (relative to repository root)
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[3]

WQ_PATH      = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
HYDRO_PATH   = (ROOT / "data" / "hydrology" / "outputs" / "final_datasets"
                / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv")
WL_PATH      = (ROOT / "data" / "hydrology" / "outputs" / "final_datasets"
                / "water_level_unified_subset_1985_2022.csv")
MASTER_PATH  = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
LULC_PATH    = ROOT / "analysis" / "cache" / "lulc_metrics_corrected_16km.csv"

# Pre-computed analysis outputs
AO             = ROOT / "analysis" / "analysis_outputs"
FE_CSV         = AO / "cod_confirmatory_fixed_effects_latest.csv"
SHAP_CSV       = AO / "primary_model_shap_latest.csv"
SHAP_HC_CSV    = AO / "shap_with_hydro_climate_latest.csv"
VP_CSV         = AO / "primary_variance_partitioning_latest.csv"
BUF_BT_CSV     = AO / "Buffer_Correlations_BenTre.csv"
HFI_CSV        = AO / "hfi_tr_over_q_dryseason_latest.csv"
TIDAL_CSV      = AO / "Table_Tidal_Range_By_Year.csv"
MIX_CSV        = AO / "mixing_excess_summary_latest.csv"
LTG_CSV        = AO / "lulc_trajectory_groups_16km_1990_2020_latest.csv"
CP_CSV         = AO / "changepoint_effect_sizes_latest.csv"
ATTR_HC_CSV    = AO / "attribution_with_hydroclimate_comparison_latest.csv"
MAP_IMG        = ROOT / "manuscript" / "figures" / "Figure_2.1_Study_Area_Map.png"
MB_CSV         = ROOT / "manuscript" / "tables" / "Table_NW_Mass_Balance.csv"
CEE_VERIFY_DIR = ROOT / "scripts" / "99_publish" / "cee_verification" / "metrics"

# Output directories
FIG_OUT     = ROOT / "manuscript" / "figures" / "cee_composites"
METRICS_OUT = Path(__file__).resolve().parent / "metrics"
FIG_OUT.mkdir(parents=True, exist_ok=True)
METRICS_OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
BREAK_YEAR = 2004
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]

CORE_PARAMS = [
    ("COD (mg/L)", "COD", r"COD (mg $\mathrm{L^{-1}}$)"),
    ("DO (mg/L)",  "DO",  r"DO (mg $\mathrm{L^{-1}}$)"),
    ("NH4-N (mg/L)", r"NH$_4^+$-N", r"NH$_4^+$-N (mg $\mathrm{L^{-1}}$)"),
    ("TN (mg/L)", "TN", r"TN (mg $\mathrm{L^{-1}}$)"),
    ("TSS (mg/L)", "TSS", r"TSS (mg $\mathrm{L^{-1}}$)"),
]

STN_Q_MAP = {
    "Tan Chau": "TanChau",
    "Can Tho":  "Total_Inflow",
    "My Thuan": "Total_Inflow",
    "My Tho":   "Total_Inflow",
}

MM = 1 / 25.4

# ═══════════════════════════════════════════════════════════════════════
# NATURE PORTFOLIO / CEE MATPLOTLIB STYLE (exact match to _nw_common.py)
# ═══════════════════════════════════════════════════════════════════════
NW_STYLE: dict = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":             8.4,
    "axes.labelsize":        9.6,
    "axes.titlesize":        9.6,
    "xtick.labelsize":       8.4,
    "ytick.labelsize":       8.4,
    "legend.fontsize":       7.8,
    "legend.title_fontsize": 8.4,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "axes.grid":             False,
    "grid.linewidth":        0.4,
    "grid.alpha":            0.3,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "savefig.facecolor":     "white",
    "lines.linewidth":       1.0,
    "lines.markersize":      3,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
}


def apply_style() -> None:
    """Apply Nature Portfolio / CEE publication style globally."""
    plt.rcParams.update(NW_STYLE)


# ═══════════════════════════════════════════════════════════════════════
# COLOUR PALETTES  (colorblind-friendly, identical to _nw_common.py)
# ═══════════════════════════════════════════════════════════════════════
PHASE1_CLR = "#2166ac"          # blue  – Phase 1 (<=2004)
PHASE2_CLR = "#b2182b"          # red   – Phase 2 (>2004)

STATION_CLR = {
    "Tan Chau": "#009e73",  # Emerald Green – upstream freshwater baseline
    "Can Tho":  "#0072b2",  # Deep Blue – urban midstream
    "My Thuan": "#e69f00",  # Amber Orange – central transition
    "My Tho":   "#d55e00",  # Vermillion Red – downstream tidal influence
}

PARAM_CLR = {
    "COD": "#e66101", "DO":    "#5ab4ac", "pH":    "#2ca02c",
    "NH₄⁺": "#fdb863", "NO₃⁻": "#8073ac", "TN":   "#018571",
    "TP":  "#e377c2",  "TSS":  "#8c564b", "Sal":  "#bcbd22",
}

LULC_CLR = {
    "Rice":         "#66c2a5",
    "Upland Crops": "#e6ab02",
    "Upland crops": "#e6ab02",
    "Aquaculture":  "#3288bd",
    "Residential":  "#d53e4f",
    "Forest":       "#5e4fa2",
    "Wetland":      "#bf812d",
    "Water":        "#80b1d3",
}

CLR_SPATIAL  = "#3288bd"
CLR_TEMPORAL = "#d53e4f"
CLR_RESIDUAL = "#d9d9d9"


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def panel_label(ax, text: str, x: float | None = None, y: float = 1.00,
                fontsize: float = 12) -> None:
    initial_x = x if x is not None else 0.0
    ha = "left" if x is not None else "center"
    txt_obj = ax.text(initial_x, y, text, transform=ax.transAxes,
                      fontsize=fontsize, fontweight="bold", va="top", ha=ha,
                      fontfamily="sans-serif")
    if not hasattr(ax, "_nw_panel_labels"):
        ax._nw_panel_labels = []
    ax._nw_panel_labels.append((txt_obj, x, y))


def align_panel_labels(fig: plt.Figure) -> None:
    """Align y-labels and panel labels dynamically across all subplots."""
    try:
        fig.canvas.draw()
    except Exception:
        pass

    from collections import defaultdict
    cols = defaultdict(list)
    for ax in fig.axes:
        pos = ax.get_position()
        cols[round(pos.x0, 2)].append(ax)

    for col_axs in cols.values():
        try:
            fig.align_ylabels(col_axs)
        except Exception:
            pass

    try:
        fig.canvas.draw()
    except Exception:
        pass

    for ax in fig.axes:
        if hasattr(ax, "_nw_panel_labels"):
            ylabel = ax.yaxis.get_label()
            if ylabel and ylabel.get_text():
                disp_x, _ = ylabel.get_transform().transform(ylabel.get_position())
                ax_x, _ = ax.transAxes.inverted().transform((disp_x, 0))
                for txt_obj, custom_x, custom_y in ax._nw_panel_labels:
                    if custom_x is None:
                        txt_obj.set_x(ax_x)
                        txt_obj.set_ha("center")


def save_fig(fig: plt.Figure, name: str, metrics: pd.DataFrame | None = None) -> None:
    """Save figure as Type-42 vector PDF + 300 DPI PNG (<800 KB) and optional metrics CSV."""
    align_panel_labels(fig)
    fig.savefig(FIG_OUT / f"{name}.pdf", format="pdf")
    from PIL import Image
    png_path = FIG_OUT / f"{name}.png"
    fig.savefig(png_path, format="png", dpi=300)
    with Image.open(png_path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im_rgb = bg
        else:
            im_rgb = im.convert("RGB")
        im_p = im_rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.NONE)
        im_p.save(png_path, format="PNG", optimize=True, dpi=(300, 300))
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
    if "wq" not in _cache:
        wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
        wq["year"]  = wq["datetime"].dt.year
        wq["month"] = wq["datetime"].dt.month
        wq["ym"]    = wq["datetime"].dt.to_period("M")
        wq = wq[wq["year"] <= 2022].copy()
        _cache["wq"] = wq
    return _cache["wq"]


def load_hydro_monthly() -> pd.DataFrame:
    if "q_monthly" not in _cache:
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
    if "master" not in _cache:
        _cache["master"] = pd.read_csv(MASTER_PATH)
    return _cache["master"]


def load_lulc() -> pd.DataFrame:
    if "lulc" not in _cache:
        _cache["lulc"] = pd.read_csv(LULC_PATH)
    return _cache["lulc"]


def load_bt(master: pd.DataFrame | None = None) -> pd.DataFrame:
    if master is None:
        master = load_master()
    return master[~master["stationName"].isin(MRC_STATIONS)].copy()


def cq_metrics(c: np.ndarray, q: np.ndarray):
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
