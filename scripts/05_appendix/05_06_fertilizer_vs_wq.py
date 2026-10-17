#!/usr/bin/env python3
"""
Multivariate analysis of fertilizer application, paddy area, and water quality
==============================================================================

Evaluates relationships between national and delta-specific agricultural
nutrient inputs (FAOSTAT nitrogen, phosphate, and potash use; World Bank
fertilizer consumption; General Statistics Office of Viet Nam paddy planted
area), local land-use composition, hydrological forcing, and river water
quality across Mekong River Commission (MRC) and Ben Tre monitoring stations.

Analyses:
1. Delta-specific fertilizer proxy: FAO_N (kg/ha) * GSO_VMD_paddy (1000 ha)
2. Per-station bivariate correlations in levels and first differences
3. Partial first-difference correlations controlling for discharge and rainfall

Outputs:
- analysis/analysis_outputs/fertilizer_multivariate_metrics.csv
- analysis/analysis_outputs/fertilizer_partial_corr_metrics.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
FAO_PATH = ROOT / "data" / "Fertilizers" / "FAO" / "Vietnam_FAOSTAT_data.csv"
WB_PATH = (
    ROOT / "data" / "Fertilizers" / "Worldbank" / "Vietnam_Fertilizer_consumption.csv"
)
GSO_PADDY_PATH = (
    ROOT
    / "data"
    / "Fertilizers"
    / "GSO_Vietnam"
    / "Planted area of paddy by province.csv"
)
OUT_DIR = ROOT / "analysis" / "analysis_outputs"

BREAK_YEAR = 2004
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
BT_STATIONS = [
    "BL01",
    "BL02",
    "BL03",
    "BL04",
    "CC01",
    "CC02",
    "CC03",
    "CC04",
    "CC05",
    "HL01",
    "HL02",
    "HL03",
    "HL04",
    "HL05",
    "MT01",
    "MT02",
    "MT03",
    "MT04",
    "MT05",
]

WQ_PARAMS = [
    "NO3-N (mg/L)_annual",
    "COD (mg/L)_annual",
    "NH4-N (mg/L)_annual",
]
WQ_SHORT = {
    "NO3-N (mg/L)_annual": "NO₃⁻-N",
    "COD (mg/L)_annual": "COD",
    "NH4-N (mg/L)_annual": "NH₄⁺-N",
}


def load_faostat() -> pd.DataFrame:
    """Load FAOSTAT nutrient series as a wide annual table."""
    fao = pd.read_csv(FAO_PATH)
    fao["Year"] = fao["Year"].astype(int)
    rows = []
    for nutrient_item, prefix in [
        ("Nutrient nitrogen N (total)", "N"),
        ("Nutrient phosphate P2O5 (total)", "P2O5"),
        ("Nutrient potash K2O (total)", "K2O"),
    ]:
        for element, suffix in [
            ("Agricultural Use", "agri_use_t"),
            ("Use per area of cropland", "kg_ha"),
        ]:
            sub = fao[(fao["Item"] == nutrient_item) & (fao["Element"] == element)]
            sub = sub[["Year", "Value"]].copy()
            sub.columns = ["Year", f"{prefix}_{suffix}"]
            sub = sub.set_index("Year")
            rows.append(sub)
    result = rows[0]
    for r in rows[1:]:
        result = result.join(r, how="outer")
    result["N_P_ratio"] = result["N_agri_use_t"] / result["P2O5_agri_use_t"]
    result["NPK_kg_ha"] = (
        result["N_kg_ha"].fillna(0)
        + result["P2O5_kg_ha"].fillna(0)
        + result["K2O_kg_ha"].fillna(0)
    )
    return result.sort_index()


def _to_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def load_gso_paddy() -> pd.DataFrame:
    """Load GSO paddy planted area for the Mekong Delta and Ben Tre (1000 ha)."""
    df = pd.read_csv(GSO_PADDY_PATH)
    col0 = df.columns[0]
    year_cols = [c for c in df.columns[1:] if c.strip().isdigit()]
    vmd = df[df[col0].str.strip() == "Mekong River Delta"]
    bt = df[df[col0].str.strip() == "Ben Tre"]
    result = {}
    for yr_str in year_cols:
        yr = int(yr_str.strip())
        vmd_val = vmd[yr_str].values[0] if len(vmd) > 0 else np.nan
        bt_val = bt[yr_str].values[0] if len(bt) > 0 else np.nan
        result[yr] = {
            "VMD_paddy_kha": _to_float(vmd_val),
            "BenTre_paddy_kha": _to_float(bt_val),
        }
    return pd.DataFrame(result).T.rename_axis("Year").sort_index()


def load_worldbank() -> pd.DataFrame:
    df = pd.read_csv(WB_PATH)
    df.columns = ["Year", "WB_fert_kg_ha"]
    df["Year"] = df["Year"].astype(int)
    return df.set_index("Year").sort_index()


def load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_PATH)


def build_panel(
    master: pd.DataFrame, fao: pd.DataFrame, gso: pd.DataFrame, wb: pd.DataFrame
) -> pd.DataFrame:
    """Merge station-year water quality and covariates with fertilizer series."""
    keep_cols = [
        "stationName",
        "year",
        "NO3-N (mg/L)_annual",
        "COD (mg/L)_annual",
        "NH4-N (mg/L)_annual",
        "Rice_Cultivation_composition_pct",
        "Aquaculture_composition_pct",
        "Residential_Urban_composition_pct",
        "pop_density",
        "Rain_Annual_mm",
        "Q_branch_AnnualMean_m3s",
        "Q_branch_DrySeasonMean_m3s",
        "TR_CanTho_m",
        "network",
    ]
    available = [c for c in keep_cols if c in master.columns]
    panel = master[available].copy().rename(columns={"year": "Year"})
    panel = panel.merge(fao.reset_index(), on="Year", how="left")
    panel = panel.merge(gso.reset_index(), on="Year", how="left")
    panel = panel.merge(wb.reset_index(), on="Year", how="left")

    panel["N_delta_proxy"] = panel["N_kg_ha"] * panel["VMD_paddy_kha"]
    panel["P2O5_delta_proxy"] = panel["P2O5_kg_ha"] * panel["VMD_paddy_kha"]
    return panel


def first_diff(s: pd.Series) -> pd.Series:
    return s.diff().dropna()


def safe_corr(x: pd.Series, y: pd.Series, min_n: int = 5):
    mask = x.notna() & y.notna()
    xc, yc = x[mask], y[mask]
    if len(xc) < min_n:
        return np.nan, np.nan, len(xc)
    r, p = stats.pearsonr(xc, yc)
    return r, p, len(xc)


def fd_corr(x: pd.Series, y: pd.Series, min_n: int = 5):
    xd, yd = first_diff(x), first_diff(y)
    mask = xd.notna() & yd.notna()
    xd, yd = xd[mask], yd[mask]
    if len(xd) < min_n:
        return np.nan, np.nan, len(xd)
    r, p = stats.pearsonr(xd, yd)
    return r, p, len(xd)


def partial_corr_fd(
    x: pd.Series, y: pd.Series, controls: list[pd.Series], min_n: int = 6
):
    """Partial first-difference correlation controlling for discharge and rainfall."""
    xd = first_diff(x)
    yd = first_diff(y)
    cd = [first_diff(c) for c in controls]

    idx = xd.index
    for c in cd:
        idx = idx.intersection(c.index)
    idx = idx.intersection(yd.index)
    xd = xd.reindex(idx).dropna()
    yd = yd.reindex(idx)
    cd = [c.reindex(idx) for c in cd]

    valid = xd.notna() & yd.notna()
    for c in cd:
        valid = valid & c.notna()
    if valid.sum() < min_n:
        return np.nan, np.nan, int(valid.sum())

    xd_v = xd[valid].values
    yd_v = yd[valid].values
    C = np.column_stack([np.ones(int(valid.sum()))] + [c[valid].values for c in cd])

    try:
        beta_x = np.linalg.lstsq(C, xd_v, rcond=None)[0]
        beta_y = np.linalg.lstsq(C, yd_v, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan, int(valid.sum())
    res_x = xd_v - C @ beta_x
    res_y = yd_v - C @ beta_y

    r, p = stats.pearsonr(res_x, res_y)
    return r, p, int(valid.sum())


def run_bivariate_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    drivers = {
        "N_kg_ha": "FAO N (kg/ha)",
        "P2O5_kg_ha": "FAO P₂O₅ (kg/ha)",
        "K2O_kg_ha": "FAO K₂O (kg/ha)",
        "NPK_kg_ha": "FAO NPK (kg/ha)",
        "N_P_ratio": "N:P ratio",
        "N_delta_proxy": "N × VMD paddy",
        "WB_fert_kg_ha": "WB Fert (kg/ha)",
        "pop_density": "Population density",
        "Rice_Cultivation_composition_pct": "Rice LULC (%)",
        "Aquaculture_composition_pct": "Aquaculture LULC (%)",
        "Rain_Annual_mm": "Annual rainfall (mm)",
        "Q_branch_AnnualMean_m3s": "Discharge (m³/s)",
    }
    rows = []
    for stn in panel["stationName"].unique():
        s = panel[panel["stationName"] == stn].set_index("Year").sort_index()
        for drv_col, drv_label in drivers.items():
            if drv_col not in s.columns:
                continue
            for wq_col in WQ_PARAMS:
                if wq_col not in s.columns:
                    continue
                x, y = s[drv_col], s[wq_col]
                r_lev, p_lev, n_lev = safe_corr(x, y)
                r_fd, p_fd, n_fd = fd_corr(x, y)
                rows.append(
                    {
                        "station": stn,
                        "driver": drv_label,
                        "wq_param": WQ_SHORT[wq_col],
                        "N_levels": n_lev,
                        "r_levels": round(r_lev, 4) if not np.isnan(r_lev) else np.nan,
                        "p_levels": round(p_lev, 6) if not np.isnan(p_lev) else np.nan,
                        "N_fd": n_fd,
                        "r_fd": round(r_fd, 4) if not np.isnan(r_fd) else np.nan,
                        "p_fd": round(p_fd, 6) if not np.isnan(p_fd) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_partial_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    fert_cols = ["N_kg_ha", "NPK_kg_ha", "N_delta_proxy"]
    fert_labels = {
        "N_kg_ha": "FAO N (kg/ha)",
        "NPK_kg_ha": "FAO NPK (kg/ha)",
        "N_delta_proxy": "N × VMD paddy",
    }
    rows = []
    for stn in panel["stationName"].unique():
        s = panel[panel["stationName"] == stn].set_index("Year").sort_index()
        for fc in fert_cols:
            if fc not in s.columns:
                continue
            for wq_col in WQ_PARAMS:
                if wq_col not in s.columns:
                    continue
                controls = []
                for ctrl_col in ["Q_branch_AnnualMean_m3s", "Rain_Annual_mm"]:
                    if ctrl_col in s.columns and s[ctrl_col].notna().sum() > 5:
                        controls.append(s[ctrl_col])
                if not controls:
                    continue
                r_part, p_part, n_part = partial_corr_fd(s[fc], s[wq_col], controls)
                rows.append(
                    {
                        "station": stn,
                        "driver": fert_labels[fc],
                        "wq_param": WQ_SHORT[wq_col],
                        "controls": "Q + Rain",
                        "partial_r_fd": round(r_part, 4)
                        if not np.isnan(r_part)
                        else np.nan,
                        "partial_p_fd": round(p_part, 6)
                        if not np.isnan(p_part)
                        else np.nan,
                        "N": n_part,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fao = load_faostat()
    gso = load_gso_paddy()
    wb = load_worldbank()
    master = load_master()

    panel = build_panel(master, fao, gso, wb)
    print(
        f"Panel: {len(panel)} rows, {panel['stationName'].nunique()} stations, "
        f"years {panel['Year'].min()}-{panel['Year'].max()}"
    )

    biv = run_bivariate_analysis(panel)
    part = run_partial_analysis(panel)

    out_biv = OUT_DIR / "fertilizer_multivariate_metrics.csv"
    biv.to_csv(out_biv, index=False)
    print(f"Saved bivariate metrics: {out_biv} ({len(biv)} tests)")

    out_part = OUT_DIR / "fertilizer_partial_corr_metrics.csv"
    part.to_csv(out_part, index=False)
    print(f"Saved partial correlation metrics: {out_part} ({len(part)} tests)")

    print("\nFAOSTAT and GSO historical context:")
    print(
        f"  N use (kg/ha): 1985={fao.loc[1985, 'N_kg_ha']:.0f}, "
        f"2004={fao.loc[2004, 'N_kg_ha']:.0f}, "
        f"2022={fao.loc[2022, 'N_kg_ha']:.0f}"
    )
    print(
        f"  NPK use (kg/ha): 1985={fao.loc[1985, 'NPK_kg_ha']:.0f}, "
        f"2004={fao.loc[2004, 'NPK_kg_ha']:.0f}, "
        f"2022={fao.loc[2022, 'NPK_kg_ha']:.0f}"
    )
    print(
        f"  VMD paddy planted area (1000 ha): 1995={gso.loc[1995, 'VMD_paddy_kha']:.0f}, "
        f"2004={gso.loc[2004, 'VMD_paddy_kha']:.0f}, "
        f"2022={gso.loc[2022, 'VMD_paddy_kha']:.0f}"
    )


if __name__ == "__main__":
    main()
