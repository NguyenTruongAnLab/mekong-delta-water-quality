#!/usr/bin/env python3
"""
CEE Supplementary Figure 12 — Fertilizer–Water Quality Spatial Gradient
=======================================================================
Cloned directly from `nw_sfig4_fertilizer_gradient.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import _cee_common as cee

ROOT = cee.ROOT
FAO_PATH = ROOT / "data" / "Fertilizers" / "FAO" / "Vietnam_FAOSTAT_data.csv"
WB_PATH = ROOT / "data" / "Fertilizers" / "Worldbank" / "Vietnam_Fertilizer_consumption.csv"
GSO_PADDY_PATH = ROOT / "data" / "Fertilizers" / "GSO_Vietnam" / "Planted area of paddy by province.csv"

MRC_STATIONS = cee.MRC_STATIONS
BT_STATIONS = ["BL01", "BL02", "BL03", "BL04",
               "CC01", "CC02", "CC03", "CC04", "CC05",
               "HL01", "HL02", "HL03", "HL04", "HL05",
               "MT01", "MT02", "MT03", "MT04", "MT05"]

WQ_COL = "NO3-N (mg/L)_annual"


def load_faostat() -> pd.DataFrame:
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
    result["NPK_kg_ha"] = (result["N_kg_ha"].fillna(0)
                           + result["P2O5_kg_ha"].fillna(0)
                           + result["K2O_kg_ha"].fillna(0))
    return result.sort_index()


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def load_gso_paddy() -> pd.DataFrame:
    df = pd.read_csv(GSO_PADDY_PATH)
    col0 = df.columns[0]
    year_cols = [c for c in df.columns[1:] if c.strip().isdigit()]
    vmd = df[df[col0].str.strip() == "Mekong River Delta"]
    result = {}
    for yr_str in year_cols:
        yr = int(yr_str.strip())
        vmd_val = vmd[yr_str].values[0] if len(vmd) > 0 else np.nan
        result[yr] = {"VMD_paddy_kha": _to_float(vmd_val)}
    return pd.DataFrame(result).T.rename_axis("Year").sort_index()


def load_worldbank() -> pd.DataFrame:
    df = pd.read_csv(WB_PATH)
    df.columns = ["Year", "WB_fert_kg_ha"]
    df["Year"] = df["Year"].astype(int)
    return df.set_index("Year").sort_index()


def fd_corr(x: pd.Series, y: pd.Series, min_n: int = 5):
    xd, yd = x.diff().dropna(), y.diff().dropna()
    idx = xd.index.intersection(yd.index)
    if len(idx) < min_n:
        return np.nan, np.nan
    return stats.pearsonr(xd.loc[idx], yd.loc[idx])


def main() -> None:
    cee.apply_style()
    fao = load_faostat()
    gso = load_gso_paddy()
    wb = load_worldbank()
    fert = fao.join(gso, how="outer").join(wb, how="outer")
    fert["N_x_VMD_paddy"] = (fert["N_kg_ha"] * fert["VMD_paddy_kha"]) / 1000.0

    master = cee.load_master()
    no3_stn = master[master["year"] <= 2022][["stationName", "year", WQ_COL]].dropna()

    drivers = [
        ("N_kg_ha", "N use (FAO, kg/ha)"),
        ("NPK_kg_ha", "NPK use (FAO, kg/ha)"),
        ("N_x_VMD_paddy", "N × VMD paddy"),
        ("VMD_paddy_kha", "Paddy area (GSO, kha)"),
        ("WB_fert_kg_ha", "Fertilizer (WB, kg/ha)"),
    ]

    biv_rows = []
    stations = MRC_STATIONS + BT_STATIONS
    for stn in stations:
        stn_df = no3_stn[no3_stn["stationName"] == stn].copy()
        if stn_df.empty:
            continue
        stn_df["Year"] = stn_df["year"].astype(int)
        stn_df = stn_df.set_index("Year").sort_index()

        for dcol, dlabel in drivers:
            df_merged = stn_df.join(fert[[dcol]], how="inner").dropna(subset=[WQ_COL, dcol])
            r, p = fd_corr(df_merged[dcol], df_merged[WQ_COL])
            biv_rows.append({
                "station": stn, "driver": dlabel, "r_fd": r, "p_fd": p,
                "n_years": len(df_merged), "network": "MRC" if stn in MRC_STATIONS else "Ben Tre"
            })

    biv = pd.DataFrame(biv_rows)
    if biv.empty:
        return

    pivot_df = biv.pivot(index="station", columns="driver", values="r_fd").reindex(stations)[[lbl for _, lbl in drivers]]
    pivot_p = biv.pivot(index="station", columns="driver", values="p_fd").reindex(stations)[[lbl for _, lbl in drivers]]

    fig, ax = plt.subplots(figsize=(180 * cee.MM, 95 * cee.MM))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eaeaea")
    im = ax.imshow(pivot_df.values, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(drivers)))
    ax.set_xticklabels([lbl for _, lbl in drivers], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(stations)))
    ax.set_yticklabels(stations, fontsize=7.5)

    for i in range(len(stations)):
        for j in range(len(drivers)):
            val = pivot_df.values[i, j]
            pval = pivot_p.values[i, j]
            if np.isfinite(val):
                sig = "*" if (np.isfinite(pval) and pval < 0.05) else ""
                ax.text(j, i, f"{val:+.2f}{sig}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 0.55 else "black")

    n_mrc = len([s for s in MRC_STATIONS if s in pivot_df.index])
    n_bt = len([s for s in BT_STATIONS if s in pivot_df.index])
    if n_mrc > 0 and n_bt > 0:
        ax.axhline(n_mrc - 0.5, color="black", lw=1.2)
        mrc_n_mean = pivot_df.iloc[:n_mrc]["N use (FAO, kg/ha)"].mean()
        bt_n_mean = pivot_df.iloc[n_mrc:]["N use (FAO, kg/ha)"].mean()
        mrc_mean = pivot_df.iloc[:n_mrc].mean().mean()
        bt_mean = pivot_df.iloc[n_mrc:].mean().mean()
        ax.text(-1.38, n_mrc / 2 - 0.5,
                f"MRC (upstream)\nFAO N " + r"$\Delta r$" + f" = {mrc_n_mean:+.2f}\n(all = {mrc_mean:+.2f})",
                ha="right", va="center", fontsize=6.8, fontweight="bold", clip_on=False)
        ax.text(-1.38, n_mrc + n_bt / 2 - 0.5,
                f"Ben Tre (downstream)\nFAO N " + r"$\Delta r$" + f" = {bt_n_mean:+.2f}\n(all = {bt_mean:+.2f})",
                ha="right", va="center", fontsize=6.8, fontweight="bold", clip_on=False)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"First-difference Pearson r ($\Delta r$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.21, bottom=0.22, right=0.88, top=0.97)
    cee.save_fig(fig, "CEE_SFig_12_Fertilizer_Gradient", metrics=biv.copy())


if __name__ == "__main__":
    main()
