#!/usr/bin/env python3
"""
Supplementary Figure S7 — Fertilizer–Water Quality Spatial Gradient
====================================================================
Heatmap of first-difference correlations (Δr) between multiple drivers
and NO₃⁻-N at all MRC + Ben Tre stations.

Key finding: upstream MRC stations show no fertilizer signal (mean Δr ≈ −0.17),
while downstream Ben Tre stations show strong positive correlations
(mean Δr ≈ +0.52), with companion partial-Δr diagnostics exported.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Import shared publication style ─────────────────────────────────
import _nw_common as nw

ROOT = nw.ROOT
FAO_PATH = ROOT / "data" / "Fertilizers" / "FAO" / "Vietnam_FAOSTAT_data.csv"
WB_PATH = ROOT / "data" / "Fertilizers" / "Worldbank" / "Vietnam_Fertilizer_consumption.csv"
GSO_PADDY_PATH = ROOT / "data" / "Fertilizers" / "GSO_Vietnam" / "Planted area of paddy by province.csv"

BREAK_YEAR = nw.BREAK_YEAR
MRC_STATIONS = nw.MRC_STATIONS
BT_STATIONS = ["BL01", "BL02", "BL03", "BL04",
               "CC01", "CC02", "CC03", "CC04", "CC05",
               "HL01", "HL02", "HL03", "HL04", "HL05",
               "MT01", "MT02", "MT03", "MT04", "MT05"]

WQ_COL = "NO3-N (mg/L)_annual"


# ── Data loaders ────────────────────────────────────────────────────

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


# ── Statistics ──────────────────────────────────────────────────────

def fd_corr(x: pd.Series, y: pd.Series, min_n: int = 5):
    xd, yd = x.diff().dropna(), y.diff().dropna()
    idx = xd.index.intersection(yd.index)
    xd, yd = xd.reindex(idx), yd.reindex(idx)
    mask = xd.notna() & yd.notna()
    xd, yd = xd[mask], yd[mask]
    if len(xd) < min_n:
        return np.nan, np.nan, len(xd)
    r, p = stats.pearsonr(xd, yd)
    return r, p, len(xd)


def fd_partial_corr(
    x: pd.Series,
    y: pd.Series,
    controls: list[pd.Series],
    min_n: int = 5,
):
    """Partial first-difference Pearson correlation between x and y.

    Regresses Δx and Δy on control Δ-series (with intercept), then computes
    Pearson r between residuals.
    """
    xd = x.diff()
    yd = y.diff()
    aligned = pd.concat([xd.rename("x"), yd.rename("y")], axis=1)

    for i, c in enumerate(controls):
        aligned[f"c{i}"] = c.diff()

    aligned = aligned.dropna()
    if len(aligned) < min_n:
        return np.nan, np.nan, len(aligned)

    yv = aligned["y"].to_numpy(dtype=float)
    xv = aligned["x"].to_numpy(dtype=float)
    cv = aligned[[c for c in aligned.columns if c.startswith("c")]].to_numpy(dtype=float)

    # Add intercept for residualization
    X = np.column_stack([np.ones(len(aligned)), cv])

    beta_y, *_ = np.linalg.lstsq(X, yv, rcond=None)
    beta_x, *_ = np.linalg.lstsq(X, xv, rcond=None)
    y_res = yv - X @ beta_y
    x_res = xv - X @ beta_x

    if len(y_res) < min_n:
        return np.nan, np.nan, len(y_res)

    # Guard against near-constant residual vectors
    if np.isclose(np.std(y_res), 0.0) or np.isclose(np.std(x_res), 0.0):
        return np.nan, np.nan, len(y_res)

    r, p = stats.pearsonr(x_res, y_res)
    return r, p, len(y_res)


# ── Main figure ─────────────────────────────────────────────────────

def main():
    nw.apply_style()

    # Load data
    master = pd.read_csv(nw.MASTER_PATH)
    fao = load_faostat()
    gso = load_gso_paddy()
    wb = load_worldbank()

    # Merge
    keep_cols = [
        "stationName", "year", WQ_COL,
        "Rice_Cultivation_composition_pct",
        "Aquaculture_composition_pct",
        "pop_density",
        "Rain_Annual_mm",
        "Q_MyThuan_AnnualMean_m3s",
    ]
    available = [c for c in keep_cols if c in master.columns]
    panel = master[available].copy().rename(columns={"year": "Year"})
    panel = panel.merge(fao.reset_index(), on="Year", how="left")
    panel = panel.merge(gso.reset_index(), on="Year", how="left")
    panel = panel.merge(wb.reset_index(), on="Year", how="left")
    panel["N_delta_proxy"] = panel["N_kg_ha"] * panel["VMD_paddy_kha"]

    # Drivers to test
    drivers = {
        "N_kg_ha": r"N (kg ha$^{-1}$)",
        "P2O5_kg_ha": r"P$_2$O$_5$ (kg ha$^{-1}$)",
        "K2O_kg_ha": r"K$_2$O (kg ha$^{-1}$)",
        "NPK_kg_ha": r"NPK (kg ha$^{-1}$)",
        "N_P_ratio": "N:P ratio",
        "N_delta_proxy": r"N $\times$ VMD paddy",
        "WB_fert_kg_ha": r"WB total (kg ha$^{-1}$)",
        "pop_density": "Population density",
        "Rice_Cultivation_composition_pct": "Rice LULC (%)",
        "Aquaculture_composition_pct": "Aquaculture LULC (%)",
        "Rain_Annual_mm": "Annual rainfall",
        "Q_MyThuan_AnnualMean_m3s": r"Discharge (m$^3$ s$^{-1}$)",
    }

    # Compute Δr for every station × driver
    all_stations = MRC_STATIONS + sorted(BT_STATIONS)
    rows = []
    partial_rows = []
    for stn in all_stations:
        s = panel[panel["stationName"] == stn].set_index("Year").sort_index()
        if WQ_COL not in s.columns:
            continue
        for drv_col, drv_label in drivers.items():
            if drv_col not in s.columns:
                continue
            r, p, n = fd_corr(s[drv_col], s[WQ_COL])
            rows.append({"station": stn, "driver": drv_label,
                         "r_fd": r, "p_fd": p, "n": n})

        # Reproducible partial Δr for key fertilizer indicators
        c1 = "Rain_Annual_mm"
        c2 = "Q_MyThuan_AnnualMean_m3s"
        key_drivers = {
            "N_kg_ha": r"N (kg ha$^{-1}$)",
            "NPK_kg_ha": r"NPK (kg ha$^{-1}$)",
            "N_delta_proxy": r"N $\times$ VMD paddy",
        }
        if c1 in s.columns and c2 in s.columns:
            for drv_col, drv_label in key_drivers.items():
                if drv_col not in s.columns:
                    continue
                r_p, p_p, n_p = fd_partial_corr(
                    s[drv_col],
                    s[WQ_COL],
                    controls=[s[c1], s[c2]],
                )
                partial_rows.append({
                    "station": stn,
                    "driver": drv_label,
                    "r_fd_partial": r_p,
                    "p_fd_partial": p_p,
                    "n_partial": n_p,
                    "controls": "Rain_Annual_mm + Q_MyThuan_AnnualMean_m3s",
                })

    biv = pd.DataFrame(rows)
    driver_order = list(drivers.values())
    driver_order = [d for d in driver_order if d in biv["driver"].unique()]
    station_order = [s for s in all_stations if s in biv["station"].unique()]

    piv_r = biv.pivot_table(index="station", columns="driver",
                            values="r_fd", aggfunc="first")
    piv_p = biv.pivot_table(index="station", columns="driver",
                            values="p_fd", aggfunc="first")
    piv_r = piv_r.reindex(index=station_order, columns=driver_order)
    piv_p = piv_p.reindex(index=station_order, columns=driver_order)

    # Drop stations that have no data (all NaN rows)
    has_data = piv_r.notna().any(axis=1)
    piv_r = piv_r.loc[has_data]
    piv_p = piv_p.reindex(index=piv_r.index, columns=driver_order)
    station_order = list(piv_r.index)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(180 * nw.MM, 120 * nw.MM))

    im = ax.imshow(piv_r.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(driver_order)))
    ax.set_xticklabels(driver_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(station_order)))
    ax.set_yticklabels(station_order, fontsize=7.5)

    # Compute group means for labels
    mrc_mean = biv[(biv["station"].isin(MRC_STATIONS)) &
                   (biv["driver"] == r"N (kg ha$^{-1}$)")]["r_fd"].mean()
    bt_mean = biv[(biv["station"].isin(BT_STATIONS)) &
                  (biv["driver"] == r"N (kg ha$^{-1}$)")]["r_fd"].mean()

    # Separator between MRC and Ben Tre
    n_mrc = sum(1 for s in station_order if s in MRC_STATIONS)
    n_bt = len(station_order) - n_mrc
    if 0 < n_mrc < len(station_order):
        ax.axhline(n_mrc - 0.5, color="black", lw=2)
        ax.text(-3.0, n_mrc / 2 - 0.5,
                f"MRC\n(upstream)\nmean "
                + r"$\Delta r$"
                + f" = {mrc_mean:+.2f}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                rotation=90, clip_on=False)
        ax.text(-3.0, n_mrc + n_bt / 2 - 0.5,
                f"Ben Tre\n(downstream)\nmean "
                + r"$\Delta r$"
                + f" = {bt_mean:+.2f}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                rotation=90, clip_on=False)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"First-difference Pearson r ($\Delta r$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.21, bottom=0.22, right=0.88, top=0.97)

    # Save metrics
    metrics = biv.copy()
    nw.save_fig(fig, "Figure_S7_Fertilizer_Gradient", metrics=metrics)

    # Save partial-correlation companion metrics for SI narrative traceability
    partial_df = pd.DataFrame(partial_rows)
    if not partial_df.empty:
        partial_summary = (
            partial_df.groupby("driver", as_index=False)
            .apply(lambda g: pd.Series({
                "mrc_mean_partial": g[g["station"].isin(MRC_STATIONS)]["r_fd_partial"].mean(),
                "bt_mean_partial": g[g["station"].isin(BT_STATIONS)]["r_fd_partial"].mean(),
                "mrc_n": g[g["station"].isin(MRC_STATIONS)]["r_fd_partial"].notna().sum(),
                "bt_n": g[g["station"].isin(BT_STATIONS)]["r_fd_partial"].notna().sum(),
            }))
            .reset_index(drop=True)
        )

        partial_out = nw.METRICS_OUT / "Figure_S7_Fertilizer_Gradient_partial_metrics.csv"
        partial_df.to_csv(partial_out, index=False)
        partial_summary_out = nw.METRICS_OUT / "Figure_S7_Fertilizer_Gradient_partial_summary.csv"
        partial_summary.to_csv(partial_summary_out, index=False)
        print(f"    partial metrics → {partial_out}")
        print(f"    partial summary → {partial_summary_out}")

    print("Done.")


if __name__ == "__main__":
    main()
