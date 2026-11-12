#!/usr/bin/env python3
"""
CEE Supplementary Figure 11 — Rainfall Consistency Check (ERA5 vs CHIRPS)
=========================================================================
Cloned directly from `nw_sfig2_rainfall_crosscheck.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _cee_common import (
    apply_style, ROOT, MM, save_fig,
)

apply_style()

ERA5_FILE = ROOT / "data" / "hydrology" / "data" / "inputs" / "drivers" / "Mekong_Rainfall_ERA5_1985_2022.csv"
CHIRPS_FILE = ROOT / "data" / "hydrology" / "data" / "inputs" / "drivers" / "chirps_basin_rainfall_daily.csv"
BEN_TRE_GAUGE_MONTHLY_FILE = ROOT / "data" / "hydrology" / "data" / "inputs" / "drivers" / "Monthly_rain_gauge_BenTre.csv"

STUDY_START = "1985-01-01"
STUDY_END = "2022-12-31"


def panel_label(ax, text: str, x: float = -0.12, y: float = 1.06,
                fontsize: float = 12) -> None:
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            fontfamily="sans-serif")


def main() -> None:
    metrics = []

    if not CHIRPS_FILE.exists():
        print(f"  WARNING: CHIRPS file not found at {CHIRPS_FILE}, skipping CEE_SFig_11")
        return
    chirps_df = pd.read_csv(CHIRPS_FILE)
    chirps_df["Date"] = pd.to_datetime(chirps_df["Date"], errors="coerce")
    chirps_df = chirps_df.dropna(subset=["Date"]).set_index("Date").sort_index()
    chirps_daily = chirps_df["Precipitation_mm"].astype(float).resample("D").sum()
    chirps_daily = chirps_daily.loc[STUDY_START:STUDY_END]
    chirps_annual = chirps_daily.resample("YE").sum()

    bt_scatter: pd.DataFrame | None = None
    bt_scatter_r = np.nan
    bt_scatter_p = np.nan
    bt_scatter_bias = np.nan
    bt_scatter_slope = np.nan

    if BEN_TRE_GAUGE_MONTHLY_FILE.exists():
        gauge_df = pd.read_csv(BEN_TRE_GAUGE_MONTHLY_FILE, engine="python")
        gauge_df["Date"] = pd.to_datetime(gauge_df["Date"], format="%m/%Y", errors="coerce")
        gauge_df = gauge_df.dropna(subset=["Date"]).set_index("Date").sort_index()
        gauge_df = gauge_df.loc[STUDY_START:STUDY_END]

        g_idx = pd.DatetimeIndex(gauge_df.index)
        dim = g_idx.days_in_month
        bt_mm = pd.to_numeric(gauge_df["Ben Tre Station"], errors="coerce") * dim

        chirps_monthly = chirps_daily.resample("MS").sum()
        m = pd.DataFrame({
            "chirps_basin_mm_month": chirps_monthly,
            "ben_tre_gauge_mm_month": bt_mm.resample("MS").sum(min_count=1),
        }).dropna()

        bt_scatter = m[["chirps_basin_mm_month", "ben_tre_gauge_mm_month"]].dropna().copy()
        if len(bt_scatter) >= 12:
            bt_scatter_r, bt_scatter_p = stats.pearsonr(bt_scatter["chirps_basin_mm_month"], bt_scatter["ben_tre_gauge_mm_month"])
            bt_scatter_bias = float(bt_scatter["chirps_basin_mm_month"].mean() - bt_scatter["ben_tre_gauge_mm_month"].mean())
            x_bt = np.asarray(bt_scatter["ben_tre_gauge_mm_month"].values, dtype=float)
            y_bt = np.asarray(bt_scatter["chirps_basin_mm_month"].values, dtype=float)
            bt_scatter_slope = float(np.polyfit(x_bt, y_bt, deg=1)[0])

    era5_annual = None
    if ERA5_FILE.exists():
        era5_df = pd.read_csv(ERA5_FILE)
        era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], errors="coerce")
        era5_df = era5_df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        era5_daily = era5_df["precipitation_mm"].astype(float).resample("D").sum()
        era5_daily = era5_daily.loc[STUDY_START:STUDY_END]
        era5_annual = era5_daily.resample("YE").sum()

    if era5_annual is not None:
        idx = era5_annual.index.intersection(chirps_annual.index)
        era5_c = era5_annual.loc[idx]
        chirps_c = chirps_annual.loc[idx]
        years = idx.year.values.astype(int)
        corr, _ = stats.pearsonr(era5_c.values, chirps_c.values)
        bias = float(chirps_c.mean() - era5_c.mean())
    else:
        era5_c = None
        chirps_c = chirps_annual
        years = chirps_annual.index.year.values.astype(int)
        corr = 1.0
        bias = 0.0

    res_c = stats.theilslopes(chirps_c.values, years)
    tau_c, p_c = stats.kendalltau(years, chirps_c.values)
    c_trend = (res_c[0], res_c[1], tau_c, p_c)

    if era5_c is not None:
        res_e = stats.theilslopes(era5_c.values, years)
        tau_e, p_e = stats.kendalltau(years, era5_c.values)
        e_trend = (res_e[0], res_e[1], tau_e, p_e)
    else:
        e_trend = None

    fig = plt.figure(figsize=(110 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.18)

    ax_a = fig.add_subplot(gs[0])
    if era5_c is not None:
        ax_a.plot(years, era5_c.values, label="ERA5 reanalysis", color="blue", alpha=0.7, lw=1.2)
        ax_a.plot(years, e_trend[1] + e_trend[0] * years, "b--", alpha=0.5, lw=0.8)
    ax_a.plot(years, chirps_c.values, label="CHIRPS basin", color="green", alpha=0.7, lw=1.2)
    ax_a.plot(years, c_trend[1] + c_trend[0] * years, "g--", alpha=0.5, lw=0.8)

    ax_a.set_ylabel("Annual rainfall (mm)", fontsize=8.4)
    ax_a.set_xlim(1984, 2023)
    ax_a.legend(fontsize=8.4, loc="lower right", frameon=False, handletextpad=0.2)
    ax_a.grid(True, ls=":", alpha=0.3)
    ax_a.tick_params(axis="both", labelsize=8.4)
    if e_trend is not None:
        t_str = (
            f"r = {corr:.2f}\n"
            rf"CHIRPS: +{c_trend[0]*10:.1f} mm $\mathrm{{decade^{{-1}}}}$ (p = {c_trend[3]:.3f})" + "\n"
            rf"ERA5: +{e_trend[0]*10:.1f} mm $\mathrm{{decade^{{-1}}}}$ (p = {e_trend[3]:.3f})"
        )
    else:
        t_str = f"r = {corr:.2f}\n" + rf"CHIRPS: +{c_trend[0]*10:.1f} mm $\mathrm{{decade^{{-1}}}}$ (p = {c_trend[3]:.3f})"
    ax_a.text(0.05, 0.95, t_str,
              transform=ax_a.transAxes, fontsize=8.0, va="top",
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax_a.text(0.05, 0.05, rf"Mean bias: {bias:.0f} mm $\mathrm{{yr^{{-1}}}}$", transform=ax_a.transAxes, fontsize=8.0)
    ax_a.set_xticks(range(1985, 2025, 5))
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[1])
    if bt_scatter is not None:
        x = bt_scatter["ben_tre_gauge_mm_month"].values
        y = bt_scatter["chirps_basin_mm_month"].values
        months = bt_scatter.index.month
        mbt = ax_b.scatter(x, y, c=months, cmap="turbo", s=8, alpha=0.65, edgecolors="none")
        lim = float(np.max([np.max(x), np.max(y)]))
        ax_b.plot([0, lim], [0, lim], color="0.5", linewidth=1.0, linestyle=":")
        ax_b.plot(np.linspace(0, lim, 100), bt_scatter_bias + bt_scatter_slope * np.linspace(0, lim, 100), color="black", linewidth=1.2, alpha=0.85)
        ax_b.set_xlabel(r"Rain gauge (mm $\mathrm{month^{-1}}$)", fontsize=8.4)
        ax_b.set_ylabel(r"CHIRPS basin (mm $\mathrm{month^{-1}}$)", fontsize=8.4)
        ax_b.set_xlim(0, lim)
        ax_b.set_ylim(0, lim)
        ax_b.tick_params(axis="both", labelsize=8.4)
        ax_b.grid(True, ls=":", alpha=0.3)
        ax_b.text(0.03, 0.96, f"r = {bt_scatter_r:.2f}\n" + rf"Mean bias = +{bt_scatter_bias:.1f} mm $\mathrm{{month^{{-1}}}}$",
                  transform=ax_b.transAxes, fontsize=8.0, va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        metrics.append({"Figure": "CEE_SFig11", "Gauge": "BenTre", "Pearson_r": round(bt_scatter_r, 4), "p_value": round(bt_scatter_p, 6), "N": len(bt_scatter)})
        cbar = fig.colorbar(mbt, ax=ax_b, pad=0.02, fraction=0.05)
        cbar.ax.tick_params(labelsize=8.4)
        cbar.set_label("Month of year", fontsize=8.4)

    panel_label(ax_b, "b")
    metrics.append({"Figure": "CEE_SFig11", "Gauge": "CHIRPS_Basin_Annual", "Slope_mm_yr": round(c_trend[0], 4), "p_value": round(c_trend[3], 6), "N": len(years)})
    if e_trend:
        metrics.append({"Figure": "CEE_SFig11", "Gauge": "ERA5_Basin_Annual", "Slope_mm_yr": round(e_trend[0], 4), "p_value": round(e_trend[3], 6), "N": len(years)})

    save_fig(fig, "CEE_SFig_11_Rainfall_CrossCheck", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
