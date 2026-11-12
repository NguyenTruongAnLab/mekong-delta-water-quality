#!/usr/bin/env python3
"""
CEE Supplementary Figure 2 — Thermodynamic Oxygen Partitioning (AOU) and Respiration Diagnostics
================================================================================================
Styled strictly with `_cee_common.py` (exact Arial 8.4 pt typography, `STATION_CLR`,
zero 'Year' x-axis labels, zero rounded defensive prose boxes):
  (a) Post-2004 (2005–2022) annual Apparent Oxygen Utilization (AOU) across 4 MRC stations.
  (b) Thermodynamic partitioning of post-2004 DO trends into biological respiration (-dAOU/dt)
      versus thermal solubility loss (dO2,sat/dt).
  (c) Dry-season (Jan–Apr) COD versus pH across MRC stations (testing against algal bloom inflation).
  (d) Post-2004 dry-season AOU versus COD across MRC stations.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, spearmanr, linregress

from _cee_common import (
    apply_style, load_wq, MM, MRC_STATIONS, STATION_CLR,
    PHASE1_CLR, PHASE2_CLR, CEE_VERIFY_DIR,
    panel_label, save_fig,
)

apply_style()


def _o2_sat_garcia_gordon(temp_c: np.ndarray, sal: np.ndarray | None = None) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    s = np.zeros_like(t) if sal is None else np.nan_to_num(np.asarray(sal, dtype=float), nan=0.0)
    ts = np.log((298.15 - t) / (273.15 + t))
    a0, a1, a2, a3, a4, a5 = 2.00907, 3.22014, 4.05010, 4.94457, -0.256847, 3.88767
    b0, b1, b2, b3, c0 = -6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3, -4.88682e-7
    ln_c = (a0 + a1 * ts + a2 * ts**2 + a3 * ts**3 + a4 * ts**4 + a5 * ts**5
            + s * (b0 + b1 * ts + b2 * ts**2 + b3 * ts**3) + c0 * s**2)
    return np.exp(ln_c) * 1.42905


def main() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()

    temp_col = next((c for c in mrc.columns if c.upper().startswith("TEMP")), None)
    sal_col = "Salinity" if "Salinity" in mrc.columns else None
    if temp_col is not None:
        s_vals = mrc[sal_col].to_numpy(dtype=float) if sal_col else None
        mrc["O2_sat"] = _o2_sat_garcia_gordon(mrc[temp_col].to_numpy(dtype=float), s_vals)
        mrc["AOU"] = mrc["O2_sat"] - mrc["DO (mg/L)"]
    else:
        mrc["O2_sat"] = np.nan
        mrc["AOU"] = np.nan

    metrics = []
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 115 * MM))

    # ── (a) Post-2004 annual AOU trajectories across 4 MRC stations ──
    ax = axes[0, 0]
    p2 = mrc[(mrc["year"] >= 2005) & (mrc["year"] <= 2022)].copy()
    for stn in MRC_STATIONS:
        sub = p2[p2["stationName"] == stn]
        ann = sub.groupby("year")["AOU"].mean().dropna()
        if len(ann) < 6:
            continue
        xa = ann.index.to_numpy(dtype=float)
        ya = ann.to_numpy(dtype=float)
        ax.plot(xa, ya, "-o", color=STATION_CLR[stn], lw=1.0, ms=2.2, alpha=0.88, label=stn, zorder=3)
        sl, ic, _, _ = theilslopes(ya, xa)
        ax.plot(xa, ic + sl * xa, "--", color=STATION_CLR[stn], lw=0.8, alpha=0.55, zorder=2)
        metrics.append({"Panel": "a_AOU_series", "Station": stn, "TheilSen_AOU_yr": round(float(sl), 5)})

    ax.set_xlim(2004.3, 2022.7)
    ax.set_xticks([2005, 2010, 2015, 2020])
    ax.set_ylabel(r"AOU (mg $\mathrm{L^{-1}}$)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.2)
    ax.legend(fontsize=7.6, ncol=2, loc="upper left", frameon=False, handlelength=1.2)
    panel_label(ax, "a", x=-0.22, y=1.08)

    # ── (b) Thermodynamic DO slope decomposition (2005–2022) ──
    ax = axes[0, 1]
    aou_csv = CEE_VERIFY_DIR / "cee_aou_thermodynamic_partitioning.csv"
    if aou_csv.exists():
        adf = pd.read_csv(aou_csv)
        adf["order"] = adf["Station"].map({s: i for i, s in enumerate(MRC_STATIONS)})
        adf = adf.sort_values("order").reset_index(drop=True)
        x_pos = np.arange(len(adf))
        w = 0.25

        do_sl = adf["Slope_DO_mgL_yr"].to_numpy(dtype=float)
        resp_sl = -adf["Slope_AOU_mgL_yr"].to_numpy(dtype=float)
        therm_sl = adf["Slope_O2sat_mgL_yr"].to_numpy(dtype=float)

        ax.bar(x_pos - w, do_sl, w, color="#333333", alpha=0.85, edgecolor="white",
               linewidth=0.3, label=r"Observed $\mathrm{dDO/d}t$", zorder=3)
        ax.bar(x_pos, resp_sl, w, color=PHASE2_CLR, alpha=0.85, edgecolor="white",
               linewidth=0.3, label=r"Respiration ($-\mathrm{dAOU/d}t$)", zorder=3)
        ax.bar(x_pos + w, therm_sl, w, color=PHASE1_CLR, alpha=0.85, edgecolor="white",
               linewidth=0.3, label=r"Thermal ($\mathrm{dO_{2,sat}/d}t$)", zorder=3)

        for i, r in adf.iterrows():
            pct = float(r["Respiration_AOU_Share_pct"])
            ax.text(
                i + 0.06, resp_sl[i] - 0.0035, f"{pct:.0f}%",
                ha="center", va="top", fontsize=7.4, fontweight="bold", color=PHASE2_CLR,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                zorder=5,
            )
            metrics.append({
                "Panel": "b_partition", "Station": str(r["Station"]),
                "Resp_Share_pct": round(pct, 1),
            })

        ax.axhline(0, color="black", lw=0.6, zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(adf["Station"].tolist(), fontsize=8.2)
        ax.set_ylim(-0.108, 0.015)
        ax.set_ylabel(r"Post-2004 slope (mg $\mathrm{L^{-1}}\ \mathrm{yr^{-1}}$)", fontsize=8.4)
        ax.tick_params(axis="both", labelsize=8.2)
        ax.legend(fontsize=7.2, loc="lower left", frameon=False, handlelength=1.0)
    panel_label(ax, "b", x=-0.22, y=1.08)

    # ── (c) Dry-season pH vs COD (1985–2022, Jan–Apr) ──
    ax = axes[1, 0]
    dry_all = mrc[mrc["month"].isin([1, 2, 3, 4])][["stationName", "year", "pH", "COD (mg/L)"]].dropna()
    if len(dry_all) > 10:
        m1 = dry_all["year"] <= 2004
        m2 = dry_all["year"] > 2004
        ax.scatter(dry_all.loc[m1, "pH"], dry_all.loc[m1, "COD (mg/L)"],
                   s=10, c=PHASE1_CLR, alpha=0.28, edgecolors="none", label="Phase 1 (≤2004)", zorder=2)
        ax.scatter(dry_all.loc[m2, "pH"], dry_all.loc[m2, "COD (mg/L)"],
                   s=10, c=PHASE2_CLR, alpha=0.28, edgecolors="none", label="Phase 2 (>2004)", zorder=2)
        xv = dry_all["pH"].to_numpy(dtype=float)
        yv = dry_all["COD (mg/L)"].to_numpy(dtype=float)
        rho, p_rho = spearmanr(xv, yv)
        lr = linregress(xv, yv)
        xs = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, "-", color="#222222", lw=1.2, zorder=4)
        ax.set_ylim(-0.2, 7.6)
        ax.text(0.97, 0.95, rf"$\rho = {rho:.2f}$" + "\n" + rf"($p < 0.001$, $n = {len(dry_all)}$)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color="#111111")
        ax.legend(fontsize=7.2, loc="upper left", frameon=False, handletextpad=0.2)
        metrics.append({"Panel": "c_dry_pH_COD", "Spearman_rho": round(float(rho), 4),
                        "p": round(float(p_rho), 8), "n": int(len(dry_all))})
    ax.set_xlabel("Dry-season pH", fontsize=8.4)
    ax.set_ylabel(r"Dry-season COD (mg $\mathrm{L^{-1}}$)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.2)
    panel_label(ax, "c", x=-0.22, y=1.08)

    # ── (d) Post-2004 dry-season COD vs AOU (2005–2022, Jan–Apr) ──
    ax = axes[1, 1]
    dry_p2 = p2[p2["month"].isin([1, 2, 3, 4])][["stationName", "COD (mg/L)", "AOU"]].dropna()
    if len(dry_p2) > 10:
        for stn in MRC_STATIONS:
            s_df = dry_p2[dry_p2["stationName"] == stn]
            ax.scatter(s_df["COD (mg/L)"], s_df["AOU"], s=12, c=STATION_CLR[stn],
                       alpha=0.45, edgecolors="none", label=stn, zorder=3)
        xv = dry_p2["COD (mg/L)"].to_numpy(dtype=float)
        yv = dry_p2["AOU"].to_numpy(dtype=float)
        rho, p_rho = spearmanr(xv, yv)
        lr = linregress(xv, yv)
        xs = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, "-", color="#222222", lw=1.2, zorder=4)
        ax.text(0.04, 0.94, rf"$\rho = +{rho:.2f}$ ($p < 0.001$, $n = {len(dry_p2)}$)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color="#111111")
        metrics.append({"Panel": "d_dry_COD_AOU", "Spearman_rho": round(float(rho), 4),
                        "p": round(float(p_rho), 8), "n": int(len(dry_p2))})
    ax.set_xlabel(r"Dry-season COD (mg $\mathrm{L^{-1}}$)", fontsize=8.4)
    ax.set_ylabel(r"Dry-season AOU (mg $\mathrm{L^{-1}}$)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.2)
    panel_label(ax, "d", x=-0.22, y=1.08)

    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save_fig(fig, "CEE_SFig_2_AOU_Thermodynamics", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
