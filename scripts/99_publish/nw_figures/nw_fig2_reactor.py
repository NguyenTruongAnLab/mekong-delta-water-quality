#!/usr/bin/env python3
"""
Figure 2 — Biogeochemical Reactor
===================================
(a,b) Downstream enrichment gradients (NO3-N, COD; Phase 1 vs Phase 2)
(c,d) Q-normalised mass balance (NO3-N, COD)

Exported metrics CSV:
  - Station, Parameter, trend slope, p, significance
  - Non-conservative fraction per parameter per phase
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau, mannwhitneyu

from _nw_common import (
    apply_style, load_wq, MM, MRC_STATIONS,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR, MB_CSV,
    panel_label, save_fig,
)

apply_style()


def figure_2() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()

    metrics_rows = []

    # Compute a complete parameter catalogue of phase-trend metrics (for narrative selection later)
    stations_ordered = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
    candidate_params = [
        "COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)",
        "TN (mg/L)", "TP (mg/L)", "TSS (mg/L)", "pH"
    ]
    all_params = [p for p in candidate_params if p in mrc.columns]

    for pcol in all_params:
        for stn in stations_ordered:
            sub = mrc[mrc["stationName"] == stn]
            annual = sub.groupby("year")[pcol].mean().dropna()
            if len(annual) < 5:
                continue

            x = annual.index.to_numpy(dtype=float)
            y = annual.to_numpy(dtype=float)

            for phase_label, mask in [("Phase1", x <= BREAK_YEAR), ("Phase2", x > BREAK_YEAR)]:
                if np.sum(mask) <= 5:
                    continue
                slope = float(np.asarray(theilslopes(y[mask], x[mask]), dtype=float)[0])
                pval = float(np.asarray(kendalltau(x[mask], y[mask]), dtype=float)[1])
                metrics_rows.append({
                    "Panel": "meta_allparams_phase_slopes",
                    "Station": stn,
                    "Parameter": pcol.split(" (")[0],
                    "Phase": phase_label,
                    "Slope": round(slope, 6),
                    "Kendall_p": round(pval, 6),
                    "n_years": int(np.sum(mask)),
                })

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 100 * MM))

    # ── (a) Downstream NO3 enrichment gradient: Phase 1 vs Phase 2 ──
    ax = axes[0, 0]
    x_pos = np.arange(len(stations_ordered))

    # NO3 with Phase 1 vs Phase 2 comparison
    tn_col = "NO3-N (mg/L)"
    phase1_slopes, phase2_slopes = [], []
    phase1_pvals, phase2_pvals = [], []

    for stn in stations_ordered:
        sub = mrc[mrc["stationName"] == stn]
        if tn_col not in sub.columns:
            phase1_slopes.append(0); phase2_slopes.append(0)
            phase1_pvals.append(1); phase2_pvals.append(1)
            continue
        annual = sub.groupby("year")[tn_col].mean().dropna()
        if len(annual) < 5:
            phase1_slopes.append(0); phase2_slopes.append(0)
            phase1_pvals.append(1); phase2_pvals.append(1)
            continue

        x = annual.index.to_numpy(dtype=float)
        y = annual.to_numpy(dtype=float)

        # Phase 1 (≤2004) trend
        m1 = x <= BREAK_YEAR
        if m1.sum() > 5:
            slope1 = float(np.asarray(theilslopes(y[m1], x[m1]), dtype=float)[0])
            p1 = float(np.asarray(kendalltau(x[m1], y[m1]), dtype=float)[1])
        else:
            slope1, p1 = 0.0, 1.0

        # Phase 2 (>2004) trend
        m2 = x > BREAK_YEAR
        if m2.sum() > 5:
            slope2 = float(np.asarray(theilslopes(y[m2], x[m2]), dtype=float)[0])
            p2 = float(np.asarray(kendalltau(x[m2], y[m2]), dtype=float)[1])
        else:
            slope2, p2 = 0.0, 1.0

        phase1_slopes.append(slope1 * 1000)
        phase2_slopes.append(slope2 * 1000)
        phase1_pvals.append(p1)
        phase2_pvals.append(p2)

        metrics_rows.append({
            "Panel": "a_NO3_enrichment",
            "Station": stn, "Parameter": "NO3-N",
            "Phase1_Slope_x1000": round(slope1 * 1000, 4),
            "Phase1_p": round(p1, 6),
            "Phase2_Slope_x1000": round(slope2 * 1000, 4),
            "Phase2_p": round(p2, 6),
            "n_years": int(len(annual)),
        })

    width = 0.35
    ax.bar(x_pos - width / 2, phase1_slopes, width,
           color=PHASE1_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=3)
    ax.bar(x_pos + width / 2, phase2_slopes, width,
           color=PHASE2_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 2 (>{BREAK_YEAR})", zorder=3)

    # Add significance markers
    for i, (s1, p1, s2, p2) in enumerate(zip(phase1_slopes, phase1_pvals, phase2_slopes, phase2_pvals)):
        for s, p, offset in [(s1, p1, -width / 2), (s2, p2, width / 2)]:
            if p < 0.001:   sig = "***"
            elif p < 0.01:  sig = "**"
            elif p < 0.05:  sig = "*"
            else:           sig = ""
            if sig:
                y_offset = 0.5 if s >= 0 else -0.7
                ax.text(i + offset, s + y_offset, sig,
                        ha="center", va="bottom" if s >= 0 else "top",
                        fontsize=5, fontweight="bold", color="#333333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stations_ordered, fontsize=6)
    ax.set_ylabel(r"NO$_3^-$ trend ($\times 10^{-3}$ mg $\mathrm{L^{-1}\,yr^{-1}}$)", fontsize=6)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    ax.legend(fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none",
              loc="upper left", ncol=1, handlelength=1.0, columnspacing=0.6)
    ax.set_title(r"(a) NO$_3^-$ enrichment (source activation)", fontsize=7, fontweight="bold", pad=2)

    # ── (b) Downstream COD enrichment gradient: Phase 1 vs Phase 2 ──
    ax = axes[1, 0]
    cod_col = "COD (mg/L)"
    phase1_slopes, phase2_slopes = [], []
    phase1_pvals, phase2_pvals = [], []

    for stn in stations_ordered:
        sub = mrc[mrc["stationName"] == stn]
        if cod_col not in sub.columns:
            phase1_slopes.append(0); phase2_slopes.append(0)
            phase1_pvals.append(1); phase2_pvals.append(1)
            continue
        annual = sub.groupby("year")[cod_col].mean().dropna()
        if len(annual) < 5:
            phase1_slopes.append(0); phase2_slopes.append(0)
            phase1_pvals.append(1); phase2_pvals.append(1)
            continue

        x = annual.index.to_numpy(dtype=float)
        y = annual.to_numpy(dtype=float)

        m1 = x <= BREAK_YEAR
        if m1.sum() > 5:
            slope1 = float(np.asarray(theilslopes(y[m1], x[m1]), dtype=float)[0])
            p1 = float(np.asarray(kendalltau(x[m1], y[m1]), dtype=float)[1])
        else:
            slope1, p1 = 0.0, 1.0

        m2 = x > BREAK_YEAR
        if m2.sum() > 5:
            slope2 = float(np.asarray(theilslopes(y[m2], x[m2]), dtype=float)[0])
            p2 = float(np.asarray(kendalltau(x[m2], y[m2]), dtype=float)[1])
        else:
            slope2, p2 = 0.0, 1.0

        phase1_slopes.append(slope1 * 1000)
        phase2_slopes.append(slope2 * 1000)
        phase1_pvals.append(p1)
        phase2_pvals.append(p2)

        metrics_rows.append({
            "Panel": "b_COD_enrichment",
            "Station": stn, "Parameter": "COD",
            "Phase1_Slope_x1000": round(slope1 * 1000, 4),
            "Phase1_p": round(p1, 6),
            "Phase2_Slope_x1000": round(slope2 * 1000, 4),
            "Phase2_p": round(p2, 6),
            "n_years": int(len(annual)),
        })

    ax.bar(x_pos - width / 2, phase1_slopes, width,
           color=PHASE1_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=3)
    ax.bar(x_pos + width / 2, phase2_slopes, width,
           color=PHASE2_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 2 (>{BREAK_YEAR})", zorder=3)

    for i, (s1, p1, s2, p2) in enumerate(zip(phase1_slopes, phase1_pvals, phase2_slopes, phase2_pvals)):
        for s, p, offset in [(s1, p1, -width / 2), (s2, p2, width / 2)]:
            if p < 0.001:   sig = "***"
            elif p < 0.01:  sig = "**"
            elif p < 0.05:  sig = "*"
            else:           sig = ""
            if sig:
                y_offset = 2.0 if s >= 0 else -3.0
                ax.text(i + offset, s + y_offset, sig,
                        ha="center", va="bottom" if s >= 0 else "top",
                        fontsize=5, fontweight="bold", color="#333333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stations_ordered, fontsize=6)
    ax.set_ylabel(r"COD trend ($\times 10^{-3}$ mg $\mathrm{L^{-1}\,yr^{-1}}$)", fontsize=6)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    ax.set_title("(b) COD enrichment", fontsize=7, fontweight="bold", pad=2)

    # ── (c) Q-normalised NO3 mass balance ──
    ax = axes[0, 1]

    # Plot NO3 only in panel c
    tn_params = [("NO3-N",  "#756bb1", "-", r"NO$_3^-$ (→ source activation)")]
    slope_tn = np.nan
    qeff_tn = np.nan

    if MB_CSV.exists():
        mb = pd.read_csv(MB_CSV)
        for plabel, clr, ls, legend_label in tn_params:
            sub = mb[mb["parameter"] == plabel].dropna(subset=["nonconservative_pct"])
            if len(sub) < 5:
                continue
            annual_df = (
                sub.groupby("year")
                .agg(
                    nonconservative_pct=("nonconservative_pct", "mean"),
                    L_expected=("L_expected_MyThuan_tonnes", "sum"),
                    L_tanchau=("L_TanChau_tonnes", "sum"),
                )
                .dropna(subset=["nonconservative_pct"])
            )
            annual_df = annual_df[annual_df["L_tanchau"] > 0]
            annual_df["Qratio_eff"] = annual_df["L_expected"] / annual_df["L_tanchau"]
            annual = annual_df["nonconservative_pct"]

            # Raw interannual variability (faded background texture)
            ax.plot(annual.index, annual.values, "o-", color=clr,
                    lw=0.9, ms=2.2, markeredgewidth=0, alpha=0.35,
                    label=legend_label, zorder=2)

            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            ts = np.asarray(theilslopes(ya, xa), dtype=float)
            slope, intercept = float(ts[0]), float(ts[1])
            p = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            slope_tn = slope

            p1_mask = xa <= BREAK_YEAR
            p2_mask = xa > BREAK_YEAR
            p1_mean = float(np.mean(ya[p1_mask])) if p1_mask.any() else np.nan
            p2_mean = float(np.mean(ya[p2_mask])) if p2_mask.any() else np.nan
            qeff_tn = float(np.nanmean(annual_df["Qratio_eff"])) if len(annual_df) else np.nan
            qeff_p1 = float(np.nanmean(annual_df.loc[p1_mask, "Qratio_eff"])) if p1_mask.any() else np.nan
            qeff_p2 = float(np.nanmean(annual_df.loc[p2_mask, "Qratio_eff"])) if p2_mask.any() else np.nan

            mw_p = np.nan
            if p1_mask.any() and p2_mask.any():
                try:
                    mw_p = float(mannwhitneyu(ya[p1_mask], ya[p2_mask], alternative="two-sided").pvalue)
                except Exception:
                    mw_p = np.nan

            metrics_rows.append({
                "Panel": "c_NO3_mass_balance",
                "Station": "TanChau→MyThuan", "Parameter": plabel,
                "Phase1_NCF_pct": round(p1_mean, 2),
                "Phase2_NCF_pct": round(p2_mean, 2),
                "Delta_NCF_pct": round((p2_mean - p1_mean), 2) if np.isfinite(p1_mean) and np.isfinite(p2_mean) else None,
                "Trend_slope_pct_yr": round(slope, 4),
                "Kendall_p": round(p, 6),
                "PhaseShift_MW_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "Qratio_eff_mean": round(qeff_tn, 4) if np.isfinite(qeff_tn) else None,
                "Qratio_eff_phase1": round(qeff_p1, 4) if np.isfinite(qeff_p1) else None,
                "Qratio_eff_phase2": round(qeff_p2, 4) if np.isfinite(qeff_p2) else None,
                "n_years": int(len(annual)),
            })

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    ax.text(BREAK_YEAR + 0.75, 0.97, "Regime shift",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="left", va="top", fontsize=5.0, color="#525252")
    y0, y1 = ax.get_ylim()
    ax.axhspan(min(0, y0), 0, color="#deebf7", alpha=0.20, zorder=0)
    ax.axhspan(0, max(0, y1), color="#fee0d2", alpha=0.20, zorder=0)

    # Annotation for NO3 panel
    ax.text(1.02, 0.5, r"source activation ↑", fontsize=5, color="#756bb1",
            rotation=90, va="center", ha="left", transform=ax.transAxes, clip_on=False)
    ax.text(0.03, 0.92, "Net generation", transform=ax.transAxes, fontsize=5.8,
            color="#7f0000", style="italic", fontweight="bold")
    ax.text(0.03, 0.08, "Net consumption", transform=ax.transAxes, fontsize=5.8,
            color="#084081", style="italic", fontweight="bold")
    # Do not emphasize linear slope in panel c; use phase means to show step change.
    # (No flow-partition inset annotation in panel c; kept in metrics CSV only)
    ax.set_ylabel("Non-conservative fraction (%)", fontsize=6)
    ax.set_title(r"(c) NO$_3^-$ mass balance", fontsize=7, fontweight="bold", pad=2)
    ax.legend(fontsize=5, frameon=True, framealpha=0.9, edgecolor="none",
              loc="upper right", ncol=1, handlelength=1.2)
    ax.tick_params(axis="x", labelbottom=True)

    # ── (d) Q-normalised COD mass balance ──
    ax = axes[1, 1]
    
    # Plot COD only
    cod_params = [("COD", "#d62728", "-", "COD (→ loss of consumption)")]
    slope_cod = np.nan
    qeff_cod = np.nan
    
    if MB_CSV.exists():
        mb = pd.read_csv(MB_CSV)
        for plabel, clr, ls, legend_label in cod_params:
            sub = mb[mb["parameter"] == plabel].dropna(subset=["nonconservative_pct"])
            if len(sub) < 5:
                continue
            annual_df = (
                sub.groupby("year")
                .agg(
                    nonconservative_pct=("nonconservative_pct", "mean"),
                    L_expected=("L_expected_MyThuan_tonnes", "sum"),
                    L_tanchau=("L_TanChau_tonnes", "sum"),
                )
                .dropna(subset=["nonconservative_pct"])
            )
            annual_df = annual_df[annual_df["L_tanchau"] > 0]
            annual_df["Qratio_eff"] = annual_df["L_expected"] / annual_df["L_tanchau"]
            annual = annual_df["nonconservative_pct"]

            # Raw interannual variability (faded background texture)
            ax.plot(annual.index, annual.values, "o-", color=clr,
                    lw=0.9, ms=2.2, markeredgewidth=0, alpha=0.35,
                    label=legend_label, zorder=2)

            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            ts = np.asarray(theilslopes(ya, xa), dtype=float)
            slope, intercept = float(ts[0]), float(ts[1])
            p = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            slope_cod = slope

            p1_mask = xa <= BREAK_YEAR
            p2_mask = xa > BREAK_YEAR
            p1_mean = float(np.mean(ya[p1_mask])) if p1_mask.any() else np.nan
            p2_mean = float(np.mean(ya[p2_mask])) if p2_mask.any() else np.nan
            qeff_cod = float(np.nanmean(annual_df["Qratio_eff"])) if len(annual_df) else np.nan
            qeff_p1 = float(np.nanmean(annual_df.loc[p1_mask, "Qratio_eff"])) if p1_mask.any() else np.nan
            qeff_p2 = float(np.nanmean(annual_df.loc[p2_mask, "Qratio_eff"])) if p2_mask.any() else np.nan

            mw_p = np.nan
            if p1_mask.any() and p2_mask.any():
                try:
                    mw_p = float(mannwhitneyu(ya[p1_mask], ya[p2_mask], alternative="two-sided").pvalue)
                except Exception:
                    mw_p = np.nan

            metrics_rows.append({
                "Panel": "d_COD_mass_balance",
                "Station": "TanChau→MyThuan", "Parameter": plabel,
                "Phase1_NCF_pct": round(p1_mean, 2),
                "Phase2_NCF_pct": round(p2_mean, 2),
                "Delta_NCF_pct": round((p2_mean - p1_mean), 2) if np.isfinite(p1_mean) and np.isfinite(p2_mean) else None,
                "Trend_slope_pct_yr": round(slope, 4),
                "Kendall_p": round(p, 6),
                "PhaseShift_MW_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "Qratio_eff_mean": round(qeff_cod, 4) if np.isfinite(qeff_cod) else None,
                "Qratio_eff_phase1": round(qeff_p1, 4) if np.isfinite(qeff_p1) else None,
                "Qratio_eff_phase2": round(qeff_p2, 4) if np.isfinite(qeff_p2) else None,
                "n_years": int(len(annual)),
            })

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    ax.text(BREAK_YEAR + 0.75, 0.97, "Regime shift",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="left", va="top", fontsize=5.0, color="#525252")
    y0, y1 = ax.get_ylim()
    ax.axhspan(min(0, y0), 0, color="#deebf7", alpha=0.20, zorder=0)
    ax.axhspan(0, max(0, y1), color="#fee0d2", alpha=0.20, zorder=0)
    ax.set_xlabel("", fontsize=6)
    ax.set_ylabel("Non-conservative fraction (%)", fontsize=6)
    ax.text(0.03, 0.92, "Net generation", transform=ax.transAxes, fontsize=5.8,
            color="#7f0000", style="italic", fontweight="bold")
    ax.text(0.03, 0.08, "Net consumption", transform=ax.transAxes, fontsize=5.8,
            color="#084081", style="italic", fontweight="bold")
    ax.text(1.02, 0.5, r"generation $\uparrow$", fontsize=5, color="#7f0000",
            rotation=90, va="center", ha="left", transform=ax.transAxes, clip_on=False)
    # Do not emphasize linear slope in panel d; use phase means to show sink collapse.
    ax.set_title("(d) COD mass balance", fontsize=7, fontweight="bold", pad=2)
    ax.legend(fontsize=5, frameon=True, framealpha=0.9, edgecolor="none",
              loc="upper right", ncol=1, handlelength=1.2)

    # Complete mass-balance catalogue (all parameters present in table)
    if MB_CSV.exists():
        mb = pd.read_csv(MB_CSV)
        for plabel in sorted(mb["parameter"].dropna().unique()):
            sub = mb[mb["parameter"] == plabel].dropna(subset=["nonconservative_pct"])
            if len(sub) < 5:
                continue
            annual_df = (
                sub.groupby("year")
                .agg(
                    nonconservative_pct=("nonconservative_pct", "mean"),
                    L_expected=("L_expected_MyThuan_tonnes", "sum"),
                    L_tanchau=("L_TanChau_tonnes", "sum"),
                )
                .dropna(subset=["nonconservative_pct"])
            )
            annual_df = annual_df[annual_df["L_tanchau"] > 0]
            annual_df["Qratio_eff"] = annual_df["L_expected"] / annual_df["L_tanchau"]
            annual = annual_df["nonconservative_pct"]
            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            slope = float(np.asarray(theilslopes(ya, xa), dtype=float)[0])
            p = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            p1_mask = xa <= BREAK_YEAR
            p2_mask = xa > BREAK_YEAR
            p1_mean = float(np.mean(ya[p1_mask])) if p1_mask.any() else np.nan
            p2_mean = float(np.mean(ya[p2_mask])) if p2_mask.any() else np.nan
            qeff = float(np.nanmean(annual_df["Qratio_eff"])) if len(annual_df) else np.nan
            qeff_p1 = float(np.nanmean(annual_df.loc[p1_mask, "Qratio_eff"])) if p1_mask.any() else np.nan
            qeff_p2 = float(np.nanmean(annual_df.loc[p2_mask, "Qratio_eff"])) if p2_mask.any() else np.nan
            mw_p = np.nan
            if p1_mask.any() and p2_mask.any():
                try:
                    mw_p = float(mannwhitneyu(ya[p1_mask], ya[p2_mask], alternative="two-sided").pvalue)
                except Exception:
                    mw_p = np.nan
            metrics_rows.append({
                "Panel": "meta_allparams_mass_balance",
                "Station": "TanChau→MyThuan",
                "Parameter": plabel,
                "Phase1_NCF_pct": round(p1_mean, 2),
                "Phase2_NCF_pct": round(p2_mean, 2),
                "Delta_NCF_pct": round((p2_mean - p1_mean), 2) if np.isfinite(p1_mean) and np.isfinite(p2_mean) else None,
                "Trend_slope_pct_yr": round(slope, 4),
                "Kendall_p": round(p, 6),
                "PhaseShift_MW_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "Qratio_eff_mean": round(qeff, 4) if np.isfinite(qeff) else None,
                "Qratio_eff_phase1": round(qeff_p1, 4) if np.isfinite(qeff_p1) else None,
                "Qratio_eff_phase2": round(qeff_p2, 4) if np.isfinite(qeff_p2) else None,
                "n_years": int(len(annual)),
            })

    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save_fig(fig, "Figure_2_Biogeochemical_Reactor", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_2()
