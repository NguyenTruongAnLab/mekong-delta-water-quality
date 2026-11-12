#!/usr/bin/env python3
"""
CEE Figure 2 — Mainstem Biogeochemical Reactor
==============================================
Cloned directly from `nw_fig2_reactor.py` to preserve 100% of the tuned 2x2
layout, palettes, 90-degree year ticks, zero 'Year' x-labels, and zero boxes.
Surgical upgrade: uses harmonized Seasonal Mann-Kendall (`SMK_p`) significance
stars in panels (a,b) matching the verified monthly trend tests (`3/4` post-2004
COD stations `***`, My Tho n.s.; `4/4` post-2004 NO3-N stations `***`).
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau, mannwhitneyu

from _cee_common import (
    apply_style, load_wq, MM,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR, MB_CSV, CEE_VERIFY_DIR,
    panel_label, save_fig,
)

apply_style()


def _load_smk_lookup() -> dict[tuple[str, str, str], dict]:
    smk_csv = CEE_VERIFY_DIR / "cee_harmonized_smk_annual_trends.csv"
    lookup: dict[tuple[str, str, str], dict] = {}
    if smk_csv.exists():
        df = pd.read_csv(smk_csv)
        for _, r in df.iterrows():
            ph = str(r["Phase"])
            ph_short = "Phase1" if ph.startswith("Phase1") else ("Phase2" if ph.startswith("Phase2") else ph)
            lookup[(str(r["Parameter"]), str(r["Station"]), ph_short)] = {
                "smk_slope": float(r["SMK_Sen_Slope_yr"]),
                "smk_p": float(r["SMK_p"]),
                "ann_slope": float(r["Annual_TheilSen_Slope_yr"]),
                "ci_lo": float(r["Annual_CI95_lo"]),
                "ci_hi": float(r["Annual_CI95_hi"]),
            }
    return lookup


def figure_2() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    smk_lookup = _load_smk_lookup()

    metrics_rows = []
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

    # ── (a) Downstream NO3-N enrichment gradient: Phase 1 vs Phase 2 ──
    ax = axes[0, 0]
    x_pos = np.arange(len(stations_ordered))
    tn_col = "NO3-N (mg/L)"
    phase1_slopes, phase2_slopes = [], []
    phase1_pvals, phase2_pvals = [], []

    for stn in stations_ordered:
        sub = mrc[mrc["stationName"] == stn]
        annual = sub.groupby("year")[tn_col].mean().dropna()
        x = annual.index.to_numpy(dtype=float)
        y = annual.to_numpy(dtype=float)

        m1_ann = x <= BREAK_YEAR
        m2_ann = x > BREAK_YEAR
        slope1 = float(np.asarray(theilslopes(y[m1_ann], x[m1_ann]), dtype=float)[0]) if m1_ann.sum() > 5 else 0.0
        slope2 = float(np.asarray(theilslopes(y[m2_ann], x[m2_ann]), dtype=float)[0]) if m2_ann.sum() > 5 else 0.0

        info1 = smk_lookup.get(("NO3-N", stn, "Phase1"))
        info2 = smk_lookup.get(("NO3-N", stn, "Phase2"))
        p1 = info1["smk_p"] if info1 is not None else 1.0
        p2 = info2["smk_p"] if info2 is not None else 1.0

        phase1_slopes.append(slope1)
        phase2_slopes.append(slope2)
        phase1_pvals.append(p1)
        phase2_pvals.append(p2)

        metrics_rows.append({
            "Panel": "a_NO3_enrichment",
            "Station": stn, "Parameter": "NO3-N",
            "Phase1_Slope_x1000": round(slope1 * 1000, 4),
            "Phase1_SMK_p": round(p1, 6),
            "Phase2_Slope_x1000": round(slope2 * 1000, 4),
            "Phase2_SMK_p": round(p2, 6),
            "n_years": int(len(annual)),
        })

    width = 0.35
    ax.bar(x_pos - width / 2, phase1_slopes, width,
           color=PHASE1_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=3)
    ax.bar(x_pos + width / 2, phase2_slopes, width,
           color=PHASE2_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 2 (>{BREAK_YEAR})", zorder=3)

    max_slope_a = max(max(phase1_slopes), max(phase2_slopes), key=abs)
    for i, (s1, p1, s2, p2) in enumerate(zip(phase1_slopes, phase1_pvals, phase2_slopes, phase2_pvals)):
        for s, p, offset in [(s1, p1, -width / 2), (s2, p2, width / 2)]:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                y_offset = 0.03 * abs(max_slope_a) if s >= 0 else -0.03 * abs(max_slope_a)
                ax.text(i + offset, s + y_offset, sig,
                        ha="center", va="bottom" if s >= 0 else "top",
                        fontsize=10.8, fontweight="bold", color="#111111")

    y_max_a = max(max(phase1_slopes), max(phase2_slopes))
    if y_max_a > 0:
        ax.set_ylim(top=y_max_a * 1.38)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stations_ordered, fontsize=8.4)
    ax.set_ylabel(r"NO$_3^-$-N trend (mg $\mathrm{L^{-1}}\ \mathrm{yr^{-1}}$)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.4)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    ax.legend(fontsize=8.4, frameon=False, edgecolor="none",
              loc="upper right", bbox_to_anchor=(1.0, 1.08),
              ncol=2, handlelength=1.0, columnspacing=0.8)
    panel_label(ax, "a", x=-0.26, y=1.12)

    # ── (b) Downstream COD enrichment gradient: Phase 1 vs Phase 2 ──
    ax = axes[1, 0]
    cod_col = "COD (mg/L)"
    phase1_slopes, phase2_slopes = [], []
    phase1_pvals, phase2_pvals = [], []

    for stn in stations_ordered:
        sub = mrc[mrc["stationName"] == stn]
        annual = sub.groupby("year")[cod_col].mean().dropna()
        x = annual.index.to_numpy(dtype=float)
        y = annual.to_numpy(dtype=float)

        m1_ann = x <= BREAK_YEAR
        m2_ann = x > BREAK_YEAR
        slope1 = float(np.asarray(theilslopes(y[m1_ann], x[m1_ann]), dtype=float)[0]) if m1_ann.sum() > 5 else 0.0
        slope2 = float(np.asarray(theilslopes(y[m2_ann], x[m2_ann]), dtype=float)[0]) if m2_ann.sum() > 5 else 0.0

        info1 = smk_lookup.get(("COD", stn, "Phase1"))
        info2 = smk_lookup.get(("COD", stn, "Phase2"))
        p1 = info1["smk_p"] if info1 is not None else 1.0
        p2 = info2["smk_p"] if info2 is not None else 1.0

        phase1_slopes.append(slope1)
        phase2_slopes.append(slope2)
        phase1_pvals.append(p1)
        phase2_pvals.append(p2)

        metrics_rows.append({
            "Panel": "b_COD_enrichment",
            "Station": stn, "Parameter": "COD",
            "Phase1_Slope_x1000": round(slope1 * 1000, 4),
            "Phase1_SMK_p": round(p1, 6),
            "Phase2_Slope_x1000": round(slope2 * 1000, 4),
            "Phase2_SMK_p": round(p2, 6),
            "n_years": int(len(annual)),
        })

    ax.bar(x_pos - width / 2, phase1_slopes, width,
           color=PHASE1_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=3)
    ax.bar(x_pos + width / 2, phase2_slopes, width,
           color=PHASE2_CLR, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=f"Phase 2 (>{BREAK_YEAR})", zorder=3)

    max_slope_b = max(max(phase1_slopes), max(phase2_slopes), key=abs)
    for i, (s1, p1, s2, p2) in enumerate(zip(phase1_slopes, phase1_pvals, phase2_slopes, phase2_pvals)):
        for s, p, offset in [(s1, p1, -width / 2), (s2, p2, width / 2)]:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                y_offset = 0.03 * abs(max_slope_b) if s >= 0 else -0.03 * abs(max_slope_b)
                ax.text(i + offset, s + y_offset, sig,
                        ha="center", va="bottom" if s >= 0 else "top",
                        fontsize=10.8, fontweight="bold", color="#111111")

    y_max_b = max(max(phase1_slopes), max(phase2_slopes))
    if y_max_b > 0:
        ax.set_ylim(top=y_max_b * 1.28)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stations_ordered, fontsize=8.4)
    ax.set_ylabel(r"COD trend (mg $\mathrm{L^{-1}}\ \mathrm{yr^{-1}}$)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.4)
    ax.axhline(0, color="black", lw=0.6, zorder=1)
    panel_label(ax, "b", x=-0.26, y=1.12)

    # ── (c) Q-normalised NO3-N mass balance ──
    ax = axes[0, 1]
    tn_params = [("NO3-N", "#756bb1", "-", r"NO$_3^-$-N")]
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

            ax.plot(annual.index, annual.values, "o-", color=clr,
                    lw=1.2, ms=3.0, markeredgewidth=0, alpha=0.85,
                    label=legend_label, zorder=2)

            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            ts = np.asarray(theilslopes(ya, xa), dtype=float)
            slope = float(ts[0])
            p = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            p1_mask = xa <= BREAK_YEAR
            p2_mask = xa > BREAK_YEAR
            p1_mean = float(np.mean(ya[p1_mask])) if p1_mask.any() else np.nan
            p2_mean = float(np.mean(ya[p2_mask])) if p2_mask.any() else np.nan
            mw_p = float(mannwhitneyu(ya[p1_mask], ya[p2_mask], alternative="two-sided").pvalue) if (p1_mask.any() and p2_mask.any()) else np.nan

            metrics_rows.append({
                "Panel": "c_NO3_mass_balance",
                "Station": "TanChau→MyThuan", "Parameter": plabel,
                "Phase1_NCF_pct": round(p1_mean, 2),
                "Phase2_NCF_pct": round(p2_mean, 2),
                "Delta_NCF_pct": round((p2_mean - p1_mean), 2) if np.isfinite(p1_mean) and np.isfinite(p2_mean) else None,
                "Trend_slope_pct_yr": round(slope, 4),
                "Kendall_p": round(p, 6),
                "PhaseShift_MW_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "n_years": int(len(annual)),
            })

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    ax.text(BREAK_YEAR + 0.75, 0.97, "Regime shift",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="left", va="top", fontsize=8.4, color="#525252")
    y0, y1 = ax.get_ylim()
    ax.axhspan(min(0, y0), 0, color="#deebf7", alpha=0.20, zorder=0)
    ax.axhspan(0, max(0, y1), color="#fee0d2", alpha=0.20, zorder=0)
    ax.text(0.03, 0.92, "Net generation", transform=ax.transAxes, fontsize=8.4,
            color="#7f0000", style="italic", fontweight="bold")
    ax.text(0.03, 0.03, "Net consumption", transform=ax.transAxes, fontsize=8.4,
            color="#084081", style="italic", fontweight="bold")
    ax.set_ylabel("Non-conservative fraction (%)", fontsize=8.4)
    ax.tick_params(axis="both", labelsize=8.4)
    panel_label(ax, "c", x=-0.26, y=1.12)
    ax.legend(fontsize=8.4, frameon=False, edgecolor="none",
              loc="upper right", ncol=1, handlelength=1.2)
    ax.set_xticks([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020])
    ax.tick_params(axis="x", labelbottom=True, rotation=90)

    # ── (d) Q-normalised COD mass balance ──
    ax = axes[1, 1]
    cod_params = [("COD", "#d62728", "-", "COD")]
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
            annual = annual_df["nonconservative_pct"]

            ax.plot(annual.index, annual.values, "o-", color=clr,
                    lw=1.2, ms=3.0, markeredgewidth=0, alpha=0.85,
                    label=legend_label, zorder=2)

            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            slope = float(np.asarray(theilslopes(ya, xa), dtype=float)[0])
            p = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            p1_mask = xa <= BREAK_YEAR
            p2_mask = xa > BREAK_YEAR
            p1_mean = float(np.mean(ya[p1_mask])) if p1_mask.any() else np.nan
            p2_mean = float(np.mean(ya[p2_mask])) if p2_mask.any() else np.nan
            mw_p = float(mannwhitneyu(ya[p1_mask], ya[p2_mask], alternative="two-sided").pvalue) if (p1_mask.any() and p2_mask.any()) else np.nan

            metrics_rows.append({
                "Panel": "d_COD_mass_balance",
                "Station": "TanChau→MyThuan", "Parameter": plabel,
                "Phase1_NCF_pct": round(p1_mean, 2),
                "Phase2_NCF_pct": round(p2_mean, 2),
                "Delta_NCF_pct": round((p2_mean - p1_mean), 2) if np.isfinite(p1_mean) and np.isfinite(p2_mean) else None,
                "Trend_slope_pct_yr": round(slope, 4),
                "Kendall_p": round(p, 6),
                "PhaseShift_MW_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "n_years": int(len(annual)),
            })

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    y0, y1 = ax.get_ylim()
    ax.axhspan(min(0, y0), 0, color="#deebf7", alpha=0.20, zorder=0)
    ax.axhspan(0, max(0, y1), color="#fee0d2", alpha=0.20, zorder=0)
    ax.set_ylabel("Non-conservative fraction (%)", fontsize=8.4)
    ax.set_xticks([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020])
    ax.tick_params(axis="x", labelbottom=True, rotation=90, labelsize=8.4)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.text(0.03, 0.92, "Net generation", transform=ax.transAxes, fontsize=8.4,
            color="#7f0000", style="italic", fontweight="bold")
    ax.text(0.03, 0.03, "Net consumption", transform=ax.transAxes, fontsize=8.4,
            color="#084081", style="italic", fontweight="bold")
    panel_label(ax, "d", x=-0.26, y=1.12)
    ax.legend(fontsize=8.4, frameon=False, edgecolor="none",
              loc="upper right", ncol=1, handlelength=1.2)

    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save_fig(fig, "CEE_Figure_2_Mainstem_Reactor", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_2()
