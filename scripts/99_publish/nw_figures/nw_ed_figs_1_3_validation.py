#!/usr/bin/env python3
"""
Extended Data Figures 1-3 — Validation & Trends
=================================================
ED Fig 1: Cross-network validation (MRC vs Ben Tre)
ED Fig 2: Buffer scale sensitivity analysis
ED Fig 3: Extended 8-parameter regime shift

All export companion metrics CSVs.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes, kendalltau

from _nw_common import (
    apply_style, load_wq, MM, MRC_STATIONS, STATION_CLR,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR, BUF_BT_CSV, ROOT, AO,
    panel_label, save_fig,
)

apply_style()


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 1 — Cross-Network Validation
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_1() -> None:
    wq = load_wq()
    metrics = []

    pairs = [
        {"MRC": "Can Tho", "BenTre": "CC01", "label": "Can Tho / CC01"},
        {"MRC": "My Tho",  "BenTre": "MT01", "label": "My Tho / MT01"},
    ]
    params = [
        ("COD (mg/L)", "COD"), ("DO (mg/L)",  "DO"),
        ("NH4-N (mg/L)", r"NH$_4$-N"), ("NO3-N (mg/L)", r"NO$_3$-N"),
    ]

    fig = plt.figure(figsize=(180 * MM, 140 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

    for pidx, (pcol, plabel) in enumerate(params):
        row, col = divmod(pidx, 2)
        ax = fig.add_subplot(gs[row, col])
        all_mrc, all_bt = [], []

        for pair in pairs:
            mrc_data = wq[(wq["stationName"] == pair["MRC"]) & (wq["Data source"] == "MRC")]
            bt_data = wq[(wq["stationName"] == pair["BenTre"]) & (wq["Data source"] == "Ben Tre")]
            if pcol not in mrc_data.columns or pcol not in bt_data.columns:
                continue

            mrc_ann = mrc_data.groupby("year")[pcol].mean().dropna()
            bt_ann = bt_data.groupby("year")[pcol].mean().dropna()
            common = mrc_ann.index.intersection(bt_ann.index)
            if len(common) < 3:
                continue

            m_vals = mrc_ann.loc[common].values
            b_vals = bt_ann.loc[common].values
            all_mrc.extend(m_vals); all_bt.extend(b_vals)

            clr = STATION_CLR.get(pair["MRC"], "#333333")
            ax.scatter(m_vals, b_vals, s=18, c=clr, edgecolors="#333333",
                       linewidths=0.3, alpha=0.8, label=pair["label"], zorder=3)

        if all_mrc and all_bt:
            mn = min(min(all_mrc), min(all_bt)) * 0.8
            mx = max(max(all_mrc), max(all_bt)) * 1.2
            ax.plot([mn, mx], [mn, mx], "--", color="#969696", lw=0.8, zorder=1)
            r_val, p_val = stats.pearsonr(all_mrc, all_bt)
            ax.text(0.05, 0.95, f"r = {r_val:.2f}\np = {p_val:.3f}",
                    transform=ax.transAxes, fontsize=5.5, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
            metrics.append({"Figure": "ED1", "Parameter": plabel,
                            "Pearson_r": round(r_val, 4), "p_value": round(p_val, 6),
                            "N_paired_years": len(all_mrc)})

        ax.set_xlabel(f"MRC {plabel}", fontsize=7)
        ax.set_ylabel(f"Ben Tre {plabel}", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        if pidx == 0:
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="none", loc="lower right")
        panel_label(ax, chr(ord("a") + pidx))

    save_fig(fig, "ED_Figure_1_Network_CrossValidation", pd.DataFrame(metrics))


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 2 — Buffer Scale Sensitivity (modelled on generate_buffer_scale_figure.py)
# ═══════════════════════════════════════════════════════════════════════

# Category colours (colorblind-friendly)
_CAT_COLORS = {
    "Residential_Urban": "#D55E00", "Water_Bodies": "#0072B2",
    "Aquaculture": "#CC79A7", "Rice_Cultivation": "#009E73",
    "Rice_Paddy": "#009E73", "Forest": "#228B22",
    "Wetland": "#56B4E9", "Other_Crops": "#E69F00", "Others": "#999999",
}

_METRIC_STYLES = {
    "composition_pct": "-", "pct": "-", "%": "-",
    "LPI": "--", "PD": "-.", "ED": ":", "LSI": (0, (5, 5)),
}


def _get_style_cat(metric: str):
    clr, ls, cat = "#333333", "-", "Unknown"
    for c, col in _CAT_COLORS.items():
        if c in metric:
            clr = col; cat = c; break
    for m, st in _METRIC_STYLES.items():
        if m in metric:
            ls = st; break
    return clr, ls, cat


def _kneedle_elbow(buffers, correlations):
    """Kneedle algorithm (Satopaa et al., 2011)."""
    x = np.array(buffers, dtype=float)
    y = np.abs(np.array(correlations))
    if len(x) < 3:
        return x[y.argmax()]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    p1, p2 = np.array([x_n[0], y_n[0]]), np.array([x_n[-1], y_n[-1]])
    dists = [abs(np.cross(p2 - p1, p1 - np.array([x_n[i], y_n[i]])))
             / (np.linalg.norm(p2 - p1) + 1e-10) for i in range(len(x_n))]
    return x[np.argmax(dists)]


def _clean_label(metric: str) -> str:
    # Supports both underscore-style and space-style metric names
    s = str(metric)
    s = (s.replace("Residential_Urban", "Urban")
           .replace("Rice_Cultivation", "Rice").replace("Rice_Paddy", "Rice")
           .replace("Aquaculture", "Aqua").replace("Water_Bodies", "Water")
           .replace("Other_Crops", "Other")
           .replace("_composition_pct", " %")
           .replace("composition_pct", "%")
           .replace("_config_", " ")
           .replace("_", " "))
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def ed_figure_2() -> None:
    metrics = []

    # Use FDR-corrected bootstrap file (same as user's generate_buffer_scale_figure.py)
    fdr_csv = AO / "Table_Temporal_Bootstrap_FDR_Corrected.csv"
    if not fdr_csv.exists():
        # Fall back to plain buffer correlations
        if BUF_BT_CSV.exists():
            fdr_csv = BUF_BT_CSV
        else:
            print("  WARNING: No buffer scale data found, skipping ED Figure 2")
            return

    df = pd.read_csv(fdr_csv)
    if df.empty:
        print("  WARNING: Buffer scale data empty, skipping")
        return

    # Determine column names adaptively
    buf_col = "Buffer_km" if "Buffer_km" in df.columns else "buffer_km"
    wq_col = "WQ_Parameter" if "WQ_Parameter" in df.columns else "WQ_Param"
    lulc_col = "LULC_Metric" if "LULC_Metric" in df.columns else "LULC_Category"
    r_col = "Spearman_r" if "Spearman_r" in df.columns else "spearman_r"
    p_col = "Spearman_p" if "Spearman_p" in df.columns else "p_value"
    has_ci = "r_CI_lower" in df.columns and "r_CI_upper" in df.columns

    # Core parameters matching manuscript narrative
    wq_order = ["COD", "DO", "NH4", "TN", "TSS"]
    wq_params = [p for p in wq_order if p in set(df[wq_col].dropna().astype(str).unique())]
    TOP_N = 5

    # Simplified 2×3 grid for core parameters
    n_params = len(wq_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(180 * MM, 55 * n_rows * MM), sharex=True)
    if n_rows == 1:
        axes_flat = list(axes_arr)
    else:
        axes_flat = axes_arr.flatten()

    debug = os.environ.get("NW_DEBUG", "0").strip() not in {"", "0", "false", "False"}
    if debug:
        print(f"  ED2: rows={len(df):,}, WQ params={wq_params}")

    for idx, wq in enumerate(wq_params):
        ax = axes_flat[idx]
        wq_data = df[df[wq_col] == wq].copy()
        if wq_data.empty:
            ax.text(0.5, 0.5, f"{wq}\n(no data)", transform=ax.transAxes, ha="center")
            panel_label(ax, chr(ord("a") + idx))
            continue

        # Find champion metric per LULC category (mirrors generate_buffer_scale_figure.py)
        cat_champs: dict[str, tuple[str, float, float, float, float]] = {}
        unknown_count = 0
        for metric in wq_data[lulc_col].dropna().astype(str).unique():
            _, _, cat = _get_style_cat(metric)
            if cat == "Unknown":
                unknown_count += 1
                continue
            sub = wq_data[wq_data[lulc_col].astype(str) == metric]
            if sub.empty:
                continue
            peak_idx = sub[r_col].abs().idxmax()
            r_val = float(sub.loc[peak_idx, r_col])
            buf_peak = float(sub.loc[peak_idx, buf_col])
            sub_sorted = sub.sort_values(buf_col)
            elbow = float(_kneedle_elbow(sub_sorted[buf_col].values, sub_sorted[r_col].values))
            # Expected sign: DO is typically negative; most other parameters positive.
            expected = (r_val < 0) if wq == "DO" else (r_val > 0)
            score = abs(r_val) + (1.0 if expected else 0.0)
            if cat not in cat_champs or score > cat_champs[cat][1]:
                cat_champs[cat] = (metric, score, r_val, buf_peak, elbow)

        ranked = sorted(cat_champs.values(), key=lambda x: x[1], reverse=True)[:TOP_N]
        if debug:
            print(f"    {wq}: metrics={wq_data[lulc_col].nunique()}, unknown={unknown_count}, ranked={len(ranked)}")

        # Fallback: if category parsing yields nothing, plot top-|r| metrics so figure is never blank.
        if not ranked:
            tmp = (wq_data[[buf_col, lulc_col, r_col]].dropna().copy())
            if not tmp.empty:
                top_metrics = (
                    tmp.groupby(lulc_col)[r_col]
                    .apply(lambda s: float(s.abs().max()))
                    .sort_values(ascending=False)
                    .head(TOP_N)
                    .index.astype(str)
                    .tolist()
                )
                ranked = []
                for m in top_metrics:
                    sub = wq_data[wq_data[lulc_col].astype(str) == m].sort_values(buf_col)
                    if sub.empty:
                        continue
                    peak_idx = sub[r_col].abs().idxmax()
                    r_val = float(sub.loc[peak_idx, r_col])
                    buf_peak = float(sub.loc[peak_idx, buf_col])
                    elbow = float(_kneedle_elbow(sub[buf_col].values, sub[r_col].values))
                    ranked.append((m, abs(r_val), r_val, buf_peak, elbow))
                ax.text(0.02, 0.98, "Fallback: top-|r| metrics",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=5.5, color="#7f7f7f",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
            if not ranked:
                ax.text(0.5, 0.5, f"{wq}\n(no valid metrics)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="#969696")
                panel_label(ax, chr(ord("a") + idx))
                continue

        best_metric, _, best_r, peak_buf, elbow_buf = ranked[0]

        for j, (metric, _, r_val, buf_peak, buf_elbow) in enumerate(ranked):
            sub = wq_data[wq_data[lulc_col].astype(str) == str(metric)].sort_values(buf_col)
            clr, ls, _ = _get_style_cat(str(metric))
            lw = 1.8 if j == 0 else 1.0
            max_r = sub.loc[sub[r_col].abs().idxmax(), r_col]
            line_pk = sub.loc[sub[r_col].abs().idxmax(), buf_col]
            label = f"{_clean_label(metric)} ($r$={max_r:.2f})"
            ax.plot(sub[buf_col], sub[r_col], label=label, color=clr,
                    linestyle=ls, linewidth=lw, zorder=10 - j)
            # No uncertainty bands - cleaner figure

        metrics.append({"Figure": "ED2", "WQ": wq, "Peak_km": int(peak_buf),
                        "Elbow_km": int(elbow_buf), "Best_metric": best_metric,
                        "Peak_r": round(float(best_r), 4)})

        # Only show elbow indicator (key diagnostic, simpler)
        ax.axvline(elbow_buf, color="green", ls="--", lw=1.0, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax.set_title(f"{wq} (elbow: {int(elbow_buf)} km)",
                     fontsize=7, fontweight="bold")
        ax.set_xlim(0, 31)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=5.0, frameon=True, framealpha=0.95, edgecolor="lightgray",
                  loc="best")
        if idx >= len(wq_params) - n_cols:
            ax.set_xlabel("Buffer radius (km)")
        if idx % 3 == 0:
            ax.set_ylabel(r"Spearman $r_s$")
        panel_label(ax, chr(ord("a") + idx))

    # Hide unused axes
    for i in range(len(wq_params), len(axes_flat)):
        axes_flat[i].axis("off")

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save_fig(fig, "ED_Figure_2_Buffer_Scale", pd.DataFrame(metrics))


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 3 — Extended 8-Parameter Trends
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_3() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    metrics = []

    params = [
        ("COD (mg/L)",   r"COD (mg $\mathrm{L^{-1}}$)",  BREAK_YEAR),
        ("DO (mg/L)",    r"DO (mg $\mathrm{L^{-1}}$)",   BREAK_YEAR),
        ("pH",           "pH",             BREAK_YEAR),
        ("NH4-N (mg/L)", r"NH$_4$-N (mg $\mathrm{L^{-1}}$)",BREAK_YEAR),
        ("NO3-N (mg/L)", r"NO$_3$-N (mg $\mathrm{L^{-1}}$)",BREAK_YEAR),
        ("TN (mg/L)",    r"TN (mg $\mathrm{L^{-1}}$)",   BREAK_YEAR),
        ("TP (mg/L)",    r"TP (mg $\mathrm{L^{-1}}$)",   BREAK_YEAR),
        ("TSS (mg/L)",   r"TSS (mg $\mathrm{L^{-1}}$)",  BREAK_YEAR),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(180 * MM, 150 * MM), sharex=True)

    for idx, (pcol, ylabel, bp) in enumerate(params):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for stn in MRC_STATIONS:
            sub = mrc[mrc["stationName"] == stn]
            if pcol not in sub.columns:
                continue
            annual = sub.groupby("year")[pcol].mean().dropna()
            if len(annual) < 5:
                continue

            ax.plot(annual.index, annual.values, "-o",
                    color=STATION_CLR[stn], lw=0.9, ms=2, alpha=0.85,
                    label=stn if idx == 0 else None, zorder=3)

            x_arr = annual.index.values.astype(float)
            y_arr = annual.values
            for phase, mask in [("Phase1", x_arr <= bp), ("Phase2", x_arr > bp)]:
                xm, ym = x_arr[mask], y_arr[mask]
                if len(xm) > 5:
                    slope, intercept, _, _ = theilslopes(ym, xm)
                    tau, p = kendalltau(xm, ym)
                    ax.plot(xm, intercept + slope * xm, "-",
                            color=STATION_CLR[stn], lw=1.5, alpha=0.7)
                    metrics.append({
                        "Figure": "ED3", "Station": stn,
                        "Parameter": pcol.split(" (")[0], "Phase": phase,
                        "Slope": round(slope, 6), "p": round(p, 6),
                    })

        ax.axvline(bp + 0.5, color="#525252", lw=0.8, ls="--", alpha=0.6)
        ax.axvspan(1985, bp + 0.5, color=PHASE1_CLR, alpha=0.02, zorder=0)
        ax.axvspan(bp + 0.5, 2022, color=PHASE2_CLR, alpha=0.02, zorder=0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.locator_params(axis="y", nbins=4)
        if row < 3:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("")
        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper left",
                      frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.2)

    # Ensure uniform x-axis limits and ticks across all subplots
    for ax in axes.flat:
        ax.set_xlim(1984, 2023)
        ax.set_xticks(range(1985, 2025, 5))

    fig.tight_layout(h_pad=0.6, w_pad=1.5)
    save_fig(fig, "ED_Figure_1_Extended_Trends", pd.DataFrame(metrics))


if __name__ == "__main__":
    print("=" * 65)
    print("Extended Data Figures 1-3: Validation & Trends")
    print("=" * 65)
    ed_figure_1()
    ed_figure_2()
    ed_figure_3()
    print("Done.")
