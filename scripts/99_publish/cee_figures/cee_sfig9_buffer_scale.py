#!/usr/bin/env python3
"""
CEE Supplementary Figure 9 — Buffer Scale Sensitivity
=====================================================
Cloned directly from `nw_sfig3_buffer_scale.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _cee_common import (
    apply_style, MM, BUF_BT_CSV, AO,
    panel_label, save_fig,
)

apply_style()

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
    x = np.array(buffers, dtype=float)
    y = np.abs(np.array(correlations))
    if len(x) < 3:
        return x[y.argmax()]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dists = [abs(dx * (p1[1] - y_n[i]) - dy * (p1[0] - x_n[i]))
             / (np.hypot(dx, dy) + 1e-10) for i in range(len(x_n))]
    return x[np.argmax(dists)]


def _clean_label(metric: str) -> str:
    s = str(metric)
    s = (s.replace("Residential_Urban", "Urban")
           .replace("Rice_Cultivation", "Rice").replace("Rice_Paddy", "Rice")
           .replace("Aquaculture", "Aqua").replace("Water_Bodies", "Water")
           .replace("Other_Crops", "Upland crops")
           .replace("_composition_pct", " %")
           .replace("composition_pct", "%")
           .replace("_config_", " ")
           .replace("_", " "))
    return " ".join(s.split())


def main() -> None:
    metrics = []
    fdr_csv = AO / "Table_Temporal_Bootstrap_FDR_Corrected.csv"
    if not fdr_csv.exists():
        if BUF_BT_CSV.exists():
            fdr_csv = BUF_BT_CSV
        else:
            print("  WARNING: No buffer scale data found, skipping CEE_SFig_9")
            return

    df = pd.read_csv(fdr_csv)
    if df.empty:
        return

    buf_col = "Buffer_km" if "Buffer_km" in df.columns else "buffer_km"
    wq_col = "WQ_Parameter" if "WQ_Parameter" in df.columns else "WQ_Param"
    lulc_col = "LULC_Metric" if "LULC_Metric" in df.columns else "LULC_Category"
    r_col = "Spearman_r" if "Spearman_r" in df.columns else "spearman_r"

    wq_order = ["COD", "DO", "NH4", "NO3", "TN", "TSS"]
    wq_params = [p for p in wq_order if p in set(df[wq_col].dropna().astype(str).unique())]
    TOP_N = 5

    n_params = len(wq_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(180 * MM, 58 * n_rows * MM), sharex=True)
    axes_flat = list(axes_arr) if n_rows == 1 else axes_arr.flatten()

    for idx, wq in enumerate(wq_params):
        ax = axes_flat[idx]
        wq_data = df[df[wq_col] == wq].copy()
        if wq_data.empty:
            ax.text(0.5, 0.5, f"{wq}\n(no data)", transform=ax.transAxes, ha="center")
            panel_label(ax, chr(ord("a") + idx))
            continue

        cat_champs: dict[str, tuple[str, float, float, float, float]] = {}
        for metric in wq_data[lulc_col].dropna().astype(str).unique():
            _, _, cat = _get_style_cat(metric)
            if cat == "Unknown":
                continue
            sub = wq_data[wq_data[lulc_col].astype(str) == metric]
            if sub.empty:
                continue
            peak_idx = sub[r_col].abs().idxmax()
            r_val = float(sub.loc[peak_idx, r_col])
            buf_peak = float(sub.loc[peak_idx, buf_col])
            sub_sorted = sub.sort_values(buf_col)
            elbow = float(_kneedle_elbow(sub_sorted[buf_col].values, sub_sorted[r_col].values))
            expected = (r_val < 0) if wq == "DO" else (r_val > 0)
            score = abs(r_val) + (1.0 if expected else 0.0)
            if cat not in cat_champs or score > cat_champs[cat][1]:
                cat_champs[cat] = (metric, score, r_val, buf_peak, elbow)

        ranked = sorted(cat_champs.values(), key=lambda x: x[1], reverse=True)[:TOP_N]
        if not ranked:
            continue

        best_metric, _, best_r, peak_buf, elbow_buf = ranked[0]
        all_y_vals = []
        right_y_vals = []
        for j, (metric, _, r_val, buf_peak, buf_elbow) in enumerate(ranked):
            sub = wq_data[wq_data[lulc_col].astype(str) == str(metric)].sort_values(buf_col)
            clr, ls, _ = _get_style_cat(str(metric))
            lw = 1.8 if j == 0 else 1.0
            max_r = sub.loc[sub[r_col].abs().idxmax(), r_col]
            label = f"{_clean_label(metric)} ($r$={max_r:.2f})"
            all_y_vals.extend(sub[r_col].dropna().tolist())
            right_y_vals.extend(sub.loc[sub[buf_col] >= 9, r_col].dropna().tolist())
            ax.plot(sub[buf_col], sub[r_col], label=label, color=clr,
                    linestyle=ls, linewidth=lw, zorder=10 - j)

        metrics.append({"Figure": "CEE_SFig9", "WQ": wq, "Peak_km": int(peak_buf),
                        "Elbow_km": int(elbow_buf), "Best_metric": best_metric,
                        "Peak_r": round(float(best_r), 4)})

        ax.axvline(elbow_buf, color="green", ls="--", lw=1.0, alpha=0.5, zorder=1)
        title_wq = "NH$_4^+$-N" if wq in {"NH4", "NH4-N"} else ("NO$_3^-$-N" if wq in {"NO3", "NO3-N"} else wq)
        ax.set_title(f"{title_wq} (elbow: {int(elbow_buf)} km)",
                     fontsize=7.5, fontweight="bold", pad=4)
        ax.set_xlim(0, 31)
        if all_y_vals:
            y_min, y_max = min(all_y_vals), max(all_y_vals)
            y_span = max(y_max - y_min, 0.05)
            r_min = min(right_y_vals) if right_y_vals else y_min
            r_max = max(right_y_vals) if right_y_vals else y_max
            if wq == "DO":
                y_bot = max(-1.0, y_min - 0.08 * y_span)
                y_top = min(1.0, max(y_max + 0.08 * y_span, (r_max - 0.58 * y_bot) / 0.42))
            else:
                y_top = min(1.0, y_max + 0.08 * y_span)
                y_bot = max(-1.0, min(y_min - 0.08 * y_span, (r_min - 0.42 * y_top) / 0.58))
            ax.set_ylim(y_bot, y_top)
        ax.grid(True, ls=":", alpha=0.3)
        leg_loc = "upper right" if wq == "DO" else "lower right"
        leg = ax.legend(fontsize=5.8, frameon=True, framealpha=0.95, edgecolor="lightgray",
                        loc=leg_loc, handletextpad=0.28, labelspacing=0.24, borderpad=0.28)
        leg.set_zorder(20)
        if idx >= len(wq_params) - n_cols:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Buffer radius (km)")
        ax.set_ylabel(r"Spearman $r_s$")
        panel_label(ax, chr(ord("a") + idx), x=-0.24, y=1.08)

    for i in range(len(wq_params), len(axes_flat)):
        axes_flat[i].axis("off")

    fig.tight_layout(h_pad=2.2, w_pad=1.0)
    save_fig(fig, "CEE_SFig_9_Buffer_Scale", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
