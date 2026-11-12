#!/usr/bin/env python3
"""
CEE Figure 5 — Tidal Range Amplification and Reversal
=====================================================
Cloned directly from `nw_ed_fig6_tidal_decomposition.py` to preserve 100% of
the tuned 1x2 layout (180 x 68 mm), rotated slope labels, zero 'Year' x-labels,
and zero defensive callout boxes.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau

from _cee_common import (
    apply_style, MM, STATION_CLR, BREAK_YEAR, TIDAL_CSV,
    panel_label, save_fig,
)

apply_style()


def _compute_tidal_range_from_raw() -> pd.DataFrame:
    from pathlib import Path
    wl_file = Path(__file__).resolve().parent.parent.parent.parent / (
        "data/hydrology/data/inputs/agency_cleaned/"
        "water_level_hourly_unified_cleaned.csv"
    )
    if not wl_file.exists():
        raise FileNotFoundError(f"Raw hourly water level file not found: {wl_file}")
    df = pd.read_csv(wl_file, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    rows: list[dict] = []
    for col, stn in [("H_CanTho", "CanTho"), ("H_MyThuan", "MyThuan")]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        daily_range = (s.resample("D").max() - s.resample("D").min()).dropna()
        daily_count = s.resample("D").count()
        daily_range = daily_range[daily_count >= 20]

        ann = daily_range.resample("YE").mean().dropna()
        ann_count = daily_range.resample("YE").count()
        for ts, val in ann[ann_count >= 300].items():
            rows.append({"Station": stn, "year": ts.year,
                         "Metric": "AnnualMeanDailyTidalRange_m", "Value": float(val)})

        dry = daily_range[daily_range.index.month.isin([1, 2, 3, 4])]
        dry_ann = dry.resample("YE").mean().dropna()
        dry_count = dry.resample("YE").count()
        for ts, val in dry_ann[dry_count >= 90].items():
            rows.append({"Station": stn, "year": ts.year,
                         "Metric": "DrySeasonMeanDailyTidalRange_m", "Value": float(val)})

        wet = daily_range[daily_range.index.month.isin([5, 6, 7, 8, 9, 10])]
        wet_ann = wet.resample("YE").mean().dropna()
        wet_count = wet.resample("YE").count()
        for ts, val in wet_ann[wet_count >= 130].items():
            rows.append({"Station": stn, "year": ts.year,
                         "Metric": "WetSeasonMeanDailyTidalRange_m", "Value": float(val)})
    return pd.DataFrame(rows)


def _find_pooled_minimum(tidal: pd.DataFrame) -> int:
    common_years: set[int] | None = None
    series_data: list[tuple[np.ndarray, np.ndarray]] = []
    for stn in tidal["Station"].unique():
        for metric in tidal["Metric"].unique():
            sub = tidal[(tidal["Station"] == stn) & (tidal["Metric"] == metric)].sort_values("year")
            if len(sub) < 10:
                continue
            yrs = sub["year"].values.astype(int)
            vals = sub["Value"].values.astype(float)
            series_data.append((yrs, vals))
            s = set(yrs)
            common_years = s if common_years is None else common_years & s
    if common_years is None or len(common_years) < 10:
        return BREAK_YEAR
    common = sorted(common_years)
    combined = []
    for yrs, vals in series_data:
        yr_list = list(yrs)
        idx = [yr_list.index(c) for c in common]
        v = vals[idx].astype(float)
        v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        combined.append(v)
    pooled = np.nanmean(combined, axis=0)
    min_idx = int(np.argmin(pooled))
    return common[min_idx]


def main() -> None:
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM))

    if TIDAL_CSV.exists():
        tidal = pd.read_csv(TIDAL_CSV)
        if "WetSeasonMeanDailyTidalRange_m" not in tidal["Metric"].values:
            tidal = _compute_tidal_range_from_raw()
            tidal.to_csv(TIDAL_CSV, index=False)
    else:
        tidal = _compute_tidal_range_from_raw()
        tidal.to_csv(TIDAL_CSV, index=False)

    if not tidal.empty:
        tidal = tidal[(tidal["year"] >= 1985) & (tidal["year"] <= 2025)]

    cp_year = _find_pooled_minimum(tidal)
    stn_label = {"CanTho": "Can Tho (Hau/Bassac)", "MyThuan": "My Thuan (Tien/Mekong)"}
    stn_clr = {"CanTho": STATION_CLR.get("Can Tho", "#4393c3"),
               "MyThuan": STATION_CLR.get("My Thuan", "#f4a582")}

    ylim = (1.05, 2.55)
    yticks = [1.2, 1.5, 1.8, 2.1, 2.4]

    for panel_idx, (metric_key, metric_label, ylabel) in enumerate([
        ("DrySeasonMeanDailyTidalRange_m", "DrySeasonTR", "Dry-season tidal range (m)"),
        ("WetSeasonMeanDailyTidalRange_m", "WetSeasonTR", "Wet-season tidal range (m)"),
    ]):
        ax = axes[panel_idx]
        annotations_to_add = []
        for stn in tidal["Station"].unique():
            sub = tidal[(tidal["Station"] == stn) & (tidal["Metric"] == metric_key)].sort_values("year")
            if sub.empty:
                continue
            clr = stn_clr.get(stn, "#333333")
            label = stn_label.get(stn, stn)
            ax.plot(sub["year"], sub["Value"], "-o", color=clr,
                    lw=1.0, ms=2.5, alpha=0.85, label=label, zorder=3)
            x = sub["year"].values.astype(float)
            y = sub["Value"].values.astype(float)
            for mask, period_label in [(x <= cp_year, "pre"), (x > cp_year, "post")]:
                xm, ym = x[mask], y[mask]
                if len(xm) > 3:
                    slope, intercept, _, _ = theilslopes(ym, xm)
                    _, p = kendalltau(xm, ym)
                    x_line = np.array([xm.min(), cp_year]) if period_label == "pre" else np.array([cp_year, xm.max()])
                    ax.plot(x_line, intercept + slope * x_line, "--", color=clr, lw=0.8, alpha=0.6)
                    cm_yr = slope * 100
                    sign = "+" if cm_yr >= 0 else ""
                    txt = f"{sign}{cm_yr:.1f} cm yr$^{{-1}}$"
                    if period_label == "pre":
                        lx = 1991.5
                        offset = (0.14 if metric_label == "WetSeasonTR" else 0.09) if stn == "CanTho" else -0.10
                    else:
                        if stn == "CanTho":
                            lx = 2017.5
                            offset = -0.14
                        else:
                            lx = 2014.5 if metric_label == "WetSeasonTR" else 2016.5
                            offset = 0.14 if metric_label == "WetSeasonTR" else 0.13
                    ly = intercept + slope * lx
                    annotations_to_add.append((lx, ly, slope, offset, txt, clr))
                    metrics.append({
                        "Figure": "CEE_Fig5", "Station": stn,
                        "Metric": metric_label, "Period": period_label,
                        "Changepoint_year": cp_year,
                        "Trend_cm_per_decade": round(slope * 1000, 2),
                        "Trend_cm_per_year": round(cm_yr, 2),
                        "p": round(p, 6),
                    })
        ax.axvline(cp_year, color="#555555", lw=0.8, ls=":", alpha=0.6, zorder=1)
        ax.set_xlim(1984.0, 2026.0)
        ax.set_xticks(np.arange(1985, 2026, 5))
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        panel_label(ax, chr(ord("a") + panel_idx), x=-0.14, y=1.05)
        if panel_idx == 0:
            ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")

        fig.canvas.draw()
        for lx, ly, slope, offset, txt, clr in annotations_to_add:
            p0 = ax.transData.transform((lx, ly))
            p1 = ax.transData.transform((lx + 1.0, ly + slope))
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            ax.text(lx, ly + offset, txt, fontsize=8, color=clr,
                    ha="center", va="bottom" if offset > 0 else "top",
                    fontweight="bold", rotation=angle, rotation_mode="anchor",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, pad=0.5), zorder=5)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "CEE_Figure_5_Tidal_Decomposition", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
