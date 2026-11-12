#!/usr/bin/env python3
"""
CEE Figure 1 — Spatiotemporal Regime Shift
==========================================
Cloned directly from `nw_fig1_regime_shift.py` to preserve 100% of the tuned
3x2 layout, cartographic design, station trajectories, internal frameless legends,
45-degree year ticks, and zero 'Year' x-labels. Adds only a subtle dotted
horizontal threshold line at DO = 5.0 mg L^-1 in subplot (b).
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from _cee_common import (
    apply_style, load_wq, MM, MRC_STATIONS, STATION_CLR,
    BREAK_YEAR, ROOT, panel_label, save_fig, require_file,
)

apply_style()


def _plot_study_area_map(ax) -> None:
    """Generate study area map using GeoPandas (identical to nw_fig1_regime_shift.py)."""
    DATA_DIR = ROOT / "data"

    water_bodies_file = DATA_DIR / "spatial" / "waterbody_Mekong_Saigon_Water.gpkg"
    require_file(water_bodies_file, label="waterbody_Mekong_Saigon_Water.gpkg")
    if not water_bodies_file.exists():
        ax.text(0.5, 0.5, "Map data not found", ha="center", va="center", transform=ax.transAxes)
        return

    water_bodies = gpd.read_file(water_bodies_file).to_crs(epsg=32648)

    wq_file = DATA_DIR / "mekong_delta_water_quality_1985_2023.csv"
    stations_df = pd.read_csv(wq_file)
    stations_unique = stations_df[["stationName", "latitude", "longitude", "River", "Data source"]].drop_duplicates()
    stations_gdf = gpd.GeoDataFrame(
        stations_unique,
        geometry=gpd.points_from_xy(stations_unique.longitude, stations_unique.latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=32648)

    mrc_stations = stations_gdf[stations_gdf["Data source"] == "MRC"]
    bentre_stations = stations_gdf[stations_gdf["Data source"] == "Ben Tre"]

    x0, y0, x1, y1 = stations_gdf.total_bounds
    margin = 20000
    xmin, xmax = x0 - margin, x1 + margin
    ymin, ymax = y0 - margin, y1 + margin

    from shapely.geometry import box
    water_bodies = gpd.clip(water_bodies, box(xmin, ymin, xmax, ymax)).copy()
    water_bodies["geometry"] = water_bodies.geometry.simplify(15.0, preserve_topology=True)
    water_bodies.plot(ax=ax, color="#4292c6", linewidth=0.8, alpha=0.8, zorder=1, rasterized=True)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    mrc_stations.plot(ax=ax, color="#d62728", marker="o", markersize=45,
                      edgecolor="#d62728", linewidth=0.5, zorder=10)
    bentre_stations.plot(ax=ax, color="#ff7f0e", marker="^", markersize=24,
                         edgecolor="#ff7f0e", linewidth=0.5, zorder=9)

    for _, row in mrc_stations.iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = row.stationName
        offset = (0, 4000)
        ax.annotate(label, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                    fontsize=8.0, fontweight="bold", color="#d62728",
                    ha="center", va="bottom", zorder=11)

    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="MRC network (1985–2022)",
               markerfacecolor="#d62728", markersize=8, markeredgecolor="white", markeredgewidth=1),
        Line2D([0], [0], marker="^", color="w", label="Ben Tre network (2005–2022)",
               markerfacecolor="#ff7f0e", markersize=7, markeredgecolor="white", markeredgewidth=1),
    ]

    ax.legend(handles=legend_elements, loc="lower left", fontsize=8.0, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.90, borderpad=0.4, ncol=1)
    ax.text(0.28, 0.72, "Tien River", transform=ax.transAxes, fontsize=7.6, style="italic",
            color="#08519c", fontweight="bold", rotation=-28, ha="center", va="center", zorder=12)
    ax.text(0.18, 0.44, "Hau River", transform=ax.transAxes, fontsize=7.6, style="italic",
            color="#08519c", fontweight="bold", rotation=-34, ha="center", va="center", zorder=12)
    ax.text(0.84, 0.24, "East Sea\nof Vietnam", transform=ax.transAxes, fontsize=8.0, style="italic",
            color="#08519c", ha="center", va="center", zorder=12)

    scale_bar_length = 20000
    scale_bar_x = xmax - 30000
    scale_bar_y = ymin + 18000
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], [scale_bar_y, scale_bar_y],
            "k-", linewidth=4, solid_capstyle="butt", zorder=10)
    ax.text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 6000, "20 km",
            ha="center", va="top", fontsize=8.5, fontweight="bold")

    ax.annotate("N", xy=(0.92, 0.92), xycoords="axes fraction",
                fontsize=11.0, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="black", linewidth=1))
    ax.annotate(r"$\uparrow$", xy=(0.92, 0.97), xycoords="axes fraction",
                fontsize=9.8, fontweight="bold", ha="center", va="center")


def figure_1() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()

    params = [
        ("COD (mg/L)",   r"COD (mg $\mathrm{L^{-1}}$)",        BREAK_YEAR),
        ("DO (mg/L)",    r"DO (mg $\mathrm{L^{-1}}$)",         BREAK_YEAR),
        ("NH4-N (mg/L)", r"NH$_4^+$-N (mg $\mathrm{L^{-1}}$)", BREAK_YEAR),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg $\mathrm{L^{-1}}$)", BREAK_YEAR),
    ]

    metrics_rows = []
    fig = plt.figure(figsize=(180 * MM, 120 * MM))

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.2, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.20, hspace=0.10,
    )

    ax_map = fig.add_subplot(gs[0:2, 0])
    if HAS_GEOPANDAS:
        _plot_study_area_map(ax_map)
    else:
        ax_map.text(0.5, 0.5, "GeoPandas not available", ha="center", va="center",
                    transform=ax_map.transAxes, fontsize=8)
        ax_map.axis("off")

    panel_label(ax_map, "a", x=-0.01, y=1.02)

    axes_by_param = {
        "COD (mg/L)":   fig.add_subplot(gs[0, 1]),
        "DO (mg/L)":    fig.add_subplot(gs[2, 0]),
        "NH4-N (mg/L)": fig.add_subplot(gs[1, 1]),
        "NO3-N (mg/L)": fig.add_subplot(gs[2, 1]),
    }
    panel_ids_by_param = {
        "DO (mg/L)":    "b",
        "COD (mg/L)":   "c",
        "NH4-N (mg/L)": "d",
        "NO3-N (mg/L)": "e",
    }

    ytick_config = {
        "DO (mg/L)":    ([5.0, 5.6, 6.2, 6.8, 7.4], (4.75, 7.6)),
        "COD (mg/L)":   ([1.0, 2.0, 3.0, 4.0, 5.0], (0.8, 5.2)),
        "NH4-N (mg/L)": ([0.00, 0.03, 0.06, 0.09, 0.12], (-0.005, 0.125)),
        "NO3-N (mg/L)": ([0.0, 0.25, 0.50, 0.75, 1.00], (0.0, 1.05)),
    }

    for pcol, ylabel, bp in params:
        ax = axes_by_param[pcol]
        if pcol in ytick_config:
            yticks, ylim = ytick_config[pcol]
            ax.set_yticks(yticks)
            ax.set_ylim(ylim)

        if pcol == "DO (mg/L)":
            ax.axhline(5.0, color="#737373", lw=0.75, ls=":", alpha=0.8, zorder=1)

        for stn in MRC_STATIONS:
            sub = mrc[mrc["stationName"] == stn]
            if pcol not in sub.columns:
                continue
            annual = sub.groupby("year")[pcol].mean().dropna()
            if len(annual) < 5:
                continue

            x_plot = annual.index.to_numpy(dtype=float)
            y_plot = annual.to_numpy(dtype=float)

            ax.plot(x_plot, y_plot, "-o",
                    color=STATION_CLR[stn], lw=1.0, ms=2.0,
                    alpha=0.85, label=stn, zorder=3)

            x_arr = annual.index.to_numpy(dtype=float)
            y_arr = annual.to_numpy(dtype=float)
            for phase_label, mask in [("Phase1", x_arr <= bp),
                                       ("Phase2", x_arr > bp)]:
                xm, ym = x_arr[mask], y_arr[mask]
                if len(xm) > 5:
                    ts = np.asarray(theilslopes(ym, xm), dtype=float)
                    slope = float(ts[0])
                    intercept = float(ts[1])
                    kt = np.asarray(kendalltau(xm, ym), dtype=float)
                    p = float(kt[1])
                    ax.plot(xm, intercept + slope * xm, "--",
                            color=STATION_CLR[stn], lw=0.7, alpha=0.5)
                    metrics_rows.append({
                        "Station": stn,
                        "Parameter": pcol.split(" (")[0],
                        "Phase": phase_label,
                        "Break_Year": bp,
                        "Theil_Sen_Slope": round(slope, 6),
                        "Kendall_p": round(p, 6),
                        "n_years": int(len(xm)),
                        "mean_value": round(float(ym.mean()), 4),
                    })

        ax.axvline(bp + 0.5, color="#555555", lw=1.1, ls="-", alpha=0.5, zorder=1)
        ax.set_ylabel(ylabel, fontsize=8.5)

        ax.set_xlim(1983, 2024)
        ax.set_xticks([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020])
        ax.tick_params(axis="x", rotation=45, labelsize=8.0)
        ax.tick_params(axis="y", labelsize=8.0)

        is_bottom = pcol in {"DO (mg/L)", "NO3-N (mg/L)"}
        ax.tick_params(axis="x", labelbottom=is_bottom)

        if pcol == "NO3-N (mg/L)":
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=8.0, loc="upper left", ncol=2,
                      frameon=False, handlelength=1.2)
        elif pcol == "NH4-N (mg/L)":
            handles = [
                Line2D([0], [0], color="gray", lw=1.0, ls="--", label="Trend line"),
                Line2D([0], [0], color="#555555", lw=1.1, ls="-", alpha=0.5, label="Regime shift (2004–2005)"),
            ]
            labels = ["Trend line", "Regime shift (2004–2005)"]
            ax.legend(handles, labels, fontsize=8.0, loc="upper left", ncol=1,
                      frameon=False, handlelength=1.2)

        panel_coords = {
            "DO (mg/L)":    (-0.18, 1.06),
            "COD (mg/L)":   (-0.24, 1.06),
            "NH4-N (mg/L)": (-0.24, 1.06),
            "NO3-N (mg/L)": (-0.24, 1.06),
        }
        px, py = panel_coords[pcol]
        panel_label(ax, panel_ids_by_param[pcol], x=px, y=py)

    metrics_df = pd.DataFrame(metrics_rows)
    fig.tight_layout(pad=0.25)
    save_fig(fig, "CEE_Figure_1_Regime_Shift", metrics=metrics_df)


if __name__ == "__main__":
    figure_1()
