#!/usr/bin/env python3
"""
Figure 4 — Landscape Trajectories
===================================
(a) Stacked area chart of mean LULC composition 1990–2020
(b) Key category trajectories + population overlay

Exported metrics CSV:
  - Category, Year, Composition (%), change from 1990 to 2020
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from pathlib import Path

try:
    import geopandas as gpd  # optional
except Exception:  # pragma: no cover
    gpd = None

from _nw_common import (
    apply_style, load_lulc, load_master, load_bt,
    MM, MRC_STATIONS, LULC_CLR, BREAK_YEAR, AO,
    panel_label, save_fig,
    ROOT,
)

apply_style()


def _debug_enabled() -> bool:
    return os.environ.get("NW_DEBUG", "0").strip() not in {"", "0", "false", "False"}


def _gso_mekong_delta_population_density_series(year_min: int = 1990, year_max: int = 2020) -> pd.Series | None:
    """GSO official population for Mekong Delta provinces.
    
    Uses the most authoritative source: GSO Vietnam census data.
    Mekong Delta area ≈ 40,000 km² (approximation for density calculation).
    """
    gso_csv = ROOT / "data" / "Population" / "GSO_Vietnam" / "GSO_population_province_1995_2024.csv"
    if not gso_csv.exists():
        return None

    df = pd.read_csv(gso_csv)
    # Find the Mekong River Delta row
    mekong_row = df[df["Cities, provincies"].str.contains("Mekong River Delta", case=False, na=False)]
    if mekong_row.empty:
        return None

    mekong_row = mekong_row.iloc[0]
    area_km2 = 40000.0  # Approximate Mekong Delta area

    years = list(range(year_min, year_max + 1))
    dens = {}
    for y in years:
        col = f"Total {y}"
        if col in df.columns:
            try:
                pop = float(mekong_row[col])  # Population in thousands
                if np.isfinite(pop) and pop > 0:
                    dens[y] = (pop * 1000) / area_km2  # Convert to people/km²
            except (ValueError, TypeError):
                continue

    if not dens:
        return None
    return pd.Series(dens).sort_index()


def _province_population_density_series(year_min: int = 1990, year_max: int = 2020) -> pd.Series | None:
    """Province-based population density time series for the Mekong Delta.

    Uses reconstruction gpkg with pop_YYYY columns. Density is computed as:
      sum(pop_YYYY across provinces) / sum(province area)
    which avoids spikes caused by changing station coverage in buffer-based products.
    """
    if gpd is None:
        return None

    gpkg = (ROOT / "data" / "Population" / "outputs" / "reconstruction"
            / "vietnam_adm1_population_1985_2025_mekong_delta.gpkg")
    if not gpkg.exists():
        return None

    gdf = gpd.read_file(gpkg)
    if "geometry" not in gdf.columns:
        return None

    # Total area (km^2) using an equal-area projection.
    try:
        area_km2 = float(gdf.to_crs("EPSG:6933").geometry.area.sum() / 1e6)
    except Exception:
        area_km2 = float(gdf.to_crs("EPSG:3857").geometry.area.sum() / 1e6)
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return None

    years = list(range(year_min, year_max + 1))
    dens = {}
    for y in years:
        col = f"pop_{y}"
        if col not in gdf.columns:
            continue
        total_pop = float(pd.to_numeric(gdf[col], errors="coerce").fillna(0).sum())
        if total_pop <= 0:
            continue
        dens[y] = total_pop / area_km2

    if not dens:
        return None
    return pd.Series(dens).sort_index()


def _stationbuffer_population_density_series(year_min: int = 1990, year_max: int = 2020) -> pd.Series | None:
    pop_path = AO / "population_stationbuffer_1990_2022_16km_latest.csv"
    if not pop_path.exists():
        return None
    pop_df = pd.read_csv(pop_path)
    if not {"year", "pop_density"}.issubset(pop_df.columns):
        return None
    # Use median across stations per year to reduce spikes if station coverage changes.
    if "stationName" in pop_df.columns:
        s = pop_df.groupby(["year", "stationName"])["pop_density"].mean().reset_index()
        pop_yr = s.groupby("year")["pop_density"].median().dropna()
    else:
        pop_yr = pop_df.groupby("year")["pop_density"].median().dropna()
    pop_yr = pop_yr[(pop_yr.index >= year_min) & (pop_yr.index <= year_max)]
    return pop_yr if len(pop_yr) > 3 else None


def figure_4() -> None:
    lulc = load_lulc()
    lulc_v = lulc[(lulc["year"] >= 1990) & (lulc["year"] <= 2020)].copy()

    col_map = {
        "Rice":        "Rice_Cultivation_composition_pct",
        "Upland Crops":"Other_Crops_composition_pct",
        "Aquaculture": "Aquaculture_composition_pct",
        "Residential": "Residential_Urban_composition_pct",
        "Forest":      "Forest_composition_pct",
        "Wetland":     "Wetland_composition_pct",
        "Water":       "Water_Bodies_composition_pct",
    }

    years = sorted(lulc_v["year"].unique())
    cats = list(col_map.keys())
    data = {}
    for cat, col in col_map.items():
        if col in lulc_v.columns:
            means = lulc_v.groupby("year")[col].mean()
            data[cat] = np.array([means.get(y, 0) for y in years])
        else:
            data[cat] = np.zeros(len(years))

    # Build metrics
    metrics_rows = []
    y_stack = np.array([data[cat] for cat in cats])
    totals = y_stack.sum(axis=0)
    totals[totals == 0] = 1
    y_stack_pct = y_stack / totals * 100

    for ci, cat in enumerate(cats):
        for yi, yr in enumerate(years):
            metrics_rows.append({
                "Category": cat, "Year": yr,
                "Mean_Composition_Pct": round(float(y_stack_pct[ci, yi]), 2),
                "Raw_Value": round(float(data[cat][yi]), 4),
            })
        # Summary: change 1990 to 2020
        first_val = float(y_stack_pct[ci, 0])
        last_val  = float(y_stack_pct[ci, -1])
        metrics_rows.append({
            "Category": cat, "Year": "change_1990_2020",
            "Mean_Composition_Pct": round(last_val - first_val, 2),
            "Raw_Value": None,
        })

    fig, axes = plt.subplots(2, 1, figsize=(180 * MM, 130 * MM))

    # ── (a) Stacked area ──
    ax = axes[0]
    ax.stackplot(years, y_stack_pct,
                 labels=cats,
                 colors=[LULC_CLR[c] for c in cats],
                 alpha=0.85, linewidth=0.3, edgecolor="white")
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Composition (%)")
    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    ax.text(BREAK_YEAR - 0.5, 102, "regime shift", fontsize=5, color="#525252",
            ha="right", va="bottom", style="italic")
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labs[::-1], fontsize=5.5,
              loc="center left", bbox_to_anchor=(0.01, 0.5),
              ncol=1, frameon=True, framealpha=0.9, edgecolor="none")
    panel_label(ax, "a")

    # ── (b) Key trajectories ──
    ax = axes[1]
    focus = ["Aquaculture", "Residential", "Rice", "Wetland"]
    totals_arr = np.array([sum(data[c][i] for c in cats) for i in range(len(years))])
    totals_arr[totals_arr == 0] = 1

    for cat in focus:
        pct = data[cat] / totals_arr * 100
        ax.plot(years, pct, "-o", color=LULC_CLR[cat], lw=1.4,
                ms=2.5, alpha=0.9, label=cat, zorder=3)

    # Population overlay
    # Priority: GSO official data > province reconstruction > station-buffer
    pop_yr = _gso_mekong_delta_population_density_series(1990, 2020)
    pop_source = "GSO-official"

    if pop_yr is None or len(pop_yr) <= 3:
        pop_yr = _province_population_density_series(1990, 2020)
        pop_source = "province-reconstruction"

    if pop_yr is None or len(pop_yr) <= 3:
        pop_yr = _stationbuffer_population_density_series(1990, 2020)
        pop_source = "station-buffer"

    # Fallback: old Ben Tre-derived series (may only start ~2005)
    if pop_yr is None or len(pop_yr) <= 3:
        bt = load_bt()
        if "pop_density" in bt.columns:
            pop_yr = bt.groupby("year")["pop_density"].mean().dropna()
            pop_yr = pop_yr[(pop_yr.index >= 1990) & (pop_yr.index <= 2020)]
            pop_source = "bentre-fallback"

    if pop_yr is not None and len(pop_yr) > 3:
        ax2 = ax.twinx()
        ax2.plot(pop_yr.index, pop_yr.values, "s--", color="#737373",
                 lw=1.0, ms=2.5, alpha=0.75, label="Pop. density", zorder=2)
        ax2.set_ylabel(r"Pop. density (people $\mathrm{km^{-2}}$)", fontsize=7, color="#737373")
        ax2.tick_params(axis="y", labelcolor="#737373")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("#bdbdbd")
        ax2.spines["right"].set_linewidth(0.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        # Single-row legend above the plot for clean layout
        ax.legend(h1 + h2, l1 + l2, fontsize=5.5,
                  loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
                  ncol=len(h1 + h2), frameon=True, framealpha=0.9, edgecolor="none")

        if _debug_enabled():
            print(f"  Figure4: population overlay source = {pop_source}, n_years={len(pop_yr)}")

    ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year")
    ax.set_ylabel("Composition (%)")
    if pop_yr is None or len(pop_yr) <= 3:
        ax.legend(fontsize=6,
                  loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
                  ncol=len(focus), frameon=True, framealpha=0.9, edgecolor="none")
    panel_label(ax, "b")

    fig.tight_layout(h_pad=2.0, rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, "Figure_4_Landscape_Trajectories",
             metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_4()
