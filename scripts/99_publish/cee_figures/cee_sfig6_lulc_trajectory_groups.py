#!/usr/bin/env python3
"""
CEE Supplementary Figure 6 — LULC Trajectory Groups and Hierarchical Clustering
===============================================================================
Cloned directly from `nw_ed_fig4_lulc_trajectory_groups.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette

from _cee_common import (
    apply_style, load_lulc, load_wq, MM,
    ROOT, panel_label, save_fig, MRC_STATIONS,
)

apply_style()

try:
    import geopandas as gpd
except Exception:
    gpd = None


def main() -> None:
    lulc = load_lulc()
    lulc_v = lulc[(lulc["year"] >= 1990) & (lulc["year"] <= 2020)].copy()
    lulc_v = lulc_v[~lulc_v["stationName"].isin(MRC_STATIONS)]
    metrics = []

    col_map_full = {
        "Residential":  "Residential_Urban_composition_pct",
        "Rice":         "Rice_Cultivation_composition_pct",
        "Upland crops": "Other_Crops_composition_pct",
        "Aquaculture":  "Aquaculture_composition_pct",
        "Forest":       "Forest_composition_pct",
        "Wetland":      "Wetland_composition_pct",
    }
    categories_full = list(col_map_full.keys())
    display_categories = ["Residential", "Rice", "Upland crops", "Aquaculture", "Wetland"]

    df_proc = lulc_v[["stationName", "year"]].copy()
    for name, col in col_map_full.items():
        df_proc[name] = lulc_v[col].values if col in lulc_v.columns else 0

    years = sorted(df_proc["year"].unique())
    df_pivot = df_proc.pivot(index="stationName", columns="year", values=categories_full)
    df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
    df_pivot.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_pivot)
    n_clusters = 4
    linked = linkage(scaled, method="ward")
    df_pivot["Cluster"] = fcluster(linked, t=n_clusters, criterion="maxclust") - 1

    cluster_colors = {0: "#D55E00", 1: "#0072B2", 2: "#009E73", 3: "#E69F00"}
    cluster_trends = df_pivot.groupby("Cluster").mean()

    cluster_labels = {
        0: "G1: \u2191Aquaculture, \u2193Wetland/Rice",
        1: "G2: \u2191Aquaculture, \u2193Wetland (coastal)",
        2: "G3: \u2191Residential, \u2193Rice",
        3: "G4: \u2191Residential/Aqua, \u2193Rice",
    }

    wq = load_wq()
    coords = wq.groupby("stationName")[["latitude", "longitude"]].mean().reset_index()
    coords = coords[~coords["stationName"].isin(MRC_STATIONS)]
    map_data = df_pivot.reset_index().merge(coords, on="stationName", how="left").dropna(subset=["longitude", "latitude"])

    fig = plt.figure(figsize=(180 * MM, 125 * MM), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1.1, 0.9])

    ax_map = fig.add_subplot(gs[0, 0])
    water_bodies_path = ROOT / "data" / "spatial" / "waterbody_Mekong_Saigon_Water.gpkg"
    if gpd is not None and water_bodies_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from shapely.geometry import box
            wb = gpd.read_file(water_bodies_path)
            if wb.crs is None:
                wb = wb.set_crs("EPSG:4326")
            wb = wb.to_crs("EPSG:4326")
            if len(map_data) > 0:
                min_lon, max_lon = float(map_data["longitude"].min()), float(map_data["longitude"].max())
                min_lat, max_lat = float(map_data["latitude"].min()), float(map_data["latitude"].max())
                wb = gpd.clip(wb, box(min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1)).copy()
            wb.geometry = wb.geometry.simplify(tolerance=0.001, preserve_topology=True)
            wb.plot(ax=ax_map, color="#a6cee3", edgecolor="none", zorder=1)

    _label_offsets = {
        "My Thuan": (-4, 5, "right", "bottom"),
        "CC01": (-5, -4, "right", "top"), "HL01": (-5, 3, "right", "bottom"),
        "BL01": (0, -6, "center", "top"), "MT01": (0, 6, "center", "bottom"),
        "HL02": (-5, -3, "right", "top"), "BL02": (5, -2, "left", "top"),
        "MT02": (5, 3, "left", "bottom"), "BL03": (5, -1, "left", "center"),
        "MT03": (5, 3, "left", "bottom"), "BL04": (-6, -4, "right", "top"),
        "MT04": (5, 3, "left", "bottom"), "MT05": (5, -4, "left", "top"),
        "CC02": (-5, -3, "right", "top"), "CC03": (-5, -3, "right", "top"),
        "CC04": (-5, -3, "right", "top"), "CC05": (5, -3, "left", "top"),
        "HL03": (-5, -3, "right", "top"), "HL04": (-5, -3, "right", "top"),
        "HL05": (5, -3, "left", "top"),
    }
    for cid in range(n_clusters):
        cdata = map_data[map_data["Cluster"] == cid]
        if cdata.empty:
            continue
        ax_map.scatter(cdata["longitude"], cdata["latitude"], s=42, c=cluster_colors[cid],
                       edgecolors="black", linewidths=0.5, zorder=10, alpha=1,
                       label=cluster_labels.get(cid, f"G{cid + 1}"))
        for _, row in cdata.iterrows():
            sn = str(row["stationName"])
            ox, oy, ha_val, va_val = _label_offsets.get(sn, (4, 2, "left", "bottom"))
            ax_map.annotate(sn, xy=(float(row["longitude"]), float(row["latitude"])),
                            xytext=(ox, oy), textcoords="offset points", ha=ha_val, va=va_val,
                            fontsize=5.8, fontweight="bold", zorder=20,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    if len(map_data) > 0:
        ax_map.set_xlim(float(map_data["longitude"].min()) - 0.1, float(map_data["longitude"].max()) + 0.1)
        ax_map.set_ylim(float(map_data["latitude"].min()) - 0.1, float(map_data["latitude"].max()) + 0.1)
    ax_map.set_axis_off()
    panel_label(ax_map, "a", x=-0.06, y=1.06)

    ax_hca = fig.add_subplot(gs[1, 0])
    cut_dist = linked[-(n_clusters - 1), 2]
    station_names = df_pivot.index.tolist()
    set_link_color_palette([cluster_colors[i] for i in range(n_clusters)])
    dendrogram(linked, ax=ax_hca, labels=station_names, leaf_rotation=90, leaf_font_size=7.8,
               color_threshold=cut_dist, above_threshold_color="#999999")
    set_link_color_palette(None)
    ax_hca.axhline(y=cut_dist, color="#999999", ls="--", lw=0.7, alpha=0.6)
    ax_hca.set_ylabel("Ward distance", fontsize=8)
    ax_hca.tick_params(labelsize=7.8)
    ax_hca.grid(False)
    label_clr = df_pivot["Cluster"].map(cluster_colors).to_dict()
    for lbl in ax_hca.get_xticklabels():
        if lbl.get_text() in label_clr:
            lbl.set_color(label_clr[lbl.get_text()])
    panel_label(ax_hca, "b", x=-0.13, y=1.07)

    gs_tr = gridspec.GridSpecFromSubplotSpec(
        len(display_categories) + 1, 1, subplot_spec=gs[:, 1], hspace=0.10,
        height_ratios=[0.26] + [1.0] * len(display_categories),
    )
    ax_leg = fig.add_subplot(gs_tr[0, 0])
    ax_leg.axis("off")
    panel_label(ax_leg, "c", x=-0.16, y=0.95)

    for ci, cat in enumerate(display_categories):
        ax_tr = fig.add_subplot(gs_tr[ci + 1, 0])
        for cid in range(n_clusters):
            tcols = [f"{cat}_{yr}" for yr in years if f"{cat}_{yr}" in cluster_trends.columns]
            if not tcols:
                continue
            vals = cluster_trends.loc[cid, tcols].values
            yr_subset = [int(c.split("_")[-1]) for c in tcols]
            ax_tr.plot(yr_subset, vals, linestyle="-", color=cluster_colors[cid], lw=1.2,
                       label=cluster_labels.get(cid, f"G{cid + 1}"))
        ax_tr.set_ylabel(f"{cat}\n(%)", fontsize=7, fontweight="bold")
        ax_tr.set_xlim(1990, 2020)
        ax_tr.set_xticks(range(1990, 2025, 5))
        ax_tr.grid(False)
        ax_tr.tick_params(labelsize=7.8)
        if ci == 0:
            ax_tr.set_ylim(0, 10.5)
            handles, labels = ax_tr.get_legend_handles_labels()
            leg = ax_leg.legend(handles, labels, fontsize=6.1, loc="center",
                                bbox_to_anchor=(0.46, 0.45), ncol=2, frameon=False,
                                handlelength=1.1, handletextpad=0.28, columnspacing=1.0, labelspacing=0.25)
            leg.set_in_layout(False)
        if ci < len(display_categories) - 1:
            ax_tr.tick_params(axis="x", labelbottom=False)

    for cid in range(n_clusters):
        metrics.append({"Figure": "SFig6", "Cluster": cid,
                        "Label": cluster_labels[cid],
                        "N_stations": int((df_pivot["Cluster"] == cid).sum())})

    save_fig(fig, "CEE_SFig_6_LULC_Trajectory_Groups", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
