#!/usr/bin/env python3
"""
Extended Data Figures 5-6 — LULC & Attribution
================================================
ED Fig 5: Station-level LULC trajectory clustering
ED Fig 6: SHAP COD baseline vs HydroClimate + model comparison

All export companion metrics CSVs.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from _nw_common import (
    apply_style, load_lulc, load_wq, MM, LULC_CLR, BREAK_YEAR,
    SHAP_CSV, SHAP_HC_CSV, ATTR_HC_CSV, LTG_CSV, ROOT,
    panel_label, save_fig, MRC_STATIONS,
)

apply_style()


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 7 — LULC Trajectory Groups (comprehensive, 3-panel design
#   inspired by 03_07_station_clustering.py)
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_7() -> None:
    """Multi-panel LULC clustering figure: map + dendrogram + trajectories."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    import warnings

    try:
        import geopandas as gpd
    except Exception:  # pragma: no cover
        gpd = None

    lulc = load_lulc()
    lulc_v = lulc[(lulc["year"] >= 1990) & (lulc["year"] <= 2020)].copy()
    lulc_v = lulc_v[~lulc_v["stationName"].isin(MRC_STATIONS)]
    metrics = []

    col_map_full = {
        "Residential": "Residential_Urban_composition_pct",
        "Rice":        "Rice_Cultivation_composition_pct",
        "Upland Crops":"Other_Crops_composition_pct",
        "Aquaculture": "Aquaculture_composition_pct",
        "Forest":      "Forest_composition_pct",
        "Wetland":     "Wetland_composition_pct",
    }
    categories_full = list(col_map_full.keys())
    display_categories = ["Residential", "Rice", "Upland Crops", "Aquaculture", "Wetland"]

    df_proc = lulc_v[["stationName", "year"]].copy()
    for name, col in col_map_full.items():
        df_proc[name] = lulc_v[col].values if col in lulc_v.columns else 0

    years = sorted(df_proc["year"].unique())
    df_pivot = df_proc.pivot(index="stationName", columns="year", values=categories_full)
    df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
    df_pivot.dropna(inplace=True)

    if df_pivot.empty:
        print("  WARNING: No LULC data for clustering, skipping ED Figure 7")
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_pivot)
    n_clusters = 4

    # K-means for cluster assignment (correct grouping)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_pivot["Cluster"] = kmeans.fit_predict(scaled)

    # Ward linkage for dendrogram (supporting visualization)
    linked = linkage(scaled, method="ward")

    cluster_colors = {0: "#D55E00", 1: "#0072B2", 2: "#009E73", 3: "#E69F00"}
    cluster_trends = df_pivot.groupby("Cluster").mean()

    cluster_labels = {}
    for cid in range(n_clusters):
        if cid not in cluster_trends.index:
            continue
        trends = cluster_trends.loc[cid].filter(like="_").values
        n_full_cols = len(categories_full) * len(years)
        full_trends = trends[:n_full_cols].reshape(len(categories_full), len(years))
        changes = full_trends[:, -1] - full_trends[:, 0]
        inc = [(categories_full[i], c) for i, c in enumerate(changes) if c > 0]
        dec = [(categories_full[i], c) for i, c in enumerate(changes) if c < 0]
        top_inc = sorted(inc, key=lambda x: x[1], reverse=True)
        top_dec = sorted(dec, key=lambda x: x[1])
        parts = []
        if top_inc and top_inc[0][1] > 1.5:
            parts.append(f"\u2191{top_inc[0][0]}")
        if top_dec and abs(top_dec[0][1]) > 1.5:
            parts.append(f"\u2193{top_dec[0][0]}")
        if parts:
            cluster_labels[cid] = f"G{cid + 1}: {', '.join(parts)}"
        else:
            means = np.mean(full_trends, axis=1)
            cluster_labels[cid] = f"G{cid + 1}: {categories_full[np.argmax(means)]}"

    wq = load_wq()
    coords = wq.groupby("stationName")[["latitude", "longitude"]].mean().reset_index()
    coords = coords[~coords["stationName"].isin(MRC_STATIONS)]
    map_data = df_pivot.reset_index().merge(coords, on="stationName", how="left").dropna(subset=["longitude", "latitude"])

    # ── Layout ──
    fig = plt.figure(figsize=(180 * MM, 125 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2],
                           height_ratios=[1.1, 0.9], wspace=0.25, hspace=0.18)

    # ── (A) Map — simple fixed-offset labels, no adjustText ──
    ax_map = fig.add_subplot(gs[0, 0])

    water_bodies_path = ROOT / "data" / "spatial" / "waterbody_Mekong_Saigon_Water.gpkg"
    if gpd is not None and water_bodies_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                wb = gpd.read_file(water_bodies_path)
                if wb.crs is None:
                    wb = wb.set_crs("EPSG:4326")
                wb = wb.to_crs("EPSG:4326")
                wb.geometry = wb.geometry.simplify(tolerance=0.001)
                wb.plot(ax=ax_map, color="#a6cee3", edgecolor="none", zorder=1)
            except Exception as exc:
                print(f"  WARNING: water body rendering failed: {exc}")

    # Hand-tuned label offsets for dense upstream stations
    _label_offsets = {
        "My Thuan": (-3, 5),
        "CC01":     (-12, -6),
        "HL01":     (-12, 3),
        "BL01":     (3, 5),
        "MT01":     (3, -6),
        "HL02":     (-12, 0),
        "BL02":     (3, -6),
    }
    texts = []
    for cid in range(n_clusters):
        cdata = map_data[map_data["Cluster"] == cid]
        if cdata.empty:
            continue
        ax_map.scatter(cdata["longitude"], cdata["latitude"],
                       s=42, c=cluster_colors[cid], edgecolors="black",
                       linewidths=0.5, zorder=10, alpha=1,
                       label=cluster_labels.get(cid, f"G{cid + 1}"))
        for _, row in cdata.iterrows():
            sn = str(row["stationName"])
            ox, oy = _label_offsets.get(sn, (3, 2))
            ax_map.annotate(sn,
                            xy=(float(row["longitude"]), float(row["latitude"])),
                            xytext=(ox, oy), textcoords="offset points",
                            fontsize=4, fontweight="bold", zorder=20)

    if len(map_data) > 0:
        min_lon, max_lon = float(map_data["longitude"].min()), float(map_data["longitude"].max())
        min_lat, max_lat = float(map_data["latitude"].min()), float(map_data["latitude"].max())
        ax_map.set_xlim(min_lon - 0.1, max_lon + 0.1)
        ax_map.set_ylim(min_lat - 0.1, max_lat + 0.1)

    ax_map.set_axis_off()
    ax_map.legend(fontsize=5.5, loc="lower left", frameon=True, framealpha=0.9,
                  edgecolor="none", handlelength=1.2,
                  prop={"size": 5.5, "weight": "bold"})
    ax_map.text(-0.06, 1.06, "(a)", transform=ax_map.transAxes,
               fontsize=8, fontweight="bold", va="top", ha="left")

    # ── (B) HCA Dendrogram — Ward linkage, leaf labels colored by K-means ──
    ax_hca = fig.add_subplot(gs[1, 0])
    cut_dist = linked[-(n_clusters - 1), 2]
    station_names = df_pivot.index.tolist()
    # Map HCA subclusters → k-means cluster colors for branch coloring
    from collections import Counter
    from scipy.cluster.hierarchy import set_link_color_palette
    hca_raw = fcluster(linked, t=n_clusters, criterion="maxclust")
    hca_to_km = {}
    for hc in sorted(set(hca_raw)):
        members = [station_names[i] for i, h in enumerate(hca_raw) if h == hc]
        km_ids = [int(df_pivot.loc[m, "Cluster"]) for m in members if m in df_pivot.index]
        if km_ids:
            hca_to_km[hc] = Counter(km_ids).most_common(1)[0][0]
    palette = [cluster_colors[hca_to_km.get(i + 1, 0)] for i in range(n_clusters)]
    set_link_color_palette(palette)
    dend = dendrogram(linked, ax=ax_hca, labels=station_names,
                      leaf_rotation=90, leaf_font_size=6,
                      color_threshold=cut_dist,
                      above_threshold_color="#999999")
    set_link_color_palette(None)
    ax_hca.axhline(y=cut_dist, color="#999999", ls="--", lw=0.7, alpha=0.6)
    ax_hca.set_ylabel("Ward distance", fontsize=8)
    ax_hca.tick_params(labelsize=6)
    ax_hca.grid(axis="y", ls="--", alpha=0.5)
    label_clr = df_pivot["Cluster"].map(cluster_colors).to_dict()
    for lbl in ax_hca.get_xticklabels():
        if lbl.get_text() in label_clr:
            lbl.set_color(label_clr[lbl.get_text()])
    ax_hca.text(-0.06, 1.06, "(b)", transform=ax_hca.transAxes,
               fontsize=8, fontweight="bold", va="top", ha="left")

    # ── (C) Trajectories: 5 stacked subplots (no Forest) ──
    gs_tr = gridspec.GridSpecFromSubplotSpec(len(display_categories), 1,
                                             subplot_spec=gs[:, 1], hspace=0.12)
    # Best legend position per subplot (avoid overlap with data lines)
    _legend_loc = {
        "Residential": "upper left",
        "Rice": "upper right",
        "Upland Crops": "lower left",
        "Aquaculture": "upper left",
        "Wetland": "upper right",
    }
    for ci, cat in enumerate(display_categories):
        ax_tr = fig.add_subplot(gs_tr[ci, 0])
        # Identify groups whose label mentions this category
        relevant_cids = [cid for cid in range(n_clusters)
                         if cat in cluster_labels.get(cid, "")]
        for cid in range(n_clusters):
            tcols = [f"{cat}_{yr}" for yr in years]
            available = [c for c in tcols if c in cluster_trends.columns]
            if not available:
                continue
            vals = cluster_trends.loc[cid, available].values
            yr_subset = [int(c.split("_")[-1]) for c in available]
            show_label = cid in relevant_cids
            ax_tr.plot(yr_subset, vals, linestyle="-",
                       color=cluster_colors[cid], lw=1.2,
                       label=cluster_labels.get(cid, f"G{cid + 1}") if show_label else None)

        ax_tr.set_ylabel(f"{cat}\n(%)", fontsize=7, fontweight="bold")
        ax_tr.grid(True, ls=":", alpha=0.5)
        ax_tr.tick_params(labelsize=6)
        if relevant_cids:
            ax_tr.legend(fontsize=5, loc=_legend_loc.get(cat, "best"), ncol=1,
                         frameon=True, framealpha=0.9, edgecolor="none",
                         prop={"size": 5, "weight": "bold"})
        if ci < len(display_categories) - 1:
            ax_tr.tick_params(axis="x", labelbottom=False)
        if ci == 0:
            ax_tr.text(-0.08, 1.06, "(c)", transform=ax_tr.transAxes,
                       fontsize=8, fontweight="bold", va="top", ha="left")

    for cid in range(n_clusters):
        n_stns = int((df_pivot["Cluster"] == cid).sum())
        metrics.append({"Figure": "ED7", "Cluster": cid,
                        "Label": cluster_labels[cid], "N_stations": n_stns})

    fig.tight_layout()
    save_fig(fig, "ED_Figure_5_LULC_Trajectory_Groups", pd.DataFrame(metrics))


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 8 — SHAP COD & DO: LULC-only vs Augmented (Hydro+Climate+Pop)
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_8() -> None:
    """
    SHAP comparison for COD and model performance.
    
    The 'Augmented' model includes:
      - LULC (composition and configuration)
      - Hydrology (discharge, tidal range, HFI)
      - Climate (rainfall, dry/wet season indices)
      - Population density
    """
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 75 * MM))

    # ── (a) SHAP comparison bars ──
    ax = axes[0]
    has_data = False

    if SHAP_CSV.exists() and SHAP_HC_CSV.exists():
        shap_base = pd.read_csv(SHAP_CSV)
        shap_hc = pd.read_csv(SHAP_HC_CSV)

        cod_base = shap_base[
            (shap_base["Parameter"] == "COD") &
            (shap_base["Feature_Set"] == "LULC-only")
        ].sort_values("Rank").head(10)
        cod_hc = shap_hc[shap_hc["Parameter"] == "COD"].head(10)

        if len(cod_base) > 0 and len(cod_hc) > 0:
            has_data = True
            all_feats = list(dict.fromkeys(
                list(cod_hc["Driver"].values) + list(cod_base["Driver"].values)))[:10]
            base_vals = {r["Driver"]: r["Mean_Abs_SHAP"] for _, r in cod_base.iterrows()}
            hc_vals = {r["Driver"]: r["Mean_Abs_SHAP"] for _, r in cod_hc.iterrows()}

            y_pos = np.arange(len(all_feats))
            w = 0.35
            ax.barh(y_pos - w / 2, [base_vals.get(f, 0) for f in all_feats], w,
                    color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.3,
                    label="LULC-only", zorder=3)
            ax.barh(y_pos + w / 2, [hc_vals.get(f, 0) for f in all_feats], w,
                    color="#F58518", alpha=0.85, edgecolor="white", linewidth=0.3,
                    label="Augmented", zorder=3)

            feat_labels = []
            _rename = {
                "TidalRange_MyThuan_Dry_m": "Tidal range (My Thuan, dry)",
                "Rice PD": "Rice patch density",
                "Q_MyThuan_AnnualMean_z": "Discharge (My Thuan, annual)",
                "Q_MyThuan_DrySeasonMean_z": "Discharge (My Thuan, dry)",
                "Aquaculture ED": "Aquaculture edge density",
                "Rain_Annual_z": "Rainfall (annual)",
                "Rice %": "Rice (%)",
                "Urban PD": "Urban patch density",
                "Urban %": "Urban (%)",
                "Aquaculture %": "Aquaculture (%)",
                "Rain_Dry_z": "Rainfall (dry)",
            }
            for f in all_feats:
                feat_labels.append(_rename.get(f, f))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_labels, fontsize=5.5)
            ax.set_xlabel("Mean |SHAP|", fontsize=7)
            ax.set_title("(a) Predictor importance for organic pollution (COD)",
                         fontsize=7, fontweight="bold", pad=3)
            ax.legend(fontsize=5.5, frameon=True, framealpha=0.9,
                      edgecolor="none", loc="lower right")
            ax.invert_yaxis()

            # Annotate Augmented model composition
            ax.text(0.02, 0.02, "Augmented = LULC + Hydro + Rainfall + Pop.",
                    fontsize=4.5, color="#666666", style="italic",
                    transform=ax.transAxes, va="bottom", ha="left")

            for f in all_feats:
                metrics.append({"Figure": "ED8", "Feature": f,
                                "LULC_only_SHAP": round(base_vals.get(f, 0), 4),
                                "Augmented_SHAP": round(hc_vals.get(f, 0), 4)})

    if not has_data:
        ax.text(0.5, 0.5, "SHAP data not available", transform=ax.transAxes,
                ha="center", fontsize=8, color="#969696")

    # ── (b) In-sample vs LOSO R² from attribution_with_hydroclimate_comparison CSV ──
    ax = axes[1]

    if ATTR_HC_CSV.exists():
        comp = pd.read_csv(ATTR_HC_CSV)
        labels = comp["Predictor_Set"].tolist()
        insample = comp["OLS_R2"].values
        loso = comp["LOSO_R2"].values

        x_c = np.arange(len(comp))
        w_c = 0.35
        ax.bar(x_c - w_c / 2, insample, w_c,
               color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.3,
               label=r"In-sample R$^2$", zorder=3)
        ax.bar(x_c + w_c / 2, loso, w_c,
               color="#F58518", alpha=0.85, edgecolor="white", linewidth=0.3,
               label=r"LOSO R$^2$", zorder=3)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x_c)
        clean = [l.replace("_", " ") for l in labels]
        ax.set_xticklabels(clean, fontsize=6.5)
        ax.set_ylabel(r"R$^2$", fontsize=7)
        ax.set_title("(b) Spatial transferability across predictor sets (COD)",
                     fontsize=7, fontweight="bold", pad=3)
        ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")

        for i in range(len(comp)):
            gap = float(insample[i]) - float(loso[i])
            if gap > 0.1:
                mid_y = (float(insample[i]) + float(loso[i])) / 2
                ax.text(i + 0.4, mid_y, rf"$\Delta$={gap:.2f}", fontsize=5,
                        color="#d62728", ha="left", va="center")
            metrics.append({"Figure": "ED8", "Model": labels[i],
                            "InSample_R2": round(float(insample[i]), 4),
                            "LOSO_R2": round(float(loso[i]), 4),
                            "N": int(comp["N"].iloc[i])})
    else:
        ax.text(0.5, 0.5, "Attribution data not available",
                transform=ax.transAxes, ha="center", fontsize=8, color="#969696")

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "ED_Figure_6_SHAP_COD_HydroClimate", pd.DataFrame(metrics))


if __name__ == "__main__":
    print("=" * 65)
    print("Extended Data Figures 7-8: LULC & Attribution")
    print("=" * 65)
    ed_figure_7()
    ed_figure_8()
    print("Done.")
