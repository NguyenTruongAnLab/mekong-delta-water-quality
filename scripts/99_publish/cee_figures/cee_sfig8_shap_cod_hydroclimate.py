#!/usr/bin/env python3
"""
CEE Supplementary Figure 8 — SHAP COD Baseline vs HydroClimate Comparison
=========================================================================
Cloned directly from `nw_ed_fig5_shap_cod_hydroclimate.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _cee_common import (
    apply_style, MM, SHAP_CSV, VP_CSV,
    panel_label, save_fig,
)

apply_style()


def main() -> None:
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 100 * MM))

    # ── (a) SHAP comparison bars (11 features: LULC-only vs Primary Augmented) ──
    ax = axes[0]
    has_data = False

    if SHAP_CSV.exists():
        shap_df = pd.read_csv(SHAP_CSV)
        cod_base = shap_df[
            (shap_df["Parameter"] == "COD") &
            (shap_df["Feature_Set"] == "LULC-only")
        ].sort_values("Rank")
        cod_hc = shap_df[
            (shap_df["Parameter"] == "COD") &
            (shap_df["Feature_Set"] == "Primary (LULC+Pop+Hydro)")
        ].sort_values("Rank")

        if len(cod_base) > 0 and len(cod_hc) > 0:
            has_data = True
            all_feats = list(dict.fromkeys(
                list(cod_hc["Driver"].values) + list(cod_base["Driver"].values)
            ))
            base_vals = {r["Driver"]: float(r["Mean_Abs_SHAP"]) for _, r in cod_base.iterrows()}
            hc_vals = {r["Driver"]: float(r["Mean_Abs_SHAP"]) for _, r in cod_hc.iterrows()}

            y_pos = np.arange(len(all_feats))
            w = 0.36
            ax.barh(y_pos - w / 2, [base_vals.get(f, 0.0) for f in all_feats], w,
                    color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.3,
                    label="LULC-only (6)", zorder=3)
            ax.barh(y_pos + w / 2, [hc_vals.get(f, 0.0) for f in all_feats], w,
                    color="#F58518", alpha=0.85, edgecolor="white", linewidth=0.3,
                    label="Augmented (11)", zorder=3)

            _rename = {
                "Tidal range (m)": "Tidal range (m)",
                "Discharge (My Thuan, annual z)": "Discharge (My Thuan)",
                "Rice PD": "Rice patch density",
                "Rainfall (dry z)": "Rainfall (dry)",
                "Urban PD": "Urban patch density",
                "Aquaculture ED": "Aquaculture edge density",
                "Population density": "Population density",
                "Rainfall (annual z)": "Rainfall (annual)",
                "Urban %": "Urban (%)",
                "Rice %": "Rice (%)",
                "Aquaculture %": "Aquaculture (%)",
            }
            feat_labels = [_rename.get(f, f) for f in all_feats]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_labels, fontsize=8.0)
            ax.set_xlabel("Mean |SHAP|", fontsize=8.4)
            ax.tick_params(axis="x", labelsize=8.4)
            panel_label(ax, "a", y=1.06)
            ax.legend(fontsize=8.0, frameon=False, loc="lower right")
            ax.invert_yaxis()

            for f in all_feats:
                metrics.append({
                    "Figure": "CEE_SFig8",
                    "Feature": f,
                    "LULC_only_SHAP": round(base_vals.get(f, 0.0), 4),
                    "Augmented_SHAP": round(hc_vals.get(f, 0.0), 4),
                })

    if not has_data:
        ax.text(0.5, 0.5, "SHAP data not available", transform=ax.transAxes,
                ha="center", fontsize=8.4, color="#969696")

    # ── (b) In-sample vs LOSO R² across 4 predictor sets (N = 229) ──
    ax = axes[1]

    if VP_CSV.exists():
        vp = pd.read_csv(VP_CSV)
        comp = vp[vp["Parameter"] == "COD"].copy()
        set_order = ["LULC-only", "LULC+Pop", "Hydro-only", "Primary (LULC+Pop+Hydro)"]
        comp["order"] = comp["Predictor_Set"].map({k: i for i, k in enumerate(set_order)})
        comp = comp.dropna(subset=["order"]).sort_values("order").reset_index(drop=True)

        display_map = {
            "LULC-only": "LULC-only\n(6)",
            "LULC+Pop": "LULC+Pop\n(7)",
            "Hydro-only": "Hydro-only\n(4)",
            "Primary (LULC+Pop+Hydro)": "Combined\n(11)",
        }
        csv_name_map = {
            "LULC-only": "LULC-only",
            "LULC+Pop": "LULC+Pop",
            "Hydro-only": "Hydro-only",
            "Primary (LULC+Pop+Hydro)": "Combined",
        }
        raw_labels = comp["Predictor_Set"].tolist()
        clean = [display_map.get(l, l) for l in raw_labels]
        insample = comp["OLS_R2"].to_numpy(dtype=float)
        loso = comp["LOSO_R2_Ridge"].to_numpy(dtype=float)

        x_c = np.arange(len(comp))
        w_c = 0.35
        ax.bar(x_c - w_c / 2, insample, w_c,
               color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.3,
               label=r"In-sample $R^2$", zorder=3)
        ax.bar(x_c + w_c / 2, loso, w_c,
               color="#F58518", alpha=0.85, edgecolor="white", linewidth=0.3,
               label=r"LOSO $R^2$", zorder=3)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x_c)
        ax.set_xticklabels(clean, fontsize=8.0)
        ax.set_ylabel(r"$R^2$", fontsize=8.4)
        ax.tick_params(axis="both", labelsize=8.4)
        panel_label(ax, "b", y=1.06)
        ax.legend(fontsize=8.0, frameon=False, loc="upper left")
        ax.set_ylim(top=0.50)

        for i in range(len(comp)):
            gap = float(insample[i]) - float(loso[i])
            if gap > 0:
                top_y = max(float(insample[i]), float(loso[i]))
                ax.text(i, top_y + 0.015, rf"$\Delta$={gap:.2f}", fontsize=8.0,
                        color="black", ha="center", va="bottom", weight="bold")
            metrics.append({
                "Figure": "CEE_SFig8",
                "Model": csv_name_map.get(raw_labels[i], raw_labels[i]),
                "InSample_R2": round(float(insample[i]), 4),
                "LOSO_R2": round(float(loso[i]), 4),
                "N": int(comp["N"].iloc[i]),
                "N_features": int(comp["N_features"].iloc[i]),
            })
    else:
        ax.text(0.5, 0.5, "Attribution data not available",
                transform=ax.transAxes, ha="center", fontsize=8, color="#969696")

    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "CEE_SFig_8_SHAP_COD_HydroClimate", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
