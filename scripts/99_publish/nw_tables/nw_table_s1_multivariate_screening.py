#!/usr/bin/env python3
"""
Supplementary Table S1: Multivariate Screening Diagnostics
==========================================================

Computes from RAW DATA:
  1. RDA (Redundancy Analysis) biplot scores for constrained axes 1–2
  2. Variance Inflation Factors (VIF) for the 6 baseline LULC predictors

Data sources: raw WQ CSV + LULC metrics CSV → merge → RDA + VIF

Output:
  manuscript/tables/Supplementary_Table_S1_Multivariate_Screening.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skbio.stats.ordination import rda
from statsmodels.stats.outliers_influence import variance_inflation_factor

from _nw_common_tables import (
    merge_wq_lulc, prepare_xy, MANU_TABLES, write_table_md, write_metrics_csv,
)


def build_s1() -> pd.DataFrame:
    """
    Compute RDA biplot scores (axes 1–2) and VIF from raw data.

    Steps:
      1. Load raw WQ + LULC → merge → analysis-ready DataFrame
      2. Extract Y (WQ) and X (LULC) matrices
      3. Run RDA via skbio on standardized X, scaled Y → extract biplot scores
      4. Compute VIF for each predictor
      5. Merge biplot + VIF → sorted by vector length
    """
    # ── Step 1: Load and merge raw data ──────────────────────────────
    merged = merge_wq_lulc(bentre_only=True)
    Y, X, _ = prepare_xy(merged, use_display_labels=True)

    print(f"  Data: {len(Y)} station-years, {X.shape[1]} LULC predictors, "
          f"{Y.shape[1]} WQ parameters")

    # ── Step 2: RDA ──────────────────────────────────────────────────
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    ord_result = rda(Y, X_scaled, scale_Y=True, scaling=2)

    # Extract biplot scores for constrained axes 1–2
    biplot = ord_result.biplot_scores
    biplot_df = pd.DataFrame(
        biplot.values[:, :2],
        columns=["RDA1", "RDA2"],
        index=biplot.index,
    )
    biplot_df["Vector Length"] = np.sqrt(biplot_df["RDA1"]**2 + biplot_df["RDA2"]**2)
    biplot_df.index.name = "LULC Driver"
    biplot_df = biplot_df.reset_index()

    # Proportion explained
    prop = ord_result.proportion_explained
    print(f"  RDA1 explains {prop.iloc[0]:.1%}, RDA2 explains {prop.iloc[1]:.1%}"
          f"  (total: {prop.iloc[0]+prop.iloc[1]:.1%})")

    # ── Step 3: VIF ──────────────────────────────────────────────────
    X_vif = X.values.astype(float)
    vif_data = pd.DataFrame({
        "LULC Driver": X.columns,
        "VIF": [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])],
    })

    # ── Step 4: Merge and sort ───────────────────────────────────────
    result = pd.merge(biplot_df, vif_data, on="LULC Driver", how="left")
    result = result.sort_values("Vector Length", ascending=False).reset_index(drop=True)

    return result


def main():
    print("=" * 60)
    print("Table S1: Multivariate Screening Diagnostics")
    print("=" * 60)

    df = build_s1()

    # Write Markdown table
    md_content = df.to_markdown(index=False, floatfmt=".3f")
    write_table_md(
        content=md_content,
        out_path=MANU_TABLES / "Supplementary_Table_S1_Multivariate_Screening.md",
        caption=(
            "Supplementary Table S1. Redundancy analysis (RDA) biplot scores and "
            "variance inflation factors (VIF) for six LULC predictors. The two "
            "configuration metrics with the longest biplot vectors (Aquaculture edge "
            "density [ED] and Urban patch density [PD]) exhibit persistent "
            "multicollinearity (VIF = 91.4 and 73.9, respectively), exceeding the "
            "conventional threshold of 10. This collinearity reflects the inherent "
            "correlation between composition and configuration within the same "
            "land-cover class: fragmentation necessarily scales with areal extent. "
            "These results motivate the orthogonalization sensitivity test in "
            "@tbl-s2-shap; if SHAP rankings shifted substantially after "
            "residualizing configuration on composition, the baseline attribution "
            "would be unreliable."
        ),
        label="#tbl-s1-screening",
    )

    # Write companion metrics
    write_metrics_csv(df, "S1_multivariate_screening")

    print("\n  Table S1 written successfully.\n")


if __name__ == "__main__":
    main()
