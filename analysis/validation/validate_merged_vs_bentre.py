#!/usr/bin/env python3
"""validate_merged_vs_bentre.py

Exploratory check: how sensitive COD driver rankings are to merging monitoring
networks (Ben Tre + MRC), given known analytical offsets (e.g., COD methods).

Important:
    - The manuscript's primary spatial attribution is *within-network* (Ben Tre only)
        to avoid network bias.
    - This script is therefore *diagnostic*, not a recommended attribution workflow.

To minimize apples-to-oranges comparisons, the defaults are aligned with the
main attribution subset (16 km buffer; complete-case 2013-2020).
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
import shap

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
WQ_FILE = str(REPO_ROOT / 'data' / 'mekong_delta_water_quality_1985_2023.csv')
LULC_FILE = str(ANALYSIS_DIR / 'cache' / 'lulc_metrics_corrected_16km.csv')
MRC_STATIONS = ['Tan Chau', 'Can Tho', 'My Thuan', 'My Tho']

# Match the main multivariate attribution subset by default
YEAR_MIN = 2013
YEAR_MAX = 2020

print("="*80)
print("VALIDATION: MERGED DATASET CONSISTENCY CHECK")
print("="*80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\nLoading Data...")
MASTER_FILE = str(ANALYSIS_DIR / "cache" / "station_year_master_latest.csv")
if os.path.exists(WQ_FILE) and os.path.exists(LULC_FILE):
    df_wq = pd.read_csv(WQ_FILE)
    df_wq['year'] = pd.to_datetime(df_wq['datetime'], errors='coerce').dt.year
    wq_annual = df_wq.groupby(['stationName', 'year']).mean(numeric_only=True).reset_index()
    df_lulc = pd.read_csv(LULC_FILE)
    df_full = pd.merge(wq_annual, df_lulc, on=['stationName', 'year'], how='inner')
elif os.path.exists(MASTER_FILE):
    df_full = pd.read_csv(MASTER_FILE)
    df_full['COD (mg/L)'] = df_full['COD (mg/L)_annual']
else:
    raise FileNotFoundError(WQ_FILE)

# Filter to the aligned analysis window
df_full = df_full[(df_full['year'] >= YEAR_MIN) & (df_full['year'] <= YEAR_MAX)]

print(f"Full Merged Dataset ({YEAR_MIN}-{YEAR_MAX}): {len(df_full)} obs")

# Define Variables
wq_target = 'COD (mg/L)'
lulc_vars = [
    'Residential_Urban_composition_pct', 'Residential_Urban_config_PD',
    'Aquaculture_composition_pct', 'Aquaculture_config_ED',
    'Rice_Cultivation_composition_pct', 'Rice_Cultivation_config_PD'
]

# Rename for display
clean_names = {
    'Residential_Urban_composition_pct': 'Urban %',
    'Residential_Urban_config_PD': 'Urban PD',
    'Aquaculture_composition_pct': 'Aquaculture %',
    'Aquaculture_config_ED': 'Aquaculture ED',
    'Rice_Cultivation_composition_pct': 'Rice %',
    'Rice_Cultivation_config_PD': 'Rice PD'
}

# ==============================================================================
# 2. ANALYSIS FUNCTION
# ==============================================================================
def run_rf_analysis(df_subset, subset_name):
    print(f"\nAnalyzing: {subset_name} (N={len(df_subset)})")
    
    # Prep Data
    df_clean = df_subset.dropna(subset=[wq_target] + lulc_vars)
    if len(df_clean) < 10:
        print("  Not enough data.")
        return None
    
    X = df_clean[lulc_vars].rename(columns=clean_names)
    y = df_clean[wq_target]
    groups = df_clean['stationName']
    
    # Train RF
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    # Performance (grouped CV)
    logo = LeaveOneGroupOut()
    y_pred_cv = cross_val_predict(rf, X, y, groups=groups, cv=logo)
    r2_cv = r2_score(y, y_pred_cv)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    print(f"  CV R2 Score (LOGO): {r2_cv:.3f}")
    print(f"  CV RMSE (LOGO):     {rmse_cv:.3f}")

    # CV-aggregated SHAP (leakage-safe): compute SHAP on held-out stations
    abs_sum = np.zeros(X.shape[1], dtype=float)
    n_total = 0

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        rf_fold = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        rf_fold.fit(X_train, y_train)

        explainer_fold = shap.TreeExplainer(rf_fold)
        shap_vals_fold = explainer_fold.shap_values(X_test)
        abs_sum += np.abs(shap_vals_fold).sum(axis=0)
        n_total += shap_vals_fold.shape[0]

    mean_abs_shap = abs_sum / max(n_total, 1)
    imp_df = pd.DataFrame({
        'Driver': X.columns,
        'SHAP': mean_abs_shap
    }).sort_values('SHAP', ascending=False)
    
    imp_df['Rank'] = range(1, len(imp_df)+1)
    
    # Get Top Driver
    top_driver = imp_df.iloc[0]['Driver']
    top_shap = imp_df.iloc[0]['SHAP']
    
    # Get Urban Rank
    urban_rank = imp_df[imp_df['Driver'].str.contains('Urban')]['Rank'].min()
    
    return {
        'CV_R2': r2_cv,
        'CV_RMSE': rmse_cv,
        'Top_Driver': top_driver,
        'Top_SHAP': top_shap,
        'Urban_Rank': urban_rank,
        'Importance_Table': imp_df
    }

# ==============================================================================
# 3. RUN COMPARISON
# ==============================================================================

# Subset 1: Ben Tre Only
df_bentre = df_full[~df_full['stationName'].isin(MRC_STATIONS)]
res_bentre = run_rf_analysis(df_bentre, "Ben Tre Only")

# Subset 2: Merged
res_merged = df_full.copy()
res_merged_analysis = run_rf_analysis(res_merged, "Merged (Ben Tre + MRC)")

# ==============================================================================
# 4. REPORT
# ==============================================================================
print("\n" + "="*80)
print(f"CONSISTENCY CHECK: COD DRIVERS ({YEAR_MIN}-{YEAR_MAX})")
print("="*80)
print(f"{'Metric':<20} | {'Ben Tre Only':<25} | {'Merged (BT+MRC)':<25} | {'Consistent?'}")
print("-" * 90)

# Compare Top Driver
match_top = (res_bentre['Top_Driver'] == res_merged_analysis['Top_Driver'])
print(f"{'Top Driver':<20} | {res_bentre['Top_Driver']:<25} | {res_merged_analysis['Top_Driver']:<25} | {'✅' if match_top else '⚠️'}")

# Compare Urban Rank
match_rank = (res_bentre['Urban_Rank'] == res_merged_analysis['Urban_Rank'])
print(f"{'Urban Best Rank':<20} | {res_bentre['Urban_Rank']:<25} | {res_merged_analysis['Urban_Rank']:<25} | {'✅' if match_rank else '⚠️'}")

# Compare CV R2
print(f"{'Model CV R2':<20} | {res_bentre['CV_R2']:.3f}{'':<20} | {res_merged_analysis['CV_R2']:.3f}{'':<20} | {'-'}")

print("\nFull Rankings (Ben Tre):")
print(res_bentre['Importance_Table'].head(3).to_string(index=False))

print("\nFull Rankings (Merged):")
print(res_merged_analysis['Importance_Table'].head(3).to_string(index=False))

print("\nCONCLUSION:")
if match_top:
    print("Adding MRC stations DOES NOT change the primary driver identification.")
    print("Within this restricted window, the top COD correlate is stable to merging.")
else:
    print("Adding MRC stations changes the rankings, consistent with expected network bias/offset effects.")
    print("For defensible attribution, prefer within-network spatial modeling (Ben Tre only),")
    print("and treat cross-network comparisons as validation of *directional* patterns rather than merged attribution.")
