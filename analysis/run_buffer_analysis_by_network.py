#!/usr/bin/env python3
"""
NETWORK-SEPARATED BUFFER SCALE ANALYSIS
========================================
Runs buffer scale sensitivity analysis SEPARATELY for:
1. MRC stations only (N=4, 1985-2022)
2. Ben Tre stations only (N=19, 2005-2022)

Compares Kneedle optimal buffers and correlation patterns between networks.
This provides RIGOROUS evidence that buffer scale findings are consistent.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CACHE_DIR = "cache"
OUTPUT_DIR = "analysis_outputs"
WQ_CSV = 'mekong_delta_water_quality_1985_2023.csv'

# Buffer range (same as main analysis)
BUFFERS = list(range(1, 31))

# Station definitions
MRC_STATIONS = ['Tan Chau', 'My Thuan', 'Can Tho', 'My Tho']
BEN_TRE_PREFIXES = ['BL', 'CC', 'HL', 'MT']  # Ben Tre station prefixes

# Key parameters for comparison
KEY_PARAMS = ['COD', 'DO', 'TN', 'TSS', 'NH4']
KEY_LULC = ['Residential_Urban_composition_pct', 'Aquaculture_composition_pct']

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80)
print("NETWORK-SEPARATED BUFFER SCALE ANALYSIS")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================================
# KNEEDLE ALGORITHM
# ==============================================================================
def kneedle_elbow(x, y):
    """Find elbow point using Kneedle algorithm (Satopaa et al., 2011)."""
    x = np.array(x).astype(float)
    y = np.abs(np.array(y))
    
    if len(x) < 3:
        return x[y.argmax()]
    
    # Normalize to [0,1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Find point with maximum distance from diagonal
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    
    distances = []
    for i in range(len(x_n)):
        p = np.array([x_n[i], y_n[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return x[elbow_idx]

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\n" + "-" * 80)
print("LOADING DATA")
print("-" * 80)

# Load water quality data
wq_df = pd.read_csv(WQ_CSV, low_memory=False)
wq_df['datetime'] = pd.to_datetime(wq_df['datetime'], errors='coerce')
wq_df['year'] = wq_df['datetime'].dt.year.astype('Int64')

# Standardize parameter names
WQ_MAP = {
    'COD (mg/L)': 'COD', 'BOD5 (mg/L)': 'BOD5', 'DO (mg/L)': 'DO',
    'TN (mg/L)': 'TN', 'TP (mg/L)': 'TP',
    'NO3-N (mg/L)': 'NO3', 'NH4-N (mg/L)': 'NH4', 'PO4-P (mg/L)': 'PO4',
    'TSS (mg/L)': 'TSS'
}
wq_df.rename(columns=WQ_MAP, inplace=True)

# Available WQ parameters
available_wq = [c for c in WQ_MAP.values() if c in wq_df.columns]
print(f"Available WQ params: {available_wq}")

# Create network filter
def get_network(station_name):
    if station_name in MRC_STATIONS:
        return 'MRC'
    for prefix in BEN_TRE_PREFIXES:
        if str(station_name).startswith(prefix) and station_name not in MRC_STATIONS:
            return 'BenTre'
    return 'Unknown'

wq_df['Network'] = wq_df['stationName'].apply(get_network)

print(f"\nNetwork breakdown:")
print(wq_df.groupby('Network')['stationName'].nunique())

# ==============================================================================
# LOAD LULC DATA (for specific buffer)
# ==============================================================================
def load_lulc_buffer(buffer_km):
    """Load LULC data for a specific buffer distance."""
    cache_file = os.path.join(CACHE_DIR, f'lulc_metrics_corrected_{buffer_km}km.csv')
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    return None

# Test loading
test_lulc = load_lulc_buffer(16)
if test_lulc is not None:
    print(f"LULC data loaded: {len(test_lulc)} rows, {test_lulc.columns.tolist()[:5]}...")
else:
    print("ERROR: Cannot load LULC cache files!")

# ==============================================================================
# RUN BUFFER ANALYSIS FOR EACH NETWORK
# ==============================================================================
def run_buffer_analysis(wq_data, lulc_data, network_name, year_range=None):
    """Run buffer scale analysis for a specific network."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {network_name}")
    print(f"{'='*60}")
    
    # Filter by year if specified
    if year_range:
        wq_filtered = wq_data[(wq_data['year'] >= year_range[0]) & 
                              (wq_data['year'] <= year_range[1])].copy()
    else:
        wq_filtered = wq_data.copy()
    
    n_stations = wq_filtered['stationName'].nunique()
    print(f"Stations: {n_stations}")
    print(f"Years: {wq_filtered['year'].min()}-{wq_filtered['year'].max()}")
    print(f"Observations: {len(wq_filtered)}")
    
    if n_stations < 2:
        print("ERROR: Not enough stations!")
        return None
    
    # Calculate annual means
    wq_annual = wq_filtered.groupby(['stationName', 'year'])[available_wq].mean().reset_index()
    print(f"Station-years: {len(wq_annual)}")
    
    results = []
    
    for buffer in BUFFERS:
        lulc = load_lulc_buffer(buffer)
        if lulc is None:
            continue
        
        # Merge with WQ
        merged = pd.merge(wq_annual, lulc, on=['stationName', 'year'], how='inner')
        
        if len(merged) < 10:
            continue
        
        for wq_param in KEY_PARAMS:
            if wq_param not in merged.columns:
                continue
                
            for lulc_metric in KEY_LULC:
                if lulc_metric not in merged.columns:
                    continue
                
                # Get valid pairs
                valid = merged[[wq_param, lulc_metric]].dropna()
                
                if len(valid) < 10:
                    continue
                
                # Calculate Spearman correlation
                r, p = spearmanr(valid[wq_param], valid[lulc_metric])
                
                results.append({
                    'Network': network_name,
                    'Buffer_km': buffer,
                    'WQ_Parameter': wq_param,
                    'LULC_Metric': lulc_metric,
                    'Spearman_r': r,
                    'p_value': p,
                    'n_pairs': len(valid)
                })
    
    return pd.DataFrame(results)

# ==============================================================================
# RUN SEPARATE ANALYSES
# ==============================================================================
print("\n" + "=" * 80)
print("RUNNING SEPARATE BUFFER ANALYSES")
print("=" * 80)

# MRC analysis (1985-2022)
mrc_wq = wq_df[wq_df['Network'] == 'MRC']
print(f"\nMRC stations: {mrc_wq['stationName'].unique().tolist()}")

# Ben Tre analysis (2005-2022) - same period comparison
bt_wq = wq_df[wq_df['Network'] == 'BenTre']
print(f"Ben Tre stations: {bt_wq['stationName'].nunique()}")

# Run analyses
mrc_results = run_buffer_analysis(mrc_wq, None, 'MRC', year_range=(1985, 2022))
bt_results = run_buffer_analysis(bt_wq, None, 'BenTre', year_range=(2005, 2022))

# Also run MRC for 2005-2022 only (same period as Ben Tre)
mrc_results_2005 = run_buffer_analysis(mrc_wq, None, 'MRC_2005_2022', year_range=(2005, 2022))

# ==============================================================================
# COMPUTE KNEEDLE OPTIMAL BUFFERS
# ==============================================================================
print("\n" + "=" * 80)
print("COMPUTING KNEEDLE OPTIMAL BUFFERS")
print("=" * 80)

def compute_kneedle_for_network(results_df, network_name):
    """Compute Kneedle optimal buffer for each WQ-LULC combination."""
    if results_df is None or len(results_df) == 0:
        print(f"  {network_name}: No results to analyze")
        return pd.DataFrame(columns=['Network', 'WQ_Parameter', 'LULC_Metric', 
                                     'Peak_Buffer_km', 'Peak_r', 'Kneedle_Buffer_km'])
    
    kneedle_results = []
    
    for wq in results_df['WQ_Parameter'].unique():
        for lulc in results_df['LULC_Metric'].unique():
            subset = results_df[(results_df['WQ_Parameter'] == wq) & 
                               (results_df['LULC_Metric'] == lulc)].sort_values('Buffer_km')
            
            if len(subset) < 5:
                continue
            
            buffers = subset['Buffer_km'].values
            correlations = subset['Spearman_r'].values
            
            # Skip if all NaN
            if np.all(np.isnan(correlations)):
                continue
            
            # Find peak correlation
            peak_idx = np.nanargmax(np.abs(correlations))
            peak_buffer = buffers[peak_idx]
            peak_r = correlations[peak_idx]
            
            # Find Kneedle elbow
            try:
                elbow_buffer = kneedle_elbow(buffers, correlations)
            except:
                elbow_buffer = peak_buffer
            
            kneedle_results.append({
                'Network': network_name,
                'WQ_Parameter': wq,
                'LULC_Metric': lulc.replace('_composition_pct', ''),
                'Peak_Buffer_km': int(peak_buffer),
                'Peak_r': float(peak_r),
                'Kneedle_Buffer_km': int(elbow_buffer)
            })
    
    return pd.DataFrame(kneedle_results)

mrc_kneedle = compute_kneedle_for_network(mrc_results, 'MRC')
bt_kneedle = compute_kneedle_for_network(bt_results, 'BenTre')
mrc_2005_kneedle = compute_kneedle_for_network(mrc_results_2005, 'MRC_2005_2022')

# ==============================================================================
# COMPARE RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("CROSS-NETWORK BUFFER SCALE COMPARISON")
print("=" * 80)

if len(mrc_kneedle) > 0 and len(bt_kneedle) > 0:
    print("\nMRC Kneedle Optimal Buffers:")
    print(mrc_kneedle.to_string(index=False))
    
    print("\nBen Tre Kneedle Optimal Buffers:")
    print(bt_kneedle.to_string(index=False))
    
    # Summary comparison
    print("\n" + "-" * 60)
    print("SUMMARY COMPARISON:")
    print("-" * 60)
    
    # Compare Urban-COD
    mrc_urban_cod = mrc_kneedle[(mrc_kneedle['WQ_Parameter'] == 'COD') & 
                                (mrc_kneedle['LULC_Metric'] == 'Residential_Urban')]
    bt_urban_cod = bt_kneedle[(bt_kneedle['WQ_Parameter'] == 'COD') & 
                              (bt_kneedle['LULC_Metric'] == 'Residential_Urban')]
    
    if len(mrc_urban_cod) > 0 and len(bt_urban_cod) > 0:
        print(f"\nUrban-COD Kneedle Buffer:")
        print(f"  MRC:     {mrc_urban_cod['Kneedle_Buffer_km'].values[0]} km")
        print(f"  BenTre:  {bt_urban_cod['Kneedle_Buffer_km'].values[0]} km")
    
    # Overall median comparison
    mrc_median = mrc_kneedle['Kneedle_Buffer_km'].median()
    bt_median = bt_kneedle['Kneedle_Buffer_km'].median()
    
    print(f"\nMedian Kneedle Buffer:")
    print(f"  MRC:     {mrc_median:.0f} km")
    print(f"  BenTre:  {bt_median:.0f} km")
    print(f"  Selected: 16 km")
    
    # Consistency check
    diff = abs(mrc_median - bt_median)
    print(f"\n  Difference: {diff:.0f} km")
    if diff <= 5:
        print("  STATUS: CONSISTENT (within 5 km)")
    else:
        print("  STATUS: CHECK NEEDED (>5 km difference)")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Combine all Kneedle results
all_kneedle = pd.concat([mrc_kneedle, bt_kneedle, mrc_2005_kneedle], ignore_index=True)
kneedle_file = os.path.join(OUTPUT_DIR, f'Table_Kneedle_ByNetwork_{TIMESTAMP}.csv')
all_kneedle.to_csv(kneedle_file, index=False)
print(f"Saved: {kneedle_file}")

# Also save to manuscript tables
manuscript_file = os.path.join(OUTPUT_DIR, 'Table_Kneedle_ByNetwork.csv')
all_kneedle.to_csv(manuscript_file, index=False)
print(f"Saved: {manuscript_file}")

# Save correlation results
if mrc_results is not None:
    mrc_results.to_csv(os.path.join(OUTPUT_DIR, 'Buffer_Correlations_MRC.csv'), index=False)
if bt_results is not None:
    bt_results.to_csv(os.path.join(OUTPUT_DIR, 'Buffer_Correlations_BenTre.csv'), index=False)

# ==============================================================================
# CREATE COMPARISON TABLE FOR APPENDIX
# ==============================================================================
comparison_data = []

if len(mrc_kneedle) > 0 and len(bt_kneedle) > 0:
    for wq in KEY_PARAMS:
        for lulc_name in ['Residential_Urban', 'Aquaculture']:
            mrc_row = mrc_kneedle[(mrc_kneedle['WQ_Parameter'] == wq) & 
                                  (mrc_kneedle['LULC_Metric'] == lulc_name)]
            bt_row = bt_kneedle[(bt_kneedle['WQ_Parameter'] == wq) & 
                                (bt_kneedle['LULC_Metric'] == lulc_name)]
            
            mrc_val = mrc_row['Kneedle_Buffer_km'].values[0] if len(mrc_row) > 0 else np.nan
            bt_val = bt_row['Kneedle_Buffer_km'].values[0] if len(bt_row) > 0 else np.nan
            mrc_r = mrc_row['Peak_r'].values[0] if len(mrc_row) > 0 else np.nan
            bt_r = bt_row['Peak_r'].values[0] if len(bt_row) > 0 else np.nan
            
            # Handle NaN comparison
            if np.isnan(mrc_val) or np.isnan(bt_val):
                diff = np.nan
                consistent = 'N/A'
            else:
                diff = abs(mrc_val - bt_val)
                consistent = 'Yes' if diff <= 5 else 'Check'
            
            comparison_data.append({
                'WQ_Parameter': wq,
                'LULC_Driver': lulc_name,
                'MRC_Kneedle_km': mrc_val,
                'MRC_Peak_r': mrc_r,
                'BenTre_Kneedle_km': bt_val,
                'BenTre_Peak_r': bt_r,
                'Buffer_Difference_km': diff,
                'Consistent': consistent
            })

comparison_df = pd.DataFrame(comparison_data)
if len(comparison_df) > 0:
    comparison_file = os.path.join(OUTPUT_DIR, 'Table_Buffer_CrossNetwork_Comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved: {comparison_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
else:
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - No comparison table generated (insufficient data)")
    print("=" * 80)

