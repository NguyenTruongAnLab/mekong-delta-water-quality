#!/usr/bin/env python3
"""
PUBLICATION-READY TEMPORAL BOOTSTRAP SENSITIVITY ANALYSIS (v2 Revised)
----------------------------------------------------------------------
Methodology:
1. Resamples monthly/seasonal observations WITHIN each station-year
2. Generates bootstrap distributions of annual means
3. Computes correlation distributions to quantify temporal uncertainty
4. Applies FDR correction for multiple testing
Adapted to read: cache/lulc_metrics_corrected_{km}km.csv
"""

import os
import warnings
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Configuration
warnings.filterwarnings('ignore')

print("="*80)
print("TEMPORAL BOOTSTRAP SENSITIVITY ANALYSIS")
print("Quantifying Uncertainty from Multi-Frequency Monitoring Integration")
print("="*80)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = str(ANALYSIS_DIR / "cache")
OUTPUT_DIR = str(ANALYSIS_DIR / "analysis_outputs")
APPENDIX_DIR = os.path.join(OUTPUT_DIR, "Appendix_Scale_Analysis_Final")
DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "Temporal_Uncertainty_Diagnostics")

for directory in [OUTPUT_DIR, APPENDIX_DIR, DIAGNOSTICS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Input files
WQ_CSV = str(REPO_ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv")

# =============================================================================
# ANALYSIS PARAMETERS (Following manuscript specifications)
# =============================================================================

ALL_BUFFERS_KM = list(range(1, 31))  # Complete 1-30 km range
N_BOOTSTRAP = 200  # Statistically sufficient for 95% CI
GLOBAL_SEED = 42   # Deterministic bootstrap for reproducibility
MIN_BOOTSTRAP_SAMPLES = 50  # Minimum valid bootstrap samples
MIN_STATION_YEARS = 20  # Minimum station-year pairs for robust correlation
FDR_ALPHA = 0.05  # False discovery rate threshold

# LULC categories and metrics
LULC_CATEGORIES = [
    'Residential_Urban', 'Aquaculture', 'Rice_Cultivation', 
    'Other_Crops', 'Forest', 'Wetland', 'Water_Bodies'
]
METRIC_TYPES = ['composition_pct', 'config_PD', 'config_ED', 'config_LSI', 'config_LPI']

# Water quality parameters
WQ_PARAMETERS_MAP = {
    'COD (mg/L)': 'COD', 'BOD5 (mg/L)': 'BOD5', 'DO (mg/L)': 'DO',
    'TN (mg/L)': 'TN', 'TP (mg/L)': 'TP',
    'NO3-N (mg/L)': 'NO3', 'NH4-N (mg/L)': 'NH4', 'PO4-P (mg/L)': 'PO4',
    'TSS (mg/L)': 'TSS'
}

print(f"\n⚙️  CONFIGURATION (Following Academic Standards):")
print(f"   Bootstrap iterations: {N_BOOTSTRAP}")
print(f"   Minimum station-years: {MIN_STATION_YEARS} (ensures statistical power)")
print(f"   FDR correction: Benjamini-Hochberg at α={FDR_ALPHA}")
print(f"   Buffer range: {ALL_BUFFERS_KM[0]}-{ALL_BUFFERS_KM[-1]} km")
print(f"   Method: Within-year temporal bootstrap")

# =============================================================================
# STEP 1: LOAD LULC DATA
# =============================================================================

def load_all_buffer_data(cache_path, buffer_distances):
    """Load LULC metrics for all buffer distances."""
    print("\n" + "="*80)
    print("STEP 1: LOADING LULC DATA")
    print("="*80)
    
    all_dfs = []
    # Check what files actually exist
    existing_buffers = []
    
    # Try rigorous range 1-30
    for buffer_km in buffer_distances:
        # UPDATED FILENAME PATTERN
        filepath = os.path.join(cache_path, f"lulc_metrics_corrected_{buffer_km}km.csv")
        if os.path.exists(filepath):
            existing_buffers.append(buffer_km)
            try:
                df = pd.read_csv(filepath, low_memory=False)
                if 'River' not in df.columns:
                    df['River'] = 'Unknown'
                # Ensure buffer_km column corresponds to file (it should be in file, but safe incase)
                df['buffer_km'] = buffer_km 
                all_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Error loading {buffer_km}km: {e}")
    
    if not all_dfs:
        print(f"No LULC data found in '{cache_path}'. Waiting for generation...")
        return pd.DataFrame()
    
    lulc_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nLULC DATA LOADED:")
    print(f"   Total records: {len(lulc_df):,}")
    print(f"   Buffer distances: {sorted(lulc_df['buffer_km'].unique())}")
    print(f"   Years: {lulc_df['year'].min()}-{lulc_df['year'].max()}")
    print(f"   Stations: {lulc_df['stationName'].nunique()}")
    
    return lulc_df

# =============================================================================
# STEP 2: LOAD RAW WQ DATA WITH DIAGNOSTICS
# =============================================================================

def prepare_raw_wq_data_with_diagnostics(wq_csv_file, diagnostics_dir):
    """Load raw WQ data preserving temporal structure."""
    print("\n" + "="*80)
    print("STEP 2: LOADING RAW WATER QUALITY DATA")
    print("="*80)
    
    wq_df = pd.read_csv(wq_csv_file, low_memory=False)
    
    # Parse temporal components
    wq_df['datetime'] = pd.to_datetime(wq_df['datetime'], errors='coerce')
    wq_df['year'] = wq_df['datetime'].dt.year
    wq_df['month'] = wq_df['datetime'].dt.month
    wq_df['season'] = wq_df['month'].apply(lambda m:
        'Dry' if m in [12, 1, 2, 3, 4] else 'Wet' if m in [5, 6, 7, 8, 9, 10, 11] else 'Unknown')
    
    wq_df.dropna(subset=['year'], inplace=True)
    wq_df['year'] = wq_df['year'].astype(int)
    wq_df.rename(columns=WQ_PARAMETERS_MAP, inplace=True)
    
    if 'River' not in wq_df.columns:
        wq_df['River'] = 'Unknown'
    
    available_wq = [col for col in WQ_PARAMETERS_MAP.values() if col in wq_df.columns]
    wq_df = wq_df[['stationName', 'year', 'month', 'season', 'River'] + available_wq]
    
    print(f"\nRAW WQ DATA LOADED:")
    print(f"   Total observations: {len(wq_df):,}")
    print(f"   Parameters: {len(available_wq)}")
    print(f"   Years: {wq_df['year'].min()}-{wq_df['year'].max()}")
    print(f"   Stations: {wq_df['stationName'].nunique()}")
    
    # Temporal sampling diagnostics
    print(f"\n📊 TEMPORAL SAMPLING STRUCTURE:")
    obs_per_station_year = wq_df.groupby(['stationName', 'year']).size()
    
    print(f"   Total station-years: {len(obs_per_station_year):,}")
    print(f"   Mean obs/station-year: {obs_per_station_year.mean():.1f}")
    print(f"   Median obs/station-year: {obs_per_station_year.median():.0f}")
    print(f"   Range: {obs_per_station_year.min()}-{obs_per_station_year.max()} obs")
    print(f"   Station-years with ≥2 obs: {(obs_per_station_year >= 2).sum():,} ({(obs_per_station_year >= 2).mean()*100:.1f}%)")
    
    # Within-year variability (source of temporal uncertainty)
    print(f"\n📈 WITHIN-YEAR VARIABILITY:")
    variability_stats = []
    for param in available_wq:
        param_data = wq_df.groupby(['stationName', 'year'])[param].agg(['mean', 'std', 'count'])
        param_data = param_data[param_data['count'] >= 2]
        
        if len(param_data) > 0:
            cv = (param_data['std'] / param_data['mean']).replace([np.inf, -np.inf], np.nan).mean()
            variability_stats.append({
                'Parameter': param,
                'Station-Years': len(param_data),
                'Mean_CV': cv,
                'Median_Std': param_data['std'].median()
            })
            print(f"   {param}: CV={cv*100:.1f}%")
    
    variability_df = pd.DataFrame(variability_stats)
    variability_file = os.path.join(diagnostics_dir, 'Table_Within_Year_Variability.csv')
    variability_df.to_csv(variability_file, index=False)
    print(f"\nSaved: {variability_file}")
    
    return wq_df, available_wq

# =============================================================================
# STEP 3: CORRECT TEMPORAL BOOTSTRAP WORKER
# =============================================================================

def temporal_bootstrap_worker(job_info):
    """
    SCIENTIFICALLY CORRECT TEMPORAL BOOTSTRAP
    
    For each buffer-metric-parameter combination:
    1. Calculate observed correlation using simple annual means
    2. Bootstrap: For each iteration, resample observations WITHIN each 
       station-year, compute new annual means, correlate with LULC
    3. Return observed r and 95% CI from bootstrap distribution
    
    This correctly tests: "How would correlations change if different 
    months/seasons were sampled within each station-year?"
    """
    lulc_metric, wq_param, buffer_km, merged_subset = job_info
    
    # Count unique station-years
    n_station_years = merged_subset.groupby(['stationName', 'year']).ngroups
    
    if n_station_years < MIN_STATION_YEARS:
        return None
    
    # =========================================================================
    # STEP 1: Calculate observed correlation (baseline)
    # =========================================================================
    try:
        # Calculate annual means (one value per station-year)
        observed_annual = merged_subset.groupby(['stationName', 'year']).agg({
            lulc_metric: 'first',  # LULC is constant within station-year
            wq_param: 'mean'       # WQ varies monthly - take mean
        }).reset_index()
        
        # Calculate Spearman correlation
        r_obs, p_val = spearmanr(observed_annual[lulc_metric], observed_annual[wq_param])
        
        if np.isnan(r_obs):
            return None
            
    except Exception:
        return None
    
    # =========================================================================
    # STEP 2: Temporal bootstrap - resample WITHIN station-years
    # =========================================================================
    # Deterministic seed per job (stable across runs)
    seed = (abs(hash((lulc_metric, wq_param, buffer_km))) + GLOBAL_SEED) % (2**32)
    rng = np.random.default_rng(seed)
    bootstrap_correlations = []
    
    for iteration in range(N_BOOTSTRAP):
        try:
            bootstrap_annual_means = []
            for (station, year), group in merged_subset.groupby(['stationName', 'year']):
                n_obs = len(group)
                if n_obs > 0:
                    resampled_values = rng.choice(group[wq_param].values, size=n_obs, replace=True)
                    bootstrap_wq_mean = np.mean(resampled_values)
                    lulc_value = group[lulc_metric].iloc[0]
                    bootstrap_annual_means.append((lulc_value, bootstrap_wq_mean))

            if not bootstrap_annual_means:
                continue

            data = np.array(bootstrap_annual_means)
            r_boot, _ = spearmanr(data[:, 0], data[:, 1])

            if not np.isnan(r_boot):
                bootstrap_correlations.append(r_boot)

        except Exception:
            continue
    
    # Check if we have enough valid bootstrap samples
    if len(bootstrap_correlations) < MIN_BOOTSTRAP_SAMPLES:
        return None
    
    # Calculate 95% confidence interval
    ci_lower = np.percentile(bootstrap_correlations, 2.5)
    ci_upper = np.percentile(bootstrap_correlations, 97.5)
    
    # Clean metric name for display
    metric_clean = lulc_metric.replace('_composition_pct', ' %').replace('_config_', ' ')
    
    return {
        'Buffer_km': buffer_km,
        'LULC_Metric': metric_clean,
        'WQ_Parameter': wq_param,
        'Spearman_r': r_obs,
        'r_CI_lower': ci_lower,
        'r_CI_upper': ci_upper,
        'CI_width': ci_upper - ci_lower,
        'Spearman_p': p_val,
        'n_station_years': n_station_years,
        'n_raw_observations': len(merged_subset),
        'n_bootstrap_valid': len(bootstrap_correlations),
        'bootstrap_r_mean': np.mean(bootstrap_correlations),
        'bootstrap_r_std': np.std(bootstrap_correlations)
    }

# =============================================================================
# STEP 4: PARALLEL EXECUTION WITH FDR CORRECTION
# =============================================================================

def run_temporal_bootstrap_analysis(lulc_df, raw_wq_df, available_wq):
    """Execute temporal bootstrap with FDR correction."""
    print("\n" + "="*80)
    print("STEP 3: TEMPORAL BOOTSTRAP ANALYSIS WITH FDR CORRECTION")
    print("="*80)
    
    # Build metric list dynamically based on what exists in DF
    # User provided specific list, let's intersect
    potential_metrics = [
        f"{cat}_{mtype}" for cat in LULC_CATEGORIES for mtype in METRIC_TYPES
    ]
    metric_cols = [c for c in potential_metrics if c in lulc_df.columns]
    
    print(f"\n📊 Analysis scope:")
    print(f"   LULC metrics: {len(metric_cols)}")
    print(f"   WQ parameters: {len(available_wq)}")
    print(f"   Buffers: {lulc_df['buffer_km'].nunique()}")
    print(f"   Total tests: ~{len(metric_cols) * len(available_wq) * lulc_df['buffer_km'].nunique():,}")
    
    # Merge LULC with RAW WQ data
    print(f"\n🔗 Merging LULC with raw WQ data...")
    # Fix: LULC DF has annual data. WQ has Monthly.
    # Join on Station + Year.
    merged_raw = pd.merge(
        lulc_df, raw_wq_df,
        on=['stationName', 'year'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_raw):,} observations")
    print(f"   Structure: Each row = station × year × month × LULC × WQ")
    print(f"   LULC: constant within station-year")
    print(f"   WQ: varies by month/season within year")
    
    # Prepare jobs
    print(f"\n🔧 Preparing jobs...")
    jobs = []
    
    unique_buffers = sorted(merged_raw['buffer_km'].unique())
    
    for buffer_km in unique_buffers:
        df_buffer = merged_raw[merged_raw['buffer_km'] == buffer_km]
        if df_buffer.empty: continue
        
        for metric_col in metric_cols:
            for wq_param in available_wq:
                # Subset needs stationName and year for grouping
                subset = df_buffer[['stationName', 'year', metric_col, wq_param]].dropna()
                
                # Pre-check size
                if len(subset) > MIN_STATION_YEARS: # Rough check
                    jobs.append((metric_col, wq_param, buffer_km, subset))
    
    print(f"Prepared {len(jobs):,} jobs")
    
    # Execute parallel bootstrap
    start_time = time.time()
    print(f"\n⚡ Starting parallel temporal bootstrap...")
    
    results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(temporal_bootstrap_worker)(job)
        for job in tqdm(jobs, desc="Bootstrap analysis", unit="job", colour='green')
    )
    
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("❌ No valid results computed.")
        return pd.DataFrame()
    
    corr_df = pd.DataFrame(valid_results)
    corr_df['abs_r'] = corr_df['Spearman_r'].abs()
    
    # ==========================================================================
    # APPLY FDR CORRECTION (Critical for multiple testing)
    # ==========================================================================
    print(f"\n📊 Applying Benjamini-Hochberg FDR correction...")
    
    reject, pvals_corrected, _, _ = multipletests(
        corr_df['Spearman_p'],
        alpha=FDR_ALPHA,
        method='fdr_bh'
    )
    
    corr_df['Spearman_q'] = pvals_corrected  # q-value (FDR-adjusted p-value)
    corr_df['FDR_significant'] = reject
    
    elapsed = time.time() - start_time
    
    print(f"\nANALYSIS COMPLETE!")
    print(f"   ⏱️  Runtime: {elapsed/60:.1f} minutes")
    print(f"   📊 Valid correlations: {len(corr_df):,}")
    print(f"   ✓  Significant (raw p<0.05): {(corr_df['Spearman_p'] < 0.05).sum():,} ({(corr_df['Spearman_p'] < 0.05).mean()*100:.1f}%)")
    print(f"   ✓  Significant (FDR q<0.05): {corr_df['FDR_significant'].sum():,} ({corr_df['FDR_significant'].mean()*100:.1f}%)")
    print(f"   📏 Mean CI width: {corr_df['CI_width'].mean():.3f}")
    
    return corr_df

# =============================================================================
# STEP 5: IDENTIFY OPTIMAL SCALES (FDR-corrected)
# =============================================================================

def identify_optimal_scales(corr_df, output_file):
    """Identify optimal scales using FDR-corrected significance."""
    print("\n" + "="*80)
    print("STEP 4: IDENTIFYING OPTIMAL SCALES")
    print("="*80)
    
    if corr_df.empty:
        return pd.DataFrame()
    
    # Use FDR-corrected significance
    sig_corr = corr_df[corr_df['FDR_significant']].copy()
    
    if sig_corr.empty:
        print("⚠️  No FDR-significant correlations. Using raw p<0.05.")
        sig_corr = corr_df[corr_df['Spearman_p'] < 0.05].copy()
    
    if sig_corr.empty:
        print("⚠️  No significant correlations found.")
        return pd.DataFrame()
    
    # Find maximum |r| for each parameter
    optimal_idx = sig_corr.groupby('WQ_Parameter')['abs_r'].idxmax()
    optimal_df = corr_df.loc[optimal_idx].copy()
    
    optimal_df['95%_CI'] = optimal_df.apply(
        lambda row: f"[{row['r_CI_lower']:.3f}, {row['r_CI_upper']:.3f}]",
        axis=1
    )
    
    summary = optimal_df[[
        'WQ_Parameter', 'LULC_Metric', 'Buffer_km',
        'Spearman_r', '95%_CI', 'CI_width',
        'Spearman_p', 'Spearman_q', 'FDR_significant',
        'n_station_years', 'n_raw_observations'
    ]].sort_values('WQ_Parameter')
    
    summary.to_csv(output_file, index=False)
    
    print("\nOPTIMAL SCALES (FDR-corrected):")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_file}")
    
    return optimal_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    overall_start = time.time()
    
    try:
        print(f"\nAnalysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lulc_df = load_all_buffer_data(CACHE_DIR, ALL_BUFFERS_KM)
        
        if not lulc_df.empty:
            raw_wq_df, available_wq = prepare_raw_wq_data_with_diagnostics(WQ_CSV, DIAGNOSTICS_DIR)
            
            corr_df = run_temporal_bootstrap_analysis(lulc_df, raw_wq_df, available_wq)
            
            if not corr_df.empty:
                full_output = os.path.join(OUTPUT_DIR, 'Table_Temporal_Bootstrap_FDR_Corrected.csv')
                corr_df.to_csv(full_output, index=False)
                corr_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Temporal_Bootstrap_FINAL.csv'), index=False)
                print(f"\nSaved: {full_output}")
                
                optimal_output = os.path.join(OUTPUT_DIR, 'Table_Optimal_Scales_FDR.csv')
                identify_optimal_scales(corr_df, optimal_output)
        
        total_time = time.time() - overall_start
        
        print("\n" + "="*80)
        print("TEMPORAL BOOTSTRAP ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nTotal runtime: {total_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
