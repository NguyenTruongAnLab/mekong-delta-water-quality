
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import os
import glob
from datetime import timedelta

# Configuration
WL_DIR = 'hydrology_MRC'
OUTPUT_DIR = 'analysis_outputs'

# Transect for Lag Analysis: Upstream -> Downstream -> Estuary
# Tan Chau -> My Thuan -> My Tho -> Vam Kenh (Tien River Branch)
# Can Tho is on Hau River, can compare separately.
TRANSECT_STATIONS = {
    'Tan_Chau': 'Upstream',
    'My_Thuan': 'Midstream',
    'My_Tho': 'Downstream',
    'Vam_Kenh': 'Estuary'
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_hourly_wl_data(station_keywords):
    print("Loading Hourly Water Level Data...")
    wl_dfs = {}
    files = glob.glob(os.path.join(WL_DIR, '*points.csv'))
    
    for key, role in station_keywords.items():
        matches = [f for f in files if key in f]
        if not matches:
            print(f"Warning: No file for {key}")
            continue
        
        path = matches[0]
        print(f"Reading {key} from {os.path.basename(path)}...")
        df = pd.read_csv(path)
        
        # Parse timestamp
        # Using format='mixed' or inferring to handle potential variations, mainly ISO8601
        df['dt'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('dt', inplace=True)
        
        # Sort and deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Resample to regular hourly grid to ensure alignment for cross-corr
        # Interpolate small gaps (max 6 hours)
        df_hourly = df['value'].resample('h').mean().interpolate(method='linear', limit=6)
        
        wl_dfs[key] = df_hourly
        
    return wl_dfs

def calculate_lag_days(series1, series2):
    """
    Calculates lag of series2 relative to series1 using cross-correlation on Daily data.
    Returns lag in DAYS.
    """
    # Align intersection
    common_idx = series1.index.intersection(series2.index)
    if len(common_idx) < 14: 
        return np.nan
        
    s1 = series1.loc[common_idx]
    s2 = series2.loc[common_idx]
    
    s1 = s1 - s1.mean()
    s2 = s2 - s2.mean()
    
    correlation = signal.correlate(s2.values, s1.values, mode='full')
    lags = signal.correlation_lags(len(s1), len(s2), mode='full')
    
    lag_idx = np.argmax(correlation)
    lag_days = lags[lag_idx]
    
    return lag_days

def analyze_residence_time(wl_dfs):
    print("\n--- Analyzing Residence Time (Lag Analysis) ---\n")
    
    # Define Periods
    # Phase 1: 1985-2004
    # Phase 2: 2005-2022
    periods = {
        'Phase 1': ('1985-01-01', '2004-12-31'),
        'Phase 2': ('2005-01-01', '2022-12-31')
    }
    
    # Define Seasons
    # Dry: Jan, Feb, Mar, Apr
    # Wet: Aug, Sep, Oct, Nov (Peak flood)
    seasons = {
        'Dry': [1, 2, 3, 4],
        'Wet': [8, 9, 10, 11]
    }
    
    pairs = [
        ('Tan_Chau', 'My_Thuan'),
        ('My_Thuan', 'My_Tho'),
        ('My_Tho', 'Vam_Kenh'),
        ('Tan_Chau', 'Vam_Kenh') # Total System Lag
    ]
    
    results = []
    
    for label, (start_str, end_str) in periods.items():
        start = pd.Timestamp(start_str).tz_localize('UTC')
        end = pd.Timestamp(end_str).tz_localize('UTC')
        
        print(f"Processing {label}...")
        
        for p1, p2 in pairs:
            if p1 not in wl_dfs or p2 not in wl_dfs: continue
            
            # Slice for Period
            s1 = wl_dfs[p1][start:end]
            s2 = wl_dfs[p2][start:end]
            
            if s1.empty or s2.empty: continue
            
            # Sub-slice for Seasons
            for season_name, months in seasons.items():
                # Filter by month
                s1_seas = s1[s1.index.month.isin(months)]
                s2_seas = s2[s2.index.month.isin(months)]
                
                # strategy: Calculate lag on DAILY means to find Flood Wave travel time (Residence Time proxy)
                
                # Resample to Daily
                s1_daily = s1_seas.resample('D').mean().dropna()
                s2_daily = s2_seas.resample('D').mean().dropna()
                
                yearly_lags = []
                unique_years = s1_daily.index.year.unique()
                
                for yr in unique_years:
                    s1_yr = s1_daily[s1_daily.index.year == yr]
                    s2_yr = s2_daily[s2_daily.index.year == yr]
                    
                    if len(s1_yr) < 30: continue # Need at least a month of daily data
                    
                    # Calculate Lag in DAYS
                    lag_days = calculate_lag_days(s1_yr, s2_yr)
                    
                    if not np.isnan(lag_days) and 0 <= lag_days <= 10: # Reasonable lag 0-10 days
                        yearly_lags.append(lag_days * 24) # Convert to hours for reporting
                
                if yearly_lags:
                    avg_lag = np.mean(yearly_lags)
                    std_lag = np.std(yearly_lags)
                    results.append({
                        'Period': label,
                        'Season': season_name,
                        'Pair': f"{p1} -> {p2}",
                        'Lag_Hours': avg_lag,
                        'Lag_Std': std_lag,
                        'N_Years': len(yearly_lags)
                    })
    
    # Save Results
    df_res = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, 'Table_Residence_Time_Lags.csv')
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved Residence Time results to {out_path}")
    print(df_res)
    return df_res

def analyze_flushing_capacity(wl_dfs):
    print("\n--- Analyzing Flushing Capacity (Tidal Range) ---\n")
    
    stats = []
    
    for station, series in wl_dfs.items():
        # Calculate Daily Range
        daily_max = series.resample('D').max()
        daily_min = series.resample('D').min()
        daily_range = daily_max - daily_min
        
        # Overall Mean Range
        mean_range = daily_range.mean()
        
        # 95th Percentile (Max Flushing)
        max_range = daily_range.quantile(0.95)
        
        # 5th Percentile (Min Flushing)
        min_range = daily_range.quantile(0.05)
        
        stats.append({
            'Station': station,
            'Mean_Tidal_Range_m': mean_range,
            'Max_Tidal_Range_m': max_range,
            'Min_Tidal_Range_m': min_range
        })
        
    df_stats = pd.DataFrame(stats).sort_values('Mean_Tidal_Range_m', ascending=False)
    out_path = os.path.join(OUTPUT_DIR, 'Table_Flushing_Capacity.csv')
    df_stats.to_csv(out_path, index=False)
    print(f"Saved Flushing Capacity stats to {out_path}")
    print(df_stats)
    return df_stats

def main():
    wl_dfs = load_hourly_wl_data(TRANSECT_STATIONS)
    df_lags = analyze_residence_time(wl_dfs)
    df_flushing = analyze_flushing_capacity(wl_dfs)
    
    # Visualization: Lag Comparison
    if not df_lags.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_lags, x='Pair', y='Lag_Hours', hue='Season', ci=None)
        plt.title('Water Residence Time (Lag) by Season: Wet vs Dry')
        plt.ylabel('Lag Time (Hours)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Residence_Time_Season.png'))
        print("Saved Figure_Residence_Time_Season.png")

if __name__ == "__main__":
    main()
