
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob

# Configuration
WQ_FILE = 'mekong_delta_water_quality_1985_2023.csv'
WL_DIR = 'hydrology_MRC'
OUTPUT_DIR = 'analysis/analysis_outputs'

# Station Mapping (WQ Name -> WL File Keyword)
STATION_MAP = {
    'Tan Chau': 'Tan_Chau',
    'My Thuan': 'My_Thuan',
    'My Tho': 'My_Tho',
    'Can Tho': 'Can_Tho',
    'My Hoa': 'My_Hoa', # Ham Luong Middle
    'Vam Kenh': 'Vam_Kenh', # My Tho Estuary
    'Tra Vinh': 'Tra_Vinh', # Co Chien Estuary (approx)
    # Check if 'Tra Vinh' is the WQ station name or if it's CC4/CC5? 
    # For now we use the keys to find the WL files and then match to WQ.
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_wl_data():
    print("Loading Water Level Data...")
    wl_dfs = {}
    
    files = glob.glob(os.path.join(WL_DIR, '*points.csv'))
    
    for station, keyword in STATION_MAP.items():
        # Find matching file
        matches = [f for f in files if keyword in f]
        if not matches:
            print(f"Warning: No WL file found for {station} (keyword: {keyword})")
            continue
            
        path = matches[0]
        print(f"Processing {station} from {os.path.basename(path)}...")
        
        # Read
        df = pd.read_csv(path)
        # Parse timestamp (handled UTC offset)
        df['dt'] = pd.to_datetime(df['timestamp'], utc=True)
        df['Date'] = df['dt'].dt.date
        
        # Calculate Daily Metrics
        daily = df.groupby('Date')['value'].agg(['mean', 'max', 'min']).reset_index()
        daily['Tidal_Range'] = daily['max'] - daily['min']
        daily.rename(columns={'mean': 'WL_Mean', 'max': 'WL_Max', 'min': 'WL_Min'}, inplace=True)
        
        wl_dfs[station] = daily
        
    return wl_dfs

def analyze_tidal_influence(wl_dfs):
    print("Loading Water Quality Data...")
    df_wq = pd.read_csv(WQ_FILE)
    df_wq['Date'] = pd.to_datetime(df_wq['datetime']).dt.date
    
    results = []
    
    # Analyze correlations for each station
    for station, df_wl in wl_dfs.items():
        # Match WQ station name
        # Note: WQ station names might differ slightly.
        # 'My Tho' in WL might match 'My Tho' or 'My Tho_prov' in WQ?
        # Let's try exact match first
        
        st_wq = df_wq[df_wq['stationName'] == station]
        
        # Special handling for My Hoa / Tra Vinh if they are not exact WQ names
        # Assuming user knows mappings or we check 'stationName' list
        # We will proceed with exact matches available
        
        if st_wq.empty:
            print(f"No WQ data for {station} (Exact match). Skipping WQ correlation.")
            continue
            
        # Merge
        merged = pd.merge(st_wq, df_wl, on='Date', how='inner')
        
        if len(merged) < 10:
            print(f"Insufficient overlap for {station} (n={len(merged)})")
            continue
            
        print(f"Analyzing {station} (n={len(merged)})...")
        
        # Check Correlations
        params = ['COD (mg/L)', 'TSS (mg/L)', 'NH4-N (mg/L)', 'Salinity', 'Salinity (g/L)']
        
        for p in params:
            if p not in merged.columns or merged[p].dropna().empty: continue
            
            # Corr with Mean WL
            r_wl, p_wl = stats.spearmanr(merged[p], merged['WL_Mean'], nan_policy='omit')
            
            # Corr with Tidal Range (Mixing Energy)
            r_tr, p_tr = stats.spearmanr(merged[p], merged['Tidal_Range'], nan_policy='omit')
            
            results.append({
                'Station': station,
                'Parameter': p,
                'R_WL_Mean': r_wl,
                'P_WL_Mean': p_wl,
                'R_Tidal_Range': r_tr,
                'P_Tidal_Range': p_tr
            })
            
    # Save Results
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, 'Table_Tidal_Influence_Correlations.csv')
        df_res.to_csv(out_path, index=False)
        print(f"Saved correlations to {out_path}")
        print(df_res.head(10))
    else:
        print("No significant WQ-WL overlaps found.")

def plot_tidal_trends(wl_dfs):
    # Plot Long-Term Evolution of Tidal Range
    print("Plotting Tidal Trends...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for station, df in wl_dfs.items():
        if station not in ['Vam Kenh', 'My Tho', 'Tan Chau']: continue # Key contrasts
        
        df['year'] = pd.to_datetime(df['Date']).dt.year
        annual = df.groupby('year')['Tidal_Range'].mean()
        
        ax.plot(annual.index, annual.values, label=station)
        
    ax.set_title('Evolution of Annual Mean Tidal Range (1985-2022)')
    ax.set_ylabel('Tidal Range (m)')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Tidal_Range_Trends.png'))
    print("Saved Figure_Tidal_Range_Trends.png")

def main():
    wl_dfs = load_wl_data()
    analyze_tidal_influence(wl_dfs)
    plot_tidal_trends(wl_dfs)

if __name__ == "__main__":
    main()
