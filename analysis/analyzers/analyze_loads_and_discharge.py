
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis.lib.gold_standard_hydro import load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import load_gold_standard

WQ_PATH = str(REPO_ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv")
DISCHARGE_PATH = str(
    REPO_ROOT
    / "data"
    / "hydrology"
    / "outputs"
    / "final_datasets"
    / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
)
OUTPUT_DIR = str(ANALYSIS_DIR / "analysis_outputs")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    print("Loading Water Quality Data...")
    df_wq = pd.read_csv(WQ_PATH)
    df_wq['Date'] = pd.to_datetime(df_wq['datetime']).dt.date
    
    print("Loading Discharge Data...")
    df_q = load_gold_standard(DISCHARGE_PATH).reset_index().rename(columns={"Timestamp": "Date"})
    df_q['Date'] = pd.to_datetime(df_q['Date']).dt.date
    
    # Merge
    print("Merging datasets...")
    df_merged = pd.merge(df_wq, df_q, on='Date', how='inner')
    print(f"Matched {len(df_merged)} samples with daily discharge.")
    
    return df_merged

def calculate_loads(df):
    # Load = Concentration (mg/L) * Discharge (m3/s) * 0.0864 -> Tons/day
    
    # Station to Discharge Column Mapping
    # Tan Chau -> Upstream Tien (Direct Estimate)
    # Can Tho -> Hau River (Q_Hau_Est)
    # My Tho -> My Tho Branch (Q_MyTho_Est)
    # My Thuan -> Main Tien Downstream (Sum of branches ~ 48% of Total)
    
    # We need to map row-by-row or join.
    # Let's create a 'Q_Station' column first.
    
    def get_q_col(station):
        if station == 'Tan Chau': return 'Q_TanChau_Est'
        if station == 'Can Tho': return 'Q_Hau_Est'
        if station == 'My Tho': return 'Q_MyTho_Est'
        if station == 'My Thuan': return 'Q_Tien_Sum' # We need to create this sum
        return 'Q_Total_Est' # Default fallback
    
    # Create Q_Tien_Sum (sum of CoChien + HamLuong + MyTho) when available.
    if {'CoChien', 'HamLuong', 'MyTho'}.issubset(df.columns):
        df['Q_Tien_Sum'] = df['CoChien'] + df['HamLuong'] + df['MyTho']
    else:
        df['Q_Tien_Sum'] = np.nan
        
    params = ['COD (mg/L)', 'NO3-N (mg/L)', 'NH4-N (mg/L)', 'TSS (mg/L)']
    
    # Vectorized approach: Create Q_Specific column
    station_str = df['stationName'].astype(str)
    conditions = [
        df['stationName'] == 'Tan Chau',
        df['stationName'] == 'Can Tho',
        df['stationName'] == 'My Thuan',
        # My Tho (MRC) and MTxx (Provincial) -> My Tho Branch
        station_str.str.startswith('My Tho') | station_str.str.startswith('MT'),
        # Ham Luong (Provincial HLxx)
        station_str.str.startswith('HL'),
        # Co Chien (Provincial CCxx)
        station_str.str.startswith('CC')
    ]
    
    # Check if provincial columns exist, otherwise fallback or handle errors
    # Assuming columns exist since we checked file content previously
    q_hamluong = df['HamLuong'] if 'HamLuong' in df.columns else np.nan
    q_cochien = df['CoChien'] if 'CoChien' in df.columns else np.nan

    q_tanchau = df['TanChau'] if 'TanChau' in df.columns else np.nan
    q_cantho = df['CanTho'] if 'CanTho' in df.columns else np.nan
    q_mythuan = df['MyThuan'] if 'MyThuan' in df.columns else np.nan
    q_mytho = df['MyTho'] if 'MyTho' in df.columns else np.nan

    choices = [
        q_tanchau,
        q_cantho,
        q_mythuan,
        q_mytho,
        q_hamluong,
        q_cochien
    ]
    
    # Use np.select to assign specific discharge
    df['Q_Specific'] = np.select(conditions, choices, default=np.nan)
    
    for p in params:
        if p in df.columns:
            load_col = p.replace('(mg/L)', 'Load (tons/day)')
            # Calculate load only where Q_Specific is valid
            df[load_col] = df[p] * df['Q_Specific'] * 0.0864
            
    return df


def analyze_trends(df):
    results = []
    
    # Analyze ALL stations that have valid Load data (excluding Ba Lai likely)
    stations = df['stationName'].unique()
    
    params = [
        ('COD (mg/L)', 'COD Load (tons/day)'),
        ('NO3-N (mg/L)', 'NO3-N Load (tons/day)'),
        ('NH4-N (mg/L)', 'NH4-N Load (tons/day)')
    ]
    
    for station in stations:
        st_df = df[df['stationName'] == station].copy()
        
        # Skip if station has no valid discharge data (e.g., Ba Lai)
        if st_df['Q_Specific'].isna().all():
            continue
            
        for conc_col, load_col in params:
            if conc_col not in df.columns or load_col not in df.columns: continue
            
            # Annual Means
            st_df['year'] = pd.to_datetime(st_df['Date']).dt.year
            annual = st_df.groupby('year')[[conc_col, load_col]].mean()
            
            # Trends (2005-2022) - The Acceleration Phase
            recent = annual[annual.index >= 2005]
            if len(recent) < 5: continue
            
            # Concentration Trend
            # Handle NaNs in annual means
            recent_c = recent.dropna(subset=[conc_col])
            if len(recent_c) >= 5:
                slope_c, _, _, p_c, _ = stats.linregress(recent_c.index, recent_c[conc_col])
            else:
                slope_c, p_c = np.nan, np.nan

            # Load Trend
            recent_l = recent.dropna(subset=[load_col])
            if len(recent_l) >= 5:
                slope_l, _, _, p_l, _ = stats.linregress(recent_l.index, recent_l[load_col])
            else:
                slope_l, p_l = np.nan, np.nan
            
            # Correlation with Flow (Dilution Check)
            # Correlation between Concentration and Q at daily scale
            # Use Q_Specific for correct local hydrology
            clean_idx = st_df[[conc_col, 'Q_Specific']].dropna().index
            if len(clean_idx) > 10:
                res = stats.spearmanr(st_df.loc[clean_idx, conc_col], st_df.loc[clean_idx, 'Q_Specific'])
                corr_cq = float(getattr(res, 'correlation', np.nan))
                p_cq = float(getattr(res, 'pvalue', np.nan))
            else:
                corr_cq, p_cq = np.nan, np.nan
            
            results.append({
                'Station': station,
                'Parameter': conc_col,
                'Conc_Trend': slope_c,
                'Conc_P': p_c,
                'Load_Trend': slope_l,
                'Load_P': p_l,
                'Corr_Conc_Flow': corr_cq,
                'Dilution_Effect': 'Yes' if (not np.isnan(corr_cq)) and corr_cq < -0.3 and p_cq < 0.05 else 'No'
            })
            
    return pd.DataFrame(results)

def save_trend_summary(df_res):
    df_res['Conc_Trend_Sig'] = df_res['Conc_P'].apply(lambda x: '*' if x < 0.05 else 'ns')
    df_res['Load_Trend_Sig'] = df_res['Load_P'].apply(lambda x: '*' if x < 0.05 else 'ns')
    
    out_path = os.path.join(OUTPUT_DIR, "Table_Load_vs_Concentration_Trends.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Saved trend comparison to {out_path}")
    print(df_res[['Station', 'Parameter', 'Conc_Trend', 'Load_Trend', 'Corr_Conc_Flow', 'Dilution_Effect']].head())

def main():
    df = load_data()
    df = calculate_loads(df)
    df_trends = analyze_trends(df)
    save_trend_summary(df_trends)
    df.to_csv(os.path.join(OUTPUT_DIR, "VMD_WQ_Loads_Calculated.csv"), index=False)

if __name__ == "__main__":
    main()
