
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

try:
    from analysis.lib.gold_standard_hydro import load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import load_gold_standard

# Paths (resolve relative to this script for robustness)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WQ_PATH = os.path.join(SCRIPT_DIR, "mekong_delta_water_quality_1985_2023.csv")
DISCHARGE_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "data",
    "hydrology",
    "outputs",
    "final_datasets",
    "Mekong_Delta_Gold_Standard_Master_1985_2025.csv",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_outputs")

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

def plot_provincial_loads(df):
    """
    Visualize Load Trends for key provincial stations to confirm 'Urban Masking' hypothesis.
    Selects representative stations from each branch.
    """
    print("\nGenerating Provincial Load Trend Figure...")
    
    # Representative stations showing the effect using the new Q_Specific logic
    # Ham Luong (Aquaculture), My Tho (Urban), Co Chien (Urban/Mixed)
    rep_stations = ['HL04', 'CC02', 'MT02'] 
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, station in enumerate(rep_stations):
        ax = axes[idx]
        st_data = df[df['stationName'] == station].copy()
        
        if len(st_data) < 5: 
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')
            continue
            
        st_data['year'] = pd.to_datetime(st_data['Date']).dt.year
        # Group by year
        annual = st_data.groupby('year')[['COD Load (tons/day)', 'Q_Specific']].mean()
        annual = annual.replace([np.inf, -np.inf], np.nan).dropna(subset=['COD Load (tons/day)'])
        if annual.shape[0] < 2:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')
            continue
        
        # Plot Load
        ax.plot(annual.index, annual['COD Load (tons/day)'], marker='o', color='purple', linewidth=2, label='COD Load')
        
        # Trendline
        try:
            x = annual.index.to_numpy(dtype=float)
            y = annual['COD Load (tons/day)'].to_numpy(dtype=float)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(annual.index, p(x), "k--", alpha=0.7, label=f"Trend: {z[0]:+.2f} t/d/yr")
        except Exception:
            # If polyfit fails (rare numerical issues), skip trendline but keep the series plot.
            pass
        
        ax.set_title(f"Station {station} (Provincial)\nOrganic Load Trend", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("COD Load (tons/day)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Provincial_Load_Trends.png'), dpi=300)
    print("✓ Saved Figure_Provincial_Load_Trends.png")

def plot_defense(df_res):
    # Create valid table for manuscript
    df_res['Conc_Trend_Sig'] = df_res['Conc_P'].apply(lambda x: '*' if x < 0.05 else 'ns')
    df_res['Load_Trend_Sig'] = df_res['Load_P'].apply(lambda x: '*' if x < 0.05 else 'ns')
    
    out_path = os.path.join(OUTPUT_DIR, "Table_Load_vs_Concentration_Trends.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Saved trend comparison to {out_path}")
    print(df_res[['Station', 'Parameter', 'Conc_Trend', 'Load_Trend', 'Corr_Conc_Flow', 'Dilution_Effect']].head())

def analyze_hysteresis_and_cq(df):
    """
    Generate CEE-Strategy 1 Figures:
    1. Hysteresis Loop: Annual Mean Q vs Annual Mean COD
    2. C-Q Relationships: Scatter plot by Phase
    """
    print("\nGenerating Hysteresis and C-Q Analysis...")
    stations = ['Tan Chau', 'Can Tho', 'My Thuan', 'My Tho']
    
    # Prepare Data
    df['year'] = pd.to_datetime(df['Date']).dt.year
    df['Phase'] = df['year'].apply(lambda y: 'Phase 1 (1985-2004)' if y < 2005 else 'Phase 2 (2005-2022)')
    
    # 1. HYSTERESIS (Annual Scale)
    # Use Q_Specific for correct station hydrology
    annual = df.groupby(['stationName', 'year']).agg({
        'Q_Specific': 'mean',
        'COD (mg/L)': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, station in enumerate(stations):
        ax = axes[idx]
        st_data = annual[annual['stationName'] == station].sort_values('year')
        if len(st_data) < 5: continue
        
        # Plot Loop
        sc = ax.scatter(st_data['Q_Specific'], st_data['COD (mg/L)'], c=st_data['year'], cmap='viridis', zorder=2)
        ax.plot(st_data['Q_Specific'], st_data['COD (mg/L)'], color='gray', alpha=0.5, linewidth=1, zorder=1)
        
        # Annotate Start/End/Regime Shift
        for y in [st_data['year'].min(), 2004, st_data['year'].max()]:
            row = st_data[st_data['year'] == y]
            if not row.empty:
                ax.text(row['Q_Specific'].values[0], row['COD (mg/L)'].values[0], str(y), 
                       fontsize=8, fontweight='bold')
        
        ax.set_title(station, fontweight='bold')
        ax.set_xlabel('Annual Mean Discharge (m3/s)')
        ax.set_ylabel('Annual Mean COD (mg/L)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.colorbar(sc, ax=axes, label='Year')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_CEE_Hysteresis_Loop.png'), dpi=300)
    print("✓ Saved Figure_CEE_Hysteresis_Loop.png")

    # 2. C-Q SCATTER (Monthly/Sample Scale)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    
    for idx, station in enumerate(stations):
        ax = axes2[idx]
        st_data = df[df['stationName'] == station].copy()
        
        # Filter outliers for plot clarity
        st_data = st_data[st_data['COD (mg/L)'] < 50] 
        st_data = st_data.dropna(subset=['Q_Specific', 'COD (mg/L)'])
        
        # Plot by Phase
        for phase, color in [('Phase 1 (1985-2004)', 'blue'), ('Phase 2 (2005-2022)', 'red')]:
            subset = st_data[st_data['Phase'] == phase]
            if len(subset) > 5:
                # Scatter
                ax.scatter(subset['Q_Specific'], subset['COD (mg/L)'], label=phase, alpha=0.4, s=15, color=color)
                
                # Trendline
                lr = stats.linregress(subset['Q_Specific'], subset['COD (mg/L)'])
                slope = float(lr[0])  # type: ignore[arg-type]
                intercept = float(lr[1])  # type: ignore[arg-type]
                x_vals = np.array([subset['Q_Specific'].min(), subset['Q_Specific'].max()])
                ax.plot(x_vals, slope*x_vals + intercept, color=color, linewidth=2, linestyle='--')
                
                # Annotate Slope
                ax.text(0.05, 0.9 if phase.startswith('Phase 1') else 0.8, 
                       f"{phase.split(' ')[0]}: Slope={slope*10000:.2f}e-4", 
                       transform=ax.transAxes, color=color, fontweight='bold')

        ax.set_title(f"{station}: C-Q Relationship", fontweight='bold')
        ax.set_xlabel('Daily Discharge (m3/s)')
        ax.set_ylabel('COD (mg/L)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_CEE_CQ_Relationships.png'), dpi=300)
    print("✓ Saved Figure_CEE_CQ_Relationships.png")

def main():
    df = load_data()
    df = calculate_loads(df)
    
    # Run CEE Analysis
    analyze_hysteresis_and_cq(df)
    
    df_trends = analyze_trends(df)
    plot_defense(df_trends)
    plot_provincial_loads(df)
    
    # Save processed dataframe for verification
    df.to_csv(os.path.join(OUTPUT_DIR, "VMD_WQ_Loads_Calculated.csv"), index=False)

if __name__ == "__main__":
    main()
