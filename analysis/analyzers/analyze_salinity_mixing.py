
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration
WQ_FILE = 'mekong_delta_water_quality_1985_2023.csv'
OUTPUT_DIR = 'analysis_outputs'
BEN_TRE_STATIONS = [
    'An Hiep', 'An Thuan', 'Ben Trai', 'Binh Dai', 'Cho Lach', 'Cua Dai', 'Ham Luong',
    'Huong My', 'Khanh Thanh Tan', 'Loc Thuan', 'My Hoa', 'My Tho_prov', 'Phu Da', 
    'Phu Khanh', 'Son Doc', 'Tan Thanh', 'Tan Xuan', 'Thanh Phu', 'Vam Mon'
] 
# Note: 'My Tho_prov' might be named differently in CSV. We will rely on whatever non-MRC stations are there.

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def analyze_mixing_diagrams():
    print("Loading Data for Salinity Mixing Analysis...")
    df = pd.read_csv(WQ_FILE)
    
    # Check columns
    if 'Salinity' not in df.columns:
        print("Error: 'Salinity' column not found.")
        return

    # Filter for Ben Tre stations (Non-MRC)
    # Filter for years 2005-2022 (Phase 2)
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    df_bt = df[
        (df['year'] >= 2005) & 
        (df['year'] <= 2022) &
        (~df['stationName'].isin(['Tan Chau', 'Can Tho', 'My Thuan', 'My Tho', 'Chau Doc', 'Vam Nao'])) &
        (df['Salinity'].notna()) &
        (df['COD (mg/L)'].notna())
    ].copy()
    
    print(f"Filtered to {len(df_bt)} samples (Ben Tre, 2005-2022)")
    
    # Define End Members for Theoretical Dilution Line (TDL)
    # Freshwater (Upstream): Mean COD of fresh samples (< 0.5 g/L)
    fresh_end = df_bt[df_bt['Salinity'] < 0.5]['COD (mg/L)'].median()
    # Seawater (Coastal): Approx 30-35 g/L. COD assumed low (~2-4 mg/L) or min observed at high salinity
    sea_end_sal = 30
    sea_end_cod = df_bt[df_bt['Salinity'] > 20]['COD (mg/L)'].min()
    if np.isnan(sea_end_cod): sea_end_cod = 3.0 # Fallback
    
    print(f"End Members: Fresh(S=0, COD={fresh_end:.1f}), Sea(S={sea_end_sal}, COD={sea_end_cod:.1f})")
    
    # Plot Mixing Diagram
    plt.figure(figsize=(10, 8))
    
    # Scatter points colored by Station (or Station Group if too many)
    sns.scatterplot(
        data=df_bt, x='Salinity', y='COD (mg/L)', 
        hue='stationName', alpha=0.6, s=50, legend=False, palette='tab20'
    )
    
    # Theoretical Dilution Line
    plt.plot([0, sea_end_sal], [fresh_end, sea_end_cod], 'k--', linewidth=2, label='Theoretical Dilution Line')
    
    plt.title('Mixing Diagram: COD vs Salinity (Ben Tre, 2005-2022)', fontweight='bold')
    plt.xlabel('Salinity (g/L)')
    plt.ylabel('COD (mg/L)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate "Urban Sources"
    # Points significantly ABOVE the line
    plt.text(5, fresh_end + 5, 'Points above line\n= Internal Sources\n(Urban/Aquaculture)', 
             color='red', fontweight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, 'Figure_CEE_Mixing_Diagram.png')
    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved {out_path}")
    
    # 2. Dry Season Trends in Freshwater Zone Only
    # Filter: Dry Season (Jan-May), Salinity < 2 g/L
    df_bt['month'] = pd.to_datetime(df_bt['datetime']).dt.month
    df_fresh_dry = df_bt[
        (df_bt['month'].isin([1,2,3,4,5])) & 
        (df_bt['Salinity'] < 2)
    ]
    
    if len(df_fresh_dry) > 10:
        plt.figure(figsize=(8, 6))
        
        # Annual trend
        annual_fresh = df_fresh_dry.groupby('year')['COD (mg/L)'].mean().reset_index()
        
        sns.regplot(data=annual_fresh, x='year', y='COD (mg/L)', color='red')
        
        slope, intercept, r, p, se = stats.linregress(annual_fresh['year'], annual_fresh['COD (mg/L)'])
        
        plt.title(f'Dry Season COD Trend in Freshwater Zone (Sal < 2g/L)\nSlope = +{slope:.3f} mg/L/yr (p={p:.4f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_CEE_Freshwater_Dry_Trend.png'))
        print("✓ Saved Figure_CEE_Freshwater_Dry_Trend.png")
    else:
        print("Not enough freshwater dry season samples for trend analysis.")

if __name__ == "__main__":
    analyze_mixing_diagrams()
