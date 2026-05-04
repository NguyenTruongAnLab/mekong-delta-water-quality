import pandas as pd
df = pd.read_csv(r'h:\My Drive\Project\CARE projects 2024-20xx\4. Mekong long-term\Python\JAXA\scripts\analysis\cache\station_year_master_latest.csv')
hydro_cols = [c for c in df.columns if 'Q_branch' in c or 'Rain' in c]
print('Hydro cols:', hydro_cols)
mrc = ['MyThuan','CanTho','ChauDoc','TanChau']
bt = df[~df['stationName'].isin(mrc)]
for c in hydro_cols[:4]:
    v = bt.dropna(subset=[c])
    print(f'{c}: years {v["year"].min()}-{v["year"].max()}, N={len(v)}')
print()
# Check FE model input: all required predictors non-null
fe_preds = ['Aquaculture_composition_pct','Residential_Urban_composition_pct',
            'Rice_Cultivation_composition_pct','Wetland_composition_pct',
            'COD (mg/L)_annual'] + hydro_cols[:4]
v2 = bt.dropna(subset=[c2 for c2 in fe_preds if c2 in bt.columns])
print(f'All FE predictors non-null: N={len(v2)}, stations={v2["stationName"].nunique()}')
print(f'Years: {sorted(v2["year"].unique())}')
