"""Quick check: data availability when removing 2013-2020 filter."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The WQ file is referenced relative to SCRIPT_DIR in the analyzer,
# but actually lives in data/. Let's find it.
wq_candidates = [
    os.path.join(SCRIPT_DIR, "mekong_delta_water_quality_1985_2023.csv"),
    os.path.join(SCRIPT_DIR, "..", "..", "data", "mekong_delta_water_quality_1985_2023.csv"),
]
wq_path = next((p for p in wq_candidates if os.path.exists(p)), None)
print(f"WQ file: {wq_path}")

lulc_path = os.path.join(SCRIPT_DIR, "cache", "lulc_metrics_corrected_16km.csv")
print(f"LULC file exists: {os.path.exists(lulc_path)}")

if wq_path is None:
    print("WQ file not found, trying data/ path from root")
    wq_path = os.path.join(SCRIPT_DIR, "..", "data", "mekong_delta_water_quality_1985_2023.csv")
    print(f"Alt WQ: {os.path.exists(wq_path)}")

wq = pd.read_csv(wq_path)
wq["year"] = pd.to_datetime(wq["datetime"], errors="coerce").dt.year
wa = wq.groupby(["stationName", "River", "year"], as_index=False).mean(numeric_only=True)

lulc = pd.read_csv(lulc_path)
df = pd.merge(wa, lulc, on=["stationName", "year"], how="inner")
bt = df[df["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")].copy()
print(f"After merge + BT filter: N={len(bt)}, stations={bt['stationName'].nunique()}")
print(f"Years: {bt['year'].min()}-{bt['year'].max()}")

need = ["COD (mg/L)", "Residential_Urban_config_PD", "Aquaculture_config_ED",
        "Rice_Cultivation_config_PD", "Salinity"]
cc = bt.dropna(subset=need)
print(f"Complete case (core vars): N={len(cc)}, stations={cc['stationName'].nunique()}")
print(f"Year range: {cc['year'].min()}-{cc['year'].max()}")
print(f"Year distribution:\n{cc['year'].value_counts().sort_index()}")

# Also check hydro availability
from lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate
h = join_branch_hydroclimate(cc[["stationName", "year"]].copy())
hcols = HydroClimateColumns()
hydro_vars = [hcols.q_branch_annual_z, hcols.q_branch_dry_z,
              hcols.rain_annual_z, hcols.rain_dry_z]
for c in hydro_vars:
    if c in h.columns:
        cc[c] = h[c].values
cc2 = cc.dropna(subset=[c for c in hydro_vars if c in cc.columns])
print(f"\nWith hydro complete-case: N={len(cc2)}, stations={cc2['stationName'].nunique()}")
print(f"Year range: {cc2['year'].min()}-{cc2['year'].max()}")
print(f"Year distribution:\n{cc2['year'].value_counts().sort_index()}")
