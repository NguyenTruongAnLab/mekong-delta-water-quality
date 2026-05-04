#!/usr/bin/env python3
"""Quick verification of SHAP & LOSO R² values against manuscript claims."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# SHAP values
shap = pd.read_csv(ROOT / 'analysis/analysis_outputs/primary_model_shap_latest.csv')
cod_shap = shap[(shap['Parameter'] == 'COD') & (shap['Feature_Set'] == 'Primary (LULC+Pop+Hydro)')].sort_values('Rank').head(8)
print('COD SHAP (Primary):')
for _, r in cod_shap.iterrows():
    print('  Rank {} {}: |SHAP| = {:.2f}'.format(int(r['Rank']), r['Driver'], r['Mean_Abs_SHAP']))

do_shap = shap[(shap['Parameter'] == 'DO') & (shap['Feature_Set'] == 'Primary (LULC+Pop+Hydro)')].sort_values('Rank').head(5)
print('\nDO SHAP (Primary):')
for _, r in do_shap.iterrows():
    print('  Rank {} {}: |SHAP| = {:.2f}'.format(int(r['Rank']), r['Driver'], r['Mean_Abs_SHAP']))

print('\nMS claims:')
print('  COD: Tidal range=3.35, Rice PD=0.73, Discharge=0.50')
print('  DO: Aqua comp #1 (0.35), Edge density #2 (0.15)')

# LOSO
vp = pd.read_csv(ROOT / 'analysis/analysis_outputs/primary_variance_partitioning_latest.csv')
print('\nLOSO R2:')
for param in ['COD', 'DO', 'NH4-N', 'NO3-N', 'TSS']:
    for ps in ['LULC-only', 'Primary (LULC+Pop+Hydro)']:
        row = vp[(vp['Parameter'] == param) & (vp['Predictor_Set'] == ps)]
        if not row.empty:
            r2 = row.iloc[0]['LOSO_R2_Ridge']
            n = int(row.iloc[0]['N'])
            print('  {} [{}]: LOSO_R2={:.4f}, N={}'.format(param, ps, r2, n))

print('\nMS claims:')
print('  COD LULC-only LOSO R2=-0.11 (N=103)')
print('  DO LULC-only LOSO R2=-0.12 (N=120)')

# Check Panel FE Table
fe = pd.read_csv(ROOT / 'manuscript/tables/Table_NW_Panel_FE_Results.csv')
print('\nPanel FE Table (model_r2 and N):')
for p in fe['wq_parameter'].unique():
    sub = fe[fe['wq_parameter'] == p]
    r2 = sub['model_r2'].iloc[0]
    n = int(sub['n'].iloc[0])
    print('  {}: model_r2={:.4f}, N={}'.format(p, r2, n))

print('\nMS Table 1 claims:')
print('  COD: FE+LULC R2=0.654, N=229')
print('  DO:  FE+LULC R2=0.641, N=152')
print('  NH4: FE+LULC R2=0.369, N=297')
print('  NO3: FE+LULC R2=0.930, N=236')
print('  TSS: FE+LULC R2=0.474, N=294')
