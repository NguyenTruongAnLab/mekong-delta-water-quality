
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis.lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard

ERA5_FILE = REPO_ROOT / 'data' / 'hydrology' / 'data' / 'inputs' / 'drivers' / 'Mekong_Rainfall_ERA5_1985_2022.csv'
CHIRPS_FILE = REPO_ROOT / 'data' / 'hydrology' / 'data' / 'inputs' / 'drivers' / 'chirps_basin_rainfall_daily.csv'
BEN_TRE_GAUGE_MONTHLY_FILE = (
    REPO_ROOT / 'data' / 'hydrology' / 'data' / 'inputs' / 'drivers' / 'Monthly_rain_gauge_BenTre.csv'
)

GOLD_STANDARD_FILE = (
    REPO_ROOT
    / 'data'
    / 'hydrology'
    / 'outputs'
    / 'final_datasets'
    / 'Mekong_Delta_Gold_Standard_Master_1985_2025.csv'
)

OUTPUT_DIR = REPO_ROOT / 'analysis' / 'analysis_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seasons (monsoon context)
DRY_MONTHS = [1, 2, 3, 4]
WET_MONTHS = [6, 7, 8, 9]

# Study window for manuscript consistency
STUDY_START = "1985-01-01"
STUDY_END = "2022-12-31"


def _seasonal_totals(series: pd.Series, months: list[int]) -> pd.Series:
    """Aggregate a time-indexed precipitation series to annual seasonal totals (mm).

    Parameters
    ----------
    series:
        Time-indexed precipitation (mm per timestep).
    months:
        Months to include in the seasonal sum.
    """
    s = series.copy()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep='first')]
    # Enforce a DatetimeIndex for month filtering
    idx = pd.DatetimeIndex(s.index)
    s = s[idx.month.isin(months)]
    return s.resample('YE').sum()


def _trend_linreg(years: np.ndarray, values) -> dict:
    """Simple linear trend summary."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if mask.sum() < 5:
        return {'slope': np.nan, 'pvalue': np.nan}
    res = stats.linregress(years[mask], v[mask])
    # linregress returns a tuple-like result across scipy versions
    return {
        'slope': float(res[0]),  # type: ignore[arg-type]
        'pvalue': float(res[3]),  # type: ignore[arg-type]
    }


def _trend_sen_kendall(years: np.ndarray, values) -> dict:
    """Robust trend summary using Sen slope + Kendall tau.

    Notes
    -----
    This is not a full Mann–Kendall test implementation (ties/seasonality), but
    Kendall tau vs time provides a monotonic-trend significance check that is
    more robust than OLS slope assumptions.
    """
    from scipy.stats import kendalltau, theilslopes

    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if mask.sum() < 5:
        return {'sen_slope': np.nan, 'sen_ci_lo': np.nan, 'sen_ci_hi': np.nan, 'tau': np.nan, 'pvalue': np.nan}

    # theilslopes returns: slope, intercept, lower_slope, upper_slope
    slope, _intercept, lo, hi = theilslopes(v[mask], years[mask], 0.95)
    tau, pv = kendalltau(years[mask], v[mask])
    return {
        # scipy typeshed stubs can be overly strict here; these are scalars.
        'sen_slope': float(slope),  # type: ignore[arg-type]
        'sen_ci_lo': float(lo),  # type: ignore[arg-type]
        'sen_ci_hi': float(hi),  # type: ignore[arg-type]
        'tau': float(tau),  # type: ignore[arg-type]
        'pvalue': float(pv),  # type: ignore[arg-type]
    }

def analyze_cross_product():
    print("Loading Datasets...")

    # CHIRPS basin daily (authoritative driver series)
    # Prefer the curated Gold Standard master (which already contains the basin-mean CHIRPS driver)
    chirps_daily = None
    try:
        gs = load_gold_standard(GOLD_STANDARD_FILE)
        rain_col = GoldStandardColumns().rainfall_mm
        chirps_daily = gs[rain_col].astype(float).resample('D').sum()
    except Exception:
        chirps_daily = None

    if chirps_daily is None:
        if not os.path.exists(CHIRPS_FILE):
            raise FileNotFoundError(
                f"Missing CHIRPS driver series. Expected either Gold Standard at {GOLD_STANDARD_FILE} "
                f"or raw CHIRPS file at {CHIRPS_FILE}"
            )
        chirps_df = pd.read_csv(CHIRPS_FILE)
        chirps_df['Date'] = pd.to_datetime(chirps_df['Date'], errors='coerce')
        chirps_df = chirps_df.dropna(subset=['Date'])
        chirps_df = chirps_df.set_index('Date').sort_index()
        chirps_daily = chirps_df['Precipitation_mm'].astype(float).resample('D').sum()

    # Restrict to manuscript study window
    chirps_daily = chirps_daily.loc[STUDY_START:STUDY_END]

    chirps_annual = chirps_daily.resample('YE').sum()

    if BEN_TRE_GAUGE_MONTHLY_FILE.exists():
        gauge_df = pd.read_csv(BEN_TRE_GAUGE_MONTHLY_FILE, engine='python')
        gauge_df['Date'] = pd.to_datetime(gauge_df['Date'], format='%m/%Y', errors='coerce')
        gauge_df = gauge_df.dropna(subset=['Date']).set_index('Date').sort_index()
        gauge_df = gauge_df.loc[STUDY_START:STUDY_END]

        g_idx = pd.DatetimeIndex(gauge_df.index)
        dim = g_idx.days_in_month
        bt_mm = pd.to_numeric(gauge_df['Ben Tre Station'], errors='coerce') * dim
        btri_mm = pd.to_numeric(gauge_df['Ba Tri Station'], errors='coerce') * dim

        chirps_monthly = chirps_daily.resample('MS').sum()
        m = pd.DataFrame({
            'chirps_basin_mm_month': chirps_monthly,
            'ben_tre_gauge_mm_month': bt_mm.resample('MS').sum(min_count=1),
            'ba_tri_gauge_mm_month': btri_mm.resample('MS').sum(min_count=1),
        }).dropna()

        from scipy.stats import pearsonr, spearmanr

        def _corr_row(label: str, a: pd.Series, b: pd.Series) -> dict:
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
            r, p = pearsonr(aa, bb)
            rho, p_s = spearmanr(aa, bb)
            return {
                'Metric': label,
                'Pearson_r': float(r),  # type: ignore[arg-type]
                'Pearson_p': float(p),  # type: ignore[arg-type]
                'Spearman_rho': float(rho),  # type: ignore[arg-type]
                'Spearman_p': float(p_s),  # type: ignore[arg-type]
                'Mean_CHIRPS_mm_month': float(np.nanmean(aa)),
                'Mean_Gauge_mm_month': float(np.nanmean(bb)),
                'N_months': int(np.isfinite(aa).sum()),
            }

        rows = []
        rows.append(_corr_row('CHIRPS vs Ben Tre (raw monthly)', m['chirps_basin_mm_month'], m['ben_tre_gauge_mm_month']))
        rows.append(_corr_row('CHIRPS vs Ba Tri (raw monthly)', m['chirps_basin_mm_month'], m['ba_tri_gauge_mm_month']))

        m_month = pd.DatetimeIndex(m.index).month
        wet = m[m_month.isin(WET_MONTHS)]
        if len(wet) >= 10:
            rows.append(_corr_row('CHIRPS vs Ben Tre (wet months only)', wet['chirps_basin_mm_month'], wet['ben_tre_gauge_mm_month']))
            rows.append(_corr_row('CHIRPS vs Ba Tri (wet months only)', wet['chirps_basin_mm_month'], wet['ba_tri_gauge_mm_month']))

        mo = pd.DatetimeIndex(m.index).month
        anom = m.copy()
        for col in ['chirps_basin_mm_month', 'ben_tre_gauge_mm_month', 'ba_tri_gauge_mm_month']:
            anom[col] = m[col] - m.groupby(mo)[col].transform('mean')
        rows.append(_corr_row('CHIRPS vs Ben Tre (monthly anomalies)', anom['chirps_basin_mm_month'], anom['ben_tre_gauge_mm_month']))
        rows.append(_corr_row('CHIRPS vs Ba Tri (monthly anomalies)', anom['chirps_basin_mm_month'], anom['ba_tri_gauge_mm_month']))

        df_corr = pd.DataFrame(rows)
        out_corr = OUTPUT_DIR / 'chirps_vs_gauge_monthly_stats_latest.csv'
        df_corr.to_csv(out_corr, index=False)

        print("\nCHIRPS vs gauge (monthly totals, basin vs point):")
        for _i, r in df_corr.iterrows():
            print(
                f"{r['Metric']}: Pearson r={r['Pearson_r']:.3f} (p={r['Pearson_p']:.2g}); "
                f"Spearman rho={r['Spearman_rho']:.3f} (p={r['Spearman_p']:.2g}); "
                f"mean CHIRPS={r['Mean_CHIRPS_mm_month']:.1f} mm/mo, mean gauge={r['Mean_Gauge_mm_month']:.1f} mm/mo"
            )
        print(f"Saved CHIRPS-vs-gauge stats to {out_corr}")

    # ERA5 (optional, for cross-product comparison)
    era5_annual = None
    era5_df = None
    if ERA5_FILE.exists():
        era5_df = pd.read_csv(ERA5_FILE)
        era5_df['timestamp'] = pd.to_datetime(era5_df['timestamp'], errors='coerce')
        era5_df = era5_df.dropna(subset=['timestamp'])
        era5_df = era5_df.set_index('timestamp').sort_index()
        era5_daily = era5_df['precipitation_mm'].astype(float).resample('D').sum()
        era5_daily = era5_daily.loc[STUDY_START:STUDY_END]
        era5_annual = era5_daily.resample('YE').sum()

    if era5_annual is not None:
        idx = era5_annual.index.intersection(chirps_annual.index)
        era5_c = era5_annual.loc[idx]
        chirps_c = chirps_annual.loc[idx]
        years = pd.DatetimeIndex(idx).year
    else:
        idx = chirps_annual.index
        chirps_c = chirps_annual
        years = pd.DatetimeIndex(idx).year
    
    # 1. Stats Comparison
    print(f"\nCross-Product Stats (1985-2022):")
    c_trend = stats.linregress(years, chirps_c.values)
    c_rob = _trend_sen_kendall(np.asarray(years), chirps_c.values)
    print(
        f"CHIRPS Trend: {float(c_trend[0]):.1f} mm/yr "  # type: ignore[arg-type]
        f"(p={float(c_trend[3]):.3f})"  # type: ignore[arg-type]
    )
    print(
        "CHIRPS Robust: "
        f"Sen={c_rob['sen_slope']:.2f} mm/yr "
        f"(95% CI {c_rob['sen_ci_lo']:.2f} to {c_rob['sen_ci_hi']:.2f}); "
        f"Kendall tau={c_rob['tau']:.3f} (p={c_rob['pvalue']:.3f})"
    )

    corr, p_val, bias, e_trend = np.nan, np.nan, np.nan, None
    if era5_annual is not None:
        corr, p_val = stats.pearsonr(era5_c, chirps_c)
        bias = chirps_c.mean() - era5_c.mean()
        e_trend = stats.linregress(years, era5_c.values)
        e_rob = _trend_sen_kendall(np.asarray(years), era5_c.values)
        print(f"Correlation (r): {corr:.2f} (p={p_val:.4f})")
        print(f"Mean Bias (CHIRPS - ERA5): {bias:.1f} mm/yr")
        print(
            f"ERA5 Trend: {float(e_trend[0]):.1f} mm/yr "  # type: ignore[arg-type]
            f"(p={float(e_trend[3]):.3f})"  # type: ignore[arg-type]
        )
        print(
            "ERA5 Robust: "
            f"Sen={e_rob['sen_slope']:.2f} mm/yr "
            f"(95% CI {e_rob['sen_ci_lo']:.2f} to {e_rob['sen_ci_hi']:.2f}); "
            f"Kendall tau={e_rob['tau']:.3f} (p={e_rob['pvalue']:.3f})"
        )

    # Seasonal decomposition: dry vs wet season totals
    chirps_dry = _seasonal_totals(chirps_daily, DRY_MONTHS)
    chirps_wet = _seasonal_totals(chirps_daily, WET_MONTHS)

    era5_dry = era5_wet = None
    if era5_annual is not None and era5_df is not None:
        era5_daily = era5_df['precipitation_mm'].astype(float).resample('D').sum()
        era5_daily = era5_daily.loc[STUDY_START:STUDY_END]
        era5_dry = _seasonal_totals(era5_daily, DRY_MONTHS)
        era5_wet = _seasonal_totals(era5_daily, WET_MONTHS)

    idx_s = chirps_dry.index if era5_dry is None else era5_dry.index.intersection(chirps_dry.index)
    years_s = pd.DatetimeIndex(idx_s).year.values

    seasonal_rows = []
    rows_in = [
        ('CHIRPS', 'Dry (Jan-Apr)', chirps_dry.loc[idx_s]),
        ('CHIRPS', 'Wet (Jun-Sep)', chirps_wet.loc[idx_s]),
    ]
    if era5_dry is not None and era5_wet is not None:
        rows_in = [
            ('ERA5', 'Dry (Jan-Apr)', era5_dry.loc[idx_s]),
            ('ERA5', 'Wet (Jun-Sep)', era5_wet.loc[idx_s]),
        ] + rows_in

    for dataset, season, s in rows_in:
        tr = _trend_linreg(years_s, s.values)
        seasonal_rows.append({
            'Dataset': dataset,
            'Season': season,
            'Period_Start': int(years_s.min()),
            'Period_End': int(years_s.max()),
            'Mean_mm': float(np.nanmean(np.asarray(s.values, dtype=float))),
            'Trend_mm_per_year': tr['slope'],
            'Trend_pvalue': tr['pvalue'],
        })

    df_seasonal = pd.DataFrame(seasonal_rows)
    df_seasonal = df_seasonal.sort_values(['Season', 'Dataset'])
    seasonal_out = OUTPUT_DIR / 'rainfall_seasonal_trends_latest.csv'
    df_seasonal.to_csv(seasonal_out, index=False)
    print(f"Saved seasonal rainfall trends to {seasonal_out}")
    
    # 2. Reality Check: 2015 Drought
    lt_mean_c = chirps_c.mean()
    val_2015_c = chirps_c.loc[pd.Timestamp('2015-12-31')]
    anom_c = (val_2015_c - lt_mean_c) / lt_mean_c * 100
    
    print(f"\nReality Check (2015 Drought):")
    print(f"CHIRPS 2015 Value: {val_2015_c:.1f} mm")
    print(f"CHIRPS LTA: {lt_mean_c:.1f} mm")
    print(f"CHIRPS Anomaly: {anom_c:.1f}%")
    
    # 3. Save Stats
    with open(OUTPUT_DIR / 'cross_product_stats.txt', "w") as f:
        if e_trend is not None:
            f.write(f"ERA5_Trend,{float(e_trend[0]):.1f}\n")  # type: ignore[arg-type]
            f.write(f"ERA5_SenSlope,{e_rob['sen_slope']:.2f}\n")
            f.write(f"ERA5_KendallTau,{e_rob['tau']:.3f}\n")
            f.write(f"ERA5_KendallP,{e_rob['pvalue']:.3f}\n")
        f.write(f"CHIRPS_Trend,{float(c_trend[0]):.1f}\n")  # type: ignore[arg-type]
        f.write(f"CHIRPS_SenSlope,{c_rob['sen_slope']:.2f}\n")
        f.write(f"CHIRPS_KendallTau,{c_rob['tau']:.3f}\n")
        f.write(f"CHIRPS_KendallP,{c_rob['pvalue']:.3f}\n")
        if era5_annual is not None:
            f.write(f"Correlation,{corr:.2f}\n")
        f.write(f"CHIRPS_2015_Anom,{anom_c:.1f}\n")

if __name__ == "__main__":
    analyze_cross_product()
