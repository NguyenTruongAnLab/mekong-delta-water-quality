
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy import stats
import os
import glob

try:
    # When analysis is on PYTHONPATH
    from analysis.lib.gold_standard_hydro import load_gold_standard
except Exception:
    # When running as `python analysis/<script>.py`
    from lib.gold_standard_hydro import load_gold_standard

# Configuration (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WL_MASTER_FILE = os.path.join(
    SCRIPT_DIR,
    '..',
    'data',
    'hydrology',
    'data',
    'inputs',
    'agency_cleaned',
    'water_level_hourly_unified_cleaned.csv',
)
WQ_FILE = os.path.join(SCRIPT_DIR, 'mekong_delta_water_quality_1985_2023.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_outputs')

# Also export key figures to the manuscript for Quarto to pick up
MANUSCRIPT_FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'manuscript', 'figures')
os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)

# Primary tidal-river stations with continuous hourly coverage across the study period.
# Can Tho (Hau branch) and My Thuan (Tien branch) are ~90–100 km from the estuarine mouth
# and capture tidal influence on the two main branches.
STATIONS = {
    'CanTho': 'H_CanTho',
    'MyThuan': 'H_MyThuan',
}

STUDY_START = '1985-01-01'
STUDY_END = '2022-12-31'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def _as_float(x: object) -> float:
    """Coerce a scalar-like value (float, numpy scalar, 0-d array) to float."""
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        a = np.asarray(x, dtype=float).ravel()
        return float(a[0])

def load_water_level_master(station_cols: dict[str, str]) -> dict[str, pd.Series]:
    """Load unified agency-cleaned water level series for selected stations.

    The master file contains *mixed* temporal resolution depending on station and year
    (some periods are hourly; some are daily means). We keep the native timestamps here
    and regularize later (inside filtering) as needed.
    """

    print("Loading Unified Water Level Master...")
    if not os.path.exists(WL_MASTER_FILE):
        raise FileNotFoundError(f"Missing unified water level master CSV: {WL_MASTER_FILE}")

    df = pd.read_csv(WL_MASTER_FILE)
    if 'Timestamp' not in df.columns:
        raise ValueError("Unified water level master CSV missing required column: Timestamp")

    df['dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['dt']).set_index('dt').sort_index()
    # Study-period clip
    df = df.loc[STUDY_START:STUDY_END]

    out: dict[str, pd.Series] = {}
    for key, col in station_cols.items():
        if col not in df.columns:
            print(f"Warning: No column for station {key}: {col}")
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        s.name = key
        if len(s) == 0:
            print(f"Warning: Station {key} has no non-null values in {col}")
            continue
        out[key] = s

    return out

def separate_tides(series: pd.Series, cutoff_period_hours: float | None = None):
    """Separate short-term variability from a low-frequency residual using a Butterworth filter.

    Notes
    -----
    The input series may be *hourly* for some periods and *daily means* for others.
    We regularize to a uniform grid before filtering.

    Parameters
    ----------
    series:
        Time-indexed water-level series.
    cutoff_period_hours:
        Low-pass cutoff period expressed in hours.
        If None, choose a default based on sampling cadence:
        - hourly/sub-hourly: 30 hours (remove tides)
        - daily: 30 days (low-frequency residual diagnostic)
    """
    if len(series) < 365:
        return series, series  # Not enough data

    # Infer sampling interval (hours)
    dt = pd.DatetimeIndex(series.index).to_series().diff().median()
    dt_hours = float(dt.total_seconds() / 3600.0) if pd.notnull(dt) else 24.0
    if dt_hours <= 0:
        dt_hours = 24.0

    # Regularize to a uniform grid (required for filtfilt)
    freq = 'h' if dt_hours <= 2.0 else 'D'
    s_reg = series.sort_index().resample(freq).mean()
    filled = s_reg.interpolate(method='time').bfill().ffill()

    if cutoff_period_hours is None:
        cutoff_period_hours = 30.0 if freq == 'h' else 24.0 * 30.0

    fs = 1.0 / dt_hours
    nyquist = 0.5 * fs
    cutoff_freq = (1.0 / float(cutoff_period_hours)) / nyquist
    
    b, a = butter(2, cutoff_freq, btype='low')  # type: ignore[misc]
    
    residual = filtfilt(b, a, filled.values)
    residual_series = pd.Series(residual, index=filled.index)

    # Bring residual back to the original timestamps for downstream calculations/plots
    residual_on_native = residual_series.reindex(series.index, method='nearest')
    tidal_series = series - residual_on_native
    
    return residual_on_native, tidal_series

def analyze_advanced_hydrodynamics(wl_dfs):
    print("\n--- Advanced Hydrodynamics Analysis ---\n")

    base_stations = list(wl_dfs.keys())

    # Tidal/residual decomposition for robust hourly stations.
    station_rows = []
    for st in base_stations:
        s = wl_dfs[st]
        res, short = separate_tides(s, cutoff_period_hours=None)
        wl_dfs[f'{st}_Residual'] = res
        wl_dfs[f'{st}_ShortTerm'] = short

        annual_res = res.resample('YE').mean().dropna()
        if len(annual_res) >= 10:
            years = pd.DatetimeIndex(annual_res.index).year.values
            lr = stats.linregress(years, annual_res.values)
            slope, p_val = lr[0], lr[3]
            print(
                f"{st} residual WL trend ({years.min()}-{years.max()}): "
                f"Slope={float(slope) * 1000.0:.2f} mm/yr, P={float(p_val):.4f}"  # type: ignore[arg-type]
            )

            station_rows.append({
                'Station': st,
                'Residual_Trend_mm_per_yr': float(slope) * 1000.0,  # type: ignore[arg-type]
                'Residual_Trend_pvalue': float(p_val),  # type: ignore[arg-type]
                'Residual_Mean_m': float(np.nanmean(np.asarray(annual_res.to_numpy(dtype=float), dtype=float))),
                'ShortTerm_Std_m': float(np.nanstd(np.asarray(short.dropna().to_numpy(dtype=float), dtype=float))) if len(short.dropna()) else float('nan'),
            })

    if station_rows:
        pd.DataFrame(station_rows).to_csv(os.path.join(OUTPUT_DIR, 'Table_Hydro_Indices.csv'), index=False)

    # Tidal influence indices: daily tidal range, and an optional dry-season (range / Q) index.
    tidal_rows = []
    tidal_year_rows = []
    for st in base_stations:
        s = wl_dfs[st]
        # daily range from hourly series
        daily_range = s.resample('D').max() - s.resample('D').min()
        daily_range = daily_range.dropna()

        # year-resolved diagnostics
        ann = daily_range.resample('YE').mean().dropna()
        if len(ann):
            for ts, val in ann.items():
                tidal_year_rows.append({
                    'Station': st,
                    'year': int(pd.DatetimeIndex([ts]).year[0]),
                    'Metric': 'AnnualMeanDailyTidalRange_m',
                    'Value': float(val),
                })

        dr = daily_range.loc[daily_range.index.month.isin([1, 2, 3, 4])]
        dr_ann = dr.resample('YE').mean().dropna()
        if len(dr_ann):
            for ts, val in dr_ann.items():
                tidal_year_rows.append({
                    'Station': st,
                    'year': int(pd.DatetimeIndex([ts]).year[0]),
                    'Metric': 'DrySeasonMeanDailyTidalRange_m',
                    'Value': float(val),
                })

        annual_range = daily_range.resample('YE').mean().dropna()
        if len(annual_range) >= 10:
            years = pd.DatetimeIndex(annual_range.index).year.values
            slope, intercept, r_val, p_val, std_err = stats.linregress(years, annual_range.values)
            tidal_rows.append({
                'Station': st,
                'Metric': 'AnnualMeanDailyTidalRange_m',
                'Trend_per_year': _as_float(slope),
                'Trend_pvalue': _as_float(p_val),
                'Mean': float(np.nanmean(annual_range.to_numpy(dtype=float))),
            })

        # Dry-season mean daily range (Jan–Apr)
        dr = daily_range.copy()
        dr = dr[dr.index.month.isin([1, 2, 3, 4])]
        annual_dr = dr.resample('YE').mean().dropna()
        if len(annual_dr) >= 10:
            years = pd.DatetimeIndex(annual_dr.index).year.values
            slope, intercept, r_val, p_val, std_err = stats.linregress(years, annual_dr.values)
            tidal_rows.append({
                'Station': st,
                'Metric': 'DrySeasonMeanDailyTidalRange_m',
                'Trend_per_year': _as_float(slope),
                'Trend_pvalue': _as_float(p_val),
                'Mean': float(np.nanmean(annual_dr.to_numpy(dtype=float))),
            })

    # Optional: compute a simple dry-season flushing proxy = (daily tidal range) / (mean discharge)
    # using the curated daily discharge dataset when available.
    try:
        q = load_gold_standard().loc[STUDY_START:STUDY_END]
        q = q[['CanTho', 'MyThuan']].copy()
        q = q.loc[pd.DatetimeIndex(q.index).month.isin([1, 2, 3, 4])]
        q = q.resample('YE').mean(numeric_only=True)

        for st, qcol in [('CanTho', 'CanTho'), ('MyThuan', 'MyThuan')]:
            if st not in wl_dfs:
                continue
            daily_range = (wl_dfs[st].resample('D').max() - wl_dfs[st].resample('D').min()).dropna()
            dr = daily_range[daily_range.index.month.isin([1, 2, 3, 4])].resample('YE').mean().dropna()

            common = dr.index.intersection(q.index)
            if len(common) < 10:
                continue
            idx = (dr.loc[common] / q[qcol].loc[common]).dropna()
            if len(idx) < 10:
                continue
            years = pd.DatetimeIndex(idx.index).year.values
            slope, intercept, r_val, p_val, std_err = stats.linregress(years, idx.values)
            tidal_rows.append({
                'Station': st,
                'Metric': 'DrySeasonTidalRange_over_Discharge',
                'Trend_per_year': _as_float(slope),
                'Trend_pvalue': _as_float(p_val),
                'Mean': float(np.nanmean(idx.to_numpy(dtype=float))),
            })
    except Exception as e:
        print(f"Note: skipping discharge-normalized tidal index (could not load discharge dataset): {e}")

    if tidal_rows:
        df_tidal = pd.DataFrame(tidal_rows)
        # Safety: ensure we only report the intended base stations.
        df_tidal = df_tidal.loc[df_tidal['Station'].isin(base_stations)].copy()
        df_tidal.to_csv(os.path.join(OUTPUT_DIR, 'Table_Tidal_Influence_Indices.csv'), index=False)

    if tidal_year_rows:
        pd.DataFrame(tidal_year_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'Table_Tidal_Range_By_Year.csv'),
            index=False,
        )

    return wl_dfs

def plot_hydrodynamics(wl_dfs):
    print("Plotting Hydrodynamics...")
    
    # Figure: two-station decomposition (Can Tho vs My Thuan)
    bases = [b for b in ['CanTho', 'MyThuan'] if b in wl_dfs and f'{b}_Residual' in wl_dfs and f'{b}_ShortTerm' in wl_dfs]
    if len(bases) == 2:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)

        for j, base in enumerate(bases):
            # (A) total
            wl_dfs[base].plot(ax=axes[0, j], color='black', lw=0.4, label='Total WL')
            axes[0, j].set_title(f'(A) Total Water Level ({base})')
            axes[0, j].legend(loc='upper left')

            # (B) short-term (last 30 days) — independent x-axis so it doesn't collapse to a vertical line
            end = wl_dfs[f'{base}_ShortTerm'].index.max()
            start = end - pd.Timedelta(days=30)
            wl_dfs[f'{base}_ShortTerm'].loc[start:end].plot(ax=axes[1, j], color='blue', lw=0.8, label='Short-term (high-pass)')
            axes[1, j].set_title('(B) Short-term Component (Last 30 days)')
            axes[1, j].set_xlim(start, end)

            # (C) residual
            wl_dfs[f'{base}_Residual'].plot(ax=axes[2, j], color='red', lw=0.6, label='Residual (low-pass)')
            axes[2, j].set_title('(C) Low-frequency Residual Water Level')

        plt.tight_layout()
        out_local = os.path.join(OUTPUT_DIR, 'Figure_Tidal_Decomposition.png')
        plt.savefig(out_local)
        out_manu = os.path.join(MANUSCRIPT_FIG_DIR, 'Figure_Tidal_Decomposition.png')
        plt.savefig(out_manu, dpi=300)
        print('Saved Figure_Tidal_Decomposition.png')

def main():
    wl_dfs = load_water_level_master(STATIONS)
    wl_dfs = analyze_advanced_hydrodynamics(wl_dfs)
    plot_hydrodynamics(wl_dfs)

if __name__ == "__main__":
    main()
