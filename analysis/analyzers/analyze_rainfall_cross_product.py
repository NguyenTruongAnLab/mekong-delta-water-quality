
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Ensure the repository root (`scripts/`) is on sys.path so absolute imports work
# even when executing this file directly (e.g. `python analysis/.../file.py`).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis.lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard

# Configuration
# IMPORTANT: Resolve paths relative to the repository root (the `scripts/` folder),
# not relative to the current working directory.
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

# Keep outputs alongside other analysis artifacts used by the manuscript
OUTPUT_DIR = REPO_ROOT / 'analysis' / 'analysis_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Manuscript figure location (so running this script updates the QMD-linked asset)
MANUSCRIPT_FIG_DIR = REPO_ROOT / 'manuscript' / 'figures'
MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    # Light, journal-friendly defaults (Nature-style-ish): compact fonts, minimal ink.
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

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

    # ---------------------------------------------------------------------
    # Optional: compare CHIRPS basin monthly totals vs local rain-gauge monthly totals.
    # NOTE: This is *not* a like-for-like spatial comparison (basin mean vs point gauges),
    # but is still useful to confirm the seasonal cycle and provide a sanity check.
    # ---------------------------------------------------------------------
    # Hold CHIRPS-vs-gauge monthly data for optional scatter subplots.
    bt_scatter: pd.DataFrame | None = None
    btri_scatter: pd.DataFrame | None = None
    bt_scatter_r = np.nan
    bt_scatter_p = np.nan
    bt_scatter_slope = np.nan
    bt_scatter_bias = np.nan
    btri_scatter_r = np.nan
    btri_scatter_p = np.nan
    btri_scatter_slope = np.nan
    btri_scatter_bias = np.nan

    if BEN_TRE_GAUGE_MONTHLY_FILE.exists():
        gauge_df = pd.read_csv(BEN_TRE_GAUGE_MONTHLY_FILE, engine='python')
        gauge_df['Date'] = pd.to_datetime(gauge_df['Date'], format='%m/%Y', errors='coerce')
        gauge_df = gauge_df.dropna(subset=['Date']).set_index('Date').sort_index()
        gauge_df = gauge_df.loc[STUDY_START:STUDY_END]

        # Values appear to be monthly mean daily rainfall (mm/day). Convert to monthly totals (mm/month).
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

        # Full monthly series correlation (seasonality dominates here)
        rows = []
        rows.append(_corr_row('CHIRPS vs Ben Tre (raw monthly)', m['chirps_basin_mm_month'], m['ben_tre_gauge_mm_month']))
        rows.append(_corr_row('CHIRPS vs Ba Tri (raw monthly)', m['chirps_basin_mm_month'], m['ba_tri_gauge_mm_month']))

        # Keep raw-monthly data for scatter subplots (clearer than adding additional
        # noisy series onto the annual time-series panel).
        bt_scatter = m[['chirps_basin_mm_month', 'ben_tre_gauge_mm_month']].dropna().copy()
        btri_scatter = m[['chirps_basin_mm_month', 'ba_tri_gauge_mm_month']].dropna().copy()

        if len(bt_scatter) >= 12:
            bt_scatter_r = float(rows[0]['Pearson_r'])
            bt_scatter_p = float(rows[0]['Pearson_p'])
            bt_scatter_bias = float(
                bt_scatter['chirps_basin_mm_month'].mean() - bt_scatter['ben_tre_gauge_mm_month'].mean()
            )
            x_bt = np.asarray(bt_scatter['ben_tre_gauge_mm_month'].values, dtype=float)
            y_bt = np.asarray(bt_scatter['chirps_basin_mm_month'].values, dtype=float)
            bt_scatter_slope = float(np.polyfit(x_bt, y_bt, deg=1)[0])

        if len(btri_scatter) >= 12:
            btri_scatter_r = float(rows[1]['Pearson_r'])
            btri_scatter_p = float(rows[1]['Pearson_p'])
            btri_scatter_bias = float(
                btri_scatter['chirps_basin_mm_month'].mean() - btri_scatter['ba_tri_gauge_mm_month'].mean()
            )
            x_btri = np.asarray(btri_scatter['ba_tri_gauge_mm_month'].values, dtype=float)
            y_btri = np.asarray(btri_scatter['chirps_basin_mm_month'].values, dtype=float)
            btri_scatter_slope = float(np.polyfit(x_btri, y_btri, deg=1)[0])

        # Wet season only (reduces seasonal-cycle leverage)
        m_month = pd.DatetimeIndex(m.index).month
        wet = m[m_month.isin(WET_MONTHS)]
        if len(wet) >= 10:
            rows.append(_corr_row('CHIRPS vs Ben Tre (wet months only)', wet['chirps_basin_mm_month'], wet['ben_tre_gauge_mm_month']))
            rows.append(_corr_row('CHIRPS vs Ba Tri (wet months only)', wet['chirps_basin_mm_month'], wet['ba_tri_gauge_mm_month']))

        # Monthly anomalies (remove mean seasonal cycle month-by-month)
        mo = pd.DatetimeIndex(m.index).month
        anom = m.copy()
        for col in ['chirps_basin_mm_month', 'ben_tre_gauge_mm_month', 'ba_tri_gauge_mm_month']:
            anom[col] = m[col] - m.groupby(mo)[col].transform('mean')
        rows.append(_corr_row('CHIRPS vs Ben Tre (monthly anomalies)', anom['chirps_basin_mm_month'], anom['ben_tre_gauge_mm_month']))
        rows.append(_corr_row('CHIRPS vs Ba Tri (monthly anomalies)', anom['chirps_basin_mm_month'], anom['ba_tri_gauge_mm_month']))

        df_corr = pd.DataFrame(rows)
        out_corr = OUTPUT_DIR / 'chirps_vs_gauge_monthly_stats_latest.csv'
        df_corr.to_csv(out_corr, index=False)

        print("\nCHIRPS vs gauge (monthly totals, basin vs point; interpret with care):")
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
        # Align
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

    # ---------------------------------------------------------------------
    # Seasonal decomposition: dry vs wet season totals
    # ---------------------------------------------------------------------
    chirps_dry = _seasonal_totals(chirps_daily, DRY_MONTHS)
    chirps_wet = _seasonal_totals(chirps_daily, WET_MONTHS)

    era5_dry = era5_wet = None
    if era5_annual is not None and era5_df is not None:
        era5_daily = era5_df['precipitation_mm'].astype(float).resample('D').sum()
        era5_daily = era5_daily.loc[STUDY_START:STUDY_END]
        era5_dry = _seasonal_totals(era5_daily, DRY_MONTHS)
        era5_wet = _seasonal_totals(era5_daily, WET_MONTHS)

    # Align all seasonal series to common year-end index
    idx_s = chirps_dry.index if era5_dry is None else era5_dry.index.intersection(chirps_dry.index)
    years_s = pd.DatetimeIndex(idx_s).year.values

    seasonal_rows = []
    rows_in = [
        ('CHIRPS', 'Dry (Jan–Apr)', chirps_dry.loc[idx_s]),
        ('CHIRPS', 'Wet (Jun–Sep)', chirps_wet.loc[idx_s]),
    ]
    if era5_dry is not None and era5_wet is not None:
        rows_in = [
            ('ERA5', 'Dry (Jan–Apr)', era5_dry.loc[idx_s]),
            ('ERA5', 'Wet (Jun–Sep)', era5_wet.loc[idx_s]),
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
    
    # 2. Plotting The "Consistency Check" (Figure C1)
    # Three-panel layout:
    #   (a) Annual basin-scale time series (ERA5 vs CHIRPS)
    #   (b) Monthly scatter: CHIRPS vs Ben Tre gauge
    #   (c) Monthly scatter: CHIRPS vs Ba Tri gauge
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.6])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_bt = fig.add_subplot(gs[1, 0])
    ax_btri = fig.add_subplot(gs[1, 1], sharey=ax_bt)

    years_plot = np.asarray(years, dtype=int)

    # (a) Annual basin-scale rainfall totals
    if era5_annual is not None:
        ax_ts.plot(
            years_plot,
            np.asarray(era5_c.values, dtype=float),
            label='ERA5 (basin)',
            color='blue',
            alpha=0.65,
        )
    ax_ts.plot(
        years_plot,
        np.asarray(chirps_c.values, dtype=float),
        label='CHIRPS (basin)',
        color='green',
        alpha=0.65,
    )

    # Trend lines (basin series)
    if e_trend is not None:
        e_slope = float(e_trend[0])  # type: ignore[arg-type]
        e_intercept = float(e_trend[1])  # type: ignore[arg-type]
        ax_ts.plot(
            years_plot,
            e_intercept + e_slope * years_plot,
            'b--',
            alpha=0.85,
            linewidth=1.5,
        )

    c_slope = float(c_trend[0])  # type: ignore[arg-type]
    c_intercept = float(c_trend[1])  # type: ignore[arg-type]
    ax_ts.plot(
        years_plot,
        c_intercept + c_slope * years_plot,
        'g--',
        alpha=0.85,
        linewidth=1.5,
    )

    if era5_annual is not None:
        ax_ts.set_title(f"Rainfall consistency check (1985–2022)\nCHIRPS vs ERA5: r={corr:.2f}, bias={bias:.0f} mm/yr")
    else:
        ax_ts.set_title("Rainfall time series (CHIRPS basin, 1985–2022)")
    ax_ts.set_ylabel('Annual rainfall (mm) — basin mean')
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='upper left')

    # Compact annotation with key basin-scale metrics.
    ann_lines = [
        f"CHIRPS trend: {float(c_trend[0]):.1f} mm/yr (p={float(c_trend[3]):.3f})",  # type: ignore[arg-type]
    ]
    if e_trend is not None:
        ann_lines.append(
            f"ERA5 trend: {float(e_trend[0]):.1f} mm/yr (p={float(e_trend[3]):.3f})"  # type: ignore[arg-type]
        )
    ax_ts.text(
        0.02,
        0.02,
        "\n".join(ann_lines),
        transform=ax_ts.transAxes,
        fontsize=9,
        va='bottom',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='0.7'),
    )

    def _scatter_panel(
        ax,
        df: pd.DataFrame | None,
        xcol: str,
        ycol: str,
        title: str,
        r: float,
        p: float,
        slope: float,
        bias: float,
    ):
        ax.set_title(title)
        ax.set_xlabel('Gauge rainfall (mm/month)')
        ax.grid(False)

        if df is None or len(df) < 12:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
            return None

        x = np.asarray(df[xcol].values, dtype=float)
        y = np.asarray(df[ycol].values, dtype=float)
        months = np.asarray(pd.DatetimeIndex(df.index).month, dtype=int)

        mappable = ax.scatter(
            x,
            y,
            c=months,
            cmap='turbo',
            s=16,
            alpha=0.65,
            edgecolors='none',
            rasterized=True,
        )

        lim = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        ax.plot([0, lim], [0, lim], color='0.5', linewidth=1.0, linestyle=':', label='1:1')
        fit_slope, fit_intercept = np.polyfit(x, y, deg=1)
        xx = np.linspace(0, lim, 100)
        ax.plot(xx, fit_intercept + fit_slope * xx, color='black', linewidth=1.2, alpha=0.85, label='Fit')
        ax.legend(loc='lower right', frameon=False)

        # Put text in the least intrusive corner.
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"r={r:.2f} (p={p:.1g})",
                    f"slope={slope:.2f}",
                    f"bias={bias:.1f} mm/mo",
                ]
            ),
            transform=ax.transAxes,
            fontsize=8.5,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='0.8'),
        )
        return mappable

    ax_bt.set_ylabel('CHIRPS basin rainfall (mm/month)')
    mbt = _scatter_panel(
        ax_bt,
        bt_scatter,
        xcol='ben_tre_gauge_mm_month',
        ycol='chirps_basin_mm_month',
        title='CHIRPS vs Ben Tre gauge (monthly totals)',
        r=bt_scatter_r,
        p=bt_scatter_p,
        slope=bt_scatter_slope,
        bias=bt_scatter_bias,
    )
    mbtri = _scatter_panel(
        ax_btri,
        btri_scatter,
        xcol='ba_tri_gauge_mm_month',
        ycol='chirps_basin_mm_month',
        title='CHIRPS vs Ba Tri gauge (monthly totals)',
        r=btri_scatter_r,
        p=btri_scatter_p,
        slope=btri_scatter_slope,
        bias=btri_scatter_bias,
    )

    # One shared colorbar for month-of-year across both scatter panels.
    m_for_cbar = mbtri if mbtri is not None else mbt
    if m_for_cbar is not None:
        cbar = fig.colorbar(m_for_cbar, ax=[ax_bt, ax_btri], pad=0.01, fraction=0.05)
        cbar.set_label('Month')
        cbar.set_ticks([1, 3, 6, 9, 12])

    out_file = OUTPUT_DIR / 'Figure_Cross_Product_Rainfall.pdf'
    fig.savefig(out_file, dpi=300)
    print(f"Saved Figure C1 to {out_file}")

    # Keep the manuscript-linked figure in sync
    manuscript_copy = MANUSCRIPT_FIG_DIR / 'Figure_Cross_Product_Rainfall.pdf'
    try:
        shutil.copyfile(out_file, manuscript_copy)
        print(f"Copied figure to {manuscript_copy}")
    except Exception as e:
        print(f"Warning: failed to copy figure to manuscript folder: {e}")
    
    # 3. Reality Check: 2015 Drought
    # Calculate anomaly for 2015
    lt_mean_c = chirps_c.mean()
    val_2015_c = chirps_c.loc[pd.Timestamp('2015-12-31')]
    anom_c = (val_2015_c - lt_mean_c) / lt_mean_c * 100
    
    print(f"\nReality Check (2015 Drought):")
    print(f"CHIRPS 2015 Value: {val_2015_c:.1f} mm")
    print(f"CHIRPS LTA: {lt_mean_c:.1f} mm")
    print(f"CHIRPS Anomaly: {anom_c:.1f}%")
    
    # 4. Save Stats for Table C2
    with open(OUTPUT_DIR / 'cross_product_stats.txt', "w") as f:
        if e_trend is not None:
            f.write(f"ERA5_Trend,{float(e_trend[0]):.1f}\n")  # type: ignore[arg-type]
            # Robust
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
