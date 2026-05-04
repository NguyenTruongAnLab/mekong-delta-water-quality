#!/usr/bin/env python3
"""
Extended Data Figures 2, 4 — Hydrodynamics
==========================================
ED Fig 2: Conservative mixing diagnostics (salinity as tracer)
ED Fig 4: Tidal range decomposition & trends

All export companion metrics CSVs.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau

from _nw_common import (
    apply_style, load_wq, MM, MRC_STATIONS, STATION_CLR,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR,
    HFI_CSV, TIDAL_CSV, MIX_CSV,
    panel_label, save_fig,
)

apply_style()


def _robust_endmembers(df: pd.DataFrame, ycol: str, *, min_n: int = 25) -> tuple[float, float] | None:
    """Compute robust fresh/sea endmembers for conservative mixing.

    Uses salinity thresholds when possible and falls back to salinity quantiles
    if sampling is sparse at the endpoints.
    """
    if "Salinity" not in df.columns or ycol not in df.columns:
        return None
    d = df[["Salinity", ycol]].dropna().copy()
    if d.empty:
        return None
    d["Salinity"] = d["Salinity"].astype(float)
    d[ycol] = d[ycol].astype(float)

    fresh = d.loc[d["Salinity"] <= 0.5, ycol]
    sea = d.loc[d["Salinity"] >= 20.0, ycol]

    if len(fresh) < min_n:
        q = float(np.nanpercentile(d["Salinity"].to_numpy(dtype=float), 10))
        fresh = d.loc[d["Salinity"] <= q, ycol]
    if len(sea) < min_n:
        q = float(np.nanpercentile(d["Salinity"].to_numpy(dtype=float), 90))
        sea = d.loc[d["Salinity"] >= q, ycol]

    if len(fresh) < 8 or len(sea) < 8:
        return None

    return float(np.nanmedian(fresh)), float(np.nanmedian(sea))


def _c_mix(sal: pd.Series | np.ndarray, c_fresh: float, c_sea: float, *, s_sea: float) -> np.ndarray:
    s = np.asarray(sal, dtype=float)
    f = np.clip(s / float(s_sea), 0.0, 1.0)
    return c_fresh + (c_sea - c_fresh) * f


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 4 — Conservative Mixing Diagnostics
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_4() -> None:
    wq = load_wq()
    bt = wq[wq["Data source"] == "Ben Tre"].copy()
    metrics = []

    # Periods used for Ben Tre mixing diagnostics in the manuscript
    early = (2005, 2012)
    late = (2013, 2022)
    p1_label = f"{early[0]}–{early[1]}"
    p2_label = f"{late[0]}–{late[1]}"

    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.40,
                           height_ratios=[1.05, 1.0])

    # ── (a) Non-conservative excess summary ──
    ax = fig.add_subplot(gs[0, :])

    S_SEA = 30.0
    # Parameters to summarize in panel (a). Note: these are *diagnostics* of
    # non-conservative behaviour; they are not assumed conservative.
    candidate_params = [
        ("COD (mg/L)", "COD"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N"),
        ("TN (mg/L)", "TN"),
        ("TP (mg/L)", "TP"),
        ("TSS (mg/L)", "TSS"),
        ("DO (mg/L)", "DO"),
    ]

    available = []
    for pcol, short in candidate_params:
        if pcol in bt.columns and "Salinity" in bt.columns:
            em = _robust_endmembers(bt, pcol)
            if em is not None:
                available.append((pcol, short, em))

    if not available:
        ax.text(0.5, 0.5, "Mixing diagnostics unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#969696")
    else:
        x_pos = np.arange(len(available))
        w = 0.35
        period_clrs = {p1_label: PHASE1_CLR, p2_label: PHASE2_CLR}
        all_vals: list[float] = []

        for pidx, (label, yr0, yr1) in enumerate([(p1_label, early[0], early[1]), (p2_label, late[0], late[1])] ):
            vals = []
            offset = -w / 2 + pidx * w
            for i, (pcol, short, (fresh, sea)) in enumerate(available):
                sub = bt[(bt["year"].astype(int) >= yr0) & (bt["year"].astype(int) <= yr1)][["Salinity", pcol]].dropna()
                expected = _c_mix(sub["Salinity"], fresh, sea, s_sea=S_SEA)
                excess = (sub[pcol].to_numpy(dtype=float) - expected)
                val = float(np.nanmean(excess)) if excess.size else np.nan
                vals.append(val)
                if np.isfinite(val):
                    all_vals.append(val)
                metrics.append({
                    "Figure": "ED4",
                    "Panel": "a",
                    "Variable": pcol,
                    "Variable_short": short,
                    "Period": label,
                    "Excess_Mean": None if not np.isfinite(val) else round(val, 4),
                    "Fresh_Endmember": round(fresh, 4),
                    "Sea_Endmember": round(sea, 4),
                    "N": int(len(sub)),
                })

            ax.bar(x_pos + offset, vals, w,
                   color=period_clrs.get(label, "#969696"), alpha=0.85,
                   edgecolor="white", linewidth=0.3, label=label, zorder=3)

            for i, v in enumerate(vals):
                if np.isfinite(v) and abs(v) > 0.001:
                    ax.text(x_pos[i] + offset, v, f"{v:.3f}",
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7, fontweight="bold", color="#333333")

        ax.axhline(0, color="black", lw=0.5)

        abs_vals = np.abs(np.asarray(all_vals, dtype=float))
        abs_vals = abs_vals[np.isfinite(abs_vals) & (abs_vals > 0)]
        if abs_vals.size:
            linthresh = float(np.clip(np.percentile(abs_vals, 20), 1e-4, 1e3))
            ax.set_yscale("symlog", linthresh=linthresh)

        # Clean y-ticks: show only 0 and major decades (±1, ±10, ±100)
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_locator(mticker.FixedLocator(
            [-100, -10, -1, 0, 1, 10, 100]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: "0" if x == 0 else f"{x:g}"))

        ax.set_xticks(x_pos)
        ax.set_xticklabels([short for _, short, _ in available], fontsize=8)
        ax.set_ylabel(r"Non-conservative excess" + "\n" + r"(observed $-$ expected)", fontsize=8)
        panel_label(ax, "(a)")
        ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")
        ax.text(0.98, 0.97, "Above zero = in-delta source",
                transform=ax.transAxes, fontsize=6, color="#d62728",
                ha="right", va="top", style="italic")
        ax.text(0.98, 0.88, "Below zero = in-delta sink",
                transform=ax.transAxes, fontsize=6, color="#1f77b4",
                ha="right", va="top", style="italic")

    # ── (b–d) C–Salinity mixing plots for key solutes ──
    mixing_params = [
        ("COD (mg/L)", r"COD (mg L$^{-1}$)"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N (mg L$^{-1}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg L$^{-1}$)"),
    ]

    for pidx, (pcol, ylabel) in enumerate(mixing_params):
        ax = fig.add_subplot(gs[1, pidx])
        has_data = "Salinity" in bt.columns and pcol in bt.columns
        if not has_data:
            ax.text(0.5, 0.5, "Data not available", transform=ax.transAxes, ha="center")
            continue

        sub_all = bt[["year", "Salinity", pcol]].dropna().copy()
        em = _robust_endmembers(sub_all, pcol)
        if em is None:
            ax.text(0.5, 0.5, "Insufficient endpoints", transform=ax.transAxes, ha="center")
            continue
        fresh, saline = em

        # Assign period as object dtype to avoid numpy string/NaN dtype promotion issues
        sub_all["period"] = pd.Series(index=sub_all.index, dtype="object")
        m1 = (sub_all["year"].astype(int) >= early[0]) & (sub_all["year"].astype(int) <= early[1])
        m2 = (sub_all["year"].astype(int) >= late[0]) & (sub_all["year"].astype(int) <= late[1])
        sub_all.loc[m1, "period"] = p1_label
        sub_all.loc[m2, "period"] = p2_label
        sub_all = sub_all.dropna(subset=["period"])

        s_line = np.linspace(0, S_SEA, 100)
        c_mix = fresh + (saline - fresh) * (s_line / S_SEA)
        ax.plot(s_line, c_mix, "-", color="#969696", lw=1.2, alpha=0.7,
                label="Conservative mixing", zorder=2)

        for label, clr in [(p1_label, PHASE1_CLR), (p2_label, PHASE2_CLR)]:
            p_sub = sub_all[sub_all["period"] == label]
            if len(p_sub) > 0:
                ax.scatter(p_sub["Salinity"], p_sub[pcol],
                           s=12, c=clr, alpha=0.35, edgecolors="none",
                           label=label, zorder=3)
                # Mean excess
                expected = fresh + (saline - fresh) * (p_sub["Salinity"] / S_SEA)
                excess = (p_sub[pcol].values - expected.values).mean()
                param_short = pcol.split(" (")[0]
                metrics.append({"Figure": "ED4", "Panel": f"{chr(ord('b') + pidx)}_{param_short}",
                                "Period": label,
                                "Mean_Excess": round(excess, 3),
                                "N": len(p_sub)})

        ax.set_xlabel("Salinity", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        panel_titles = [
            r"(b) COD vs salinity",
            r"(c) NH$_4^+$-N vs salinity",
            r"(d) NO$_3^-$-N vs salinity",
        ]
        ax.set_title(panel_titles[pidx], fontsize=8, fontweight="bold", pad=4)
        if pidx == 0:
            ax.legend(fontsize=6, frameon=True, framealpha=0.9,
                      edgecolor="none", loc="upper left")

    save_fig(fig, "ED_Figure_2_Mixing_Diagnostics", pd.DataFrame(metrics))


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 5 — Hydrodynamic Flushing Index (TR/Q)
# ═══════════════════════════════════════════════════════════════════════
def ed_figure_5() -> None:
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM))

    # Map raw station names (no spaces) → display names and STATION_CLR keys
    _stn_display = {"CanTho": "Can Tho (Hau/Bassac)", "MyThuan": "My Thuan (Tien/Mekong)",
                    "Can Tho": "Can Tho (Hau/Bassac)", "My Thuan": "My Thuan (Tien/Mekong)"}
    _stn_clr_key = {"CanTho": "Can Tho", "MyThuan": "My Thuan",
                    "Can Tho": "Can Tho", "My Thuan": "My Thuan"}

    if HFI_CSV.exists():
        hfi = pd.read_csv(HFI_CSV)

        ax = axes[0]
        for stn in hfi["Station"].unique():
            sub = hfi[hfi["Station"] == stn].sort_values("year")
            clr = STATION_CLR.get(_stn_clr_key.get(stn, stn), "#333333")
            display = _stn_display.get(stn, stn)
            hfi_col = "HFI_TR_over_Q" if "HFI_TR_over_Q" in sub.columns else sub.columns[-1]
            ax.plot(sub["year"], sub[hfi_col] * 1000, "-o",
                    color=clr, lw=1.0, ms=2, alpha=0.85, label=display, zorder=3)
            x = sub["year"].values.astype(float)
            y = (sub[hfi_col] * 1000).values
            if len(x) > 5:
                # Two-period Theil-Sen trends split at BREAK_YEAR
                m1 = x <= BREAK_YEAR
                m2 = x > BREAK_YEAR
                for mask, period_label in [(m1, "pre"), (m2, "post")]:
                    xm, ym = x[mask], y[mask]
                    if len(xm) > 3:
                        slope, intercept, _, _ = theilslopes(ym, xm)
                        _, p = kendalltau(xm, ym)
                        ax.plot(xm, intercept + slope * xm, "--", color=clr, lw=0.7, alpha=0.5)
                        metrics.append({"Figure": "ED5", "Station": stn,
                                        "Period": period_label,
                                        "HFI_trend_per_yr": round(slope, 6),
                                        "p": round(p, 6)})

        ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(r"HFI: TR/Q ($\times 10^{-3}$ m/($\mathrm{m^3\,s^{-1}}$))")
        ax.set_title("(a) Hydrodynamic flushing index", fontsize=8, fontweight="bold", pad=4)
        ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")
        ax.annotate(r"$\rightarrow$ increasing tidal dominance",
                    xy=(2020, ax.get_ylim()[1] * 0.85),
                    fontsize=6, color="#d62728", style="italic", ha="right")

        ax = axes[1]
        q_col = [c for c in hfi.columns if "Q" in c and "dry" in c.lower()]
        tr_col = [c for c in hfi.columns if "TR" in c and "dry" in c.lower()]
        if q_col and tr_col:
            for si, stn in enumerate(hfi["Station"].unique()):
                sub = hfi[hfi["Station"] == stn].sort_values("year")
                clr = STATION_CLR.get(_stn_clr_key.get(stn, stn), "#333333")
                display = _stn_display.get(stn, stn)
                ax.plot(sub["year"], sub[q_col[0]], "-o", color=clr,
                        lw=1.0, ms=2, alpha=0.85, zorder=3,
                        label=f"{display.split(' (')[0]} Q")
            ax.set_ylabel(r"Dry-season Q (m$^3$ s$^{-1}$)", fontsize=8)
            ax.tick_params(axis="y")
            ax2 = ax.twinx()
            for si, stn in enumerate(hfi["Station"].unique()):
                sub = hfi[hfi["Station"] == stn].sort_values("year")
                clr = STATION_CLR.get(_stn_clr_key.get(stn, stn), "#333333")
                display = _stn_display.get(stn, stn)
                ax2.plot(sub["year"], sub[tr_col[0]], "s--", color=clr,
                         lw=0.8, ms=2, alpha=0.55, zorder=2,
                         label=f"{display.split(' (')[0]} TR")
            ax2.set_ylabel("Dry-season TR (m)", fontsize=8)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color("#bdbdbd")
            # Combine legends from both axes
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=True,
                      framealpha=0.9, edgecolor="none", loc="upper left")
        ax.axvline(BREAK_YEAR + 0.5, color="#525252", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(r"(b) Dry-season discharge & tidal range", fontsize=8, fontweight="bold", pad=4)
    else:
        axes[0].text(0.5, 0.5, "HFI data not found", transform=axes[0].transAxes,
                     ha="center", fontsize=8, color="#969696")

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "ED_Figure_3_HFI_TR_over_Q", pd.DataFrame(metrics))


# ═══════════════════════════════════════════════════════════════════════
# ED FIGURE 6 — Tidal Range Decomposition
# ═══════════════════════════════════════════════════════════════════════
def _detect_changepoint(y: np.ndarray) -> int | None:
    """Detect a single changepoint in a time series using PELT (L2 cost).

    Uses a BIC-like penalty scaled conservatively to favour a clear break.
    """
    import ruptures as rpt
    if len(y) < 10:
        return None
    algo = rpt.Pelt(model="l2", min_size=5).fit(y)
    bkps = algo.predict(pen=np.log(len(y)) * np.var(y))
    # bkps always ends with len(y); take the first interior breakpoint
    interior = [b for b in bkps if b < len(y)]
    return interior[0] if interior else None


def _detect_changepoint_pooled(series_list: list[tuple[np.ndarray, np.ndarray]]) -> int | None:
    """Detect a common changepoint by averaging multiple aligned time series.

    Each element is (years, values).  Aligns on the intersection of years,
    z-normalises each series, averages, then runs Binseg (L2, n_bkps=1).
    Returns the changepoint year (not index).
    """
    import ruptures as rpt
    # Find common year range
    all_years = None
    for years, _ in series_list:
        s = set(years.astype(int))
        all_years = s if all_years is None else all_years & s
    if all_years is None or len(all_years) < 10:
        return None
    common = sorted(all_years)
    combined = []
    for years, values in series_list:
        yr_int = years.astype(int)
        idx = [list(yr_int).index(c) for c in common]
        v = values[idx].astype(float)
        # z-normalise
        v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        combined.append(v)
    pooled = np.nanmean(combined, axis=0)
    algo = rpt.Binseg(model="l2", min_size=5).fit(pooled)
    bkps = algo.predict(n_bkps=1)
    cp_idx = bkps[0]
    if cp_idx >= len(common):
        return None
    return common[cp_idx]


def _compute_tidal_range_from_raw() -> pd.DataFrame:
    """Compute annual and dry-season mean daily tidal range from raw hourly data.

    Returns a long-format DataFrame with columns: Station, year, Metric, Value.
    """
    from pathlib import Path
    wl_file = Path(__file__).resolve().parent.parent.parent.parent / (
        "data/hydrology/data/inputs/agency_cleaned/"
        "water_level_hourly_unified_cleaned.csv"
    )
    if not wl_file.exists():
        raise FileNotFoundError(f"Raw hourly water level file not found: {wl_file}")
    df = pd.read_csv(wl_file, parse_dates=["Timestamp"]).set_index("Timestamp").sort_index()
    rows: list[dict] = []
    for col, stn in [("H_CanTho", "CanTho"), ("H_MyThuan", "MyThuan")]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        daily_range = (s.resample("D").max() - s.resample("D").min()).dropna()
        # Require >= 20 hourly obs per day
        daily_count = s.resample("D").count()
        daily_range = daily_range[daily_count >= 20]
        # Annual mean
        ann = daily_range.resample("YE").mean().dropna()
        ann_count = daily_range.resample("YE").count()
        for ts, val in ann[ann_count >= 300].items():
            rows.append({"Station": stn, "year": ts.year,
                         "Metric": "AnnualMeanDailyTidalRange_m", "Value": float(val)})
        # Dry season (Jan-Apr)
        dry = daily_range[daily_range.index.month.isin([1, 2, 3, 4])]
        dry_ann = dry.resample("YE").mean().dropna()
        dry_count = dry.resample("YE").count()
        for ts, val in dry_ann[dry_count >= 90].items():
            rows.append({"Station": stn, "year": ts.year,
                         "Metric": "DrySeasonMeanDailyTidalRange_m", "Value": float(val)})
    return pd.DataFrame(rows)


def _find_pooled_minimum(tidal: pd.DataFrame) -> int:
    """Find the year of minimum pooled tidal range across both stations and metrics.

    Z-normalises each series, averages, and returns the year of the minimum
    — the physical nadir where tidal range reverses from decreasing to increasing.
    """
    common_years: set[int] | None = None
    series_data: list[tuple[np.ndarray, np.ndarray]] = []
    for stn in tidal["Station"].unique():
        for metric in tidal["Metric"].unique():
            sub = tidal[(tidal["Station"] == stn) & (tidal["Metric"] == metric)].sort_values("year")
            if len(sub) < 10:
                continue
            yrs = sub["year"].values.astype(int)
            vals = sub["Value"].values.astype(float)
            series_data.append((yrs, vals))
            s = set(yrs)
            common_years = s if common_years is None else common_years & s
    if common_years is None or len(common_years) < 10:
        return BREAK_YEAR
    common = sorted(common_years)
    combined = []
    for yrs, vals in series_data:
        yr_list = list(yrs)
        idx = [yr_list.index(c) for c in common]
        v = vals[idx].astype(float)
        v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        combined.append(v)
    pooled = np.nanmean(combined, axis=0)
    min_idx = int(np.argmin(pooled))
    return common[min_idx]


def ed_figure_6() -> None:
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM))

    # Load tidal range data — prefer pre-computed CSV, fall back to raw hourly
    if TIDAL_CSV.exists():
        tidal = pd.read_csv(TIDAL_CSV)
    else:
        print("  TIDAL_CSV not found — computing from raw hourly water levels...")
        tidal = _compute_tidal_range_from_raw()

    if tidal.empty:
        axes[0].text(0.5, 0.5, "Tidal data not available", transform=axes[0].transAxes,
                     ha="center", fontsize=8, color="#969696")
        fig.tight_layout(w_pad=2.0)
        save_fig(fig, "ED_Figure_4_Tidal_Decomposition", pd.DataFrame(metrics))
        return

    # Single pooled break year (physical nadir of tidal range)
    cp_year = _find_pooled_minimum(tidal)
    print(f"  Pooled tidal-range minimum year: {cp_year}")

    # Station display names and colors
    stn_label = {"CanTho": "Can Tho (Hau/Bassac)", "MyThuan": "My Thuan (Tien/Mekong)"}
    stn_clr = {"CanTho": STATION_CLR.get("Can Tho", "#4393c3"),
               "MyThuan": STATION_CLR.get("My Thuan", "#f4a582")}

    for panel_idx, (metric_key, metric_label, ylabel) in enumerate([
        ("AnnualMeanDailyTidalRange_m", "AnnualMeanTR",
         "Annual mean tidal range (m)"),
        ("DrySeasonMeanDailyTidalRange_m", "DrySeasonTR",
         "Dry-season tidal range (m)"),
    ]):
        ax = axes[panel_idx]
        for stn in tidal["Station"].unique():
            sub = tidal[(tidal["Station"] == stn) &
                        (tidal["Metric"] == metric_key)].sort_values("year")
            if sub.empty:
                continue
            clr = stn_clr.get(stn, "#333333")
            label = stn_label.get(stn, stn)
            ax.plot(sub["year"], sub["Value"], "-o", color=clr,
                    lw=1.0, ms=2.5, alpha=0.85, label=label, zorder=3)
            x = sub["year"].values.astype(float)
            y = sub["Value"].values.astype(float)
            # Two-period Theil-Sen regression around the pooled break
            for mask, period_label in [(x <= cp_year, "pre"), (x > cp_year, "post")]:
                xm, ym = x[mask], y[mask]
                if len(xm) > 3:
                    slope, intercept, _, _ = theilslopes(ym, xm)
                    _, p = kendalltau(xm, ym)
                    ax.plot(xm, intercept + slope * xm, "--", color=clr, lw=0.7, alpha=0.5)
                    cm_yr = slope * 100  # m/yr → cm/yr
                    sign = "+" if cm_yr >= 0 else ""
                    txt = f"{sign}{cm_yr:.1f} cm yr$^{{-1}}$"
                    # Place label at the outer end of the segment (left for pre, right for post)
                    if period_label == "pre":
                        lx = xm[1]  # near the left edge
                        ly = intercept + slope * lx
                        ha = "left"
                    else:
                        lx = xm[-2]  # near the right edge
                        ly = intercept + slope * lx
                        ha = "right"
                    # Vertical offset: CanTho above, MyThuan below
                    offset = 0.06 if stn == "CanTho" else -0.08
                    ax.text(lx, ly + offset, txt, fontsize=5.5, color=clr,
                            ha=ha, va="bottom" if offset > 0 else "top",
                            fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.7, pad=0.5))
                    metrics.append({"Figure": "ED6", "Station": stn,
                                    "Metric": metric_label, "Period": period_label,
                                    "Changepoint_year": cp_year,
                                    "Trend_cm_per_decade": round(slope * 1000, 2),
                                    "Trend_cm_per_year": round(cm_yr, 2),
                                    "p": round(p, 6)})
        ax.axvline(cp_year + 0.5, color="#555555", lw=0.8, ls=":", alpha=0.5, zorder=1)
        ax.set_ylabel(ylabel, fontsize=8)
        panel_letter = chr(ord("a") + panel_idx)
        ax.set_title(f"({panel_letter}) {ylabel.replace(' (m)', '')}",
                     fontsize=8, fontweight="bold", pad=4)
        ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "ED_Figure_4_Tidal_Decomposition", pd.DataFrame(metrics))


if __name__ == "__main__":
    print("=" * 65)
    print("Extended Data Figures 4-6: Hydrodynamics")
    print("=" * 65)
    ed_figure_4()
    ed_figure_5()
    ed_figure_6()
    print("Done.")
