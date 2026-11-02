"""Phase 5: Changepoint stability analysis.

Evaluates PELT changepoint robustness across a penalty grid and reports
breakpoint support probability and effect sizes (Δmean, Δslope) with
bootstrap confidence intervals.

Outputs
-------
- analysis/analysis_outputs/changepoint_stability_grid_latest.csv
- analysis/analysis_outputs/changepoint_effect_sizes_latest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "analysis" / "analysis_outputs"
CACHE_DIR = ROOT / "analysis" / "cache"
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"

# MRC stations for regime-shift analysis
MRC_STATIONS = ["Can Tho", "My Tho", "My Thuan", "Tan Chau"]
PARAMS = {"COD (mg/L)": "COD", "DO (mg/L)": "DO", "NO3-N (mg/L)": "NO3",
          "NH4-N (mg/L)": "NH4", "TSS (mg/L)": "TSS"}

PENALTY_GRID = np.arange(0.5, 6.1, 0.5)


def _annual_mean_series(wq: pd.DataFrame, station: str, param: str) -> pd.Series:
    """Return annual mean series for one station + parameter (capped at 2022)."""
    sub = wq[(wq["stationName"] == station)].copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"])
    sub["year"] = sub["datetime"].dt.year
    sub = sub[sub["year"] <= 2022]
    annual = sub.groupby("year")[param].mean().dropna()
    return annual


def detect_breakpoints_grid(series: pd.Series, penalties: np.ndarray,
                             model: str = "rbf", min_size: int = 4) -> pd.DataFrame:
    """Run PELT across penalty grid and record detected breakpoints."""
    vals = series.values.astype(float)
    years = series.index.values

    rows = []
    for pen in penalties:
        try:
            algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(vals)
            bkps = algo.predict(pen=float(pen))
            # Convert indices to years (PELT returns 1-based end indices)
            break_years = [int(years[min(b, len(years)-1)]) for b in bkps if b < len(years)]
            for by in break_years:
                rows.append({"penalty": float(pen), "break_year": by})
        except Exception:
            pass

    return pd.DataFrame(rows)


def breakpoint_support(grid_df: pd.DataFrame, tolerance: int = 2) -> pd.DataFrame:
    """Compute support probability for each breakpoint cluster."""
    if grid_df.empty:
        return pd.DataFrame()

    n_penalties = grid_df["penalty"].nunique()
    all_years = grid_df["break_year"].values

    # Cluster breakpoints within tolerance
    unique_years = sorted(set(all_years))
    clusters = []
    visited = set()
    for y in unique_years:
        if y in visited:
            continue
        cluster = [yy for yy in unique_years if abs(yy - y) <= tolerance]
        for c in cluster:
            visited.add(c)
        center = int(np.median(cluster))
        count = sum(1 for by in all_years if any(abs(by - c) <= tolerance for c in cluster))
        clusters.append({
            "break_cluster_center": center,
            "support_count": count,
            "support_pct": 100.0 * count / n_penalties,
            "cluster_years": cluster,
        })

    return pd.DataFrame(clusters).sort_values("support_pct", ascending=False)


def compute_effect_sizes(series: pd.Series, break_year: int,
                          n_bootstrap: int = 1000) -> dict:
    """Compute Δmean and Δslope pre/post breakpoint with bootstrap CIs."""
    pre = series[series.index < break_year]
    post = series[series.index >= break_year]

    if len(pre) < 3 or len(post) < 3:
        return {}

    delta_mean = float(post.mean() - pre.mean())

    # Theil-Sen slopes
    pre_slope = theilslopes(pre.values, pre.index.values)[0]
    post_slope = theilslopes(post.values, post.index.values)[0]
    delta_slope = float(post_slope - pre_slope)

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    boot_dmean = []
    boot_dslope = []

    for _ in range(n_bootstrap):
        pre_idx = rng.choice(len(pre), size=len(pre), replace=True)
        post_idx = rng.choice(len(post), size=len(post), replace=True)
        pre_b = pre.iloc[pre_idx]
        post_b = post.iloc[post_idx]
        boot_dmean.append(post_b.mean() - pre_b.mean())
        if len(pre_b) >= 3 and len(post_b) >= 3:
            s1 = theilslopes(pre_b.values, pre.index.values[pre_idx])[0]
            s2 = theilslopes(post_b.values, post.index.values[post_idx])[0]
            boot_dslope.append(s2 - s1)

    return {
        "delta_mean": delta_mean,
        "delta_mean_ci_lo": float(np.percentile(boot_dmean, 2.5)),
        "delta_mean_ci_hi": float(np.percentile(boot_dmean, 97.5)),
        "pre_slope": float(pre_slope),
        "post_slope": float(post_slope),
        "delta_slope": delta_slope,
        "delta_slope_ci_lo": float(np.percentile(boot_dslope, 2.5)) if boot_dslope else np.nan,
        "delta_slope_ci_hi": float(np.percentile(boot_dslope, 97.5)) if boot_dslope else np.nan,
        "n_pre": len(pre),
        "n_post": len(post),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    wq = pd.read_csv(WQ_PATH)

    stability_rows = []
    effect_rows = []

    for station in MRC_STATIONS:
        for param_col, param_label in PARAMS.items():
            series = _annual_mean_series(wq, station, param_col)
            if len(series) < 10:
                continue

            # Penalty grid
            grid = detect_breakpoints_grid(series, PENALTY_GRID)
            if grid.empty:
                continue

            support = breakpoint_support(grid)
            for _, row in support.iterrows():
                stability_rows.append({
                    "station": station,
                    "parameter": param_label,
                    "break_cluster_center": row["break_cluster_center"],
                    "support_pct": row["support_pct"],
                    "support_count": row["support_count"],
                })

            # Effect sizes for the highest-support breakpoint
            if not support.empty:
                best_year = int(support.iloc[0]["break_cluster_center"])
                effects = compute_effect_sizes(series, best_year)
                if effects:
                    effects.update({
                        "station": station,
                        "parameter": param_label,
                        "break_year": best_year,
                    })
                    effect_rows.append(effects)

    return pd.DataFrame(stability_rows), pd.DataFrame(effect_rows)


def main():
    stability_df, effects_df = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p1 = OUT_DIR / "changepoint_stability_grid_latest.csv"
    stability_df.to_csv(p1, index=False)
    print(f"[Phase 5] Wrote {p1}  ({stability_df.shape[0]} rows)")

    p2 = OUT_DIR / "changepoint_effect_sizes_latest.csv"
    effects_df.to_csv(p2, index=False)
    print(f"[Phase 5] Wrote {p2}  ({effects_df.shape[0]} rows)")

    # Summary
    if not stability_df.empty:
        print("\n=== Breakpoint Stability (top per station×param) ===")
        top = stability_df.sort_values("support_pct", ascending=False).drop_duplicates(
            subset=["station", "parameter"])
        print(top.to_string())

    if not effects_df.empty:
        print("\n=== Effect Sizes ===")
        print(effects_df.to_string())


if __name__ == "__main__":
    main()
