#!/usr/bin/env python3
"""
Bootstrap 95 % confidence intervals for key manuscript effect sizes.
=====================================================================

Computes bias-corrected and accelerated (BCa) bootstrap CIs for:

1. **CI ratio** (chemostatic index = CV_C / CV_Q) at Tan Chau
   – Phase 1 (≤ 2004) and Phase 2 (> 2004)
   – Phase difference (ΔCI)
   – Block-bootstrapped by year to preserve intra-annual autocorrelation.

2. **Mass-balance NCF %** (non-conservative fraction)
   – COD, NO₃⁻, TN along the Tien River (Tan Chau → My Thuan)
   – Phase 1 mean, Phase 2 mean, Δ
   – Bootstrapped by year.

3. **CV_C decomposition** — CIs on the CV_C and CV_Q changes that
   underlie the CI shift (addresses reviewer question: is the CI drop
   driven by concentration or discharge variance?).

Output: ``metrics/Bootstrap_Effect_CIs.csv``

Usage
-----
    conda run -n pyfes_2026 python nw_bootstrap_effect_cis.py
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _nw_common import (
    load_wq, load_hydro_monthly, cq_metrics,
    BREAK_YEAR, MB_CSV, METRICS_OUT, require_file,
)

N_BOOT = 2000
RNG = np.random.default_rng(seed=42)
CI_LEVEL = 0.95


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ═══════════════════════════════════════════════════════════════════════

def _bca_ci(
    theta_hat: float,
    boot_dist: np.ndarray,
    jackknife_vals: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval.

    Falls back to percentile method if jackknife acceleration is
    degenerate (all identical values).
    """
    z0 = sp_stats.norm.ppf(np.mean(boot_dist < theta_hat))
    # acceleration factor
    jk_mean = np.mean(jackknife_vals)
    num = np.sum((jk_mean - jackknife_vals) ** 3)
    den = 6.0 * (np.sum((jk_mean - jackknife_vals) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo = sp_stats.norm.ppf(alpha / 2)
    z_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def _adj(z: float) -> float:
        denom = 1 - a * (z0 + z)
        if denom == 0:
            return z
        return sp_stats.norm.cdf(z0 + (z0 + z) / denom)

    p_lo = _adj(z_lo)
    p_hi = _adj(z_hi)
    if not (0 < p_lo < 1 and 0 < p_hi < 1):
        # fallback to simple percentile
        p_lo = alpha / 2
        p_hi = 1 - alpha / 2
    return float(np.percentile(boot_dist, 100 * p_lo)), float(np.percentile(boot_dist, 100 * p_hi))


def _block_bootstrap_ci_stat(
    years: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    stat_fn,  # callable(c_arr, q_arr) -> float
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Block bootstrap (resampling years) for a statistic involving C and Q."""
    unique_years = np.unique(years)
    n_years = len(unique_years)
    theta_hat = stat_fn(c, q)

    # --- bootstrap distribution ---
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled_years = RNG.choice(unique_years, size=n_years, replace=True)
        idx = np.concatenate([np.where(years == yr)[0] for yr in sampled_years])
        boot[b] = stat_fn(c[idx], q[idx])

    # --- jackknife for BCa ---
    jk = np.empty(n_years)
    for i, drop_yr in enumerate(unique_years):
        keep = years != drop_yr
        jk[i] = stat_fn(c[keep], q[keep])

    lo, hi = _bca_ci(theta_hat, boot, jk, alpha=1 - CI_LEVEL)
    return theta_hat, lo, hi


def _ci_ratio(c: np.ndarray, q: np.ndarray) -> float:
    """CV_C / CV_Q (chemostatic index)."""
    mask = np.isfinite(c) & np.isfinite(q) & (c > 0) & (q > 0)
    c, q = c[mask], q[mask]
    if len(c) < 10:
        return np.nan
    cv_c = np.std(c) / np.mean(c)
    cv_q = np.std(q) / np.mean(q)
    return cv_c / cv_q if cv_q > 0 else np.nan


def _cv_c(c: np.ndarray, q: np.ndarray) -> float:
    mask = np.isfinite(c) & (c > 0)
    c = c[mask]
    return np.std(c) / np.mean(c) if len(c) >= 10 else np.nan


def _cv_q(c: np.ndarray, q: np.ndarray) -> float:
    mask = np.isfinite(q) & (q > 0)
    q = q[mask]
    return np.std(q) / np.mean(q) if len(q) >= 10 else np.nan


# ═══════════════════════════════════════════════════════════════════════
# 1. CI ratio bootstrap
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci_ratio() -> list[dict]:
    """Block-bootstrap CIs for the chemostatic index at Tan Chau (COD)."""
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    q_monthly = load_hydro_monthly()

    merged = mrc[mrc["stationName"] == "Tan Chau"].merge(
        q_monthly[["ym", "TanChau"]], on="ym", how="inner"
    )

    params = [
        ("COD (mg/L)", "COD"),
        ("DO (mg/L)", "DO"),
        ("NH4-N (mg/L)", "NH4-N"),
        ("NO3-N (mg/L)", "NO3-N"),
    ]

    rows = []
    for pcol, plabel in params:
        if pcol not in merged.columns:
            continue
        c_all = merged[pcol].to_numpy(dtype=float)
        q_all = merged["TanChau"].to_numpy(dtype=float)
        yr_all = merged["year"].to_numpy(dtype=float)

        for stat_name, stat_fn in [("CI", _ci_ratio), ("CV_C", _cv_c), ("CV_Q", _cv_q)]:
            for phase_label, mask in [
                ("Phase1", yr_all <= BREAK_YEAR),
                ("Phase2", yr_all > BREAK_YEAR),
            ]:
                c_p = c_all[mask]
                q_p = q_all[mask]
                yr_p = yr_all[mask]
                est, lo, hi = _block_bootstrap_ci_stat(yr_p, c_p, q_p, stat_fn)
                rows.append({
                    "Analysis": "CI_ratio",
                    "Parameter": plabel,
                    "Station": "Tan Chau",
                    "Statistic": stat_name,
                    "Phase": phase_label,
                    "Estimate": round(est, 4),
                    "CI_lo": round(lo, 4),
                    "CI_hi": round(hi, 4),
                    "N_boot": N_BOOT,
                })

    print(f"  CI ratio bootstrap: {len(rows)} rows")
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 2. Mass-balance NCF% bootstrap
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_mass_balance() -> list[dict]:
    """Bootstrap CIs for mass-balance non-conservative fraction (%) by phase."""
    require_file(MB_CSV, label="Mass balance CSV")
    mb = pd.read_csv(MB_CSV)

    target_params = ["COD", "NO3-N", "TN", "NH4-N", "TP", "TSS"]
    rows = []

    for plabel in target_params:
        sub = mb[mb["parameter"] == plabel].dropna(subset=["nonconservative_pct"])
        if len(sub) < 5:
            continue
        annual = (
            sub.groupby("year")["nonconservative_pct"]
            .mean()
            .dropna()
        )
        if len(annual) < 5:
            continue

        xa = annual.index.to_numpy(dtype=float)
        ya = annual.to_numpy(dtype=float)

        for phase_label, mask in [
            ("Phase1", xa <= BREAK_YEAR),
            ("Phase2", xa > BREAK_YEAR),
        ]:
            y_p = ya[mask]
            x_p = xa[mask]
            if len(y_p) < 3:
                continue
            est = float(np.mean(y_p))

            # Bootstrap by year
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = RNG.choice(len(y_p), size=len(y_p), replace=True)
                boot[b] = np.mean(y_p[idx])

            # Jackknife for BCa
            jk = np.empty(len(y_p))
            for i in range(len(y_p)):
                jk[i] = np.mean(np.delete(y_p, i))

            lo, hi = _bca_ci(est, boot, jk, alpha=1 - CI_LEVEL)

            rows.append({
                "Analysis": "Mass_balance",
                "Parameter": plabel,
                "Station": "TanChau→MyThuan",
                "Statistic": "NCF_pct",
                "Phase": phase_label,
                "Estimate": round(est, 2),
                "CI_lo": round(lo, 2),
                "CI_hi": round(hi, 2),
                "N_boot": N_BOOT,
            })

        # Phase difference bootstrap
        p1 = ya[xa <= BREAK_YEAR]
        p2 = ya[xa > BREAK_YEAR]
        if len(p1) >= 3 and len(p2) >= 3:
            delta_hat = float(np.mean(p2) - np.mean(p1))
            boot_delta = np.empty(N_BOOT)
            for b in range(N_BOOT):
                i1 = RNG.choice(len(p1), size=len(p1), replace=True)
                i2 = RNG.choice(len(p2), size=len(p2), replace=True)
                boot_delta[b] = np.mean(p2[i2]) - np.mean(p1[i1])

            # Jackknife on pooled data (leave-one-out from each phase)
            jk_delta = []
            for i in range(len(p1)):
                p1_jk = np.delete(p1, i)
                jk_delta.append(float(np.mean(p2) - np.mean(p1_jk)))
            for i in range(len(p2)):
                p2_jk = np.delete(p2, i)
                jk_delta.append(float(np.mean(p2_jk) - np.mean(p1)))
            jk_delta = np.array(jk_delta)

            lo, hi = _bca_ci(delta_hat, boot_delta, jk_delta, alpha=1 - CI_LEVEL)

            rows.append({
                "Analysis": "Mass_balance",
                "Parameter": plabel,
                "Station": "TanChau→MyThuan",
                "Statistic": "Delta_NCF_pct",
                "Phase": "P2-P1",
                "Estimate": round(delta_hat, 2),
                "CI_lo": round(lo, 2),
                "CI_hi": round(hi, 2),
                "N_boot": N_BOOT,
            })

    print(f"  Mass-balance bootstrap: {len(rows)} rows")
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Computing bootstrap 95% CIs for manuscript effect sizes...")
    print(f"  N_BOOT = {N_BOOT}, CI_LEVEL = {CI_LEVEL}")

    all_rows = []

    print("\n[1/2] CI ratio bootstrap (block by year)...")
    all_rows.extend(bootstrap_ci_ratio())

    print("\n[2/2] Mass-balance NCF% bootstrap...")
    all_rows.extend(bootstrap_mass_balance())

    df = pd.DataFrame(all_rows)
    out = METRICS_OUT / "Bootstrap_Effect_CIs.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved {len(df)} rows to {out}")

    # Summary: key manuscript values
    print("\n" + "=" * 70)
    print("KEY MANUSCRIPT VALUES WITH 95% CIs:")
    print("=" * 70)

    ci_cod = df[(df["Analysis"] == "CI_ratio") & (df["Parameter"] == "COD") & (df["Statistic"] == "CI")]
    for _, r in ci_cod.iterrows():
        print(f"  CI COD {r['Phase']}: {r['Estimate']:.2f} (95% CI: {r['CI_lo']:.2f}–{r['CI_hi']:.2f})")

    cv_c = df[(df["Analysis"] == "CI_ratio") & (df["Parameter"] == "COD") & (df["Statistic"] == "CV_C")]
    for _, r in cv_c.iterrows():
        print(f"  CV_C COD {r['Phase']}: {r['Estimate']:.4f} (95% CI: {r['CI_lo']:.4f}–{r['CI_hi']:.4f})")

    mb_key = df[(df["Analysis"] == "Mass_balance") & (df["Parameter"].isin(["COD", "NO3-N", "TN"]))]
    for _, r in mb_key.iterrows():
        print(f"  {r['Parameter']} {r['Statistic']} {r['Phase']}: "
              f"{r['Estimate']:.1f}% (95% CI: {r['CI_lo']:.1f}–{r['CI_hi']:.1f}%)")


if __name__ == "__main__":
    main()
