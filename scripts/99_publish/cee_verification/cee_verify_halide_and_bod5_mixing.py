#!/usr/bin/env python3
"""
CEE Phase 1 Verification Script 2:
Estuarine Conservative-Mixing Sink-to-Source Validation with Double Halide Safeguards:
  1. Dichromate COD_Cr restricted strictly to HgSO4-masked halide-safe Salinity <= 3.6
     (Cl- <= 2,000 mg/L, APHA 5220 / TCVN 6491) and freshwater Salinity <= 1.0.
  2. Biological Oxygen Demand (BOD5, n = 718, 100% immune to chloride oxidation!) across
     full salinity, Salinity <= 3.6, and Salinity <= 1.0, plus NH4+-N, NO3--N, DO, and TSS.
Exports exact observation conservation counts, bootstrap 95% CIs, and two-sided Mann-Whitney p.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
OUT_DIR = Path(__file__).resolve().parent / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S_SEA = 30.0
EARLY = (2005, 2012)
LATE = (2013, 2022)


def robust_endmembers(df: pd.DataFrame, ycol: str, min_n: int = 25) -> tuple[float, float] | None:
    """Identical end-member median protocol to ED Figure 2 (Salinity <= 0.5 fresh, >= 20.0 marine)."""
    d = df[["Salinity", ycol]].dropna().copy()
    if d.empty:
        return None
    d["Salinity"] = d["Salinity"].astype(float)
    d[ycol] = d[ycol].astype(float)

    fresh = d.loc[d["Salinity"] <= 0.5, ycol]
    sea = d.loc[d["Salinity"] >= 20.0, ycol]
    if len(fresh) < min_n:
        q10 = float(np.nanpercentile(d["Salinity"].to_numpy(), 10))
        fresh = d.loc[d["Salinity"] <= q10, ycol]
    if len(sea) < min_n:
        q90 = float(np.nanpercentile(d["Salinity"].to_numpy(), 90))
        sea = d.loc[d["Salinity"] >= q90, ycol]
    if len(fresh) < 8 or len(sea) < 8:
        return None
    return float(np.nanmedian(fresh)), float(np.nanmedian(sea))


def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(arr) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    raw = pd.read_csv(WQ_PATH)
    raw["dt"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw["year"] = raw["dt"].dt.year

    bt_all = raw[raw["Data source"] == "Ben Tre"].copy()
    bt = bt_all[(bt_all["year"] >= 2005) & (bt_all["year"] <= 2022)].copy()
    n_excl = len(bt_all) - len(bt)
    print(f"[Conservation] Ben Tre raw rows = {len(bt_all)} = {n_excl} (outside 2005-2022) + {len(bt)} (2005-2022 window, {bt['stationName'].nunique()} stations)")

    params = [
        ("COD (mg/L)", "COD_Cr"),
        ("BOD5 (mg/L)", "BOD5"),
        ("NH4-N (mg/L)", "NH4-N"),
        ("NO3-N (mg/L)", "NO3-N"),
        ("DO (mg/L)", "DO"),
        ("TSS (mg/L)", "TSS"),
    ]

    sal_regimes = [
        ("All_Salinity", lambda s: s >= 0.0),
        ("HalideSafe_Sal_le_3.6", lambda s: s <= 3.6),
        ("Freshwater_Sal_le_1.0", lambda s: s <= 1.0),
        ("Oligohaline_Sal_le_2.0", lambda s: s <= 2.0),
    ]

    rows = []
    stn_level_rows = []
    for pcol, pshort in params:
        if pcol not in bt.columns:
            continue
        em = robust_endmembers(bt, pcol)
        if em is None:
            continue
        c_fresh, c_sea = em
        valid_full = bt[["stationName", "year", "Salinity", pcol]].dropna().copy()
        n_miss = len(bt) - len(valid_full)
        print(f"[Conservation] {pshort}: {len(bt)} = {n_miss} (missing {pshort} or Salinity) + {len(valid_full)} valid paired obs (C_fresh={c_fresh:.3f}, C_sea={c_sea:.3f})")

        f_sea = np.clip(valid_full["Salinity"].to_numpy(dtype=float) / S_SEA, 0.0, 1.0)
        valid_full["expected_mix"] = c_fresh + (c_sea - c_fresh) * f_sea
        valid_full["excess"] = valid_full[pcol].to_numpy(dtype=float) - valid_full["expected_mix"].to_numpy(dtype=float)

        for reg_name, reg_fn in sal_regimes:
            sub = valid_full[reg_fn(valid_full["Salinity"])].copy()
            e1 = sub[(sub["year"] >= EARLY[0]) & (sub["year"] <= EARLY[1])]["excess"].to_numpy(dtype=float)
            e2 = sub[(sub["year"] >= LATE[0]) & (sub["year"] <= LATE[1])]["excess"].to_numpy(dtype=float)
            if len(e1) < 5 or len(e2) < 5:
                continue
            ci1_lo, ci1_hi = bootstrap_mean_ci(e1, n_boot=5000, seed=101)
            ci2_lo, ci2_hi = bootstrap_mean_ci(e2, n_boot=5000, seed=202)
            mw_u, mw_p = mannwhitneyu(e1, e2, alternative="two-sided")

            # Also compute paired station-level shift across the 19 stations
            s1 = sub[(sub["year"] >= EARLY[0]) & (sub["year"] <= EARLY[1])].groupby("stationName")["excess"].mean()
            s2 = sub[(sub["year"] >= LATE[0]) & (sub["year"] <= LATE[1])].groupby("stationName")["excess"].mean()
            common_stns = s1.index.intersection(s2.index)
            if len(common_stns) >= 6:
                try:
                    _, p_wilcox_stn = wilcoxon(s2.loc[common_stns].to_numpy() - s1.loc[common_stns].to_numpy(), alternative="two-sided")
                except Exception:
                    p_wilcox_stn = np.nan
            else:
                p_wilcox_stn = np.nan

            rows.append({
                "Parameter": pshort,
                "Salinity_Regime": reg_name,
                "Fresh_Endmember": round(c_fresh, 3),
                "Marine_Endmember": round(c_sea, 3),
                "n_2005_2012": int(len(e1)),
                "Mean_Excess_2005_2012": round(float(np.mean(e1)), 3),
                "Median_Excess_2005_2012": round(float(np.median(e1)), 3),
                "CI95_2005_2012_lo": round(ci1_lo, 3),
                "CI95_2005_2012_hi": round(ci1_hi, 3),
                "n_2013_2022": int(len(e2)),
                "Mean_Excess_2013_2022": round(float(np.mean(e2)), 3),
                "Median_Excess_2013_2022": round(float(np.median(e2)), 3),
                "CI95_2013_2022_lo": round(ci2_lo, 3),
                "CI95_2013_2022_hi": round(ci2_hi, 3),
                "Delta_Mean_Excess": round(float(np.mean(e2) - np.mean(e1)), 3),
                "MannWhitney_U": float(mw_u),
                "MannWhitney_p": float(mw_p),
                "n_Paired_Stations": int(len(common_stns)),
                "Paired_Station_Wilcoxon_p": float(p_wilcox_stn) if np.isfinite(p_wilcox_stn) else np.nan,
            })

            if reg_name in ("All_Salinity", "HalideSafe_Sal_le_3.6"):
                for stn in common_stns:
                    stn_level_rows.append({
                        "Parameter": pshort,
                        "Salinity_Regime": reg_name,
                        "stationName": stn,
                        "Excess_2005_2012": round(float(s1.loc[stn]), 4),
                        "Excess_2013_2022": round(float(s2.loc[stn]), 4),
                        "Delta_Excess": round(float(s2.loc[stn] - s1.loc[stn]), 4),
                    })

    df_out = pd.DataFrame(rows)
    df_stn = pd.DataFrame(stn_level_rows)
    df_out.to_csv(OUT_DIR / "cee_estuarine_mixing_halide_bod5.csv", index=False)
    df_stn.to_csv(OUT_DIR / "cee_estuarine_mixing_station_means.csv", index=False)

    print("\n--- Estuarine Conservative Mixing Excess Summary (2005-2012 vs. 2013-2022) ---")
    cols_show = [
        "Parameter", "Salinity_Regime", "n_2005_2012", "Mean_Excess_2005_2012",
        "CI95_2005_2012_lo", "CI95_2005_2012_hi", "n_2013_2022", "Mean_Excess_2013_2022",
        "CI95_2013_2022_lo", "CI95_2013_2022_hi", "MannWhitney_p", "Paired_Station_Wilcoxon_p"
    ]
    print(df_out[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
