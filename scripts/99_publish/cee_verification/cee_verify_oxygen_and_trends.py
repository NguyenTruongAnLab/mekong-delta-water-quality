#!/usr/bin/env python3
"""
CEE Phase 1 Verification Script 1:
Oxygen Thermodynamics (AOU vs. O2_sat), Sub-lethal DO < 5 mg/L Exceedances,
Dry-Season Heterotrophic Respiration vs. Algal-pH Test, and Harmonized
Seasonal Mann-Kendall + Livezey-Chen (1983) Field Significance.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, theilslopes, spearmanr, fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
OUT_DIR = Path(__file__).resolve().parent / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]
BREAK_YEAR = 2004


def o2_sat_garcia_gordon(temp_c: np.ndarray, sal: np.ndarray | None = None) -> np.ndarray:
    """
    Dissolved oxygen saturation (mg/L) from Garcia & Gordon (1992, Limnol. Oceanogr. 37, 1307-1312)
    Benson & Krause (1984) fit. Converts umol/kg (or mL/L) to mg/L for freshwater/oligohaline waters.
    """
    t = np.asarray(temp_c, dtype=float)
    s = np.zeros_like(t) if sal is None else np.nan_to_num(np.asarray(sal, dtype=float), nan=0.0)
    ts = np.log((298.15 - t) / (273.15 + t))

    # Garcia & Gordon (1992) Table 1 constants for mL(STP)/L (Benson & Krause 1984 data)
    a0 = 2.00907
    a1 = 3.22014
    a2 = 4.05010
    a3 = 4.94457
    a4 = -2.56847e-1
    a5 = 3.88767
    b0 = -6.24523e-3
    b1 = -7.37614e-3
    b2 = -1.03410e-2
    b3 = -8.17083e-3
    c0 = -4.88682e-7

    ln_c_ml = (
        a0 + a1 * ts + a2 * (ts ** 2) + a3 * (ts ** 3) + a4 * (ts ** 4) + a5 * (ts ** 5)
        + s * (b0 + b1 * ts + b2 * (ts ** 2) + b3 * (ts ** 3))
        + c0 * (s ** 2)
    )
    # 1 mL O2 (STP) = 1.42905 mg O2
    return np.exp(ln_c_ml) * 1.42905


def seasonal_mann_kendall(years: np.ndarray, months: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """
    Hirsch, Slack & Smith (1982) Seasonal Mann-Kendall test & Seasonal Sen slope estimator.
    Aggregates duplicate year-month cells to monthly means first so each (year, month) is unique.
    """
    df = pd.DataFrame({"year": years, "month": months, "val": values}).dropna()
    df = df.groupby(["year", "month"], as_index=False)["val"].mean()
    s_total = 0.0
    var_total = 0.0
    slopes: list[float] = []
    n_obs = len(df)

    for m in range(1, 13):
        sub = df[df["month"] == m].sort_values("year")
        n_m = len(sub)
        if n_m < 2:
            continue
        y = sub["val"].to_numpy(dtype=float)
        t = sub["year"].to_numpy(dtype=float)
        for i in range(n_m - 1):
            dy = y[i + 1 :] - y[i]
            dt = t[i + 1 :] - t[i]
            s_total += float(np.sum(np.sign(dy)))
            valid_dt = dt != 0
            slopes.extend((dy[valid_dt] / dt[valid_dt]).tolist())

        # Tie correction within month m
        _, counts = np.unique(y, return_counts=True)
        tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
        var_m = (n_m * (n_m - 1) * (2 * n_m + 5) - tie_term) / 18.0
        var_total += var_m

    if var_total <= 0 or not slopes:
        return {"S": np.nan, "Z": np.nan, "p_two_sided": np.nan, "sen_slope_yr": np.nan, "n_ym": int(n_obs)}

    if s_total > 0:
        z = (s_total - 1.0) / np.sqrt(var_total)
    elif s_total < 0:
        z = (s_total + 1.0) / np.sqrt(var_total)
    else:
        z = 0.0
    p_val = float(2.0 * stats.norm.sf(abs(z)))
    sen_slope = float(np.median(slopes))
    return {
        "S": float(s_total),
        "Z": float(z),
        "p_two_sided": p_val,
        "sen_slope_yr": sen_slope,
        "n_ym": int(n_obs),
    }


def _fast_smk_matrix(vals_ym_stn: np.ndarray, years_ym: np.ndarray, months_ym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Seasonal Mann-Kendall Z and two-sided p across K stations simultaneously.
    vals_ym_stn: shape (T, K) where each row is a unique (year, month).
    """
    K = vals_ym_stn.shape[1]
    s_tot = np.zeros(K, dtype=float)
    var_tot = np.zeros(K, dtype=float)
    for m in range(1, 13):
        idx_m = np.where(months_ym == m)[0]
        if len(idx_m) < 2:
            continue
        order = np.argsort(years_ym[idx_m])
        ym_sub = vals_ym_stn[idx_m[order], :]  # (n_m, K)
        for k in range(K):
            yk = ym_sub[:, k]
            yk = yk[np.isfinite(yk)]
            n_m = len(yk)
            if n_m < 2:
                continue
            diff = yk[None, :] - yk[:, None]  #diff[i, j] = yk[j] - yk[i]
            iu = np.triu_indices(n_m, k=1)
            s_tot[k] += float(np.sum(np.sign(diff[iu])))
            _, counts = np.unique(yk, return_counts=True)
            tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
            var_tot[k] += (n_m * (n_m - 1) * (2 * n_m + 5) - tie_term) / 18.0

    z = np.zeros(K, dtype=float)
    for k in range(K):
        if var_tot[k] <= 0:
            z[k] = np.nan
        elif s_tot[k] > 0:
            z[k] = (s_tot[k] - 1.0) / np.sqrt(var_tot[k])
        elif s_tot[k] < 0:
            z[k] = (s_tot[k] + 1.0) / np.sqrt(var_tot[k])
    p = 2.0 * stats.norm.sf(np.abs(z))
    return z, p


def livezey_chen_field_significance(
    mrc_df: pd.DataFrame,
    param_col: str,
    phase_mask: pd.Series,
    alpha_local: float = 0.05,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Livezey & Chen (1983) field significance permutation test across the 4 MRC stations.
    Permutes the shared year order simultaneously across all stations within each calendar month
    so spatial cross-correlation among stations is 100% preserved.
    """
    rng = np.random.default_rng(seed)
    sub = mrc_df.loc[phase_mask, ["stationName", "year", "month", param_col]].dropna().copy()
    sub = sub[sub["stationName"].isin(MRC_STATIONS)]
    piv = sub.pivot_table(index=["year", "month"], columns="stationName", values=param_col, aggfunc="mean")
    stns = [s for s in MRC_STATIONS if s in piv.columns]
    if len(stns) < 2 or len(piv) < 24:
        return {"k_sig_obs": 0, "n_stations": len(stns), "p_field_count": np.nan, "p_field_absZ": np.nan}

    years_ym = piv.index.get_level_values("year").to_numpy(dtype=int)
    months_ym = piv.index.get_level_values("month").to_numpy(dtype=int)
    mat = piv[stns].to_numpy(dtype=float)

    z_obs, p_obs = _fast_smk_matrix(mat, years_ym, months_ym)
    obs_sig = int(np.sum(np.isfinite(p_obs) & (p_obs < alpha_local)))
    obs_z_sum = float(np.nansum(np.abs(z_obs)))

    month_indices = [np.where(months_ym == m)[0] for m in range(1, 13) if np.any(months_ym == m)]
    exceed_count = 0
    exceed_z = 0
    years_perm = years_ym.copy()
    for _ in range(n_perm):
        for idx_m in month_indices:
            years_perm[idx_m] = rng.permutation(years_ym[idx_m])
        zp, pp = _fast_smk_matrix(mat, years_perm, months_ym)
        k_perm = int(np.sum(np.isfinite(pp) & (pp < alpha_local)))
        z_perm = float(np.nansum(np.abs(zp)))
        if k_perm >= obs_sig:
            exceed_count += 1
        if z_perm >= obs_z_sum:
            exceed_z += 1

    return {
        "k_sig_obs": int(obs_sig),
        "n_stations": int(len(stns)),
        "p_field_count": float((exceed_count + 1) / (n_perm + 1)),
        "p_field_absZ": float((exceed_z + 1) / (n_perm + 1)),
    }


def main() -> None:
    raw = pd.read_csv(WQ_PATH)
    n_raw_total = len(raw)
    raw["dt"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw["year"] = raw["dt"].dt.year
    raw["month"] = raw["dt"].dt.month

    # Conservation of observations for MRC 1985-2022
    mrc_all = raw[raw["Data source"] == "MRC"].copy()
    n_mrc_raw = len(mrc_all)
    mrc_post2022 = mrc_all[mrc_all["year"] > 2022]
    mrc = mrc_all[(mrc_all["year"] >= 1985) & (mrc_all["year"] <= 2022)].copy()
    n_mrc_valid_window = len(mrc)

    print(f"[Conservation] Total WQ CSV rows = {n_raw_total}")
    print(f"[Conservation] MRC raw rows = {n_mrc_raw} = {len(mrc_post2022)} (year > 2022 excluded) + {n_mrc_valid_window} (1985-2022 analysis window)")
    assert n_mrc_raw == len(mrc_post2022) + n_mrc_valid_window

    # 1. Sub-lethal biological oxygen stress (DO < 5.0 mg/L and DO < 4.0 mg/L) across 1985-2022
    do_valid = mrc.dropna(subset=["DO (mg/L)"]).copy()
    n_do_missing = n_mrc_valid_window - len(do_valid)
    print(f"[Conservation] MRC DO rows: {n_mrc_valid_window} = {n_do_missing} (missing DO) + {len(do_valid)} (valid DO)")

    pre_do = do_valid[do_valid["year"] <= BREAK_YEAR]["DO (mg/L)"].to_numpy(dtype=float)
    post_do = do_valid[do_valid["year"] > BREAK_YEAR]["DO (mg/L)"].to_numpy(dtype=float)

    do_exceed_rows = []
    for scope_label, d_sub in [("All_4_MRC_Stations", do_valid)] + [
        (stn, do_valid[do_valid["stationName"] == stn]) for stn in MRC_STATIONS
    ]:
        p1 = d_sub[d_sub["year"] <= BREAK_YEAR]["DO (mg/L)"].to_numpy(dtype=float)
        p2 = d_sub[d_sub["year"] > BREAK_YEAR]["DO (mg/L)"].to_numpy(dtype=float)
        k1_lt5 = int(np.sum(p1 < 5.0))
        k2_lt5 = int(np.sum(p2 < 5.0))
        k1_lt4 = int(np.sum(p1 < 4.0))
        k2_lt4 = int(np.sum(p2 < 4.0))
        pct1 = 100.0 * k1_lt5 / len(p1) if len(p1) else np.nan
        pct2 = 100.0 * k2_lt5 / len(p2) if len(p2) else np.nan
        table_2x2 = [[k2_lt5, len(p2) - k2_lt5], [k1_lt5, len(p1) - k1_lt5]]
        odds_ratio, p_fisher = fisher_exact(table_2x2, alternative="two-sided")
        mw_p = float(mannwhitneyu(p1, p2, alternative="two-sided").pvalue)
        do_exceed_rows.append({
            "Scope": scope_label,
            "n_Phase1": int(len(p1)),
            "n_Phase2": int(len(p2)),
            "Mean_DO_Phase1": round(float(np.mean(p1)), 3),
            "Mean_DO_Phase2": round(float(np.mean(p2)), 3),
            "Delta_Mean_DO": round(float(np.mean(p2) - np.mean(p1)), 3),
            "MannWhitney_p": p_fisher if False else mw_p,
            "Count_DO_lt5_Phase1": k1_lt5,
            "Pct_DO_lt5_Phase1": round(pct1, 3),
            "Count_DO_lt5_Phase2": k2_lt5,
            "Pct_DO_lt5_Phase2": round(pct2, 3),
            "Fold_Ratio_lt5": round(pct2 / pct1, 2) if pct1 > 0 else np.nan,
            "Fisher_OR_lt5": round(float(odds_ratio), 2) if np.isfinite(odds_ratio) else np.nan,
            "Fisher_p_lt5": float(p_fisher),
            "Count_DO_lt4_Phase1": k1_lt4,
            "Count_DO_lt4_Phase2": k2_lt4,
        })
    df_do_exceed = pd.DataFrame(do_exceed_rows)
    df_do_exceed.to_csv(OUT_DIR / "cee_do_exceedance_regime_shift.csv", index=False)
    print("\n--- 1. Sub-lethal DO < 5.0 mg/L Exceedance Summary ---")
    print(df_do_exceed.to_string(index=False))

    # 2. Post-2004 (2005-2022) Thermodynamic vs. Biological (AOU) Oxygen Partitioning
    post_mrc = mrc[mrc["year"] >= 2005].copy()
    post_do_temp = post_mrc.dropna(subset=["DO (mg/L)", "Temp (C)"]).copy()
    print(f"\n[Conservation] Post-2004 MRC rows = {len(post_mrc)}; complete (DO + Temp) = {len(post_do_temp)} ({100*len(post_do_temp)/len(post_mrc):.2f}%)")

    post_do_temp["O2_sat"] = o2_sat_garcia_gordon(
        post_do_temp["Temp (C)"].to_numpy(dtype=float),
        post_do_temp["Salinity"].to_numpy(dtype=float) if "Salinity" in post_do_temp.columns else None,
    )
    post_do_temp["AOU"] = post_do_temp["O2_sat"] - post_do_temp["DO (mg/L)"]
    post_do_temp["DO_sat_pct"] = 100.0 * post_do_temp["DO (mg/L)"] / post_do_temp["O2_sat"]

    aou_rows = []
    rng = np.random.default_rng(2026)
    for stn in MRC_STATIONS:
        s_df = post_do_temp[post_do_temp["stationName"] == stn].copy()
        ann = s_df.groupby("year").agg(
            DO=("DO (mg/L)", "mean"),
            Temp=("Temp (C)", "mean"),
            O2_sat=("O2_sat", "mean"),
            AOU=("AOU", "mean"),
            DO_sat_pct=("DO_sat_pct", "mean"),
            n_obs=("DO (mg/L)", "count"),
        ).reset_index()
        yrs = ann["year"].to_numpy(dtype=float)

        # OLS & Theil-Sen slopes on annual means (DO = O2_sat - AOU => slope_DO = slope_O2sat - slope_AOU exactly in OLS!)
        sl_do_ols, _, _, p_do_ols, se_do_ols = stats.linregress(yrs, ann["DO"].to_numpy())
        sl_sat_ols, _, _, p_sat_ols, se_sat_ols = stats.linregress(yrs, ann["O2_sat"].to_numpy())
        sl_aou_ols, _, _, p_aou_ols, se_aou_ols = stats.linregress(yrs, ann["AOU"].to_numpy())
        sl_t_ols, _, _, p_t_ols, _ = stats.linregress(yrs, ann["Temp"].to_numpy())

        # Seasonal Mann-Kendall on monthly observations
        smk_do = seasonal_mann_kendall(s_df["year"].to_numpy(), s_df["month"].to_numpy(), s_df["DO (mg/L)"].to_numpy())
        smk_sat = seasonal_mann_kendall(s_df["year"].to_numpy(), s_df["month"].to_numpy(), s_df["O2_sat"].to_numpy())
        smk_aou = seasonal_mann_kendall(s_df["year"].to_numpy(), s_df["month"].to_numpy(), s_df["AOU"].to_numpy())

        # Exact additive share of deoxygenation: (-sl_do_ols) = (-sl_sat_ols) + (+sl_aou_ols)
        total_deox_rate = -sl_do_ols
        resp_share_pct = 100.0 * (sl_aou_ols / total_deox_rate) if total_deox_rate > 0 else np.nan
        therm_share_pct = 100.0 * (-sl_sat_ols / total_deox_rate) if total_deox_rate > 0 else np.nan

        # Bootstrap 95% CI on annual slopes (2,000 resamples)
        b_do, b_sat, b_aou = [], [], []
        for _ in range(2000):
            idx = rng.integers(0, len(ann), size=len(ann))
            if len(np.unique(yrs[idx])) < 5:
                continue
            b_do.append(stats.linregress(yrs[idx], ann["DO"].to_numpy()[idx])[0])
            b_sat.append(stats.linregress(yrs[idx], ann["O2_sat"].to_numpy()[idx])[0])
            b_aou.append(stats.linregress(yrs[idx], ann["AOU"].to_numpy()[idx])[0])

        aou_rows.append({
            "Station": stn,
            "n_obs": int(len(s_df)),
            "n_years": int(len(ann)),
            "Mean_Temp_C": round(float(ann["Temp"].mean()), 2),
            "Mean_DO_mgL": round(float(ann["DO"].mean()), 3),
            "Mean_O2sat_mgL": round(float(ann["O2_sat"].mean()), 3),
            "Mean_AOU_mgL": round(float(ann["AOU"].mean()), 3),
            "Mean_DOsat_pct": round(float(ann["DO_sat_pct"].mean()), 2),
            "Slope_Temp_C_yr": round(float(sl_t_ols), 4),
            "p_Temp_OLS": round(float(p_t_ols), 5),
            "Slope_DO_mgL_yr": round(float(sl_do_ols), 4),
            "CI95_DO_lo": round(float(np.percentile(b_do, 2.5)), 4),
            "CI95_DO_hi": round(float(np.percentile(b_do, 97.5)), 4),
            "p_DO_OLS": round(float(p_do_ols), 5),
            "SMK_Sen_DO_yr": round(float(smk_do["sen_slope_yr"]), 4),
            "SMK_p_DO": round(float(smk_do["p_two_sided"]), 6),
            "Slope_O2sat_mgL_yr": round(float(sl_sat_ols), 4),
            "CI95_O2sat_lo": round(float(np.percentile(b_sat, 2.5)), 4),
            "CI95_O2sat_hi": round(float(np.percentile(b_sat, 97.5)), 4),
            "p_O2sat_OLS": round(float(p_sat_ols), 5),
            "SMK_Sen_O2sat_yr": round(float(smk_sat["sen_slope_yr"]), 4),
            "SMK_p_O2sat": round(float(smk_sat["p_two_sided"]), 6),
            "Slope_AOU_mgL_yr": round(float(sl_aou_ols), 4),
            "CI95_AOU_lo": round(float(np.percentile(b_aou, 2.5)), 4),
            "CI95_AOU_hi": round(float(np.percentile(b_aou, 97.5)), 4),
            "p_AOU_OLS": round(float(p_aou_ols), 5),
            "SMK_Sen_AOU_yr": round(float(smk_aou["sen_slope_yr"]), 4),
            "SMK_p_AOU": round(float(smk_aou["p_two_sided"]), 6),
            "Respiration_AOU_Share_pct": round(float(resp_share_pct), 1),
            "Thermal_Solubility_Share_pct": round(float(therm_share_pct), 1),
        })
    df_aou = pd.DataFrame(aou_rows)
    df_aou.to_csv(OUT_DIR / "cee_aou_thermodynamic_partitioning.csv", index=False)
    print("\n--- 2. Post-2004 (2005-2022) AOU vs. O2_sat Slope Partitioning ---")
    print(df_aou[[
        "Station", "n_obs", "Mean_AOU_mgL", "Slope_DO_mgL_yr", "Slope_O2sat_mgL_yr",
        "Slope_AOU_mgL_yr", "SMK_p_AOU", "Respiration_AOU_Share_pct", "Thermal_Solubility_Share_pct"
    ]].to_string(index=False))

    # 3. Dry-season (Jan-Apr and Nov-Apr) TSS-pH-COD-AOU Spearman tests
    corr_rows = []
    dry_mrc = mrc[mrc["month"].isin([1, 2, 3, 4])].copy()
    dry_mrc_post = post_do_temp[post_do_temp["month"].isin([1, 2, 3, 4])].copy()
    pairs = [
        ("1985-2022_Dry_JanApr", dry_mrc, "TSS (mg/L)", "pH"),
        ("1985-2022_Dry_JanApr", dry_mrc, "pH", "COD (mg/L)"),
        ("1985-2022_Dry_JanApr", dry_mrc, "TSS (mg/L)", "COD (mg/L)"),
        ("2005-2022_Dry_JanApr", dry_mrc_post, "COD (mg/L)", "AOU"),
        ("2005-2022_Dry_JanApr", dry_mrc_post, "pH", "AOU"),
    ]
    for window_lbl, df_src, col_x, col_y in pairs:
        for stn in ["Pooled_MRC"] + MRC_STATIONS:
            sub = df_src if stn == "Pooled_MRC" else df_src[df_src["stationName"] == stn]
            valid = sub[[col_x, col_y]].dropna()
            if len(valid) >= 10:
                rho, p_val = spearmanr(valid[col_x].to_numpy(), valid[col_y].to_numpy())
                corr_rows.append({
                    "Window": window_lbl,
                    "Station": stn,
                    "Var_X": col_x,
                    "Var_Y": col_y,
                    "n": int(len(valid)),
                    "Spearman_rho": round(float(rho), 4),
                    "p_two_sided": float(p_val),
                })
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(OUT_DIR / "cee_dry_season_respiration_correlations.csv", index=False)
    print("\n--- 3. Dry-Season Heterotrophic Respiration vs. Algal-pH Test ---")
    print(df_corr[df_corr["Station"].isin(["Pooled_MRC", "Tan Chau", "My Thuan"])].to_string(index=False))

    # 4. Harmonized Seasonal Mann-Kendall + Annual Theil-Sen + Livezey-Chen Field Significance
    trend_params = ["COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)", "TN (mg/L)", "TSS (mg/L)"]
    trend_rows = []
    field_rows = []

    for pcol in trend_params:
        if pcol not in mrc.columns:
            continue
        for phase_name, pmask in [
            ("Phase1_1985_2004", mrc["year"] <= BREAK_YEAR),
            ("Phase2_2005_2022", mrc["year"] > BREAK_YEAR),
            ("Full_1985_2022", mrc["year"] >= 1985),
        ]:
            fs = livezey_chen_field_significance(mrc, pcol, pmask, alpha_local=0.05, n_perm=1000, seed=42)
            field_rows.append({
                "Parameter": pcol,
                "Phase": phase_name,
                **fs,
            })

            for stn in MRC_STATIONS:
                sub = mrc.loc[pmask & (mrc["stationName"] == stn), ["year", "month", pcol]].dropna()
                if len(sub) < 12:
                    continue
                smk = seasonal_mann_kendall(sub["year"].to_numpy(), sub["month"].to_numpy(), sub[pcol].to_numpy())
                ann = sub.groupby("year")[pcol].mean().dropna()
                ts = theilslopes(ann.to_numpy(), ann.index.to_numpy(dtype=float))
                _, p_ann_mk = kendalltau(ann.index.to_numpy(dtype=float), ann.to_numpy())
                trend_rows.append({
                    "Parameter": pcol.split(" (")[0],
                    "Station": stn,
                    "Phase": phase_name,
                    "n_ym": smk["n_ym"],
                    "SMK_Sen_Slope_yr": round(float(smk["sen_slope_yr"]), 5),
                    "SMK_Z": round(float(smk["Z"]), 3),
                    "SMK_p": round(float(smk["p_two_sided"]), 6),
                    "n_years": int(len(ann)),
                    "Annual_TheilSen_Slope_yr": round(float(ts[0]), 5),
                    "Annual_CI95_lo": round(float(ts[2]), 5),
                    "Annual_CI95_hi": round(float(ts[3]), 5),
                    "Annual_MK_p": round(float(p_ann_mk), 6),
                })

    df_trends = pd.DataFrame(trend_rows)
    df_field = pd.DataFrame(field_rows)
    df_trends.to_csv(OUT_DIR / "cee_harmonized_smk_annual_trends.csv", index=False)
    df_field.to_csv(OUT_DIR / "cee_livezey_chen_field_significance.csv", index=False)
    print("\n--- 4a. Harmonized Phase 2 Trends (COD, DO, NO3-N) ---")
    print(df_trends[(df_trends["Phase"] == "Phase2_2005_2022") & (df_trends["Parameter"].isin(["COD", "DO", "NO3-N"]))].to_string(index=False))
    print("\n--- 4b. Livezey-Chen (1983) Field Significance ---")
    print(df_field[df_field["Phase"] == "Phase2_2005_2022"].to_string(index=False))


if __name__ == "__main__":
    main()
