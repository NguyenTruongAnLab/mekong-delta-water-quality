#!/usr/bin/env python3
"""
Net in-delta reach mass balance and residence-time sensitivity analysis
=======================================================================

Computes downstream minus upstream loads and discharge-normalized
non-conservative fractions (NCF) along the 120 km Tan Chau to My Thuan
mainstream reach using matched monthly discharge and concentration records
at Mekong River Commission (MRC) stations.

Also executes:
1. Deterministic Monte Carlo sensitivity analysis (10,000 realizations)
   evaluating dynamic Vam Nao canal flow-split perturbations.
2. Advective flushing and 1D longitudinal tidal dispersion sensitivity
   across bathymetric incision scenarios.

Outputs:
- manuscript/tables/Table_NW_Mass_Balance.csv
- scripts/99_publish/nw_verification/metrics/vamnao_mc_sensitivity.csv
- scripts/99_publish/nw_verification/metrics/residence_time_sensitivity.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, theilslopes

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HYDRO_PATH = (
    ROOT
    / "data"
    / "hydrology"
    / "outputs"
    / "final_datasets"
    / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
)
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
TBL_DIR = ROOT / "manuscript" / "tables"
METRICS_DIR = ROOT / "scripts" / "99_publish" / "nw_verification" / "metrics"
TBL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_monthly_discharge() -> pd.DataFrame:
    """Load daily discharge and aggregate to monthly means."""
    if not HYDRO_PATH.exists():
        raise FileNotFoundError(
            f"Raw hydrology dataset required: {HYDRO_PATH.name} "
            "(archived on Zenodo: https://doi.org/10.5281/zenodo.20024194)."
        )
    hydro = pd.read_csv(HYDRO_PATH)
    hydro["date"] = pd.to_datetime(hydro["Timestamp"], format="mixed", dayfirst=False)
    hydro["year"] = hydro["date"].dt.year
    hydro["month"] = hydro["date"].dt.month

    q_monthly = (
        hydro.groupby(["year", "month"])
        .agg(
            TanChau_Q=("TanChau", "mean"),
            MyThuan_Q=("MyThuan", "mean"),
            CanTho_Q=("CanTho", "mean"),
            Total_Inflow_Q=("Total_Inflow", "mean"),
        )
        .reset_index()
    )
    return q_monthly


def load_monthly_wq() -> tuple[pd.DataFrame, list[str]]:
    """Load MRC water quality records and aggregate to monthly means per station."""
    if not WQ_PATH.exists():
        raise FileNotFoundError(
            f"Raw water quality dataset required: {WQ_PATH.name} "
            "(archived on Zenodo: https://doi.org/10.5281/zenodo.20024194)."
        )
    wq = pd.read_csv(WQ_PATH, parse_dates=["datetime"])
    mrc = wq[wq["Data source"] == "MRC"].copy()
    mrc["year"] = mrc["datetime"].dt.year
    mrc["month"] = mrc["datetime"].dt.month

    params = [
        "COD (mg/L)",
        "TN (mg/L)",
        "TP (mg/L)",
        "TSS (mg/L)",
        "NO3-N (mg/L)",
        "NH4-N (mg/L)",
    ]
    available = [p for p in params if p in mrc.columns]

    records = []
    for stn in ["Tan Chau", "My Thuan", "Can Tho", "My Tho"]:
        sub = mrc[mrc["stationName"] == stn]
        monthly = sub.groupby(["year", "month"])[available].mean().reset_index()
        monthly["stationName"] = stn
        records.append(monthly)

    return pd.concat(records, ignore_index=True), available


def compute_mass_balance() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute monthly loads and discharge-normalized reach mass balance."""
    q_monthly = load_monthly_discharge()
    wq_monthly, params = load_monthly_wq()

    conv = 86400 * 30.44 * 1e-6  # m3/s * g/m3 -> tonnes/month
    stn_q_map = {
        "Tan Chau": "TanChau_Q",
        "My Thuan": "MyThuan_Q",
        "Can Tho": "CanTho_Q",
    }

    results = []
    for param in params:
        for stn, q_col in stn_q_map.items():
            wq_sub = wq_monthly[wq_monthly["stationName"] == stn][
                ["year", "month", param]
            ].dropna()
            merged = wq_sub.merge(
                q_monthly[["year", "month", q_col]], on=["year", "month"]
            )
            merged["load_tonnes_month"] = merged[q_col] * merged[param] * conv
            merged["param"] = param
            merged["station"] = stn
            results.append(
                merged[
                    [
                        "year",
                        "month",
                        "param",
                        "station",
                        q_col,
                        param,
                        "load_tonnes_month",
                    ]
                ]
            )

    loads = pd.concat(results, ignore_index=True)

    balance_records = []
    for param in params:
        tc = loads[(loads["station"] == "Tan Chau") & (loads["param"] == param)][
            ["year", "month", "load_tonnes_month"]
        ].rename(columns={"load_tonnes_month": "L_TanChau"})

        mt = loads[(loads["station"] == "My Thuan") & (loads["param"] == param)][
            ["year", "month", "load_tonnes_month"]
        ].rename(columns={"load_tonnes_month": "L_MyThuan"})

        ct = loads[(loads["station"] == "Can Tho") & (loads["param"] == param)][
            ["year", "month", "load_tonnes_month"]
        ].rename(columns={"load_tonnes_month": "L_CanTho"})

        merged = tc.merge(mt, on=["year", "month"]).merge(
            ct, on=["year", "month"], how="left"
        )
        merged["dL_Tien"] = merged["L_MyThuan"] - merged["L_TanChau"]
        merged["L_downstream"] = merged["L_MyThuan"] + merged["L_CanTho"].fillna(0)
        merged["dL_total"] = merged["L_downstream"] - merged["L_TanChau"]
        merged["enrichment_pct"] = (merged["dL_Tien"] / merged["L_TanChau"]) * 100
        merged["param"] = param
        balance_records.append(merged)

    balance = pd.concat(balance_records, ignore_index=True)

    for param in params:
        mask = balance["param"] == param
        sub = balance.loc[mask].copy()
        q_sub = q_monthly[["year", "month", "TanChau_Q", "MyThuan_Q"]].copy()
        sub = sub.merge(q_sub, on=["year", "month"], how="left")

        sub["Q_ratio"] = sub["MyThuan_Q"] / sub["TanChau_Q"]
        sub["L_expected_MyThuan"] = sub["L_TanChau"] * sub["Q_ratio"]
        sub["L_nonconservative"] = sub["L_MyThuan"] - sub["L_expected_MyThuan"]
        sub["noncon_pct"] = (
            sub["L_nonconservative"] / sub["L_expected_MyThuan"]
        ) * 100

        balance.loc[mask, "Q_ratio"] = sub["Q_ratio"].values
        balance.loc[mask, "L_expected_MyThuan"] = sub["L_expected_MyThuan"].values
        balance.loc[mask, "L_nonconservative"] = sub["L_nonconservative"].values
        balance.loc[mask, "noncon_pct"] = sub["noncon_pct"].values

    return loads, balance, params


def print_and_save_results(loads: pd.DataFrame, balance: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """Print summary statistics and save Table_NW_Mass_Balance.csv."""
    print("=" * 80)
    print("NET IN-DELTA MASS BALANCE TIME SERIES")
    print("=" * 80)

    annual_summary = []

    for param in params:
        param_short = param.replace(" (mg/L)", "")
        bal = balance[balance["param"] == param].copy()

        if len(bal) < 10:
            continue

        annual = (
            bal.groupby("year")
            .agg(
                L_TanChau=("L_TanChau", "sum"),
                L_MyThuan=("L_MyThuan", "sum"),
                L_CanTho=("L_CanTho", "sum"),
                dL_Tien=("dL_Tien", "sum"),
                dL_total=("dL_total", "sum"),
                n_months=("dL_Tien", "count"),
            )
            .reset_index()
        )
        annual = annual[annual["n_months"] >= 6].copy()
        annual["enrichment_pct"] = (annual["dL_Tien"] / annual["L_TanChau"]) * 100

        nc = (
            bal[bal["noncon_pct"].notna()].copy()
            if "noncon_pct" in bal.columns
            else pd.DataFrame()
        )
        nc_lookup = {}
        if len(nc) > 10:
            nc_annual = (
                nc.groupby("year")
                .agg(
                    L_noncon=("L_nonconservative", "sum"),
                    L_expected=("L_expected_MyThuan", "sum"),
                    Q_ratio=("Q_ratio", "mean"),
                    n_nc=("L_nonconservative", "count"),
                )
                .reset_index()
            )
            nc_annual = nc_annual[nc_annual["n_nc"] >= 6]
            nc_annual["noncon_pct"] = (
                nc_annual["L_noncon"] / nc_annual["L_expected"]
            ) * 100

            print(f"\n  {param_short} (years with >=6 months: {len(nc_annual)}):")
            for phase_name, yr1, yr2 in [
                ("Phase 1 (<=2004)", 1985, 2004),
                ("Phase 2 (>2004)", 2005, 2023),
            ]:
                phase = nc_annual[
                    (nc_annual["year"] >= yr1) & (nc_annual["year"] <= yr2)
                ]
                if len(phase) >= 3:
                    print(
                        f"    {phase_name}: non-conservative fraction = "
                        f"{phase['noncon_pct'].mean():+.1f}% +/- {phase['noncon_pct'].std():.1f}%"
                    )

            if len(nc_annual) >= 10:
                x_nc = nc_annual["year"].values.astype(float)
                y_nc = nc_annual["noncon_pct"].values
                slope_nc, _, _, _ = theilslopes(y_nc, x_nc)
                _, p_nc = kendalltau(x_nc, y_nc)
                print(
                    f"    Theil-Sen trend in NCF: {slope_nc:+.2f} %/yr (p={p_nc:.4f})"
                )

            nc_lookup = (
                nc.groupby("year")
                .agg(
                    L_noncon=("L_nonconservative", "sum"),
                    L_expected=("L_expected_MyThuan", "sum"),
                )
                .to_dict("index")
            )

        for _, row in annual.iterrows():
            yr = row["year"]
            rec = {
                "parameter": param_short,
                "year": yr,
                "L_TanChau_tonnes": row["L_TanChau"],
                "L_MyThuan_tonnes": row["L_MyThuan"],
                "L_CanTho_tonnes": row["L_CanTho"],
                "dL_Tien_tonnes": row["dL_Tien"],
                "dL_total_tonnes": row["dL_total"],
                "enrichment_pct": row["enrichment_pct"],
                "n_months": row["n_months"],
            }
            if yr in nc_lookup:
                rec["L_expected_MyThuan_tonnes"] = nc_lookup[yr]["L_expected"]
                rec["L_nonconservative_tonnes"] = nc_lookup[yr]["L_noncon"]
                if nc_lookup[yr]["L_expected"] != 0:
                    rec["nonconservative_pct"] = (
                        nc_lookup[yr]["L_noncon"] / nc_lookup[yr]["L_expected"]
                    ) * 100
            annual_summary.append(rec)

    df_out = pd.DataFrame(annual_summary)
    csv_path = TBL_DIR / "Table_NW_Mass_Balance.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.1f")
    print(f"\nSaved {csv_path} (shape={df_out.shape})")
    return df_out


def run_vamnao_mc_sensitivity(
    balance: pd.DataFrame, n_iter: int = 10000, seed: int = 42
) -> pd.DataFrame:
    """Deterministic Monte Carlo sensitivity analysis on the Vam Nao flow-split ratio."""
    rng = np.random.default_rng(seed)
    target_params = ["COD (mg/L)", "TN (mg/L)", "NO3-N (mg/L)"]
    rows_out = []

    print("\n" + "=" * 80)
    print(f"VAM NAO FLOW-SPLIT MONTE CARLO SENSITIVITY (n_iter={n_iter}, seed={seed})")
    print("=" * 80)

    for param in target_params:
        param_short = param.replace(" (mg/L)", "")
        sub = balance[(balance["param"] == param) & balance["noncon_pct"].notna()].copy()
        if sub.empty:
            continue

        yr_counts = sub.groupby("year")["L_nonconservative"].count()
        valid_years = np.sort(yr_counts[yr_counts >= 6].index.to_numpy(dtype=int))
        sub = (
            sub[sub["year"].isin(valid_years)]
            .sort_values(["year", "month"])
            .reset_index(drop=True)
        )

        n_obs = len(sub)
        n_yrs = len(valid_years)
        years_float = valid_years.astype(float)
        p1_mask = valid_years <= 2004
        p2_mask = valid_years > 2004

        base_ann = (
            sub.groupby("year")
            .agg(
                L_noncon=("L_nonconservative", "sum"),
                L_expected=("L_expected_MyThuan", "sum"),
            )
            .loc[valid_years]
        )
        y_base = (
            base_ann["L_noncon"].to_numpy(dtype=float)
            / base_ann["L_expected"].to_numpy(dtype=float)
        ) * 100.0

        base_p1 = float(np.mean(y_base[p1_mask]))
        base_p2 = float(np.mean(y_base[p2_mask]))
        base_delta = base_p2 - base_p1
        base_slope = float(theilslopes(y_base, years_float)[0])
        _, base_p = kendalltau(years_float, y_base)
        base_mw_p = float(
            mannwhitneyu(
                y_base[p1_mask], y_base[p2_mask], alternative="two-sided"
            ).pvalue
        )

        yr_to_idx = {yr: i for i, yr in enumerate(valid_years)}
        obs_yr_idx = np.array([yr_to_idx[int(y)] for y in sub["year"].to_numpy()])
        M_yr = np.zeros((n_obs, n_yrs), dtype=float)
        M_yr[np.arange(n_obs), obs_yr_idx] = 1.0

        obs_year_frac = (sub["year"].to_numpy(dtype=float) - 1985.0) / (
            2023.0 - 1985.0
        )
        q_ratio_base = sub["Q_ratio"].to_numpy(dtype=float)
        L_tc = sub["L_TanChau"].to_numpy(dtype=float)
        L_mt_base = sub["L_MyThuan"].to_numpy(dtype=float)

        idx_i, idx_j = np.triu_indices(n_yrs, k=1)
        dx_pairs = years_float[idx_j] - years_float[idx_i]

        for mode_name, drift_bound, noise_sd, coupled in [
            ("coupled_flow_split_15pct_drift_10pct_noise", 0.15, 0.10, True),
            ("uncompensated_kumar2026_2p5pct_drift_5pct_noise", 0.025, 0.05, False),
        ]:
            drifts = rng.uniform(-drift_bound, drift_bound, size=(n_iter, 1))
            trend_mult = 1.0 + drifts * obs_year_frac[None, :]
            noise = rng.normal(1.0, noise_sd, size=(n_iter, n_obs))
            q_ratio_pert = np.clip(
                q_ratio_base[None, :] * trend_mult * noise, 0.3, 0.9
            )

            L_exp_pert = L_tc[None, :] * q_ratio_pert
            if coupled:
                scale_factor = q_ratio_pert / q_ratio_base[None, :]
                L_mt_pert = L_mt_base[None, :] * scale_factor
            else:
                L_mt_pert = np.broadcast_to(L_mt_base[None, :], (n_iter, n_obs))

            L_noncon_pert = L_mt_pert - L_exp_pert
            ann_noncon = L_noncon_pert @ M_yr
            ann_exp = L_exp_pert @ M_yr
            Y_mc = (ann_noncon / ann_exp) * 100.0

            p1_means = np.mean(Y_mc[:, p1_mask], axis=1)
            p2_means = np.mean(Y_mc[:, p2_mask], axis=1)
            deltas = p2_means - p1_means

            dy_pairs = Y_mc[:, idx_j] - Y_mc[:, idx_i]
            slopes_mc = np.median(dy_pairs / dx_pairs[None, :], axis=1)

            pvals_mc = np.empty(n_iter, dtype=float)
            mw_pvals_mc = np.empty(n_iter, dtype=float)
            for k in range(n_iter):
                _, pk = kendalltau(years_float, Y_mc[k])
                pvals_mc[k] = pk
                mw_pvals_mc[k] = mannwhitneyu(
                    Y_mc[k, p1_mask], Y_mc[k, p2_mask], alternative="two-sided"
                ).pvalue

            rec = {
                "parameter": param_short,
                "mode": mode_name,
                "n_iter": n_iter,
                "n_years": n_yrs,
                "drift_bound_pct": drift_bound * 100.0,
                "monthly_noise_sd_pct": noise_sd * 100.0,
                "baseline_phase1_mean_pct": round(base_p1, 2),
                "mc_phase1_median_pct": round(float(np.median(p1_means)), 2),
                "mc_phase1_p2_5_pct": round(float(np.percentile(p1_means, 2.5)), 2),
                "mc_phase1_p97_5_pct": round(float(np.percentile(p1_means, 97.5)), 2),
                "baseline_phase2_mean_pct": round(base_p2, 2),
                "mc_phase2_median_pct": round(float(np.median(p2_means)), 2),
                "mc_phase2_p2_5_pct": round(float(np.percentile(p2_means, 2.5)), 2),
                "mc_phase2_p97_5_pct": round(float(np.percentile(p2_means, 97.5)), 2),
                "baseline_delta_p2_minus_p1_pct": round(base_delta, 2),
                "mc_delta_median_pct": round(float(np.median(deltas)), 2),
                "mc_delta_p2_5_pct": round(float(np.percentile(deltas, 2.5)), 2),
                "mc_delta_p97_5_pct": round(float(np.percentile(deltas, 97.5)), 2),
                "baseline_theil_sen_slope_pct_yr": round(base_slope, 4),
                "mc_slope_median_pct_yr": round(float(np.median(slopes_mc)), 4),
                "mc_slope_p2_5_pct_yr": round(float(np.percentile(slopes_mc, 2.5)), 4),
                "mc_slope_p97_5_pct_yr": round(float(np.percentile(slopes_mc, 97.5)), 4),
                "baseline_mk_p": round(float(base_p), 6),
                "mc_mk_p_median": round(float(np.median(pvals_mc)), 6),
                "mc_mk_p_p2_5": round(float(np.percentile(pvals_mc, 2.5)), 6),
                "mc_mk_p_p97_5": round(float(np.percentile(pvals_mc, 97.5)), 6),
                "frac_mk_p_lt_0_05": round(float(np.mean(pvals_mc < 0.05)), 4),
                "frac_mk_p_lt_0_01": round(float(np.mean(pvals_mc < 0.01)), 4),
                "baseline_mw_p": round(base_mw_p, 6),
                "mc_mw_p_median": round(float(np.median(mw_pvals_mc)), 6),
                "frac_mw_p_lt_0_05": round(float(np.mean(mw_pvals_mc < 0.05)), 4),
                "frac_delta_positive": round(float(np.mean(deltas > 0)), 4),
            }
            rows_out.append(rec)

    df_mc = pd.DataFrame(rows_out)
    out_csv = METRICS_DIR / "vamnao_mc_sensitivity.csv"
    df_mc.to_csv(out_csv, index=False)
    print(f"Saved Vam Nao Monte Carlo sensitivity metrics to {out_csv}")
    return df_mc


def run_residence_time_sensitivity() -> pd.DataFrame:
    """Compute Eulerian advective flushing time and 1D tidal dispersion sensitivity."""
    hydro = pd.read_csv(HYDRO_PATH)
    hydro["date"] = pd.to_datetime(hydro["Timestamp"], format="mixed", dayfirst=False)
    hydro["year"] = hydro["date"].dt.year
    hydro["month"] = hydro["date"].dt.month
    hydro = hydro[(hydro["year"] >= 1985) & (hydro["year"] <= 2022)].copy()

    q_mon = (
        hydro.groupby(["year", "month"])
        .agg(MyThuan=("MyThuan", "mean"), CanTho=("CanTho", "mean"))
        .reset_index()
    )

    stations_cfg = [
        ("My Thuan (Tien)", "MyThuan", 120_000.0, 6_000.0),
        ("Can Tho (Hau)", "CanTho", 120_000.0, 5_000.0),
        ("My Thuan -> Coast (100 km)", "MyThuan", 100_000.0, 5_600.0),
    ]

    seasons_cfg = [
        ("Dry_JanApr", [1, 2, 3, 4]),
        ("Dry_NovApr", [11, 12, 1, 2, 3, 4]),
        ("Wet_MayOct", [5, 6, 7, 8, 9, 10]),
        ("Annual", list(range(1, 13))),
    ]

    area_scenarios = [
        ("Static_A0", 0.0, 0.0),
        ("Incision_MyThuan_3p4pct_per_5yr", 0.034, 0.034 * (2022 - 2004) / 5.0),
        ("Incision_HamLuong_11p8pct_per_5yr", 0.118, 0.118 * (2022 - 2004) / 5.0),
    ]

    rows = []
    for stn_label, q_col, L_m, A0 in stations_cfg:
        for season_name, months in seasons_cfg:
            sub_s = q_mon[q_mon["month"].isin(months)].copy()
            for scen_name, rate_5yr, endpoint_frac_2022 in area_scenarios:
                years_arr = sub_s["year"].to_numpy(dtype=float)
                exp_frac_t = np.where(
                    years_arr > 2004, rate_5yr * (years_arr - 2004.0) / 5.0, 0.0
                )
                sub_s["A_t"] = A0 * (1.0 + exp_frac_t)
                sub_s["A_endpoint"] = np.where(
                    years_arr > 2004, A0 * (1.0 + endpoint_frac_2022), A0
                )
                sub_s["tau_adv_mon_d"] = (L_m * sub_s["A_t"]) / (
                    sub_s[q_col] * 86400.0
                )
                sub_s["tau_adv_end_mon_d"] = (L_m * sub_s["A_endpoint"]) / (
                    sub_s[q_col] * 86400.0
                )

                for phase_label, yr_min, yr_max in [
                    ("Phase1_1985_2004", 1985, 2004),
                    ("Phase2_2005_2022", 2005, 2022),
                ]:
                    p_df = sub_s[(sub_s["year"] >= yr_min) & (sub_s["year"] <= yr_max)]
                    Q_mean = float(p_df[q_col].mean())
                    A_mean = float(p_df["A_t"].mean())
                    A_end = float(p_df["A_endpoint"].mean())

                    v_mean_ms = Q_mean / A_mean
                    v_end_ms = Q_mean / A_end
                    tau_adv_meanQ_d = (L_m * A_mean) / (Q_mean * 86400.0)
                    tau_adv_endQ_d = (L_m * A_end) / (Q_mean * 86400.0)
                    tau_adv_monthly_mean_d = float(p_df["tau_adv_mon_d"].mean())
                    tau_adv_end_monthly_mean_d = float(p_df["tau_adv_end_mon_d"].mean())

                    rec = {
                        "Station": stn_label,
                        "Reach_L_km": L_m / 1000.0,
                        "Season": season_name,
                        "Area_Scenario": scen_name,
                        "Phase": phase_label,
                        "Q_mean_m3s": round(Q_mean, 1),
                        "A_phase_mean_m2": round(A_mean, 1),
                        "A_2022_endpoint_m2": round(A_end, 1),
                        "v_mean_ms": round(v_mean_ms, 4),
                        "v_2022_endpoint_ms": round(v_end_ms, 4),
                        "tau_adv_meanQ_d": round(tau_adv_meanQ_d, 3),
                        "tau_adv_meanQ_h": round(tau_adv_meanQ_d * 24.0, 2),
                        "tau_adv_2022_endpoint_meanQ_d": round(tau_adv_endQ_d, 3),
                        "tau_adv_2022_endpoint_meanQ_h": round(tau_adv_endQ_d * 24.0, 2),
                        "tau_adv_monthly_mean_d": round(tau_adv_monthly_mean_d, 3),
                        "tau_adv_2022_endpoint_monthly_mean_d": round(
                            tau_adv_end_monthly_mean_d, 3
                        ),
                    }

                    for Kx in [100, 300, 500]:
                        pe = (v_end_ms * L_m) / float(Kx)
                        red_factor = 1.0 / (1.0 + (1.0 / pe))
                        tau_disp_d = tau_adv_endQ_d * red_factor
                        tau_disp_mon_d = tau_adv_end_monthly_mean_d * red_factor
                        rec[f"Pe_Kx{Kx}"] = round(pe, 2)
                        rec[f"disp_factor_Kx{Kx}"] = round(red_factor, 4)
                        rec[f"tau_disp_meanQ_d_Kx{Kx}"] = round(tau_disp_d, 3)
                        rec[f"tau_disp_monthly_d_Kx{Kx}"] = round(tau_disp_mon_d, 3)

                    rows.append(rec)

    df_rt = pd.DataFrame(rows)
    out_csv = METRICS_DIR / "residence_time_sensitivity.csv"
    df_rt.to_csv(out_csv, index=False)
    print(f"Saved residence time sensitivity metrics to {out_csv}")
    return df_rt


def main() -> None:
    loads, balance, params = compute_mass_balance()
    print_and_save_results(loads, balance, params)
    run_vamnao_mc_sensitivity(balance, n_iter=10000, seed=42)
    run_residence_time_sensitivity()


if __name__ == "__main__":
    main()
