"""Hydro/climate covariates derived from the curated Gold Standard dataset.

Purpose
-------
Centralize how we derive annual/seasonal discharge and rainfall covariates for
model robustness tests (CEE reviewer-facing). This avoids ad-hoc logic across
scripts and makes joins deterministic.

Authoritative input (project-internal):
- data/hydrology/outputs/final_datasets/Mekong_Delta_Gold_Standard_Master_1985_2025.csv

Notes
-----
- Discharge columns are assumed to be in m3/s (daily), rainfall in mm/day.
- Annual discharge covariates are computed as annual means of daily discharge.
- Seasonal discharge covariates are computed as mean discharge over specified months.
- Rainfall covariates are computed as sums (mm) over the period.
- Anomalies are standardized (z-score) over the study window.

This module is deterministic and does not perform any I/O besides reading the
Gold Standard CSV through the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from analysis.lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard
except ModuleNotFoundError:
    from lib.gold_standard_hydro import GoldStandardColumns, load_gold_standard


DRY_MONTHS: list[int] = [1, 2, 3, 4]
WET_MONTHS: list[int] = [6, 7, 8, 9]


@dataclass(frozen=True)
class HydroClimateColumns:
    """Canonical covariate column names produced by this module."""

    # Rainfall totals (mm)
    rain_annual_mm: str = "Rain_Annual_mm"
    rain_dry_mm: str = "Rain_Dry_mm"
    rain_wet_mm: str = "Rain_Wet_mm"

    # Standardized anomalies (z-score)
    rain_annual_z: str = "Rain_Annual_z"
    rain_dry_z: str = "Rain_Dry_z"
    rain_wet_z: str = "Rain_Wet_z"

    # Branch discharge (m3/s)
    q_branch_annual: str = "Q_branch_AnnualMean_m3s"
    q_branch_dry: str = "Q_branch_DrySeasonMean_m3s"

    # Standardized anomalies
    q_branch_annual_z: str = "Q_branch_AnnualMean_z"
    q_branch_dry_z: str = "Q_branch_DrySeasonMean_z"

    # My Thuan mainstem discharge (year-level, same for all stations)
    q_mythuan_annual: str = "Q_MyThuan_AnnualMean_m3s"
    q_mythuan_dry: str = "Q_MyThuan_DrySeasonMean_m3s"
    q_mythuan_annual_z: str = "Q_MyThuan_AnnualMean_z"
    q_mythuan_dry_z: str = "Q_MyThuan_DrySeasonMean_z"


def _zscore(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    mu = float(np.nanmean(v.to_numpy(dtype=float)))
    sd = float(np.nanstd(v.to_numpy(dtype=float)))
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(np.nan, index=x.index)
    return (v - mu) / sd


def _annual_sum(series: pd.Series) -> pd.Series:
    return series.resample("YE").sum(min_count=1)


def _annual_mean(series: pd.Series) -> pd.Series:
    return series.resample("YE").mean()


def _seasonal_sum(series: pd.Series, months: Iterable[int]) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    s = series.loc[idx.month.isin(list(months))]
    return s.resample("YE").sum(min_count=1)


def _seasonal_mean(series: pd.Series, months: Iterable[int]) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    s = series.loc[idx.month.isin(list(months))]
    return s.resample("YE").mean()


def annual_hydroclimate_table(
    study_start: str = "1985-01-01",
    study_end: str = "2022-12-31",
) -> pd.DataFrame:
    """Return annual/seasonal hydro-climate covariates by year.

    Output columns include rainfall totals/anomalies and discharge means/anomalies
    for the three Tien distributary proxies (CoChien, HamLuong, MyTho).
    """

    gs = load_gold_standard().loc[study_start:study_end]
    cols = GoldStandardColumns()

    # Rainfall (mm/day)
    rain = gs[cols.rainfall_mm].astype(float)

    rain_ann = _annual_sum(rain)
    rain_dry = _seasonal_sum(rain, DRY_MONTHS)
    rain_wet = _seasonal_sum(rain, WET_MONTHS)

    # Discharge (m3/s): Tien distributaries used for Ben Tre station-years
    q_cc = gs[cols.q_cochien].astype(float)
    q_hl = gs[cols.q_hamluong].astype(float)
    q_mt = gs[cols.q_mytho].astype(float)

    # My Thuan mainstem discharge (upstream of all Ben Tre branches)
    q_mythuan = gs[cols.q_mythuan].astype(float)

    # Annual means
    q_cc_ann = _annual_mean(q_cc)
    q_hl_ann = _annual_mean(q_hl)
    q_mt_ann = _annual_mean(q_mt)
    q_mythuan_ann = _annual_mean(q_mythuan)

    # Dry-season means
    q_cc_dry = _seasonal_mean(q_cc, DRY_MONTHS)
    q_hl_dry = _seasonal_mean(q_hl, DRY_MONTHS)
    q_mt_dry = _seasonal_mean(q_mt, DRY_MONTHS)
    q_mythuan_dry = _seasonal_mean(q_mythuan, DRY_MONTHS)

    # Build table
    idx = rain_ann.index
    years = pd.DatetimeIndex(idx).year

    hcc = HydroClimateColumns()
    df = pd.DataFrame({
        "year": years,
        # rainfall
        hcc.rain_annual_mm: rain_ann.values,
        hcc.rain_dry_mm: rain_dry.reindex(idx).values,
        hcc.rain_wet_mm: rain_wet.reindex(idx).values,
        # discharge by branch (stored as separate columns; station-year join will select one)
        "Q_CoChien_AnnualMean_m3s": q_cc_ann.reindex(idx).values,
        "Q_HamLuong_AnnualMean_m3s": q_hl_ann.reindex(idx).values,
        "Q_MyTho_AnnualMean_m3s": q_mt_ann.reindex(idx).values,
        "Q_CoChien_DrySeasonMean_m3s": q_cc_dry.reindex(idx).values,
        "Q_HamLuong_DrySeasonMean_m3s": q_hl_dry.reindex(idx).values,
        "Q_MyTho_DrySeasonMean_m3s": q_mt_dry.reindex(idx).values,
        # My Thuan mainstem (year-level, same for all stations)
        hcc.q_mythuan_annual: q_mythuan_ann.reindex(idx).values,
        hcc.q_mythuan_dry: q_mythuan_dry.reindex(idx).values,
    })

    # anomalies (z-score)
    df[HydroClimateColumns().rain_annual_z] = _zscore(df[HydroClimateColumns().rain_annual_mm])
    df[HydroClimateColumns().rain_dry_z] = _zscore(df[HydroClimateColumns().rain_dry_mm])
    df[HydroClimateColumns().rain_wet_z] = _zscore(df[HydroClimateColumns().rain_wet_mm])
    df[hcc.q_mythuan_annual_z] = _zscore(df[hcc.q_mythuan_annual])
    df[hcc.q_mythuan_dry_z] = _zscore(df[hcc.q_mythuan_dry])

    return df


def branch_prefix(station_name: str) -> str | None:
    """Return Ben Tre branch prefix for station codes (HLxx/CCxx/MTxx), else None."""
    s = str(station_name)
    if len(s) >= 2 and s[:2] in {"HL", "CC", "MT"}:
        return s[:2]
    return None


def join_branch_hydroclimate(
    df_station_year: pd.DataFrame,
    study_start: str = "1985-01-01",
    study_end: str = "2022-12-31",
) -> pd.DataFrame:
    """Join branch discharge + rainfall covariates onto a station-year dataframe.

    Requires columns: stationName, year

    Adds columns (see HydroClimateColumns):
    - Q_branch annual/dry mean + z
    - Rain annual/dry/wet totals + z

    For non-BenTre code stations (no HL/CC/MT prefix), Q_branch columns are left as NaN.
    """

    df = df_station_year.copy()
    if "stationName" not in df.columns or "year" not in df.columns:
        raise ValueError("df_station_year must have columns: stationName, year")

    cov = annual_hydroclimate_table(study_start=study_start, study_end=study_end)
    cov = cov.copy()
    hcc = HydroClimateColumns()

    # attach rainfall + My Thuan mainstem Q covariates by year (year-level, all stations)
    df = df.merge(cov[[
        "year",
        hcc.rain_annual_mm,
        hcc.rain_dry_mm,
        hcc.rain_wet_mm,
        hcc.rain_annual_z,
        hcc.rain_dry_z,
        hcc.rain_wet_z,
        hcc.q_mythuan_annual,
        hcc.q_mythuan_dry,
        hcc.q_mythuan_annual_z,
        hcc.q_mythuan_dry_z,
    ]], on="year", how="left")

    # branch selection
    b = df["stationName"].astype(str).apply(branch_prefix)
    df["_branch"] = b

    # Map branch -> discharge columns in cov
    mapping = {
        "CC": ("Q_CoChien_AnnualMean_m3s", "Q_CoChien_DrySeasonMean_m3s"),
        "HL": ("Q_HamLuong_AnnualMean_m3s", "Q_HamLuong_DrySeasonMean_m3s"),
        "MT": ("Q_MyTho_AnnualMean_m3s", "Q_MyTho_DrySeasonMean_m3s"),
    }

    # Build year->values lookup
    cov_idx = cov.set_index("year")

    q_ann = []
    q_dry = []
    for yr, br in zip(df["year"].to_numpy(), df["_branch"].to_numpy()):
        if br not in mapping or int(yr) not in cov_idx.index:
            q_ann.append(np.nan)
            q_dry.append(np.nan)
            continue
        ann_col, dry_col = mapping[str(br)]
        q_ann.append(float(np.asarray(cov_idx.loc[int(yr), ann_col], dtype=float).ravel()[0]))
        q_dry.append(float(np.asarray(cov_idx.loc[int(yr), dry_col], dtype=float).ravel()[0]))

    df[HydroClimateColumns().q_branch_annual] = q_ann
    df[HydroClimateColumns().q_branch_dry] = q_dry

    # z-score anomalies for Q (computed over the subset of rows where Q exists)
    df[HydroClimateColumns().q_branch_annual_z] = _zscore(df[HydroClimateColumns().q_branch_annual])
    df[HydroClimateColumns().q_branch_dry_z] = _zscore(df[HydroClimateColumns().q_branch_dry])

    df = df.drop(columns=["_branch"])
    return df


def hydroclimate_feature_columns() -> list[str]:
    """Feature columns intended for Phase 3 robustness models."""
    c = HydroClimateColumns()
    return [
        c.rain_annual_z,
        c.rain_dry_z,
        c.q_mythuan_annual_z,
        c.q_mythuan_dry_z,
    ]
