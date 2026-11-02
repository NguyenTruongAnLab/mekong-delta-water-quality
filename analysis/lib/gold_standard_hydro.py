"""Utilities for the curated daily hydrology dataset.

This module centralizes loading/parsing and common aggregations so that
hydrology/rainfall/load analyses stay consistent across scripts.

Authoritative input (project-internal):
- data/hydrology/outputs/final_datasets/Mekong_Delta_Gold_Standard_Master_1985_2025.csv

Expected columns (at minimum):
- Timestamp (daily)
- Basin_Rainfall_mm
- TanChau, ChauDoc, CanTho, MyThuan (station discharges; m3/s)
- CoChien, HamLuong, MyTho (downstream branches; m3/s)

All functions here are deterministic and avoid hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLS = {
    "Timestamp",
    "Basin_Rainfall_mm",
    "TanChau",
    "ChauDoc",
    "CanTho",
    "MyThuan",
    "CoChien",
    "HamLuong",
    "MyTho",
}


def default_gold_standard_path() -> Path:
    # analysis/lib/gold_standard_hydro.py -> analysis/lib -> analysis -> scripts root
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "hydrology"
        / "outputs"
        / "final_datasets"
        / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
    )


def load_gold_standard(path: str | Path | None = None) -> pd.DataFrame:
    """Load the curated hydrology dataset as a daily time-indexed dataframe."""

    p = Path(path) if path is not None else default_gold_standard_path()
    if not p.exists():
        raise FileNotFoundError(f"Curated hydrology master CSV not found: {p}")

    df = pd.read_csv(p)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Hydrology master CSV missing required columns: {sorted(missing)}")

    # Robust parsing (your file is M/D/YYYY in sample)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        bad = int(df["Timestamp"].isna().sum())
        raise ValueError(f"Failed to parse {bad} Timestamp values in hydrology master CSV")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df = df.set_index("Timestamp")

    # Ensure numeric columns are numeric (coerce bad strings to NaN)
    for c in REQUIRED_COLS - {"Timestamp"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def annual_mean(df_daily: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Annual mean (year-end index) for selected columns."""
    out = df_daily[list(cols)].resample("YE").mean(numeric_only=True)
    out.index.name = "Timestamp"
    return out


def annual_sum(df_daily: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Annual sum (year-end index) for selected columns."""
    out = df_daily[list(cols)].resample("YE").sum(numeric_only=True)
    out.index.name = "Timestamp"
    return out


def seasonal_total(series_daily_mm: pd.Series, months: list[int]) -> pd.Series:
    """Annual seasonal totals (mm) for a daily precipitation series."""
    s = series_daily_mm.copy()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    idx = pd.DatetimeIndex(s.index)
    s = s[idx.month.isin(months)]
    return s.resample("YE").sum()


@dataclass(frozen=True)
class GoldStandardColumns:
    """Canonical column names in the curated hydrology CSV."""

    rainfall_mm: str = "Basin_Rainfall_mm"

    # Main stations
    q_tanchau: str = "TanChau"
    q_chaudoc: str = "ChauDoc"
    q_cantho: str = "CanTho"
    q_mythuan: str = "MyThuan"

    # Downstream branches
    q_cochien: str = "CoChien"
    q_hamluong: str = "HamLuong"
    q_mytho: str = "MyTho"
