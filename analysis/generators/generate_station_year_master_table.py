"""Phase 1: Build unified annual station-year master table.

Joins water quality annual means, LULC 16km metrics, population density,
and hydroclimate covariates into a single analysis-ready table.

Outputs
-------
- analysis/cache/station_year_master_latest.csv

Usage
-----
    python -m analysis.generators.generate_station_year_master_table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.lib.hydro_ingest import join_branch_hydroclimate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
LULC_PATH = ROOT / "analysis" / "cache" / "lulc_metrics_corrected_16km.csv"
POP_PATH = ROOT / "analysis" / "analysis_outputs" / "population_stationbuffer_1990_2022_16km_latest.csv"
TR_PATH = ROOT / "analysis" / "analysis_outputs" / "Table_Tidal_Range_By_Year.csv"
OUT_DIR = ROOT / "analysis" / "cache"

DRY_MONTHS = [1, 2, 3, 4]
WET_MONTHS = [6, 7, 8, 9]

WQ_PARAMS = ["COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)",
             "TSS (mg/L)", "Salinity", "pH"]


def _branch_code(station: str) -> str:
    """Return branch group code for grouped CV."""
    s = str(station)
    if s[:2] in ("HL", "CC", "MT", "BL"):
        return s[:2]
    # MRC stations
    name_map = {"Can Tho": "MRC_CT", "My Thuan": "MRC_MT",
                "My Tho": "MRC_MyTho", "Tan Chau": "MRC_TC"}
    return name_map.get(s, "Unknown")


def build_wq_annual(wq: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WQ to annual and seasonal means per station-year."""
    wq = wq.copy()
    wq["datetime"] = pd.to_datetime(wq["datetime"])
    wq["year"] = wq["datetime"].dt.year
    wq["month"] = wq["datetime"].dt.month

    # Annual means
    annual = wq.groupby(["stationName", "year"])[WQ_PARAMS].mean()
    annual.columns = [f"{c}_annual" for c in annual.columns]

    # Dry season means
    dry = wq[wq["month"].isin(DRY_MONTHS)].groupby(
        ["stationName", "year"])[WQ_PARAMS].mean()
    dry.columns = [f"{c}_dry" for c in dry.columns]

    # Wet season means
    wet = wq[wq["month"].isin(WET_MONTHS)].groupby(
        ["stationName", "year"])[WQ_PARAMS].mean()
    wet.columns = [f"{c}_wet" for c in wet.columns]

    # Observation count
    n_obs = wq.groupby(["stationName", "year"]).size().rename("n_obs_year")

    out = annual.join(dry).join(wet).join(n_obs).reset_index()
    return out


def build_master_table() -> pd.DataFrame:
    """Build the unified station-year master table."""
    # 1. WQ annual aggregation
    wq = pd.read_csv(WQ_PATH)
    df = build_wq_annual(wq)

    # Add metadata
    meta = wq[["stationName", "River", "Data source", "latitude", "longitude"]].drop_duplicates(
        subset=["stationName"]).set_index("stationName")
    df = df.join(meta, on="stationName")
    df["branch_code"] = df["stationName"].apply(_branch_code)
    df["network"] = df["Data source"]

    # 2. Join LULC (16 km)
    if LULC_PATH.exists():
        lulc = pd.read_csv(LULC_PATH)
        lulc_cols = [c for c in lulc.columns
                     if c not in ("year", "stationName", "River", "buffer_km", "landscape_area_ha")]
        df = df.merge(lulc[["stationName", "year"] + lulc_cols],
                       on=["stationName", "year"], how="left")

    # 3. Join population
    if POP_PATH.exists():
        pop = pd.read_csv(POP_PATH)
        pop_cols = ["stationName", "year", "pop_total", "pop_density"]
        df = df.merge(pop[pop_cols], on=["stationName", "year"], how="left")

    # 4. Join hydroclimate (rainfall + branch discharge)
    df = join_branch_hydroclimate(df)

    # 5. Tidal range index (annual mean daily TR for Can Tho and My Thuan)
    if TR_PATH.exists():
        tr = pd.read_csv(TR_PATH)
        tr_ann = tr[tr["Metric"] == "AnnualMeanDailyTidalRange_m"].copy()
        # pivot by station
        tr_piv = tr_ann.pivot(index="year", columns="Station", values="Value")
        tr_piv.columns = [f"TR_{c}_m" for c in tr_piv.columns]
        df = df.merge(tr_piv, on="year", how="left")

    # Sort
    df = df.sort_values(["stationName", "year"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_master_table()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "station_year_master_latest.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Phase 1] Wrote {out_csv}  ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Schema manifest
    schema = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "station_years": int(df[["stationName", "year"]].drop_duplicates().shape[0]),
        "networks": sorted(df["network"].dropna().unique().tolist()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
    }
    schema_path = OUT_DIR / "station_year_master_latest.schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[Phase 1] Wrote schema: {schema_path}")


if __name__ == "__main__":
    main()
