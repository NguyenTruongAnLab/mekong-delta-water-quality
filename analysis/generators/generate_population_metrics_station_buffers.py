"""Generate station-centered population metrics in 16 km buffers (1990-2022).

Population-fragmentation validation test.

This script:
  1) Builds station buffer geometries identically to
     `analysis/generators/generate_lulc_metrics_jaxa_aligned.py`:
       - station coordinates from `data/mekong_delta_water_quality_1985_2023.csv`
       - dedupe by `stationName`
       - EPSG:4326 -> project to EPSG:32648 -> shapely buffer (km -> m)
  2) Uses the clipped Mekong Delta NetCDF population stack:
       data/Population/outputs/rasters/mekong_delta_pop_1985_2025/
         mekong_delta_pop_1985_2025.nc
  3) For each year 1990-2022 and station buffer:
       - population total in buffer (sum of raster pixel values)
       - buffer area (km^2) and density (persons/km^2)
  4) Joins against 16 km LULC metrics
       analysis/cache/lulc_metrics_corrected_16km.csv
     and computes Spearman correlations with urban fragmentation metrics.

Outputs (deterministic):
  - analysis/analysis_outputs/population_stationbuffer_1990_2022_16km_latest.csv
  - analysis/analysis_outputs/population_fragmentation_metrics_latest.csv
  - analysis/analysis_outputs/figure_population_vs_fragmentation_points_latest.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from scipy.stats import spearmanr


STATION_CRS = "EPSG:4326"
BUFFER_CRS = "EPSG:32648"
BUFFER_KM = 16
YEAR_START = 1990
YEAR_END = 2022


def time_coord_to_years(time_coord) -> np.ndarray:
    """Convert a NetCDF/xarray time coordinate to calendar years.

    Robust to common encodings:
      - integer years (e.g., 1990)
      - datetime64 (e.g., 1990-01-01)
      - object/cftime-like values with a `.year` attribute

    Returns
    -------
    np.ndarray[int]
        Year for each element in the time coordinate.
    """
    values = time_coord.values if hasattr(time_coord, "values") else np.asarray(time_coord)
    values = np.asarray(values)

    # 1) Integer years
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)

    # 1b) Float years (rare but seen in some stacks)
    if np.issubdtype(values.dtype, np.floating):
        if np.all(np.isfinite(values)) and np.allclose(values, np.round(values)):
            return np.round(values).astype(np.int64)
        raise ValueError(f"Time coordinate looks like floats but not integer-like: dtype={values.dtype}")

    # 2) datetime64
    if np.issubdtype(values.dtype, np.datetime64):
        # Pandas handles datetime64 units robustly.
        dt = pd.to_datetime(values)
        return dt.year.astype(np.int64).to_numpy()

    # 3) Object/cftime-like
    if values.dtype == object:
        years: list[int] = []
        for v in values.tolist():
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                raise ValueError("Time coordinate contains null/NaN values")
            if hasattr(v, "year"):
                years.append(int(getattr(v, "year")))
                continue
            # Best-effort fallback for other datetime-like objects
            try:
                years.append(int(pd.to_datetime(v).year))
            except Exception as e:  # pragma: no cover
                raise TypeError(f"Unsupported time element type {type(v)}: {v!r}") from e
        return np.asarray(years, dtype=np.int64)

    raise TypeError(f"Unsupported time coordinate dtype: {values.dtype}")


@dataclass(frozen=True)
class StationBuffer:
    stationName: str
    river: str
    buffer_km: float
    area_km2: float
    geom_wgs84: object  # shapely geometry
    window: Window


def _repo_root() -> Path:
    # analysis/generators/ -> repo root
    return Path(__file__).resolve().parents[2]


def _analysis_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _paths() -> dict[str, Path]:
    root = _repo_root()
    analysis_dir = _analysis_dir()

    stations_csv = root / "data" / "mekong_delta_water_quality_1985_2023.csv"
    if not stations_csv.exists():
        for alt in [analysis_dir / "mekong_delta_water_quality_1985_2023.csv", root / "mekong_delta_water_quality_1985_2023.csv"]:
            if alt.exists():
                stations_csv = alt
                break

    pop_nc = root / "data" / "Population" / "outputs" / "rasters" / "mekong_delta_pop_1985_2025" / "mekong_delta_pop_1985_2025.nc"
    lulc_metrics = analysis_dir / "cache" / "lulc_metrics_corrected_16km.csv"
    out_dir = analysis_dir / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "analysis_dir": analysis_dir,
        "stations_csv": stations_csv,
        "pop_nc": pop_nc,
        "lulc_metrics": lulc_metrics,
        "out_dir": out_dir,
    }


def affine_from_latlon(lat: np.ndarray, lon: np.ndarray):
    """Build an Affine transform from 1D lat/lon coordinate arrays.

    Assumes coordinates represent *cell centers* on a regular grid.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected 1D lat/lon coordinate arrays")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("lat/lon must have at least 2 values to infer resolution")

    dy = float(abs(lat[1] - lat[0]))
    dx = float(abs(lon[1] - lon[0]))
    if not (dx > 0 and dy > 0):
        raise ValueError(f"Invalid inferred resolution dx={dx}, dy={dy}")

    left = float(lon.min() - dx / 2)
    right = float(lon.max() + dx / 2)
    bottom = float(lat.min() - dy / 2)
    top = float(lat.max() + dy / 2)

    return from_bounds(left, bottom, right, top, width=int(lon.size), height=int(lat.size))


def zonal_sum_for_geom(
    *,
    array: np.ndarray,
    transform,
    geom,
    nodata: float | int | None,
    all_touched: bool = True,
) -> float:
    """Zonal sum for a single geometry.

    Uses a binary inclusion mask (no fractional pixel weighting).
    """
    if geom is None or getattr(geom, "is_empty", False):
        return float("nan")

    if np.ma.isMaskedArray(array):
        base_mask = np.ma.getmaskarray(array)
        data = np.ma.filled(array, 0)
    else:
        base_mask = np.zeros(array.shape, dtype=bool)
        data = array

    data = np.asarray(data)
    base_mask = base_mask | (~np.isfinite(data))
    if nodata is not None:
        base_mask = base_mask | (data == nodata)

    # Population counts should never be negative; treat negatives as missing.
    base_mask = base_mask | (data < 0)

    mask = geometry_mask(
        [geom],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=all_touched,
    )

    m = mask & (~base_mask)
    return float(np.sum(data[m], dtype=np.float64))


def _window_for_geom_bounds(*, geom_bounds, base_transform, width: int, height: int) -> Window:
    """Compute a clamped raster window for geometry bounds."""
    left, bottom, right, top = geom_bounds
    w = window_from_bounds(left, bottom, right, top, transform=base_transform)
    w = w.round_offsets().round_lengths()
    # rasterio Window stubs can be inconsistent across versions; ignore call-arg typing.
    full = Window(0, 0, width, height)  # type: ignore[call-arg]
    try:
        w = w.intersection(full)
    except Exception:
        # No overlap
        return Window(0, 0, 0, 0)  # type: ignore[call-arg]
    return w


def build_station_buffers(*, stations_csv: Path, base_transform, raster_width: int, raster_height: int) -> list[StationBuffer]:
    stations_df = pd.read_csv(stations_csv)
    unique = stations_df.drop_duplicates(subset="stationName").reset_index(drop=True)
    required = ["stationName", "longitude", "latitude", "River"]
    missing = [c for c in required if c not in unique.columns]
    if missing:
        raise ValueError(f"Stations CSV missing required columns: {missing}; columns={list(unique.columns)}")

    gdf = gpd.GeoDataFrame(
        unique,
        geometry=gpd.points_from_xy(unique["longitude"], unique["latitude"]),
        crs=STATION_CRS,
    )
    gdf_metric = gdf.to_crs(BUFFER_CRS)
    buffer_radius_m = float(BUFFER_KM) * 1000.0
    gdf_metric["buffer_geom"] = gdf_metric.geometry.buffer(buffer_radius_m)
    gdf_metric["buffer_area_km2"] = gdf_metric["buffer_geom"].area.astype(float) / 1e6

    buffers_metric = gpd.GeoDataFrame(
        {
            "stationName": gdf_metric["stationName"].astype(str),
            "River": gdf_metric["River"].astype(str),
            "area_km2": gdf_metric["buffer_area_km2"].astype(float),
        },
        geometry=gdf_metric["buffer_geom"],
        crs=BUFFER_CRS,
    )

    buffers_wgs84 = buffers_metric.to_crs(STATION_CRS)

    out: list[StationBuffer] = []
    for _, row in buffers_wgs84.iterrows():
        geom = row.geometry
        w = _window_for_geom_bounds(
            geom_bounds=geom.bounds,
            base_transform=base_transform,
            width=raster_width,
            height=raster_height,
        )
        out.append(
            StationBuffer(
                stationName=str(row["stationName"]),
                river=str(row["River"]),
                buffer_km=float(BUFFER_KM),
                area_km2=float(row["area_km2"]),
                geom_wgs84=geom,
                window=w,
            )
        )

    # Deterministic order
    out = sorted(out, key=lambda b: b.stationName)
    return out


def compute_population_station_buffers(*, pop_ds: xr.Dataset, buffers: list[StationBuffer], years: Iterable[int]) -> pd.DataFrame:
    if "pop" not in pop_ds.data_vars:
        # Safe default: use first data variable
        first = list(pop_ds.data_vars)[0] if pop_ds.data_vars else None
        raise ValueError(f"Expected 'pop' variable in NetCDF; found {list(pop_ds.data_vars)} (first={first})")

    da = pop_ds["pop"]
    if not ("time" in da.dims and "lat" in da.dims and "lon" in da.dims):
        raise ValueError(f"Unexpected pop dims {da.dims}; expected ('time','lat','lon')")

    lat = pop_ds["lat"].values
    lon = pop_ds["lon"].values
    base_transform = affine_from_latlon(lat, lon)
    height = int(lat.size)
    width = int(lon.size)

    nodata = da.encoding.get("_FillValue", None)
    if nodata is None:
        nodata = da.attrs.get("_FillValue", None)
    # Note: population stack uses counts (units="persons"), so sum is in persons.

    records: list[dict] = []
    years = list(int(y) for y in years)

    time_years = time_coord_to_years(pop_ds["time"])
    if time_years.ndim != 1:
        raise ValueError(f"Expected 1D time coordinate; got shape {time_years.shape}")

    # Map calendar year -> index (first occurrence). Use isel for robust selection.
    year_to_index: dict[int, int] = {}
    for idx, y in enumerate(time_years.tolist()):
        yi = int(y)
        if yi not in year_to_index:
            year_to_index[yi] = int(idx)

    missing_years = [y for y in years if y not in year_to_index]
    if missing_years:
        raise ValueError(
            f"NetCDF missing required years: {missing_years[:10]}{'...' if len(missing_years)>10 else ''}; "
            f"available_years=[{min(year_to_index)}..{max(year_to_index)}] (n={len(year_to_index)})"
        )

    for year in years:
        layer = da.isel(time=year_to_index[int(year)])
        # Ensure we have a plain ndarray; keep NaNs to be masked later
        arr_full = np.asarray(layer.values)
        if arr_full.shape != (height, width):
            raise ValueError(f"Unexpected array shape for year={year}: {arr_full.shape} vs {(height,width)}")

        for buf in buffers:
            w = buf.window
            if int(w.width) == 0 or int(w.height) == 0:
                pop_total = 0.0
            else:
                r0 = int(w.row_off)
                r1 = int(w.row_off + w.height)
                c0 = int(w.col_off)
                c1 = int(w.col_off + w.width)
                arr = arr_full[r0:r1, c0:c1]
                t = window_transform(w, base_transform)
                pop_total = zonal_sum_for_geom(array=arr, transform=t, geom=buf.geom_wgs84, nodata=nodata, all_touched=True)
                if not np.isfinite(pop_total):
                    pop_total = float("nan")

            area_km2 = float(buf.area_km2)
            pop_density = (float(pop_total) / area_km2) if (area_km2 > 0 and np.isfinite(pop_total)) else float("nan")

            records.append(
                {
                    "stationName": buf.stationName,
                    "year": int(year),
                    "buffer_km": float(buf.buffer_km),
                    "pop_total": float(pop_total),
                    "buffer_area_km2": float(area_km2),
                    "pop_density": float(pop_density),
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["stationName", "year"], kind="mergesort").reset_index(drop=True)
    return df


def read_station_metadata(*, stations_csv: Path) -> pd.DataFrame:
    """Read stations CSV and return one row per stationName with optional metadata."""
    stations_df = pd.read_csv(stations_csv)
    if "stationName" not in stations_df.columns:
        raise ValueError(f"Stations CSV missing 'stationName' column; columns={list(stations_df.columns)}")

    meta_cols = ["stationName"]
    if "River" in stations_df.columns:
        meta_cols.append("River")
    if "Data source" in stations_df.columns:
        meta_cols.append("Data source")

    meta = stations_df[meta_cols].drop_duplicates(subset="stationName").copy()
    meta["stationName"] = meta["stationName"].astype(str)
    if "Data source" in meta.columns:
        meta["Data source"] = meta["Data source"].astype(str)
    return meta.reset_index(drop=True)


def station_sets_from_metadata(meta: pd.DataFrame) -> dict[str, set[str]]:
    all_stations = set(meta["stationName"].astype(str).tolist())
    if "Data source" in meta.columns:
        ben_tre = set(meta.loc[meta["Data source"].astype(str).str.strip().str.lower() == "ben tre", "stationName"].astype(str).tolist())
    else:
        ben_tre = set()
    return {
        "all_stations": all_stations,
        "ben_tre_only": ben_tre,
    }


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _spearman_table(*, df: pd.DataFrame, x: str, y: str, label: str) -> dict:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(sub))
    if n < 3:
        return {"panel": label, "x": x, "y": y, "n": n, "spearman_rho": np.nan, "p_value": np.nan}
    res = spearmanr(sub[x].values, sub[y].values)
    # SciPy may return either a (rho, p) tuple or a result object with attributes.
    rho = getattr(res, "statistic", None)
    p = getattr(res, "pvalue", None)
    if rho is None or p is None:
        rho, p = res  # type: ignore[misc]
    return {
        "panel": label,
        "x": x,
        "y": y,
        "n": n,
        "spearman_rho": float(cast(float, rho)),
        "p_value": float(cast(float, p)),
    }


def build_fragmentation_metrics(
    *,
    pop_metrics: pd.DataFrame,
    lulc_metrics_path: Path,
    station_names: set[str] | None = None,
    station_set: str = "all_stations",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lulc = pd.read_csv(lulc_metrics_path)
    lulc["year"] = lulc["year"].astype(int)
    lulc["stationName"] = lulc["stationName"].astype(str)

    cols_needed = [
        "stationName",
        "year",
        "Residential_Urban_config_PD",
        "Residential_Urban_composition_pct",
    ]
    missing = [c for c in cols_needed if c not in lulc.columns]
    if missing:
        raise ValueError(f"LULC metrics missing required columns: {missing}")

    lulc = lulc[cols_needed].copy()
    pop = pop_metrics.copy()
    pop["year"] = pop["year"].astype(int)
    pop["stationName"] = pop["stationName"].astype(str)
    if station_names is not None:
        pop = pop[pop["stationName"].isin(sorted(station_names))].copy()

    joined = pop.merge(lulc, on=["stationName", "year"], how="inner")
    joined = joined.sort_values(["stationName", "year"], kind="mergesort").reset_index(drop=True)
    joined["station_set"] = str(station_set)

    # Build two panels: station-year and station-mean
    station_year = joined.copy()
    station_mean = (
        joined.groupby("stationName", as_index=False)
        .agg(
            {
                "pop_density": "mean",
                "Residential_Urban_config_PD": "mean",
                "Residential_Urban_composition_pct": "mean",
            }
        )
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    for panel_name, frame in [("station_year", station_year), ("station_mean", station_mean)]:
        rows.append(
            _spearman_table(
                df=frame,
                x="pop_density",
                y="Residential_Urban_config_PD",
                label=panel_name,
            )
        )
        rows.append(
            _spearman_table(
                df=frame,
                x="pop_density",
                y="Residential_Urban_composition_pct",
                label=panel_name,
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["panel", "x", "y", "n", "spearman_rho", "p_value"]]
    metrics_df.insert(0, "station_set", str(station_set))
    return joined, metrics_df


def main() -> int:
    paths = _paths()
    pop_nc = paths["pop_nc"]
    stations_csv = paths["stations_csv"]
    lulc_metrics = paths["lulc_metrics"]
    out_dir = paths["out_dir"]

    if not pop_nc.exists():
        raise FileNotFoundError(f"Population NetCDF not found: {pop_nc}")
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")
    if not lulc_metrics.exists():
        raise FileNotFoundError(f"LULC metrics not found: {lulc_metrics}")

    ds = xr.open_dataset(pop_nc)
    if "pop" not in ds.data_vars:
        raise ValueError(f"NetCDF missing 'pop' variable; found vars={list(ds.data_vars)}")

    years = list(range(YEAR_START, YEAR_END + 1))

    base_transform = affine_from_latlon(ds["lat"].values, ds["lon"].values)
    buffers = build_station_buffers(
        stations_csv=stations_csv,
        base_transform=base_transform,
        raster_width=int(ds["lon"].size),
        raster_height=int(ds["lat"].size),
    )

    pop_metrics = compute_population_station_buffers(pop_ds=ds, buffers=buffers, years=years)

    meta = read_station_metadata(stations_csv=stations_csv)
    if "Data source" in meta.columns:
        pop_metrics = pop_metrics.merge(
            meta[["stationName", "Data source"]].rename(columns={"Data source": "data_source"}),
            on="stationName",
            how="left",
        )
    else:
        pop_metrics["data_source"] = np.nan

    out_pop = out_dir / "population_stationbuffer_1990_2022_16km_latest.csv"
    pop_metrics.to_csv(out_pop, index=False)

    station_sets = station_sets_from_metadata(meta)
    if not station_sets["ben_tre_only"]:
        if "Data source" not in meta.columns:
            raise ValueError(
                "Ben Tre-only station set cannot be constructed because the stations CSV is missing the 'Data source' column."
            )
        raise ValueError("Stations CSV includes 'Data source' but no Ben Tre stations were detected")

    joined_all: list[pd.DataFrame] = []
    metrics_all: list[pd.DataFrame] = []

    for station_set, names in [("all_stations", station_sets["all_stations"]), ("ben_tre_only", station_sets["ben_tre_only"])]:
        joined, metrics_df = build_fragmentation_metrics(
            pop_metrics=pop_metrics,
            lulc_metrics_path=lulc_metrics,
            station_names=names,
            station_set=station_set,
        )
        joined_all.append(joined)
        metrics_all.append(metrics_df)

    joined_all_df = pd.concat(joined_all, ignore_index=True)
    metrics_all_df = pd.concat(metrics_all, ignore_index=True)

    out_metrics = out_dir / "population_fragmentation_metrics_latest.csv"
    metrics_all_df.to_csv(out_metrics, index=False)

    out_points = out_dir / "figure_population_vs_fragmentation_points_latest.csv"
    joined_all_df.to_csv(out_points, index=False)

    print(f"Wrote: {out_pop}")
    print(f"Wrote: {out_metrics}")
    print(f"Wrote: {out_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
