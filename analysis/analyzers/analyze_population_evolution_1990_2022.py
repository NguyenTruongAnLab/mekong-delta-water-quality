"""Study-area population evolution (1990-2022).

This script summarizes population totals and densities within the study boundary
defined by `data/landuse/Boundary_sampling_sites.gpkg` (Hau + Tien system).

Data source:
  - data/Population/outputs/rasters/mekong_delta_pop_1985_2025/
      mekong_delta_pop_1985_2025.nc

Important interpretation note
-----------------------------
The NetCDF variable `pop` is treated as *population counts per grid cell*.
Therefore, population totals are computed as the sum of cell values within the
study boundary mask (no area-weighting). Study-area *density* is computed as:

  density = total_pop / boundary_area_km2

Boundary area is computed in EPSG:32648.

Outputs (deterministic):
  - analysis/analysis_outputs/population_studyarea_timeseries_1990_2022_latest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from rasterio.windows import from_bounds as window_from_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.generators.generate_population_metrics_station_buffers import (
    affine_from_latlon,
    time_coord_to_years,
    zonal_sum_for_geom,
)
from analysis.lib.study_area_mask import DEFAULT_BOUNDARY_GPKG, DEFAULT_BOUNDARY_LAYER, load_study_boundary


POP_NC_REL = Path(
    "data/Population/outputs/rasters/mekong_delta_pop_1985_2025/mekong_delta_pop_1985_2025.nc"
)

YEAR_START = 1990
YEAR_END = 2022
BOUNDARY_AREA_CRS = "EPSG:32648"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _analysis_out_dir() -> Path:
    out = Path(__file__).resolve().parents[1] / "analysis_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def boundary_area_km2(*, boundary_gdf: gpd.GeoDataFrame, boundary_geom) -> float:
    area_m2 = (
        gpd.GeoSeries([boundary_geom], crs=boundary_gdf.crs)
        .to_crs(BOUNDARY_AREA_CRS)
        .area.astype(float)
        .iloc[0]
    )
    return float(area_m2 / 1e6)


def summarize_population_timeseries(
    *,
    ds: xr.Dataset,
    boundary_geom,
    years: list[int],
) -> pd.DataFrame:
    if "pop" not in ds.data_vars:
        raise ValueError(f"NetCDF missing 'pop' variable; found {list(ds.data_vars)}")

    da = ds["pop"]
    if not ("time" in da.dims and "lat" in da.dims and "lon" in da.dims):
        raise ValueError(f"Unexpected pop dims {da.dims}; expected ('time','lat','lon')")

    time_years = time_coord_to_years(ds["time"])
    year_to_index: dict[int, int] = {}
    for i, y in enumerate(time_years.tolist()):
        yi = int(y)
        if yi not in year_to_index:
            year_to_index[yi] = int(i)

    missing = [y for y in years if y not in year_to_index]
    if missing:
        raise ValueError(f"Population stack missing required years: {missing[:10]}{'...' if len(missing)>10 else ''}")

    lat = ds["lat"].values
    lon = ds["lon"].values
    base_transform = affine_from_latlon(lat, lon)
    height = int(lat.size)
    width = int(lon.size)

    # Clamp to raster extent using a window for speed.
    left, bottom, right, top = boundary_geom.bounds
    w = window_from_bounds(left, bottom, right, top, transform=base_transform)
    w = w.round_offsets().round_lengths()
    # Clamp to full raster
    r0 = max(0, int(w.row_off))
    c0 = max(0, int(w.col_off))
    r1 = min(height, int(w.row_off + w.height))
    c1 = min(width, int(w.col_off + w.width))
    if r1 <= r0 or c1 <= c0:
        raise ValueError("Boundary has no overlap with population raster grid")

    # Transform for the cropped window (same for all years)
    win_transform = base_transform * Affine.translation(c0, r0)

    out_rows: list[dict] = []
    nodata = da.encoding.get("_FillValue", None)
    if nodata is None:
        nodata = da.attrs.get("_FillValue", None)

    for year in years:
        layer = da.isel(time=year_to_index[int(year)])
        arr_full = np.asarray(layer.values)
        arr = arr_full[r0:r1, c0:c1]
        pop_total = zonal_sum_for_geom(
            array=arr,
            transform=win_transform,
            geom=boundary_geom,
            nodata=nodata,
            all_touched=True,
        )
        if not np.isfinite(pop_total):
            pop_total = float("nan")

        out_rows.append({"year": int(year), "pop_total": float(pop_total)})

    df = pd.DataFrame(out_rows).sort_values("year").reset_index(drop=True)
    return df


def main() -> int:
    root = _repo_root()
    pop_nc = root / POP_NC_REL
    if not pop_nc.exists():
        raise FileNotFoundError(f"Population NetCDF not found: {pop_nc}")

    boundary_gdf, boundary_geom = load_study_boundary(
        str(root / DEFAULT_BOUNDARY_GPKG), layer=DEFAULT_BOUNDARY_LAYER
    )
    boundary_geom_wgs84 = (
        gpd.GeoSeries([boundary_geom], crs=boundary_gdf.crs).to_crs("EPSG:4326").iloc[0]
    )
    area_km2 = boundary_area_km2(boundary_gdf=boundary_gdf, boundary_geom=boundary_geom)
    if not (area_km2 > 0):
        raise ValueError(f"Invalid boundary area {area_km2}")

    ds = xr.open_dataset(pop_nc)
    years = list(range(YEAR_START, YEAR_END + 1))
    df = summarize_population_timeseries(ds=ds, boundary_geom=boundary_geom_wgs84, years=years)
    df["boundary_area_km2"] = float(area_km2)
    df["pop_density"] = df["pop_total"] / float(area_km2)

    out_dir = _analysis_out_dir()
    out_csv = out_dir / "population_studyarea_timeseries_1990_2022_latest.csv"

    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
