"""Study-area LULC evolution time series (1990-2022), product-separated.

This script summarizes the Hau + Tien study-area land-use/land-cover composition
for each year in a shared 8-category harmonized scheme.

IMPORTANT: Cross-product overlap validation (2017-2020) indicates that VLUCD and
LULC_2017_2022 are not directly comparable at the category level (FAIL).
Therefore, this script DOES NOT splice them into a single continuous series.
Instead it exports two product-separated series:
    - VLUCD: 1990-2020
    - LULC_2017_2022: 2017-2022

Data sources
------------
- 1990-2020: JAXA VLUCD L2 (30 m):
    data/landuse/MekongDelta_VLUCD_1990_2020/VLUCD_L2_30m_YYYY.tif
- 2017-2022: MekongDelta_LULC_2017_2022 (already reclassified to IDs 1..8):
    data/landuse/MekongDelta_LULC_2017_2022/mekong_YYYY_30m.tif

Mask
----
- Study boundary: data/landuse/Boundary_sampling_sites.gpkg :: Boundary_sampling_sites

Outputs
-------
- analysis/analysis_outputs/lulc_studyarea_timeseries_vlucd_1990_2020_latest.csv
- analysis/analysis_outputs/lulc_studyarea_timeseries_lulc_2017_2022_latest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lib.lulc_harmonize import (
    HARMONIZED_CATEGORIES,
    assert_harmonized_schema,
    harmonize_lulc2017_ids,
    harmonize_vlucd_codes_to_ids,
)
from analysis.lib.study_area_mask import load_study_boundary, rasterize_geometry_mask


DST_CRS = "EPSG:32648"
DST_RES_M = 30  # target pixel size in meters for area-consistent summaries


def get_years_vlucd() -> List[int]:
    """Years available in the VLUCD series used here."""

    return list(range(1990, 2021))


def get_years_lulc_2017_2022() -> List[int]:
    """Years available in the MekongDelta_LULC_2017_2022 product used here."""

    return list(range(2017, 2023))


def get_years_all() -> List[int]:
    """Full window of interest (1990–2022 inclusive)."""

    return list(range(1990, 2023))


def _pixel_area_m2(transform: rasterio.Affine) -> float:
    return float(abs(transform.a * transform.e))


def _crop_to_boundary(
    src,
    boundary_geom_src_crs,
    *,
    fill_nodata: int,
) -> Tuple[np.ndarray, rasterio.Affine]:
    out_image, out_transform = rio_mask(
        src,
        [mapping(boundary_geom_src_crs)],
        crop=True,
        filled=True,
        nodata=fill_nodata,
    )
    return out_image[0], out_transform


def _reproject_array_to_dst(
    arr: np.ndarray,
    *,
    src_transform: rasterio.Affine,
    src_crs,
    src_nodata: int,
    dst_crs: str = DST_CRS,
    dst_res_m: float = DST_RES_M,
    dst_nodata: int = 0,
) -> Tuple[np.ndarray, rasterio.Affine]:
    if arr.ndim != 2:
        raise ValueError("Expected 2D array")

    height, width = arr.shape
    left, bottom, right, top = array_bounds(height, width, src_transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        left,
        bottom,
        right,
        top,
        resolution=dst_res_m,
    )

    if dst_width is None or dst_height is None:
        raise ValueError("Reprojection produced undefined output shape")

    dst_h = int(dst_height)
    dst_w = int(dst_width)
    dst = np.full((dst_h, dst_w), dst_nodata, dtype=np.int16)

    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )

    return dst, dst_transform


def summarize_raster_category_areas(
    raster_path: Path,
    *,
    boundary_gdf: gpd.GeoDataFrame,
    boundary_geom,
    harmonize_to_ids: Callable[[np.ndarray], np.ndarray],
    source_name: str,
    year: int,
) -> pd.DataFrame:
    """Summarize harmonized category areas + proportions for one raster.

    Returns a long/tidy DataFrame with columns:
    - year, source, category, area_km2, proportion
    """

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(str(raster_path))

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        # Determine fill nodata for cropping.
        src_nodata = src.nodata
        if src_nodata is None:
            src_nodata = 0

        # Reproject boundary to raster CRS for cropping.
        boundary_geom_src = (
            gpd.GeoSeries([boundary_geom], crs=boundary_gdf.crs).to_crs(src_crs).iloc[0]
        )

        cropped, cropped_transform = _crop_to_boundary(
            src, boundary_geom_src, fill_nodata=int(src_nodata)
        )

    # Reproject cropped array to DST_CRS with nearest resampling.
    dst_arr, dst_transform = _reproject_array_to_dst(
        cropped,
        src_transform=cropped_transform,
        src_crs=src_crs,
        src_nodata=int(src_nodata),
        dst_crs=DST_CRS,
        dst_res_m=DST_RES_M,
        dst_nodata=0,
    )

    # Apply boundary mask again in DST_CRS to avoid edge artifacts.
    boundary_geom_dst = (
        gpd.GeoSeries([boundary_geom], crs=boundary_gdf.crs).to_crs(DST_CRS).values[0]
    )
    inside = rasterize_geometry_mask(
        boundary_geom_dst, out_shape=dst_arr.shape, transform=dst_transform, all_touched=False
    )
    dst_arr = np.where(inside, dst_arr, 0).astype(np.int16)

    # Harmonize to 0..8 IDs
    ids = harmonize_to_ids(dst_arr).astype(np.int16)

    # Compute areas.
    counts = np.bincount(ids.ravel(), minlength=9).astype(np.int64)  # 0..8
    valid_counts = counts[1:9]
    total_valid = int(valid_counts.sum())
    if total_valid == 0:
        raise ValueError(f"No valid pixels inside boundary for {source_name} {year}")

    px_area = _pixel_area_m2(dst_transform)
    total_area_m2 = total_valid * px_area

    rows: List[Dict] = []
    for cat_id, cat_name in enumerate(HARMONIZED_CATEGORIES, start=1):
        n = int(counts[cat_id])
        area_m2 = n * px_area
        rows.append(
            {
                "year": int(year),
                "source": str(source_name),
                "category": str(cat_name),
                "area_km2": float(area_m2 / 1e6),
                "proportion": float(area_m2 / total_area_m2),
            }
        )

    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], list(HARMONIZED_CATEGORIES), ordered=True)
    df = df.sort_values(["year", "category"]).reset_index(drop=True)
    return df


def assemble_timeseries_from_year_category_areas(
    year_to_category_area_km2: Mapping[int, Mapping[str, float]],
    *,
    source: str,
    categories: Sequence[str] = HARMONIZED_CATEGORIES,
) -> pd.DataFrame:
    """Helper for building a tidy long DataFrame from synthetic summaries.

    This exists mainly for unit tests; the analysis script builds the same schema
    from raster-derived areas.
    """

    rows: List[Dict] = []
    for year in sorted(int(y) for y in year_to_category_area_km2.keys()):
        cat_map = year_to_category_area_km2[year]
        total = float(sum(float(cat_map.get(cat, 0.0)) for cat in categories))
        if total <= 0:
            raise ValueError(f"Year {year} has non-positive total area")

        for cat in categories:
            a = float(cat_map.get(cat, 0.0))
            rows.append(
                {
                    "year": int(year),
                    "source": str(source),
                    "category": str(cat),
                    "area_km2": float(a),
                    "proportion": float(a / total),
                }
            )

    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], list(categories), ordered=True)
    df = df[["year", "source", "category", "area_km2", "proportion"]]
    return df.sort_values(["year", "category"]).reset_index(drop=True)


def _harmonize_vlucd(arr: np.ndarray) -> np.ndarray:
    return harmonize_vlucd_codes_to_ids(arr).ids


def _harmonize_lulc2017(arr: np.ndarray) -> np.ndarray:
    return harmonize_lulc2017_ids(arr).ids


def main() -> None:
    assert_harmonized_schema()

    root = Path(__file__).resolve().parents[2]

    boundary_path = root / "data" / "landuse" / "Boundary_sampling_sites.gpkg"
    vlucd_dir = root / "data" / "landuse" / "MekongDelta_VLUCD_1990_2020"
    lulc_dir = root / "data" / "landuse" / "MekongDelta_LULC_2017_2022"

    out_dir = root / "analysis" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv_vlucd = out_dir / "lulc_studyarea_timeseries_vlucd_1990_2020_latest.csv"
    out_csv_lulc = out_dir / "lulc_studyarea_timeseries_lulc_2017_2022_latest.csv"

    # Load boundary in its native CRS.
    boundary_gdf, boundary_geom = load_study_boundary(
        str(boundary_path), layer="Boundary_sampling_sites"
    )

    years_vlucd = get_years_vlucd()
    years_lulc = get_years_lulc_2017_2022()

    all_v: List[pd.DataFrame] = []
    all_l: List[pd.DataFrame] = []

    for year in years_vlucd:
        raster_path = vlucd_dir / f"VLUCD_L2_30m_{year}.tif"
        print(f"Summarizing {year} (VLUCD)...")
        all_v.append(
            summarize_raster_category_areas(
                raster_path,
                boundary_gdf=boundary_gdf,
                boundary_geom=boundary_geom,
                harmonize_to_ids=_harmonize_vlucd,
                source_name="VLUCD",
                year=year,
            )
        )

    for year in years_lulc:
        raster_path = lulc_dir / f"mekong_{year}_30m.tif"
        print(f"Summarizing {year} (LULC_2017_2022)...")
        all_l.append(
            summarize_raster_category_areas(
                raster_path,
                boundary_gdf=boundary_gdf,
                boundary_geom=boundary_geom,
                harmonize_to_ids=_harmonize_lulc2017,
                source_name="LULC_2017_2022",
                year=year,
            )
        )

    vlucd_df = pd.concat(all_v, ignore_index=True)
    lulc_df = pd.concat(all_l, ignore_index=True)

    for df in (vlucd_df, lulc_df):
        df["category"] = pd.Categorical(df["category"], list(HARMONIZED_CATEGORIES), ordered=True)
        df.sort_values(["year", "category"], inplace=True)

    vlucd_df.to_csv(out_csv_vlucd, index=False)
    lulc_df.to_csv(out_csv_lulc, index=False)
    print(f"Wrote {out_csv_vlucd}")
    print(f"Wrote {out_csv_lulc}")


if __name__ == "__main__":
    main()
