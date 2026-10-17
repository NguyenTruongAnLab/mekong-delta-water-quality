"""LULC overlap consistency check (2017-2020).

This script:
- Loads the study boundary (Hau + Tien study area) from a GeoPackage.
- Summarizes category composition (area + proportions) for two products:
  1) VLUCD 1990-2020 (using project mapping for L2/L1 codes)
  2) MekongDelta_LULC_2017_2022 (already reclassified to IDs 1-8)
- Reprojects data to EPSG:32648 (UTM Zone 48N) before computing areas.
- Computes differences in category proportions over 2017-2020.
- Writes deterministic artifacts to analysis/analysis_outputs/.

Outputs:
- analysis/analysis_outputs/lulc_consistency_2017_2020_latest.csv
- analysis/analysis_outputs/lulc_consistency_2017_2020_validation_summary_latest.csv
- analysis/analysis_outputs/lulc_consistency_2017_2020_validation_latest.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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


def summarize_raster_category_proportions(
    raster_path: Path,
    *,
    boundary_gdf: gpd.GeoDataFrame,
    boundary_geom,
    harmonize: Callable[[np.ndarray], np.ndarray],
    product_name: str,
    year: int,
) -> pd.DataFrame:
    """Summarize harmonized category areas + proportions for one raster."""

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
            # Conservative default matching existing scripts in this repo.
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

    # Harmonize.
    ids = harmonize(dst_arr).astype(np.int16)

    # Compute areas.
    counts = np.bincount(ids.ravel(), minlength=9).astype(np.int64)  # 0..8
    valid_counts = counts[1:9]
    total_valid = int(valid_counts.sum())
    if total_valid == 0:
        raise ValueError(f"No valid pixels inside boundary for {product_name} {year}")

    px_area = _pixel_area_m2(dst_transform)

    rows: List[Dict] = []
    for cat_id, cat_name in enumerate(HARMONIZED_CATEGORIES, start=1):
        n = int(counts[cat_id])
        area_m2 = n * px_area
        rows.append(
            {
                "year": year,
                "product": product_name,
                "category": cat_name,
                "pixel_count": n,
                "pixel_area_m2": px_area,
                "area_m2": area_m2,
                "area_km2": area_m2 / 1e6,
                "proportion": area_m2 / (total_valid * px_area),
            }
        )

    return pd.DataFrame(rows)


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

    out_csv = out_dir / "lulc_consistency_2017_2020_latest.csv"
    out_summary_csv = out_dir / "lulc_consistency_2017_2020_validation_summary_latest.csv"
    out_md = out_dir / "lulc_consistency_2017_2020_validation_latest.md"

    # Load boundary in its native CRS.
    boundary_gdf, boundary_geom = load_study_boundary(
        str(boundary_path), layer="Boundary_sampling_sites"
    )

    years = [2017, 2018, 2019, 2020]

    all_long: List[pd.DataFrame] = []

    for year in years:
        vlucd_raster = vlucd_dir / f"VLUCD_L2_30m_{year}.tif"
        lulc_raster = lulc_dir / f"mekong_{year}_30m.tif"

        print(f"Summarizing {year} (VLUCD)...")
        all_long.append(
            summarize_raster_category_proportions(
                vlucd_raster,
                boundary_gdf=boundary_gdf,
                boundary_geom=boundary_geom,
                harmonize=_harmonize_vlucd,
                product_name="VLUCD",
                year=year,
            )
        )

        print(f"Summarizing {year} (LULC_2017_2022)...")
        all_long.append(
            summarize_raster_category_proportions(
                lulc_raster,
                boundary_gdf=boundary_gdf,
                boundary_geom=boundary_geom,
                harmonize=_harmonize_lulc2017,
                product_name="LULC_2017_2022",
                year=year,
            )
        )

    long_df = pd.concat(all_long, ignore_index=True)

    # Diff table (one row per year/category)
    props = long_df.pivot_table(
        index=["year", "category"], columns="product", values="proportion", aggfunc="mean"
    ).reset_index()

    # Ensure columns exist
    if "VLUCD" not in props.columns or "LULC_2017_2022" not in props.columns:
        raise ValueError("Missing product columns after pivot; check product names")

    props = props.rename(
        columns={"VLUCD": "prop_vlucd", "LULC_2017_2022": "prop_lulc2017"}
    )
    props["diff_prop"] = props["prop_lulc2017"] - props["prop_vlucd"]
    props["abs_diff_prop"] = props["diff_prop"].abs()

    # Attach km2 areas as well
    areas = long_df.pivot_table(
        index=["year", "category"], columns="product", values="area_km2", aggfunc="mean"
    ).reset_index()
    areas = areas.rename(
        columns={"VLUCD": "area_km2_vlucd", "LULC_2017_2022": "area_km2_lulc2017"}
    )

    out_df = props.merge(areas, on=["year", "category"], how="left")

    # Additional diagnostics: relative area mismatch
    out_df["abs_area_diff_km2"] = (out_df["area_km2_lulc2017"] - out_df["area_km2_vlucd"]).abs()
    out_df["mean_area_km2"] = 0.5 * (out_df["area_km2_lulc2017"] + out_df["area_km2_vlucd"]).astype(float)
    out_df["rel_area_diff"] = out_df["abs_area_diff_km2"] / out_df["mean_area_km2"].replace({0.0: np.nan})

    # Validation thresholds (used to decide whether splicing is defensible)
    PP_THRESH = 5.0  # percentage points
    REL_AREA_THRESH = 0.10  # 10% relative difference

    out_df["abs_diff_pp"] = out_df["abs_diff_prop"] * 100.0
    out_df["pass_prop_pp"] = out_df["abs_diff_pp"] <= PP_THRESH
    out_df["pass_rel_area"] = out_df["rel_area_diff"] <= REL_AREA_THRESH
    out_df["pass_both"] = out_df["pass_prop_pp"] & out_df["pass_rel_area"]

    # Stable ordering
    out_df["category"] = pd.Categorical(out_df["category"], list(HARMONIZED_CATEGORIES), ordered=True)
    out_df = out_df.sort_values(["year", "category"]).reset_index(drop=True)

    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Summary table: mean/max mismatch per category + overall pass/fail
    summary = (
        out_df.groupby("category", observed=False)
        .agg(
            mean_abs_diff_pp=("abs_diff_pp", "mean"),
            max_abs_diff_pp=("abs_diff_pp", "max"),
            mean_rel_area_diff=("rel_area_diff", "mean"),
            max_rel_area_diff=("rel_area_diff", "max"),
            pass_all_years=("pass_both", "all"),
        )
        .reset_index()
    )
    summary["mean_rel_area_diff_pct"] = summary["mean_rel_area_diff"] * 100.0
    summary["max_rel_area_diff_pct"] = summary["max_rel_area_diff"] * 100.0
    summary = summary.sort_values(["pass_all_years", "max_abs_diff_pp"], ascending=[True, False])

    summary.to_csv(out_summary_csv, index=False)
    print(f"Wrote {out_summary_csv}")

    overall_pass = bool(summary["pass_all_years"].all())
    worst = summary.iloc[0]
    md_lines = [
        "# LULC overlap validation (2017-2020)",
        "",
        f"Thresholds: |diff| <= {PP_THRESH:.1f} pp AND relative area diff <= {REL_AREA_THRESH*100:.0f}%.",
        "",
        f"Overall result: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "Worst (by max |diff|):",
        f"- Category: {worst['category']}",
        f"- Max |diff| (pp): {worst['max_abs_diff_pp']:.1f}",
        f"- Max relative area diff (%): {worst['max_rel_area_diff_pct']:.1f}",
        "",
        "If FAIL, do not splice VLUCD with LULC_2017_2022 into a single continuous area/proportion series.",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
