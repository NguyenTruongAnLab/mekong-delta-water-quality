"""Study area boundary loading and raster masking helpers.

Phase 1 uses a single shared boundary representing the Hau + Tien study area.

Key design choices:
- Boundary is loaded from a GeoPackage layer.
- Masking is done via rasterization so it works equally well for in-memory arrays
  (including reprojected rasters).
"""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

import rasterio
from rasterio.features import rasterize


DEFAULT_BOUNDARY_GPKG = "data/landuse/Boundary_sampling_sites.gpkg"
DEFAULT_BOUNDARY_LAYER = "Boundary_sampling_sites"


def _make_valid(geom: BaseGeometry) -> BaseGeometry:
    """Return a valid geometry (best-effort) without changing CRS."""

    if geom is None:
        raise ValueError("Geometry is None")

    if geom.is_valid:
        return geom

    # Shapely 2.x: make_valid may be available.
    try:
        from shapely import make_valid  # type: ignore

        fixed = make_valid(geom)
        if fixed.is_valid:
            return fixed
    except Exception:
        pass

    # Fallback: buffer(0) trick.
    try:
        fixed = geom.buffer(0)
        return fixed
    except Exception:
        return geom


def load_study_boundary(
    gpkg_path: str = DEFAULT_BOUNDARY_GPKG,
    *,
    layer: str = DEFAULT_BOUNDARY_LAYER,
    to_crs: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, BaseGeometry]:
    """Load the study boundary layer and return (gdf, unified_geometry).

    Parameters
    ----------
    gpkg_path:
        Path to GeoPackage.
    layer:
        Layer name.
    to_crs:
        If provided, reproject the GeoDataFrame before unifying.

    Returns
    -------
    (GeoDataFrame, geometry)
        geometry is a unary_union of all features in the layer.
    """

    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.empty:
        raise ValueError(f"Boundary layer is empty: {gpkg_path}::{layer}")

    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)

    geom = unary_union(gdf.geometry.values)
    geom = _make_valid(geom)

    return gdf, geom


def rasterize_geometry_mask(
    geometry: BaseGeometry,
    *,
    out_shape: Tuple[int, int],
    transform: rasterio.Affine,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize geometry into a boolean mask (True inside geometry)."""

    if geometry is None:
        raise ValueError("geometry is None")

    geom = _make_valid(geometry)

    burned = rasterize(
        [(geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    # `rasterio.features.rasterize` returns an ndarray; cast keeps type-checkers happy.
    burned_arr = np.asarray(burned)
    return burned_arr.astype(bool)


def apply_boolean_mask(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    nodata: int = 0,
) -> np.ndarray:
    """Apply mask to a 2D array, setting outside to nodata."""

    if data.shape != mask.shape:
        raise ValueError("data and mask shapes differ")

    out = data.copy()
    out[~mask] = nodata
    return out
