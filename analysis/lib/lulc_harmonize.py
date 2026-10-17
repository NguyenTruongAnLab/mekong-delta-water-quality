"""Harmonization utilities for land-use/land-cover products.

This module defines the shared 8-category scheme used across:
- JAXA VLUCD (1990–2020) Level-2 rasters (codes align to L1 meaning in this project)
- MekongDelta_LULC_2017_2022 rasters that were already reclassified to IDs 1–8

All functions are deterministic and intentionally conservative about nodata handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


HARMONIZED_CATEGORIES: Sequence[str] = (
    "Residential_Urban",
    "Rice_Cultivation",
    "Other_Crops",
    "Aquaculture",
    "Forest",
    "Wetland",
    "Water_Bodies",
    "Others",
)

CATEGORY_TO_ID: Dict[str, int] = {name: i + 1 for i, name in enumerate(HARMONIZED_CATEGORIES)}
ID_TO_CATEGORY: Dict[int, str] = {v: k for k, v in CATEGORY_TO_ID.items()}


# MekongDelta_LULC_2017_2022 is already reclassified to these IDs.
LULC2017_ID_TO_CATEGORY: Dict[int, str] = {
    1: "Residential_Urban",
    2: "Rice_Cultivation",
    3: "Other_Crops",
    4: "Aquaculture",
    5: "Forest",
    6: "Wetland",
    7: "Water_Bodies",
    8: "Others",
}


# VLUCD mapping used in this repository (see data/landuse/MekongDelta_VLUCD_1990_2020/README_VLUCD.md)
VLUCD_CODE_TO_CATEGORY: Dict[int, str] = {
    1: "Residential_Urban",
    2: "Residential_Urban",
    3: "Rice_Cultivation",
    4: "Other_Crops",
    5: "Other_Crops",
    6: "Other_Crops",
    19: "Aquaculture",
    10: "Forest",
    11: "Forest",
    12: "Forest",
    14: "Forest",
    20: "Forest",
    15: "Wetland",
    16: "Wetland",
    18: "Water_Bodies",
    7: "Others",
    8: "Others",
    9: "Others",
}


@dataclass(frozen=True)
class HarmonizeResult:
    """Structured output for harmonization (handy for debugging/tests)."""

    ids: np.ndarray
    nodata_value: int


def _as_set(values: Iterable[int]) -> set[int]:
    return set(int(v) for v in values)


def harmonize_vlucd_codes_to_ids(
    arr: np.ndarray,
    *,
    nodata_values: Sequence[int] = (0, 255),
    unknown_to_others: bool = True,
    out_nodata: int = 0,
) -> HarmonizeResult:
    """Map VLUCD codes to harmonized 1–8 category IDs.

    Parameters
    ----------
    arr:
        2D numpy array of VLUCD codes.
    nodata_values:
        Values to treat as nodata.
    unknown_to_others:
        If True, any positive code not in the mapping is mapped to "Others".
        If False, unknown codes become nodata.
    out_nodata:
        Output nodata value.

    Returns
    -------
    HarmonizeResult
        ids is an int16 array with values in {0..8}.
    """

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")

    nodata_set = _as_set(nodata_values)
    out = np.full(arr.shape, out_nodata, dtype=np.int16)

    # Valid data are values not in nodata_set.
    valid_mask = np.ones(arr.shape, dtype=bool)
    for nd in nodata_set:
        valid_mask &= arr != nd

    if not np.any(valid_mask):
        return HarmonizeResult(ids=out, nodata_value=out_nodata)

    # Apply explicit mappings.
    for code, cat in VLUCD_CODE_TO_CATEGORY.items():
        cat_id = CATEGORY_TO_ID[cat]
        out[(arr == code) & valid_mask] = cat_id

    # Unknown codes.
    if unknown_to_others:
        others_id = CATEGORY_TO_ID["Others"]
        unknown_mask = valid_mask & (out == out_nodata) & (arr > 0)
        out[unknown_mask] = others_id

    return HarmonizeResult(ids=out, nodata_value=out_nodata)


def harmonize_lulc2017_ids(
    arr: np.ndarray,
    *,
    nodata_values: Sequence[int] = (0, 255),
    out_nodata: int = 0,
) -> HarmonizeResult:
    """Validate/passthrough for MekongDelta_LULC_2017_2022 IDs.

    The product is expected to already be in 1..8.
    Any values outside 1..8 and not nodata are coerced to nodata.
    """

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")

    out = np.full(arr.shape, out_nodata, dtype=np.int16)

    # Passthrough 1..8
    for v in range(1, 9):
        out[arr == v] = v

    # Force nodata values to out_nodata
    for nd in _as_set(nodata_values):
        out[arr == nd] = out_nodata

    # Everything else remains out_nodata
    return HarmonizeResult(ids=out, nodata_value=out_nodata)


def assert_harmonized_schema() -> None:
    """Raise if mappings do not represent the agreed 8-category scheme."""

    if list(HARMONIZED_CATEGORIES) != list(CATEGORY_TO_ID.keys()):
        raise AssertionError("CATEGORY_TO_ID key order mismatch")

    if set(LULC2017_ID_TO_CATEGORY.keys()) != set(range(1, 9)):
        raise AssertionError("LULC2017_ID_TO_CATEGORY must define IDs 1..8")

    if set(LULC2017_ID_TO_CATEGORY.values()) != set(HARMONIZED_CATEGORIES):
        raise AssertionError("LULC2017 categories must match harmonized categories")

    # Ensure VLUCD mapping only maps to allowed categories.
    bad = {cat for cat in VLUCD_CODE_TO_CATEGORY.values() if cat not in CATEGORY_TO_ID}
    if bad:
        raise AssertionError(f"VLUCD mapping has unknown category names: {sorted(bad)}")
