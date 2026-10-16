#!/usr/bin/env python3
"""
GLORYS Quality-Control (QC) Policy Module v1
(Reusable Preprocessing and Validation Module)

Provides quality-control checks on raw int16 GLORYS ocean potential temperature (thetao) fields:
- Encoding verification against policy
- Static wet mask loading and coordinate alignment verification
- Coordinate-based static exclusion lookup (e.g. Columns A & B pit drift)
- Strict bitmask flag assignment (uint8)
- Valid mask generation: (raw not fill) & static_wet & ~excluded
- Tracking of unexpected violations: FAIL on unexpected zero-codes, near-floor, dry finite, or mask drift
- Warning on above_valid_max while preserving cell validity
- Per-level extremes among valid cells
- Preserves raw input arrays without in-place mutation or imputation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import h5netcdf
import numpy as np


# =============================================================================
# QC Bitmask Flag Constants (uint8)
# =============================================================================

FLAG_FILL_IN_STATIC_WET: np.uint8 = np.uint8(1)          # Bit 0: raw is fill, static mask wet
FLAG_STATIC_EXCLUDED: np.uint8 = np.uint8(2)             # Bit 1: in static exclusion list
FLAG_ZERO_CODE: np.uint8 = np.uint8(4)                   # Bit 2: raw == zero_code_raw (-28671)
FLAG_NEAR_FLOOR: np.uint8 = np.uint8(8)                  # Bit 3: raw in [valid_min, valid_min + 50]
FLAG_ABOVE_VALID_MAX: np.uint8 = np.uint8(16)            # Bit 4: raw > valid_max (21306)
FLAG_FINITE_IN_STATIC_DRY: np.uint8 = np.uint8(32)       # Bit 5: raw not fill, static mask dry
FLAG_MASK_DIFFERS_FROM_REF: np.uint8 = np.uint8(64)      # Bit 6: finite mask differs from reference


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QCResult:
    """Encapsulates the result of applying QC policy v1 to a 3D thetao field."""
    flags: np.ndarray                                    # uint8 array (nz, ny, nx)
    valid: np.ndarray                                    # bool array (nz, ny, nx)
    status: str                                          # "PASS" | "WARN" | "FAIL"
    counts: Dict[str, int]                               # Total count per rule & valid
    counts_by_level: Dict[str, List[int]]                # Counts broken down by level
    unexpected: List[Dict[str, Any]]                     # List of unexpected violating cells
    extremes_by_level: List[Dict[str, Any]]              # Max/min decoded value per level
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary results to a serializable dictionary."""
        return {
            "status": self.status,
            "counts": self.counts,
            "counts_by_level": self.counts_by_level,
            "unexpected_count": len(self.unexpected),
            "unexpected": self.unexpected,
            "extremes_by_level": self.extremes_by_level,
            "metadata": self.metadata,
        }


# =============================================================================
# Policy and Coordinate Helpers
# =============================================================================

def load_policy(path: str) -> Dict[str, Any]:
    """Load and validate the QC policy JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"QC policy file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    # Basic schema check
    required_keys = ["policy_version", "encoding", "levels_used", "zero_code_raw", "static_exclusions"]
    for k in required_keys:
        if k not in policy:
            raise KeyError(f"Missing required key '{k}' in QC policy file {path}")

    return policy


def read_raw_day(
    path: str,
    day_index: int = 0,
    policy: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Read a single day's raw int16 thetao slice (nz, ny, nx) with no masking or scaling.
    
    Verifies the file's encoding attributes match the policy. If any attribute mismatches,
    raises a ValueError and halts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw NetCDF file not found: {path}")

    with h5netcdf.File(path, mode="r") as f:
        if "thetao" not in f.variables:
            raise KeyError(f"Variable 'thetao' not found in {path}")

        var = f.variables["thetao"]
        attrs = var.attrs

        # Verify encoding against policy if provided
        if policy is not None and "encoding" in policy:
            enc = policy["encoding"]
            scale_file = float(attrs.get("scale_factor", 0.0))
            offset_file = float(attrs.get("add_offset", 0.0))
            fill_file = int(attrs.get("_FillValue", 0))
            vmin_file = int(attrs.get("valid_min", 0))
            vmax_file = int(attrs.get("valid_max", 0))

            mismatches = []
            if not np.isclose(scale_file, float(enc["scale_factor"]), rtol=1e-7, atol=1e-12):
                mismatches.append(f"scale_factor mismatch: file={scale_file}, policy={enc['scale_factor']}")
            if not np.isclose(offset_file, float(enc["add_offset"]), rtol=1e-5, atol=1e-6):
                mismatches.append(f"add_offset mismatch: file={offset_file}, policy={enc['add_offset']}")
            if fill_file != int(enc["fill_raw"]):
                mismatches.append(f"_FillValue mismatch: file={fill_file}, policy={enc['fill_raw']}")
            if vmin_file != int(enc["valid_min_raw"]):
                mismatches.append(f"valid_min mismatch: file={vmin_file}, policy={enc['valid_min_raw']}")
            if vmax_file != int(enc["valid_max_raw"]):
                mismatches.append(f"valid_max mismatch: file={vmax_file}, policy={enc['valid_max_raw']}")

            if mismatches:
                raise ValueError(f"File encoding verification failed for {path}:\n" + "\n".join(mismatches))

        levels_used = policy.get("levels_used", 36) if policy else 36

        # Slice raw day data (without mask or scale)
        var_ndim = len(var.shape)
        if var_ndim == 4:
            # (time, depth, lat, lon)
            if day_index >= var.shape[0]:
                raise IndexError(f"day_index {day_index} out of bounds for time dimension of size {var.shape[0]}")
            raw_3d = np.array(var[day_index, :levels_used, :, :], dtype=np.int16)
        elif var_ndim == 3:
            # (depth, lat, lon)
            raw_3d = np.array(var[:levels_used, :, :], dtype=np.int16)
        else:
            raise ValueError(f"Unexpected thetao dimensions {var.shape} in {path}")

    return raw_3d


def load_coordinates(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read latitude, longitude, and depth coordinates from a NetCDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coordinate file not found: {path}")

    with h5netcdf.File(path, mode="r") as f:
        lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        lons = np.array(f.variables["longitude"][:], dtype=np.float64)
        depths = np.array(f.variables["depth"][:], dtype=np.float64)

    return lats, lons, depths


def load_static_wet_mask(
    path: str,
    ref_lats: Optional[np.ndarray] = None,
    ref_lons: Optional[np.ndarray] = None,
    ref_depths: Optional[np.ndarray] = None,
    levels_used: int = 36,
    tol_latlon: float = 1e-4,
    tol_depth: float = 1e-3,
) -> np.ndarray:
    """Load the static 3D wet mask (mask == 1) for the first levels_used levels.
    
    Verifies coordinate alignment against reference coordinates if provided:
    lat/lon <= 1e-4 deg, depth <= 1e-3 m, else raises ValueError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Static mask file not found: {path}")

    with h5netcdf.File(path, mode="r") as f:
        if "mask" not in f.variables:
            raise KeyError(f"Variable 'mask' missing in static file {path}")

        stat_lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        stat_lons = np.array(f.variables["longitude"][:], dtype=np.float64)
        stat_depths = np.array(f.variables["depth"][:], dtype=np.float64)
        stat_mask = np.array(f.variables["mask"][:levels_used, :, :], dtype=np.int8)

    # Check alignment against reference coordinates
    if ref_lats is not None:
        diff_lat = np.max(np.abs(stat_lats - ref_lats))
        if diff_lat > tol_latlon:
            raise ValueError(f"Static latitude coordinate alignment failed: max diff {diff_lat} > {tol_latlon}")

    if ref_lons is not None:
        diff_lon = np.max(np.abs(stat_lons - ref_lons))
        if diff_lon > tol_latlon:
            raise ValueError(f"Static longitude coordinate alignment failed: max diff {diff_lon} > {tol_latlon}")

    if ref_depths is not None:
        diff_depth = np.max(np.abs(stat_depths[:levels_used] - ref_depths[:levels_used]))
        if diff_depth > tol_depth:
            raise ValueError(f"Static depth coordinate alignment failed: max diff {diff_depth} > {tol_depth}")

    static_wet = (stat_mask == 1)
    return static_wet


def locate_exclusions(
    lats: np.ndarray,
    lons: np.ndarray,
    policy: Dict[str, Any],
) -> np.ndarray:
    """Locate static exclusions by coordinate lookup, creating a 3D boolean mask (nz, ny, nx).
    
    HALTS and raises ValueError if any exclusion coordinate is not found within tolerance.
    """
    levels_used = policy.get("levels_used", 36)
    tol = policy.get("coordinate_tolerance_deg", 0.001)
    exclusions = policy.get("static_exclusions", [])

    ny = len(lats)
    nx = len(lons)
    exclusion_mask = np.zeros((levels_used, ny, nx), dtype=bool)

    for entry in exclusions:
        target_lat = float(entry["lat"])
        target_lon = float(entry["lon"])
        levels = entry["levels"]
        name = entry.get("name", f"({target_lat},{target_lon})")

        dist_lat = np.abs(lats - target_lat)
        dist_lon = np.abs(lons - target_lon)

        min_dist_lat = float(np.min(dist_lat))
        min_dist_lon = float(np.min(dist_lon))

        if min_dist_lat > tol or min_dist_lon > tol:
            raise ValueError(
                f"Static exclusion '{name}' coordinate ({target_lat}, {target_lon}) not found within "
                f"tolerance {tol} deg! Nearest found was ({lats[np.argmin(dist_lat)]}, {lons[np.argmin(dist_lon)]}) "
                f"with lat diff={min_dist_lat}, lon diff={min_dist_lon}"
            )

        r_idx = int(np.argmin(dist_lat))
        c_idx = int(np.argmin(dist_lon))

        for lvl in levels:
            if 0 <= lvl < levels_used:
                exclusion_mask[lvl, r_idx, c_idx] = True

    return exclusion_mask


# =============================================================================
# QC Application
# =============================================================================

def apply_qc(
    raw: np.ndarray,
    static_wet: np.ndarray,
    exclusion_mask: np.ndarray,
    policy: Dict[str, Any],
    reference_finite: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
) -> QCResult:
    """Apply GLORYS Quality-Control Policy v1 to a 3D thetao field.
    
    Parameters:
    - raw: int16 array of shape (nz, ny, nx) with raw encoded temperatures
    - static_wet: boolean array of shape (nz, ny, nx), True where static mask == 1 (sea)
    - exclusion_mask: boolean array of shape (nz, ny, nx), True for statically excluded cells
    - policy: QC policy dictionary
    - reference_finite: optional boolean array of shape (nz, ny, nx) holding reference finite mask
    - lats, lons, depths: optional coordinate arrays for logging coordinates
    
    Returns:
    - QCResult containing flags, valid mask, counts, unexpected list, status, and extremes
    """
    # Defensive check: ensure inputs are not mutated
    if raw.shape != static_wet.shape or raw.shape != exclusion_mask.shape:
        raise ValueError(f"Shape mismatch: raw {raw.shape}, static_wet {static_wet.shape}, exclusion {exclusion_mask.shape}")

    nz, ny, nx = raw.shape
    enc = policy["encoding"]
    fill_raw = int(enc["fill_raw"])
    vmin_raw = int(enc["valid_min_raw"])
    vmax_raw = int(enc["valid_max_raw"])
    scale_factor = float(enc["scale_factor"])
    add_offset = float(enc["add_offset"])

    zero_code_raw = int(policy["zero_code_raw"])
    near_floor_margin = int(policy.get("near_floor_margin_codes", 50))
    near_floor_max = vmin_raw + near_floor_margin  # -32766 + 50 = -32716

    # -------------------------------------------------------------------------
    # Bitmask Flag Computation
    # -------------------------------------------------------------------------
    flags = np.zeros((nz, ny, nx), dtype=np.uint8)

    # 1. Fill in static wet: (raw == fill) & static_wet
    is_fill = (raw == fill_raw)
    is_finite = ~is_fill

    mask_fill_in_static_wet = is_fill & static_wet
    flags[mask_fill_in_static_wet] |= FLAG_FILL_IN_STATIC_WET

    # 2. Static excluded
    flags[exclusion_mask] |= FLAG_STATIC_EXCLUDED

    # 3. Zero code: raw == zero_code_raw (only finite cells)
    mask_zero_code = (raw == zero_code_raw) & is_finite
    flags[mask_zero_code] |= FLAG_ZERO_CODE

    # 4. Near floor: raw in [vmin_raw, vmin_raw + margin] (only finite cells; fill is excluded)
    mask_near_floor = (raw >= vmin_raw) & (raw <= near_floor_max) & is_finite
    flags[mask_near_floor] |= FLAG_NEAR_FLOOR

    # 5. Above valid max: raw > vmax_raw (only finite cells)
    mask_above_vmax = (raw > vmax_raw) & is_finite
    flags[mask_above_vmax] |= FLAG_ABOVE_VALID_MAX

    # 6. Finite in static dry: is_finite & ~static_wet
    mask_finite_in_dry = is_finite & (~static_wet)
    flags[mask_finite_in_dry] |= FLAG_FINITE_IN_STATIC_DRY

    # 7. Mask differs from reference (if reference_finite provided)
    if reference_finite is not None:
        mask_diff_ref = (is_finite != reference_finite)
        flags[mask_diff_ref] |= FLAG_MASK_DIFFERS_FROM_REF
    else:
        mask_diff_ref = np.zeros((nz, ny, nx), dtype=bool)

    # -------------------------------------------------------------------------
    # Valid Mask Computation: (raw not fill) & static_wet & ~excluded
    # -------------------------------------------------------------------------
    valid = is_finite & static_wet & (~exclusion_mask)

    # -------------------------------------------------------------------------
    # Counts (Total and Per Level)
    # -------------------------------------------------------------------------
    rule_keys = [
        ("fill_in_static_wet", FLAG_FILL_IN_STATIC_WET),
        ("static_excluded", FLAG_STATIC_EXCLUDED),
        ("zero_code", FLAG_ZERO_CODE),
        ("near_floor", FLAG_NEAR_FLOOR),
        ("above_valid_max", FLAG_ABOVE_VALID_MAX),
        ("finite_in_static_dry", FLAG_FINITE_IN_STATIC_DRY),
        ("mask_differs_from_reference", FLAG_MASK_DIFFERS_FROM_REF),
    ]

    counts: Dict[str, int] = {}
    counts_by_level: Dict[str, List[int]] = {}

    for name, bit in rule_keys:
        bit_match = (flags & bit) != 0
        counts[name] = int(np.sum(bit_match))
        counts_by_level[name] = [int(np.sum(bit_match[k])) for k in range(nz)]

    counts["valid_cells"] = int(np.sum(valid))
    counts_by_level["valid_cells"] = [int(np.sum(valid[k])) for k in range(nz)]

    # Specific sub-counts for zero_code and near_floor outside exclusion
    zero_code_unexcluded = mask_zero_code & (~exclusion_mask)
    near_floor_unexcluded = mask_near_floor & (~exclusion_mask)
    counts["zero_code_unexcluded"] = int(np.sum(zero_code_unexcluded))
    counts["near_floor_unexcluded"] = int(np.sum(near_floor_unexcluded))

    # -------------------------------------------------------------------------
    # Unexpected Cells (Violations Causing FAIL)
    # -------------------------------------------------------------------------
    unexpected: List[Dict[str, Any]] = []

    def _add_unexpected(mask: np.ndarray, rule_name: str):
        if np.any(mask):
            k_idx, r_idx, c_idx = np.where(mask)
            for k, r, c in zip(k_idx, r_idx, c_idx):
                lat_val = float(lats[r]) if lats is not None else None
                lon_val = float(lons[c]) if lons is not None else None
                raw_val = int(raw[k, r, c])
                unexpected.append({
                    "level": int(k),
                    "depth_m": float(depths[k]) if depths is not None else None,
                    "lat_idx": int(r),
                    "lon_idx": int(c),
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "raw": raw_val,
                    "decoded_C": float(raw_val * scale_factor + add_offset) if raw_val != fill_raw else None,
                    "rule": rule_name,
                })

    _add_unexpected(zero_code_unexcluded, "ZERO_CODE_OUTSIDE_EXCLUSION")
    _add_unexpected(near_floor_unexcluded, "NEAR_FLOOR_OUTSIDE_EXCLUSION")
    _add_unexpected(mask_finite_in_dry, "FINITE_IN_STATIC_DRY")
    _add_unexpected(mask_diff_ref, "MASK_DIFFERS_FROM_REFERENCE")

    # -------------------------------------------------------------------------
    # Status Determination: FAIL if unexpected > 0, else WARN if above_vmax > 0, else PASS
    # -------------------------------------------------------------------------
    if len(unexpected) > 0:
        status = "FAIL"
    elif counts["above_valid_max"] > 0:
        status = "WARN"
    else:
        status = "PASS"

    # -------------------------------------------------------------------------
    # Extremes per Level Among Valid Cells
    # -------------------------------------------------------------------------
    extremes_by_level: List[Dict[str, Any]] = []
    for k in range(nz):
        valid_k = valid[k]
        n_valid_k = int(np.sum(valid_k))
        depth_val = float(depths[k]) if depths is not None else None

        if n_valid_k > 0:
            raw_valid = raw[k][valid_k]
            max_raw = int(np.max(raw_valid))
            min_raw = int(np.min(raw_valid))

            # Coordinates of max and min
            max_r, max_c = np.where(valid_k & (raw[k] == max_raw))
            min_r, min_c = np.where(valid_k & (raw[k] == min_raw))

            max_r_idx = int(max_r[0])
            max_c_idx = int(max_c[0])
            min_r_idx = int(min_r[0])
            min_c_idx = int(min_c[0])

            max_lat = float(lats[max_r_idx]) if lats is not None else None
            max_lon = float(lons[max_c_idx]) if lons is not None else None
            min_lat = float(lats[min_r_idx]) if lats is not None else None
            min_lon = float(lons[min_c_idx]) if lons is not None else None

            extremes_by_level.append({
                "level": k,
                "depth_m": depth_val,
                "n_valid": n_valid_k,
                "max_raw": max_raw,
                "max_decoded_C": float(max_raw * scale_factor + add_offset),
                "max_lat_idx": max_r_idx,
                "max_lon_idx": max_c_idx,
                "max_latitude": max_lat,
                "max_longitude": max_lon,
                "min_raw": min_raw,
                "min_decoded_C": float(min_raw * scale_factor + add_offset),
                "min_lat_idx": min_r_idx,
                "min_lon_idx": min_c_idx,
                "min_latitude": min_lat,
                "min_longitude": min_lon,
            })
        else:
            extremes_by_level.append({
                "level": k,
                "depth_m": depth_val,
                "n_valid": 0,
                "max_raw": None,
                "max_decoded_C": None,
                "max_lat_idx": None,
                "max_lon_idx": None,
                "max_latitude": None,
                "max_longitude": None,
                "min_raw": None,
                "min_decoded_C": None,
                "min_lat_idx": None,
                "min_lon_idx": None,
                "min_latitude": None,
                "min_longitude": None,
            })

    return QCResult(
        flags=flags,
        valid=valid,
        status=status,
        counts=counts,
        counts_by_level=counts_by_level,
        unexpected=unexpected,
        extremes_by_level=extremes_by_level,
    )
