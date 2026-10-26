#!/usr/bin/env python3
"""
GLORYS Vertical Interpolation Consequence Experiment
(Diagnostic Only; Comparative Evidence Gathering; Read-Only on Raw Data)

Compares candidate vertical interpolation methods for GLORYS thetao from native
depth levels (36 levels, ~0.49 m to ~1062 m) to project target depths (15 depths:
0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m):
  - Horizontal operator: Method A (area-weighted mean over QC-valid cells)
  - Provisional validity threshold: f_T(level) >= tauT (tauT = 1.0 and 0.5)
  - V1: linear interpolation in depth between the two bracketing levels.
        Valid only if BOTH bracketing levels have f_T >= tauT.
  - V2: PCHIP (monotone cubic Hermite) through the valid contiguous prefix of
        levels of the column (levels 0..K with f_T >= tauT), evaluated at the
        target depth if K reaches the upper bracketing level (deeper index).
  - Depth 0 rules:
      V0a: value of level 0 (~0.49 m) used as the 0 m value.
      V0b: linear extrapolation from levels 0 and 1 to depth 0.

Outputs generated in outputs/diagnostics/glorys_vertical_experiment/:
  - bracket_table.csv
  - validity_by_depth.csv
  - v2_minus_v1.csv
  - top_v2_v1_cells.csv
  - depth0_alternatives.csv
  - rmse_vs_target_depth.png
  - valid_cells_vs_target_depth.png
  - day0_map_v0b_minus_v0a.png
  - glorys_vertical_experiment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocessing.glorys_qc import (
    apply_qc,
    load_coordinates,
    load_policy,
    load_static_wet_mask,
    locate_exclusions,
    read_raw_day,
)
from scripts.diagnostics.glorys_regrid_experiment import (
    compute_overlap_weights_1d,
    compute_overlap_weights_2d,
    regrid_method_A,
    DEFAULT_THETAO_PATH,
    DEFAULT_STATIC_PATH,
    DEFAULT_POLICY_PATH,
    EXPECTED_SHA256_THETAO,
    EXPECTED_SHA256_STATIC,
    TARGET_LAT_START,
    TARGET_LON_START,
    TARGET_N_LAT,
    TARGET_N_LON,
    TARGET_STEP,
    NATIVE_STEP,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "diagnostics", "glorys_vertical_experiment"
)

TARGET_DEPTHS: List[float] = [
    0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
]


# =============================================================================
# File Integrity & Metadata Helpers
# =============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file without loading the entire file into RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """Retrieve SHA256, size, and modification timestamp for a file."""
    stat = os.stat(filepath)
    sha256_val = compute_file_sha256(filepath)
    return {
        "path": os.path.abspath(filepath),
        "sha256": sha256_val,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_raw_data_integrity(
    thetao_path: str = DEFAULT_THETAO_PATH,
    static_path: str = DEFAULT_STATIC_PATH,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strictly verify SHA256 hashes of raw input data before execution."""
    meta_thetao = get_file_metadata(thetao_path)
    meta_static = get_file_metadata(static_path)

    if meta_thetao["sha256"] != EXPECTED_SHA256_THETAO:
        raise ValueError(
            f"Pre-check failed: thetao SHA256 {meta_thetao['sha256']} != expected {EXPECTED_SHA256_THETAO}"
        )
    if meta_static["sha256"] != EXPECTED_SHA256_STATIC:
        raise ValueError(
            f"Pre-check failed: static SHA256 {meta_static['sha256']} != expected {EXPECTED_SHA256_STATIC}"
        )

    return meta_thetao, meta_static


# =============================================================================
# Bracket Search & Table Generation
# =============================================================================

def bracket_search(
    level_depths: np.ndarray,
    target_depth: float,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Find bracketing levels, interpolation weight, and distance to nearest level.
    
    Parameters:
        level_depths: 1D array of native level depths (strictly increasing).
        target_depth: Target depth (m).
        tol: Tolerance for exact equality.
        
    Returns:
        Dict with keys:
            - target_depth_m: float
            - k0: int, index of shallower level (or 0 for depth 0)
            - k1: int, index of deeper level (or 1 for depth 0)
            - depth_k0_m: float
            - depth_k1_m: float
            - weight_k1: float, linear interpolation weight on level k1: w = (z_t - z0) / (z1 - z0)
            - dist_nearest_m: float, distance to nearest level
            - nearest_k: int, index of nearest level
            - is_exact: bool, whether target depth matches a level depth exactly
            - is_depth0: bool, whether this is the depth 0 surface special case
    """
    level_depths = np.asarray(level_depths, dtype=np.float64)
    nz = len(level_depths)

    # Depth 0 special case (shallower than first level)
    if target_depth < level_depths[0] - tol:
        if target_depth < -tol:
            raise ValueError(f"Target depth {target_depth} cannot be negative.")
        # Surface special case: reference levels are level 0 and level 1
        k0 = 0
        k1 = 1 if nz > 1 else 0
        z0 = level_depths[k0]
        z1 = level_depths[k1] if nz > 1 else z0
        w = (target_depth - z0) / (z1 - z0) if nz > 1 else 0.0
        dist_nearest = abs(target_depth - z0)
        nearest_k = 0
        return {
            "target_depth_m": float(target_depth),
            "k0": int(k0),
            "k1": int(k1),
            "depth_k0_m": float(z0),
            "depth_k1_m": float(z1),
            "weight_k1": float(w),
            "dist_nearest_m": float(dist_nearest),
            "nearest_k": int(nearest_k),
            "is_exact": False,
            "is_depth0": True,
        }

    # Check exact equality with any level
    diffs = np.abs(level_depths - target_depth)
    min_idx = int(np.argmin(diffs))
    if diffs[min_idx] <= tol:
        k0 = min_idx
        k1 = min_idx
        z0 = level_depths[min_idx]
        return {
            "target_depth_m": float(target_depth),
            "k0": int(k0),
            "k1": int(k1),
            "depth_k0_m": float(z0),
            "depth_k1_m": float(z0),
            "weight_k1": 0.0,
            "dist_nearest_m": 0.0,
            "nearest_k": int(k0),
            "is_exact": True,
            "is_depth0": False,
        }

    # Interior depth: bracketed by level k and k+1
    # np.searchsorted gives first index where level_depths[idx] >= target_depth
    idx = int(np.searchsorted(level_depths, target_depth))
    if idx <= 0:
        # Covered by surface case above
        k0, k1 = 0, 1
    elif idx >= nz:
        raise ValueError(
            f"Target depth {target_depth} exceeds deepest GLORYS level {level_depths[-1]:.2f} m"
        )
    else:
        k0 = idx - 1
        k1 = idx

    z0 = level_depths[k0]
    z1 = level_depths[k1]
    w = (target_depth - z0) / (z1 - z0)

    d0 = abs(target_depth - z0)
    d1 = abs(z1 - target_depth)
    if d0 <= d1:
        dist_nearest = d0
        nearest_k = k0
    else:
        dist_nearest = d1
        nearest_k = k1

    return {
        "target_depth_m": float(target_depth),
        "k0": int(k0),
        "k1": int(k1),
        "depth_k0_m": float(z0),
        "depth_k1_m": float(z1),
        "weight_k1": float(w),
        "dist_nearest_m": float(dist_nearest),
        "nearest_k": int(nearest_k),
        "is_exact": False,
        "is_depth0": False,
    }


def build_bracket_table(
    level_depths: np.ndarray,
    target_depths: List[float],
) -> pd.DataFrame:
    """Build the comprehensive bracket table for all target depths."""
    records = []
    for z_t in target_depths:
        b = bracket_search(level_depths, z_t)
        note = ""
        if b["is_depth0"]:
            note = "surface: V0a=level 0; V0b=extrap(0,1)"
        elif b["is_exact"]:
            note = "exact level match"
        else:
            note = f"interior bracket [{b['k0']}, {b['k1']}]"

        records.append({
            "target_depth_m": b["target_depth_m"],
            "k0": b["k0"],
            "k1": b["k1"],
            "depth_k0_m": b["depth_k0_m"],
            "depth_k1_m": b["depth_k1_m"],
            "weight_k1": b["weight_k1"],
            "dist_nearest_m": b["dist_nearest_m"],
            "nearest_k": b["nearest_k"],
            "notes": note,
        })

    return pd.DataFrame(records)


# =============================================================================
# Vertical Interpolation Operators (V1, V2, V0a, V0b)
# =============================================================================

def interpolate_v0a(
    vals_level0: np.ndarray,
    valid_level0: np.ndarray,
) -> np.ndarray:
    """Depth 0 rule V0a: value of level 0 (~0.49 m) used as 0 m value.
    
    Valid if and only if level 0 is valid.
    """
    return np.where(valid_level0, vals_level0, np.nan)


def interpolate_v0b(
    vals_level0: np.ndarray,
    vals_level1: np.ndarray,
    valid_level0: np.ndarray,
    valid_level1: np.ndarray,
    z0: float,
    z1: float,
) -> np.ndarray:
    """Depth 0 rule V0b: linear extrapolation from levels 0 and 1 to depth 0.
    
    V0b = V(z0) + (0 - z0)/(z1 - z0) * (V(z1) - V(z0))
    Valid if and only if BOTH levels 0 and 1 are valid.
    """
    valid_both = valid_level0 & valid_level1
    extrap_val = vals_level0 + (0.0 - z0) / (z1 - z0) * (vals_level1 - vals_level0)
    return np.where(valid_both, extrap_val, np.nan)


def interpolate_v1(
    vals_k0: np.ndarray,
    vals_k1: np.ndarray,
    valid_k0: np.ndarray,
    valid_k1: np.ndarray,
    weight_k1: float,
    is_exact: bool = False,
) -> np.ndarray:
    """Method V1: linear interpolation in depth between two bracketing levels.
    
    Valid only if BOTH bracketing levels have f_T >= tauT.
    If is_exact is True (target equals level k0), returns level k0 if valid_k0.
    """
    if is_exact:
        return np.where(valid_k0, vals_k0, np.nan)

    valid_both = valid_k0 & valid_k1
    interp_val = (1.0 - weight_k1) * vals_k0 + weight_k1 * vals_k1
    return np.where(valid_both, interp_val, np.nan)


def find_contiguous_valid_prefix_k(valid_levels_3d: np.ndarray) -> np.ndarray:
    """Compute K, the last level of the contiguous valid prefix (0..K) for each column.
    
    Parameters:
        valid_levels_3d: boolean array of shape (nz, ny, nx), True where f_T >= tauT.
        
    Returns:
        K: int64 array of shape (ny, nx). K is -1 if level 0 is invalid,
           otherwise K is the maximum index such that levels 0..K are all valid.
    """
    # Cumprod along level axis (axis=0): drops to 0 at first invalid level
    cum_valid = np.cumprod(valid_levels_3d.astype(np.int8), axis=0)
    prefix_length = np.sum(cum_valid, axis=0)
    K = prefix_length - 1
    return K


def interpolate_v2_pchip_vectorized(
    level_depths: np.ndarray,
    vals_3d: np.ndarray,
    valid_levels_3d: np.ndarray,
    target_depths: List[float],
    bracket_dict: Dict[float, Dict[str, Any]],
) -> Dict[float, np.ndarray]:
    """Method V2: PCHIP (monotone cubic Hermite) through valid contiguous prefix (0..K).
    
    Evaluated at target depth if K reaches upper bracketing level (deeper index k1).
    Columns are grouped by K to vectorize efficiently via scipy.interpolate.PchipInterpolator.
    
    Parameters:
        level_depths: 1D array of 36 level depths (m).
        vals_3d: float array of shape (36, ny, nx) with method A temperatures.
        valid_levels_3d: bool array of shape (36, ny, nx) indicating valid levels (f_T >= tauT).
        target_depths: list of target depths.
        bracket_dict: mapping from target_depth to bracket_search dict.
        
    Returns:
        Dict mapping target_depth -> 2D float array (ny, nx) of V2 results (NaN where invalid).
    """
    nz, ny, nx = vals_3d.shape
    level_depths = np.asarray(level_depths, dtype=np.float64)

    # Compute contiguous valid prefix K for all columns
    K_map = find_contiguous_valid_prefix_k(valid_levels_3d)

    # Prepare output containers
    v2_results: Dict[float, np.ndarray] = {
        z_t: np.full((ny, nx), np.nan, dtype=np.float64) for z_t in target_depths
    }

    # Flatten spatial grid for fast column grouping
    vals_2d = vals_3d.reshape(nz, ny * nx)
    K_flat = K_map.reshape(ny * nx)

    # Iterate over unique K values present
    unique_K = np.unique(K_flat)

    for k_val in unique_K:
        if k_val < 1:
            # Need at least 2 points (levels 0 and 1) for PCHIP interpolation
            # (If an exact match at level 0 occurs and k_val == 0, handled separately)
            for z_t in target_depths:
                b = bracket_dict[z_t]
                if b["is_exact"] and b["k0"] == 0 and k_val == 0:
                    col_idx = np.where(K_flat == 0)[0]
                    v2_results[z_t].reshape(ny * nx)[col_idx] = vals_2d[0, col_idx]
            continue

        col_indices = np.where(K_flat == k_val)[0]
        if len(col_indices) == 0:
            continue

        # Active levels for this group: 0..k_val
        active_depths = level_depths[: k_val + 1]
        active_vals = vals_2d[: k_val + 1, col_indices]  # shape: (k_val + 1, n_cols)

        # Identify target depths whose upper bracket k1 <= k_val (interior to prefix)
        depths_to_eval: List[float] = []
        for z_t in target_depths:
            b = bracket_dict[z_t]
            if b["is_depth0"]:
                # Depth 0 is evaluated by V0a/V0b, not interior PCHIP
                continue
            if b["k1"] <= k_val:
                depths_to_eval.append(z_t)

        if not depths_to_eval:
            continue

        # Vectorized PCHIP across all columns in this group
        # PchipInterpolator with axis=0 evaluates all columns simultaneously
        pchip = PchipInterpolator(active_depths, active_vals, axis=0)
        eval_res = pchip(depths_to_eval)  # shape: (len(depths_to_eval), n_cols)

        if len(depths_to_eval) == 1:
            eval_res = eval_res[np.newaxis, :]

        for d_idx, z_t in enumerate(depths_to_eval):
            v2_results[z_t].reshape(ny * nx)[col_indices] = eval_res[d_idx]

    return v2_results


# =============================================================================
# STEP 3: Real Data Execution & Diagnostic Accumulation
# =============================================================================

def run_vertical_experiment(
    thetao_path: str = DEFAULT_THETAO_PATH,
    static_path: str = DEFAULT_STATIC_PATH,
    policy_path: str = DEFAULT_POLICY_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute full 31-day vertical interpolation experiment and output diagnostics."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 78)
    print("GLORYS VERTICAL INTERPOLATION CONSEQUENCE EXPERIMENT")
    print("=" * 78)

    # 1. Capture file metadata before run
    meta_thetao_before = get_file_metadata(thetao_path)
    meta_static_before = get_file_metadata(static_path)

    if meta_thetao_before["sha256"] != EXPECTED_SHA256_THETAO:
        raise ValueError(
            f"Pre-check failed: thetao SHA256 {meta_thetao_before['sha256']} != expected {EXPECTED_SHA256_THETAO}"
        )
    if meta_static_before["sha256"] != EXPECTED_SHA256_STATIC:
        raise ValueError(
            f"Pre-check failed: static SHA256 {meta_static_before['sha256']} != expected {EXPECTED_SHA256_STATIC}"
        )

    print("\nPre-Execution File Verification:")
    print(f"  thetao: {meta_thetao_before['path']}")
    print(f"    Size: {meta_thetao_before['size_bytes']} bytes, SHA256: {meta_thetao_before['sha256']}")
    print(f"  static: {meta_static_before['path']}")
    print(f"    Size: {meta_static_before['size_bytes']} bytes, SHA256: {meta_static_before['sha256']}")

    # 2. Load policy, coordinates, static wet mask, and static exclusions
    policy = load_policy(policy_path)
    lats, lons, level_depths = load_coordinates(thetao_path)
    static_wet = load_static_wet_mask(
        static_path, ref_lats=lats, ref_lons=lons, ref_depths=level_depths
    )
    exclusion_mask = locate_exclusions(lats, lons, policy)

    enc = policy["encoding"]
    scale_factor = float(enc["scale_factor"])
    add_offset = float(enc["add_offset"])
    fill_raw = int(enc["fill_raw"])
    levels_used = policy.get("levels_used", 36)

    # Compute 2D horizontal overlap weights (Method A)
    w_lat_1d, _, _ = compute_overlap_weights_1d(
        native_centers=lats,
        target_start=TARGET_LAT_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LAT,
        native_step=NATIVE_STEP,
    )
    w_lon_1d, _, _ = compute_overlap_weights_1d(
        native_centers=lons,
        target_start=TARGET_LON_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LON,
        native_step=NATIVE_STEP,
    )
    w_2d = compute_overlap_weights_2d(w_lat_1d, w_lon_1d)

    # 3. Step 3a: Bracket Table
    print("\n[Step 3a] Generating Bracket Table for 15 target depths...")
    bracket_df = build_bracket_table(level_depths, TARGET_DEPTHS)
    bracket_csv_path = os.path.join(output_dir, "bracket_table.csv")
    bracket_df.to_csv(bracket_csv_path, index=False)
    print(f"  Saved: {bracket_csv_path}")

    bracket_dict = {
        z_t: bracket_search(level_depths, z_t) for z_t in TARGET_DEPTHS
    }

    # Target grid coordinates for spatial logging
    target_lats = TARGET_LAT_START + (np.arange(TARGET_N_LAT) + 0.5) * TARGET_STEP
    target_lons = TARGET_LON_START + (np.arange(TARGET_N_LON) + 0.5) * TARGET_STEP

    # -------------------------------------------------------------------------
    # Accumulators across 31 days
    # -------------------------------------------------------------------------
    # Validity counts: per target depth, for tauT = 1.0 and 0.5
    # Keys: valid_v1, valid_upper, gap_band
    val_counts_d0 = {
        "1.0": {z_t: {"v1": 0, "upper": 0, "gap": 0} for z_t in TARGET_DEPTHS},
        "0.5": {z_t: {"v1": 0, "upper": 0, "gap": 0} for z_t in TARGET_DEPTHS},
    }
    val_counts_pooled = {
        "1.0": {z_t: {"v1": 0, "upper": 0, "gap": 0} for z_t in TARGET_DEPTHS},
        "0.5": {z_t: {"v1": 0, "upper": 0, "gap": 0} for z_t in TARGET_DEPTHS},
    }

    # Differences V2 - V1 accumulators (pooled across days)
    diffs_v2_v1_pooled = {
        "1.0": {z_t: [] for z_t in TARGET_DEPTHS},
        "0.5": {z_t: [] for z_t in TARGET_DEPTHS},
    }

    # Depth 0 alternatives (V0b - V0a) accumulators
    depth0_diffs_all = []    # valid at levels 0 & 1 with tauT = 0.5
    depth0_diffs_feq1 = []   # f_T == 1.0
    depth0_diffs_flt1 = []   # 0.5 <= f_T < 1.0

    # Day 0 snapshots for top cells and maps
    day0_v1: Dict[str, Dict[float, np.ndarray]] = {"1.0": {}, "0.5": {}}
    day0_v2: Dict[str, Dict[float, np.ndarray]] = {"1.0": {}, "0.5": {}}
    day0_diff_v0 = None

    print("\nBeginning 31-day GLORYS vertical experiment across all 31 days...")

    for d in range(31):
        print(f"  Processing day {d:02d}/30...", end="", flush=True)

        # 1. Read raw slice
        raw_day = read_raw_day(thetao_path, day_index=d, policy=policy)

        # 2. QC valid mask
        qc_res = apply_qc(
            raw_day, static_wet, exclusion_mask, policy,
            lats=lats, lons=lons, depths=level_depths
        )

        # 3. Decode valid cells manually after masking fill
        thetao_decoded = np.where(
            qc_res.valid,
            raw_day.astype(np.float64) * scale_factor + add_offset,
            np.nan,
        )

        # 4. Horizontal Method A for all 36 levels
        val_A, f = regrid_method_A(thetao_decoded, qc_res.valid, weights_2d=w_2d)

        # Process both validity thresholds: primary tauT = 1.0, secondary tauT = 0.5
        for tau_str, tau_val in [("1.0", 1.0), ("0.5", 0.5)]:
            # Level validity mask: f >= tauT (with 1e-12 tolerance for float precision)
            lvl_valid = (f >= tau_val - 1e-12)

            # Compute V1 for all target depths
            v1_dict: Dict[float, np.ndarray] = {}
            for z_t in TARGET_DEPTHS:
                b = bracket_dict[z_t]
                if b["is_depth0"]:
                    # Depth 0: V0a = level 0, V0b = extrap(0, 1)
                    # For validity counting: upper level is 0, lower level is 1
                    # V1 under depth 0 is V0b (requires both levels 0 & 1)
                    v0b = interpolate_v0b(
                        val_A[0], val_A[1], lvl_valid[0], lvl_valid[1],
                        level_depths[0], level_depths[1]
                    )
                    v1_dict[z_t] = v0b
                else:
                    k0, k1 = b["k0"], b["k1"]
                    w_k1 = b["weight_k1"]
                    v1_dict[z_t] = interpolate_v1(
                        val_A[k0], val_A[k1], lvl_valid[k0], lvl_valid[k1],
                        w_k1, is_exact=b["is_exact"]
                    )

            # Compute V2 for all target depths via PCHIP
            v2_dict = interpolate_v2_pchip_vectorized(
                level_depths, val_A, lvl_valid, TARGET_DEPTHS, bracket_dict
            )

            # At depth 0: V0a and V0b
            v0a = interpolate_v0a(val_A[0], lvl_valid[0])
            v0b = interpolate_v0b(
                val_A[0], val_A[1], lvl_valid[0], lvl_valid[1],
                level_depths[0], level_depths[1]
            )

            if d == 0:
                day0_v1[tau_str] = v1_dict
                day0_v2[tau_str] = v2_dict
                if tau_str == "0.5":
                    day0_diff_v0 = v0b - v0a

            # Accumulate validity counts
            for z_t in TARGET_DEPTHS:
                b = bracket_dict[z_t]
                k0, k1 = b["k0"], b["k1"]

                # Upper level alone
                upper_valid = lvl_valid[k0]
                n_upper = int(np.sum(upper_valid))

                # Both levels valid (V1 condition)
                both_valid = ~np.isnan(v1_dict[z_t])
                n_v1 = int(np.sum(both_valid))

                # Gap band: upper is valid but lower is not
                n_gap = n_upper - n_v1

                if d == 0:
                    val_counts_d0[tau_str][z_t]["v1"] = n_v1
                    val_counts_d0[tau_str][z_t]["upper"] = n_upper
                    val_counts_d0[tau_str][z_t]["gap"] = n_gap

                val_counts_pooled[tau_str][z_t]["v1"] += n_v1
                val_counts_pooled[tau_str][z_t]["upper"] += n_upper
                val_counts_pooled[tau_str][z_t]["gap"] += n_gap

                # Accumulate V2 - V1 differences for cells valid for both methods
                if not b["is_depth0"]:
                    mask_both = (~np.isnan(v1_dict[z_t])) & (~np.isnan(v2_dict[z_t]))
                    if np.any(mask_both):
                        diff_vals = (v2_dict[z_t][mask_both] - v1_dict[z_t][mask_both]).astype(np.float64)
                        diffs_v2_v1_pooled[tau_str][z_t].append(diff_vals)

        # Accumulate depth 0 alternatives (V0b - V0a)
        # Pooled over days and cells valid at levels 0 and 1
        valid_l01 = (f[0] >= 0.5 - 1e-12) & (f[1] >= 0.5 - 1e-12)
        diff_d0 = v0b - v0a
        if np.any(valid_l01):
            vals_d0_all = diff_d0[valid_l01]
            depth0_diffs_all.append(vals_d0_all)

            # Split by f_T == 1 vs f_T < 1
            f_is_1 = np.isclose(f[0], 1.0, atol=1e-12) & np.isclose(f[1], 1.0, atol=1e-12)
            mask_feq1 = valid_l01 & f_is_1
            mask_flt1 = valid_l01 & (~f_is_1)

            if np.any(mask_feq1):
                depth0_diffs_feq1.append(diff_d0[mask_feq1])
            if np.any(mask_flt1):
                depth0_diffs_flt1.append(diff_d0[mask_flt1])

        print(" done.")

    # -------------------------------------------------------------------------
    # Step 3b: Validity Table per Target Depth
    # -------------------------------------------------------------------------
    print("\n[Step 3b] Compiling validity_by_depth.csv...")
    val_rows = []
    for z_t in TARGET_DEPTHS:
        b = bracket_dict[z_t]
        val_rows.append({
            "target_depth_m": z_t,
            "k0": b["k0"],
            "k1": b["k1"],
            "d0_valid_v1_tau10": val_counts_d0["1.0"][z_t]["v1"],
            "d0_valid_upper_tau10": val_counts_d0["1.0"][z_t]["upper"],
            "d0_gap_band_tau10": val_counts_d0["1.0"][z_t]["gap"],
            "d0_valid_v1_tau05": val_counts_d0["0.5"][z_t]["v1"],
            "d0_valid_upper_tau05": val_counts_d0["0.5"][z_t]["upper"],
            "d0_gap_band_tau05": val_counts_d0["0.5"][z_t]["gap"],
            "pooled_valid_v1_tau10": val_counts_pooled["1.0"][z_t]["v1"],
            "pooled_valid_upper_tau10": val_counts_pooled["1.0"][z_t]["upper"],
            "pooled_gap_band_tau10": val_counts_pooled["1.0"][z_t]["gap"],
            "pooled_valid_v1_tau05": val_counts_pooled["0.5"][z_t]["v1"],
            "pooled_valid_upper_tau05": val_counts_pooled["0.5"][z_t]["upper"],
            "pooled_gap_band_tau05": val_counts_pooled["0.5"][z_t]["gap"],
        })
    df_validity = pd.DataFrame(val_rows)
    validity_csv_path = os.path.join(output_dir, "validity_by_depth.csv")
    df_validity.to_csv(validity_csv_path, index=False)
    print(f"  Saved: {validity_csv_path}")

    # -------------------------------------------------------------------------
    # Step 3c: V2 - V1 Differences Table
    # -------------------------------------------------------------------------
    print("\n[Step 3c] Compiling v2_minus_v1.csv...")
    v2_v1_rows = []

    # Also compute depth 0 pooled stats for inclusion in 15-depth overview
    concat_d0_all = np.concatenate(depth0_diffs_all) if depth0_diffs_all else np.array([])
    concat_d0_feq1 = np.concatenate(depth0_diffs_feq1) if depth0_diffs_feq1 else np.array([])
    d0_spacing = float(level_depths[1] - level_depths[0])

    for tau_str, tau_val in [("0.5", 0.5), ("1.0", 1.0)]:
        for z_t in TARGET_DEPTHS:
            b = bracket_dict[z_t]
            if b["is_depth0"]:
                # Depth 0: candidate comparison is V0b - V0a
                d0_arr = concat_d0_feq1 if tau_str == "1.0" else concat_d0_all
                if len(d0_arr) > 0:
                    n_c = len(d0_arr)
                    bias = float(np.mean(d0_arr))
                    rmse = float(np.sqrt(np.mean(d0_arr ** 2)))
                    mae = float(np.mean(np.abs(d0_arr)))
                    p99 = float(np.percentile(np.abs(d0_arr), 99))
                    max_abs = float(np.max(np.abs(d0_arr)))
                    ratio = float(rmse / d0_spacing)
                else:
                    n_c = 0
                    bias = rmse = mae = p99 = max_abs = ratio = np.nan

                v2_v1_rows.append({
                    "tauT": tau_val,
                    "target_depth_m": z_t,
                    "method_pair": "V0b - V0a (surface)",
                    "n_cells_pooled": n_c,
                    "bias_degC": bias,
                    "rmse_degC": rmse,
                    "mae_degC": mae,
                    "p99_abs_diff_degC": p99,
                    "max_abs_diff_degC": max_abs,
                    "local_spacing_m": d0_spacing,
                    "ratio_rmse_to_spacing": ratio,
                })
            else:
                chunks = diffs_v2_v1_pooled[tau_str][z_t]
                spacing = float(b["depth_k1_m"] - b["depth_k0_m"])
                if chunks:
                    all_diffs = np.concatenate(chunks)
                    n_c = len(all_diffs)
                    bias = float(np.mean(all_diffs))
                    rmse = float(np.sqrt(np.mean(all_diffs ** 2)))
                    mae = float(np.mean(np.abs(all_diffs)))
                    p99 = float(np.percentile(np.abs(all_diffs), 99))
                    max_abs = float(np.max(np.abs(all_diffs)))
                    ratio = float(rmse / spacing)
                else:
                    n_c = 0
                    bias = rmse = mae = p99 = max_abs = ratio = np.nan

                v2_v1_rows.append({
                    "tauT": tau_val,
                    "target_depth_m": z_t,
                    "method_pair": "V2 - V1",
                    "n_cells_pooled": n_c,
                    "bias_degC": bias,
                    "rmse_degC": rmse,
                    "mae_degC": mae,
                    "p99_abs_diff_degC": p99,
                    "max_abs_diff_degC": max_abs,
                    "local_spacing_m": spacing,
                    "ratio_rmse_to_spacing": ratio,
                })

    df_v2_v1 = pd.DataFrame(v2_v1_rows)
    v2_v1_csv_path = os.path.join(output_dir, "v2_minus_v1.csv")
    df_v2_v1.to_csv(v2_v1_csv_path, index=False)
    print(f"  Saved: {v2_v1_csv_path}")

    # -------------------------------------------------------------------------
    # Step 3d: Top 20 Cells for 100, 200, 300 m on Day 0
    # -------------------------------------------------------------------------
    print("\n[Step 3d] Extracting top 20 cells with largest |V2 - V1| on day 0...")
    top_records = []
    lat_grid, lon_grid = np.meshgrid(target_lats, target_lons, indexing="ij")

    # Evaluate for tauT = 0.5 (superset) as requested in 3c/3d
    for z_t in [100.0, 200.0, 300.0]:
        v1_d0 = day0_v1["0.5"][z_t]
        v2_d0 = day0_v2["0.5"][z_t]
        valid_both = (~np.isnan(v1_d0)) & (~np.isnan(v2_d0))

        diff_map = v2_d0 - v1_d0
        abs_diff_map = np.abs(diff_map)

        valid_indices = np.where(valid_both)
        abs_diffs_valid = abs_diff_map[valid_indices]

        # Top 20 indices
        top_k = min(20, len(abs_diffs_valid))
        top_order = np.argsort(-abs_diffs_valid)[:top_k]

        for rank, idx in enumerate(top_order, start=1):
            r = valid_indices[0][idx]
            c = valid_indices[1][idx]
            top_records.append({
                "target_depth_m": z_t,
                "rank": rank,
                "lat": float(lat_grid[r, c]),
                "lon": float(lon_grid[r, c]),
                "lat_idx": int(r),
                "lon_idx": int(c),
                "val_v1": float(v1_d0[r, c]),
                "val_v2": float(v2_d0[r, c]),
                "diff_v2_minus_v1": float(diff_map[r, c]),
                "abs_diff": float(abs_diff_map[r, c]),
                "tauT": 0.5,
            })

    df_top_cells = pd.DataFrame(top_records)
    top_cells_csv_path = os.path.join(output_dir, "top_v2_v1_cells.csv")
    df_top_cells.to_csv(top_cells_csv_path, index=False)
    print(f"  Saved: {top_cells_csv_path}")

    # -------------------------------------------------------------------------
    # Step 3e: Depth 0 Alternatives Table (V0b - V0a)
    # -------------------------------------------------------------------------
    print("\n[Step 3e] Compiling depth0_alternatives.csv...")
    depth0_rows = []

    def _calc_stats(arr: np.ndarray, subset_name: str) -> Dict[str, Any]:
        if len(arr) == 0:
            return {
                "subset": subset_name,
                "n_cells_pooled": 0,
                "bias_degC": np.nan,
                "rmse_degC": np.nan,
                "mae_degC": np.nan,
                "p99_abs_diff_degC": np.nan,
                "max_abs_diff_degC": np.nan,
            }
        return {
            "subset": subset_name,
            "n_cells_pooled": int(len(arr)),
            "bias_degC": float(np.mean(arr)),
            "rmse_degC": float(np.sqrt(np.mean(arr ** 2))),
            "mae_degC": float(np.mean(np.abs(arr))),
            "p99_abs_diff_degC": float(np.percentile(np.abs(arr), 99)),
            "max_abs_diff_degC": float(np.max(np.abs(arr))),
        }

    concat_flt1 = np.concatenate(depth0_diffs_flt1) if depth0_diffs_flt1 else np.array([])

    depth0_rows.append(_calc_stats(concat_d0_all, "All valid (tauT >= 0.5)"))
    depth0_rows.append(_calc_stats(concat_d0_feq1, "Full coverage (f_T == 1.0)"))
    depth0_rows.append(_calc_stats(concat_flt1, "Partial/coastal (0.5 <= f_T < 1.0)"))

    df_depth0 = pd.DataFrame(depth0_rows)
    depth0_csv_path = os.path.join(output_dir, "depth0_alternatives.csv")
    df_depth0.to_csv(depth0_csv_path, index=False)
    print(f"  Saved: {depth0_csv_path}")

    # -------------------------------------------------------------------------
    # Step 3f: Plots
    # -------------------------------------------------------------------------
    print("\n[Step 3f] Generating diagnostic plots...")

    # Plot 1: RMSE(V2 - V1) vs target depth
    plt.figure(figsize=(9, 6), dpi=150)
    sub_05 = df_v2_v1[(df_v2_v1["tauT"] == 0.5) & (df_v2_v1["target_depth_m"] > 0)]
    sub_10 = df_v2_v1[(df_v2_v1["tauT"] == 1.0) & (df_v2_v1["target_depth_m"] > 0)]

    plt.plot(sub_05["target_depth_m"], sub_05["rmse_degC"], "o-", color="#1f77b4", label="tauT = 0.5 (superset)", linewidth=1.8, markersize=5)
    plt.plot(sub_10["target_depth_m"], sub_10["rmse_degC"], "s--", color="#ff7f0e", label="tauT = 1.0 (strict)", linewidth=1.5, markersize=5)
    plt.xlabel("Target Depth (m)", fontsize=11)
    plt.ylabel("RMSE(V2 - V1) (°C)", fontsize=11)
    plt.title("GLORYS Vertical Interpolation: RMSE(V2 - V1) vs Target Depth (Pooled 31 Days)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "rmse_vs_target_depth.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"  Saved: {plot1_path}")

    # Plot 2: Number of valid cells vs target depth for tauT = 1.0 and 0.5
    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(df_validity["target_depth_m"], df_validity["d0_valid_v1_tau05"], "o-", color="#1f77b4", label="Day 0 (tauT = 0.5)", linewidth=1.8, markersize=5)
    plt.plot(df_validity["target_depth_m"], df_validity["d0_valid_v1_tau10"], "s--", color="#ff7f0e", label="Day 0 (tauT = 1.0)", linewidth=1.5, markersize=5)
    plt.plot(df_validity["target_depth_m"], df_validity["d0_valid_upper_tau05"], "^:", color="#2ca02c", label="Day 0 Upper Bracket Alone (tauT = 0.5)", linewidth=1.2, markersize=4)
    plt.xlabel("Target Depth (m)", fontsize=11)
    plt.ylabel("Valid Target Cells", fontsize=11)
    plt.title("GLORYS Vertical Interpolation: Target Cell Validity vs Target Depth", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "valid_cells_vs_target_depth.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"  Saved: {plot2_path}")

    # Plot 3: Day-0 map of V0b - V0a
    plt.figure(figsize=(10, 5), dpi=150)
    # Mask dry / invalid cells
    masked_diff = np.ma.masked_invalid(day0_diff_v0)
    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad(color="#d9d9d9")

    # Symmetric color limits around 0
    vmax = np.nanpercentile(np.abs(day0_diff_v0), 99.5) if np.any(~np.isnan(day0_diff_v0)) else 0.5
    vmin = -vmax

    im = plt.pcolormesh(
        target_lons, target_lats, masked_diff,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    cbar = plt.colorbar(im, orientation="vertical", pad=0.02, shrink=0.88)
    cbar.set_label("V0b - V0a (°C)", fontsize=10)
    plt.xlabel("Longitude (°E)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.title("GLORYS Day 0: Depth 0 Alternative Difference (V0b - V0a)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "day0_map_v0b_minus_v0a.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"  Saved: {plot3_path}")

    # -------------------------------------------------------------------------
    # Step 3g: JSON Summary Output
    # -------------------------------------------------------------------------
    print("\n[Step 3g] Generating glorys_vertical_experiment.json...")
    meta_thetao_after = get_file_metadata(thetao_path)
    meta_static_after = get_file_metadata(static_path)

    json_data = {
        "experiment": "GLORYS vertical interpolation consequence experiment",
        "status": "DIAGNOSTIC ONLY; no pipeline decision",
        "file_integrity": {
            "thetao": {
                "path": meta_thetao_before["path"],
                "sha256_before": meta_thetao_before["sha256"],
                "sha256_after": meta_thetao_after["sha256"],
                "size_bytes_before": meta_thetao_before["size_bytes"],
                "size_bytes_after": meta_thetao_after["size_bytes"],
                "mtime_before": meta_thetao_before["mtime"],
                "mtime_after": meta_thetao_after["mtime"],
                "unmodified": bool(meta_thetao_before["sha256"] == meta_thetao_after["sha256"] == EXPECTED_SHA256_THETAO),
            },
            "static": {
                "path": meta_static_before["path"],
                "sha256_before": meta_static_before["sha256"],
                "sha256_after": meta_static_after["sha256"],
                "size_bytes_before": meta_static_before["size_bytes"],
                "size_bytes_after": meta_static_after["size_bytes"],
                "mtime_before": meta_static_before["mtime"],
                "mtime_after": meta_static_after["mtime"],
                "unmodified": bool(meta_static_before["sha256"] == meta_static_after["sha256"] == EXPECTED_SHA256_STATIC),
            },
        },
        "target_depths_m": TARGET_DEPTHS,
        "n_target_depths": len(TARGET_DEPTHS),
        "days_processed": 31,
        "summary_findings": {
            "v2_v1_overall": {
                "tauT_05_max_rmse_depth_m": float(sub_05.loc[sub_05["rmse_degC"].idxmax()]["target_depth_m"]) if len(sub_05) > 0 else None,
                "tauT_05_max_rmse_degC": float(sub_05["rmse_degC"].max()) if len(sub_05) > 0 else None,
                "tauT_10_max_rmse_degC": float(sub_10["rmse_degC"].max()) if len(sub_10) > 0 else None,
                "tauT_05_max_abs_diff_degC": float(sub_05["max_abs_diff_degC"].max()) if len(sub_05) > 0 else None,
                "tauT_10_max_abs_diff_degC": float(sub_10["max_abs_diff_degC"].max()) if len(sub_10) > 0 else None,
            },
            "depth0_alternatives": {
                "pooled_v0b_minus_v0a_bias_degC": float(np.mean(concat_d0_all)) if len(concat_d0_all) > 0 else None,
                "pooled_v0b_minus_v0a_rmse_degC": float(np.sqrt(np.mean(concat_d0_all ** 2))) if len(concat_d0_all) > 0 else None,
                "pooled_v0b_minus_v0a_p99_degC": float(np.percentile(np.abs(concat_d0_all), 99)) if len(concat_d0_all) > 0 else None,
                "pooled_v0b_minus_v0a_max_degC": float(np.max(np.abs(concat_d0_all))) if len(concat_d0_all) > 0 else None,
            },
            "gap_band_summary": {
                "max_gap_band_cells_day0_tau10": int(df_validity["d0_gap_band_tau10"].max()),
                "max_gap_band_depth_day0_tau10_m": float(df_validity.loc[df_validity["d0_gap_band_tau10"].idxmax()]["target_depth_m"]),
                "max_gap_band_cells_day0_tau05": int(df_validity["d0_gap_band_tau05"].max()),
                "max_gap_band_depth_day0_tau05_m": float(df_validity.loc[df_validity["d0_gap_band_tau05"].idxmax()]["target_depth_m"]),
            },
        },
        "unknown_fields": {
            "future_operational_pipeline_vertical_method": None,
            "pipeline_decision_rationale": None,
            "reason": "This task is strictly diagnostic and read-only. Pipeline decisions must be taken by the research lead.",
        },
    }

    json_path = os.path.join(output_dir, "glorys_vertical_experiment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nExperiment execution successfully completed.")
    return json_data


# =============================================================================
# CLI Entrypoint
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLORYS vertical interpolation consequence experiment."
    )
    parser.add_argument(
        "--thetao-path", type=str, default=DEFAULT_THETAO_PATH,
        help="Path to raw GLORYS thetao NetCDF file."
    )
    parser.add_argument(
        "--static-path", type=str, default=DEFAULT_STATIC_PATH,
        help="Path to raw GLORYS static mask and bathymetry NetCDF file."
    )
    parser.add_argument(
        "--policy-path", type=str, default=DEFAULT_POLICY_PATH,
        help="Path to QC policy JSON config."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSVs, plots, and JSON."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vertical_experiment(
        thetao_path=args.thetao_path,
        static_path=args.static_path,
        policy_path=args.policy_path,
        output_dir=args.output_dir,
    )
