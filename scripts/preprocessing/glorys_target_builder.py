#!/usr/bin/env python3
"""
GLORYS Target Builder v1
(Reusable Preprocessing and Target Generation Module)

Transforms raw GLORYS ocean potential temperature (thetao) into project target
regridded and vertically interpolated fields according to validated research policy decisions:
- P1: Runs QC policy v1 on each day first. QC status FAIL stops the build with an error
      naming the unexpected violating cells. WARN is recorded, not hidden.
- P2: Horizontal regridding Method A: For target cell (i, j), uses native indices 3i..3i+3 (lat)
      and 3j..3j+3 (lon) with 1D area-overlap weights [1/6, 1/3, 1/3, 1/6]. Weights are computed
      from native and target cell edges dynamically and asserted to sum to 1.
      Value = sum(w * x over QC-valid cells) / sum(w over valid cells);
      Valid fraction f_T = sum(w over valid cells).
- P3: A target cell is valid at a GLORYS level iff f_T >= tauT (tauT = 1.0).
- P4: Vertical interpolation: PCHIP (monotone cubic Hermite, scipy.interpolate.PchipInterpolator)
      through the valid contiguous prefix of levels of the target column (levels 0..K with f_T >= tauT).
      A target depth between levels k0 and k0+1 is valid iff both levels are valid (K >= k0+1).
      No extrapolation. Target depths between the last valid level and the seabed are NaN.
      Columns are grouped by K to vectorize efficiently.
- P5: Depth 0 uses the value of level 0 (~0.49 m).

Inputs (all read-only, never modified):
  - Raw GLORYS thetao NetCDF (data/raw/glorys/glorys_thetao_2019-01.nc)
  - Static wet mask and bathymetry NetCDF (data/raw/glorys_static/glorys_static_mask_bathy.nc)
  - QC policy v1 (configs/glorys_qc_policy_v1.json)
  - Target policy v1 (configs/glorys_target_policy_v1.json)

Outputs:
  - NetCDF file: data/processed/glorys_targets/glorys_targets_YYYY-MM.nc
    Variables: thetao_target (time, depth, latitude, longitude, float32, NaN for invalid)
               valid_mask (uint8, 1=valid, 0=invalid)
    Coordinates: time, depth, latitude, longitude
    zlib compression, complete provenance and global attributes.
  - Manifest JSON: data/processed/glorys_targets/glorys_targets_YYYY-MM_manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import h5netcdf
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
import xarray as xr

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocessing.glorys_qc import (
    apply_qc,
    load_coordinates,
    load_policy as load_qc_policy,
    load_static_wet_mask,
    locate_exclusions,
    read_raw_day,
    QCResult,
)


# =============================================================================
# Provenance & Hash Helpers
# =============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file without loading the whole file into RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash(repo_dir: Optional[str] = None) -> str:
    """Retrieve git commit hash or 'unavailable' if git is not available."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = repo_dir or PROJECT_ROOT
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unavailable"


# =============================================================================
# Policy Loading
# =============================================================================

def load_target_policy(path: Union[str, os.PathLike, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and validate the target policy configuration."""
    if isinstance(path, dict):
        policy = path
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target policy file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            policy = json.load(f)

    # Basic schema checks
    required_keys = [
        "policy_version",
        "target_depths",
        "tauT",
        "horizontal_method",
        "vertical_method",
        "depth0",
        "seabed_gap",
        "qc_policy_path",
        "target_grid",
    ]
    for k in required_keys:
        if k not in policy:
            raise KeyError(f"Missing required key '{k}' in target policy: {path}")

    return policy


# =============================================================================
# P2: Horizontal Method A - Dynamic Edge-based Weights
# =============================================================================

def compute_overlap_weights_1d(
    native_centers: np.ndarray,
    target_start: float,
    target_step: float,
    n_target: int,
    native_step: float = 1.0 / 12.0,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, List[List[int]], List[np.ndarray]]:
    """Compute 1D area-overlap weights from native and target cell edges dynamically.
    
    Target cell i covers [target_start + i * target_step, target_start + (i + 1) * target_step].
    Native cell k covers [native_centers[k] - native_step / 2, native_centers[k] + native_step / 2].
    
    Verifies and asserts:
    - Overlap weights sum to 1.0 for each target cell.
    - Each target cell i overlaps exactly native indices [3*i, 3*i+1, 3*i+2, 3*i+3].
    - Overlap weights equal [1/6, 1/3, 1/3, 1/6].
    - Domain edges: i = n_target - 1 uses native index 3*n_target (i=99 uses 300; j=239 uses 720).
    - Cells outside the domain are not counted.
    """
    native_centers = np.asarray(native_centers, dtype=np.float64)
    half_native = native_step / 2.0
    n_native = len(native_centers)

    all_indices: List[List[int]] = []
    all_weights: List[np.ndarray] = []

    expected_w = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=np.float64)

    for i in range(n_target):
        t_left = target_start + i * target_step
        t_right = target_start + (i + 1) * target_step

        indices: List[int] = []
        weights: List[float] = []

        for k in range(n_native):
            n_left = native_centers[k] - half_native
            n_right = native_centers[k] + half_native

            int_left = max(t_left, n_left)
            int_right = min(t_right, n_right)
            overlap = max(0.0, int_right - int_left)

            if overlap > 1e-10:
                indices.append(k)
                weights.append(overlap / target_step)

        # Assertions required by contract
        expected_indices = [3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3]
        if indices != expected_indices:
            raise AssertionError(f"Target cell {i}: expected native indices {expected_indices}, got {indices}")

        w_arr = np.array(weights, dtype=np.float64)
        if not np.allclose(w_arr, expected_w, atol=tol):
            raise AssertionError(f"Target cell {i}: expected weights {expected_w}, got {w_arr}")

        if not np.isclose(np.sum(w_arr), 1.0, atol=tol):
            raise AssertionError(f"Target cell {i}: weights do not sum to 1.0 (sum={np.sum(w_arr)})")

        all_indices.append(indices)
        all_weights.append(w_arr)

    # Edge checks at boundary
    last_i = n_target - 1
    last_indices = all_indices[last_i]
    expected_boundary_idx = 3 * n_target
    if last_indices[-1] != expected_boundary_idx:
        raise AssertionError(
            f"Domain edge assertion failed: last target cell {last_i} last index is {last_indices[-1]}, "
            f"expected {expected_boundary_idx}"
        )
    if expected_boundary_idx >= n_native:
        raise AssertionError(f"Boundary index {expected_boundary_idx} exceeds native length {n_native}")

    canonical_w_1d = expected_w
    return canonical_w_1d, all_indices, all_weights


def compute_overlap_weights_2d(
    w_lat_1d: np.ndarray,
    w_lon_1d: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Compute 2D overlap weights as the outer product of 1D lat and lon weights."""
    w_lat = np.asarray(w_lat_1d, dtype=np.float64)
    w_lon = np.asarray(w_lon_1d, dtype=np.float64)
    if w_lat.shape != (4,) or w_lon.shape != (4,):
        raise ValueError(f"1D weight arrays must have shape (4,), got {w_lat.shape} and {w_lon.shape}")

    w_2d = np.outer(w_lat, w_lon)
    if not np.isclose(np.sum(w_2d), 1.0, atol=tol):
        raise AssertionError(f"2D weights do not sum to 1.0 (sum={np.sum(w_2d)})")

    return w_2d


def regrid_method_A(
    field: np.ndarray,
    valid_mask: np.ndarray,
    weights_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Method A: exact area-weighted mean over the 16 overlapping native cells using 2D weights.

    Uses only QC-valid cells:
        value = sum(w * x over valid) / sum(w over valid)
        f_T = sum(w over valid)
    Returns (val_A, f_T) where val_A is NaN if f_T == 0.
    Uses numpy slicing across the whole array (no per-cell Python loops).

    Parameters:
        field: float array of shape (..., ny, nx) where ny = 3*N + 1, nx = 3*M + 1
        valid_mask: bool array of shape (..., ny, nx)
        weights_2d: optional (4, 4) array of weights

    Returns:
        val_A: float64 array of shape (..., N, M)
        f_T: float64 array of shape (..., N, M)
    """
    field = np.asarray(field, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if field.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch: field {field.shape} vs valid_mask {valid_mask.shape}")

    ny, nx = field.shape[-2], field.shape[-1]
    if (ny - 1) % 3 != 0 or (nx - 1) % 3 != 0:
        raise ValueError(f"Spatial dimensions must be 3*N+1 and 3*M+1, got ({ny}, {nx})")

    N = (ny - 1) // 3
    M = (nx - 1) // 3

    if weights_2d is None:
        w_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
        weights_2d = np.outer(w_1d, w_1d)
    else:
        weights_2d = np.asarray(weights_2d, dtype=np.float64)

    out_shape = field.shape[:-2] + (N, M)
    sum_wx = np.zeros(out_shape, dtype=np.float64)
    f_T = np.zeros(out_shape, dtype=np.float64)

    for m in range(4):
        s_lat = slice(m, 3 * N - 2 + m, 3)
        for n in range(4):
            s_lon = slice(n, 3 * M - 2 + n, 3)
            w = weights_2d[m, n]
            vm = valid_mask[..., s_lat, s_lon]
            val = field[..., s_lat, s_lon]

            f_T += w * vm
            safe_val = np.where(vm, val, 0.0)
            sum_wx += w * safe_val

    val_A = np.where(f_T > 0.0, sum_wx / np.where(f_T > 0.0, f_T, 1.0), np.nan)
    return val_A, f_T


# =============================================================================
# P4 & P5: Vertical Interpolation (Bracket Search & PCHIP)
# =============================================================================

def bracket_search(
    level_depths: np.ndarray,
    target_depth: float,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Find bracketing levels, interpolation weight, and surface special case flag.
    
    Parameters:
        level_depths: 1D array of native level depths (strictly increasing).
        target_depth: Target depth (m).
        tol: Tolerance for exact equality.
        
    Returns:
        Dict with keys:
            - target_depth_m: float
            - k0: int, index of shallower level
            - k1: int, index of deeper level
            - depth_k0_m: float
            - depth_k1_m: float
            - weight_k1: float, linear weight on level k1
            - is_exact: bool
            - is_depth0: bool
    """
    level_depths = np.asarray(level_depths, dtype=np.float64)
    nz = len(level_depths)

    # Depth 0 special case (shallower than first level)
    if target_depth < level_depths[0] - tol:
        if target_depth < -tol:
            raise ValueError(f"Target depth {target_depth} cannot be negative.")
        k0 = 0
        k1 = 1 if nz > 1 else 0
        z0 = level_depths[k0]
        z1 = level_depths[k1] if nz > 1 else z0
        w = (target_depth - z0) / (z1 - z0) if nz > 1 else 0.0
        return {
            "target_depth_m": float(target_depth),
            "k0": int(k0),
            "k1": int(k1),
            "depth_k0_m": float(z0),
            "depth_k1_m": float(z1),
            "weight_k1": float(w),
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
            "is_exact": True,
            "is_depth0": False,
        }

    # Interior depth: bracketed by level k0 and k1
    idx = int(np.searchsorted(level_depths, target_depth))
    if idx <= 0:
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

    return {
        "target_depth_m": float(target_depth),
        "k0": int(k0),
        "k1": int(k1),
        "depth_k0_m": float(z0),
        "depth_k1_m": float(z1),
        "weight_k1": float(w),
        "is_exact": False,
        "is_depth0": False,
    }


def find_contiguous_valid_prefix_k(valid_levels_3d: np.ndarray) -> np.ndarray:
    """Compute K, the last level of the contiguous valid prefix (0..K) for each column.
    
    Parameters:
        valid_levels_3d: boolean array of shape (nz, ny, nx), True where f_T >= tauT.
        
    Returns:
        K: int64 array of shape (ny, nx). K is -1 if level 0 is invalid,
           otherwise K is the maximum index such that levels 0..K are all valid.
    """
    cum_valid = np.cumprod(valid_levels_3d.astype(np.int8), axis=0)
    prefix_length = np.sum(cum_valid, axis=0)
    K = prefix_length - 1
    return K


def interpolate_column_pchip(
    level_depths: np.ndarray,
    vals_3d: np.ndarray,
    valid_levels_3d: np.ndarray,
    target_depths: List[float],
    bracket_dict: Dict[float, Dict[str, Any]],
    depth0_method: str = "level0",
) -> np.ndarray:
    """Vertical interpolation implementing P4 (vectorized PCHIP over prefix) and P5 (depth 0).
    
    Parameters:
        level_depths: 1D array of native level depths (m).
        vals_3d: float array of shape (nz, ny, nx) with method A temperatures.
        valid_levels_3d: bool array of shape (nz, ny, nx) where f_T >= tauT.
        target_depths: list of target depths (m).
        bracket_dict: mapping from target_depth to bracket_search dict.
        depth0_method: 'level0' (P5) uses level 0 value directly.
        
    Returns:
        targets: float32 array of shape (len(target_depths), ny, nx) with NaN for invalid cells.
    """
    nz, ny, nx = vals_3d.shape
    n_targets = len(target_depths)
    level_depths = np.asarray(level_depths, dtype=np.float64)

    # Compute contiguous valid prefix K for all columns
    K_map = find_contiguous_valid_prefix_k(valid_levels_3d)

    # Output container
    targets = np.full((n_targets, ny, nx), np.nan, dtype=np.float32)

    # Flatten spatial grid for fast vectorized column grouping
    vals_2d = vals_3d.reshape(nz, ny * nx)
    K_flat = K_map.reshape(ny * nx)
    targets_2d = targets.reshape(n_targets, ny * nx)

    # P5: Depth 0 uses value of level 0 (valid iff level 0 is valid, i.e. K >= 0)
    for d_idx, z_t in enumerate(target_depths):
        b = bracket_dict[z_t]
        if b["is_depth0"]:
            valid_d0_cols = np.where(K_flat >= 0)[0]
            if len(valid_d0_cols) > 0:
                if depth0_method == "level0":
                    targets_2d[d_idx, valid_d0_cols] = vals_2d[0, valid_d0_cols].astype(np.float32)
                else:
                    raise ValueError(f"Unsupported depth0 method: {depth0_method}")

    # Group columns by unique K values for interior PCHIP evaluation
    unique_K = np.unique(K_flat)

    for k_val in unique_K:
        if k_val < 1:
            # Need at least 2 levels (0 and 1) for PCHIP interpolation
            # Handle exact match at level 0 if any
            for d_idx, z_t in enumerate(target_depths):
                b = bracket_dict[z_t]
                if b["is_exact"] and b["k0"] == 0 and k_val == 0:
                    col_idx = np.where(K_flat == 0)[0]
                    targets_2d[d_idx, col_idx] = vals_2d[0, col_idx].astype(np.float32)
            continue

        col_indices = np.where(K_flat == k_val)[0]
        if len(col_indices) == 0:
            continue

        # Active levels for this group: 0..k_val
        active_depths = level_depths[: k_val + 1]
        active_vals = vals_2d[: k_val + 1, col_indices]  # shape: (k_val + 1, n_cols)

        # Identify target depths whose deeper bracket k1 <= k_val (interior to valid prefix)
        depths_to_eval: List[float] = []
        target_indices_to_eval: List[int] = []

        for d_idx, z_t in enumerate(target_depths):
            b = bracket_dict[z_t]
            if b["is_depth0"]:
                # Already handled by P5 surface rule
                continue
            if b["k1"] <= k_val:
                depths_to_eval.append(z_t)
                target_indices_to_eval.append(d_idx)

        if not depths_to_eval:
            continue

        # Vectorized PCHIP across all columns in this group
        pchip = PchipInterpolator(active_depths, active_vals, axis=0)
        eval_res = pchip(depths_to_eval)  # shape: (len(depths_to_eval), n_cols)

        if len(depths_to_eval) == 1:
            eval_res = eval_res[np.newaxis, :]

        for i_sub, d_idx in enumerate(target_indices_to_eval):
            targets_2d[d_idx, col_indices] = eval_res[i_sub].astype(np.float32)

    return targets


# =============================================================================
# Main Day and Month Target Builders
# =============================================================================

def build_day_targets(
    raw_day: np.ndarray,
    static_wet: np.ndarray,
    exclusion_mask: np.ndarray,
    policy: Union[Dict[str, Any], str],
    native_lats: Optional[np.ndarray] = None,
    native_lons: Optional[np.ndarray] = None,
    native_depths: Optional[np.ndarray] = None,
    target_depths: Optional[List[float]] = None,
    weights_2d: Optional[np.ndarray] = None,
    bracket_dict: Optional[Dict[float, Dict[str, Any]]] = None,
    reference_finite: Optional[np.ndarray] = None,
    qc_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """Build project target fields for a single day according to validated policy decisions.
    
    Parameters:
        raw_day: raw int16 array of shape (nz, ny, nx)
        static_wet: boolean array of shape (nz, ny, nx), True for wet ocean cells
        exclusion_mask: boolean array of shape (nz, ny, nx), True for static exclusions
        policy: target policy dictionary or path to policy JSON
        native_lats: optional native latitude centers
        native_lons: optional native longitude centers
        native_depths: optional native level depths (m)
        target_depths: optional list of target depths (m)
        weights_2d: optional precomputed (4, 4) overlap weights
        bracket_dict: optional precomputed bracket mapping
        reference_finite: optional reference finite mask for QC rule 7
        qc_policy: optional preloaded QC policy dict
        
    Returns:
        targets: (ndepths, nlat_t, nlon_t) float32 with NaN where invalid
        valid_mask: (ndepths, nlat_t, nlon_t) uint8 (1=valid, 0=invalid)
        qc_summary: dict of QC status, counts, and unexpected cells
        f_T: (nz, nlat_t, nlon_t) float64 array of valid fraction per level
    """
    target_pol = load_target_policy(policy)
    nz, ny, nx = raw_day.shape

    # Load QC policy if not provided
    if qc_policy is None:
        qc_path = target_pol["qc_policy_path"]
        if not os.path.isabs(qc_path):
            qc_path = os.path.join(PROJECT_ROOT, qc_path)
        qc_policy = load_qc_policy(qc_path)

    enc = qc_policy["encoding"]
    scale_factor = float(enc["scale_factor"])
    add_offset = float(enc["add_offset"])

    # 1. P1: Run QC policy v1 on this day first
    qc_res = apply_qc(
        raw=raw_day,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=qc_policy,
        reference_finite=reference_finite,
        lats=native_lats,
        lons=native_lons,
        depths=native_depths,
    )

    if qc_res.status == "FAIL":
        unexp_summary = [
            f"cell at level={u.get('level')} (lat_idx={u.get('lat_idx')}, lon_idx={u.get('lon_idx')}, "
            f"lat={u.get('latitude')}, lon={u.get('longitude')}, raw={u.get('raw')}): {u.get('rule')}"
            for u in qc_res.unexpected[:10]
        ]
        more = f" ... and {len(qc_res.unexpected) - 10} more" if len(qc_res.unexpected) > 10 else ""
        raise RuntimeError(
            f"GLORYS target builder halted: QC policy v1 reported FAIL with "
            f"{len(qc_res.unexpected)} unexpected violating cells:\n"
            + "\n".join(unexp_summary) + more
        )

    # 2. Decode valid cells manually after masking fill
    thetao_decoded = np.where(
        qc_res.valid,
        raw_day.astype(np.float64) * scale_factor + add_offset,
        np.nan,
    )

    # 3. P2: Horizontal Method A regridding
    grid_cfg = target_pol["target_grid"]
    N = grid_cfg.get("n_lat", (ny - 1) // 3)
    M = grid_cfg.get("n_lon", (nx - 1) // 3)

    if weights_2d is None:
        lat_start = float(grid_cfg.get("lat_start", 5.0))
        lat_step = float(grid_cfg.get("lat_step", 0.25))
        lon_start = float(grid_cfg.get("lon_start", 45.0))
        lon_step = float(grid_cfg.get("lon_step", 0.25))

        if native_lats is None:
            native_lats = lat_start + np.arange(ny) * (1.0 / 12.0)
        if native_lons is None:
            native_lons = lon_start + np.arange(nx) * (1.0 / 12.0)

        w_lat_1d, _, _ = compute_overlap_weights_1d(native_lats, lat_start, lat_step, N)
        w_lon_1d, _, _ = compute_overlap_weights_1d(native_lons, lon_start, lon_step, M)
        weights_2d = compute_overlap_weights_2d(w_lat_1d, w_lon_1d)

    val_A, f_T = regrid_method_A(thetao_decoded, qc_res.valid, weights_2d=weights_2d)

    # 4. P3: Validity threshold tauT
    tauT = float(target_pol.get("tauT", 1.0))
    lvl_valid = (f_T >= tauT - 1e-12)

    # 5. P4 & P5: Vertical PCHIP and Depth 0
    t_depths = target_depths if target_depths is not None else target_pol["target_depths"]
    if native_depths is None:
        raise ValueError("native_depths must be provided for vertical interpolation.")

    if bracket_dict is None:
        bracket_dict = {z_t: bracket_search(native_depths, z_t) for z_t in t_depths}

    depth0_rule = str(target_pol.get("depth0", "level0"))
    targets = interpolate_column_pchip(
        level_depths=native_depths,
        vals_3d=val_A,
        valid_levels_3d=lvl_valid,
        target_depths=t_depths,
        bracket_dict=bracket_dict,
        depth0_method=depth0_rule,
    )

    valid_mask = (~np.isnan(targets)).astype(np.uint8)
    qc_summary = qc_res.to_dict()

    return targets, valid_mask, qc_summary, f_T


def build_month_file(
    glorys_path: str,
    static_path: str,
    out_path: str,
    policy: Union[Dict[str, Any], str],
    qc_policy_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Process an entire monthly GLORYS file, writing a target NetCDF and manifest JSON.
    
    Parameters:
        glorys_path: path to raw glorys monthly thetao file
        static_path: path to raw glorys static mask and bathymetry file
        out_path: path for destination NetCDF file
        policy: target policy dictionary or path to target policy JSON
        qc_policy_path: optional path to QC policy JSON
        verbose: whether to print progress to stdout
        
    Returns:
        manifest: dictionary containing build summary, metadata, and per-day stats.
    """
    target_pol = load_target_policy(policy)
    t_depths = [float(z) for z in target_pol["target_depths"]]
    grid_cfg = target_pol["target_grid"]
    N = int(grid_cfg["n_lat"])
    M = int(grid_cfg["n_lon"])
    lat_start = float(grid_cfg["lat_start"])
    lon_start = float(grid_cfg["lon_start"])
    lat_step = float(grid_cfg["lat_step"])
    lon_step = float(grid_cfg["lon_step"])

    # Resolve QC policy path
    if qc_policy_path is None:
        qc_policy_path = target_pol["qc_policy_path"]
        if not os.path.isabs(qc_policy_path):
            qc_policy_path = os.path.join(PROJECT_ROOT, qc_policy_path)
    qc_pol = load_qc_policy(qc_policy_path)

    # Capture file hashes before reading
    sha_glorys = compute_file_sha256(glorys_path)
    sha_static = compute_file_sha256(static_path)
    sha_qc_pol = compute_file_sha256(qc_policy_path)
    this_script_path = os.path.abspath(__file__)
    sha_builder = compute_file_sha256(this_script_path)
    git_commit = get_git_commit_hash(PROJECT_ROOT)

    # Load coordinates from source NetCDF
    native_lats, native_lons, native_depths = load_coordinates(glorys_path)
    static_wet = load_static_wet_mask(
        static_path, ref_lats=native_lats, ref_lons=native_lons, ref_depths=native_depths
    )
    exclusion_mask = locate_exclusions(native_lats, native_lons, qc_pol)

    # Precompute horizontal overlap weights and bracket search
    w_lat_1d, _, _ = compute_overlap_weights_1d(native_lats, lat_start, lat_step, N)
    w_lon_1d, _, _ = compute_overlap_weights_1d(native_lons, lon_start, lon_step, M)
    w_2d = compute_overlap_weights_2d(w_lat_1d, w_lon_1d)
    bracket_dict = {z_t: bracket_search(native_depths, z_t) for z_t in t_depths}

    # Determine number of days in source file and load time coordinate
    with xr.open_dataset(glorys_path, decode_times=False) as ds_src:
        time_var = ds_src["time"].copy()
        time_values = ds_src["time"].values
        time_attrs = dict(ds_src["time"].attrs)
        n_days = len(time_values)

    target_lats = (lat_start + 0.5 * lat_step) + np.arange(N) * lat_step
    target_lons = (lon_start + 0.5 * lon_step) + np.arange(M) * lon_step

    # Output storage
    targets_all = np.full((n_days, len(t_depths), N, M), np.nan, dtype=np.float32)
    valid_mask_all = np.zeros((n_days, len(t_depths), N, M), dtype=np.uint8)

    per_day_manifest: List[Dict[str, Any]] = []
    pooled_valid_counts = {float(z_t): 0 for z_t in t_depths}

    if verbose:
        print(f"Building targets for {n_days} days into {out_path}...")

    for d in range(n_days):
        raw_day = read_raw_day(glorys_path, day_index=d, policy=qc_pol)
        targets_d, mask_d, qc_summary_d, _ = build_day_targets(
            raw_day=raw_day,
            static_wet=static_wet,
            exclusion_mask=exclusion_mask,
            policy=target_pol,
            native_lats=native_lats,
            native_lons=native_lons,
            native_depths=native_depths,
            target_depths=t_depths,
            weights_2d=w_2d,
            bracket_dict=bracket_dict,
            qc_policy=qc_pol,
        )

        targets_all[d] = targets_d
        valid_mask_all[d] = mask_d

        # Counts per depth for this day
        depth_counts = {}
        for z_idx, z_t in enumerate(t_depths):
            cnt = int(np.sum(mask_d[z_idx] == 1))
            depth_counts[str(z_t)] = cnt
            pooled_valid_counts[z_t] += cnt

        day_info = {
            "day_index": d,
            "qc_status": qc_summary_d["status"],
            "qc_counts": qc_summary_d["counts"],
            "unexpected_count": qc_summary_d["unexpected_count"],
            "valid_cells_by_depth": depth_counts,
        }
        per_day_manifest.append(day_info)

        if verbose:
            print(f"  Day {d:02d}/{n_days-1}: QC={qc_summary_d['status']}, valid_d0={depth_counts[str(t_depths[0])]}")

    # Build destination xarray Dataset
    coords = {
        "time": ("time", time_values, time_attrs),
        "depth": (
            "depth",
            np.array(t_depths, dtype=np.float32),
            {
                "long_name": "Depth",
                "standard_name": "depth",
                "units": "m",
                "unit_long": "Meters",
                "positive": "down",
                "axis": "Z",
            },
        ),
        "latitude": (
            "latitude",
            np.array(target_lats, dtype=np.float32),
            {
                "long_name": "Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "unit_long": "Degrees North",
                "axis": "Y",
            },
        ),
        "longitude": (
            "longitude",
            np.array(target_lons, dtype=np.float32),
            {
                "long_name": "Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "unit_long": "Degrees East",
                "axis": "X",
            },
        ),
    }

    data_vars = {
        "thetao_target": (
            ("time", "depth", "latitude", "longitude"),
            targets_all,
            {
                "long_name": "Target Sea Water Potential Temperature",
                "standard_name": "sea_water_potential_temperature",
                "units": "degrees_C",
                "unit_long": "Degrees Celsius",
            },
        ),
        "valid_mask": (
            ("time", "depth", "latitude", "longitude"),
            valid_mask_all,
            {
                "long_name": "Target Validity Mask (1=valid, 0=invalid)",
                "flag_values": np.array([0, 1], dtype=np.uint8),
                "flag_meanings": "invalid valid",
            },
        ),
    }

    created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    global_attrs = {
        "policy_name": str(target_pol.get("policy_name", "glorys_target_policy_v1")),
        "policy_version": str(target_pol.get("policy_version", "v1")),
        "tauT": float(target_pol.get("tauT", 1.0)),
        "horizontal_method": str(target_pol.get("horizontal_method", "area_weighted_valid")),
        "vertical_method": str(target_pol.get("vertical_method", "pchip_valid_prefix")),
        "depth0_method": str(target_pol.get("depth0", "level0")),
        "seabed_gap": str(target_pol.get("seabed_gap", "missing")),
        "raw_glorys_sha256": sha_glorys,
        "raw_static_sha256": sha_static,
        "qc_policy_sha256": sha_qc_pol,
        "builder_script_sha256": sha_builder,
        "git_commit": git_commit,
        "created_utc": created_utc,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "xarray_version": xr.__version__,
    }

    ds_target = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attrs)

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    encoding = {
        "thetao_target": {
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.float32(np.nan),
            "dtype": "float32",
        },
        "valid_mask": {
            "zlib": True,
            "complevel": 4,
            "dtype": "uint8",
        },
        "time": {
            "dtype": time_var.dtype,
        },
    }

    ds_target.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    ds_target.close()

    file_size = os.path.getsize(out_path)
    file_sha256 = compute_file_sha256(out_path)

    # Write manifest JSON next to the output NetCDF
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    manifest_data = {
        "output_file": {
            "path": os.path.abspath(out_path),
            "size_bytes": file_size,
            "sha256": file_sha256,
        },
        "input_files": {
            "glorys_thetao": {
                "path": os.path.abspath(glorys_path),
                "sha256": sha_glorys,
            },
            "glorys_static": {
                "path": os.path.abspath(static_path),
                "sha256": sha_static,
            },
            "qc_policy": {
                "path": os.path.abspath(qc_policy_path),
                "sha256": sha_qc_pol,
            },
        },
        "policy": target_pol,
        "target_depths": t_depths,
        "pooled_valid_counts": {str(z): pooled_valid_counts[z] for z in t_depths},
        "per_day": per_day_manifest,
        "metadata": global_attrs,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    if verbose:
        print(f"Successfully generated {out_path} ({file_size} bytes)")
        print(f"Saved manifest: {manifest_path}")

    return manifest_data


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLORYS Target Builder v1")
    parser.add_argument(
        "--glorys-path",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "glorys", "glorys_thetao_2019-01.nc"),
        help="Path to raw GLORYS thetao NetCDF",
    )
    parser.add_argument(
        "--static-path",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "glorys_static", "glorys_static_mask_bathy.nc"),
        help="Path to static bathymetry/mask NetCDF",
    )
    parser.add_argument(
        "--policy-path",
        default=os.path.join(PROJECT_ROOT, "configs", "glorys_target_policy_v1.json"),
        help="Path to target policy JSON",
    )
    parser.add_argument(
        "--out-path",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "glorys_targets", "glorys_targets_2019-01.nc"),
        help="Path for output processed NetCDF",
    )
    args = parser.parse_args()

    build_month_file(
        glorys_path=args.glorys_path,
        static_path=args.static_path,
        out_path=args.out_path,
        policy=args.policy_path,
    )


if __name__ == "__main__":
    main()
