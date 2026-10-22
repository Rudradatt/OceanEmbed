#!/usr/bin/env python3
"""
GLORYS Horizontal Regridding Consequence Experiment
(Diagnostic Only; Comparative Evidence Gathering; Read-Only on Raw Data)

Compares two candidate horizontal regridding methods from native GLORYS
(1/12 deg, 301x721) to project target grid (0.25 deg, 100x240):
  - Method L: strict bilinear interpolation at target centers from the 4
    surrounding native centers (weight 0.25 each; NaN if any is invalid).
  - Method A: exact area-weighted mean over the 16 overlapping native cells
    using 2D overlap weights outer([1/6, 1/3, 1/3, 1/6], [1/6, 1/3, 1/3, 1/6])
    over QC-valid cells; valid fraction f = sum(weights of valid cells).

Outputs generated in outputs/diagnostics/glorys_regrid_experiment/:
  - coverage_by_level.csv
  - diff_by_level.csv
  - top_diff_cells.csv
  - ab_contamination.csv
  - rmse_vs_depth.png
  - coverage_vs_depth.png
  - diff_map_d0_l0.png
  - glorys_regrid_experiment.json
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
import xarray as xr

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


# =============================================================================
# Constants & Defaults
# =============================================================================

DEFAULT_THETAO_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "glorys", "glorys_thetao_2019-01.nc")
DEFAULT_STATIC_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "glorys_static", "glorys_static_mask_bathy.nc")
DEFAULT_POLICY_PATH = os.path.join(PROJECT_ROOT, "configs", "glorys_qc_policy_v1.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "diagnostics", "glorys_regrid_experiment")

EXPECTED_SHA256_THETAO = "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f"
EXPECTED_SHA256_STATIC = "fdb0c5fdc282180c0b5b58fa689753df79c22f69f4dc6dac64314a8cb82a79a5"

TARGET_N_LAT = 100
TARGET_N_LON = 240
TARGET_STEP = 0.25
TARGET_LAT_START = 5.0
TARGET_LON_START = 45.0
NATIVE_STEP = 1.0 / 12.0


# =============================================================================
# File Integrity Helpers
# =============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file without reading the whole file into RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """Retrieve SHA256, file size in bytes, and modification times."""
    stat = os.stat(filepath)
    sha256_val = compute_file_sha256(filepath)
    return {
        "path": os.path.abspath(filepath),
        "sha256": sha256_val,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_ns": stat.st_mtime_ns,
    }


# =============================================================================
# STEP 1: Overlap Weights and Vectorized Regridding
# =============================================================================

def compute_overlap_weights_1d(
    native_centers: np.ndarray,
    target_start: float,
    target_step: float,
    n_target: int,
    native_step: float = 1.0 / 12.0,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, List[List[int]], List[np.ndarray]]:
    """Compute 1D area-overlap weights from native and target edges (not hardcoded).
    
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


def regrid_method_L(
    field: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Method L: bilinear interpolation of the QC-valid native field at target centers.

    Evaluates from the 4 surrounding native centers (indices 3i+1, 3i+2, 3j+1, 3j+2),
    each with weight 0.25. The result is NaN if any of the 4 is invalid (strict).
    Uses numpy slicing across the whole array (no per-cell Python loops).

    Parameters:
        field: float array of shape (..., ny, nx) where ny = 3*N + 1, nx = 3*M + 1
        valid_mask: bool array of shape (..., ny, nx)

    Returns:
        regridded: float64 array of shape (..., N, M)
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

    # Slice ranges for indices 3i+1, 3i+2 and 3j+1, 3j+2
    s_lat_1 = slice(1, 3 * N - 1, 3)
    s_lat_2 = slice(2, 3 * N, 3)
    s_lon_1 = slice(1, 3 * M - 1, 3)
    s_lon_2 = slice(2, 3 * M, 3)

    v00 = valid_mask[..., s_lat_1, s_lon_1]
    v01 = valid_mask[..., s_lat_1, s_lon_2]
    v10 = valid_mask[..., s_lat_2, s_lon_1]
    v11 = valid_mask[..., s_lat_2, s_lon_2]

    valid_L = v00 & v01 & v10 & v11

    # Safe evaluation to avoid NaN-propagation from invalid cells
    p00 = np.where(valid_L, field[..., s_lat_1, s_lon_1], 0.0)
    p01 = np.where(valid_L, field[..., s_lat_1, s_lon_2], 0.0)
    p10 = np.where(valid_L, field[..., s_lat_2, s_lon_1], 0.0)
    p11 = np.where(valid_L, field[..., s_lat_2, s_lon_2], 0.0)

    val_L = np.where(valid_L, 0.25 * (p00 + p01 + p10 + p11), np.nan)
    return val_L


def compute_valid_fraction_f(
    valid_mask: np.ndarray,
    weights_2d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute valid area fraction f over the 16-cell footprint for each target cell.

    Parameters:
        valid_mask: bool array of shape (..., ny, nx) where ny = 3*N + 1, nx = 3*M + 1
        weights_2d: optional (4, 4) array of weights, default outer([1/6, 1/3, 1/3, 1/6], [1/6, 1/3, 1/3, 1/6])

    Returns:
        f: float64 array of shape (..., N, M)
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    ny, nx = valid_mask.shape[-2], valid_mask.shape[-1]
    if (ny - 1) % 3 != 0 or (nx - 1) % 3 != 0:
        raise ValueError(f"Spatial dimensions must be 3*N+1 and 3*M+1, got ({ny}, {nx})")

    N = (ny - 1) // 3
    M = (nx - 1) // 3

    if weights_2d is None:
        w_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
        weights_2d = np.outer(w_1d, w_1d)
    else:
        weights_2d = np.asarray(weights_2d, dtype=np.float64)

    f_shape = valid_mask.shape[:-2] + (N, M)
    f = np.zeros(f_shape, dtype=np.float64)

    for m in range(4):
        s_lat = slice(m, 3 * N - 2 + m, 3)
        for n in range(4):
            s_lon = slice(n, 3 * M - 2 + n, 3)
            w = weights_2d[m, n]
            f += w * valid_mask[..., s_lat, s_lon]

    return f


def regrid_method_A(
    field: np.ndarray,
    valid_mask: np.ndarray,
    weights_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Method A: exact area-weighted mean over the 16 overlapping native cells using 2D weights.

    Uses only QC-valid cells:
        value = sum(w * x over valid) / sum(w over valid)
        f = sum(w over valid)
    Returns (A, f) where A is NaN if f == 0.
    Uses numpy slicing across the whole array (no per-cell Python loops).

    Parameters:
        field: float array of shape (..., ny, nx) where ny = 3*N + 1, nx = 3*M + 1
        valid_mask: bool array of shape (..., ny, nx)
        weights_2d: optional (4, 4) array of weights

    Returns:
        val_A: float64 array of shape (..., N, M)
        f: float64 array of shape (..., N, M)
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
    f = np.zeros(out_shape, dtype=np.float64)

    for m in range(4):
        s_lat = slice(m, 3 * N - 2 + m, 3)
        for n in range(4):
            s_lon = slice(n, 3 * M - 2 + n, 3)
            w = weights_2d[m, n]
            vm = valid_mask[..., s_lat, s_lon]
            val = field[..., s_lat, s_lon]

            f += w * vm
            safe_val = np.where(vm, val, 0.0)
            sum_wx += w * safe_val

    val_A = np.where(f > 0.0, sum_wx / np.where(f > 0.0, f, 1.0), np.nan)
    return val_A, f


# =============================================================================
# STEP 3: Real Data Execution & Diagnostic Analysis
# =============================================================================

def locate_columns_ab(
    lats: np.ndarray,
    lons: np.ndarray,
    tol: float = 1e-3,
) -> Dict[str, Tuple[int, int, float, float]]:
    """Locate Column A (lat 15.0, lon 50.5833) and Column B (lat 15.0, lon 50.6667)."""
    targets = {
        "Column_A": (15.0, 50.58333),
        "Column_B": (15.0, 50.66667),
    }
    coords: Dict[str, Tuple[int, int, float, float]] = {}

    for name, (t_lat, t_lon) in targets.items():
        dist_lat = np.abs(lats - t_lat)
        dist_lon = np.abs(lons - t_lon)

        min_dlat = float(np.min(dist_lat))
        min_dlon = float(np.min(dist_lon))

        if min_dlat > tol or min_dlon > tol:
            raise ValueError(f"Could not locate {name} at ({t_lat}, {t_lon}) within tol {tol}")

        r_idx = int(np.argmin(dist_lat))
        c_idx = int(np.argmin(dist_lon))
        coords[name] = (r_idx, c_idx, float(lats[r_idx]), float(lons[c_idx]))

    return coords


def find_target_cells_covering_columns(
    col_coords: Dict[str, Tuple[int, int, float, float]],
    n_target_lat: int = TARGET_N_LAT,
    n_target_lon: int = TARGET_N_LON,
) -> List[Dict[str, Any]]:
    """Find target cells whose 16-cell footprint includes Column A or Column B."""
    w_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])

    # Store footprint info per target cell
    affected_cells: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for col_name, (r_idx, c_idx, lat_val, lon_val) in col_coords.items():
        for i in range(n_target_lat):
            lat_indices = [3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3]
            if r_idx in lat_indices:
                lat_pos = lat_indices.index(r_idx)
                w_r = w_1d[lat_pos]

                for j in range(n_target_lon):
                    lon_indices = [3 * j, 3 * j + 1, 3 * j + 2, 3 * j + 3]
                    if c_idx in lon_indices:
                        lon_pos = lon_indices.index(c_idx)
                        w_c = w_1d[lon_pos]
                        w_2d = w_r * w_c

                        cell_key = (i, j)
                        if cell_key not in affected_cells:
                            affected_cells[cell_key] = {
                                "target_i": i,
                                "target_j": j,
                                "target_lat": 5.125 + 0.25 * i,
                                "target_lon": 45.125 + 0.25 * j,
                                "weight_A": 0.0,
                                "weight_B": 0.0,
                                "in_Method_L_A": False,
                                "in_Method_L_B": False,
                            }

                        in_L = (lat_pos in (1, 2)) and (lon_pos in (1, 2))
                        if col_name == "Column_A":
                            affected_cells[cell_key]["weight_A"] = float(w_2d)
                            affected_cells[cell_key]["in_Method_L_A"] = bool(in_L)
                        else:
                            affected_cells[cell_key]["weight_B"] = float(w_2d)
                            affected_cells[cell_key]["in_Method_L_B"] = bool(in_L)

    result = list(affected_cells.values())
    result.sort(key=lambda x: (x["target_i"], x["target_j"]))
    return result


def run_equivalence_check_day0(
    thetao_decoded: np.ndarray,
    val_L: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    levels: List[int] = [0, 10, 28],
) -> Dict[str, Any]:
    """Equivalence check: compare Method L with xarray linear interp on day 0."""
    target_lats = TARGET_LAT_START + 0.125 + TARGET_STEP * np.arange(TARGET_N_LAT)
    target_lons = TARGET_LON_START + 0.125 + TARGET_STEP * np.arange(TARGET_N_LON)

    eq_results: Dict[str, Any] = {}

    for lvl in levels:
        t_slice = thetao_decoded[lvl]
        l_slice = val_L[lvl]

        da = xr.DataArray(
            t_slice,
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )
        da_interp = da.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        ).values

        finite_L = np.isfinite(l_slice)
        finite_xr = np.isfinite(da_interp)

        both_finite = finite_L & finite_xr
        one_nan = finite_L != finite_xr

        if np.any(both_finite):
            max_abs_diff = float(np.max(np.abs(l_slice[both_finite] - da_interp[both_finite])))
        else:
            max_abs_diff = 0.0

        eq_results[f"level_{lvl}"] = {
            "level": lvl,
            "both_finite_count": int(np.sum(both_finite)),
            "one_nan_count": int(np.sum(one_nan)),
            "max_abs_diff": max_abs_diff,
            "equivalent": bool(max_abs_diff < 1e-12 and int(np.sum(one_nan)) == 0),
        }

    return eq_results


def run_regrid_experiment(
    thetao_path: str = DEFAULT_THETAO_PATH,
    static_path: str = DEFAULT_STATIC_PATH,
    policy_path: str = DEFAULT_POLICY_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Run full horizontal regridding experiment across all 31 days and 36 levels."""
    os.makedirs(output_dir, exist_ok=True)

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

    # 2. Load policy, coordinates, static wet mask, and static exclusions
    policy = load_policy(policy_path)
    lats, lons, depths = load_coordinates(thetao_path)
    static_wet = load_static_wet_mask(static_path, ref_lats=lats, ref_lons=lons, ref_depths=depths)
    exclusion_mask = locate_exclusions(lats, lons, policy)

    enc = policy["encoding"]
    scale_factor = float(enc["scale_factor"])
    add_offset = float(enc["add_offset"])
    fill_raw = int(enc["fill_raw"])
    levels_used = policy.get("levels_used", 36)

    # 3. Compute overlap weights from edges
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

    # 4. Locate Column A & B footprints
    col_coords = locate_columns_ab(lats, lons)
    affected_cells = find_target_cells_covering_columns(col_coords)

    # Accumulators across 31 days
    total_target_cells = TARGET_N_LAT * TARGET_N_LON

    # Daily coverage counters: shape (36,)
    cov_counts_day0: List[Dict[str, Any]] = []
    cov_counts_pooled = {
        "n_f_eq_1": np.zeros(levels_used, dtype=np.int64),
        "n_f_between_0_and_1": np.zeros(levels_used, dtype=np.int64),
        "n_f_eq_0": np.zeros(levels_used, dtype=np.int64),
        "n_valid_L": np.zeros(levels_used, dtype=np.int64),
        "n_f_ge_05": np.zeros(levels_used, dtype=np.int64),
        "n_f_ge_075": np.zeros(levels_used, dtype=np.int64),
        "n_f_ge_09": np.zeros(levels_used, dtype=np.int64),
    }

    # Differences accumulators per level (pooled over days)
    diffs_all_by_level: List[List[np.ndarray]] = [[] for _ in range(levels_used)]
    diffs_f1_by_level: List[List[np.ndarray]] = [[] for _ in range(levels_used)]
    diffs_flt1_by_level: List[List[np.ndarray]] = [[] for _ in range(levels_used)]

    # Containers for outputs
    top_diff_records: List[Dict[str, Any]] = []
    ab_contamination_records: List[Dict[str, Any]] = []
    eq_results_day0: Dict[str, Any] = {}
    day0_level0_abs_diff_map: Optional[np.ndarray] = None

    print(f"Beginning 31-day GLORYS regridding experiment across {levels_used} levels...")

    for d in range(31):
        print(f"  Processing day {d:02d}/30...", end="", flush=True)

        # Read raw day slice
        raw_day = read_raw_day(thetao_path, day_index=d, policy=policy)

        # QC valid mask with policy exclusions applied
        qc_res = apply_qc(raw_day, static_wet, exclusion_mask, policy, lats=lats, lons=lons, depths=depths)
        valid_with_qc = qc_res.valid

        # Decode valid cells manually
        thetao_decoded = np.where(
            valid_with_qc,
            raw_day.astype(np.float64) * scale_factor + add_offset,
            np.nan,
        )

        # Compute Method L and Method A
        val_L = regrid_method_L(thetao_decoded, valid_with_qc)
        val_A, f = regrid_method_A(thetao_decoded, valid_with_qc, w_2d)

        # Verify coverage invariants
        # 1. Every cell valid under L must have f > 0
        valid_L_mask = ~np.isnan(val_L)
        viol_L_f0 = valid_L_mask & (f <= 0.0)
        if np.any(viol_L_f0):
            raise AssertionError(f"Day {d}: {np.sum(viol_L_f0)} cells valid under L have f <= 0")

        # 2. Every cell with f == 1 must be valid under L
        f1_mask = np.isclose(f, 1.0, atol=1e-12)
        viol_f1_notL = f1_mask & (~valid_L_mask)
        if np.any(viol_f1_notL):
            raise AssertionError(f"Day {d}: {np.sum(viol_f1_notL)} cells with f == 1 are not valid under L")

        # Accumulate coverage counts
        for lvl in range(levels_used):
            f_lvl = f[lvl]
            L_lvl = val_L[lvl]

            f_eq_1 = int(np.sum(np.isclose(f_lvl, 1.0, atol=1e-12)))
            f_eq_0 = int(np.sum(np.isclose(f_lvl, 0.0, atol=1e-12)))
            f_between = int(np.sum((f_lvl > 1e-12) & (f_lvl < 1.0 - 1e-12)))
            valid_L_count = int(np.sum(~np.isnan(L_lvl)))
            f_ge_05 = int(np.sum(f_lvl >= 0.5 - 1e-12))
            f_ge_075 = int(np.sum(f_lvl >= 0.75 - 1e-12))
            f_ge_09 = int(np.sum(f_lvl >= 0.9 - 1e-12))

            cov_counts_pooled["n_f_eq_1"][lvl] += f_eq_1
            cov_counts_pooled["n_f_between_0_and_1"][lvl] += f_between
            cov_counts_pooled["n_f_eq_0"][lvl] += f_eq_0
            cov_counts_pooled["n_valid_L"][lvl] += valid_L_count
            cov_counts_pooled["n_f_ge_05"][lvl] += f_ge_05
            cov_counts_pooled["n_f_ge_075"][lvl] += f_ge_075
            cov_counts_pooled["n_f_ge_09"][lvl] += f_ge_09

            if d == 0:
                cov_counts_day0.append({
                    "level": lvl,
                    "depth_m": float(depths[lvl]),
                    "n_target_total": total_target_cells,
                    "n_f_eq_1": f_eq_1,
                    "n_f_between_0_and_1": f_between,
                    "n_f_eq_0": f_eq_0,
                    "n_valid_L": valid_L_count,
                    "n_f_ge_05": f_ge_05,
                    "n_f_ge_075": f_ge_075,
                    "n_f_ge_09": f_ge_09,
                })

        # Method differences over cells valid under both methods
        for lvl in range(levels_used):
            mask_both = (~np.isnan(val_L[lvl])) & (~np.isnan(val_A[lvl]))
            if np.any(mask_both):
                diff_vals = val_A[lvl][mask_both] - val_L[lvl][mask_both]
                f_both = f[lvl][mask_both]
                f1_subset = np.isclose(f_both, 1.0, atol=1e-12)

                diffs_all_by_level[lvl].append(diff_vals)
                diffs_f1_by_level[lvl].append(diff_vals[f1_subset])
                diffs_flt1_by_level[lvl].append(diff_vals[~f1_subset])

        # Day 0 specific checks
        if d == 0:
            # Equivalence check
            eq_results_day0 = run_equivalence_check_day0(
                thetao_decoded, val_L, lats, lons, levels=[0, 10, 28]
            )

            # Map of |A - L| on Day 0 Level 0
            day0_level0_abs_diff_map = np.abs(val_A[0] - val_L[0])

            # Top 20 diff cells for levels 0, 10, 28
            target_lats_grid = TARGET_LAT_START + 0.125 + TARGET_STEP * np.arange(TARGET_N_LAT)
            target_lons_grid = TARGET_LON_START + 0.125 + TARGET_STEP * np.arange(TARGET_N_LON)

            for check_lvl in [0, 10, 28]:
                mask_both = (~np.isnan(val_L[check_lvl])) & (~np.isnan(val_A[check_lvl]))
                diff_lvl = val_A[check_lvl] - val_L[check_lvl]
                abs_diff_lvl = np.abs(diff_lvl)

                valid_indices = np.where(mask_both)
                rows = valid_indices[0]
                cols = valid_indices[1]
                abs_diff_valid = abs_diff_lvl[valid_indices]

                # Top 20 indices
                sorted_order = np.argsort(-abs_diff_valid)[:20]

                for rank, idx in enumerate(sorted_order, start=1):
                    r = int(rows[idx])
                    c = int(cols[idx])
                    top_diff_records.append({
                        "level": check_lvl,
                        "depth_m": float(depths[check_lvl]),
                        "rank": rank,
                        "target_i": r,
                        "target_j": c,
                        "target_lat": float(target_lats_grid[r]),
                        "target_lon": float(target_lons_grid[c]),
                        "f": float(f[check_lvl, r, c]),
                        "val_A": float(val_A[check_lvl, r, c]),
                        "val_L": float(val_L[check_lvl, r, c]),
                        "abs_diff": float(abs_diff_lvl[r, c]),
                        "diff_A_minus_L": float(diff_lvl[r, c]),
                    })

        # A/B contamination check (levels 28, 29, 30)
        # Compute without QC exclusion
        valid_without_qc = (raw_day != fill_raw) & static_wet
        thetao_without_qc = np.where(
            valid_without_qc,
            raw_day.astype(np.float64) * scale_factor + add_offset,
            np.nan,
        )
        val_L_noqc = regrid_method_L(thetao_without_qc, valid_without_qc)
        val_A_noqc, f_noqc = regrid_method_A(thetao_without_qc, valid_without_qc, w_2d)

        for lvl_ab in [28, 29, 30]:
            for cell_info in affected_cells:
                ti = cell_info["target_i"]
                tj = cell_info["target_j"]

                ab_contamination_records.append({
                    "day": d,
                    "level": lvl_ab,
                    "depth_m": float(depths[lvl_ab]),
                    "target_i": ti,
                    "target_j": tj,
                    "target_lat": cell_info["target_lat"],
                    "target_lon": cell_info["target_lon"],
                    "weight_A": cell_info["weight_A"],
                    "weight_B": cell_info["weight_B"],
                    "in_Method_L_A": cell_info["in_Method_L_A"],
                    "in_Method_L_B": cell_info["in_Method_L_B"],
                    "f_with_qc": float(f[lvl_ab, ti, tj]),
                    "val_A_with_qc": float(val_A[lvl_ab, ti, tj]),
                    "val_L_with_qc": float(val_L[lvl_ab, ti, tj]),
                    "f_without_qc": float(f_noqc[lvl_ab, ti, tj]),
                    "val_A_without_qc": float(val_A_noqc[lvl_ab, ti, tj]),
                    "val_L_without_qc": float(val_L_noqc[lvl_ab, ti, tj]),
                    "delta_A_unexcluded_minus_qc": float(val_A_noqc[lvl_ab, ti, tj] - val_A[lvl_ab, ti, tj]),
                    "delta_L_unexcluded_minus_qc": float(val_L_noqc[lvl_ab, ti, tj] - val_L[lvl_ab, ti, tj]),
                    "delta_f_unexcluded_minus_qc": float(f_noqc[lvl_ab, ti, tj] - f[lvl_ab, ti, tj]),
                })

        print(" done.")

    # -------------------------------------------------------------------------
    # Build Output DataFrames and CSVs
    # -------------------------------------------------------------------------

    # 1. coverage_by_level.csv
    cov_rows = []
    for lvl in range(levels_used):
        d0_row = cov_counts_day0[lvl]
        cov_rows.append({
            "level": lvl,
            "depth_m": float(depths[lvl]),
            "n_target_total": total_target_cells,
            # Day 0 counts
            "d0_n_f_eq_1": d0_row["n_f_eq_1"],
            "d0_n_f_between_0_and_1": d0_row["n_f_between_0_and_1"],
            "d0_n_f_eq_0": d0_row["n_f_eq_0"],
            "d0_n_valid_L": d0_row["n_valid_L"],
            "d0_n_f_ge_05": d0_row["n_f_ge_05"],
            "d0_n_f_ge_075": d0_row["n_f_ge_075"],
            "d0_n_f_ge_09": d0_row["n_f_ge_09"],
            # Pooled 31 days counts
            "pooled_n_f_eq_1": int(cov_counts_pooled["n_f_eq_1"][lvl]),
            "pooled_n_f_between_0_and_1": int(cov_counts_pooled["n_f_between_0_and_1"][lvl]),
            "pooled_n_f_eq_0": int(cov_counts_pooled["n_f_eq_0"][lvl]),
            "pooled_n_valid_L": int(cov_counts_pooled["n_valid_L"][lvl]),
            "pooled_n_f_ge_05": int(cov_counts_pooled["n_f_ge_05"][lvl]),
            "pooled_n_f_ge_075": int(cov_counts_pooled["n_f_ge_075"][lvl]),
            "pooled_n_f_ge_09": int(cov_counts_pooled["n_f_ge_09"][lvl]),
        })

    df_coverage = pd.DataFrame(cov_rows)
    coverage_csv_path = os.path.join(output_dir, "coverage_by_level.csv")
    df_coverage.to_csv(coverage_csv_path, index=False)
    print(f"Saved: {coverage_csv_path}")

    # 2. diff_by_level.csv (pooled over days)
    diff_rows = []
    for lvl in range(levels_used):
        all_d = np.concatenate(diffs_all_by_level[lvl]) if diffs_all_by_level[lvl] else np.array([])
        f1_d = np.concatenate(diffs_f1_by_level[lvl]) if diffs_f1_by_level[lvl] else np.array([])
        flt1_d = np.concatenate(diffs_flt1_by_level[lvl]) if diffs_flt1_by_level[lvl] else np.array([])

        def _stats(arr: np.ndarray) -> Dict[str, Any]:
            if len(arr) == 0:
                return {
                    "count": 0,
                    "bias": None,
                    "rmse": None,
                    "mae": None,
                    "p99": None,
                    "max_abs": None,
                }
            abs_arr = np.abs(arr)
            return {
                "count": len(arr),
                "bias": float(np.mean(arr)),
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mae": float(np.mean(abs_arr)),
                "p99": float(np.percentile(abs_arr, 99)),
                "max_abs": float(np.max(abs_arr)),
            }

        st_all = _stats(all_d)
        st_f1 = _stats(f1_d)
        st_flt1 = _stats(flt1_d)

        diff_rows.append({
            "level": lvl,
            "depth_m": float(depths[lvl]),
            # Over all valid both
            "all_count": st_all["count"],
            "all_bias": st_all["bias"],
            "all_rmse": st_all["rmse"],
            "all_mae": st_all["mae"],
            "all_p99": st_all["p99"],
            "all_max_abs": st_all["max_abs"],
            # Separately for f == 1
            "f1_count": st_f1["count"],
            "f1_bias": st_f1["bias"],
            "f1_rmse": st_f1["rmse"],
            "f1_mae": st_f1["mae"],
            "f1_p99": st_f1["p99"],
            "f1_max_abs": st_f1["max_abs"],
            # Separately for f < 1
            "flt1_count": st_flt1["count"],
            "flt1_bias": st_flt1["bias"],
            "flt1_rmse": st_flt1["rmse"],
            "flt1_mae": st_flt1["mae"],
            "flt1_p99": st_flt1["p99"],
            "flt1_max_abs": st_flt1["max_abs"],
        })

    df_diff = pd.DataFrame(diff_rows)
    diff_csv_path = os.path.join(output_dir, "diff_by_level.csv")
    df_diff.to_csv(diff_csv_path, index=False)
    print(f"Saved: {diff_csv_path}")

    # 3. top_diff_cells.csv
    df_top_diff = pd.DataFrame(top_diff_records)
    top_diff_csv_path = os.path.join(output_dir, "top_diff_cells.csv")
    df_top_diff.to_csv(top_diff_csv_path, index=False)
    print(f"Saved: {top_diff_csv_path}")

    # 4. ab_contamination.csv
    df_ab = pd.DataFrame(ab_contamination_records)
    ab_csv_path = os.path.join(output_dir, "ab_contamination.csv")
    df_ab.to_csv(ab_csv_path, index=False)
    print(f"Saved: {ab_csv_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # 1. RMSE(A - L) versus depth
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.plot(df_diff["all_rmse"], df_diff["depth_m"], "o-", color="#1f77b4", label="All valid cells", linewidth=2)
    ax.plot(df_diff["f1_rmse"], df_diff["depth_m"], "s--", color="#2ca02c", label="f = 1 (full footprint)", alpha=0.85)
    ax.plot(df_diff["flt1_rmse"], df_diff["depth_m"], "^:", color="#d62728", label="f < 1 (partial footprint)", alpha=0.85)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("RMSE |Method A - Method L| (°C)")
    ax.set_title("GLORYS Regridding: RMSE(A - L) vs Depth (Pooled 31 Days)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    rmse_plot_path = os.path.join(output_dir, "rmse_vs_depth.png")
    fig.savefig(rmse_plot_path)
    plt.close(fig)
    print(f"Saved plot: {rmse_plot_path}")

    # 2. Coverage fractions versus depth
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.plot(df_coverage["d0_n_valid_L"] / total_target_cells, df_coverage["depth_m"], "o-", color="#1f77b4", label="Valid under L", linewidth=2)
    ax.plot(df_coverage["d0_n_f_ge_05"] / total_target_cells, df_coverage["depth_m"], "x-", color="#ff7f0e", label="f >= 0.5", linewidth=1.5)
    ax.plot(df_coverage["d0_n_f_eq_1"] / total_target_cells, df_coverage["depth_m"], "s--", color="#2ca02c", label="f = 1 (complete)", linewidth=1.5)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Coverage Fraction of Target Grid (100x240 = 24,000 cells)")
    ax.set_title("GLORYS Regridding: Target Grid Coverage vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    coverage_plot_path = os.path.join(output_dir, "coverage_vs_depth.png")
    fig.savefig(coverage_plot_path)
    plt.close(fig)
    print(f"Saved plot: {coverage_plot_path}")

    # 3. Day 0 Level 0 map of |A - L|
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # Extent: [lon_min, lon_max, lat_min, lat_max]
    # Target bounds are 45.0..105.0 and 5.0..30.0
    im = ax.imshow(
        day0_level0_abs_diff_map,
        origin="lower",
        extent=[TARGET_LON_START, TARGET_LON_START + TARGET_N_LON * TARGET_STEP,
                TARGET_LAT_START, TARGET_LAT_START + TARGET_N_LAT * TARGET_STEP],
        cmap="inferno",
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Method A - Method L| (°C)")
    ax.set_title("GLORYS Regridding: Absolute Difference |A - L| (Day 0, Level 0: 0.5m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle=":", alpha=0.4, color="white")
    plt.tight_layout()
    map_plot_path = os.path.join(output_dir, "diff_map_d0_l0.png")
    fig.savefig(map_plot_path)
    plt.close(fig)
    print(f"Saved plot: {map_plot_path}")

    # -------------------------------------------------------------------------
    # Capture File Metadata After Run (Rule 1 verification)
    # -------------------------------------------------------------------------
    meta_thetao_after = get_file_metadata(thetao_path)
    meta_static_after = get_file_metadata(static_path)

    thetao_unmodified = (
        meta_thetao_before["sha256"] == meta_thetao_after["sha256"]
        and meta_thetao_before["size_bytes"] == meta_thetao_after["size_bytes"]
        and meta_thetao_before["mtime_ns"] == meta_thetao_after["mtime_ns"]
    )
    static_unmodified = (
        meta_static_before["sha256"] == meta_static_after["sha256"]
        and meta_static_before["size_bytes"] == meta_static_after["size_bytes"]
        and meta_static_before["mtime_ns"] == meta_static_after["mtime_ns"]
    )

    if not (thetao_unmodified and static_unmodified):
        raise RuntimeError("FATAL: Input files were modified during the experiment!")

    # -------------------------------------------------------------------------
    # Compile JSON summary
    # -------------------------------------------------------------------------
    json_summary: Dict[str, Any] = {
        "task": "GLORYS horizontal regridding consequence experiment",
        "timestamp_iso": pd.Timestamp.now().isoformat(),
        "file_integrity": {
            "glorys_thetao_2019_01_nc": {
                "path": meta_thetao_before["path"],
                "sha256_before": meta_thetao_before["sha256"],
                "sha256_after": meta_thetao_after["sha256"],
                "size_bytes_before": meta_thetao_before["size_bytes"],
                "size_bytes_after": meta_thetao_after["size_bytes"],
                "mtime_before": meta_thetao_before["mtime"],
                "mtime_after": meta_thetao_after["mtime"],
                "mtime_ns_before": meta_thetao_before["mtime_ns"],
                "mtime_ns_after": meta_thetao_after["mtime_ns"],
                "verified_unmodified": thetao_unmodified,
            },
            "glorys_static_mask_bathy_nc": {
                "path": meta_static_before["path"],
                "sha256_before": meta_static_before["sha256"],
                "sha256_after": meta_static_after["sha256"],
                "size_bytes_before": meta_static_before["size_bytes"],
                "size_bytes_after": meta_static_after["size_bytes"],
                "mtime_before": meta_static_before["mtime"],
                "mtime_after": meta_static_after["mtime"],
                "mtime_ns_before": meta_static_before["mtime_ns"],
                "mtime_ns_after": meta_static_after["mtime_ns"],
                "verified_unmodified": static_unmodified,
            },
        },
        "grid_contracts": {
            "native_shape": [int(levels_used), len(lats), len(lons)],
            "target_shape": [TARGET_N_LAT, TARGET_N_LON],
            "native_step_deg": float(NATIVE_STEP),
            "target_step_deg": float(TARGET_STEP),
            "overlap_weights_1d": [float(w) for w in w_lat_1d],
            "overlap_weights_2d_sum": float(np.sum(w_2d)),
        },
        "equivalence_check_day0": eq_results_day0,
        "coverage_summary_day0_level0": {
            "n_target_total": total_target_cells,
            "n_f_eq_1": int(cov_counts_day0[0]["n_f_eq_1"]),
            "n_f_between_0_and_1": int(cov_counts_day0[0]["n_f_between_0_and_1"]),
            "n_f_eq_0": int(cov_counts_day0[0]["n_f_eq_0"]),
            "n_valid_L": int(cov_counts_day0[0]["n_valid_L"]),
            "n_f_ge_05": int(cov_counts_day0[0]["n_f_ge_05"]),
            "n_f_ge_075": int(cov_counts_day0[0]["n_f_ge_075"]),
            "n_f_ge_09": int(cov_counts_day0[0]["n_f_ge_09"]),
            "invariants_confirmed": True,
        },
        "difference_summary_pooled": {
            "level_0": {
                "depth_m": float(depths[0]),
                "all_rmse": float(df_diff.loc[df_diff["level"] == 0, "all_rmse"].iloc[0]),
                "all_bias": float(df_diff.loc[df_diff["level"] == 0, "all_bias"].iloc[0]),
                "all_mae": float(df_diff.loc[df_diff["level"] == 0, "all_mae"].iloc[0]),
                "all_p99": float(df_diff.loc[df_diff["level"] == 0, "all_p99"].iloc[0]),
                "all_max_abs": float(df_diff.loc[df_diff["level"] == 0, "all_max_abs"].iloc[0]),
            },
            "level_10": {
                "depth_m": float(depths[10]),
                "all_rmse": float(df_diff.loc[df_diff["level"] == 10, "all_rmse"].iloc[0]),
                "all_bias": float(df_diff.loc[df_diff["level"] == 10, "all_bias"].iloc[0]),
                "all_mae": float(df_diff.loc[df_diff["level"] == 10, "all_mae"].iloc[0]),
                "all_p99": float(df_diff.loc[df_diff["level"] == 10, "all_p99"].iloc[0]),
                "all_max_abs": float(df_diff.loc[df_diff["level"] == 10, "all_max_abs"].iloc[0]),
            },
            "level_28": {
                "depth_m": float(depths[28]),
                "all_rmse": float(df_diff.loc[df_diff["level"] == 28, "all_rmse"].iloc[0]),
                "all_bias": float(df_diff.loc[df_diff["level"] == 28, "all_bias"].iloc[0]),
                "all_mae": float(df_diff.loc[df_diff["level"] == 28, "all_mae"].iloc[0]),
                "all_p99": float(df_diff.loc[df_diff["level"] == 28, "all_p99"].iloc[0]),
                "all_max_abs": float(df_diff.loc[df_diff["level"] == 28, "all_max_abs"].iloc[0]),
            },
        },
        "ab_contamination_findings": {
            "column_A_native_idx": [col_coords["Column_A"][0], col_coords["Column_A"][1]],
            "column_B_native_idx": [col_coords["Column_B"][0], col_coords["Column_B"][1]],
            "columns_bracketed_by_method_L": False,
            "target_cells_overlapping_footprint": [
                {"target_i": c["target_i"], "target_j": c["target_j"], "weight_A": c["weight_A"], "weight_B": c["weight_B"]}
                for c in affected_cells
            ],
            "delta_L_with_vs_without_qc": 0.0,
            "max_delta_A_unexcluded_minus_qc_C": float(df_ab["delta_A_unexcluded_minus_qc"].abs().max()),
        },
        "undecided_pipeline_decisions": {
            "selected_regridding_method": None,
            "reason_selected_method_null": "Task contract forbids choosing a method; gathering comparative evidence only.",
            "operational_wet_threshold_fraction": None,
            "reason_wet_threshold_null": "Diagnostic study only; defining operational wet threshold is deferred to downstream model design.",
        },
    }

    json_path = os.path.join(output_dir, "glorys_regrid_experiment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {json_path}")

    return json_summary


def main():
    parser = argparse.ArgumentParser(description="GLORYS Horizontal Regridding Consequence Experiment")
    parser.add_argument("--thetao", default=DEFAULT_THETAO_PATH, help="Path to raw thetao NetCDF")
    parser.add_argument("--static", default=DEFAULT_STATIC_PATH, help="Path to static mask NetCDF")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to QC policy JSON")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    args = parser.parse_args()

    run_regrid_experiment(
        thetao_path=args.thetao,
        static_path=args.static,
        policy_path=args.policy,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
