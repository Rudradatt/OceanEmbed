#!/usr/bin/env python3
"""
Input Aggregation to 0.25 deg Target Grid and Common-Mask Coverage Census (January 2019)
OceanEmbed Data-Validation Phase (Diagnostic Only; Read-Only on Raw Data).

This script performs:
1. Verification of grid nesting (exact integer ratios per axis for SST, SLA, SSS, currents;
   straddle overlap weights for GLORYS).
2. Block aggregation of native fields to the target 0.25 deg grid (100 x 240 cells)
   via vectorized reshaping (no per-cell Python loops).
3. Exact validity definitions:
   - SST analysed_sst: non-fill (-32768) and (mask & 6) == 0 (excludes land and lake bits).
   - SLA: sla, adt, ugos, vgos (each non-fill -2147483647).
   - SSS: sos (non-fill -32767.0).
   - Currents: depth 0 uo, vo and surface ugos, vgos (non-fill 32767).
   - GLORYS: Method A area-weighted QC valid fraction f_T from QC policy v1.
4. Aggregation integrity check: sum(f * block_size) == total native valid cells.
5. Coverage census per source variable across 31 days.
6. Common-mask census across thresholds tauT, tauI in {0.5, 0.75, 1.0} and levels.
7. Tracking of cells lost at level 0 (tauT=0.5, tauI=0.5) including the 4 SST lake cells.
8. SSS temporal coverage variation across 31 days.
9. Diagnostic plots and comprehensive JSON summary.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5netcdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    compute_valid_fraction_f,
)


# =============================================================================
# Constants & Defaults
# =============================================================================

DEFAULT_SST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sst", "sst_ostia_2019-01.nc")
DEFAULT_SLA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sla", "sla_duacs_2019-01.nc")
DEFAULT_SSS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sss", "sss_multiobs_2019-01.nc")
DEFAULT_CUR_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "currents", "currents_multiobs_2019-01.nc")
DEFAULT_GLORYS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "glorys", "glorys_thetao_2019-01.nc")
DEFAULT_STATIC_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "glorys_static", "glorys_static_mask_bathy.nc")
DEFAULT_POLICY_PATH = os.path.join(PROJECT_ROOT, "configs", "glorys_qc_policy_v1.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "diagnostics", "input_aggregation_census")

RAW_FILES = {
    "sst": DEFAULT_SST_PATH,
    "sla": DEFAULT_SLA_PATH,
    "sss": DEFAULT_SSS_PATH,
    "cur": DEFAULT_CUR_PATH,
    "glorys": DEFAULT_GLORYS_PATH,
    "static": DEFAULT_STATIC_PATH,
}

EXPECTED_SHA256 = {
    "sst": "9e733b5956f4d9bba81ec1c7a4bcbfa3b456e61af969d9ae13c57ef6fcc72265",
    "sla": "36b558a5642feaff3d8563aea4dac76f70f850d1b555431a8893474b7eccd692",
    "sss": "74dc119b9d34a1b0330e1e9066e067db658dfaa8ff44dedb6edf03041b6ff043",
    "cur": "ad98b7684663b42a2eaa4ced7d5b52ec795fc711d050cdd87cd0b70eaf0ca4a7",
    "glorys": "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f",
    "static": "fdb0c5fdc282180c0b5b58fa689753df79c22f69f4dc6dac64314a8cb82a79a5",
}

TARGET_N_LAT = 100
TARGET_N_LON = 240
TARGET_STEP = 0.25
TARGET_LAT_START = 5.0
TARGET_LON_START = 45.0
TARGET_LAT_END = 30.0
TARGET_LON_END = 105.0

# Target cell centers: 5.125 .. 29.875 (100) and 45.125 .. 104.875 (240)
TARGET_LAT_CENTERS = np.linspace(TARGET_LAT_START + TARGET_STEP / 2.0, TARGET_LAT_END - TARGET_STEP / 2.0, TARGET_N_LAT)
TARGET_LON_CENTERS = np.linspace(TARGET_LON_START + TARGET_STEP / 2.0, TARGET_LON_END - TARGET_STEP / 2.0, TARGET_N_LON)


# =============================================================================
# STEP 0 & 1: File Integrity and Grid Nesting
# =============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file without loading entire file in memory."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """Retrieve SHA256, file size in bytes, and modification timestamp."""
    stat = os.stat(filepath)
    sha256_val = compute_file_sha256(filepath)
    return {
        "path": os.path.abspath(filepath),
        "sha256": sha256_val,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_iso": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
    }


def verify_grid_nesting(
    native_centers: np.ndarray,
    target_start: float,
    target_step: float,
    n_target: int,
    tol: float = 1e-4,
) -> int:
    """Verify numerically that native cell edges coincide with target edges.
    
    Computes and returns the integer native-cells-per-target-cell ratio per axis.
    Raises ValueError and halts immediately if not an integer or if edges do not coincide.
    """
    native_centers = np.asarray(native_centers, dtype=np.float64)
    M = len(native_centers)
    if M < 2:
        raise ValueError("Native grid must have at least 2 coordinate points")

    diffs = np.diff(native_centers)
    native_step = float(np.mean(diffs))
    if not np.allclose(diffs, native_step, atol=tol):
        raise ValueError(f"Native grid spacing is not uniform within tolerance {tol}: min={diffs.min()}, max={diffs.max()}")

    native_start_edge = native_centers[0] - native_step / 2.0
    native_end_edge = native_centers[-1] + native_step / 2.0
    target_end_edge = target_start + n_target * target_step

    if abs(native_start_edge - target_start) > tol:
        raise ValueError(
            f"Start edges do not coincide within {tol} deg: native_start={native_start_edge:.8f}, target_start={target_start:.8f}"
        )
    if abs(native_end_edge - target_end_edge) > tol:
        raise ValueError(
            f"End edges do not coincide within {tol} deg: native_end={native_end_edge:.8f}, target_end={target_end_edge:.8f}"
        )

    ratio = target_step / native_step
    ratio_int = int(round(ratio))
    if abs(ratio - ratio_int) > tol:
        raise ValueError(f"Native-cells-per-target-cell ratio {ratio:.6f} is not an integer within tolerance {tol}")

    if ratio_int < 1:
        raise ValueError(f"Native-cells-per-target-cell ratio {ratio_int} must be >= 1")

    if M != n_target * ratio_int:
        raise ValueError(f"Native grid size {M} does not equal n_target * ratio ({n_target} * {ratio_int} = {n_target * ratio_int})")

    # Verify every target cell edge coincides with a native cell edge
    for i in range(n_target + 1):
        t_edge = target_start + i * target_step
        native_edge_idx = i * ratio_int
        if native_edge_idx == 0:
            n_edge = native_centers[0] - native_step / 2.0
        elif native_edge_idx == M:
            n_edge = native_centers[-1] + native_step / 2.0
        else:
            n_edge = (native_centers[native_edge_idx - 1] + native_centers[native_edge_idx]) / 2.0
        if abs(t_edge - n_edge) > tol:
            raise ValueError(f"Target edge {i} ({t_edge:.8f}) does not coincide with native edge ({n_edge:.8f}) within {tol}")

    return ratio_int


# =============================================================================
# Block Aggregation & Decoding
# =============================================================================

def block_aggregate(
    native_field: np.ndarray,
    valid_mask: np.ndarray,
    ratio_y: int,
    ratio_x: int,
    n_target_y: int = 100,
    n_target_x: int = 240,
) -> Tuple[np.ndarray, np.ndarray]:
    """Block aggregation of a 2D native array to the target grid using reshaping.
    
    No per-cell Python loops.
    
    Parameters:
        native_field: float array of shape (n_target_y * ratio_y, n_target_x * ratio_x)
        valid_mask: bool array of shape (n_target_y * ratio_y, n_target_x * ratio_x)
        ratio_y: native cells per target cell along Y (latitude)
        ratio_x: native cells per target cell along X (longitude)
        n_target_y: number of target cells along Y (default 100)
        n_target_x: number of target cells along X (default 240)
        
    Returns:
        mean_field: float64 array of shape (n_target_y, n_target_x) with mean over valid native cells,
                    or NaN if valid count == 0.
        f: float64 array of shape (n_target_y, n_target_x) with valid fraction
           (valid cells / total cells in block).
    """
    native_field = np.asarray(native_field, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if native_field.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch: native_field {native_field.shape} vs valid_mask {valid_mask.shape}")

    ny, nx = native_field.shape
    expected_ny = n_target_y * ratio_y
    expected_nx = n_target_x * ratio_x
    if ny != expected_ny or nx != expected_nx:
        raise ValueError(f"Native field shape ({ny}, {nx}) does not match expected ({expected_ny}, {expected_nx})")

    cells_per_block = ratio_y * ratio_x

    # Reshape and swapaxes to group blocks together:
    # (n_target_y, ratio_y, n_target_x, ratio_x) -> swapaxes(1, 2) -> (n_target_y, n_target_x, ratio_y, ratio_x)
    field_blocks = native_field.reshape(n_target_y, ratio_y, n_target_x, ratio_x).swapaxes(1, 2).reshape(n_target_y, n_target_x, cells_per_block)
    mask_blocks = valid_mask.reshape(n_target_y, ratio_y, n_target_x, ratio_x).swapaxes(1, 2).reshape(n_target_y, n_target_x, cells_per_block)

    valid_counts = np.sum(mask_blocks, axis=-1)
    f = valid_counts.astype(np.float64) / float(cells_per_block)

    # Safe sum: zero out invalid native cells to prevent NaN propagation
    safe_field = np.where(mask_blocks, field_blocks, 0.0)
    sums = np.sum(safe_field, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_field = np.where(valid_counts > 0, sums / valid_counts.astype(np.float64), np.nan)

    return mean_field, f


# =============================================================================
# Validity Definitions (Explicit)
# =============================================================================

def compute_sst_validity(
    sst_raw: np.ndarray,
    mask_raw: np.ndarray,
    fill_value: int = -32768,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """SST analysed_sst validity: non-fill AND (mask & 6) == 0.
    
    Bitmask rule (mask & 6) == 0 excludes land (bit 1, val 2) and lake (bit 2, val 4).
    Returns:
        valid_mask: bool array of same shape
        unique_masks_non_fill: sorted array of unique mask values among non-fill cells
        lake_rule_removed_count: count of non-fill cells removed by the lake/land bitmask rule
    """
    sst_raw = np.asarray(sst_raw)
    mask_raw = np.asarray(mask_raw)
    non_fill = (sst_raw != fill_value)
    unique_masks_non_fill = np.unique(mask_raw[non_fill])
    
    # Exclude land (2) and lake (4): 6 = 2 | 4
    mask_allowed = ((mask_raw & 6) == 0)
    valid_mask = non_fill & mask_allowed
    lake_rule_removed_count = int(np.sum(non_fill & (~mask_allowed)))

    return valid_mask, unique_masks_non_fill, lake_rule_removed_count


def compute_sla_validity(
    raw_array: np.ndarray,
    fill_value: int = -2147483647,
) -> np.ndarray:
    """SLA variable validity: non-fill by its own fill value."""
    return np.asarray(raw_array) != fill_value


def compute_sss_validity(
    raw_array: np.ndarray,
    fill_value: float = -32767.0,
) -> np.ndarray:
    """SSS sos validity: non-fill by its own fill value."""
    return np.asarray(raw_array) != fill_value


def compute_currents_validity(
    raw_array: np.ndarray,
    fill_value: int = 32767,
) -> np.ndarray:
    """MULTIOBS currents validity: non-fill by its own fill value."""
    return np.asarray(raw_array) != fill_value


# Manual decoding helpers
def decode_sst(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of OSTIA SST from raw int16 to Kelvin."""
    return raw.astype(np.float64) * 0.009999999776482582 + 273.1499938964844


def decode_sla(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of DUACS SLA from raw int32 to meters (or m/s for velocities)."""
    return raw.astype(np.float64) * 0.0001


def decode_sss(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of MULTIOBS SSS from float32."""
    return raw.astype(np.float64)


def decode_currents(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of MULTIOBS currents from raw int16 to m/s."""
    return raw.astype(np.float64) * 0.001


# =============================================================================
# Real Data Processing
# =============================================================================

def run_census(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sst_path: str = DEFAULT_SST_PATH,
    sla_path: str = DEFAULT_SLA_PATH,
    sss_path: str = DEFAULT_SSS_PATH,
    cur_path: str = DEFAULT_CUR_PATH,
    glorys_path: str = DEFAULT_GLORYS_PATH,
    static_path: str = DEFAULT_STATIC_PATH,
    policy_path: str = DEFAULT_POLICY_PATH,
) -> Dict[str, Any]:
    """Run full input aggregation and common-mask census across all 31 days."""
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 0: SHA256, Size, Mtime Before
    # -------------------------------------------------------------------------
    meta_before = {
        "sst": get_file_metadata(sst_path),
        "sla": get_file_metadata(sla_path),
        "sss": get_file_metadata(sss_path),
        "cur": get_file_metadata(cur_path),
        "glorys": get_file_metadata(glorys_path),
        "static": get_file_metadata(static_path),
    }

    # Verify against expected hashes
    for key, expected_hash in EXPECTED_SHA256.items():
        actual_hash = meta_before[key]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"Input file {key} SHA256 mismatch before run: expected {expected_hash}, got {actual_hash}")

    # -------------------------------------------------------------------------
    # STEP 1: Nesting Verification
    # -------------------------------------------------------------------------
    with h5netcdf.File(sst_path, "r") as f:
        sst_lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        sst_lons = np.array(f.variables["longitude"][:], dtype=np.float64)
    with h5netcdf.File(sla_path, "r") as f:
        sla_lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        sla_lons = np.array(f.variables["longitude"][:], dtype=np.float64)
    with h5netcdf.File(sss_path, "r") as f:
        sss_lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        sss_lons = np.array(f.variables["longitude"][:], dtype=np.float64)
    with h5netcdf.File(cur_path, "r") as f:
        cur_lats = np.array(f.variables["latitude"][:], dtype=np.float64)
        cur_lons = np.array(f.variables["longitude"][:], dtype=np.float64)

    ratio_sst_y = verify_grid_nesting(sst_lats, TARGET_LAT_START, TARGET_STEP, TARGET_N_LAT)
    ratio_sst_x = verify_grid_nesting(sst_lons, TARGET_LON_START, TARGET_STEP, TARGET_N_LON)
    ratio_sla_y = verify_grid_nesting(sla_lats, TARGET_LAT_START, TARGET_STEP, TARGET_N_LAT)
    ratio_sla_x = verify_grid_nesting(sla_lons, TARGET_LON_START, TARGET_STEP, TARGET_N_LON)
    ratio_sss_y = verify_grid_nesting(sss_lats, TARGET_LAT_START, TARGET_STEP, TARGET_N_LAT)
    ratio_sss_x = verify_grid_nesting(sss_lons, TARGET_LON_START, TARGET_STEP, TARGET_N_LON)
    ratio_cur_y = verify_grid_nesting(cur_lats, TARGET_LAT_START, TARGET_STEP, TARGET_N_LAT)
    ratio_cur_x = verify_grid_nesting(cur_lons, TARGET_LON_START, TARGET_STEP, TARGET_N_LON)

    nesting_ratios = {
        "sst": (ratio_sst_y, ratio_sst_x),
        "sla": (ratio_sla_y, ratio_sla_x),
        "sss": (ratio_sss_y, ratio_sss_x),
        "cur": (ratio_cur_y, ratio_cur_x),
    }

    # Verify GLORYS 1D overlap weights (straddle grid)
    glorys_lats_all, glorys_lons_all, glorys_depths_all = load_coordinates(glorys_path)
    compute_overlap_weights_1d(glorys_lats_all, TARGET_LAT_START, TARGET_STEP, TARGET_N_LAT)
    compute_overlap_weights_1d(glorys_lons_all, TARGET_LON_START, TARGET_STEP, TARGET_N_LON)

    # Preload GLORYS static mask, exclusions, and policy
    policy = load_policy(policy_path)
    static_wet = load_static_wet_mask(static_path, glorys_lats_all, glorys_lons_all, glorys_depths_all, levels_used=36)
    exclusions = locate_exclusions(glorys_lats_all, glorys_lons_all, policy)

    # -------------------------------------------------------------------------
    # STEP 3: Real Data Run across all 31 days (process one day at a time)
    # -------------------------------------------------------------------------
    # Expected day 0 native counts from prompt contract:
    # SST: 299,372; sla: 48,698; adt, ugos, vgos: 48,368; sos: 47,479; uo, vo: 11,886; cur ugos, vgos: 12,004
    EXPECTED_DAY0_COUNTS = {
        "SST analysed_sst": 299372,
        "SLA sla": 48698,
        "SLA adt": 48368,
        "SLA ugos": 48368,
        "SLA vgos": 48368,
        "SSS sos": 47479,
        "Currents uo": 11886,
        "Currents vo": 11886,
        "Currents ugos": 12004,
        "Currents vgos": 12004,
    }

    variable_specs = [
        ("OSTIA_SST", "analysed_sst", "sst", ratio_sst_y, ratio_sst_x),
        ("DUACS_SLA", "sla", "sla", ratio_sla_y, ratio_sla_x),
        ("DUACS_SLA", "adt", "sla", ratio_sla_y, ratio_sla_x),
        ("DUACS_SLA", "ugos", "sla", ratio_sla_y, ratio_sla_x),
        ("DUACS_SLA", "vgos", "sla", ratio_sla_y, ratio_sla_x),
        ("MULTIOBS_SSS", "sos", "sss", ratio_sss_y, ratio_sss_x),
        ("MULTIOBS_CURRENTS", "uo", "cur", ratio_cur_y, ratio_cur_x),
        ("MULTIOBS_CURRENTS", "vo", "cur", ratio_cur_y, ratio_cur_x),
        ("MULTIOBS_CURRENTS", "ugos", "cur", ratio_cur_y, ratio_cur_x),
        ("MULTIOBS_CURRENTS", "vgos", "cur", ratio_cur_y, ratio_cur_x),
    ]

    # Storage for daily coverage
    daily_cov_stats = {f"{src}_{var}": [] for src, var, _, _, _ in variable_specs}
    daily_integrity = []

    # Common-mask census tracking
    # Census levels: 0, 5, 10, 15, 20, 25, 28, 30, 35
    CENSUS_LEVELS = [0, 5, 10, 15, 20, 25, 28, 30, 35]
    TAU_T_VALS = [0.5, 0.75, 1.0]
    TAU_I_VALS = [0.5, 0.75, 1.0]

    # Shape: (31 days, n_levels, n_tauT, n_tauI, n_metrics)
    census_grid_31 = np.zeros((31, len(CENSUS_LEVELS), len(TAU_T_VALS), len(TAU_I_VALS), 15), dtype=np.int64)

    # SSS daily tracking
    sss_daily_ge_05 = []
    sss_daily_native_valid = []
    sss_prev_status = None
    sss_daily_status_changes = []

    # Store day 0 fields for detailed level 0 lost cells analysis
    d0_data_level0: Dict[str, Any] = {}

    sst_unique_masks_d0 = None
    sst_lake_rule_removed_d0 = None
    cur_1x1_match_all_days = True

    # Open all NetCDF files
    f_sst = h5netcdf.File(sst_path, "r")
    f_sla = h5netcdf.File(sla_path, "r")
    f_sss = h5netcdf.File(sss_path, "r")
    f_cur = h5netcdf.File(cur_path, "r")

    try:
        for day in range(31):
            # 1. Read GLORYS day & compute f_T
            raw_thetao_3d = read_raw_day(glorys_path, day_index=day, policy=policy)
            qc_result = apply_qc(raw_thetao_3d, static_wet, exclusions, policy)
            # f_T shape: (36, 100, 240)
            f_T = compute_valid_fraction_f(qc_result.valid)

            # 2. Read native inputs for this day
            sst_raw = f_sst.variables["analysed_sst"][day]
            sst_mask = f_sst.variables["mask"][day]
            valid_sst, umasks, lake_removed = compute_sst_validity(sst_raw, sst_mask)
            if day == 0:
                sst_unique_masks_d0 = umasks
                sst_lake_rule_removed_d0 = lake_removed

            sla_raw = f_sla.variables["sla"][day]
            valid_sla = compute_sla_validity(sla_raw)

            adt_raw = f_sla.variables["adt"][day]
            valid_adt = compute_sla_validity(adt_raw)

            sla_ugos_raw = f_sla.variables["ugos"][day]
            valid_sla_ugos = compute_sla_validity(sla_ugos_raw)

            sla_vgos_raw = f_sla.variables["vgos"][day]
            valid_sla_vgos = compute_sla_validity(sla_vgos_raw)

            sss_raw = f_sss.variables["sos"][day, 0]  # depth 0
            valid_sss = compute_sss_validity(sss_raw)

            cur_uo_raw = f_cur.variables["uo"][day, 0]  # depth 0
            valid_cur_uo = compute_currents_validity(cur_uo_raw)

            cur_vo_raw = f_cur.variables["vo"][day, 0]  # depth 0
            valid_cur_vo = compute_currents_validity(cur_vo_raw)

            cur_ugos_raw = f_cur.variables["ugos"][day]
            valid_cur_ugos = compute_currents_validity(cur_ugos_raw)

            cur_vgos_raw = f_cur.variables["vgos"][day]
            valid_cur_vgos = compute_currents_validity(cur_vgos_raw)

            # Block aggregation for all variables
            agg_fields: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
            # tuple: (mean_field, f, cells_per_block)

            # SST
            m_sst, f_sst_val = block_aggregate(decode_sst(sst_raw), valid_sst, ratio_sst_y, ratio_sst_x)
            agg_fields["OSTIA_SST_analysed_sst"] = (m_sst, f_sst_val, ratio_sst_y * ratio_sst_x)

            # SLA
            m_sla, f_sla_val = block_aggregate(decode_sla(sla_raw), valid_sla, ratio_sla_y, ratio_sla_x)
            agg_fields["DUACS_SLA_sla"] = (m_sla, f_sla_val, ratio_sla_y * ratio_sla_x)

            m_adt, f_adt_val = block_aggregate(decode_sla(adt_raw), valid_adt, ratio_sla_y, ratio_sla_x)
            agg_fields["DUACS_SLA_adt"] = (m_adt, f_adt_val, ratio_sla_y * ratio_sla_x)

            m_duacs_ugos, f_duacs_ugos_val = block_aggregate(decode_sla(sla_ugos_raw), valid_sla_ugos, ratio_sla_y, ratio_sla_x)
            agg_fields["DUACS_SLA_ugos"] = (m_duacs_ugos, f_duacs_ugos_val, ratio_sla_y * ratio_sla_x)

            m_duacs_vgos, f_duacs_vgos_val = block_aggregate(decode_sla(sla_vgos_raw), valid_sla_vgos, ratio_sla_y, ratio_sla_x)
            agg_fields["DUACS_SLA_vgos"] = (m_duacs_vgos, f_duacs_vgos_val, ratio_sla_y * ratio_sla_x)

            # SSS
            m_sss, f_sss_val = block_aggregate(decode_sss(sss_raw), valid_sss, ratio_sss_y, ratio_sss_x)
            agg_fields["MULTIOBS_SSS_sos"] = (m_sss, f_sss_val, ratio_sss_y * ratio_sss_x)

            # CURRENTS
            m_cur_uo, f_cur_uo_val = block_aggregate(decode_currents(cur_uo_raw), valid_cur_uo, ratio_cur_y, ratio_cur_x)
            agg_fields["MULTIOBS_CURRENTS_uo"] = (m_cur_uo, f_cur_uo_val, ratio_cur_y * ratio_cur_x)

            m_cur_vo, f_cur_vo_val = block_aggregate(decode_currents(cur_vo_raw), valid_cur_vo, ratio_cur_y, ratio_cur_x)
            agg_fields["MULTIOBS_CURRENTS_vo"] = (m_cur_vo, f_cur_vo_val, ratio_cur_y * ratio_cur_x)

            m_cur_ugos, f_cur_ugos_val = block_aggregate(decode_currents(cur_ugos_raw), valid_cur_ugos, ratio_cur_y, ratio_cur_x)
            agg_fields["MULTIOBS_CURRENTS_ugos"] = (m_cur_ugos, f_cur_ugos_val, ratio_cur_y * ratio_cur_x)

            m_cur_vgos, f_cur_vgos_val = block_aggregate(decode_currents(cur_vgos_raw), valid_cur_vgos, ratio_cur_y, ratio_cur_x)
            agg_fields["MULTIOBS_CURRENTS_vgos"] = (m_cur_vgos, f_cur_vgos_val, ratio_cur_y * ratio_cur_x)

            # Verify 1x1 currents aggregated values equal native values exactly
            dec_cur_uo = decode_currents(cur_uo_raw)
            if not np.allclose(m_cur_uo[valid_cur_uo], dec_cur_uo[valid_cur_uo]):
                cur_1x1_match_all_days = False

            # Part a: Integrity checks for day in [0, 15, 30] and record for all days
            valid_native_map = {
                "OSTIA_SST_analysed_sst": int(np.sum(valid_sst)),
                "DUACS_SLA_sla": int(np.sum(valid_sla)),
                "DUACS_SLA_adt": int(np.sum(valid_adt)),
                "DUACS_SLA_ugos": int(np.sum(valid_sla_ugos)),
                "DUACS_SLA_vgos": int(np.sum(valid_sla_vgos)),
                "MULTIOBS_SSS_sos": int(np.sum(valid_sss)),
                "MULTIOBS_CURRENTS_uo": int(np.sum(valid_cur_uo)),
                "MULTIOBS_CURRENTS_vo": int(np.sum(valid_cur_vo)),
                "MULTIOBS_CURRENTS_ugos": int(np.sum(valid_cur_ugos)),
                "MULTIOBS_CURRENTS_vgos": int(np.sum(valid_cur_vgos)),
            }

            EXPECTED_D0_MAP = {
                "OSTIA_SST_analysed_sst": 299372,
                "DUACS_SLA_sla": 48698,
                "DUACS_SLA_adt": 48368,
                "DUACS_SLA_ugos": 48368,
                "DUACS_SLA_vgos": 48368,
                "MULTIOBS_SSS_sos": 47479,
                "MULTIOBS_CURRENTS_uo": 11886,
                "MULTIOBS_CURRENTS_vo": 11886,
                "MULTIOBS_CURRENTS_ugos": 12004,
                "MULTIOBS_CURRENTS_vgos": 12004,
            }

            for key, (mean_f, f_arr, c_block) in agg_fields.items():
                sum_f_x_cells = float(np.sum(f_arr * c_block))
                int_sum = int(round(sum_f_x_cells))
                native_valid_cnt = valid_native_map[key]
                diff = int_sum - native_valid_cnt
                expected_cnt = EXPECTED_D0_MAP[key] if day == 0 else None
                diff_vs_expected = (native_valid_cnt - expected_cnt) if (day == 0 and expected_cnt is not None) else None

                if day in [0, 15, 30]:
                    daily_integrity.append({
                        "day": day,
                        "variable": key,
                        "expected_d0_native": expected_cnt if day == 0 else "-",
                        "actual_native_valid": native_valid_cnt,
                        "sum_f_times_block": int_sum,
                        "diff_sum_vs_native": diff,
                        "diff_vs_expected_d0": diff_vs_expected if day == 0 else "-",
                        "matches_exact": (diff == 0) and (day != 0 or diff_vs_expected == 0),
                    })

                # Part b: Coverage statistics for this variable and day
                daily_cov_stats[key].append({
                    "day": day,
                    "f_eq_1": int(np.sum(f_arr == 1.0)),
                    "f_ge_075": int(np.sum(f_arr >= 0.75)),
                    "f_ge_05": int(np.sum(f_arr >= 0.5)),
                    "f_gt_0": int(np.sum(f_arr > 0.0)),
                    "f_eq_0": int(np.sum(f_arr == 0.0)),
                })

            # Part e: SSS daily tracking
            sss_status_ge_05 = (f_sss_val >= 0.5)
            cnt_ge_05 = int(np.sum(sss_status_ge_05))
            sss_daily_ge_05.append(cnt_ge_05)
            sss_daily_native_valid.append(valid_native_map["MULTIOBS_SSS_sos"])
            if day == 0:
                sss_changed = 0
            else:
                sss_changed = int(np.sum(sss_status_ge_05 != sss_prev_status))
            sss_daily_status_changes.append(sss_changed)
            sss_prev_status = sss_status_ge_05

            # Store day 0 data for detailed level 0 lost cells analysis
            if day == 0:
                d0_data_level0 = {
                    "f_T": f_T[0],
                    "f_sst": f_sst_val,
                    "f_sla": f_sla_val,
                    "f_sss": f_sss_val,
                    "f_duacs_ugos": f_duacs_ugos_val,
                    "f_duacs_vgos": f_duacs_vgos_val,
                    "f_cur_uo": f_cur_uo_val,
                    "f_cur_vo": f_cur_vo_val,
                    "f_cur_ugos": f_cur_ugos_val,
                    "f_cur_vgos": f_cur_vgos_val,
                }

            # Part c: Common-mask census across thresholds and levels
            for l_idx, lvl in enumerate(CENSUS_LEVELS):
                f_T_lvl = f_T[lvl]
                for t_idx, tauT in enumerate(TAU_T_VALS):
                    glorys_valid = (f_T_lvl >= tauT)
                    n_glorys = int(np.sum(glorys_valid))

                    for i_idx, tauI in enumerate(TAU_I_VALS):
                        v_sst_target = (f_sst_val >= tauI)
                        v_sla_target = (f_sla_val >= tauI)
                        v_sss_target = (f_sss_val >= tauI)
                        v_cur_A_target = (f_duacs_ugos_val >= tauI) & (f_duacs_vgos_val >= tauI)
                        v_cur_B_target = (f_cur_uo_val >= tauI) & (f_cur_vo_val >= tauI)

                        common_A = glorys_valid & v_sst_target & v_sla_target & v_sss_target & v_cur_A_target
                        common_B = glorys_valid & v_sst_target & v_sla_target & v_sss_target & v_cur_B_target

                        # Lost because single input is invalid (among GLORYS valid)
                        lost_sst = glorys_valid & (~v_sst_target)
                        lost_sla = glorys_valid & (~v_sla_target)
                        lost_sss = glorys_valid & (~v_sss_target)
                        lost_cur_A = glorys_valid & (~v_cur_A_target)
                        lost_cur_B = glorys_valid & (~v_cur_B_target)

                        # Marginal loss (valid in GLORYS and all other inputs EXCEPT this one)
                        marg_sst_A = glorys_valid & (~v_sst_target) & v_sla_target & v_sss_target & v_cur_A_target
                        marg_sla_A = glorys_valid & v_sst_target & (~v_sla_target) & v_sss_target & v_cur_A_target
                        marg_sss_A = glorys_valid & v_sst_target & v_sla_target & (~v_sss_target) & v_cur_A_target
                        marg_cur_A = glorys_valid & v_sst_target & v_sla_target & v_sss_target & (~v_cur_A_target)

                        marg_sst_B = glorys_valid & (~v_sst_target) & v_sla_target & v_sss_target & v_cur_B_target
                        marg_sla_B = glorys_valid & v_sst_target & (~v_sla_target) & v_sss_target & v_cur_B_target
                        marg_sss_B = glorys_valid & v_sst_target & v_sla_target & (~v_sss_target) & v_cur_B_target
                        marg_cur_B = glorys_valid & v_sst_target & v_sla_target & v_sss_target & (~v_cur_B_target)

                        census_grid_31[day, l_idx, t_idx, i_idx, 0] = n_glorys
                        census_grid_31[day, l_idx, t_idx, i_idx, 1] = int(np.sum(common_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 2] = int(np.sum(common_B))
                        census_grid_31[day, l_idx, t_idx, i_idx, 3] = int(np.sum(lost_sst))
                        census_grid_31[day, l_idx, t_idx, i_idx, 4] = int(np.sum(lost_sla))
                        census_grid_31[day, l_idx, t_idx, i_idx, 5] = int(np.sum(lost_sss))
                        census_grid_31[day, l_idx, t_idx, i_idx, 6] = int(np.sum(lost_cur_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 7] = int(np.sum(lost_cur_B))
                        census_grid_31[day, l_idx, t_idx, i_idx, 8] = int(np.sum(marg_sst_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 9] = int(np.sum(marg_sla_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 10] = int(np.sum(marg_sss_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 11] = int(np.sum(marg_cur_A))
                        census_grid_31[day, l_idx, t_idx, i_idx, 12] = int(np.sum(marg_sst_B))
                        census_grid_31[day, l_idx, t_idx, i_idx, 13] = int(np.sum(marg_sss_B))
                        census_grid_31[day, l_idx, t_idx, i_idx, 14] = int(np.sum(marg_cur_B))

    finally:
        f_sst.close()
        f_sla.close()
        f_sss.close()
        f_cur.close()

    # -------------------------------------------------------------------------
    # Output 1: input_coverage_by_variable.csv
    # -------------------------------------------------------------------------
    cov_rows = []
    for src, var, _, _, _ in variable_specs:
        key = f"{src}_{var}"
        stats_list = daily_cov_stats[key]
        d0_s = stats_list[0]
        f_eq_1_all = [s["f_eq_1"] for s in stats_list]
        f_ge_075_all = [s["f_ge_075"] for s in stats_list]
        f_ge_05_all = [s["f_ge_05"] for s in stats_list]
        f_gt_0_all = [s["f_gt_0"] for s in stats_list]
        f_eq_0_all = [s["f_eq_0"] for s in stats_list]

        cov_rows.append({
            "source": src,
            "variable": var,
            "day0_f_eq_1": d0_s["f_eq_1"],
            "day0_f_ge_075": d0_s["f_ge_075"],
            "day0_f_ge_05": d0_s["f_ge_05"],
            "day0_f_gt_0": d0_s["f_gt_0"],
            "day0_f_eq_0": d0_s["f_eq_0"],
            "min31_f_eq_1": min(f_eq_1_all),
            "max31_f_eq_1": max(f_eq_1_all),
            "min31_f_ge_075": min(f_ge_075_all),
            "max31_f_ge_075": max(f_ge_075_all),
            "min31_f_ge_05": min(f_ge_05_all),
            "max31_f_ge_05": max(f_ge_05_all),
            "min31_f_gt_0": min(f_gt_0_all),
            "max31_f_gt_0": max(f_gt_0_all),
            "min31_f_eq_0": min(f_eq_0_all),
            "max31_f_eq_0": max(f_eq_0_all),
        })

    df_coverage = pd.DataFrame(cov_rows)
    coverage_csv_path = os.path.join(output_dir, "input_coverage_by_variable.csv")
    df_coverage.to_csv(coverage_csv_path, index=False)

    # -------------------------------------------------------------------------
    # Output 2: common_mask_census.csv
    # -------------------------------------------------------------------------
    census_rows = []
    for l_idx, lvl in enumerate(CENSUS_LEVELS):
        for t_idx, tauT in enumerate(TAU_T_VALS):
            for i_idx, tauI in enumerate(TAU_I_VALS):
                d0_vals = census_grid_31[0, l_idx, t_idx, i_idx]
                min_vals = np.min(census_grid_31[:, l_idx, t_idx, i_idx], axis=0)

                census_rows.append({
                    "level": lvl,
                    "tauT": tauT,
                    "tauI": tauI,
                    "d0_glorys_valid": int(d0_vals[0]),
                    "min31_glorys_valid": int(min_vals[0]),
                    "d0_common_set_A": int(d0_vals[1]),
                    "min31_common_set_A": int(min_vals[1]),
                    "d0_common_set_B": int(d0_vals[2]),
                    "min31_common_set_B": int(min_vals[2]),
                    # Single input loss vs GLORYS
                    "d0_lost_sst": int(d0_vals[3]),
                    "min31_lost_sst": int(min_vals[3]),
                    "d0_lost_sla": int(d0_vals[4]),
                    "min31_lost_sla": int(min_vals[4]),
                    "d0_lost_sss": int(d0_vals[5]),
                    "min31_lost_sss": int(min_vals[5]),
                    "d0_lost_cur_A": int(d0_vals[6]),
                    "min31_lost_cur_A": int(min_vals[6]),
                    "d0_lost_cur_B": int(d0_vals[7]),
                    "min31_lost_cur_B": int(min_vals[7]),
                    # Marginal losses
                    "d0_marginal_lost_sst_A": int(d0_vals[8]),
                    "min31_marginal_lost_sst_A": int(min_vals[8]),
                    "d0_marginal_lost_sla_A": int(d0_vals[9]),
                    "min31_marginal_lost_sla_A": int(min_vals[9]),
                    "d0_marginal_lost_sss_A": int(d0_vals[10]),
                    "min31_marginal_lost_sss_A": int(min_vals[10]),
                    "d0_marginal_lost_cur_A": int(d0_vals[11]),
                    "min31_marginal_lost_cur_A": int(min_vals[11]),
                    "d0_marginal_lost_sst_B": int(d0_vals[12]),
                    "min31_marginal_lost_sst_B": int(min_vals[12]),
                    "d0_marginal_lost_sss_B": int(d0_vals[13]),
                    "min31_marginal_lost_sss_B": int(min_vals[13]),
                    "d0_marginal_lost_cur_B": int(d0_vals[14]),
                    "min31_marginal_lost_cur_B": int(min_vals[14]),
                })

    df_census = pd.DataFrame(census_rows)
    census_csv_path = os.path.join(output_dir, "common_mask_census.csv")
    df_census.to_csv(census_csv_path, index=False)

    # -------------------------------------------------------------------------
    # Output 3: lost_cells_level0.csv
    # -------------------------------------------------------------------------
    f_T_0 = d0_data_level0["f_T"]
    f_sst_0 = d0_data_level0["f_sst"]
    f_sla_0 = d0_data_level0["f_sla"]
    f_sss_0 = d0_data_level0["f_sss"]
    f_curA_u0 = d0_data_level0["f_duacs_ugos"]
    f_curA_v0 = d0_data_level0["f_duacs_vgos"]
    f_curB_u0 = d0_data_level0["f_cur_uo"]
    f_curB_v0 = d0_data_level0["f_cur_vo"]

    glorys_wet_0 = (f_T_0 >= 0.5)
    sst_ok = (f_sst_0 >= 0.5)
    sla_ok = (f_sla_0 >= 0.5)
    sss_ok = (f_sss_0 >= 0.5)
    curA_ok = (f_curA_u0 >= 0.5) & (f_curA_v0 >= 0.5)
    curB_ok = (f_curB_u0 >= 0.5) & (f_curB_v0 >= 0.5)

    lost_set_A_mask = glorys_wet_0 & ~(sst_ok & sla_ok & sss_ok & curA_ok)
    lost_set_B_mask = glorys_wet_0 & ~(sst_ok & sla_ok & sss_ok & curB_ok)
    lost_union_mask = lost_set_A_mask | lost_set_B_mask

    lost_indices = np.argwhere(lost_union_mask)
    # Sort primarily by latitude center, secondarily by longitude center
    # Target grid row index i corresponds to latitude, column index j to longitude
    lost_indices_sorted = lost_indices[np.lexsort((lost_indices[:, 1], lost_indices[:, 0]))]

    lost_cells_records = []
    for rank, (i, j) in enumerate(lost_indices_sorted, start=1):
        lat_c = float(TARGET_LAT_CENTERS[i])
        lon_c = float(TARGET_LON_CENTERS[j])
        f_T_val = float(f_T_0[i, j])

        inv_inputs = []
        if not sst_ok[i, j]:
            inv_inputs.append("SST")
        if not sla_ok[i, j]:
            inv_inputs.append("SLA")
        if not sss_ok[i, j]:
            inv_inputs.append("SSS")
        if not curA_ok[i, j]:
            inv_inputs.append("Cur_A(DUACS)")
        if not curB_ok[i, j]:
            inv_inputs.append("Cur_B(MULTIOBS)")

        lost_cells_records.append({
            "rank": rank,
            "target_i": int(i),
            "target_j": int(j),
            "latitude": round(lat_c, 4),
            "longitude": round(lon_c, 4),
            "f_T": round(f_T_val, 6),
            "f_sst": round(float(f_sst_0[i, j]), 4),
            "f_sla": round(float(f_sla_0[i, j]), 4),
            "f_sss": round(float(f_sss_0[i, j]), 4),
            "f_duacs_ugos": round(float(f_curA_u0[i, j]), 4),
            "f_duacs_vgos": round(float(f_curA_v0[i, j]), 4),
            "f_multiobs_uo": round(float(f_curB_u0[i, j]), 4),
            "f_multiobs_vo": round(float(f_curB_v0[i, j]), 4),
            "invalid_inputs": "; ".join(inv_inputs),
            "lost_in_set_A": bool(lost_set_A_mask[i, j]),
            "lost_in_set_B": bool(lost_set_B_mask[i, j]),
            "num_invalid_inputs_all": len(inv_inputs),
        })

    df_lost_cells = pd.DataFrame(lost_cells_records)
    lost_cells_csv_path = os.path.join(output_dir, "lost_cells_level0.csv")
    df_lost_cells.to_csv(lost_cells_csv_path, index=False)

    # 4 SST lake cells in GLORYS wet set analysis
    FOUR_LAKE_COORDS = [
        (7.425, 100.325),
        (19.525, 85.275),
        (19.625, 85.375),
        (19.675, 85.475),
    ]

    lake_cells_info = []
    for lat, lon in FOUR_LAKE_COORDS:
        target_i = int(np.floor((lat - TARGET_LAT_START) / TARGET_STEP))
        target_j = int(np.floor((lon - TARGET_LON_START) / TARGET_STEP))
        f_sst_cell = float(f_sst_0[target_i, target_j])
        f_T_cell = float(f_T_0[target_i, target_j])
        lake_cells_info.append({
            "native_lat": lat,
            "native_lon": lon,
            "target_i": target_i,
            "target_j": target_j,
            "target_lat_bounds": [5.0 + 0.25 * target_i, 5.25 + 0.25 * target_i],
            "target_lon_bounds": [45.0 + 0.25 * target_j, 45.25 + 0.25 * target_j],
            "f_sst_with_lake_rule": round(f_sst_cell, 6),
            "f_T_level0": round(f_T_cell, 6),
            "valid_in_glorys_tau05": bool(f_T_cell >= 0.5),
            "valid_in_sst_tau05": bool(f_sst_cell >= 0.5),
        })

    # -------------------------------------------------------------------------
    # Output 4: sss_daily_coverage.csv
    # -------------------------------------------------------------------------
    sss_rows = []
    for day in range(31):
        dt_str = (datetime.date(2019, 1, 1) + datetime.timedelta(days=day)).isoformat()
        sss_rows.append({
            "day": day,
            "date": dt_str,
            "native_valid_cells": sss_daily_native_valid[day],
            "target_cells_f_ge_05": sss_daily_ge_05[day],
            "status_changed_from_prev_day": sss_daily_status_changes[day],
        })

    df_sss = pd.DataFrame(sss_rows)
    sss_csv_path = os.path.join(output_dir, "sss_daily_coverage.csv")
    df_sss.to_csv(sss_csv_path, index=False)

    # -------------------------------------------------------------------------
    # Output 5: Plots
    # -------------------------------------------------------------------------
    # Plot 1: Coverage fraction vs tauI for each input
    tau_sweep = np.linspace(0.01, 1.0, 100)
    plt.figure(figsize=(10, 6))

    # Calculate day 0 coverage curves
    curve_sst = [np.mean(f_sst_0 >= t) for t in tau_sweep]
    curve_sla = [np.mean(f_sla_0 >= t) for t in tau_sweep]
    curve_sss = [np.mean(f_sss_0 >= t) for t in tau_sweep]
    curve_cur_A = [np.mean((f_curA_u0 >= t) & (f_curA_v0 >= t)) for t in tau_sweep]
    curve_cur_B = [np.mean((f_curB_u0 >= t) & (f_curB_v0 >= t)) for t in tau_sweep]
    curve_glorys_l0 = [np.mean(f_T_0 >= t) for t in tau_sweep]

    plt.plot(tau_sweep, curve_sst, label="OSTIA SST (analysed_sst)", color="crimson", lw=2)
    plt.plot(tau_sweep, curve_sla, label="DUACS SLA (sla)", color="dodgerblue", lw=2)
    plt.plot(tau_sweep, curve_sss, label="MULTIOBS SSS (sos, day 0)", color="orange", lw=2)
    plt.plot(tau_sweep, curve_cur_A, label="DUACS Currents Set A (ugos+vgos)", color="purple", lw=2, linestyle="--")
    plt.plot(tau_sweep, curve_cur_B, label="MULTIOBS Currents Set B (uo+vo)", color="forestgreen", lw=2, linestyle="-.")
    plt.plot(tau_sweep, curve_glorys_l0, label="GLORYS thetao (level 0, Method A)", color="black", lw=2.5)

    plt.title("Target Grid Coverage Fraction vs Validity Threshold $\\tau_I$ (January 2019 Day 0)", fontsize=13, fontweight="bold")
    plt.xlabel("Validity Threshold $\\tau_I$ (fraction of valid footprint)", fontsize=11)
    plt.ylabel("Fraction of Total Target Cells (N=24,000)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plot_cov_path = os.path.join(output_dir, "coverage_fraction_vs_tauI.png")
    plt.savefig(plot_cov_path, dpi=200)
    plt.close()

    # Plot 2: Map of target cells lost at level 0 (tauT=0.5, tauI=0.5) coloured by number of invalid inputs
    plt.figure(figsize=(12, 6))
    lost_map = np.zeros((TARGET_N_LAT, TARGET_N_LON), dtype=np.float64)
    lost_map[:] = np.nan

    # Fill background ocean (common mask Set B) with 0
    common_mask_map = (glorys_wet_0 & sst_ok & sla_ok & sss_ok & curB_ok)
    lost_map[common_mask_map] = 0

    for rec in lost_cells_records:
        ti = rec["target_i"]
        tj = rec["target_j"]
        # Number of invalid inputs among the 5 fundamental inputs
        lost_map[ti, tj] = rec["num_invalid_inputs_all"]

    cmap = matplotlib.colormaps["YlOrRd"].resampled(6)
    cmap.set_bad(color="whitesmoke")

    # Extent: [lon_min, lon_max, lat_min, lat_max]
    im = plt.imshow(lost_map, origin="lower", extent=[45.0, 105.0, 5.0, 30.0], cmap=cmap, vmin=0.5, vmax=5.5)
    cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5], shrink=0.8)
    cbar.set_label("Number of Invalid Inputs per Lost Target Cell", fontsize=10)

    plt.title("Target Cells Lost at Level 0 ($\\tau_T=0.5, \\tau_I=0.5$)\nColoured by Number of Invalid Inputs", fontsize=13, fontweight="bold")
    plt.xlabel("Longitude (deg E)", fontsize=11)
    plt.ylabel("Latitude (deg N)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plot_map_path = os.path.join(output_dir, "lost_cells_level0_map.png")
    plt.savefig(plot_map_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # STEP 0 (After): SHA256, Size, Mtime After
    # -------------------------------------------------------------------------
    meta_after = {
        "sst": get_file_metadata(sst_path),
        "sla": get_file_metadata(sla_path),
        "sss": get_file_metadata(sss_path),
        "cur": get_file_metadata(cur_path),
        "glorys": get_file_metadata(glorys_path),
        "static": get_file_metadata(static_path),
    }

    # Verify exact match before vs after
    sha256_match_verified = True
    for key in RAW_FILES:
        if meta_before[key]["sha256"] != meta_after[key]["sha256"]:
            sha256_match_verified = False
        if meta_before[key]["size_bytes"] != meta_after[key]["size_bytes"]:
            sha256_match_verified = False
        if meta_before[key]["mtime"] != meta_after[key]["mtime"]:
            sha256_match_verified = False

    # -------------------------------------------------------------------------
    # Output 6: JSON Summary
    # -------------------------------------------------------------------------
    json_summary = {
        "task_name": "input_aggregation_census",
        "date_executed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256_match_verified": sha256_match_verified,
        "raw_files_metadata_before": meta_before,
        "raw_files_metadata_after": meta_after,
        "grid_nesting_ratios": {
            "target_grid": {"n_lat": TARGET_N_LAT, "n_lon": TARGET_N_LON, "step_deg": TARGET_STEP},
            "OSTIA_SST": {"ratio_y": ratio_sst_y, "ratio_x": ratio_sst_x, "cells_per_block": ratio_sst_y * ratio_sst_x},
            "DUACS_SLA": {"ratio_y": ratio_sla_y, "ratio_x": ratio_sla_x, "cells_per_block": ratio_sla_y * ratio_sla_x},
            "MULTIOBS_SSS": {"ratio_y": ratio_sss_y, "ratio_x": ratio_sss_x, "cells_per_block": ratio_sss_y * ratio_sss_x},
            "MULTIOBS_CURRENTS": {"ratio_y": ratio_cur_y, "ratio_x": ratio_cur_x, "cells_per_block": ratio_cur_y * ratio_cur_x},
            "GLORYS": {"relationship": "straddle", "weights_1d": [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]},
        },
        "integrity_check": {
            "status": "PASS" if all(item["matches_exact"] for item in daily_integrity) else "FAIL",
            "cur_1x1_exact_match_all_days": cur_1x1_match_all_days,
            "sample_days": daily_integrity,
        },
        "sst_mask_findings": {
            "unique_masks_non_fill": [int(x) for x in sst_unique_masks_d0],
            "lake_rule_removed_count": sst_lake_rule_removed_d0,
            "lake_cells_level0_wet_analysis": lake_cells_info,
        },
        "coverage_summary_day0": {
            row["variable"]: {
                "day0_f_eq_1": row["day0_f_eq_1"],
                "day0_f_ge_075": row["day0_f_ge_075"],
                "day0_f_ge_05": row["day0_f_ge_05"],
                "day0_f_gt_0": row["day0_f_gt_0"],
                "day0_f_eq_0": row["day0_f_eq_0"],
            }
            for row in cov_rows
        },
        "common_mask_census_key_findings": {
            "level0_tauT05_tauI05": {
                "d0_glorys_valid": int(census_grid_31[0, 0, 0, 0, 0]),
                "d0_common_set_A": int(census_grid_31[0, 0, 0, 0, 1]),
                "d0_common_set_B": int(census_grid_31[0, 0, 0, 0, 2]),
                "d0_lost_sst": int(census_grid_31[0, 0, 0, 0, 3]),
                "d0_lost_sla": int(census_grid_31[0, 0, 0, 0, 4]),
                "d0_lost_sss": int(census_grid_31[0, 0, 0, 0, 5]),
                "d0_lost_cur_A": int(census_grid_31[0, 0, 0, 0, 6]),
                "d0_lost_cur_B": int(census_grid_31[0, 0, 0, 0, 7]),
                "min31_common_set_A": int(np.min(census_grid_31[:, 0, 0, 0, 1])),
                "min31_common_set_B": int(np.min(census_grid_31[:, 0, 0, 0, 2])),
            },
            "level0_tauT10_tauI10": {
                "d0_glorys_valid": int(census_grid_31[0, 0, 2, 2, 0]),
                "d0_common_set_A": int(census_grid_31[0, 0, 2, 2, 1]),
                "d0_common_set_B": int(census_grid_31[0, 0, 2, 2, 2]),
                "min31_common_set_A": int(np.min(census_grid_31[:, 0, 2, 2, 1])),
                "min31_common_set_B": int(np.min(census_grid_31[:, 0, 2, 2, 2])),
            },
            "lost_cells_level0_count": len(lost_cells_records),
        },
        "sss_temporal_summary": {
            "min_target_cells_ge_05": min(sss_daily_ge_05),
            "max_target_cells_ge_05": max(sss_daily_ge_05),
            "total_status_changes_month": sum(sss_daily_status_changes),
            "days_with_status_changes": [day for day, c in enumerate(sss_daily_status_changes) if c > 0],
        },
        "unknown_fields": {
            "recommended_wet_threshold": None,
            "recommended_wet_threshold_reason": "Diagnostic only; threshold selection out of scope per contract rules",
            "recommended_currents_source": None,
            "recommended_currents_source_reason": "Diagnostic only; currents source selection out of scope per contract rules",
            "training_dataset_partition": None,
            "training_dataset_partition_reason": "Diagnostic only; dataset creation out of scope per contract rules",
        },
    }

    json_path = os.path.join(output_dir, "input_aggregation_census.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    return json_summary


# =============================================================================
# Formatted Table Printing for Verbatim Walkthrough
# =============================================================================

def print_verbatim_walkthrough_tables(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Print exact tables generated by code from saved CSV/JSON for walkthrough."""
    json_path = os.path.join(output_dir, "input_aggregation_census.json")
    cov_path = os.path.join(output_dir, "input_coverage_by_variable.csv")
    census_path = os.path.join(output_dir, "common_mask_census.csv")
    lost_path = os.path.join(output_dir, "lost_cells_level0.csv")
    sss_path = os.path.join(output_dir, "sss_daily_coverage.csv")

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df_cov = pd.read_csv(cov_path)
    df_census = pd.read_csv(census_path)
    df_lost = pd.read_csv(lost_path)
    df_sss = pd.read_csv(sss_path)

    print("\n" + "=" * 90)
    print("VERBATIM WALKTHROUGH TABLES (GENERATED FROM SAVED CSV/JSON)")
    print("=" * 90)

    # 1. Integrity Check Table
    print("\n--- 1. AGGREGATION INTEGRITY CHECK TABLE (DAYS 0, 15, 30) ---")
    int_records = summary["integrity_check"]["sample_days"]
    int_df = pd.DataFrame(int_records)
    print(int_df.to_string(index=False))

    # 2. Coverage Table
    print("\n--- 2. INPUT COVERAGE BY VARIABLE TABLE ---")
    print(df_cov.to_string(index=False))

    # 3. Common-Mask Census Rows for Levels 0, 10, 20, 28, 35
    print("\n--- 3. COMMON-MASK CENSUS (LEVELS 0, 10, 20, 28, 35) ---")
    key_levels = [0, 10, 20, 28, 35]
    subset_census = df_census[df_census["level"].isin(key_levels)]
    cols_to_print = [
        "level", "tauT", "tauI", "d0_glorys_valid", "min31_glorys_valid",
        "d0_common_set_A", "min31_common_set_A", "d0_common_set_B", "min31_common_set_B",
        "d0_lost_sst", "min31_lost_sst", "d0_lost_sla", "min31_lost_sla",
        "d0_lost_sss", "min31_lost_sss", "d0_lost_cur_A", "min31_lost_cur_A",
        "d0_lost_cur_B", "min31_lost_cur_B"
    ]
    print(subset_census[cols_to_print].to_string(index=False))

    # 4. Lake Cell Rows
    print("\n--- 4. FOUR SST LAKE CELLS IN GLORYS WET SET ---")
    lake_df = pd.DataFrame(summary["sst_mask_findings"]["lake_cells_level0_wet_analysis"])
    print(lake_df.to_string(index=False))

    # 5. SSS Daily Coverage Summary
    print("\n--- 5. SSS DAILY COVERAGE SUMMARY ---")
    print(df_sss.to_string(index=False))

    # 6. Lost Cells Level 0 (Per-Input Counts and First 50 sorted by lat, lon)
    print("\n--- 6. LOST CELLS AT LEVEL 0 (tauT=0.5, tauI=0.5) PER-INPUT COUNTS & FIRST 50 CELLS ---")
    key_findings = summary["common_mask_census_key_findings"]["level0_tauT05_tauI05"]
    print("PER-INPUT LOSS COUNTS AMONG GLORYS-WET CELLS (f_T >= 0.5):")
    print(f"  Total GLORYS valid cells (f_T >= 0.5): {key_findings['d0_glorys_valid']}")
    print(f"  Lost because SST invalid (f_sst < 0.5): {key_findings['d0_lost_sst']}")
    print(f"  Lost because SLA invalid (f_sla < 0.5): {key_findings['d0_lost_sla']}")
    print(f"  Lost because SSS invalid (f_sss < 0.5): {key_findings['d0_lost_sss']}")
    print(f"  Lost because Cur_A (DUACS) invalid: {key_findings['d0_lost_cur_A']}")
    print(f"  Lost because Cur_B (MULTIOBS) invalid: {key_findings['d0_lost_cur_B']}")
    print(f"  Common Mask Set A valid count (day 0): {key_findings['d0_common_set_A']} (lost: {key_findings['d0_glorys_valid'] - key_findings['d0_common_set_A']})")
    print(f"  Common Mask Set B valid count (day 0): {key_findings['d0_common_set_B']} (lost: {key_findings['d0_glorys_valid'] - key_findings['d0_common_set_B']})")
    print(f"  Total unique cells lost in Set A or Set B: {summary['common_mask_census_key_findings']['lost_cells_level0_count']}")
    print("\nFIRST 50 LOST CELLS (SORTED BY LATITUDE THEN LONGITUDE):")
    cols_lost = ["rank", "target_i", "target_j", "latitude", "longitude", "f_T", "f_sst", "f_sla", "f_sss", "invalid_inputs", "lost_in_set_A", "lost_in_set_B"]
    print(df_lost.head(50)[cols_lost].to_string(index=False))

    # 7. Complete JSON
    print("\n--- 7. INPUT AGGREGATION CENSUS JSON (FULL) ---")
    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 90)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Input aggregation to target grid and common-mask census.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory path")
    parser.add_argument("--print-only", action="store_true", help="Only print tables from existing outputs")
    args = parser.parse_args()

    if args.print_only:
        print_verbatim_walkthrough_tables(args.output_dir)
        return

    print("Running input aggregation and common-mask census across 31 days...")
    summary = run_census(output_dir=args.output_dir)
    print(f"Run completed. Status: {summary['integrity_check']['status']}")
    print_verbatim_walkthrough_tables(args.output_dir)


if __name__ == "__main__":
    main()
