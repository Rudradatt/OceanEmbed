#!/usr/bin/env python3
"""
Input Builder v1 (OSTIA SST, DUACS SLA, MULTIOBS SSS, MULTIOBS Currents)
OceanEmbed Preprocessing Phase.

Builds standardized 0.25-degree daily input fields from multi-source remote sensing
and reanalysis products for the OceanEmbed pipeline according to policy decisions:
- I1: Value = block mean over valid native cells; frac = valid native cells / total native cells in block.
- I2: Store the value wherever frac > 0. Apply no threshold in the builder.
- I3: Validity:
      - SST analysed_sst: non-fill (-32768) AND (mask & 6) == 0 (excludes land bit 2 and lake bit 4).
      - SLA sla: non-fill (-2147483647).
      - SSS sos: non-fill (-32767.0).
      - Currents uo, vo: non-fill (32767) at depth index 0 (0 m).
      Manual decoding without relying on valid_min/valid_max attributes.
- I4: Units:
      - SST: degC (decoded Kelvin minus 273.15).
      - SLA: m (decoded from raw int32 * 0.0001).
      - SSS: .001 (recorded from raw file units attribute).
      - Currents: m/s (decoded from raw int16 * 0.001).
- I5: No normalization, anomalies, climatology, imputation or error-field aggregation.
- I6: Time axis copied directly from reference target file (glorys_targets_2019-01.nc);
      verify that each source day equals the reference date.

Self-contained module: does not depend on diagnostics scripts.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import h5netcdf
import numpy as np
import xarray as xr


# Default paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_POLICY_PATH = os.path.join(PROJECT_ROOT, "configs", "input_policy_v1.json")
DEFAULT_TARGET_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "glorys_targets", "glorys_targets_2019-01.nc"
)
DEFAULT_SST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sst", "sst_ostia_2019-01.nc")
DEFAULT_SLA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sla", "sla_duacs_2019-01.nc")
DEFAULT_SSS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sss", "sss_multiobs_2019-01.nc")
DEFAULT_CUR_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "currents", "currents_multiobs_2019-01.nc"
)
DEFAULT_OUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "inputs", "inputs_2019-01.nc"
)


# =============================================================================
# Provenance and Hashing Helpers
# =============================================================================

def compute_file_sha256(filepath: Union[str, os.PathLike]) -> str:
    """Compute SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash(repo_dir: Optional[str] = None) -> str:
    """Get current git commit hash, or 'unavailable' if git is not available."""
    if repo_dir is None:
        repo_dir = PROJECT_ROOT
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unavailable"


def load_policy(policy: Union[Dict[str, Any], str, os.PathLike]) -> Dict[str, Any]:
    """Load policy dict or load JSON from path."""
    if isinstance(policy, dict):
        return policy
    with open(policy, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Time Parsing and Verification Helpers
# =============================================================================

def parse_time_units(units_str: str) -> Tuple[str, datetime.datetime]:
    """Parse CF time units string (e.g. 'seconds since 1981-01-01 00:00:00')."""
    match = re.match(
        r"(seconds|minutes|hours|days)\s+since\s+([\d]{4}-[\d]{2}-[\d]{2}(?:[T\s][\d]{2}:[\d]{2}:[\d]{2}(?:\.[\d]+)?)?)",
        units_str.strip(),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized time unit format: '{units_str}'")
    unit_type = match.group(1).lower()
    base_str = match.group(2).replace("T", " ")
    if len(base_str) == 10:
        base_dt = datetime.datetime.strptime(base_str, "%Y-%m-%d")
    else:
        base_dt = datetime.datetime.strptime(base_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
    base_dt = base_dt.replace(tzinfo=datetime.timezone.utc)
    return unit_type, base_dt


def decode_timestamps(time_array: np.ndarray, units_str: str) -> List[datetime.datetime]:
    """Decode raw numeric time array into list of timezone-aware UTC datetimes."""
    unit_type, base_dt = parse_time_units(units_str)
    decoded: List[datetime.datetime] = []
    for val in time_array:
        v = float(val)
        if unit_type == "seconds":
            dt = base_dt + datetime.timedelta(seconds=v)
        elif unit_type == "minutes":
            dt = base_dt + datetime.timedelta(minutes=v)
        elif unit_type == "hours":
            dt = base_dt + datetime.timedelta(hours=v)
        elif unit_type == "days":
            dt = base_dt + datetime.timedelta(days=v)
        else:
            raise ValueError(f"Unsupported time unit: {unit_type}")
        decoded.append(dt)
    return decoded


def verify_time_match(
    src_times: np.ndarray,
    src_units: str,
    ref_times: np.ndarray,
    ref_units: str,
    source_name: str = "source",
) -> List[str]:
    """Verify that source timestamps have exactly 00:00:00 UTC and equal reference dates.
    
    Returns list of ISO date strings 'YYYY-MM-DD'.
    Raises ValueError on any mismatch.
    """
    src_dts = decode_timestamps(src_times, src_units)
    ref_dts = decode_timestamps(ref_times, ref_units)

    if len(src_dts) != len(ref_dts):
        raise ValueError(
            f"{source_name} time count ({len(src_dts)}) does not match reference ({len(ref_dts)})"
        )

    date_strings: List[str] = []
    for i, (s_dt, r_dt) in enumerate(zip(src_dts, ref_dts)):
        if s_dt.date() != r_dt.date():
            raise ValueError(
                f"{source_name} date mismatch at index {i}: source date {s_dt.date()} != ref date {r_dt.date()}"
            )
        if s_dt.hour != 0 or s_dt.minute != 0 or s_dt.second != 0:
            raise ValueError(
                f"{source_name} timestamp at index {i} is not 00:00:00 UTC: {s_dt.time()}"
            )
        date_strings.append(s_dt.strftime("%Y-%m-%d"))

    return date_strings


# =============================================================================
# Grid Nesting Check
# =============================================================================

def nesting_check(
    native_centers: np.ndarray,
    target_start: float,
    target_step: float,
    n_target: int,
    tol: float = 1e-4,
) -> int:
    """Verify numerically that native cell edges nest exactly into target edges.
    
    Computes and returns the integer ratio (native cells per target cell).
    Raises ValueError and halts immediately if not an integer or if edges do not coincide.
    """
    native_centers = np.asarray(native_centers, dtype=np.float64)
    m = len(native_centers)
    if m < 1:
        raise ValueError("Native grid must have at least 1 coordinate point")

    if m == 1:
        native_step = target_step
    else:
        diffs = np.diff(native_centers)
        native_step = float(np.mean(diffs))
        if not np.allclose(diffs, native_step, atol=tol):
            raise ValueError(
                f"Native grid spacing is not uniform within tolerance {tol}: min={diffs.min()}, max={diffs.max()}"
            )

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
        raise ValueError(
            f"Native-cells-per-target-cell ratio {ratio:.6f} is not an integer within tolerance {tol}"
        )

    if ratio_int < 1:
        raise ValueError(f"Native-cells-per-target-cell ratio {ratio_int} must be >= 1")

    if m != n_target * ratio_int:
        raise ValueError(
            f"Native grid size {m} does not equal n_target * ratio ({n_target} * {ratio_int} = {n_target * ratio_int})"
        )

    # Verify every target cell edge coincides with a native cell edge
    for i in range(n_target + 1):
        t_edge = target_start + i * target_step
        native_edge_idx = i * ratio_int
        if native_edge_idx == 0:
            n_edge = native_centers[0] - native_step / 2.0
        elif native_edge_idx == m:
            n_edge = native_centers[-1] + native_step / 2.0
        else:
            n_edge = (native_centers[native_edge_idx - 1] + native_centers[native_edge_idx]) / 2.0
        if abs(t_edge - n_edge) > tol:
            raise ValueError(
                f"Target edge {i} ({t_edge:.8f}) does not coincide with native edge ({n_edge:.8f}) within {tol}"
            )

    return ratio_int


# =============================================================================
# Vectorized Block Aggregation
# =============================================================================

def block_aggregate(
    values: np.ndarray,
    valid: np.ndarray,
    ratio: Union[int, Tuple[int, int]],
    n_target: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Block aggregation of a 2D native array to the target grid using reshaping.
    
    No per-cell Python loops.
    
    Parameters:
        values: float or int array of shape (ny, nx)
        valid: bool array of shape (ny, nx)
        ratio: integer ratio or tuple (ratio_y, ratio_x)
        n_target: optional tuple (n_target_y, n_target_x); inferred if None
        
    Returns:
        (value, frac):
            value: float64 array of shape (n_target_y, n_target_x) with mean over valid native cells,
                   or NaN if valid cells == 0.
            frac: float64 array of shape (n_target_y, n_target_x) with valid fraction (0.0 .. 1.0).
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if values.shape != valid.shape:
        raise ValueError(f"Shape mismatch: values {values.shape} vs valid {valid.shape}")

    if isinstance(ratio, (int, np.integer)):
        ratio_y = int(ratio)
        ratio_x = int(ratio)
    else:
        ratio_y = int(ratio[0])
        ratio_x = int(ratio[1])

    if ratio_y < 1 or ratio_x < 1:
        raise ValueError(f"Aggregation ratios must be >= 1, got ({ratio_y}, {ratio_x})")

    ny, nx = values.shape
    if n_target is None:
        if ny % ratio_y != 0 or nx % ratio_x != 0:
            raise ValueError(
                f"Native dimensions ({ny}, {nx}) not divisible by ratios ({ratio_y}, {ratio_x})"
            )
        n_target_y = ny // ratio_y
        n_target_x = nx // ratio_x
    else:
        n_target_y, n_target_x = n_target
        if ny != n_target_y * ratio_y or nx != n_target_x * ratio_x:
            raise ValueError(
                f"Native shape ({ny}, {nx}) does not match expected ({n_target_y * ratio_y}, {n_target_x * ratio_x})"
            )

    cells_per_block = ratio_y * ratio_x

    # Reshape and swap axes to group native cells into blocks
    val_blocks = (
        values.reshape(n_target_y, ratio_y, n_target_x, ratio_x)
        .swapaxes(1, 2)
        .reshape(n_target_y, n_target_x, cells_per_block)
    )
    mask_blocks = (
        valid.reshape(n_target_y, ratio_y, n_target_x, ratio_x)
        .swapaxes(1, 2)
        .reshape(n_target_y, n_target_x, cells_per_block)
    )

    valid_counts = np.sum(mask_blocks, axis=-1)
    frac = valid_counts.astype(np.float64) / float(cells_per_block)

    # Replace invalid cells with 0.0 before summing to prevent NaN leakage
    safe_values = np.where(mask_blocks, val_blocks, 0.0)
    sums = np.sum(safe_values, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(valid_counts > 0, sums / valid_counts.astype(np.float64), np.nan)

    return value, frac


# =============================================================================
# Validity Rules and Manual Decoders
# =============================================================================

def compute_sst_validity(
    sst_raw: np.ndarray,
    mask_raw: np.ndarray,
    fill_value: int = -32768,
) -> np.ndarray:
    """SST analysed_sst validity: non-fill AND (mask & 6) == 0 (excludes land bit 2 and lake bit 4)."""
    sst_raw = np.asarray(sst_raw)
    mask_raw = np.asarray(mask_raw)
    non_fill = sst_raw != fill_value
    mask_allowed = (mask_raw & 6) == 0
    return non_fill & mask_allowed


def compute_sla_validity(
    sla_raw: np.ndarray,
    fill_value: int = -2147483647,
) -> np.ndarray:
    """SLA validity: non-fill by its own fill value."""
    return np.asarray(sla_raw) != fill_value


def compute_sss_validity(
    sss_raw: np.ndarray,
    fill_value: float = -32767.0,
) -> np.ndarray:
    """SSS sos validity: non-fill by its own fill value."""
    return np.asarray(sss_raw) != fill_value


def compute_currents_validity(
    cur_raw: np.ndarray,
    fill_value: int = 32767,
) -> np.ndarray:
    """MULTIOBS currents validity: non-fill by its own fill value."""
    return np.asarray(cur_raw) != fill_value


def decode_sst(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of OSTIA SST raw int16 to Kelvin."""
    return np.asarray(raw, dtype=np.float64) * 0.009999999776482582 + 273.1499938964844


def decode_sst_degc(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of OSTIA SST raw int16 to degrees Celsius (decoded Kelvin minus 273.15)."""
    return decode_sst(raw) - 273.15


def decode_sla(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of DUACS SLA raw int32 to meters."""
    return np.asarray(raw, dtype=np.float64) * 0.0001


def decode_sss(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of MULTIOBS SSS float32 (identity)."""
    return np.asarray(raw, dtype=np.float64)


def decode_currents(raw: np.ndarray) -> np.ndarray:
    """Manual decoding of MULTIOBS currents raw int16 to m/s."""
    return np.asarray(raw, dtype=np.float64) * 0.001


# =============================================================================
# Per-Source Single-Day Aggregation
# =============================================================================

def build_day_sst(
    sst_raw: np.ndarray,
    mask_raw: np.ndarray,
    ratio: Union[int, Tuple[int, int]] = (5, 5),
    n_target: Tuple[int, int] = (100, 240),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build single day SST: decode to degC and block aggregate to target grid."""
    valid = compute_sst_validity(sst_raw, mask_raw)
    degc = decode_sst_degc(sst_raw)
    return block_aggregate(degc, valid, ratio=ratio, n_target=n_target)


def build_day_sla(
    sla_raw: np.ndarray,
    ratio: Union[int, Tuple[int, int]] = (2, 2),
    n_target: Tuple[int, int] = (100, 240),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build single day SLA: decode to meters and block aggregate to target grid."""
    valid = compute_sla_validity(sla_raw)
    decoded_m = decode_sla(sla_raw)
    return block_aggregate(decoded_m, valid, ratio=ratio, n_target=n_target)


def build_day_sss(
    sss_raw: np.ndarray,
    ratio: Union[int, Tuple[int, int]] = (2, 2),
    n_target: Tuple[int, int] = (100, 240),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build single day SSS: decode to source units and block aggregate to target grid."""
    if sss_raw.ndim == 3 and sss_raw.shape[0] == 1:
        sss_raw_2d = sss_raw[0]
    elif sss_raw.ndim == 2:
        sss_raw_2d = sss_raw
    else:
        raise ValueError(f"Unexpected SSS raw shape: {sss_raw.shape}")

    valid = compute_sss_validity(sss_raw_2d)
    decoded = decode_sss(sss_raw_2d)
    return block_aggregate(decoded, valid, ratio=ratio, n_target=n_target)


def build_day_currents(
    uo_raw: np.ndarray,
    vo_raw: np.ndarray,
    ratio: Union[int, Tuple[int, int]] = (1, 1),
    n_target: Tuple[int, int] = (100, 240),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build single day currents (uo, vo) at depth index 0: decode to m/s and aggregate."""
    if uo_raw.ndim == 3:
        uo_raw_2d = uo_raw[0]
        vo_raw_2d = vo_raw[0]
    elif uo_raw.ndim == 2:
        uo_raw_2d = uo_raw
        vo_raw_2d = vo_raw
    else:
        raise ValueError(f"Unexpected currents raw shape: {uo_raw.shape}")

    valid_uo = compute_currents_validity(uo_raw_2d)
    valid_vo = compute_currents_validity(vo_raw_2d)

    dec_uo = decode_currents(uo_raw_2d)
    dec_vo = decode_currents(vo_raw_2d)

    uo_val, uo_frac = block_aggregate(dec_uo, valid_uo, ratio=ratio, n_target=n_target)
    vo_val, vo_frac = block_aggregate(dec_vo, valid_vo, ratio=ratio, n_target=n_target)

    return uo_val, uo_frac, vo_val, vo_frac


# =============================================================================
# Full Month Pipeline
# =============================================================================

def build_month_file(
    policy_path: str = DEFAULT_POLICY_PATH,
    sst_path: str = DEFAULT_SST_PATH,
    sla_path: str = DEFAULT_SLA_PATH,
    sss_path: str = DEFAULT_SSS_PATH,
    cur_path: str = DEFAULT_CUR_PATH,
    target_path: str = DEFAULT_TARGET_PATH,
    out_path: str = DEFAULT_OUT_PATH,
    manifest_out_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Process all sources for the month, writing standardized inputs NetCDF and manifest JSON.
    
    Returns manifest dictionary.
    """
    pol = load_policy(policy_path)
    grid_cfg = pol["target_grid"]
    n_lat = int(grid_cfg["n_lat"])
    n_lon = int(grid_cfg["n_lon"])
    lat_start = float(grid_cfg["lat_start"])
    lon_start = float(grid_cfg["lon_start"])
    lat_step = float(grid_cfg["lat_step"])
    lon_step = float(grid_cfg["lon_step"])

    # Compute provenance hashes
    sha_builder = compute_file_sha256(os.path.abspath(__file__))
    sha_policy = compute_file_sha256(policy_path)
    sha_sst = compute_file_sha256(sst_path)
    sha_sla = compute_file_sha256(sla_path)
    sha_sss = compute_file_sha256(sss_path)
    sha_cur = compute_file_sha256(cur_path)
    sha_target = compute_file_sha256(target_path)

    # Optional glorys reference files in policy
    ref_cfg = pol.get("reference", {})
    glorys_raw_path = ref_cfg.get("glorys_thetao_path")
    static_raw_path = ref_cfg.get("glorys_static_path")
    sha_glorys = compute_file_sha256(glorys_raw_path) if glorys_raw_path and os.path.exists(glorys_raw_path) else None
    sha_static = compute_file_sha256(static_raw_path) if static_raw_path and os.path.exists(static_raw_path) else None

    git_commit = get_git_commit_hash(PROJECT_ROOT)
    created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Load target reference coordinates and time
    with h5netcdf.File(target_path, "r") as f_target:
        t_lat = np.array(f_target.variables["latitude"][:], dtype=np.float32)
        t_lon = np.array(f_target.variables["longitude"][:], dtype=np.float32)
        ref_time_values = np.array(f_target.variables["time"][:])
        ref_time_attrs = dict(f_target.variables["time"].attrs)
        ref_time_units = str(ref_time_attrs.get("units", ""))
        ref_time_calendar = str(ref_time_attrs.get("calendar", "gregorian"))

    if not ref_time_units or not ref_time_calendar:
        raise ValueError(
            f"Target file {target_path} is missing time units ('{ref_time_units}') or calendar ('{ref_time_calendar}'). STOP."
        )

    n_days = len(ref_time_values)

    # Nesting and time checks for each source
    with h5netcdf.File(sst_path, "r") as f_sst:
        sst_lats = np.array(f_sst.variables["latitude"][:])
        sst_lons = np.array(f_sst.variables["longitude"][:])
        sst_times = np.array(f_sst.variables["time"][:])
        sst_time_units = str(f_sst.variables["time"].attrs.get("units", ""))
        ratio_sst_y = nesting_check(sst_lats, lat_start, lat_step, n_lat)
        ratio_sst_x = nesting_check(sst_lons, lon_start, lon_step, n_lon)
        dates_sst = verify_time_match(sst_times, sst_time_units, ref_time_values, ref_time_units, "SST")

    with h5netcdf.File(sla_path, "r") as f_sla:
        sla_lats = np.array(f_sla.variables["latitude"][:])
        sla_lons = np.array(f_sla.variables["longitude"][:])
        sla_times = np.array(f_sla.variables["time"][:])
        sla_time_units = str(f_sla.variables["time"].attrs.get("units", ""))
        ratio_sla_y = nesting_check(sla_lats, lat_start, lat_step, n_lat)
        ratio_sla_x = nesting_check(sla_lons, lon_start, lon_step, n_lon)
        dates_sla = verify_time_match(sla_times, sla_time_units, ref_time_values, ref_time_units, "SLA")

    with h5netcdf.File(sss_path, "r") as f_sss:
        sss_lats = np.array(f_sss.variables["latitude"][:])
        sss_lons = np.array(f_sss.variables["longitude"][:])
        sss_times = np.array(f_sss.variables["time"][:])
        sss_time_units = str(f_sss.variables["time"].attrs.get("units", ""))
        ratio_sss_y = nesting_check(sss_lats, lat_start, lat_step, n_lat)
        ratio_sss_x = nesting_check(sss_lons, lon_start, lon_step, n_lon)
        dates_sss = verify_time_match(sss_times, sss_time_units, ref_time_values, ref_time_units, "SSS")

    with h5netcdf.File(cur_path, "r") as f_cur:
        cur_lats = np.array(f_cur.variables["latitude"][:])
        cur_lons = np.array(f_cur.variables["longitude"][:])
        cur_times = np.array(f_cur.variables["time"][:])
        cur_time_units = str(f_cur.variables["time"].attrs.get("units", ""))
        ratio_cur_y = nesting_check(cur_lats, lat_start, lat_step, n_lat)
        ratio_cur_x = nesting_check(cur_lons, lon_start, lon_step, n_lon)
        dates_cur = verify_time_match(cur_times, cur_time_units, ref_time_values, ref_time_units, "Currents")

    # Allocate full 3D destination arrays (float32)
    sst_c_all = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sst_frac_all = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    sla_m_all = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sla_frac_all = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    sss_all = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sss_frac_all = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    uo_ms_all = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    uo_frac_all = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    vo_ms_all = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    vo_frac_all = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)

    daily_manifest: List[Dict[str, Any]] = []

    if verbose:
        print(f"Building month file for {n_days} days into {out_path}...")

    # Open files for day-by-day streaming
    with h5netcdf.File(sst_path, "r") as f_sst, \
         h5netcdf.File(sla_path, "r") as f_sla, \
         h5netcdf.File(sss_path, "r") as f_sss, \
         h5netcdf.File(cur_path, "r") as f_cur:

        for d in range(n_days):
            date_str = dates_sst[d]

            # SST
            sst_raw = f_sst.variables["analysed_sst"][d]
            mask_raw = f_sst.variables["mask"][d]
            sst_c, sst_f = build_day_sst(sst_raw, mask_raw, ratio=(ratio_sst_y, ratio_sst_x), n_target=(n_lat, n_lon))
            sst_c_all[d] = sst_c.astype(np.float32)
            sst_frac_all[d] = sst_f.astype(np.float32)

            # SLA
            sla_raw = f_sla.variables["sla"][d]
            sla_m, sla_f = build_day_sla(sla_raw, ratio=(ratio_sla_y, ratio_sla_x), n_target=(n_lat, n_lon))
            sla_m_all[d] = sla_m.astype(np.float32)
            sla_frac_all[d] = sla_f.astype(np.float32)

            # SSS
            sss_raw = f_sss.variables["sos"][d]
            sss_val, sss_f = build_day_sss(sss_raw, ratio=(ratio_sss_y, ratio_sss_x), n_target=(n_lat, n_lon))
            sss_all[d] = sss_val.astype(np.float32)
            sss_frac_all[d] = sss_f.astype(np.float32)

            # Currents (depth index 0)
            uo_raw = f_cur.variables["uo"][d, 0]
            vo_raw = f_cur.variables["vo"][d, 0]
            uo_val, uo_f, vo_val, vo_f = build_day_currents(
                uo_raw, vo_raw, ratio=(ratio_cur_y, ratio_cur_x), n_target=(n_lat, n_lon)
            )
            uo_ms_all[d] = uo_val.astype(np.float32)
            uo_frac_all[d] = uo_f.astype(np.float32)
            vo_ms_all[d] = vo_val.astype(np.float32)
            vo_frac_all[d] = vo_f.astype(np.float32)

            def get_counts(f_arr: np.ndarray) -> Dict[str, int]:
                return {
                    "frac_eq_1": int(np.sum(f_arr >= 1.0 - 1e-6)),
                    "frac_ge_075": int(np.sum(f_arr >= 0.75 - 1e-6)),
                    "frac_ge_05": int(np.sum(f_arr >= 0.5 - 1e-6)),
                    "frac_gt_0": int(np.sum(f_arr > 1e-6)),
                    "frac_eq_0": int(np.sum(f_arr < 1e-6)),
                }

            day_entry = {
                "day_index": d,
                "date": date_str,
                "counts": {
                    "analysed_sst": get_counts(sst_f),
                    "sla": get_counts(sla_f),
                    "sos": get_counts(sss_f),
                    "uo": get_counts(uo_f),
                    "vo": get_counts(vo_f),
                },
            }
            daily_manifest.append(day_entry)

            if verbose and (d in [0, 15, 30] or d == n_days - 1):
                print(
                    f"  Day {d:02d} ({date_str}): SST valid>0={day_entry['counts']['analysed_sst']['frac_gt_0']}, "
                    f"SLA valid>0={day_entry['counts']['sla']['frac_gt_0']}, "
                    f"SSS valid>0={day_entry['counts']['sos']['frac_gt_0']}, "
                    f"UO valid>0={day_entry['counts']['uo']['frac_gt_0']}"
                )

    # Coordinates
    coords = {
        "time": ("time", ref_time_values, ref_time_attrs),
        "latitude": (
            "latitude",
            t_lat,
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
            t_lon,
            {
                "long_name": "Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "unit_long": "Degrees East",
                "axis": "X",
            },
        ),
    }

    # Data variables
    data_vars = {
        "sst_c": (
            ("time", "latitude", "longitude"),
            sst_c_all,
            {
                "long_name": "Sea Surface Temperature in degC",
                "standard_name": "sea_surface_temperature",
                "units": "degC",
                "unit_long": "Degrees Celsius",
            },
        ),
        "sst_frac": (
            ("time", "latitude", "longitude"),
            sst_frac_all,
            {
                "long_name": "SST Valid Native Cell Fraction in Block",
                "units": "1",
                "unit_long": "Fraction",
            },
        ),
        "sla_m": (
            ("time", "latitude", "longitude"),
            sla_m_all,
            {
                "long_name": "Sea Level Anomaly in meters",
                "standard_name": "sea_surface_height_above_sea_level",
                "units": "m",
                "unit_long": "Meters",
            },
        ),
        "sla_frac": (
            ("time", "latitude", "longitude"),
            sla_frac_all,
            {
                "long_name": "SLA Valid Native Cell Fraction in Block",
                "units": "1",
                "unit_long": "Fraction",
            },
        ),
        "sss": (
            ("time", "latitude", "longitude"),
            sss_all,
            {
                "long_name": "Sea Surface Salinity",
                "standard_name": "sea_surface_salinity",
                "units": ".001",
                "unit_long": "Practical Salinity Unit (.001)",
            },
        ),
        "sss_frac": (
            ("time", "latitude", "longitude"),
            sss_frac_all,
            {
                "long_name": "SSS Valid Native Cell Fraction in Block",
                "units": "1",
                "unit_long": "Fraction",
            },
        ),
        "uo_ms": (
            ("time", "latitude", "longitude"),
            uo_ms_all,
            {
                "long_name": "Eastward Ocean Surface Current Velocity",
                "standard_name": "eastward_sea_water_velocity",
                "units": "m/s",
                "unit_long": "Meters per second",
            },
        ),
        "uo_frac": (
            ("time", "latitude", "longitude"),
            uo_frac_all,
            {
                "long_name": "uo Valid Native Cell Fraction in Block",
                "units": "1",
                "unit_long": "Fraction",
            },
        ),
        "vo_ms": (
            ("time", "latitude", "longitude"),
            vo_ms_all,
            {
                "long_name": "Northward Ocean Surface Current Velocity",
                "standard_name": "northward_sea_water_velocity",
                "units": "m/s",
                "unit_long": "Meters per second",
            },
        ),
        "vo_frac": (
            ("time", "latitude", "longitude"),
            vo_frac_all,
            {
                "long_name": "vo Valid Native Cell Fraction in Block",
                "units": "1",
                "unit_long": "Fraction",
            },
        ),
    }

    # Global attributes
    global_attrs = {
        "policy_name": pol.get("policy_name", "input_policy_v1"),
        "policy_version": pol.get("policy_version", "v1"),
        "target_grid": f"{n_lat}x{n_lon}_0.25deg",
        "raw_sst_sha256": sha_sst,
        "raw_sla_sha256": sha_sla,
        "raw_sss_sha256": sha_sss,
        "raw_cur_sha256": sha_cur,
        "raw_glorys_sha256": sha_glorys or "not_provided",
        "raw_static_sha256": sha_static or "not_provided",
        "target_file_sha256": sha_target,
        "policy_sha256": sha_policy,
        "builder_script_sha256": sha_builder,
        "git_commit": git_commit,
        "created_utc": created_utc,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "h5netcdf_version": h5netcdf.__version__,
    }

    ds_inputs = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attrs)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.float32(np.nan),
            "dtype": "float32",
        }
        for var in data_vars
    }
    encoding["time"] = {"dtype": ref_time_values.dtype}
    encoding["latitude"] = {"dtype": "float32"}
    encoding["longitude"] = {"dtype": "float32"}

    ds_inputs.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    ds_inputs.close()

    file_size = os.path.getsize(out_path)
    file_sha256 = compute_file_sha256(out_path)

    if manifest_out_path is None:
        manifest_out_path = os.path.splitext(out_path)[0] + "_manifest.json"

    # Compute overall range across 31 days per variable
    summary_31days: Dict[str, Any] = {}
    for var_key in ["analysed_sst", "sla", "sos", "uo", "vo"]:
        eq_1_vals = [d["counts"][var_key]["frac_eq_1"] for d in daily_manifest]
        ge_075_vals = [d["counts"][var_key]["frac_ge_075"] for d in daily_manifest]
        ge_05_vals = [d["counts"][var_key]["frac_ge_05"] for d in daily_manifest]
        gt_0_vals = [d["counts"][var_key]["frac_gt_0"] for d in daily_manifest]
        eq_0_vals = [d["counts"][var_key]["frac_eq_0"] for d in daily_manifest]

        summary_31days[var_key] = {
            "day0": daily_manifest[0]["counts"][var_key],
            "min31_f_eq_1": min(eq_1_vals),
            "max31_f_eq_1": max(eq_1_vals),
            "min31_f_ge_075": min(ge_075_vals),
            "max31_f_ge_075": max(ge_075_vals),
            "min31_f_ge_05": min(ge_05_vals),
            "max31_f_ge_05": max(ge_05_vals),
            "min31_f_gt_0": min(gt_0_vals),
            "max31_f_gt_0": max(gt_0_vals),
            "min31_f_eq_0": min(eq_0_vals),
            "max31_f_eq_0": max(eq_0_vals),
        }

    manifest_data = {
        "output_file": {
            "path": os.path.abspath(out_path),
            "size_bytes": file_size,
            "sha256": file_sha256,
        },
        "policy": {
            "name": pol.get("policy_name", "input_policy_v1"),
            "version": pol.get("policy_version", "v1"),
            "path": os.path.abspath(policy_path),
            "sha256": sha_policy,
        },
        "builder": {
            "path": os.path.abspath(__file__),
            "sha256": sha_builder,
            "git_commit": git_commit,
            "created_utc": created_utc,
        },
        "input_files": {
            "sst": {"path": os.path.abspath(sst_path), "sha256": sha_sst, "size": os.path.getsize(sst_path)},
            "sla": {"path": os.path.abspath(sla_path), "sha256": sha_sla, "size": os.path.getsize(sla_path)},
            "sss": {"path": os.path.abspath(sss_path), "sha256": sha_sss, "size": os.path.getsize(sss_path)},
            "cur": {"path": os.path.abspath(cur_path), "sha256": sha_cur, "size": os.path.getsize(cur_path)},
            "glorys_targets": {"path": os.path.abspath(target_path), "sha256": sha_target, "size": os.path.getsize(target_path)},
        },
        "coverage_summary_31days": summary_31days,
        "daily_manifest": daily_manifest,
    }

    with open(manifest_out_path, "w", encoding="utf-8") as f_m:
        json.dump(manifest_data, f_m, indent=2)

    if verbose:
        print(f"Build complete. Output NetCDF size: {file_size} bytes, SHA256: {file_sha256}")
        print(f"Manifest written to: {manifest_out_path}")

    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="OceanEmbed Input Builders v1")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to input policy JSON")
    parser.add_argument("--sst", default=DEFAULT_SST_PATH, help="Path to raw SST NetCDF")
    parser.add_argument("--sla", default=DEFAULT_SLA_PATH, help="Path to raw SLA NetCDF")
    parser.add_argument("--sss", default=DEFAULT_SSS_PATH, help="Path to raw SSS NetCDF")
    parser.add_argument("--cur", default=DEFAULT_CUR_PATH, help="Path to raw currents NetCDF")
    parser.add_argument("--target", default=DEFAULT_TARGET_PATH, help="Path to reference target NetCDF")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output NetCDF path")
    parser.add_argument("--manifest", default=None, help="Output manifest JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    args = parser.parse_args()

    build_month_file(
        policy_path=args.policy,
        sst_path=args.sst,
        sla_path=args.sla,
        sss_path=args.sss,
        cur_path=args.cur,
        target_path=args.target,
        out_path=args.out,
        manifest_out_path=args.manifest,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

