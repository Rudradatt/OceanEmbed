#!/usr/bin/env python3
"""
GLORYS Annual Snapshot Census 1993-2020 & Pre-Freeze Trajectory Diagnostic
(Read-Only Forensic Analysis)

Gathers forensic evidence across:
- Section A: Inventory & coordinate check (28 snapshot files 1993-2020)
- Section B: Mask consistency (finite mask vs Jan 2019 day-0 and static 3D mask)
- Section C: Per-snapshot, per-level census (zero codes, <0C, near-floor, near-ceiling)
- Section D: Extremes (5 coldest and 5 warmest cells per year/level)
- Section E: Column A & B snapshot cross-check against 1993-2020 window file
- Section F: Pit-candidate cells analysis (static 8-neighbor geometry & snapshot MAD distance)
- Section G: Pre-freeze trajectory in window file (Columns A & B at levels 28-30)
- Section H: Diagnostic visualisations (bar chart, heatmaps, time series)
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


# =============================================================================
# Helper: File Integrity Check
# =============================================================================

def compute_file_metadata(filepath: str) -> Dict[str, Any]:
    """Compute SHA256, size (bytes), and mtime for a file without modifying it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    stat = os.stat(filepath)
    mtime_dt = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)

    return {
        "filepath": os.path.abspath(filepath),
        "filename": os.path.basename(filepath),
        "sha256": sha256.hexdigest().lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": mtime_dt.isoformat(),
    }


def verify_integrity_unchanged(
    before_meta: Dict[str, Dict[str, Any]],
    after_meta: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Verify that all files have identical SHA256, size, and mtime before and after."""
    mismatches = []
    for path, b_info in before_meta.items():
        if path not in after_meta:
            mismatches.append(f"Missing in after_meta: {path}")
            continue
        a_info = after_meta[path]
        if b_info["sha256"] != a_info["sha256"]:
            mismatches.append(f"SHA256 changed for {path}: {b_info['sha256']} -> {a_info['sha256']}")
        if b_info["size_bytes"] != a_info["size_bytes"]:
            mismatches.append(f"Size changed for {path}: {b_info['size_bytes']} -> {a_info['size_bytes']}")
        if b_info["mtime_utc"] != a_info["mtime_utc"]:
            mismatches.append(f"mtime changed for {path}: {b_info['mtime_utc']} -> {a_info['mtime_utc']}")

    return (len(mismatches) == 0, mismatches)


# =============================================================================
# Helper: Decoding
# =============================================================================

def decode_thetao_array(
    raw: np.ndarray,
    fill_value: int,
    scale_factor: float,
    add_offset: float,
) -> np.ma.MaskedArray:
    """Mask _FillValue and decode int16 raw values to float64 Celsius."""
    masked_raw = np.ma.masked_equal(raw, fill_value)
    decoded = masked_raw.astype(np.float64) * scale_factor + add_offset
    return decoded


# =============================================================================
# Section A: Inventory & Coordinates Check
# =============================================================================

def check_snapshot_inventory(
    snapshot_dir: str,
    jan2019_file: str,
    years: List[int] = list(range(1993, 2021)),
    coord_tol_latlon: float = 1e-4,
    coord_tol_depth: float = 1e-3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Confirm 28 snapshots exist, check shapes, encodings, and coordinates against Jan 2019."""
    if not os.path.exists(jan2019_file):
        raise FileNotFoundError(f"Reference file not found: {jan2019_file}")

    # Read reference header and coordinates
    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        ref_lats = np.array(f_ref.variables["latitude"][:], dtype=np.float64)
        ref_lons = np.array(f_ref.variables["longitude"][:], dtype=np.float64)
        ref_depths = np.array(f_ref.variables["depth"][:], dtype=np.float64)
        ref_var = f_ref.variables["thetao"]
        ref_attrs = {
            "scale_factor": float(ref_var.attrs["scale_factor"]),
            "add_offset": float(ref_var.attrs["add_offset"]),
            "_FillValue": int(ref_var.attrs["_FillValue"]),
            "valid_min": int(ref_var.attrs["valid_min"]),
            "valid_max": int(ref_var.attrs["valid_max"]),
        }
        ref_nz = len(ref_depths)
        ref_ny = len(ref_lats)
        ref_nx = len(ref_lons)

    records = []
    base_epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)

    for y in years:
        fname = f"glorys_thetao_{y}-07-01.nc"
        fpath = os.path.join(snapshot_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required snapshot file missing: {fpath}")

        meta = compute_file_metadata(fpath)

        with h5netcdf.File(fpath, mode="r") as f_snap:
            # Check thetao variable
            if "thetao" not in f_snap.variables:
                raise KeyError(f"'thetao' variable missing in {fpath}")
            snap_var = f_snap.variables["thetao"]
            snap_shape = tuple(snap_var.shape)
            expected_shape = (1, ref_nz, ref_ny, ref_nx)
            if snap_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in {fname}: got {snap_shape}, expected {expected_shape}"
                )

            # Check encoding attributes
            snap_attrs = {
                "scale_factor": float(snap_var.attrs["scale_factor"]),
                "add_offset": float(snap_var.attrs["add_offset"]),
                "_FillValue": int(snap_var.attrs["_FillValue"]),
                "valid_min": int(snap_var.attrs["valid_min"]),
                "valid_max": int(snap_var.attrs["valid_max"]),
            }
            encoding_match = (
                abs(snap_attrs["scale_factor"] - ref_attrs["scale_factor"]) < 1e-12
                and abs(snap_attrs["add_offset"] - ref_attrs["add_offset"]) < 1e-6
                and snap_attrs["_FillValue"] == ref_attrs["_FillValue"]
                and snap_attrs["valid_min"] == ref_attrs["valid_min"]
                and snap_attrs["valid_max"] == ref_attrs["valid_max"]
            )
            if not encoding_match:
                raise ValueError(
                    f"Encoding attributes mismatch in {fname}: {snap_attrs} vs {ref_attrs}"
                )

            # Check coordinates
            snap_lats = np.array(f_snap.variables["latitude"][:], dtype=np.float64)
            snap_lons = np.array(f_snap.variables["longitude"][:], dtype=np.float64)
            snap_depths = np.array(f_snap.variables["depth"][:], dtype=np.float64)

            max_diff_lat = float(np.max(np.abs(snap_lats - ref_lats)))
            max_diff_lon = float(np.max(np.abs(snap_lons - ref_lons)))
            max_diff_depth = float(np.max(np.abs(snap_depths - ref_depths)))

            if max_diff_lat > coord_tol_latlon:
                raise ValueError(
                    f"Latitude coordinate difference {max_diff_lat} in {fname} exceeds tolerance {coord_tol_latlon}"
                )
            if max_diff_lon > coord_tol_latlon:
                raise ValueError(
                    f"Longitude coordinate difference {max_diff_lon} in {fname} exceeds tolerance {coord_tol_latlon}"
                )
            if max_diff_depth > coord_tol_depth:
                raise ValueError(
                    f"Depth coordinate difference {max_diff_depth} in {fname} exceeds tolerance {coord_tol_depth}"
                )

            # Parse timestamp
            raw_time = float(f_snap.variables["time"][0])
            snap_dt = base_epoch + datetime.timedelta(hours=raw_time)
            snap_dt_str = snap_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            records.append({
                "year": y,
                "filename": fname,
                "filepath": os.path.abspath(fpath),
                "sha256": meta["sha256"],
                "size_bytes": meta["size_bytes"],
                "mtime_utc": meta["mtime_utc"],
                "raw_time_hours": raw_time,
                "timestamp_utc": snap_dt_str,
                "shape": str(snap_shape),
                "scale_factor": snap_attrs["scale_factor"],
                "add_offset": snap_attrs["add_offset"],
                "fill_value": snap_attrs["_FillValue"],
                "valid_min": snap_attrs["valid_min"],
                "valid_max": snap_attrs["valid_max"],
                "max_diff_lat": max_diff_lat,
                "max_diff_lon": max_diff_lon,
                "max_diff_depth": max_diff_depth,
                "encoding_match": encoding_match,
                "coords_match": True,
            })

    df = pd.DataFrame(records)
    summary = {
        "total_files": len(df),
        "years_covered": f"{years[0]}-{years[-1]}",
        "all_shapes_match": bool(np.all(df["shape"] == str((1, ref_nz, ref_ny, ref_nx)))),
        "all_encodings_match": bool(np.all(df["encoding_match"])),
        "all_coords_match": bool(np.all(df["coords_match"])),
        "max_diff_lat_overall": float(df["max_diff_lat"].max()),
        "max_diff_lon_overall": float(df["max_diff_lon"].max()),
        "max_diff_depth_overall": float(df["max_diff_depth"].max()),
    }
    return df, summary


# =============================================================================
# Section B: Mask Consistency
# =============================================================================

def compare_masks(
    snap_raw: np.ndarray,
    ref_day0_raw: np.ndarray,
    static_mask_3d: np.ndarray,
    fill_value: int = -32767,
) -> Dict[str, Any]:
    """Compare snapshot finite mask against Jan 2019 day-0 finite mask and static 3D mask.
    
    Shapes must be (nz, ny, nx) where nz <= 36.
    """
    snap_finite = (snap_raw != fill_value)
    jan_finite = (ref_day0_raw != fill_value)
    stat_wet = (static_mask_3d == 1)

    # vs Jan 2019 day 0
    diff_jan = (snap_finite != jan_finite)
    diff_jan_count = int(np.sum(diff_jan))
    jan_m0_finite = int(np.sum((~jan_finite) & snap_finite))
    jan_m1_fill = int(np.sum(jan_finite & (~snap_finite)))

    # vs Static 3D mask
    diff_stat = (snap_finite != stat_wet)
    diff_stat_count = int(np.sum(diff_stat))
    stat_m0_finite = int(np.sum((~stat_wet) & snap_finite))
    stat_m1_fill = int(np.sum(stat_wet & (~snap_finite)))

    return {
        "diff_vs_jan2019": diff_jan_count,
        "jan2019_m0_snap_finite": jan_m0_finite,
        "jan2019_m1_snap_fill": jan_m1_fill,
        "diff_vs_static": diff_stat_count,
        "static_m0_snap_finite": stat_m0_finite,
        "static_m1_snap_fill": stat_m1_fill,
    }


def find_mask_differing_cells(
    snap_raw: np.ndarray,
    ref_day0_raw: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    year: int,
    fill_value: int = -32767,
    max_cells: int = 50,
) -> List[Dict[str, Any]]:
    """List up to max_cells differing cells between snapshot finite mask and Jan 2019 mask."""
    snap_finite = (snap_raw != fill_value)
    jan_finite = (ref_day0_raw != fill_value)
    diff_indices = np.where(snap_finite != jan_finite)

    cells = []
    n_diff = len(diff_indices[0])
    for i in range(min(n_diff, max_cells)):
        k, r, c = int(diff_indices[0][i]), int(diff_indices[1][i]), int(diff_indices[2][i])
        cells.append({
            "year": year,
            "level": k,
            "depth_m": float(depths[k]),
            "lat_idx": r,
            "lon_idx": c,
            "latitude": float(lats[r]),
            "longitude": float(lons[c]),
            "snap_raw": int(snap_raw[k, r, c]),
            "snap_finite": bool(snap_finite[k, r, c]),
            "jan_raw": int(ref_day0_raw[k, r, c]),
            "jan_finite": bool(jan_finite[k, r, c]),
        })
    return cells


# =============================================================================
# Section C: Per-Snapshot, Per-Level Census
# =============================================================================

def compute_level_census(
    raw_level: np.ndarray,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
    valid_min: int = -32766,
    valid_max: int = 21306,
) -> Dict[str, int]:
    """Compute counts of finite cells and forensic categories for a 2D level slice."""
    finite = (raw_level != fill_value)
    n_finite = int(np.sum(finite))
    if n_finite == 0:
        return {
            "n_total": raw_level.size,
            "n_finite": 0,
            "n_raw_neg_28671": 0,
            "n_abs_lt_1e-3": 0,
            "n_subzero": 0,
            "n_near_floor": 0,
            "n_near_ceiling": 0,
        }

    raw_finite = raw_level[finite]
    n_neg_28671 = int(np.sum(raw_finite == -28671))

    # Near encoding boundaries
    n_near_floor = int(np.sum((raw_finite >= valid_min) & (raw_finite <= valid_min + 50)))
    n_near_ceiling = int(np.sum((raw_finite <= valid_max) & (raw_finite >= valid_max - 50)))

    # Decoded values (only for finite cells)
    dec_finite = raw_finite.astype(np.float64) * scale_factor + add_offset
    n_abs_lt_1em3 = int(np.sum(np.abs(dec_finite) < 1e-3))
    n_subzero = int(np.sum(dec_finite < 0.0))

    return {
        "n_total": raw_level.size,
        "n_finite": n_finite,
        "n_raw_neg_28671": n_neg_28671,
        "n_abs_lt_1e-3": n_abs_lt_1em3,
        "n_subzero": n_subzero,
        "n_near_floor": n_near_floor,
        "n_near_ceiling": n_near_ceiling,
    }


def find_flagged_cells_level(
    raw_level: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depth_m: float,
    level: int,
    year: int,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
    valid_min: int = -32766,
) -> List[Dict[str, Any]]:
    """List all cells in a level where raw == -28671 OR raw <= valid_min + 50 OR decoded < 0."""
    finite = (raw_level != fill_value)
    if not np.any(finite):
        return []

    is_neg_28671 = (raw_level == -28671)
    is_near_floor = (raw_level >= valid_min) & (raw_level <= valid_min + 50) & finite

    # For decoded < 0: compute safely on finite cells
    is_subzero = np.zeros_like(finite, dtype=bool)
    finite_indices = np.where(finite)
    dec_finite = raw_level[finite].astype(np.float64) * scale_factor + add_offset
    is_subzero[finite_indices] = (dec_finite < 0.0)

    flagged_mask = is_neg_28671 | is_near_floor | is_subzero
    if not np.any(flagged_mask):
        return []

    r_idx, c_idx = np.where(flagged_mask)
    flagged = []
    for r, c in zip(r_idx, c_idx):
        raw_val = int(raw_level[r, c])
        dec_val = float(raw_val * scale_factor + add_offset)
        reasons = []
        if raw_val == -28671:
            reasons.append("raw_neg_28671")
        if valid_min <= raw_val <= valid_min + 50:
            reasons.append("near_floor")
        if dec_val < 0.0:
            reasons.append("subzero")

        flagged.append({
            "year": year,
            "level": level,
            "depth_m": depth_m,
            "lat_idx": int(r),
            "lon_idx": int(c),
            "latitude": float(lats[r]),
            "longitude": float(lons[c]),
            "raw": raw_val,
            "decoded": dec_val,
            "reasons": ";".join(reasons),
        })
    return flagged


# =============================================================================
# Section D: Extremes
# =============================================================================

def extract_level_extremes(
    raw_level: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    level: int,
    depth_m: float,
    year: int,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
    lat_a_idx: int = 120,
    lon_a_idx: int = 67,
    lat_b_idx: int = 120,
    lon_b_idx: int = 68,
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find the k coldest and k warmest finite cells for a single 2D level slice."""
    finite = (raw_level != fill_value)
    if not np.any(finite):
        return [], []

    r_idx, c_idx = np.where(finite)
    raw_finite = raw_level[finite]
    n_pts = len(raw_finite)
    actual_k = min(k, n_pts)

    # raw sort is monotonically equivalent to decoded sort because scale_factor > 0
    sort_order = np.argsort(raw_finite)

    cold_idx = sort_order[:actual_k]
    warm_idx = sort_order[::-1][:actual_k]

    cold_records = []
    for rank_idx, idx in enumerate(cold_idx, start=1):
        r = int(r_idx[idx])
        c = int(c_idx[idx])
        raw_val = int(raw_finite[idx])
        dec_val = float(raw_val * scale_factor + add_offset)
        cold_records.append({
            "year": year,
            "level": level,
            "depth_m": depth_m,
            "extreme_type": "coldest",
            "rank": rank_idx,
            "lat_idx": r,
            "lon_idx": c,
            "latitude": float(lats[r]),
            "longitude": float(lons[c]),
            "raw": raw_val,
            "decoded": dec_val,
            "is_A": bool(r == lat_a_idx and c == lon_a_idx),
            "is_B": bool(r == lat_b_idx and c == lon_b_idx),
        })

    warm_records = []
    for rank_idx, idx in enumerate(warm_idx, start=1):
        r = int(r_idx[idx])
        c = int(c_idx[idx])
        raw_val = int(raw_finite[idx])
        dec_val = float(raw_val * scale_factor + add_offset)
        warm_records.append({
            "year": year,
            "level": level,
            "depth_m": depth_m,
            "extreme_type": "warmest",
            "rank": rank_idx,
            "lat_idx": r,
            "lon_idx": c,
            "latitude": float(lats[r]),
            "longitude": float(lons[c]),
            "raw": raw_val,
            "decoded": dec_val,
            "is_A": bool(r == lat_a_idx and c == lon_a_idx),
            "is_B": bool(r == lat_b_idx and c == lon_b_idx),
        })

    return cold_records, warm_records


# =============================================================================
# Section E: Columns A & B at Snapshot Dates
# =============================================================================

def audit_ab_snapshots(
    snapshot_dir: str,
    window_file: str,
    depths: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    years: List[int] = list(range(1993, 2021)),
    levels: List[int] = list(range(26, 32)),  # 26 to 31
    lat_target: float = 15.0,
    lon_a_target: float = 50.583332,
    lon_b_target: float = 50.666668,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> Tuple[pd.DataFrame, int, int, List[Dict[str, Any]]]:
    """Audit Columns A & B at levels 26-31 for all snapshot dates, cross-checking against window file."""
    # Lookup indices in full domain
    lat_idx_full = int(np.argmin(np.abs(lats - lat_target)))
    lon_a_idx_full = int(np.argmin(np.abs(lons - lon_a_target)))
    lon_b_idx_full = int(np.argmin(np.abs(lons - lon_b_target)))

    # Read window file coordinates and index mapping
    with h5netcdf.File(window_file, mode="r") as fw:
        w_lats = np.array(fw.variables["latitude"][:], dtype=np.float64)
        w_lons = np.array(fw.variables["longitude"][:], dtype=np.float64)
        w_times = np.array(fw.variables["time"][:], dtype=np.float64)
        w_thetao = fw.variables["thetao"]

        w_lat_idx = int(np.argmin(np.abs(w_lats - lat_target)))
        w_lon_a_idx = int(np.argmin(np.abs(w_lons - lon_a_target)))
        w_lon_b_idx = int(np.argmin(np.abs(w_lons - lon_b_target)))

        base_epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
        records = []
        cells_compared = 0
        cells_mismatched = 0
        mismatches = []

        for y in years:
            fname = f"glorys_thetao_{y}-07-01.nc"
            fpath = os.path.join(snapshot_dir, fname)
            date_str = f"{y}-07-01"

            with h5netcdf.File(fpath, mode="r") as fs:
                snap_time = float(fs.variables["time"][0])
                snap_thetao = fs.variables["thetao"][0]  # shape (36, ny, nx)

                # Locate time index in window file
                t_matches = np.where(np.abs(w_times - snap_time) < 1e-4)[0]
                if len(t_matches) == 0:
                    raise ValueError(f"Snapshot timestamp {snap_time} ({date_str}) not found in window file")
                w_t_idx = int(t_matches[0])

                # Read window slice for this day
                win_slice = w_thetao[w_t_idx]  # shape (36, 13, 15)

                for col_name, c_full, c_win in [("A", lon_a_idx_full, w_lon_a_idx), ("B", lon_b_idx_full, w_lon_b_idx)]:
                    for lvl in levels:
                        cells_compared += 1
                        snap_raw = int(snap_thetao[lvl, lat_idx_full, c_full])
                        win_raw = int(win_slice[lvl, w_lat_idx, c_win])

                        snap_dec = (
                            float(snap_raw * scale_factor + add_offset)
                            if snap_raw != fill_value else np.nan
                        )
                        win_dec = (
                            float(win_raw * scale_factor + add_offset)
                            if win_raw != fill_value else np.nan
                        )

                        is_match = (snap_raw == win_raw)
                        if not is_match:
                            cells_mismatched += 1
                            if len(mismatches) < 20:
                                mismatches.append({
                                    "year": y,
                                    "date": date_str,
                                    "column": col_name,
                                    "level": lvl,
                                    "depth_m": float(depths[lvl]),
                                    "snap_raw": snap_raw,
                                    "win_raw": win_raw,
                                })

                        records.append({
                            "year": y,
                            "date": date_str,
                            "column": col_name,
                            "level": lvl,
                            "depth_m": float(depths[lvl]),
                            "lat": float(lats[lat_idx_full]),
                            "lon": float(lons[c_full]),
                            "snap_raw": snap_raw,
                            "snap_decoded": snap_dec,
                            "window_raw": win_raw,
                            "window_decoded": win_dec,
                            "match": is_match,
                        })

    df = pd.DataFrame(records)
    return df, cells_compared, cells_mismatched, mismatches


# =============================================================================
# Section F: Pit-Candidate Cells (Static Geometry & Snapshot Anomaly)
# =============================================================================

def compute_wet_8neighbors(mask_2d: np.ndarray) -> np.ndarray:
    """Count wet 8-neighbors for each cell in a 2D mask (1=wet, 0=land).
    
    Cells outside the domain boundaries strictly count as not wet (0).
    Uses 2D zero-padding.
    """
    m = (mask_2d == 1).astype(np.int32)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    n = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2]             + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    return n


def identify_pit_candidates_static(
    static_mask_3d: np.ndarray,
    depths: np.ndarray,
    lat_a_idx: int = 120,
    lon_a_idx: int = 67,
    lat_b_idx: int = 120,
    lon_b_idx: int = 68,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Identify wet cells with < 3 wet neighbors for levels 0-35 of static 3D mask.
    
    Returns summary DataFrame per level and boolean 3D mask of pit candidates.
    """
    nz = min(36, static_mask_3d.shape[0])
    ny, nx = static_mask_3d.shape[1], static_mask_3d.shape[2]
    pit_mask_3d = np.zeros((nz, ny, nx), dtype=bool)

    records = []
    for lvl in range(nz):
        m = static_mask_3d[lvl]
        n_neighbors = compute_wet_8neighbors(m)
        is_wet = (m == 1)
        is_pit = is_wet & (n_neighbors < 3)
        pit_mask_3d[lvl] = is_pit

        n_wet = int(np.sum(is_wet))
        n_pit = int(np.sum(is_pit))

        records.append({
            "level": lvl,
            "depth_m": float(depths[lvl]),
            "n_wet_cells": n_wet,
            "n_pit_candidates": n_pit,
            "A_is_wet": bool(is_wet[lat_a_idx, lon_a_idx]),
            "A_neighbors": int(n_neighbors[lat_a_idx, lon_a_idx]),
            "A_is_pit": bool(is_pit[lat_a_idx, lon_a_idx]),
            "B_is_wet": bool(is_wet[lat_b_idx, lon_b_idx]),
            "B_neighbors": int(n_neighbors[lat_b_idx, lon_b_idx]),
            "B_is_pit": bool(is_pit[lat_b_idx, lon_b_idx]),
        })

    df = pd.DataFrame(records)
    return df, pit_mask_3d


def analyze_pit_candidates_snapshot(
    raw_3d: np.ndarray,
    pit_mask_3d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    year: int,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
    valid_min: int = -32766,
    lat_a_idx: int = 120,
    lon_a_idx: int = 67,
    lat_b_idx: int = 120,
    lon_b_idx: int = 68,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Analyze pit-candidate cells in a snapshot: count flagged codes and find top 10 farthest in MAD."""
    nz = min(36, raw_3d.shape[0])
    counts = {
        "year": year,
        "n_pit_raw_neg_28671": 0,
        "n_pit_subzero": 0,
        "n_pit_near_floor": 0,
        "n_pit_flagged_any": 0,
    }

    all_pit_cells = []

    for lvl in range(nz):
        raw_lvl = raw_3d[lvl]
        finite_lvl = (raw_lvl != fill_value)
        pit_lvl = pit_mask_3d[lvl] & finite_lvl

        # Compute level domain median and MAD over all finite cells of this level
        if np.any(finite_lvl):
            dec_finite = raw_lvl[finite_lvl].astype(np.float64) * scale_factor + add_offset
            lvl_median = float(np.median(dec_finite))
            lvl_mad = float(np.median(np.abs(dec_finite - lvl_median)))
        else:
            lvl_median = np.nan
            lvl_mad = np.nan

        if not np.any(pit_lvl):
            continue

        r_pts, c_pts = np.where(pit_lvl)
        for r, c in zip(r_pts, c_pts):
            raw_val = int(raw_lvl[r, c])
            dec_val = float(raw_val * scale_factor + add_offset)

            is_neg_28671 = (raw_val == -28671)
            is_subzero = (dec_val < 0.0)
            is_near_floor = (valid_min <= raw_val <= valid_min + 50)

            if is_neg_28671:
                counts["n_pit_raw_neg_28671"] += 1
            if is_subzero:
                counts["n_pit_subzero"] += 1
            if is_near_floor:
                counts["n_pit_near_floor"] += 1
            if is_neg_28671 or is_subzero or is_near_floor:
                counts["n_pit_flagged_any"] += 1

            if not np.isnan(lvl_mad) and lvl_mad > 0:
                mad_dist = float(abs(dec_val - lvl_median) / lvl_mad)
            else:
                mad_dist = 0.0 if dec_val == lvl_median else 999999.0

            all_pit_cells.append({
                "year": year,
                "level": lvl,
                "depth_m": float(depths[lvl]),
                "lat_idx": int(r),
                "lon_idx": int(c),
                "latitude": float(lats[r]),
                "longitude": float(lons[c]),
                "raw": raw_val,
                "decoded": dec_val,
                "level_median": lvl_median,
                "level_mad": lvl_mad,
                "mad_distance": mad_dist,
                "is_A": bool(r == lat_a_idx and c == lon_a_idx),
                "is_B": bool(r == lat_b_idx and c == lon_b_idx),
            })

    # Sort pit cells by mad_distance descending and pick top 10
    all_pit_cells.sort(key=lambda x: x["mad_distance"], reverse=True)
    top_10 = all_pit_cells[:10]
    for rank_idx, cell in enumerate(top_10, start=1):
        cell["rank"] = rank_idx

    return counts, top_10


# =============================================================================
# Section G: Pre-Freeze Trajectory (Window File)
# =============================================================================

def find_final_continuous_run_start(arr: np.ndarray, target: int = -28671) -> int:
    """Find index of the first day of the final continuous run of target ending at arr[-1].
    
    If the series does not end with target, returns -1.
    Correctly ignores earlier runs if there is an interior non-target gap.
    """
    if len(arr) == 0 or arr[-1] != target:
        return -1
    idx = len(arr) - 1
    while idx >= 0 and arr[idx] == target:
        idx -= 1
    return idx + 1


def audit_prefreeze_trajectory(
    window_file: str,
    lat_target: float = 15.0,
    lon_a_target: float = 50.583332,
    lon_b_target: float = 50.666668,
    fill_value: int = -32767,
    valid_min: int = -32766,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Audit pre-freeze trajectory and level-28 tails for Columns A & B in the window file."""
    with h5netcdf.File(window_file, mode="r") as fw:
        w_lats = np.array(fw.variables["latitude"][:], dtype=np.float64)
        w_lons = np.array(fw.variables["longitude"][:], dtype=np.float64)
        w_depths = np.array(fw.variables["depth"][:], dtype=np.float64)
        w_times = np.array(fw.variables["time"][:], dtype=np.float64)

        lat_idx = int(np.argmin(np.abs(w_lats - lat_target)))
        lon_a_idx = int(np.argmin(np.abs(w_lons - lon_a_target)))
        lon_b_idx = int(np.argmin(np.abs(w_lons - lon_b_target)))

        base_epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
        dates = [base_epoch + datetime.timedelta(hours=float(t)) for t in w_times]

        valid_min_dec = valid_min * scale_factor + add_offset

        summary_findings = {}
        trajectory_records = []

        # Analyze levels 29 and 30 for A and B
        target_freeze_code = -28671
        for col_name, c_idx in [("A", lon_a_idx), ("B", lon_b_idx)]:
            for lvl in [29, 30]:
                series = np.array(fw.variables["thetao"][:, lvl, lat_idx, c_idx])
                start_idx = find_final_continuous_run_start(series, target_freeze_code)
                if start_idx == -1:
                    freeze_date_str = None
                    freeze_run_len = 0
                else:
                    freeze_date_str = dates[start_idx].strftime("%Y-%m-%d")
                    freeze_run_len = len(series) - start_idx

                # Min decoded over record
                finite = (series != fill_value)
                min_raw = int(np.min(series[finite]))
                min_idx = int(np.argmin(np.where(finite, series, 999999)))
                min_dec = float(min_raw * scale_factor + add_offset)
                min_date_str = dates[min_idx].strftime("%Y-%m-%d")
                diff_from_valid_min = float(min_dec - valid_min_dec)

                summary_findings[f"{col_name}_lvl{lvl}"] = {
                    "column": col_name,
                    "level": lvl,
                    "depth_m": float(w_depths[lvl]),
                    "freeze_start_date": freeze_date_str,
                    "freeze_start_index": start_idx,
                    "freeze_run_length_days": freeze_run_len,
                    "record_min_raw": min_raw,
                    "record_min_decoded": min_dec,
                    "record_min_date": min_date_str,
                    "diff_from_valid_min_C": diff_from_valid_min,
                }

                # 40 days before and 5 days after
                if start_idx != -1:
                    window_start = max(0, start_idx - 40)
                    window_end = min(len(series) - 1, start_idx + 5)
                    for idx in range(window_start, window_end + 1):
                        rel_day = idx - start_idx
                        raw_val = int(series[idx])
                        dec_val = float(raw_val * scale_factor + add_offset) if raw_val != fill_value else np.nan
                        trajectory_records.append({
                            "column": col_name,
                            "level": lvl,
                            "depth_m": float(w_depths[lvl]),
                            "relative_day": rel_day,
                            "day_index": idx,
                            "date": dates[idx].strftime("%Y-%m-%d"),
                            "raw": raw_val,
                            "decoded": dec_val,
                            "is_freeze_start": bool(idx == start_idx),
                        })

        df_prefreeze = pd.DataFrame(trajectory_records)

        # Analyze level 28 for A and B
        tail_records = []
        n_days = len(dates)
        for col_name, c_idx in [("A", lon_a_idx), ("B", lon_b_idx)]:
            lvl = 28
            series = np.array(fw.variables["thetao"][:, lvl, lat_idx, c_idx])
            finite = (series != fill_value)
            min_raw = int(np.min(series[finite]))
            min_idx = int(np.argmin(np.where(finite, series, 999999)))
            min_dec = float(min_raw * scale_factor + add_offset)
            min_date_str = dates[min_idx].strftime("%Y-%m-%d")

            near_floor_days = int(np.sum((series >= valid_min) & (series <= valid_min + 50)))

            summary_findings[f"{col_name}_lvl28"] = {
                "column": col_name,
                "level": lvl,
                "depth_m": float(w_depths[lvl]),
                "record_min_raw": min_raw,
                "record_min_decoded": min_dec,
                "record_min_date": min_date_str,
                "days_within_50_codes_of_valid_min": near_floor_days,
            }

            # Last 40 days of record
            for idx in range(n_days - 40, n_days):
                rel_day = idx - (n_days - 1)
                raw_val = int(series[idx])
                dec_val = float(raw_val * scale_factor + add_offset) if raw_val != fill_value else np.nan
                tail_records.append({
                    "column": col_name,
                    "level": lvl,
                    "depth_m": float(w_depths[lvl]),
                    "relative_day_to_end": rel_day,
                    "day_index": idx,
                    "date": dates[idx].strftime("%Y-%m-%d"),
                    "raw": raw_val,
                    "decoded": dec_val,
                })

        df_level28 = pd.DataFrame(tail_records)

    return df_prefreeze, df_level28, summary_findings


# =============================================================================
# Section H: Visualizations
# =============================================================================

def generate_plots(
    census_df: pd.DataFrame,
    window_file: str,
    prefreeze_findings: Dict[str, Any],
    output_dir: str,
    lat_target: float = 15.0,
    lon_a_target: float = 50.583332,
    lon_b_target: float = 50.666668,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> Dict[str, str]:
    """Generate the four required diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # 1. Per-year bar chart of cells with raw == -28671 (all levels)
    # Sum over all levels for each year
    raw_28671_by_year = census_df.groupby("year")["n_raw_neg_28671"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(raw_28671_by_year["year"], raw_28671_by_year["n_raw_neg_28671"], color="#1f77b4", edgecolor="#0e4b75")
    ax.set_title("GLORYS July-01 Snapshots: Annual Count of Cells with Raw Code -28671 (All 36 Levels)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year (July 01 Snapshot)", fontsize=10)
    ax.set_ylabel("Count of Cells (raw == -28671)", fontsize=10)
    ax.set_xticks(raw_28671_by_year["year"])
    ax.set_xticklabels(raw_28671_by_year["year"], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "plot1_raw_28671_by_year.png")
    fig.savefig(p1)
    plt.close(fig)
    plot_paths["plot1_raw_28671_by_year"] = os.path.abspath(p1)

    # 2. Heatmap (level x year) of the coldest decoded value
    pivot_coldest = census_df.pivot(index="level", columns="year", values="min_decoded_C")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    im = ax.imshow(pivot_coldest.values, aspect="auto", cmap="coolwarm", origin="upper", interpolation="none")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Coldest Decoded Temperature (°C)", fontsize=10)
    ax.set_title("GLORYS Snapshots: Coldest Decoded Temperature (°C) by Level (0-35) and Year (1993-2020)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Depth Level Index (0 = Surface, 35 = ~1062m)", fontsize=10)
    years_list = sorted(census_df["year"].unique())
    ax.set_xticks(range(len(years_list)))
    ax.set_xticklabels(years_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, 36, 2))
    ax.set_yticklabels(range(0, 36, 2), fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(output_dir, "plot2_coldest_decoded_heatmap.png")
    fig.savefig(p2)
    plt.close(fig)
    plot_paths["plot2_coldest_decoded_heatmap"] = os.path.abspath(p2)

    # 3. Heatmap (level x year) of the number of cells with decoded < 0 C
    pivot_subzero = census_df.pivot(index="level", columns="year", values="n_subzero")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    im = ax.imshow(pivot_subzero.values, aspect="auto", cmap="viridis", origin="upper", interpolation="none")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Cells with Decoded T < 0 °C", fontsize=10)
    ax.set_title("GLORYS Snapshots: Count of Sub-Zero Decoded Cells (< 0 °C) by Level and Year", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Depth Level Index", fontsize=10)
    ax.set_xticks(range(len(years_list)))
    ax.set_xticklabels(years_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, 36, 2))
    ax.set_yticklabels(range(0, 36, 2), fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(output_dir, "plot3_subzero_count_heatmap.png")
    fig.savefig(p3)
    plt.close(fig)
    plot_paths["plot3_subzero_count_heatmap"] = os.path.abspath(p3)

    # 4. A and B decoded temperature at levels 28-30 over 1993-2020 from window file
    with h5netcdf.File(window_file, mode="r") as fw:
        w_lats = np.array(fw.variables["latitude"][:], dtype=np.float64)
        w_lons = np.array(fw.variables["longitude"][:], dtype=np.float64)
        w_times = np.array(fw.variables["time"][:], dtype=np.float64)

        lat_idx = int(np.argmin(np.abs(w_lats - lat_target)))
        lon_a_idx = int(np.argmin(np.abs(w_lons - lon_a_target)))
        lon_b_idx = int(np.argmin(np.abs(w_lons - lon_b_target)))

        base_epoch = datetime.datetime(1950, 1, 1, tzinfo=datetime.timezone.utc)
        d_times = [base_epoch + datetime.timedelta(hours=float(t)) for t in w_times]

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, dpi=150)
        level_map = {0: 28, 1: 29, 2: 30}
        colors = {"A": "#1f77b4", "B": "#d62728"}

        for ax_idx, lvl in level_map.items():
            ax = axes[ax_idx]
            raw_a = fw.variables["thetao"][:, lvl, lat_idx, lon_a_idx]
            raw_b = fw.variables["thetao"][:, lvl, lat_idx, lon_b_idx]

            dec_a = np.where(raw_a != fill_value, raw_a * scale_factor + add_offset, np.nan)
            dec_b = np.where(raw_b != fill_value, raw_b * scale_factor + add_offset, np.nan)

            ax.plot(d_times, dec_a, label="Column A (15.000N, 50.583E)", color=colors["A"], lw=1.0, alpha=0.85)
            ax.plot(d_times, dec_b, label="Column B (15.000N, 50.667E)", color=colors["B"], lw=1.0, alpha=0.85)

            # Mark freeze start if applicable
            for col_name, c_color in [("A", colors["A"]), ("B", colors["B"])]:
                key = f"{col_name}_lvl{lvl}"
                if key in prefreeze_findings and prefreeze_findings[key].get("freeze_start_date"):
                    f_date_str = prefreeze_findings[key]["freeze_start_date"]
                    f_date = datetime.datetime.strptime(f_date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                    ax.axvline(f_date, color=c_color, linestyle="--", lw=1.2, label=f"{col_name} freeze start ({f_date_str})")

            ax.set_ylabel(f"Level {lvl} T (°C)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)

        axes[0].set_title("GLORYS 1993-2020 Window: Columns A & B Daily Temperature (Levels 28, 29, 30)", fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("Date", fontsize=10)
        fig.tight_layout()
        p4 = os.path.join(output_dir, "plot4_ab_levels28_30_timeseries.png")
        fig.savefig(p4)
        plt.close(fig)
        plot_paths["plot4_ab_levels28_30_timeseries"] = os.path.abspath(p4)

    return plot_paths


# =============================================================================
# Main Orchestration
# =============================================================================

def run_snapshot_census(
    snapshot_dir: str = "data/raw/glorys_snapshots",
    jan2019_file: str = "data/raw/glorys/glorys_thetao_2019-01.nc",
    window_file: str = "data/raw/glorys_AB_window/glorys_thetao_AB_window_1993-2020.nc",
    static_file: str = "data/raw/glorys_static/glorys_static_mask_bathy.nc",
    output_dir: str = "outputs/diagnostics/glorys_snapshot_census",
    years: List[int] = list(range(1993, 2021)),
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Execute complete GLORYS snapshot census and pre-freeze forensic analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GLORYS SNAPSHOT CENSUS 1993-2020 & PRE-FREEZE TRAJECTORY")
    print("=" * 80)

    # ---------------------------------------------------------
    # Initial File Metadata (BEFORE)
    # ---------------------------------------------------------
    input_files = [jan2019_file, window_file, static_file]
    for y in years:
        input_files.append(os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc"))

    print(f"\nComputing SHA256 and metadata for {len(input_files)} input files (BEFORE)...")
    meta_before = {}
    for fpath in input_files:
        meta_before[os.path.abspath(fpath)] = compute_file_metadata(fpath)

    print(f"  Jan 2019 SHA256 : {meta_before[os.path.abspath(jan2019_file)]['sha256']}")
    print(f"  Window SHA256   : {meta_before[os.path.abspath(window_file)]['sha256']}")
    print(f"  Static SHA256   : {meta_before[os.path.abspath(static_file)]['sha256']}")
    print(f"  Snapshots (28)  : {meta_before[os.path.abspath(input_files[3])]['sha256']} (1993) ... {meta_before[os.path.abspath(input_files[-1])]['sha256']} (2020)")

    # ---------------------------------------------------------
    # Section A: Inventory & Coordinates Check
    # ---------------------------------------------------------
    print("\n--- Section A: Inventory and Integrity ---")
    df_inventory, summary_A = check_snapshot_inventory(snapshot_dir, jan2019_file, years)
    inventory_csv = os.path.join(output_dir, "snapshot_inventory.csv")
    df_inventory.to_csv(inventory_csv, index=False)
    print(f"Confirmed exactly {summary_A['total_files']} snapshots exist ({summary_A['years_covered']}).")
    print(f"All shapes match: {summary_A['all_shapes_match']}")
    print(f"All encodings match Jan 2019: {summary_A['all_encodings_match']}")
    print(f"Max coordinate differences: lat={summary_A['max_diff_lat_overall']:.2e} deg, lon={summary_A['max_diff_lon_overall']:.2e} deg, depth={summary_A['max_diff_depth_overall']:.2e} m")
    print(f"Saved: {inventory_csv}")

    # Read reference data for later sections
    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        jan_day0_raw = f_ref.variables["thetao"][0]  # shape (36, 301, 721)
        ref_lats = np.array(f_ref.variables["latitude"][:], dtype=np.float64)
        ref_lons = np.array(f_ref.variables["longitude"][:], dtype=np.float64)
        ref_depths = np.array(f_ref.variables["depth"][:], dtype=np.float64)
        scale_factor = float(f_ref.variables["thetao"].attrs["scale_factor"])
        add_offset = float(f_ref.variables["thetao"].attrs["add_offset"])
        fill_val = int(f_ref.variables["thetao"].attrs["_FillValue"])
        valid_min = int(f_ref.variables["thetao"].attrs["valid_min"])
        valid_max = int(f_ref.variables["thetao"].attrs["valid_max"])

    with h5netcdf.File(static_file, mode="r") as f_stat:
        static_mask_3d = np.array(f_stat.variables["mask"][:36], dtype=np.int8)

    lat_a_idx = int(np.argmin(np.abs(ref_lats - 15.0)))
    lat_b_idx = lat_a_idx
    lon_a_idx = int(np.argmin(np.abs(ref_lons - 50.583332)))
    lon_b_idx = int(np.argmin(np.abs(ref_lons - 50.666668)))

    # ---------------------------------------------------------
    # Section B: Mask Consistency
    # ---------------------------------------------------------
    print("\n--- Section B: Mask Consistency ---")
    mask_records = []
    differing_cells_list = []
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            s_raw = fs.variables["thetao"][0]
            cmp_res = compare_masks(s_raw, jan_day0_raw, static_mask_3d, fill_val)
            cmp_res["year"] = y
            cmp_res["date"] = f"{y}-07-01"
            mask_records.append(cmp_res)

            if cmp_res["diff_vs_jan2019"] > 0:
                diff_cells = find_mask_differing_cells(
                    s_raw, jan_day0_raw, ref_lats, ref_lons, ref_depths, y, fill_val, max_cells=50
                )
                differing_cells_list.extend(diff_cells)

    df_mask = pd.DataFrame(mask_records)
    mask_csv = os.path.join(output_dir, "snapshot_mask_consistency.csv")
    df_mask.to_csv(mask_csv, index=False)
    total_diff_jan = int(df_mask["diff_vs_jan2019"].sum())
    total_diff_static = int(df_mask["diff_vs_static"].sum())
    print(f"Total differences vs Jan 2019 day-0 finite mask across all 28 years: {total_diff_jan}")
    print(f"Total differences vs static 3D mask across all 28 years: {total_diff_static}")
    if total_diff_jan > 0:
        print(f"Listing first {min(len(differing_cells_list), 50)} differing cells vs Jan 2019:")
        for dc in differing_cells_list[:10]:
            print(f"  Year {dc['year']} Lvl {dc['level']} ({dc['depth_m']:.1f}m) at ({dc['latitude']:.3f}N, {dc['longitude']:.3f}E): snap_raw={dc['snap_raw']}, jan_raw={dc['jan_raw']}")
    print(f"Saved: {mask_csv}")

    # ---------------------------------------------------------
    # Section C: Per-Snapshot, Per-Level Census
    # ---------------------------------------------------------
    print("\n--- Section C: Per-Snapshot, Per-Level Census ---")
    census_records = []
    all_flagged_cells = []
    year_flagged_counts = {}

    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        year_flags = []
        with h5netcdf.File(snap_path, mode="r") as fs:
            s_raw = fs.variables["thetao"][0]
            for lvl in range(36):
                depth_m = float(ref_depths[lvl])
                raw_lvl = s_raw[lvl]
                stats = compute_level_census(
                    raw_lvl, fill_val, scale_factor, add_offset, valid_min, valid_max
                )
                finite_lvl = (raw_lvl != fill_val)
                if np.any(finite_lvl):
                    dec_finite = raw_lvl[finite_lvl].astype(np.float64) * scale_factor + add_offset
                    min_dec = float(np.min(dec_finite))
                    max_dec = float(np.max(dec_finite))
                else:
                    min_dec = np.nan
                    max_dec = np.nan

                census_records.append({
                    "year": y,
                    "level": lvl,
                    "depth_m": depth_m,
                    "n_total": stats["n_total"],
                    "n_finite": stats["n_finite"],
                    "n_raw_neg_28671": stats["n_raw_neg_28671"],
                    "n_abs_lt_1e-3": stats["n_abs_lt_1e-3"],
                    "n_subzero": stats["n_subzero"],
                    "n_near_floor": stats["n_near_floor"],
                    "n_near_ceiling": stats["n_near_ceiling"],
                    "min_decoded_C": min_dec,
                    "max_decoded_C": max_dec,
                })

                flags = find_flagged_cells_level(
                    raw_lvl, ref_lats, ref_lons, depth_m, lvl, y,
                    fill_val, scale_factor, add_offset, valid_min
                )
                year_flags.extend(flags)

        year_flagged_counts[y] = len(year_flags)
        all_flagged_cells.extend(year_flags)

    df_census = pd.DataFrame(census_records)
    census_csv = os.path.join(output_dir, "census_by_year_level.csv")
    df_census.to_csv(census_csv, index=False)

    df_flagged = pd.DataFrame(all_flagged_cells)
    flagged_csv = os.path.join(output_dir, "flagged_cells.csv")
    df_flagged.to_csv(flagged_csv, index=False)

    print(f"Total finite cells evaluated across 28 snapshots: {df_census['n_finite'].sum():,}")
    print(f"Total occurrences of raw == -28671: {df_census['n_raw_neg_28671'].sum()}")
    print(f"Total occurrences of decoded < 0 C: {df_census['n_subzero'].sum()}")
    print(f"Total occurrences of raw near floor (<= -32716): {df_census['n_near_floor'].sum()}")
    print(f"Total occurrences of raw near ceiling (>= 21256): {df_census['n_near_ceiling'].sum()}")
    print(f"Total flagged cells saved to CSV: {len(df_flagged)}")
    for y in years:
        if year_flagged_counts[y] > 500:
            print(f"  Year {y}: {year_flagged_counts[y]} flagged cells (>500 limit). Detailed rows written to CSV; printing per-level counts only.")
    print(f"Saved: {census_csv}")
    print(f"Saved: {flagged_csv}")

    # ---------------------------------------------------------
    # Section D: Extremes
    # ---------------------------------------------------------
    print("\n--- Section D: Extremes ---")
    extremes_records = []
    ab_in_extremes_count = 0
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            s_raw = fs.variables["thetao"][0]
            for lvl in range(36):
                depth_m = float(ref_depths[lvl])
                raw_lvl = s_raw[lvl]
                cold_recs, warm_recs = extract_level_extremes(
                    raw_lvl, ref_lats, ref_lons, lvl, depth_m, y,
                    fill_val, scale_factor, add_offset,
                    lat_a_idx, lon_a_idx, lat_b_idx, lon_b_idx, k=5
                )
                extremes_records.extend(cold_recs)
                extremes_records.extend(warm_recs)
                for rec in cold_recs + warm_recs:
                    if rec["is_A"] or rec["is_B"]:
                        ab_in_extremes_count += 1

    df_extremes = pd.DataFrame(extremes_records)
    extremes_csv = os.path.join(output_dir, "extremes_by_year_level.csv")
    df_extremes.to_csv(extremes_csv, index=False)
    print(f"Generated {len(df_extremes)} extreme records (5 coldest & 5 warmest for 28 years x 36 levels).")
    print(f"Total appearances of Column A or B among top-5 extremes: {ab_in_extremes_count}")
    print(f"Saved: {extremes_csv}")

    # ---------------------------------------------------------
    # Section E: Columns A & B Snapshot Values
    # ---------------------------------------------------------
    print("\n--- Section E: Columns A & B at Snapshot Dates ---")
    df_ab, cells_cmp, cells_mismatch, mismatches = audit_ab_snapshots(
        snapshot_dir, window_file, ref_depths, ref_lats, ref_lons,
        years, levels=list(range(26, 32)),
        lat_target=15.0, lon_a_target=50.583332, lon_b_target=50.666668,
        fill_value=fill_val, scale_factor=scale_factor, add_offset=add_offset
    )
    ab_csv = os.path.join(output_dir, "ab_snapshot_values.csv")
    df_ab.to_csv(ab_csv, index=False)
    print(f"Columns A & B evaluated at levels 26-31 across 28 snapshot dates:")
    print(f"  Total cells compared against window file: {cells_cmp}")
    print(f"  Total mismatches: {cells_mismatch}")
    if cells_mismatch > 0:
        print(f"  First mismatches: {mismatches[:5]}")
    print(f"Saved: {ab_csv}")

    # ---------------------------------------------------------
    # Section F: Pit-Candidate Cells (Static Geometry)
    # ---------------------------------------------------------
    print("\n--- Section F: Pit-Candidate Cells ---")
    df_pit_levels, pit_mask_3d = identify_pit_candidates_static(
        static_mask_3d, ref_depths, lat_a_idx, lon_a_idx, lat_b_idx, lon_b_idx
    )
    pit_level_csv = os.path.join(output_dir, "pit_candidates_by_level.csv")
    df_pit_levels.to_csv(pit_level_csv, index=False)
    print(f"Total pit candidates across levels 0-35: {df_pit_levels['n_pit_candidates'].sum()}")
    print("Columns A and B pit status at levels 26-31:")
    for _, row in df_pit_levels[df_pit_levels["level"].isin(range(26, 32))].iterrows():
        lvl = int(row["level"])
        print(f"  Level {lvl:2d} ({row['depth_m']:6.1f}m): A_pit={row['A_is_pit']} (wet={row['A_is_wet']}, neighbors={row['A_neighbors']}), B_pit={row['B_is_pit']} (wet={row['B_is_wet']}, neighbors={row['B_neighbors']})")

    pit_year_records = []
    top_10_pit_records = []
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            s_raw = fs.variables["thetao"][0]
            counts, top_10 = analyze_pit_candidates_snapshot(
                s_raw, pit_mask_3d, ref_lats, ref_lons, ref_depths, y,
                fill_val, scale_factor, add_offset, valid_min,
                lat_a_idx, lon_a_idx, lat_b_idx, lon_b_idx
            )
            pit_year_records.append(counts)
            top_10_pit_records.extend(top_10)

    df_pit_year = pd.DataFrame(top_10_pit_records)
    pit_year_csv = os.path.join(output_dir, "pit_candidates_by_year.csv")
    df_pit_year.to_csv(pit_year_csv, index=False)

    df_pit_counts = pd.DataFrame(pit_year_records)
    print(f"Total pit cells with raw == -28671 across years: {df_pit_counts['n_pit_raw_neg_28671'].sum()}")
    print(f"Total pit cells with decoded < 0 C across years: {df_pit_counts['n_pit_subzero'].sum()}")
    print(f"Total pit cells near floor across years: {df_pit_counts['n_pit_near_floor'].sum()}")
    print(f"Saved: {pit_level_csv}")
    print(f"Saved: {pit_year_csv}")

    # ---------------------------------------------------------
    # Section G: Pre-Freeze Trajectory
    # ---------------------------------------------------------
    print("\n--- Section G: Pre-Freeze Trajectory (Window File) ---")
    df_prefreeze, df_level28, prefreeze_summary = audit_prefreeze_trajectory(
        window_file, lat_target=15.0, lon_a_target=50.583332, lon_b_target=50.666668,
        fill_value=fill_val, valid_min=valid_min,
        scale_factor=scale_factor, add_offset=add_offset
    )
    prefreeze_csv = os.path.join(output_dir, "prefreeze_trajectory.csv")
    df_prefreeze.to_csv(prefreeze_csv, index=False)

    level28_csv = os.path.join(output_dir, "level28_tail.csv")
    df_level28.to_csv(level28_csv, index=False)

    for col in ["A", "B"]:
        for lvl in [29, 30]:
            info = prefreeze_summary[f"{col}_lvl{lvl}"]
            print(f"  Col {col} Lvl {lvl} ({info['depth_m']:.1f}m): Final zero run start = {info['freeze_start_date']} (len={info['freeze_run_length_days']} days)")
            print(f"    Record min = {info['record_min_decoded']:.5f} °C (raw={info['record_min_raw']}) on {info['record_min_date']}, diff from valid_min = {info['diff_from_valid_min_C']:.5f} °C")
    for col in ["A", "B"]:
        info = prefreeze_summary[f"{col}_lvl28"]
        print(f"  Col {col} Lvl 28 ({info['depth_m']:.1f}m): Record min = {info['record_min_decoded']:.5f} °C (raw={info['record_min_raw']}) on {info['record_min_date']}, days within 50 codes of valid_min = {info['days_within_50_codes_of_valid_min']}")

    print(f"Saved: {prefreeze_csv}")
    print(f"Saved: {level28_csv}")

    # ---------------------------------------------------------
    # Section H: Plots
    # ---------------------------------------------------------
    plot_paths = {}
    if not skip_plots:
        print("\n--- Section H: Diagnostic Visualizations ---")
        plot_paths = generate_plots(
            df_census, window_file, prefreeze_summary, output_dir,
            lat_target=15.0, lon_a_target=50.583332, lon_b_target=50.666668,
            fill_value=fill_val, scale_factor=scale_factor, add_offset=add_offset
        )
        for k, p in plot_paths.items():
            print(f"  Generated {k}: {p}")

    # ---------------------------------------------------------
    # Final File Metadata (AFTER) & Integrity Check
    # ---------------------------------------------------------
    print(f"\nComputing SHA256 and metadata for {len(input_files)} input files (AFTER)...")
    meta_after = {}
    for fpath in input_files:
        meta_after[os.path.abspath(fpath)] = compute_file_metadata(fpath)

    integrity_ok, mismatches = verify_integrity_unchanged(meta_before, meta_after)
    if not integrity_ok:
        raise RuntimeError(f"RAW DATA INTEGRITY VIOLATION! Mismatches: {mismatches}")
    print(f"RAW DATA INTEGRITY VERIFIED: All {len(input_files)} input files are 100% byte-for-byte identical.")

    # ---------------------------------------------------------
    # Scalar Findings & JSON Dump
    # ---------------------------------------------------------
    results_json = {
        "analysis_name": "glorys_snapshot_census_and_prefreeze_trajectory",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "integrity": {
            "all_files_unchanged": integrity_ok,
            "total_files_checked": len(input_files),
            "files": {
                p: {
                    "filename": meta_before[p]["filename"],
                    "sha256_before": meta_before[p]["sha256"],
                    "sha256_after": meta_after[p]["sha256"],
                    "size_bytes": meta_before[p]["size_bytes"],
                    "mtime_before": meta_before[p]["mtime_utc"],
                    "mtime_after": meta_after[p]["mtime_utc"],
                    "match": bool(meta_before[p]["sha256"] == meta_after[p]["sha256"]),
                }
                for p in meta_before
            },
        },
        "section_A_inventory": summary_A,
        "section_B_mask_consistency": {
            "total_diff_vs_jan2019": total_diff_jan,
            "total_diff_vs_static": total_diff_static,
            "yearly_mask_diffs": df_mask.to_dict(orient="records"),
            "sample_differing_cells": differing_cells_list[:50],
        },
        "section_C_census": {
            "total_finite_cells": int(df_census["n_finite"].sum()),
            "total_raw_neg_28671": int(df_census["n_raw_neg_28671"].sum()),
            "total_abs_lt_1e-3": int(df_census["n_abs_lt_1e-3"].sum()),
            "total_subzero": int(df_census["n_subzero"].sum()),
            "total_near_floor": int(df_census["n_near_floor"].sum()),
            "total_near_ceiling": int(df_census["n_near_ceiling"].sum()),
            "total_flagged_cells": len(df_flagged),
            "year_flagged_counts": year_flagged_counts,
        },
        "section_D_extremes": {
            "total_extreme_records": len(df_extremes),
            "ab_in_extremes_count": ab_in_extremes_count,
            "overall_coldest_record": df_extremes[df_extremes["extreme_type"] == "coldest"].sort_values("decoded").iloc[0].to_dict() if len(df_extremes) > 0 else None,
            "overall_warmest_record": df_extremes[df_extremes["extreme_type"] == "warmest"].sort_values("decoded", ascending=False).iloc[0].to_dict() if len(df_extremes) > 0 else None,
        },
        "section_E_ab_snapshots": {
            "cells_compared": cells_cmp,
            "cells_mismatched": cells_mismatch,
            "first_mismatches": mismatches[:20],
        },
        "section_F_pit_candidates": {
            "total_pit_candidates": int(df_pit_levels["n_pit_candidates"].sum()),
            "pit_candidates_by_level": df_pit_levels.to_dict(orient="records"),
            "pit_flagged_counts_by_year": df_pit_counts.to_dict(orient="records"),
        },
        "section_G_prefreeze_trajectory": prefreeze_summary,
        "section_H_plots": plot_paths,
        "unknown_fields": {
            "physical_cause_of_pit_drift": None,
            "reason_for_April_2005_colA_dip": None,
        },
    }

    json_path = os.path.join(output_dir, "glorys_snapshot_census.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(results_json, fj, indent=2)
    print(f"\nSaved master JSON summary: {json_path}")
    print("=" * 80)

    return results_json


def main():
    parser = argparse.ArgumentParser(description="GLORYS Annual Snapshot Census Diagnostic")
    parser.add_argument("--snapshot-dir", default="data/raw/glorys_snapshots", help="Path to snapshot directory")
    parser.add_argument("--jan2019-file", default="data/raw/glorys/glorys_thetao_2019-01.nc", help="Path to Jan 2019 reference file")
    parser.add_argument("--window-file", default="data/raw/glorys_AB_window/glorys_thetao_AB_window_1993-2020.nc", help="Path to AB window file")
    parser.add_argument("--static-file", default="data/raw/glorys_static/glorys_static_mask_bathy.nc", help="Path to static mask/bathy file")
    parser.add_argument("--output-dir", default="outputs/diagnostics/glorys_snapshot_census", help="Output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    run_snapshot_census(
        snapshot_dir=args.snapshot_dir,
        jan2019_file=args.jan2019_file,
        window_file=args.window_file,
        static_file=args.static_file,
        output_dir=args.output_dir,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
