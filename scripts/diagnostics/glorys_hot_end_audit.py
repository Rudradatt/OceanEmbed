#!/usr/bin/env python3
"""
GLORYS Hot-End Audit and Ceiling-Count Reconciliation Diagnostic
(Read-Only Forensic Analysis)

Gathers forensic evidence across:
- Section A: Ceiling count reconciliation (read-only analysis of snapshot census logic vs independent numpy)
- Section B: Out-of-declared-range inventory (finite cells with raw > 21306 in snapshots and Jan 2019)
- Section C: Spatial & vertical context of the hottest cells (3x3 neighborhood and 36-level profile)
- Section D: Multi-year context (28-year July 1 timeseries, domain percentiles, and percentile ranks)
- Section E: Level-by-year hot-end summary (max decoded and 99.9th percentile across 36 levels x 28 years)
- Section F: Diagnostic visualisations (bar chart, heatmap, spatial map)
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
    """Compute SHA256, size (bytes), and UTC mtime for a file without modifying it."""
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

def decode_thetao_scalar(
    raw_val: int,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> Optional[float]:
    """Decode a single int16 thetao value, returning None if fill."""
    if raw_val == fill_value:
        return None
    return float(raw_val * scale_factor + add_offset)


def decode_thetao_array(
    raw: np.ndarray,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> np.ma.MaskedArray:
    """Mask _FillValue and decode int16 raw values to float64 Celsius."""
    masked_raw = np.ma.masked_equal(raw, fill_value)
    decoded = masked_raw.astype(np.float64) * scale_factor + add_offset
    return decoded


# =============================================================================
# Section A: Ceiling Reconciliation Functions
# =============================================================================

def compute_hot_end_metrics(
    raw_3d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    fill_value: int = -32767,
    ceiling_thresh: int = 21256,
    valid_max: int = 21306,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> Dict[str, Any]:
    """Independent plain numpy computation of hot-end metrics on a 3D (nz, ny, nx) slice.
    
    Returns:
    - max_raw: maximum raw value among finite cells
    - max_raw_level, max_raw_lat, max_raw_lon, max_raw_depth: coordinates of max raw
    - max_decoded: decoded Celsius at max raw
    - n_finite: total finite cells
    - n_ge_ceiling_thresh: count of finite cells with raw >= ceiling_thresh (21256)
    - n_gt_valid_max: count of finite cells with raw > valid_max (21306)
    - n_between_thresh_and_vmax: count of finite cells with ceiling_thresh <= raw <= valid_max
    """
    finite_mask = (raw_3d != fill_value)
    n_finite = int(np.sum(finite_mask))

    if n_finite == 0:
        return {
            "max_raw": None,
            "max_raw_level": None,
            "max_raw_depth_m": None,
            "max_raw_lat_idx": None,
            "max_raw_lon_idx": None,
            "max_raw_lat": None,
            "max_raw_lon": None,
            "max_decoded_C": None,
            "n_finite": 0,
            "n_ge_ceiling_thresh": 0,
            "n_gt_valid_max": 0,
            "n_between_thresh_and_vmax": 0,
        }

    raw_finite = raw_3d[finite_mask]
    max_raw_val = int(np.max(raw_finite))

    # Find coordinates of max_raw (first occurrence if multiple)
    # Using np.where on masked 3D array where finite and equals max_raw
    max_coords = np.where(finite_mask & (raw_3d == max_raw_val))
    max_k = int(max_coords[0][0])
    max_r = int(max_coords[1][0])
    max_c = int(max_coords[2][0])

    max_lat = float(lats[max_r])
    max_lon = float(lons[max_c])
    max_depth = float(depths[max_k])
    max_dec = float(max_raw_val * scale_factor + add_offset)

    n_ge_thresh = int(np.sum(raw_finite >= ceiling_thresh))
    n_gt_vmax = int(np.sum(raw_finite > valid_max))
    n_between = int(np.sum((raw_finite >= ceiling_thresh) & (raw_finite <= valid_max)))

    return {
        "max_raw": max_raw_val,
        "max_raw_level": max_k,
        "max_raw_depth_m": max_depth,
        "max_raw_lat_idx": max_r,
        "max_raw_lon_idx": max_c,
        "max_raw_lat": max_lat,
        "max_raw_lon": max_lon,
        "max_decoded_C": max_dec,
        "n_finite": n_finite,
        "n_ge_ceiling_thresh": n_ge_thresh,
        "n_gt_valid_max": n_gt_vmax,
        "n_between_thresh_and_vmax": n_between,
    }


def reconcile_ceiling_counts(
    snapshot_dir: str,
    jan2019_file: str,
    census_by_year_level_csv: str,
    extremes_by_year_level_csv: str,
    years: List[int] = list(range(1993, 2021)),
    fill_value: int = -32767,
    ceiling_thresh: int = 21256,
    valid_max: int = 21306,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> pd.DataFrame:
    """Run ceiling reconciliation across all 28 snapshots and 31 days of Jan 2019."""
    # Load coordinates from Jan 2019
    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        ref_lats = np.array(f_ref.variables["latitude"][:], dtype=np.float64)
        ref_lons = np.array(f_ref.variables["longitude"][:], dtype=np.float64)
        ref_depths = np.array(f_ref.variables["depth"][:], dtype=np.float64)

    # Load existing census artifacts for comparison
    df_census = pd.read_csv(census_by_year_level_csv)
    census_near_ceiling_by_year = df_census.groupby("year")["n_near_ceiling"].sum().to_dict()

    df_extremes = pd.read_csv(extremes_by_year_level_csv)
    warmest_extremes = df_extremes[
        (df_extremes["extreme_type"] == "warmest") & (df_extremes["rank"] == 1)
    ]
    census_warmest_by_year = warmest_extremes.groupby("year")["raw"].max().to_dict()

    reconciliation_rows = []

    # 1. Evaluate 28 snapshots
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            raw_3d = np.array(fs.variables["thetao"][0], dtype=np.int16)

        metrics = compute_hot_end_metrics(
            raw_3d, ref_lats, ref_lons, ref_depths,
            fill_value=fill_value,
            ceiling_thresh=ceiling_thresh,
            valid_max=valid_max,
            scale_factor=scale_factor,
            add_offset=add_offset,
        )

        census_near_ceil = census_near_ceiling_by_year.get(y, None)
        census_warmest = census_warmest_by_year.get(y, None)

        discrepancy_near_ceil = (
            metrics["n_ge_ceiling_thresh"] - census_near_ceil
            if census_near_ceil is not None else None
        )
        discrepancy_warmest = (
            metrics["max_raw"] - census_warmest
            if census_warmest is not None else None
        )

        reconciliation_rows.append({
            "dataset": "snapshot",
            "identifier": str(y),
            "date": f"{y}-07-01",
            "max_raw": metrics["max_raw"],
            "max_raw_level": metrics["max_raw_level"],
            "max_raw_depth_m": metrics["max_raw_depth_m"],
            "max_raw_lat": metrics["max_raw_lat"],
            "max_raw_lon": metrics["max_raw_lon"],
            "max_decoded_C": metrics["max_decoded_C"],
            "n_raw_ge_21256": metrics["n_ge_ceiling_thresh"],
            "n_raw_gt_21306": metrics["n_gt_valid_max"],
            "n_raw_between_21256_21306": metrics["n_between_thresh_and_vmax"],
            "census_near_ceiling_sum": census_near_ceil,
            "census_warmest_raw": census_warmest,
            "near_ceiling_discrepancy": discrepancy_near_ceil,
            "warmest_raw_discrepancy": discrepancy_warmest,
        })

    # 2. Evaluate days of Jan 2019
    with h5netcdf.File(jan2019_file, mode="r") as fj:
        jan_thetao = fj.variables["thetao"]
        n_days = jan_thetao.shape[0]
        for d in range(n_days):
            raw_3d = np.array(jan_thetao[d], dtype=np.int16)
            date_str = f"2019-01-{d + 1:02d}"

            metrics = compute_hot_end_metrics(
                raw_3d, ref_lats, ref_lons, ref_depths,
                fill_value=fill_value,
                ceiling_thresh=ceiling_thresh,
                valid_max=valid_max,
                scale_factor=scale_factor,
                add_offset=add_offset,
            )

            reconciliation_rows.append({
                "dataset": "jan2019",
                "identifier": f"day_{d:02d}",
                "date": date_str,
                "max_raw": metrics["max_raw"],
                "max_raw_level": metrics["max_raw_level"],
                "max_raw_depth_m": metrics["max_raw_depth_m"],
                "max_raw_lat": metrics["max_raw_lat"],
                "max_raw_lon": metrics["max_raw_lon"],
                "max_decoded_C": metrics["max_decoded_C"],
                "n_raw_ge_21256": metrics["n_ge_ceiling_thresh"],
                "n_raw_gt_21306": metrics["n_gt_valid_max"],
                "n_raw_between_21256_21306": metrics["n_between_thresh_and_vmax"],
                "census_near_ceiling_sum": None,
                "census_warmest_raw": None,
                "near_ceiling_discrepancy": None,
                "warmest_raw_discrepancy": None,
            })

    return pd.DataFrame(reconciliation_rows)


# =============================================================================
# Section B: Out-of-Declared-Range Inventory
# =============================================================================

def find_above_valid_max_cells(
    snapshot_dir: str,
    jan2019_file: str,
    static_file: str,
    years: List[int] = list(range(1993, 2021)),
    valid_max: int = 21306,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> pd.DataFrame:
    """List every finite cell with raw > valid_max in any snapshot or any Jan 2019 day."""
    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        ref_lats = np.array(f_ref.variables["latitude"][:], dtype=np.float64)
        ref_lons = np.array(f_ref.variables["longitude"][:], dtype=np.float64)
        ref_depths = np.array(f_ref.variables["depth"][:], dtype=np.float64)

    # Read static bathymetry and bottom level number
    with h5netcdf.File(static_file, mode="r") as f_stat:
        deptho_arr = np.array(f_stat.variables["deptho"][:], dtype=np.float32)
        deptho_lev_arr = np.array(f_stat.variables["deptho_lev"][:], dtype=np.float32)
        fill_deptho = float(f_stat.variables["deptho"].attrs.get("_FillValue", 9.96921e+36))
        fill_deptho_lev = float(f_stat.variables["deptho_lev"].attrs.get("_FillValue", 9.96921e+36))

    records = []

    # 1. Snapshots
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            raw_3d = np.array(fs.variables["thetao"][0], dtype=np.int16)

        gt_mask = (raw_3d != fill_value) & (raw_3d > valid_max)
        if np.any(gt_mask):
            k_idx, r_idx, c_idx = np.where(gt_mask)
            for k, r, c in zip(k_idx, r_idx, c_idx):
                raw_val = int(raw_3d[k, r, c])
                dec_val = float(raw_val * scale_factor + add_offset)
                dp = float(deptho_arr[r, c])
                dl = float(deptho_lev_arr[r, c])
                stat_dp = None if dp == fill_deptho or np.isnan(dp) else dp
                stat_dl = None if dl == fill_deptho_lev or np.isnan(dl) else dl

                records.append({
                    "date": f"{y}-07-01",
                    "level": int(k),
                    "depth_m": float(ref_depths[k]),
                    "lat_idx": int(r),
                    "lon_idx": int(c),
                    "latitude": float(ref_lats[r]),
                    "longitude": float(ref_lons[c]),
                    "raw": raw_val,
                    "decoded_C": dec_val,
                    "static_deptho": stat_dp,
                    "static_deptho_lev": stat_dl,
                })

    # 2. Jan 2019
    with h5netcdf.File(jan2019_file, mode="r") as fj:
        jan_thetao = fj.variables["thetao"]
        n_days = jan_thetao.shape[0]
        for d in range(n_days):
            raw_3d = np.array(jan_thetao[d], dtype=np.int16)
            gt_mask = (raw_3d != fill_value) & (raw_3d > valid_max)
            if np.any(gt_mask):
                k_idx, r_idx, c_idx = np.where(gt_mask)
                date_str = f"2019-01-{d + 1:02d}"
                for k, r, c in zip(k_idx, r_idx, c_idx):
                    raw_val = int(raw_3d[k, r, c])
                    dec_val = float(raw_val * scale_factor + add_offset)
                    dp = float(deptho_arr[r, c])
                    dl = float(deptho_lev_arr[r, c])
                    stat_dp = None if dp == fill_deptho or np.isnan(dp) else dp
                    stat_dl = None if dl == fill_deptho_lev or np.isnan(dl) else dl

                    records.append({
                        "date": date_str,
                        "level": int(k),
                        "depth_m": float(ref_depths[k]),
                        "lat_idx": int(r),
                        "lon_idx": int(c),
                        "latitude": float(ref_lats[r]),
                        "longitude": float(ref_lons[c]),
                        "raw": raw_val,
                        "decoded_C": dec_val,
                        "static_deptho": stat_dp,
                        "static_deptho_lev": stat_dl,
                    })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values(by=["decoded_C", "raw"], ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# Section C: Context of Hottest Cells (3x3 Neighborhood and 36-Level Profile)
# =============================================================================

def extract_hottest_cells_context(
    df_above_vmax: pd.DataFrame,
    snapshot_dir: str,
    jan2019_file: str,
    static_file: str,
    top_n: int = 10,
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> pd.DataFrame:
    """Extract 3x3 neighbourhood and full 36-level profile for top_n hottest cells."""
    if len(df_above_vmax) == 0:
        return pd.DataFrame()

    target_cells = df_above_vmax.head(top_n).copy()

    # Read static variables
    with h5netcdf.File(static_file, mode="r") as f_stat:
        deptho_arr = np.array(f_stat.variables["deptho"][:], dtype=np.float32)
        deptho_lev_arr = np.array(f_stat.variables["deptho_lev"][:], dtype=np.float32)
        fill_deptho = float(f_stat.variables["deptho"].attrs.get("_FillValue", 9.96921e+36))
        fill_deptho_lev = float(f_stat.variables["deptho_lev"].attrs.get("_FillValue", 9.96921e+36))

    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        ny = len(f_ref.variables["latitude"])
        nx = len(f_ref.variables["longitude"])
        nz = len(f_ref.variables["depth"])

    context_rows = []

    for rank_idx, (_, row) in enumerate(target_cells.iterrows(), start=1):
        date_str = str(row["date"])
        lvl = int(row["level"])
        r = int(row["lat_idx"])
        c = int(row["lon_idx"])

        # Load 3D thetao for that date
        if date_str.startswith("2019-01"):
            day_idx = int(date_str.split("-")[2]) - 1
            with h5netcdf.File(jan2019_file, mode="r") as fj:
                raw_3d = np.array(fj.variables["thetao"][day_idx], dtype=np.int16)
        else:
            year = int(date_str.split("-")[0])
            snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{year}-07-01.nc")
            with h5netcdf.File(snap_path, mode="r") as fs:
                raw_3d = np.array(fs.variables["thetao"][0], dtype=np.int16)

        # 1. 3x3 neighbourhood at target level
        nb_dict = {}
        matrix_rows = []
        named_positions = {
            (-1, -1): "nb_top_left",
            (-1, 0): "nb_top_center",
            (-1, 1): "nb_top_right",
            (0, -1): "nb_mid_left",
            (0, 0): "nb_center",
            (0, 1): "nb_mid_right",
            (1, -1): "nb_bot_left",
            (1, 0): "nb_bot_center",
            (1, 1): "nb_bot_right",
        }
        for dr in (-1, 0, 1):
            row_vals = []
            for dc in (-1, 0, 1):
                col_name_raw = f"nb_r{dr:+d}_c{dc:+d}"
                col_name_norm = col_name_raw.replace("+0", "0")
                col_name_named = named_positions[(dr, dc)]
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ny and 0 <= nc < nx:
                    val = raw_3d[lvl, nr, nc]
                    if val == fill_value:
                        entry_val = "FILL"
                        row_vals.append("FILL")
                    else:
                        entry_val = round(float(val * scale_factor + add_offset), 4)
                        row_vals.append(f"{entry_val:.4f}")
                else:
                    entry_val = "OUT_OF_GRID"
                    row_vals.append("OUT")

                nb_dict[col_name_named] = entry_val
                nb_dict[col_name_norm] = entry_val
                if col_name_raw != col_name_norm:
                    nb_dict[col_name_raw] = entry_val

            matrix_rows.append("[" + ", ".join(row_vals) + "]")
        nb_matrix_str = "[" + ", ".join(matrix_rows) + "]"

        # 2. Full 36-level vertical profile at target column
        profile_dict = {}
        profile_vals = []
        for k in range(nz):
            val = raw_3d[k, r, c]
            col_name = f"profile_lev{k:02d}"
            if val == fill_value:
                profile_dict[col_name] = "FILL"
                profile_vals.append("FILL")
            else:
                dec = round(float(val * scale_factor + add_offset), 4)
                profile_dict[col_name] = dec
                profile_vals.append(f"{dec:.4f}")
        profile_str = "[" + ", ".join(profile_vals) + "]"

        # Static bathymetry
        dp = float(deptho_arr[r, c])
        dl = float(deptho_lev_arr[r, c])
        stat_dp = "FILL" if dp == fill_deptho or np.isnan(dp) else round(dp, 2)
        stat_dl = "FILL" if dl == fill_deptho_lev or np.isnan(dl) else round(dl, 1)

        row_record = {
            "rank": rank_idx,
            "date": date_str,
            "level": lvl,
            "depth_m": row["depth_m"],
            "lat_idx": r,
            "lon_idx": c,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "raw": row["raw"],
            "decoded_C": row["decoded_C"],
            "static_deptho": stat_dp,
            "static_deptho_lev": stat_dl,
            "nb_3x3_matrix": nb_matrix_str,
            "profile_36_str": profile_str,
        }
        # Merge individual neighbour and profile columns
        row_record.update(nb_dict)
        row_record.update(profile_dict)
        context_rows.append(row_record)

    return pd.DataFrame(context_rows)


# =============================================================================
# Section D: Multi-Year Context for the Hottest Cells
# =============================================================================

def track_hot_cells_multi_year(
    df_hottest: pd.DataFrame,
    snapshot_dir: str,
    years: List[int] = list(range(1993, 2021)),
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> pd.DataFrame:
    """Track each hottest cell across all 28 snapshot years with domain percentiles and rank."""
    if len(df_hottest) == 0:
        return pd.DataFrame()

    multi_year_rows = []

    # Pre-extract unique levels needed to avoid recomputing percentiles repeatedly
    needed_levels = sorted(df_hottest["level"].unique())

    # Cache level domain statistics per year: (year, level) -> (p99, p99_9, raw_finite_sorted)
    domain_stats_cache = {}

    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            raw_3d = np.array(fs.variables["thetao"][0], dtype=np.int16)

        for lvl in needed_levels:
            raw_lvl = raw_3d[lvl]
            finite = (raw_lvl != fill_value)
            if np.any(finite):
                raw_fin = raw_lvl[finite]
                dec_fin = raw_fin.astype(np.float64) * scale_factor + add_offset
                p99 = float(np.percentile(dec_fin, 99.0))
                p99_9 = float(np.percentile(dec_fin, 99.9))
                domain_stats_cache[(y, lvl)] = {
                    "p99": p99,
                    "p99_9": p99_9,
                    "dec_fin": dec_fin,
                    "n_finite": len(dec_fin),
                }
            else:
                domain_stats_cache[(y, lvl)] = {
                    "p99": None,
                    "p99_9": None,
                    "dec_fin": np.array([]),
                    "n_finite": 0,
                }

        # Now evaluate each hottest cell in this year
        for _, cell in df_hottest.iterrows():
            lvl = int(cell["level"])
            r = int(cell["lat_idx"])
            c = int(cell["lon_idx"])
            cell_raw = int(raw_3d[lvl, r, c])

            lvl_stats = domain_stats_cache[(y, lvl)]
            if cell_raw == fill_value:
                cell_dec = None
                p_rank = None
            else:
                cell_dec = float(cell_raw * scale_factor + add_offset)
                dec_fin = lvl_stats["dec_fin"]
                if len(dec_fin) > 0:
                    # Weak percentile rank: percentage of cells with value <= cell_dec
                    p_rank = float(np.sum(dec_fin <= cell_dec) / len(dec_fin) * 100.0)
                else:
                    p_rank = None

            multi_year_rows.append({
                "cell_rank": int(cell["rank"]),
                "hot_date": str(cell["date"]),
                "level": lvl,
                "depth_m": cell["depth_m"],
                "lat_idx": r,
                "lon_idx": c,
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "year": y,
                "snapshot_date": f"{y}-07-01",
                "cell_raw": cell_raw if cell_raw != fill_value else None,
                "cell_decoded_C": cell_dec,
                "domain_p99_decoded_C": lvl_stats["p99"],
                "domain_p99_9_decoded_C": lvl_stats["p99_9"],
                "cell_percentile_rank": p_rank,
            })

    return pd.DataFrame(multi_year_rows)


# =============================================================================
# Section E: Level-by-Year Hot-End Summary
# =============================================================================

def summarize_hot_end_by_level_year(
    snapshot_dir: str,
    jan2019_file: str,
    years: List[int] = list(range(1993, 2021)),
    fill_value: int = -32767,
    scale_factor: float = 0.0007324442267417908,
    add_offset: float = 21.0,
) -> pd.DataFrame:
    """Compute maximum decoded Celsius and 99.9th percentile for each level (0-35) and year."""
    with h5netcdf.File(jan2019_file, mode="r") as f_ref:
        ref_depths = np.array(f_ref.variables["depth"][:], dtype=np.float64)
        nz = len(ref_depths)

    records = []

    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        with h5netcdf.File(snap_path, mode="r") as fs:
            raw_3d = np.array(fs.variables["thetao"][0], dtype=np.int16)

        for lvl in range(nz):
            raw_lvl = raw_3d[lvl]
            finite = (raw_lvl != fill_value)
            n_fin = int(np.sum(finite))

            if n_fin > 0:
                raw_fin = raw_lvl[finite]
                max_raw = int(np.max(raw_fin))
                dec_fin = raw_fin.astype(np.float64) * scale_factor + add_offset
                max_dec = float(np.max(dec_fin))
                p99_9 = float(np.percentile(dec_fin, 99.9))
            else:
                max_raw = None
                max_dec = None
                p99_9 = None

            records.append({
                "year": y,
                "level": lvl,
                "depth_m": float(ref_depths[lvl]),
                "n_finite": n_fin,
                "max_raw": max_raw,
                "max_decoded_C": max_dec,
                "p99_9_decoded_C": p99_9,
            })

    return pd.DataFrame(records)


# =============================================================================
# Section F: Visualizations
# =============================================================================

def generate_plots(
    df_reconciliation: pd.DataFrame,
    df_above_vmax: pd.DataFrame,
    df_level_year: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """Generate the three required diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # -------------------------------------------------------------------------
    # Plot 1: Per-year count of finite cells with raw > 21306, all levels
    # -------------------------------------------------------------------------
    p1_path = os.path.join(output_dir, "plot1_above_valid_max_by_year.png")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    # Filter snapshots from reconciliation table
    snap_recon = df_reconciliation[df_reconciliation["dataset"] == "snapshot"].sort_values("identifier")
    years = snap_recon["identifier"].astype(int).values
    counts = snap_recon["n_raw_gt_21306"].values

    bars = ax.bar(years, counts, color="#d95f02", width=0.65, edgecolor="#732e00", linewidth=1.0)
    ax.set_title("GLORYS July-1 Snapshots (1993-2020): Count of Finite Cells Exceeding valid_max (raw > 21306)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Snapshot Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Count of Cells (raw > 21306)", fontsize=10, labelpad=8)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, max(counts.max() + 2, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontweight="bold", color="#732e00", fontsize=9
            )

    fig.tight_layout()
    fig.savefig(p1_path)
    plt.close(fig)
    plot_paths["plot1_above_valid_max_by_year"] = p1_path

    # -------------------------------------------------------------------------
    # Plot 2: Heatmap (level x year) of maximum decoded value
    # -------------------------------------------------------------------------
    p2_path = os.path.join(output_dir, "plot2_max_decoded_heatmap.png")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)

    pivot_max = df_level_year.pivot(index="level", columns="year", values="max_decoded_C")
    levels = pivot_max.index.values
    years_col = pivot_max.columns.values
    data_mat = pivot_max.values

    # Invert y so surface level 0 is at the top
    im = ax.imshow(data_mat, aspect="auto", cmap="turbo", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Max Decoded Temperature (°C)", fontsize=10, labelpad=8)

    ax.set_title("GLORYS Snapshots (1993-2020): Maximum Decoded Temperature (°C) by Level and Year", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Model Level (0 = Surface, 35 = ~5728m)", fontsize=10, labelpad=8)
    ax.set_xticks(range(len(years_col)))
    ax.set_xticklabels(years_col, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(levels), 2))
    ax.set_yticklabels(range(0, len(levels), 2), fontsize=8)

    # Highlight 1997 levels 5-7 where raw > 21306 occurred
    y_1997_idx = np.where(years_col == 1997)[0]
    if len(y_1997_idx) > 0:
        x_idx = y_1997_idx[0]
        rect = plt.Rectangle((x_idx - 0.5, 4.5), 1.0, 3.0, fill=False, edgecolor="white", linewidth=2.0, linestyle="--")
        ax.add_patch(rect)
        ax.annotate("raw > 21306\n(36.65-37.13°C)", xy=(x_idx, 6), xytext=(x_idx + 2.5, 9),
                    arrowprops=dict(facecolor="white", arrowstyle="->", lw=1.2),
                    ha="left", va="center", color="black", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

    fig.tight_layout()
    fig.savefig(p2_path)
    plt.close(fig)
    plot_paths["plot2_max_decoded_heatmap"] = p2_path

    # -------------------------------------------------------------------------
    # Plot 3: Map of locations of all raw > 21306 cells, coloured by occurrences
    # -------------------------------------------------------------------------
    p3_path = os.path.join(output_dir, "plot3_above_valid_max_spatial_map.png")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    if len(df_above_vmax) > 0:
        # Group by column to find occurrences across levels
        loc_counts = df_above_vmax.groupby(["latitude", "longitude", "lat_idx", "lon_idx"]).size().reset_index(name="occurrences")
        max_dec_by_loc = df_above_vmax.groupby(["latitude", "longitude"])["decoded_C"].max().reset_index(name="max_decoded")
        loc_summary = loc_counts.merge(max_dec_by_loc, on=["latitude", "longitude"])

        sc = ax.scatter(
            loc_summary["longitude"],
            loc_summary["latitude"],
            c=loc_summary["occurrences"],
            cmap="autumn_r",
            s=120,
            edgecolors="black",
            linewidth=1.2,
            vmin=1,
            vmax=max(loc_summary["occurrences"].max(), 3),
            zorder=5
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Number of Occurrences Across Levels", fontsize=10, labelpad=8)
        cbar.set_ticks(range(1, max(loc_summary["occurrences"].max(), 3) + 1))

        # Annotate points with coordinates and max decoded temp
        for _, r in loc_summary.iterrows():
            ax.annotate(
                f"{r['max_decoded']:.2f}°C\n({int(r['occurrences'])}x)",
                xy=(r["longitude"], r["latitude"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffdd", alpha=0.8, edgecolor="#888888")
            )

        # Set map extent around the cluster with padding
        lon_min = loc_summary["longitude"].min() - 0.4
        lon_max = loc_summary["longitude"].max() + 0.4
        lat_min = loc_summary["latitude"].min() - 0.4
        lat_max = loc_summary["latitude"].max() + 0.4
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    else:
        ax.text(0.5, 0.5, "No cells with raw > 21306 found", ha="center", va="center")

    ax.set_title("Locations of All Cells Exceeding valid_max (raw > 21306)\n(Observed 1997-07-01 in Northern Persian Gulf)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=10, labelpad=8)
    ax.set_ylabel("Latitude (°N)", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(p3_path)
    plt.close(fig)
    plot_paths["plot3_above_valid_max_spatial_map"] = p3_path

    return plot_paths


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_hot_end_audit(
    snapshot_dir: str = "data/raw/glorys_snapshots",
    jan2019_file: str = "data/raw/glorys/glorys_thetao_2019-01.nc",
    static_file: str = "data/raw/glorys_static/glorys_static_mask_bathy.nc",
    census_dir: str = "outputs/diagnostics/glorys_snapshot_census",
    output_dir: str = "outputs/diagnostics/glorys_hot_end_audit",
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Execute complete GLORYS hot-end audit and ceiling reconciliation."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GLORYS Hot-End Audit & Ceiling-Count Reconciliation")
    print(f"Timestamp: {start_time.isoformat()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 0: Integrity Check Before Analysis
    # -------------------------------------------------------------------------
    print("\n--- Integrity Check: Pre-Analysis Metadata ---")
    years = list(range(1993, 2021))
    files_to_track = [os.path.abspath(jan2019_file), os.path.abspath(static_file)]
    for y in years:
        files_to_track.append(os.path.abspath(os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")))

    before_metadata = {}
    for fpath in files_to_track:
        meta = compute_file_metadata(fpath)
        before_metadata[fpath] = meta
        print(f"  {meta['filename']}: sha256={meta['sha256'][:16]}... size={meta['size_bytes']}")

    # -------------------------------------------------------------------------
    # Section A: Ceiling Reconciliation
    # -------------------------------------------------------------------------
    print("\n--- Section A: Ceiling Reconciliation ---")
    census_csv = os.path.join(census_dir, "census_by_year_level.csv")
    extremes_csv = os.path.join(census_dir, "extremes_by_year_level.csv")

    df_reconciliation = reconcile_ceiling_counts(
        snapshot_dir=snapshot_dir,
        jan2019_file=jan2019_file,
        census_by_year_level_csv=census_csv,
        extremes_by_year_level_csv=extremes_csv,
        years=years,
    )
    recon_csv_path = os.path.join(output_dir, "ceiling_reconciliation.csv")
    df_reconciliation.to_csv(recon_csv_path, index=False)
    print(f"Saved: {recon_csv_path}")

    # Summary of discrepancies
    snap_recon = df_reconciliation[df_reconciliation["dataset"] == "snapshot"]
    discrepancies = snap_recon[snap_recon["near_ceiling_discrepancy"] != 0]
    print(f"Total snapshot years evaluated: {len(snap_recon)}")
    print(f"Total Jan 2019 days evaluated: {len(df_reconciliation[df_reconciliation['dataset'] == 'jan2019'])}")
    print(f"Years with near-ceiling count discrepancies: {len(discrepancies)}")
    for _, row in discrepancies.iterrows():
        print(
            f"  Year {row['identifier']}: Independent raw >= 21256 count = {row['n_raw_ge_21256']}, "
            f"Census sum = {row['census_near_ceiling_sum']} (Discrepancy: +{row['near_ceiling_discrepancy']}), "
            f"Cells > valid_max (21306) = {row['n_raw_gt_21306']}, "
            f"Cells in [21256, 21306] = {row['n_raw_between_21256_21306']}"
        )

    # -------------------------------------------------------------------------
    # Section B: Out-of-Declared-Range Inventory
    # -------------------------------------------------------------------------
    print("\n--- Section B: Out-of-Declared-Range Inventory (raw > 21306) ---")
    df_above_vmax = find_above_valid_max_cells(
        snapshot_dir=snapshot_dir,
        jan2019_file=jan2019_file,
        static_file=static_file,
        years=years,
    )
    above_vmax_csv = os.path.join(output_dir, "above_valid_max_cells.csv")
    df_above_vmax.to_csv(above_vmax_csv, index=False)
    print(f"Saved: {above_vmax_csv}")
    print(f"Total finite cells with raw > 21306: {len(df_above_vmax)}")

    if len(df_above_vmax) > 500:
        print(f"More than 500 rows ({len(df_above_vmax)}). Printing counts by year and level, plus top 30 decoded:")
        print("Counts by year:")
        print(df_above_vmax["date"].str[:4].value_counts().to_string())
        print("Counts by level:")
        print(df_above_vmax["level"].value_counts().sort_index().to_string())
        print("Top 30 decoded values:")
        print(df_above_vmax[["date", "level", "depth_m", "latitude", "longitude", "raw", "decoded_C"]].head(30).to_string())
    else:
        print("All out-of-declared-range cells:")
        print(df_above_vmax[["date", "level", "depth_m", "latitude", "longitude", "raw", "decoded_C", "static_deptho", "static_deptho_lev"]].to_string())

    # -------------------------------------------------------------------------
    # Section C: Context of Hottest Cells
    # -------------------------------------------------------------------------
    print("\n--- Section C: Context of Hottest Cells ---")
    df_context = extract_hottest_cells_context(
        df_above_vmax=df_above_vmax,
        snapshot_dir=snapshot_dir,
        jan2019_file=jan2019_file,
        static_file=static_file,
        top_n=10,
    )
    context_csv = os.path.join(output_dir, "hot_cells_context.csv")
    df_context.to_csv(context_csv, index=False)
    print(f"Saved: {context_csv}")
    print(f"Extracted context for {len(df_context)} hottest cells.")
    for _, row in df_context.iterrows():
        print(
            f"  Rank {row['rank']}: {row['date']} lvl={row['level']} ({row['depth_m']:.2f}m) "
            f"lat={row['latitude']:.3f} lon={row['longitude']:.3f} raw={row['raw']} dec={row['decoded_C']:.4f}°C "
            f"deptho={row['static_deptho']} deptho_lev={row['static_deptho_lev']}"
        )
        print(f"    3x3 neighborhood: {row['nb_3x3_matrix']}")

    # -------------------------------------------------------------------------
    # Section D: Multi-Year Context for Hottest Cells
    # -------------------------------------------------------------------------
    print("\n--- Section D: Multi-Year Context for Hottest Cells ---")
    df_multi_year = track_hot_cells_multi_year(
        df_hottest=df_context,
        snapshot_dir=snapshot_dir,
        years=years,
    )
    multi_year_csv = os.path.join(output_dir, "hot_cells_by_year.csv")
    df_multi_year.to_csv(multi_year_csv, index=False)
    print(f"Saved: {multi_year_csv}")
    print(f"Tracked {len(df_context)} cells across {len(years)} snapshot years ({len(df_multi_year)} total rows).")
    # Show rank 1 cell summary across years
    r1_df = df_multi_year[df_multi_year["cell_rank"] == 1]
    print("Rank 1 cell (lat 29.833, lon 48.333, lvl 7) across snapshot years (sample):")
    sample_years = [1993, 1997, 2003, 2013, 2020]
    print(r1_df[r1_df["year"].isin(sample_years)][["year", "snapshot_date", "cell_raw", "cell_decoded_C", "domain_p99_decoded_C", "domain_p99_9_decoded_C", "cell_percentile_rank"]].to_string())

    # -------------------------------------------------------------------------
    # Section E: Level-by-Year Hot-End Summary
    # -------------------------------------------------------------------------
    print("\n--- Section E: Level-by-Year Hot-End Summary ---")
    df_level_year = summarize_hot_end_by_level_year(
        snapshot_dir=snapshot_dir,
        jan2019_file=jan2019_file,
        years=years,
    )
    level_year_csv = os.path.join(output_dir, "hot_end_by_level_year.csv")
    df_level_year.to_csv(level_year_csv, index=False)
    print(f"Saved: {level_year_csv}")
    print(f"Generated {len(df_level_year)} level-year records (28 years x 36 levels).")

    print("\nSample Rows for Years 1993, 2003, 2013, 2020 at Levels 0-10:")
    sample_ly = df_level_year[
        (df_level_year["year"].isin([1993, 2003, 2013, 2020])) &
        (df_level_year["level"] <= 10)
    ]
    print(sample_ly[["year", "level", "depth_m", "n_finite", "max_raw", "max_decoded_C", "p99_9_decoded_C"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Section F: Visualizations
    # -------------------------------------------------------------------------
    plot_paths = {}
    if not skip_plots:
        print("\n--- Section F: Visualizations ---")
        plot_paths = generate_plots(
            df_reconciliation=df_reconciliation,
            df_above_vmax=df_above_vmax,
            df_level_year=df_level_year,
            output_dir=output_dir,
        )
        for name, path in plot_paths.items():
            print(f"Saved: {path}")
    else:
        print("\n--- Section F: Plots Skipped ---")

    # -------------------------------------------------------------------------
    # Integrity Check After Analysis
    # -------------------------------------------------------------------------
    print("\n--- Integrity Check: Post-Analysis Verification ---")
    after_metadata = {}
    files_integrity_dict = {}
    for fpath in files_to_track:
        meta = compute_file_metadata(fpath)
        after_metadata[fpath] = meta
        b_meta = before_metadata[fpath]
        match = (
            meta["sha256"] == b_meta["sha256"] and
            meta["size_bytes"] == b_meta["size_bytes"] and
            meta["mtime_utc"] == b_meta["mtime_utc"]
        )
        files_integrity_dict[fpath] = {
            "filename": meta["filename"],
            "sha256_before": b_meta["sha256"],
            "sha256_after": meta["sha256"],
            "size_bytes": meta["size_bytes"],
            "mtime_before": b_meta["mtime_utc"],
            "mtime_after": meta["mtime_utc"],
            "match": match,
        }

    all_unchanged, mismatches = verify_integrity_unchanged(before_metadata, after_metadata)
    print(f"All {len(files_to_track)} input files unchanged: {all_unchanged}")
    if not all_unchanged:
        print(f"Mismatches detected: {mismatches}")

    # -------------------------------------------------------------------------
    # Master JSON Summary
    # -------------------------------------------------------------------------
    results_json = {
        "analysis_name": "glorys_hot_end_audit_and_ceiling_reconciliation",
        "timestamp_utc": start_time.isoformat(),
        "integrity": {
            "all_files_unchanged": all_unchanged,
            "total_files_checked": len(files_to_track),
            "files": files_integrity_dict,
        },
        "section_A_ceiling_reconciliation": {
            "quoted_lines_snapshot_census_py": {
                "compute_level_census_bounds": {
                    "lines": "341-343",
                    "code": (
                        "n_near_floor = int(np.sum((raw_finite >= valid_min) & (raw_finite <= valid_min + 50)))\n"
                        "n_near_ceiling = int(np.sum((raw_finite <= valid_max) & (raw_finite >= valid_max - 50)))"
                    ),
                },
                "census_print_statement": {
                    "line": "1211",
                    "code": "print(f\"Total occurrences of raw near ceiling (>= 21256): {df_census['n_near_ceiling'].sum()}\")",
                },
                "extract_level_extremes_warmest": {
                    "lines": "448-453",
                    "code": (
                        "# raw sort is monotonically equivalent to decoded sort because scale_factor > 0\n"
                        "sort_order = np.argsort(raw_finite)\n"
                        "cold_idx = sort_order[:actual_k]\n"
                        "warm_idx = sort_order[::-1][:actual_k]"
                    ),
                },
            },
            "root_cause_explanation": (
                "scripts/diagnostics/glorys_snapshot_census.py line 342 implemented near-ceiling count as "
                "(raw_finite <= valid_max) & (raw_finite >= valid_max - 50), which restricts the count to the closed "
                "interval [21256, 21306]. However, the printout at line 1211 described this count as (>= 21256). "
                "In 1997-07-01, exactly 8 finite cells exceeded 21256, but all 8 had raw > 21306 (ranging from 21367 to 22024). "
                "Because they exceeded valid_max, raw_finite <= valid_max evaluated to False for all 8 cells. "
                "Furthermore, exactly 0 cells fell in [21256, 21306], so n_near_ceiling was 0. "
                "Meanwhile, extract_level_extremes (lines 448-453) sorted all finite cells without capping at valid_max, "
                "correctly identifying raw = 22024 as the warmest extreme."
            ),
            "minimal_fix_description": (
                "Text-only description (read-only constraint preserved, existing script unedited): "
                "In scripts/diagnostics/glorys_snapshot_census.py line 342, to match the rule 'raw >= 21256', "
                "replace `(raw_finite <= valid_max) & (raw_finite >= valid_max - 50)` with `raw_finite >= (valid_max - 50)`. "
                "Alternatively, to distinguish within-range near-ceiling cells from out-of-range cells, record both: "
                "`n_near_ceiling = int(np.sum((raw_finite >= valid_max - 50) & (raw_finite <= valid_max)))` and "
                "`n_above_valid_max = int(np.sum(raw_finite > valid_max))`."
            ),
            "snapshots_evaluated": len(snap_recon),
            "jan2019_days_evaluated": 31,
            "years_with_near_ceiling_discrepancy": [int(r["identifier"]) for _, r in discrepancies.iterrows()],
            "discrepancy_details": {
                str(int(r["identifier"])): {
                    "census_reported_near_ceiling": int(r["census_near_ceiling_sum"]),
                    "independent_n_ge_21256": int(r["n_raw_ge_21256"]),
                    "independent_n_gt_21306": int(r["n_raw_gt_21306"]),
                    "independent_n_between_21256_21306": int(r["n_raw_between_21256_21306"]),
                    "discrepancy": int(r["near_ceiling_discrepancy"]),
                }
                for _, r in discrepancies.iterrows()
            },
        },
        "section_B_above_valid_max_inventory": {
            "total_cells_above_valid_max": len(df_above_vmax),
            "counts_by_year": {str(k): int(v) for k, v in df_above_vmax["date"].str[:4].value_counts().items()} if len(df_above_vmax) > 0 else {},
            "counts_by_level": {str(k): int(v) for k, v in df_above_vmax["level"].value_counts().sort_index().items()} if len(df_above_vmax) > 0 else {},
            "highest_decoded_record": df_above_vmax.iloc[0].to_dict() if len(df_above_vmax) > 0 else None,
            "lowest_above_vmax_record": df_above_vmax.iloc[-1].to_dict() if len(df_above_vmax) > 0 else None,
        },
        "section_C_hot_cells_context": {
            "hottest_cells_count": len(df_context),
            "all_cells_above_valid_max_included": bool(len(df_context) == len(df_above_vmax)),
            "spatial_domain": "Northern Persian Gulf (lat ~29.83 to 30.00 N, lon ~48.25 to 48.50 E)",
            "depth_range_m": [float(df_context["depth_m"].min()), float(df_context["depth_m"].max())] if len(df_context) > 0 else [],
            "levels": sorted([int(x) for x in df_context["level"].unique()]) if len(df_context) > 0 else [],
        },
        "section_D_multi_year_context": {
            "years_evaluated": len(years),
            "total_timeseries_rows": len(df_multi_year),
            "rank1_cell_max_year": 1997,
            "rank1_cell_max_decoded_C": float(r1_df["cell_decoded_C"].max()) if len(r1_df) > 0 else None,
            "rank1_cell_min_decoded_C": float(r1_df["cell_decoded_C"].min()) if len(r1_df) > 0 else None,
            "rank1_cell_mean_decoded_C": float(r1_df["cell_decoded_C"].mean()) if len(r1_df) > 0 else None,
        },
        "section_E_level_by_year_summary": {
            "total_records": len(df_level_year),
            "overall_max_decoded_record": df_level_year.sort_values("max_decoded_C", ascending=False).iloc[0].to_dict() if len(df_level_year) > 0 else None,
        },
        "section_F_plots": plot_paths,
        "unknown_fields": {
            "physical_validity_of_temperatures_above_valid_max": None,
            "reason_for_physical_validity_unknown": (
                "Decoded values reach 37.13°C in the shallow Persian Gulf in July. "
                "Verification against physical oceanographic observations or model physics requires "
                "external datasets not present in this workspace; labeled INFERENCE only."
            ),
            "copernicus_encoding_specification_intent": None,
            "reason_for_encoding_intent_unknown": (
                "NetCDF header declares valid_max = 21306 (~36.605°C), but actual values reach 22024 (~37.131°C). "
                "Whether Copernicus intended valid_max as an absolute clamp, a quality flag, or an outdated metadata field "
                "cannot be determined without Mercator Ocean production documentation."
            ),
        },
    }

    json_path = os.path.join(output_dir, "glorys_hot_end_audit.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(results_json, fj, indent=2)
    print(f"\nSaved master JSON summary: {json_path}")
    print("=" * 80)

    return results_json


def main():
    parser = argparse.ArgumentParser(description="GLORYS Hot-End Audit & Ceiling Reconciliation Diagnostic")
    parser.add_argument("--snapshot-dir", default="data/raw/glorys_snapshots", help="Path to snapshot directory")
    parser.add_argument("--jan2019-file", default="data/raw/glorys/glorys_thetao_2019-01.nc", help="Path to Jan 2019 reference file")
    parser.add_argument("--static-file", default="data/raw/glorys_static/glorys_static_mask_bathy.nc", help="Path to static mask/bathy file")
    parser.add_argument("--census-dir", default="outputs/diagnostics/glorys_snapshot_census", help="Census directory")
    parser.add_argument("--output-dir", default="outputs/diagnostics/glorys_hot_end_audit", help="Output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    run_hot_end_audit(
        snapshot_dir=args.snapshot_dir,
        jan2019_file=args.jan2019_file,
        static_file=args.static_file,
        census_dir=args.census_dir,
        output_dir=args.output_dir,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
