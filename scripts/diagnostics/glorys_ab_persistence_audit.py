#!/usr/bin/env python3
"""
GLORYS A/B Window Persistence Audit over 1993-2020 (Read-Only Forensic Script)

Audits temporal persistence of anomalies in GLORYS thetao around Columns A & B:
- Section A: Window file integrity, calendar continuity, and encoding structure
- Section B: Cross-download reproducibility (Jan 2019 slice vs full-domain file)
- Section C: Per-column, per-level audit for A & B over 1993-2020 (all 36 levels & per-year)
- Section D: Wet-mask history and temporal invariance check
- Section E: Window-wide census of -28671 code and longest continuous runs
- Section F: Diagnostic visualizations (time series, heatmaps, spatial anomaly map)
"""

from __future__ import annotations

import argparse
import datetime
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
import h5netcdf


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
        "sha256": sha256.hexdigest().lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": mtime_dt.isoformat(),
    }


# =============================================================================
# Helper: Coordinate Lookup with Strict Tolerance
# =============================================================================

def lookup_coord_index(
    coords: np.ndarray,
    target_val: float,
    coord_name: str,
    tolerance: float = 1e-3,
) -> int:
    """Find index of coordinate closest to target_val; raise ValueError if beyond tolerance."""
    coords_arr = np.asarray(coords)
    diffs = np.abs(coords_arr - target_val)
    min_idx = int(np.argmin(diffs))
    min_diff = float(diffs[min_idx])
    if min_diff > tolerance:
        raise ValueError(
            f"Nearest {coord_name} {coords_arr[min_idx]:.6f} exceeds tolerance {tolerance:.6f} "
            f"from target {target_val:.6f} (diff = {min_diff:.6e})"
        )
    return min_idx


# =============================================================================
# Helper: Run-Length Longest Run Computation
# =============================================================================

def compute_longest_run(arr_1d: np.ndarray) -> Tuple[int, int]:
    """Compute the length and raw value of the longest run of identical values in arr_1d.
    
    Correctly handles runs at the start, middle, and end of the array.
    """
    if len(arr_1d) == 0:
        return 0, 0
    if len(arr_1d) == 1:
        return 1, int(arr_1d[0])

    diff = (arr_1d[1:] != arr_1d[:-1])
    change_idx = np.where(diff)[0] + 1
    run_starts = np.r_[0, change_idx]
    run_ends = np.r_[change_idx, len(arr_1d)]
    run_lengths = run_ends - run_starts
    max_pos = int(np.argmax(run_lengths))
    return int(run_lengths[max_pos]), int(arr_1d[run_starts[max_pos]])


# =============================================================================
# Section A: Window File Integrity and Structure
# =============================================================================

def dump_window_integrity_and_structure(
    window_path: str,
    reference_path: str,
) -> Dict[str, Any]:
    """Dump dimensions, thetao attributes, calendar continuity, and compare encoding."""
    with h5netcdf.File(window_path, mode="r") as fw:
        w_dims = {k: len(v) for k, v in fw.dimensions.items()}
        w_glob = dict(fw.attrs)
        w_thetao_attrs = dict(fw.variables["thetao"].attrs)
        w_lats = np.array(fw.variables["latitude"])
        w_lons = np.array(fw.variables["longitude"])
        w_depths = np.array(fw.variables["depth"])
        w_times = np.array(fw.variables["time"])

    with h5netcdf.File(reference_path, mode="r") as fr:
        r_thetao_attrs = dict(fr.variables["thetao"].attrs)

    # Clean attribute dicts for json serialization
    def clean_val(v):
        if hasattr(v, "item"):
            return v.item()
        return v

    w_thetao_attrs_clean = {k: clean_val(v) for k, v in w_thetao_attrs.items()}
    r_thetao_attrs_clean = {k: clean_val(v) for k, v in r_thetao_attrs.items()}

    # Compare key encoding attributes
    encoding_keys = ["scale_factor", "add_offset", "_FillValue", "valid_min", "valid_max"]
    encoding_differences = {}
    for k in encoding_keys:
        wv = w_thetao_attrs_clean.get(k)
        rv = r_thetao_attrs_clean.get(k)
        if wv != rv:
            encoding_differences[k] = {"window": wv, "reference": rv}

    # Calendar analysis: times are 'hours since 1950-01-01'
    base_dt = datetime.datetime(1950, 1, 1, 0, 0, 0)
    dates = [base_dt + datetime.timedelta(hours=float(h)) for h in w_times]

    n_days = len(dates)
    unique_days = len(set(d.date() for d in dates))

    # Check daily steps
    time_step_diffs = set(dates[i+1] - dates[i] for i in range(n_days - 1))
    has_gaps_or_dups = bool(unique_days != n_days or time_step_diffs != {datetime.timedelta(days=1)})

    expected_start = datetime.datetime(1993, 1, 1, 0, 0, 0)
    expected_end = datetime.datetime(2020, 12, 31, 0, 0, 0)
    expected_days = (expected_end - expected_start).days + 1  # 10227

    return {
        "dimensions": w_dims,
        "global_attributes": {k: repr(v) for k, v in w_glob.items()},
        "thetao_attributes": w_thetao_attrs_clean,
        "reference_thetao_attributes": r_thetao_attrs_clean,
        "encoding_identical_to_reference": (len(encoding_differences) == 0),
        "encoding_differences": encoding_differences,
        "time_start_utc": dates[0].isoformat(),
        "time_end_utc": dates[-1].isoformat(),
        "total_days": n_days,
        "expected_days": expected_days,
        "unique_days_count": unique_days,
        "time_step_intervals": [str(x) for x in time_step_diffs],
        "has_calendar_gaps_or_duplicates": has_gaps_or_dups,
        "latitude_range": [float(w_lats[0]), float(w_lats[-1])],
        "latitude_count": len(w_lats),
        "longitude_range": [float(w_lons[0]), float(w_lons[-1])],
        "longitude_count": len(w_lons),
        "depth_range_m": [float(w_depths[0]), float(w_depths[-1])],
        "depth_levels_count": len(w_depths),
    }


# =============================================================================
# Section B: Cross-Download Reproducibility
# =============================================================================

def compare_cross_download_reproducibility(
    window_path: str,
    full_path: str,
    year: int = 2019,
    month: int = 1,
) -> Dict[str, Any]:
    """Compare January 2019 raw int16 slice of window file with the full-domain file."""
    with h5netcdf.File(window_path, mode="r") as fw, h5netcdf.File(full_path, mode="r") as ff:
        w_times = np.array(fw.variables["time"])
        base_dt = datetime.datetime(1950, 1, 1)
        w_dates = [base_dt + datetime.timedelta(hours=float(h)) for h in w_times]

        jan2019_w_indices = [i for i, d in enumerate(w_dates) if d.year == year and d.month == month]
        if not jan2019_w_indices:
            raise ValueError(f"No dates found for {year}-{month:02d} in window file!")

        w_lats = np.array(fw.variables["latitude"])
        w_lons = np.array(fw.variables["longitude"])
        w_depths = np.array(fw.variables["depth"])

        f_lats = np.array(ff.variables["latitude"])
        f_lons = np.array(ff.variables["longitude"])
        f_depths = np.array(ff.variables["depth"])

        # Match spatial coordinates
        f_lat_idx = [lookup_coord_index(f_lats, lat, "latitude", tolerance=1e-3) for lat in w_lats]
        f_lon_idx = [lookup_coord_index(f_lons, lon, "longitude", tolerance=1e-3) for lon in w_lons]

        # Verify depth coordinates match
        depth_diff = np.max(np.abs(w_depths - f_depths[:len(w_depths)]))
        if depth_diff > 1e-3:
            raise ValueError(f"Depth coordinate mismatch: max diff {depth_diff} m")

        # Slice window file: (31, 36, n_lat, n_lon)
        t_start = jan2019_w_indices[0]
        t_end = jan2019_w_indices[-1] + 1
        raw_w = np.asarray(fw.variables["thetao"][t_start:t_end, :, :, :], dtype=np.int16)

        # Slice full file
        raw_f = np.asarray(
            ff.variables["thetao"][:, :len(w_depths), f_lat_idx[0]:f_lat_idx[-1]+1, f_lon_idx[0]:f_lon_idx[-1]+1],
            dtype=np.int16,
        )

    cells_compared = raw_w.size
    is_exact_match = bool(np.array_equal(raw_w, raw_f))

    mismatches = []
    if not is_exact_match:
        diff_indices = np.where(raw_w != raw_f)
        for i in range(min(20, len(diff_indices[0]))):
            t_i = int(diff_indices[0][i])
            k_i = int(diff_indices[1][i])
            y_i = int(diff_indices[2][i])
            x_i = int(diff_indices[3][i])
            mismatches.append({
                "day_index_in_slice": t_i,
                "date": w_dates[t_start + t_i].strftime("%Y-%m-%d"),
                "level": k_i,
                "depth_m": float(w_depths[k_i]),
                "lat": float(w_lats[y_i]),
                "lon": float(w_lons[x_i]),
                "window_raw_code": int(raw_w[t_i, k_i, y_i, x_i]),
                "full_raw_code": int(raw_f[t_i, k_i, y_i, x_i]),
            })

    return {
        "year": year,
        "month": month,
        "days_in_slice": len(jan2019_w_indices),
        "cells_compared": cells_compared,
        "is_exact_match": is_exact_match,
        "mismatch_count": int(np.sum(raw_w != raw_f)),
        "first_20_mismatches": mismatches,
    }


# =============================================================================
# Section C: Per-Column, Per-Level Audit for A & B over Whole Record
# =============================================================================

def audit_columns_ab_levels(
    window_path: str,
    col_a_idx: Tuple[int, int],
    col_b_idx: Tuple[int, int],
) -> Dict[str, Any]:
    """Perform per-column, per-level audit across 1993-2020 for A & B."""
    ya, xa = col_a_idx
    yb, xb = col_b_idx

    with h5netcdf.File(window_path, mode="r") as fw:
        var = fw.variables["thetao"]
        scale_factor = float(var.attrs.get("scale_factor", 0.0007324442267417908))
        add_offset = float(var.attrs.get("add_offset", 21.0))
        fill_value = int(var.attrs.get("_FillValue", -32767))

        depths = np.array(fw.variables["depth"])
        times = np.array(fw.variables["time"])

        raw_A = np.asarray(var[:, :, ya, xa], dtype=np.int16)  # (10227, 36)
        raw_B = np.asarray(var[:, :, yb, xb], dtype=np.int16)  # (10227, 36)

    base_dt = datetime.datetime(1950, 1, 1)
    dates = [base_dt + datetime.timedelta(hours=float(h)) for h in times]
    years = np.array([d.year for d in dates])
    unique_years = sorted(list(set(years)))

    n_times, n_depths = raw_A.shape
    level_summary_rows = []

    for col_name, raw_arr in [("A", raw_A), ("B", raw_B)]:
        for k in range(n_depths):
            col_k = raw_arr[:, k]
            fin_mask = (col_k != fill_value)
            days_finite = int(np.sum(fin_mask))
            days_raw_28671 = int(np.sum(col_k == -28671))

            if days_finite > 0:
                fin_codes = col_k[fin_mask]
                dec_vals = fin_codes.astype(np.float64) * scale_factor + add_offset
                days_near_zero = int(np.sum(np.abs(dec_vals) < 1e-3))
                days_sub_neg_half = int(np.sum(dec_vals < -0.5))
                num_distinct_codes = int(len(np.unique(fin_codes)))
                min_dec = float(np.min(dec_vals))
                mean_dec = float(np.mean(dec_vals))
                max_dec = float(np.max(dec_vals))
            else:
                days_near_zero = 0
                days_sub_neg_half = 0
                num_distinct_codes = 0
                min_dec = np.nan
                mean_dec = np.nan
                max_dec = np.nan

            level_summary_rows.append({
                "column": col_name,
                "level": k,
                "depth_m": float(depths[k]),
                "days_finite": days_finite,
                "days_raw_28671": days_raw_28671,
                "days_near_zero": days_near_zero,
                "days_sub_neg_half": days_sub_neg_half,
                "num_distinct_raw_codes": num_distinct_codes,
                "min_decoded_C": min_dec,
                "mean_decoded_C": mean_dec,
                "max_decoded_C": max_dec,
            })

    df_level_summary = pd.DataFrame(level_summary_rows)

    # Per-year table for levels 27 to 31
    year_level_rows = []
    target_levels = [27, 28, 29, 30, 31]

    for yr in unique_years:
        yr_mask = (years == yr)
        days_in_yr = int(np.sum(yr_mask))

        for col_name, raw_arr in [("A", raw_A), ("B", raw_B)]:
            for k in target_levels:
                col_k_yr = raw_arr[yr_mask, k]
                fin_mask = (col_k_yr != fill_value)
                days_finite = int(np.sum(fin_mask))
                days_raw_28671 = int(np.sum(col_k_yr == -28671))
                frac_raw_28671 = float(days_raw_28671 / days_in_yr) if days_in_yr > 0 else 0.0

                if days_finite > 0:
                    dec_vals = col_k_yr[fin_mask].astype(np.float64) * scale_factor + add_offset
                    min_dec = float(np.min(dec_vals))
                    mean_dec = float(np.mean(dec_vals))
                    max_dec = float(np.max(dec_vals))
                else:
                    min_dec = np.nan
                    mean_dec = np.nan
                    max_dec = np.nan

                year_level_rows.append({
                    "year": yr,
                    "column": col_name,
                    "level": k,
                    "depth_m": float(depths[k]),
                    "days_in_year": days_in_yr,
                    "days_finite": days_finite,
                    "days_raw_28671": days_raw_28671,
                    "fraction_raw_28671": frac_raw_28671,
                    "min_decoded_C": min_dec,
                    "mean_decoded_C": mean_dec,
                    "max_decoded_C": max_dec,
                })

    df_year_level_summary = pd.DataFrame(year_level_rows)

    return {
        "level_summary_df": df_level_summary,
        "year_level_summary_df": df_year_level_summary,
        "raw_A": raw_A,
        "raw_B": raw_B,
        "dates": dates,
        "years": years,
        "unique_years": unique_years,
        "depths": depths,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "fill_value": fill_value,
    }


# =============================================================================
# Section D: Wet-Mask History in the Window
# =============================================================================

def audit_wet_mask_history(
    window_path: str,
) -> Dict[str, Any]:
    """Compare finite mask of every day with day 0 over the entire window."""
    with h5netcdf.File(window_path, mode="r") as fw:
        th = np.asarray(fw.variables["thetao"][:], dtype=np.int16)
        fill_val = int(fw.variables["thetao"].attrs.get("_FillValue", -32767))
        times = np.array(fw.variables["time"])
        lats = np.array(fw.variables["latitude"])
        lons = np.array(fw.variables["longitude"])
        depths = np.array(fw.variables["depth"])

    base_dt = datetime.datetime(1950, 1, 1)
    dates = [base_dt + datetime.timedelta(hours=float(h)) for h in times]

    n_times, n_depths, ny, nx = th.shape
    day0_mask = (th[0] != fill_val)

    total_differing_points = 0
    differing_day_records = []

    for t in range(1, n_times):
        m_t = (th[t] != fill_val)
        diff_locs = np.where(m_t != day0_mask)
        n_diff = len(diff_locs[0])
        if n_diff > 0:
            total_differing_points += n_diff
            for i in range(min(5, n_diff)):
                k_i = int(diff_locs[0][i])
                y_i = int(diff_locs[1][i])
                x_i = int(diff_locs[2][i])
                differing_day_records.append({
                    "day_index": t,
                    "date": dates[t].strftime("%Y-%m-%d"),
                    "level": k_i,
                    "depth_m": float(depths[k_i]),
                    "lat": float(lats[y_i]),
                    "lon": float(lons[x_i]),
                    "was_finite_day0": bool(day0_mask[k_i, y_i, x_i]),
                    "is_finite_day_t": bool(m_t[k_i, y_i, x_i]),
                })

    is_completely_time_invariant = bool(total_differing_points == 0)

    return {
        "total_days_evaluated": n_times,
        "is_completely_time_invariant": is_completely_time_invariant,
        "total_differing_points": total_differing_points,
        "differing_days_count": len(set(r["day_index"] for r in differing_day_records)),
        "differing_points_sample": differing_day_records[:20],
    }


# =============================================================================
# Section E: Window-Wide Census & Longest Consecutive Runs
# =============================================================================

def census_window_wide(
    window_path: str,
) -> Dict[str, Any]:
    """Census of raw == -28671 and longest consecutive identical runs across the window."""
    with h5netcdf.File(window_path, mode="r") as fw:
        th = np.asarray(fw.variables["thetao"][:], dtype=np.int16)
        scale_factor = float(fw.variables["thetao"].attrs.get("scale_factor", 0.0007324442267417908))
        add_offset = float(fw.variables["thetao"].attrs.get("add_offset", 21.0))
        fill_val = int(fw.variables["thetao"].attrs.get("_FillValue", -32767))

        lats = np.array(fw.variables["latitude"])
        lons = np.array(fw.variables["longitude"])
        depths = np.array(fw.variables["depth"])

    n_times, n_depths, ny, nx = th.shape

    # 1. Zero code census (-28671 and |decoded| < 1e-3)
    is_28671 = (th == -28671)
    days_28671 = np.sum(is_28671, axis=0)  # (36, 13, 15)

    # Near zero decoded
    fin_mask = (th != fill_val)
    dec = np.where(fin_mask, th.astype(np.float64) * scale_factor + add_offset, np.nan)
    is_near_zero = (np.abs(dec) < 1e-3)
    days_near_zero = np.sum(is_near_zero, axis=0)

    cells_zero_idx = np.where((days_28671 > 0) | (days_near_zero > 0))
    zero_code_rows = []

    for k, y, x in zip(*cells_zero_idx):
        zero_code_rows.append({
            "level": int(k),
            "depth_m": float(depths[k]),
            "lat": float(lats[y]),
            "lon": float(lons[x]),
            "y_idx": int(y),
            "x_idx": int(x),
            "days_raw_28671": int(days_28671[k, y, x]),
            "days_near_zero_decoded": int(days_near_zero[k, y, x]),
        })

    df_zero_cells = pd.DataFrame(zero_code_rows)
    if not df_zero_cells.empty:
        df_zero_cells = df_zero_cells.sort_values(by="days_raw_28671", ascending=False)

    # 2. Longest consecutive runs for all finite cells
    fin_spatial = fin_mask[0]  # shape (36, 13, 15)
    ks, ys, xs = np.where(fin_spatial)

    longest_run_rows = []
    for k, y, x in zip(ks, ys, xs):
        col_ts = th[:, k, y, x]
        run_len, raw_code = compute_longest_run(col_ts)
        dec_code = float(raw_code * scale_factor + add_offset) if raw_code != fill_val else np.nan
        longest_run_rows.append({
            "level": int(k),
            "depth_m": float(depths[k]),
            "lat": float(lats[y]),
            "lon": float(lons[x]),
            "y_idx": int(y),
            "x_idx": int(x),
            "longest_run_days": run_len,
            "raw_code": raw_code,
            "decoded_C": dec_code,
        })

    df_runs = pd.DataFrame(longest_run_rows)
    df_top30_runs = df_runs.sort_values(by="longest_run_days", ascending=False).head(30)

    return {
        "zero_cells_df": df_zero_cells,
        "total_zero_code_cells": len(df_zero_cells),
        "total_zero_code_cell_days": int(np.sum(days_28671)),
        "longest_runs_top30_df": df_top30_runs,
        "total_finite_cells": len(df_runs),
        "days_28671_grid": days_28671,
    }


# =============================================================================
# Section F: Visualization Plots
# =============================================================================

def generate_plots(
    audit_c: Dict[str, Any],
    census_e: Dict[str, Any],
    col_a_idx: Tuple[int, int],
    col_b_idx: Tuple[int, int],
    output_dir: str,
) -> Dict[str, str]:
    """Generate the 3 required diagnostic figures."""
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    raw_A = audit_c["raw_A"]
    raw_B = audit_c["raw_B"]
    dates = audit_c["dates"]
    depths = audit_c["depths"]
    scale_factor = audit_c["scale_factor"]
    add_offset = audit_c["add_offset"]
    fill_value = audit_c["fill_value"]

    # -------------------------------------------------------------------------
    # Plot 1: Decoded Time Series 1993-2020 for A & B at levels 28, 29, 30, 31
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

    levels_to_plot = [28, 29, 30, 31]
    colors = {28: "forestgreen", 29: "dodgerblue", 30: "crimson", 31: "purple"}

    for lev in levels_to_plot:
        # Col A
        col_a_k = raw_A[:, lev]
        fin_a = (col_a_k != fill_value)
        if np.any(fin_a):
            dec_a = np.where(fin_a, col_a_k.astype(np.float64) * scale_factor + add_offset, np.nan)
            ax1.plot(dates, dec_a, label=f"Lev {lev} ({depths[lev]:.1f}m)", color=colors[lev], lw=0.9, alpha=0.85)

        # Col B
        col_b_k = raw_B[:, lev]
        fin_b = (col_b_k != fill_value)
        if np.any(fin_b):
            dec_b = np.where(fin_b, col_b_k.astype(np.float64) * scale_factor + add_offset, np.nan)
            ax2.plot(dates, dec_b, label=f"Lev {lev} ({depths[lev]:.1f}m)", color=colors[lev], lw=0.9, alpha=0.85)

    ax1.set_title("Column A (15.000°N, 50.583°E) Temperature Time Series 1993–2020", fontsize=11, pad=8)
    ax1.set_ylabel("Temperature (°C)", fontsize=10)
    ax1.axhline(0.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_title("Column B (15.000°N, 50.667°E) Temperature Time Series 1993–2020", fontsize=11, pad=8)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    ax2.axhline(0.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    p1_path = os.path.join(output_dir, "ab_timeseries_1993_2020.png")
    fig.tight_layout()
    fig.savefig(p1_path)
    plt.close(fig)
    generated["timeseries_plot"] = p1_path

    # -------------------------------------------------------------------------
    # Plot 2: Heatmap (year x level) of fraction of days raw == -28671 for A & B
    # -------------------------------------------------------------------------
    df_yr = audit_c["year_level_summary_df"]
    years = audit_c["unique_years"]
    target_levels = [27, 28, 29, 30, 31]

    grid_A = np.zeros((len(target_levels), len(years)), dtype=float)
    grid_B = np.zeros((len(target_levels), len(years)), dtype=float)

    for i_k, k in enumerate(target_levels):
        for i_y, yr in enumerate(years):
            sub_a = df_yr[(df_yr["column"] == "A") & (df_yr["level"] == k) & (df_yr["year"] == yr)]
            if not sub_a.empty:
                grid_A[i_k, i_y] = sub_a.iloc[0]["fraction_raw_28671"]
            sub_b = df_yr[(df_yr["column"] == "B") & (df_yr["level"] == k) & (df_yr["year"] == yr)]
            if not sub_b.empty:
                grid_B[i_k, i_y] = sub_b.iloc[0]["fraction_raw_28671"]

    fig, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True)

    im1 = ax_h1.imshow(grid_A, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax_h1.set_title("Column A: Annual Fraction of Days with Raw Code -28671 (0.0°C)", fontsize=11, pad=8)
    ax_h1.set_yticks(range(len(target_levels)))
    ax_h1.set_yticklabels([f"Lev {k} ({depths[k]:.1f}m)" for k in target_levels], fontsize=9)
    ax_h1.set_ylabel("Vertical Level", fontsize=10)

    im2 = ax_h2.imshow(grid_B, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax_h2.set_title("Column B: Annual Fraction of Days with Raw Code -28671 (0.0°C)", fontsize=11, pad=8)
    ax_h2.set_yticks(range(len(target_levels)))
    ax_h2.set_yticklabels([f"Lev {k} ({depths[k]:.1f}m)" for k in target_levels], fontsize=9)
    ax_h2.set_ylabel("Vertical Level", fontsize=10)
    ax_h2.set_xlabel("Year", fontsize=10)
    ax_h2.set_xticks(range(0, len(years), 2))
    ax_h2.set_xticklabels([str(y) for y in years[::2]], rotation=45, fontsize=9)

    cbar = fig.colorbar(im2, ax=[ax_h1, ax_h2], orientation="vertical", pad=0.02, shrink=0.85)
    cbar.set_label("Fraction of Days in Year", fontsize=10)

    p2_path = os.path.join(output_dir, "ab_zero_code_heatmap.png")
    fig.savefig(p2_path, bbox_inches="tight")
    plt.close(fig)
    generated["heatmap_plot"] = p2_path

    # -------------------------------------------------------------------------
    # Plot 3: Map of total days with raw == -28671 summed over all levels
    # -------------------------------------------------------------------------
    days_grid_3d = census_e["days_28671_grid"]  # (36, 13, 15)
    days_sum_2d = np.sum(days_grid_3d, axis=0)   # (13, 15)

    with h5netcdf.File(audit_c["raw_A"].shape and "data/raw/glorys_AB_window/glorys_thetao_AB_window_1993-2020.nc", mode="r") as fw:
        w_lats = np.array(fw.variables["latitude"])
        w_lons = np.array(fw.variables["longitude"])

    fig, ax_m = plt.subplots(figsize=(8, 6), dpi=150)
    X, Y = np.meshgrid(w_lons, w_lats)

    im_m = ax_m.pcolormesh(X, Y, days_sum_2d, cmap="hot_r", shading="auto")
    cbar_m = fig.colorbar(im_m, ax=ax_m, orientation="vertical", pad=0.03)
    cbar_m.set_label("Total Days with Raw Code -28671 (Summed Across Levels)", fontsize=10)

    ya, xa = col_a_idx
    yb, xb = col_b_idx
    ax_m.scatter([w_lons[xa]], [w_lats[ya]], color="cyan", marker="*", s=160, edgecolor="black", label=f"Col A (15.000N, 50.583E: {int(days_sum_2d[ya, xa])} d)")
    ax_m.scatter([w_lons[xb]], [w_lats[yb]], color="lime", marker="o", s=100, edgecolor="black", label=f"Col B (15.000N, 50.667E: {int(days_sum_2d[yb, xb])} d)")

    ax_m.set_title("Window Map: Cumulative Days with Code -28671 (1993–2020)", fontsize=11, pad=8)
    ax_m.set_xlabel("Longitude (°E)", fontsize=10)
    ax_m.set_ylabel("Latitude (°N)", fontsize=10)
    ax_m.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_m.grid(True, linestyle="--", alpha=0.4)

    p3_path = os.path.join(output_dir, "window_zero_code_map.png")
    fig.tight_layout()
    fig.savefig(p3_path)
    plt.close(fig)
    generated["map_plot"] = p3_path

    return generated


# =============================================================================
# Main Pipeline
# =============================================================================

def run_persistence_audit(
    window_path: str,
    full_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Execute complete GLORYS A/B window persistence audit."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GLORYS A/B WINDOW PERSISTENCE AUDIT OVER 1993-2020 (READ-ONLY)")
    print("=" * 80)

    # Initial file integrity
    print("\n--- Verifying Raw File Integrity (Initial Baseline) ---")
    meta_w_before = compute_file_metadata(window_path)
    meta_f_before = compute_file_metadata(full_path)

    expected_full_sha = "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f"

    print(f"Window File : {meta_w_before['filepath']}")
    print(f"  SHA256    : {meta_w_before['sha256']}")
    print(f"  Size      : {meta_w_before['size_bytes']} bytes")
    print(f"  MTime UTC : {meta_w_before['mtime_utc']}")

    print(f"Full File   : {meta_f_before['filepath']}")
    print(f"  SHA256    : {meta_f_before['sha256']}")
    print(f"  Size      : {meta_f_before['size_bytes']} bytes")
    print(f"  MTime UTC : {meta_f_before['mtime_utc']}")

    if meta_f_before["sha256"] != expected_full_sha:
        raise ValueError(
            f"Full-domain file SHA256 mismatch!\nExpected: {expected_full_sha}\nGot: {meta_f_before['sha256']}"
        )
    print("Full-domain SHA256 matches established reference.")

    # Locate Columns A and B by coordinate lookup in window file
    with h5netcdf.File(window_path, mode="r") as fw:
        w_lats = np.array(fw.variables["latitude"])
        w_lons = np.array(fw.variables["longitude"])

    col_a_y = lookup_coord_index(w_lats, 15.0, "latitude", tolerance=1e-3)
    col_a_x = lookup_coord_index(w_lons, 50.583332, "longitude", tolerance=1e-3)
    col_b_y = lookup_coord_index(w_lats, 15.0, "latitude", tolerance=1e-3)
    col_b_x = lookup_coord_index(w_lons, 50.666668, "longitude", tolerance=1e-3)

    col_a_idx = (col_a_y, col_a_x)
    col_b_idx = (col_b_y, col_b_x)

    print(f"\nCoordinate Lookup: Column A located at y={col_a_y}, x={col_a_x} (lat={w_lats[col_a_y]:.4f}, lon={w_lons[col_a_x]:.4f})")
    print(f"Coordinate Lookup: Column B located at y={col_b_y}, x={col_b_x} (lat={w_lats[col_b_y]:.4f}, lon={w_lons[col_b_x]:.4f})")

    # -------------------------------------------------------------------------
    # STEP 1A: Window File Integrity and Structure
    # -------------------------------------------------------------------------
    print("\n--- SECTION A: Window File Integrity and Structure ---")
    struct_info = dump_window_integrity_and_structure(window_path, full_path)
    print(f"Dimensions: {struct_info['dimensions']}")
    print(f"Time range: {struct_info['time_start_utc']} to {struct_info['time_end_utc']} ({struct_info['total_days']} days)")
    print(f"Calendar gaps or duplicates: {struct_info['has_calendar_gaps_or_duplicates']}")
    print(f"Latitude range: {struct_info['latitude_range']} ({struct_info['latitude_count']} points)")
    print(f"Longitude range: {struct_info['longitude_range']} ({struct_info['longitude_count']} points)")
    print(f"Depth range: {struct_info['depth_range_m']} ({struct_info['depth_levels_count']} levels)")
    print(f"Encoding identical to Jan 2019 reference: {struct_info['encoding_identical_to_reference']}")
    if struct_info["encoding_differences"]:
        print(f"Differences: {struct_info['encoding_differences']}")

    # -------------------------------------------------------------------------
    # STEP 1B: Cross-Download Reproducibility
    # -------------------------------------------------------------------------
    print("\n--- SECTION B: Cross-Download Reproducibility ---")
    repro_info = compare_cross_download_reproducibility(window_path, full_path, year=2019, month=1)
    print(f"Cells compared for Jan 2019: {repro_info['cells_compared']}")
    print(f"Bit-for-bit exact match    : {repro_info['is_exact_match']}")
    print(f"Mismatches                 : {repro_info['mismatch_count']}")

    # -------------------------------------------------------------------------
    # STEP 1C: Columns A & B Audit Across Whole Record
    # -------------------------------------------------------------------------
    print("\n--- SECTION C: Per-Column, Per-Level Audit for A & B ---")
    audit_c = audit_columns_ab_levels(window_path, col_a_idx, col_b_idx)

    csv_level_summary = os.path.join(output_dir, "ab_level_summary.csv")
    audit_c["level_summary_df"].to_csv(csv_level_summary, index=False)
    print(f"Per-level summary written to: {csv_level_summary}")

    csv_year_summary = os.path.join(output_dir, "ab_year_level_summary.csv")
    audit_c["year_level_summary_df"].to_csv(csv_year_summary, index=False)
    print(f"Per-year summary written to: {csv_year_summary}")

    print("\nSummary for Column A (selected levels 27-31):")
    df_lev = audit_c["level_summary_df"]
    print(df_lev[(df_lev["column"] == "A") & (df_lev["level"].isin([27, 28, 29, 30, 31]))].to_string(index=False))

    print("\nSummary for Column B (selected levels 27-31):")
    print(df_lev[(df_lev["column"] == "B") & (df_lev["level"].isin([27, 28, 29, 30, 31]))].to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 1D: Wet-Mask History in Window
    # -------------------------------------------------------------------------
    print("\n--- SECTION D: Wet-Mask History in the Window ---")
    mask_hist = audit_wet_mask_history(window_path)
    print(f"Evaluated across all {mask_hist['total_days_evaluated']} days.")
    print(f"100% time-invariant mask : {mask_hist['is_completely_time_invariant']}")
    print(f"Total differing points   : {mask_hist['total_differing_points']}")

    # -------------------------------------------------------------------------
    # STEP 1E: Window-Wide Census & Longest Consecutive Runs
    # -------------------------------------------------------------------------
    print("\n--- SECTION E: Window-Wide Census & Longest Runs ---")
    census_e = census_window_wide(window_path)

    csv_zero_cells = os.path.join(output_dir, "window_zero_code_cells.csv")
    census_e["zero_cells_df"].to_csv(csv_zero_cells, index=False)
    print(f"Window zero-code cells written to: {csv_zero_cells}")
    print(census_e["zero_cells_df"].to_string(index=False))

    csv_longest_runs = os.path.join(output_dir, "window_longest_runs_top30.csv")
    census_e["longest_runs_top30_df"].to_csv(csv_longest_runs, index=False)
    print(f"\nTop 30 longest runs written to: {csv_longest_runs}")
    print(census_e["longest_runs_top30_df"].to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 1F: Visualization Plots
    # -------------------------------------------------------------------------
    print("\n--- SECTION F: Diagnostic Plots ---")
    plot_paths = generate_plots(audit_c, census_e, col_a_idx, col_b_idx, output_dir)
    for k_p, path_p in plot_paths.items():
        print(f"  {k_p:20s}: {path_p}")

    # -------------------------------------------------------------------------
    # Final Integrity Check
    # -------------------------------------------------------------------------
    print("\n--- Verifying Raw File Integrity (Final Check) ---")
    meta_w_after = compute_file_metadata(window_path)
    meta_f_after = compute_file_metadata(full_path)

    w_sha_match = (meta_w_before["sha256"] == meta_w_after["sha256"])
    w_size_match = (meta_w_before["size_bytes"] == meta_w_after["size_bytes"])
    w_mtime_match = (meta_w_before["mtime_utc"] == meta_w_after["mtime_utc"])

    f_sha_match = (meta_f_before["sha256"] == meta_f_after["sha256"])
    f_size_match = (meta_f_before["size_bytes"] == meta_f_after["size_bytes"])
    f_mtime_match = (meta_f_before["mtime_utc"] == meta_f_after["mtime_utc"])

    if not (w_sha_match and w_size_match and w_mtime_match):
        raise RuntimeError(f"Window raw file was mutated! Before: {meta_w_before}, After: {meta_w_after}")
    if not (f_sha_match and f_size_match and f_mtime_match):
        raise RuntimeError(f"Full raw file was mutated! Before: {meta_f_before}, After: {meta_f_after}")

    print("Immutability verified: Both raw NetCDF files are byte-for-byte identical before and after analysis.")

    # -------------------------------------------------------------------------
    # Write JSON Summary
    # -------------------------------------------------------------------------
    json_summary = {
        "metadata": {
            "analysis_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "analysis_script": os.path.abspath(__file__),
            "output_directory": os.path.abspath(output_dir),
        },
        "file_integrity": {
            "window_file": {
                "filepath": meta_w_before["filepath"],
                "sha256_before": meta_w_before["sha256"],
                "sha256_after": meta_w_after["sha256"],
                "sha256_matches": w_sha_match,
                "size_bytes_before": meta_w_before["size_bytes"],
                "size_bytes_after": meta_w_after["size_bytes"],
                "size_matches": w_size_match,
                "mtime_utc_before": meta_w_before["mtime_utc"],
                "mtime_utc_after": meta_w_after["mtime_utc"],
                "mtime_matches": w_mtime_match,
            },
            "full_reference_file": {
                "filepath": meta_f_before["filepath"],
                "sha256_before": meta_f_before["sha256"],
                "sha256_after": meta_f_after["sha256"],
                "sha256_matches": f_sha_match,
                "size_bytes_before": meta_f_before["size_bytes"],
                "size_bytes_after": meta_f_after["size_bytes"],
                "size_matches": f_size_match,
                "mtime_utc_before": meta_f_before["mtime_utc"],
                "mtime_utc_after": meta_f_after["mtime_utc"],
                "mtime_matches": f_mtime_match,
            },
        },
        "section_a_window_structure": {
            "dimensions": struct_info["dimensions"],
            "time_range": {
                "start": struct_info["time_start_utc"],
                "end": struct_info["time_end_utc"],
                "total_days": struct_info["total_days"],
                "expected_days": struct_info["expected_days"],
                "has_gaps_or_duplicates": struct_info["has_calendar_gaps_or_duplicates"],
            },
            "coordinate_ranges": {
                "latitude": struct_info["latitude_range"],
                "longitude": struct_info["longitude_range"],
                "depth_m": struct_info["depth_range_m"],
                "depth_levels_count": struct_info["depth_levels_count"],
            },
            "encoding_attributes": struct_info["thetao_attributes"],
            "encoding_identical_to_jan2019": struct_info["encoding_identical_to_reference"],
            "encoding_differences": struct_info["encoding_differences"],
        },
        "section_b_cross_download_reproducibility": {
            "jan2019_cells_compared": repro_info["cells_compared"],
            "jan2019_exact_match": repro_info["is_exact_match"],
            "jan2019_mismatches_count": repro_info["mismatch_count"],
        },
        "section_c_columns_ab_audit": {
            "column_A_indices": [col_a_y, col_a_x],
            "column_B_indices": [col_b_y, col_b_x],
            "column_A_level_29_zero_code_days": int(df_lev[(df_lev["column"] == "A") & (df_lev["level"] == 29)]["days_raw_28671"].iloc[0]),
            "column_A_level_30_zero_code_days": int(df_lev[(df_lev["column"] == "A") & (df_lev["level"] == 30)]["days_raw_28671"].iloc[0]),
            "column_B_level_29_zero_code_days": int(df_lev[(df_lev["column"] == "B") & (df_lev["level"] == 29)]["days_raw_28671"].iloc[0]),
            "column_B_level_30_zero_code_days": int(df_lev[(df_lev["column"] == "B") & (df_lev["level"] == 30)]["days_raw_28671"].iloc[0]),
            "column_A_level_31_is_finite": bool(df_lev[(df_lev["column"] == "A") & (df_lev["level"] == 31)]["days_finite"].iloc[0] == struct_info["total_days"]),
            "column_B_level_31_is_finite": bool(df_lev[(df_lev["column"] == "B") & (df_lev["level"] == 31)]["days_finite"].iloc[0] > 0),
        },
        "section_d_wet_mask_history": {
            "total_days_evaluated": mask_hist["total_days_evaluated"],
            "is_completely_time_invariant": mask_hist["is_completely_time_invariant"],
            "total_differing_points": mask_hist["total_differing_points"],
        },
        "section_e_window_census": {
            "total_zero_code_cells": census_e["total_zero_code_cells"],
            "total_zero_code_cell_days": census_e["total_zero_code_cell_days"],
            "zero_code_cells": census_e["zero_cells_df"].to_dict(orient="records"),
            "top_longest_runs": census_e["longest_runs_top30_df"].to_dict(orient="records"),
        },
        "section_f_plots": plot_paths,
        "unknown_fields": {
            "physical_or_numerical_trigger_for_2003_freeze": None,
            "physical_or_numerical_trigger_for_2003_freeze_reason": (
                "UNKNOWN / None: Data-validation forensic audit only. Why GLORYS reanalysis froze "
                "Columns A & B at 0.0°C beginning in March 2003 cannot be determined from the reanalysis NetCDF alone."
            ),
        },
    }

    json_path = os.path.join(output_dir, "glorys_ab_persistence_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nScalar findings written to JSON: {json_path}")
    print("=" * 80)
    print("AUDIT COMPLETE.")
    print("=" * 80)

    return json_summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLORYS A/B Window Persistence Audit over 1993-2020 (read-only)."
    )
    parser.add_argument(
        "--window-path",
        type=str,
        default="data/raw/glorys_AB_window/glorys_thetao_AB_window_1993-2020.nc",
        help="Path to GLORYS A/B window NetCDF file.",
    )
    parser.add_argument(
        "--full-path",
        type=str,
        default="data/raw/glorys/glorys_thetao_2019-01.nc",
        help="Path to GLORYS full-domain January 2019 NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/diagnostics/glorys_ab_persistence_audit",
        help="Directory to save output files and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_persistence_audit(
        window_path=args.window_path,
        full_path=args.full_path,
        output_dir=args.output_dir,
    )
