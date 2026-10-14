#!/usr/bin/env python3
"""
GLORYS Defect Census
Data-validation read-only forensic script for GLORYS NetCDF products.

Investigates whether the column A/B anomaly is an isolated defect or
one example of a wider class:
- Section A: A/B vertical structure (Day 0 and Day 30) & compact wet/dry maps
- Section B: Frozen-cell census (temporal invariability across the month)
- Section C: Local-consistency census (day-mean decoded spatial residuals)
- Section D: Extremes (coldest and warmest cells per level)
- Section E: Forensic plots
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

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.diagnostics.glorys_encoding_forensics import (
    compute_file_metadata,
    RawNetCDFReader,
    lookup_nearest_index,
    check_mask_time_invariance,
)


EXPECTED_SHA256 = "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f"


# =============================================================================
# Section A: A/B Vertical Structure and Wet/Dry Maps
# =============================================================================

def analyze_ab_vertical_structure(
    reader: RawNetCDFReader,
    scale_factor: float,
    add_offset: float,
    fill_value: int,
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], str]:
    """Extract Day 0 and Day 30 vertical structure for A and B, plus wet/dry maps for levels 20-35."""
    dims = reader.get_dimensions()
    n_time = dims["time"]
    n_depth = dims["depth"]
    depths = reader.get_raw_variable("depth")
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    days_to_extract = [0, n_time - 1]  # Day 0 and Day 30
    profile_rows = []

    for d in days_to_extract:
        day_raw = reader.get_raw_slice("thetao", (d, slice(None), slice(None), slice(None)))

        for col_name, y_idx, x_idx in [("A", lat_A_idx, lon_A_idx), ("B", lat_B_idx, lon_B_idx)]:
            col_lat = float(lats[y_idx])
            col_lon = float(lons[x_idx])

            for k in range(n_depth):
                depth_m = float(depths[k])
                raw_code = int(day_raw[k, y_idx, x_idx])
                is_fill = bool(raw_code == fill_value)
                decoded_val = float(np.float64(raw_code) * scale + offset) if not is_fill else np.nan

                profile_rows.append({
                    "day": d,
                    "column": col_name,
                    "lat": col_lat,
                    "lon": col_lon,
                    "lat_index": y_idx,
                    "lon_index": x_idx,
                    "level_index": k,
                    "depth_m": depth_m,
                    "raw_code": raw_code,
                    "is_fill": is_fill,
                    "decoded_C": decoded_val,
                })

    profiles_df = pd.DataFrame(profile_rows)

    # Wet/Dry maps around A and B for levels 20 to 35 on Day 0
    # Center A: (lat_A_idx, lon_A_idx), Center B: (lat_B_idx, lon_B_idx)
    wet_dry_maps = {}
    map_text_lines = []
    map_text_lines.append("WET/DRY MAPS (W = finite, . = fill) FOR LEVELS 20 TO 35 AROUND A & B")
    map_text_lines.append("=" * 70)

    # Read Day 0 slice for levels 20..35
    day0_deep = reader.get_raw_slice("thetao", (0, slice(20, 36), slice(None), slice(None)))

    for k_idx in range(20, min(36, n_depth)):
        rel_k = k_idx - 20
        depth_m = float(depths[k_idx])

        # 3x3 for A: y in [lat_A_idx-1 .. lat_A_idx+1], x in [lon_A_idx-1 .. lon_A_idx+1]
        grid_A = []
        for dy in [-1, 0, 1]:
            row_chars = []
            for dx in [-1, 0, 1]:
                ny, nx = lat_A_idx + dy, lon_A_idx + dx
                code = day0_deep[rel_k, ny, nx]
                row_chars.append("W" if code != fill_value else ".")
            grid_A.append("".join(row_chars))

        # 3x3 for B: y in [lat_B_idx-1 .. lat_B_idx+1], x in [lon_B_idx-1 .. lon_B_idx+1]
        grid_B = []
        for dy in [-1, 0, 1]:
            row_chars = []
            for dx in [-1, 0, 1]:
                ny, nx = lat_B_idx + dy, lon_B_idx + dx
                code = day0_deep[rel_k, ny, nx]
                row_chars.append("W" if code != fill_value else ".")
            grid_B.append("".join(row_chars))

        # Combined 3x4 window (y in [119..121], x in [66..69]):
        # Column 66: West of A, 67: Col A, 68: Col B, 69: East of B
        grid_combined = []
        for dy in [-1, 0, 1]:
            row_chars = []
            for x in range(lon_A_idx - 1, lon_B_idx + 2):
                ny = lat_A_idx + dy
                code = day0_deep[rel_k, ny, x]
                row_chars.append("W" if code != fill_value else ".")
            grid_combined.append("".join(row_chars))

        wet_dry_maps[k_idx] = {
            "level_index": k_idx,
            "depth_m": depth_m,
            "grid_A": grid_A,
            "grid_B": grid_B,
            "grid_combined_3x4": grid_combined,
        }

        map_text_lines.append(f"Level {k_idx:2d} ({depth_m:7.2f} m):  Combined [Col A=x:67, Col B=x:68]")
        for r_idx, r_str in enumerate(grid_combined):
            y_offset = ["(lat 15.083 N)", "(lat 15.000 N)", "(lat 14.917 N)"][r_idx]
            map_text_lines.append(f"    y={lat_A_idx + r_idx - 1} {y_offset:<16}  | {r_str} |")
        map_text_lines.append("")

    return profiles_df, wet_dry_maps, "\n".join(map_text_lines)


# =============================================================================
# Section B: Frozen-Cell Census (Whole File)
# =============================================================================

def census_frozen_cells(
    reader: RawNetCDFReader,
    scale_factor: float,
    add_offset: float,
    fill_value: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Check every finite cell at every level for temporal invariability across all 31 days."""
    dims = reader.get_dimensions()
    n_depth = dims["depth"]
    n_time = dims["time"]
    depths = reader.get_raw_variable("depth")
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    by_level_rows = []
    all_frozen_cells = []
    frozen_codes_counts = {}

    total_finite_all = 0
    total_frozen_all = 0
    total_at_most_3_all = 0

    for k in range(n_depth):
        depth_m = float(depths[k])
        # Read full month slice for level k: shape (31, 301, 721)
        raw_level = reader.get_raw_slice("thetao", (slice(None), k, slice(None), slice(None)))

        # Finite mask is time-invariant
        mask = (raw_level[0] != fill_value)
        n_finite = int(np.sum(mask))
        total_finite_all += n_finite

        if n_finite == 0:
            by_level_rows.append({
                "level_index": k,
                "depth_m": depth_m,
                "finite_cells": 0,
                "frozen_cells": 0,
                "at_most_3_distinct": 0,
            })
            continue

        # Extract finite time-series: shape (31, n_finite)
        finite_ts = raw_level[:, mask]

        # Vectorized calculation of number of distinct raw codes across the 31 days
        sorted_ts = np.sort(finite_ts, axis=0)
        n_distinct = 1 + np.sum(sorted_ts[1:] != sorted_ts[:-1], axis=0)

        is_frozen = (n_distinct == 1)
        is_at_most_3 = (n_distinct <= 3)

        n_frozen = int(np.sum(is_frozen))
        n_at_most_3 = int(np.sum(is_at_most_3))

        total_frozen_all += n_frozen
        total_at_most_3_all += n_at_most_3

        # Record frozen cells details
        if n_frozen > 0:
            frozen_indices = np.where(is_frozen)[0]
            # Map back to 2D grid coordinates
            grid_y, grid_x = np.where(mask)
            frozen_y = grid_y[frozen_indices]
            frozen_x = grid_x[frozen_indices]
            frozen_raw_codes = finite_ts[0, frozen_indices]

            for y, x, code in zip(frozen_y, frozen_x, frozen_raw_codes):
                code_int = int(code)
                dec_c = float(np.float64(code_int) * scale + offset)
                frozen_codes_counts[code_int] = frozen_codes_counts.get(code_int, 0) + 1

                all_frozen_cells.append({
                    "lat": float(lats[y]),
                    "lon": float(lons[x]),
                    "lat_index": int(y),
                    "lon_index": int(x),
                    "level_index": int(k),
                    "depth_m": depth_m,
                    "raw_code": code_int,
                    "decoded_C": dec_c,
                })

        by_level_rows.append({
            "level_index": k,
            "depth_m": depth_m,
            "finite_cells": n_finite,
            "frozen_cells": n_frozen,
            "at_most_3_distinct": n_at_most_3,
        })

    by_level_df = pd.DataFrame(by_level_rows)
    frozen_list_df = pd.DataFrame(all_frozen_cells)

    # Top 10 most common frozen codes
    sorted_frozen_codes = sorted(frozen_codes_counts.items(), key=lambda item: item[1], reverse=True)
    top_10_frozen_codes = [
        {"raw_code": code, "count": cnt, "decoded_C": float(np.float64(code) * scale + offset)}
        for code, cnt in sorted_frozen_codes[:10]
    ]

    summary = {
        "total_finite_cells": total_finite_all,
        "total_frozen_cells": total_frozen_all,
        "total_at_most_3_distinct": total_at_most_3_all,
        "top_10_frozen_codes": top_10_frozen_codes,
        "all_frozen_cells_count": len(all_frozen_cells),
        "is_frozen_count_le_300": bool(len(all_frozen_cells) <= 300),
    }

    return by_level_df, frozen_list_df, summary


# =============================================================================
# Section C: Local-Consistency Census (Day-Mean Decoded Field)
# =============================================================================

def census_local_consistency(
    reader: RawNetCDFReader,
    scale_factor: float,
    add_offset: float,
    fill_value: int,
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[int, np.ndarray]]:
    """Compute 31-day mean decoded fields, 8-neighbor spatial consistency residuals, and extremes."""
    dims = reader.get_dimensions()
    n_depth = dims["depth"]
    depths = reader.get_raw_variable("depth")
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    consistency_by_level = []
    top_10_by_level = []
    mean_decoded_fields = {}
    summary_dict = {}

    for k in range(n_depth):
        depth_m = float(depths[k])
        # Read full month slice for level k: shape (31, 301, 721)
        raw_level = reader.get_raw_slice("thetao", (slice(None), k, slice(None), slice(None)))

        mask = (raw_level[0] != fill_value)
        n_finite = int(np.sum(mask))

        if n_finite == 0:
            mean_decoded_fields[k] = np.full(mask.shape, np.nan, dtype=np.float64)
            consistency_by_level.append({
                "level_index": k,
                "depth_m": depth_m,
                "finite_cells": 0,
                "evaluable_cells": 0,
                "unevaluable_finite_cells": 0,
                "median_residual_C": np.nan,
                "mad_residual_C": np.nan,
                "residual_gt_5_mad": 0,
                "residual_gt_10_mad": 0,
                "residual_gt_20_mad": 0,
            })
            continue

        # 31-day mean of decoded values
        # Because decoded = raw * scale + offset, mean(decoded) = mean(raw) * scale + offset
        raw_mean = np.mean(raw_level.astype(np.float64), axis=0)
        mean_decoded = np.where(mask, raw_mean * scale + offset, np.nan)
        mean_decoded_fields[k] = mean_decoded

        # 8-neighbour array construction with NaN padding
        n_lat, n_lon = mask.shape
        pad = np.pad(mean_decoded, 1, constant_values=np.nan)
        nbrs = np.zeros((8, n_lat, n_lon), dtype=np.float64)

        idx = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nbrs[idx] = pad[1 + dy : 1 + dy + n_lat, 1 + dx : 1 + dx + n_lon]
                idx += 1

        finite_nbr_counts = np.sum(np.isfinite(nbrs), axis=0)

        # Evaluable condition: finite cell with >= 3 finite neighbours
        eval_mask = mask & (finite_nbr_counts >= 3)
        uneval_mask = mask & (finite_nbr_counts < 3)

        n_evaluable = int(np.sum(eval_mask))
        n_unevaluable = int(np.sum(uneval_mask))

        if n_evaluable > 0:
            nbr_med = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
            nbr_med[eval_mask] = np.nanmedian(nbrs[:, eval_mask], axis=0)
            residuals = np.where(eval_mask, mean_decoded - nbr_med, np.nan)

            eval_res = residuals[eval_mask]
            med_res = float(np.median(eval_res))
            # MAD = median(|residual - median_residual|)
            mad_res = float(np.median(np.abs(eval_res - med_res)))

            # Counts of |residual| > 5, 10, 20 MAD
            abs_res = np.abs(eval_res)
            cnt_gt_5_mad = int(np.sum(abs_res > 5.0 * mad_res)) if mad_res > 0 else 0
            cnt_gt_10_mad = int(np.sum(abs_res > 10.0 * mad_res)) if mad_res > 0 else 0
            cnt_gt_20_mad = int(np.sum(abs_res > 20.0 * mad_res)) if mad_res > 0 else 0

            # Find 10 largest |residual| cells
            eval_y, eval_x = np.where(eval_mask)
            eval_abs_res = np.abs(eval_res)
            top_order = np.argsort(eval_abs_res)[::-1][:10]

            for rank_idx, top_i in enumerate(top_order):
                y = int(eval_y[top_i])
                x = int(eval_x[top_i])
                res_val = float(eval_res[top_i])
                cell_val = float(mean_decoded[y, x])
                med_val = float(nbr_med[y, x])

                is_A = (y == lat_A_idx and x == lon_A_idx)
                is_B = (y == lat_B_idx and x == lon_B_idx)
                col_tag = "Col A" if is_A else ("Col B" if is_B else "")

                top_10_by_level.append({
                    "level_index": k,
                    "depth_m": depth_m,
                    "rank": rank_idx + 1,
                    "lat": float(lats[y]),
                    "lon": float(lons[x]),
                    "lat_index": y,
                    "lon_index": x,
                    "cell_value_C": cell_val,
                    "neighbor_median_C": med_val,
                    "residual_C": res_val,
                    "abs_residual_C": abs(res_val),
                    "mad_multiple": (abs(res_val) / mad_res) if mad_res > 0 else np.nan,
                    "column_tag": col_tag,
                })
        else:
            med_res = np.nan
            mad_res = np.nan
            cnt_gt_5_mad = 0
            cnt_gt_10_mad = 0
            cnt_gt_20_mad = 0

        # Status of Column A and Column B at level k
        def get_col_status(y_idx, x_idx):
            if not mask[y_idx, x_idx]:
                return "FILL/LAND"
            if eval_mask[y_idx, x_idx]:
                return f"EVALUABLE (res={residuals[y_idx, x_idx]:.6f} C)"
            return "UNEVALUABLE (<3 finite neighbors)"

        status_A = get_col_status(lat_A_idx, lon_A_idx)
        status_B = get_col_status(lat_B_idx, lon_B_idx)

        consistency_by_level.append({
            "level_index": k,
            "depth_m": depth_m,
            "finite_cells": n_finite,
            "evaluable_cells": n_evaluable,
            "unevaluable_finite_cells": n_unevaluable,
            "median_residual_C": med_res,
            "mad_residual_C": mad_res,
            "residual_gt_5_mad": cnt_gt_5_mad,
            "residual_gt_10_mad": cnt_gt_10_mad,
            "residual_gt_20_mad": cnt_gt_20_mad,
            "status_col_A": status_A,
            "status_col_B": status_B,
        })

    consistency_df = pd.DataFrame(consistency_by_level)
    top10_df = pd.DataFrame(top_10_by_level)

    summary_dict = {
        "levels_analyzed": n_depth,
        "total_evaluable_cells_across_levels": int(consistency_df["evaluable_cells"].sum()),
        "total_unevaluable_cells_across_levels": int(consistency_df["unevaluable_finite_cells"].sum()),
        "total_gt_10_mad_across_levels": int(consistency_df["residual_gt_10_mad"].sum()),
    }

    return consistency_df, top10_df, summary_dict, mean_decoded_fields


# =============================================================================
# Section D: Extremes
# =============================================================================

def census_extremes(
    mean_decoded_fields: Dict[int, np.ndarray],
    reader: RawNetCDFReader,
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Determine minimum and maximum day-mean decoded temperatures per level, plus 5 coldest/warmest."""
    dims = reader.get_dimensions()
    n_depth = dims["depth"]
    depths = reader.get_raw_variable("depth")
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")

    rows = []
    level_extremes_summary = {}

    for k in range(n_depth):
        depth_m = float(depths[k])
        field = mean_decoded_fields[k]

        mask = np.isfinite(field)
        n_finite = int(np.sum(mask))

        if n_finite == 0:
            rows.append({
                "level_index": k,
                "depth_m": depth_m,
                "finite_cells": 0,
                "min_mean_C": np.nan,
                "max_mean_C": np.nan,
                "contains_col_A_coldest5": False,
                "contains_col_B_coldest5": False,
                "contains_col_A_warmest5": False,
                "contains_col_B_warmest5": False,
                "coldest_5_details": "[]",
                "warmest_5_details": "[]",
            })
            continue

        finite_vals = field[mask]
        y_indices, x_indices = np.where(mask)

        # Min and Max
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))

        # 5 coldest
        coldest_order = np.argsort(finite_vals)[:5]
        coldest_details = []
        coldest_coords = set()
        for idx in coldest_order:
            y = int(y_indices[idx])
            x = int(x_indices[idx])
            val = float(finite_vals[idx])
            coldest_coords.add((y, x))
            coldest_details.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "lat_index": y,
                "lon_index": x,
                "value_C": val,
            })

        # 5 warmest
        warmest_order = np.argsort(finite_vals)[::-1][:5]
        warmest_details = []
        warmest_coords = set()
        for idx in warmest_order:
            y = int(y_indices[idx])
            x = int(x_indices[idx])
            val = float(finite_vals[idx])
            warmest_coords.add((y, x))
            warmest_details.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "lat_index": y,
                "lon_index": x,
                "value_C": val,
            })

        has_A_cold = (lat_A_idx, lon_A_idx) in coldest_coords
        has_B_cold = (lat_B_idx, lon_B_idx) in coldest_coords
        has_A_warm = (lat_A_idx, lon_A_idx) in warmest_coords
        has_B_warm = (lat_B_idx, lon_B_idx) in warmest_coords

        rows.append({
            "level_index": k,
            "depth_m": depth_m,
            "finite_cells": n_finite,
            "min_mean_C": min_val,
            "max_mean_C": max_val,
            "contains_col_A_coldest5": has_A_cold,
            "contains_col_B_coldest5": has_B_cold,
            "contains_col_A_warmest5": has_A_warm,
            "contains_col_B_warmest5": has_B_warm,
            "coldest_5_details": json.dumps(coldest_details),
            "warmest_5_details": json.dumps(warmest_details),
        })

        level_extremes_summary[f"level_{k}"] = {
            "level_index": k,
            "depth_m": depth_m,
            "min_mean_C": min_val,
            "max_mean_C": max_val,
            "contains_col_A_coldest5": has_A_cold,
            "contains_col_B_coldest5": has_B_cold,
            "contains_col_A_warmest5": has_A_warm,
            "contains_col_B_warmest5": has_B_warm,
            "coldest_5": coldest_details,
            "warmest_5": warmest_details,
        }

    extremes_df = pd.DataFrame(rows)
    return extremes_df, level_extremes_summary


# =============================================================================
# Section E: Plots
# =============================================================================

def generate_defect_plots(
    frozen_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    output_dir: str,
) -> List[str]:
    """Generate forensic diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # Plot 1: Frozen-cell count per level
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = frozen_df["level_index"]
    depths = frozen_df["depth_m"]
    frozen_counts = frozen_df["frozen_cells"]

    bars = ax.bar(levels, frozen_counts, color="steelblue", edgecolor="black", alpha=0.85)

    # Highlight levels 29 and 30 where A & B reside
    for l, bar in zip(levels, bars):
        if l in (29, 30):
            bar.set_color("crimson")
            bar.set_edgecolor("black")

    ax.set_xlabel("Depth Level Index (0 to 35)")
    ax.set_ylabel("Frozen Cell Count (Raw Code Invariant on all 31 Days)")
    ax.set_title("GLORYS thetao (Jan 2019): Frozen-Cell Count per Vertical Level")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xticks(levels[::2])

    # Annotate non-zero counts
    for l, cnt in zip(levels, frozen_counts):
        if cnt > 0:
            ax.annotate(f"{cnt}", (l, cnt), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, weight="bold")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "frozen_cells_by_level.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    generated.append(p1)

    # Plot 2: Unevaluable-finite-cell count and count of |residual| > 10 MAD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(levels, consistency_df["unevaluable_finite_cells"], color="darkorange", marker="o", linewidth=2.0, label="Unevaluable Finite Cells (<3 Finite Neighbors)")
    ax1.set_ylabel("Unevaluable Cell Count")
    ax1.set_title("GLORYS thetao (Jan 2019): Spatial Consistency Diagnostics Across Levels")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(levels, consistency_df["residual_gt_10_mad"], color="firebrick", marker="s", linewidth=2.0, label="Cells with |Residual| > 10 MAD (Descriptive Multiplier)")
    ax2.plot(levels, consistency_df["residual_gt_5_mad"], color="purple", linestyle="--", marker="^", alpha=0.7, label="Cells with |Residual| > 5 MAD")
    ax2.set_xlabel("Depth Level Index (0 to 35)")
    ax2.set_ylabel("Count of Outlier Multiples")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p2 = os.path.join(output_dir, "local_consistency_by_level.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    generated.append(p2)

    return generated


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_defect_census(input_file: str, output_dir: str) -> Dict[str, Any]:
    """Execute complete GLORYS defect census."""
    print("\n" + "=" * 80)
    print("STEP 0: INPUT FILE CHECKSUM VERIFICATION (BEFORE ANALYSIS)")
    print("=" * 80)
    meta_before = compute_file_metadata(input_file)
    print(f"File: {meta_before['filepath']}")
    print(f"Size: {meta_before['size_bytes']} bytes")
    print(f"MTime (UTC): {meta_before['mtime_utc']}")
    print(f"SHA256 (before): {meta_before['sha256']}")
    print(f"Expected SHA256: {EXPECTED_SHA256}")

    if meta_before["sha256"] != EXPECTED_SHA256:
        raise RuntimeError(
            f"STOP: Input file SHA256 ({meta_before['sha256']}) does not match expected ({EXPECTED_SHA256})!"
        )

    os.makedirs(output_dir, exist_ok=True)

    with RawNetCDFReader(input_file) as reader:
        # Check mask time-invariance upfront
        print("\nVerifying mask time-invariance upfront...")
        thetao_attrs = reader.get_var_attrs("thetao")
        fill_val = int(thetao_attrs.get("_FillValue", -32767))
        scale_fac = float(thetao_attrs.get("scale_factor", 1.0))
        add_off = float(thetao_attrs.get("add_offset", 0.0))

        mask_inv = check_mask_time_invariance(reader, fill_value=fill_val)
        print(f"Mask is strictly time-invariant: {mask_inv['is_time_invariant']} (diff points: {mask_inv['total_differing_points']})")
        if not mask_inv["is_time_invariant"]:
            raise RuntimeError(
                f"STOP: Finite mask is NOT time-invariant! Differing points: {mask_inv['total_differing_points']}"
            )

        # Locate Columns A and B
        lats = reader.get_raw_variable("latitude")
        lons = reader.get_raw_variable("longitude")
        lat_A_idx, exact_lat_A = lookup_nearest_index(lats, 15.000, "lat_A", tolerance=1e-3)
        lon_A_idx, exact_lon_A = lookup_nearest_index(lons, 50.583, "lon_A", tolerance=1e-3)
        lat_B_idx, exact_lat_B = lookup_nearest_index(lats, 15.000, "lat_B", tolerance=1e-3)
        lon_B_idx, exact_lon_B = lookup_nearest_index(lons, 50.667, "lon_B", tolerance=1e-3)

        # Section A: A/B vertical structure & wet/dry maps
        print("\n" + "=" * 80)
        print("SECTION A: A/B VERTICAL STRUCTURE (DAY 0 & DAY 30) AND WET/DRY MAPS")
        print("=" * 80)
        dims = reader.get_dimensions()
        n_time = dims["time"]
        n_depth = dims["depth"]

        profiles_df, wet_dry_maps, map_text = analyze_ab_vertical_structure(
            reader=reader,
            scale_factor=scale_fac,
            add_offset=add_off,
            fill_value=fill_val,
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
        )
        ab_profiles_path = os.path.join(output_dir, "ab_profiles.csv")
        profiles_df.to_csv(ab_profiles_path, index=False)
        print(f"A/B profiles written to: {ab_profiles_path}")

        print("\n--- Column A & B Vertical Structure Table (Day 0 & Day 30) ---")
        for day_val in [0, n_time - 1]:
            print(f"\nDay {day_val}:")
            print(f"{'Lvl':<4} {'Depth (m)':<11} {'Col A Raw':<10} {'Col A Decoded':<15} {'Col A Fill':<11} {'Col B Raw':<10} {'Col B Decoded':<15} {'Col B Fill':<11}")
            p_day = profiles_df[profiles_df["day"] == day_val]
            p_A = p_day[p_day["column"] == "A"].set_index("level_index")
            p_B = p_day[p_day["column"] == "B"].set_index("level_index")
            for k in range(n_depth):
                rA = p_A.loc[k]
                rB = p_B.loc[k]
                decA = f"{rA['decoded_C']:.6f} C" if not rA['is_fill'] else "FILL"
                decB = f"{rB['decoded_C']:.6f} C" if not rB['is_fill'] else "FILL"
                print(f"{k:<4d} {rA['depth_m']:<11.2f} {rA['raw_code']:<10d} {decA:<15} {str(rA['is_fill']):<11} {rB['raw_code']:<10d} {decB:<15} {str(rB['is_fill']):<11}")

        print("\n" + map_text)

        # Section B: Frozen-cell census
        print("\n" + "=" * 80)
        print("SECTION B: FROZEN-CELL CENSUS (WHOLE FILE)")
        print("=" * 80)
        frozen_by_level_df, frozen_list_df, frozen_summary = census_frozen_cells(
            reader=reader,
            scale_factor=scale_fac,
            add_offset=add_off,
            fill_value=fill_val,
        )
        frozen_by_level_path = os.path.join(output_dir, "frozen_cells_by_level.csv")
        frozen_list_path = os.path.join(output_dir, "frozen_cells_list.csv")
        frozen_by_level_df.to_csv(frozen_by_level_path, index=False)
        frozen_list_df.to_csv(frozen_list_path, index=False)
        print(f"Frozen cells by level written to: {frozen_by_level_path}")
        print(f"Frozen cells list written to: {frozen_list_path}")
        print(f"Total finite cells across domain: {frozen_summary['total_finite_cells']:,}")
        print(f"Total frozen cells (raw code invariant all 31 days): {frozen_summary['total_frozen_cells']}")
        print(f"Total cells with at most 3 distinct raw codes: {frozen_summary['total_at_most_3_distinct']:,}")
        print("\n10 Most Common Frozen Raw Codes:")
        for idx, item in enumerate(frozen_summary["top_10_frozen_codes"]):
            print(f"  {idx+1:2d}. Raw code {item['raw_code']:6d} -> {item['decoded_C']:12.6f} °C (Count: {item['count']})")

        if frozen_summary["total_frozen_cells"] <= 300:
            print(f"\nListing all {frozen_summary['total_frozen_cells']} frozen cells:")
            print(f"{'Lat':<8} {'Lon':<8} {'Lvl':<5} {'Depth (m)':<11} {'Raw Code':<10} {'Decoded (°C)':<14}")
            for _, r in frozen_list_df.iterrows():
                print(f"{r['lat']:<8.3f} {r['lon']:<8.3f} {int(r['level_index']):<5d} {r['depth_m']:<11.2f} {int(r['raw_code']):<10d} {r['decoded_C']:<14.6f}")

        # Section C: Local-consistency census
        print("\n" + "=" * 80)
        print("SECTION C: LOCAL-CONSISTENCY CENSUS (DAY-MEAN DECODED FIELD)")
        print("=" * 80)
        consistency_df, top10_df, consistency_summary, mean_decoded_fields = census_local_consistency(
            reader=reader,
            scale_factor=scale_fac,
            add_offset=add_off,
            fill_value=fill_val,
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
        )
        consistency_path = os.path.join(output_dir, "local_consistency_by_level.csv")
        top10_path = os.path.join(output_dir, "local_consistency_top10.csv")
        consistency_df.to_csv(consistency_path, index=False)
        top10_df.to_csv(top10_path, index=False)
        print(f"Local consistency by level written to: {consistency_path}")
        print(f"Local consistency top 10 written to: {top10_path}")
        print(f"Total evaluable cells across levels: {consistency_summary['total_evaluable_cells_across_levels']:,}")
        print(f"Total unevaluable finite cells across levels: {consistency_summary['total_unevaluable_cells_across_levels']:,}")
        print(f"Total cells with |residual| > 10 MAD: {consistency_summary['total_gt_10_mad_across_levels']:,}")

        # Section D: Extremes
        print("\n" + "=" * 80)
        print("SECTION D: EXTREMES (COLDEST AND WARMEST CELLS PER LEVEL)")
        print("=" * 80)
        extremes_df, extremes_summary = census_extremes(
            mean_decoded_fields=mean_decoded_fields,
            reader=reader,
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
        )
        extremes_path = os.path.join(output_dir, "level_extremes.csv")
        extremes_df.to_csv(extremes_path, index=False)
        print(f"Level extremes written to: {extremes_path}")

        # Check where A or B appear in extremes
        print("\nColumn A/B Presence in Extremes:")
        for k in range(dims["depth"]):
            row = extremes_df.loc[extremes_df["level_index"] == k].iloc[0]
            if row["contains_col_A_coldest5"] or row["contains_col_B_coldest5"] or row["contains_col_A_warmest5"] or row["contains_col_B_warmest5"]:
                print(f"  Level {k:2d} ({row['depth_m']:7.2f} m): "
                      f"Col A Coldest-5: {row['contains_col_A_coldest5']}, Col B Coldest-5: {row['contains_col_B_coldest5']} | "
                      f"Col A Warmest-5: {row['contains_col_A_warmest5']}, Col B Warmest-5: {row['contains_col_B_warmest5']}")

        # Section E: Plots
        print("\n" + "=" * 80)
        print("SECTION E: GENERATING FORENSIC PLOTS")
        print("=" * 80)
        plots = generate_defect_plots(
            frozen_df=frozen_by_level_df,
            consistency_df=consistency_df,
            output_dir=output_dir,
        )
        for p in plots:
            print(f"Generated plot: {p}")

    # Step Final: Checksum verification after analysis
    print("\n" + "=" * 80)
    print("STEP FINAL: INPUT FILE INTEGRITY VERIFICATION (AFTER ANALYSIS)")
    print("=" * 80)
    meta_after = compute_file_metadata(input_file)
    print(f"SHA256 (before): {meta_before['sha256']}")
    print(f"SHA256 (after):  {meta_after['sha256']}")
    print(f"Size (before):   {meta_before['size_bytes']} bytes")
    print(f"Size (after):    {meta_after['size_bytes']} bytes")
    print(f"MTime (before):  {meta_before['mtime_utc']}")
    print(f"MTime (after):   {meta_after['mtime_utc']}")

    sha_match = (meta_before["sha256"] == meta_after["sha256"])
    sz_match = (meta_before["size_bytes"] == meta_after["size_bytes"])
    mt_match = (meta_before["mtime_utc"] == meta_after["mtime_utc"])

    if not sha_match or not sz_match or not mt_match:
        raise RuntimeError("FATAL: Input raw data file was modified during analysis!")

    # Write summary JSON
    json_path = os.path.join(output_dir, "glorys_defect_census.json")
    json_data = {
        "analysis_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_file_integrity": {
            "filepath": meta_before["filepath"],
            "expected_sha256": EXPECTED_SHA256,
            "sha256_before": meta_before["sha256"],
            "sha256_after": meta_after["sha256"],
            "sha256_match": sha_match,
            "size_bytes_before": meta_before["size_bytes"],
            "size_bytes_after": meta_after["size_bytes"],
            "size_bytes_match": sz_match,
            "mtime_utc_before": meta_before["mtime_utc"],
            "mtime_utc_after": meta_after["mtime_utc"],
            "mtime_match": mt_match,
        },
        "unknown_fields": {
            "physical_origin_of_frozen_cells": None,
            "physical_origin_of_frozen_cells_reason": "UNKNOWN: Data-validation census only. Whether frozen or high-residual cells represent numerical bottom-boundary artifacts, land mask leakage, or physical stagnation cannot be determined from file alone.",
            "defect_labeling_rule": None,
            "defect_labeling_rule_reason": "UNKNOWN: By protocol, this analysis only gathers evidence and strictly does not label any cell an error or define mask rules.",
        },
        "section_A_ab_structure": {
            "column_A": {"lat": 15.0, "lon": 50.583332, "indices": [lat_A_idx, lon_A_idx]},
            "column_B": {"lat": 15.0, "lon": 50.666668, "indices": [lat_B_idx, lon_B_idx]},
            "wet_dry_maps": wet_dry_maps,
        },
        "section_B_frozen_cells": frozen_summary,
        "section_C_local_consistency": consistency_summary,
        "section_D_extremes": {
            "presence_in_coldest_5": {
                "col_A_levels": [v["level_index"] for v in extremes_summary.values() if v["contains_col_A_coldest5"]],
                "col_B_levels": [v["level_index"] for v in extremes_summary.values() if v["contains_col_B_coldest5"]],
            },
            "presence_in_warmest_5": {
                "col_A_levels": [v["level_index"] for v in extremes_summary.values() if v["contains_col_A_warmest5"]],
                "col_B_levels": [v["level_index"] for v in extremes_summary.values() if v["contains_col_B_warmest5"]],
            },
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nForensic defect census JSON written to: {json_path}")

    return json_data


def main():
    parser = argparse.ArgumentParser(description="GLORYS Defect Census")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/glorys/glorys_thetao_2019-01.nc",
        help="Path to GLORYS NetCDF input file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/diagnostics/glorys_defect_census",
        help="Path to output directory for census artifacts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_defect_census(args.input, args.output_dir)


if __name__ == "__main__":
    main()
