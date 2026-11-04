"""
GLORYS Hot-End Audit (2010-2020): Cells Above valid_max, SST Comparison, and Target Impact.

Audits GLORYS 1993-2020 reanalysis (2010-2020 focus) for:
- Cells exceeding declared valid_max (raw > 21306, decoded > 36.6055 C).
- Manifest reconciliation and anomalous zero-code investigation (2016-12).
- Native hot cell-days inventory and vertical structure (20 hottest and 20 random).
- Comparison with OSTIA foundation SST (nearest-neighbour ocean cells).
- Method-A target footprint impact and whole-record target extremes by depth.

STRICTLY READ-ONLY on data. NO NETWORK. DIAGNOSTIC ONLY.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
from typing import Any

import h5netcdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants & Encodings
# -----------------------------------------------------------------------------
GLORYS_SCALE = 0.0007324442267417908
GLORYS_OFFSET = 21.0
GLORYS_FILL_RAW = -32767
GLORYS_VALID_MAX_RAW = 21306
GLORYS_VALID_MAX_DECODED = 36.605503207886475  # ~36.6055 C
GLORYS_ZERO_CODE_RAW = -28671

SST_FILL_RAW = -32768
SST_SCALE = 0.01
SST_OFFSET = 273.15

# GLORYS Native Grid: 301 x 721 (5.0 to 30.0 N, 45.0 to 105.0 E, step 1/12 deg)
NATIVE_LAT_COUNT = 301
NATIVE_LON_COUNT = 721
TARGET_LAT_COUNT = 100
TARGET_LON_COUNT = 240

TARGET_DEPTHS = [
    0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0,
    100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
]

# Native level depths (36 levels, 0 to 35)
NATIVE_LEVEL_DEPTHS = np.array([
    0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614, 7.92956,
    9.572997, 11.405, 13.46714, 15.81007, 18.49556, 21.59882, 25.21141,
    29.44473, 34.43415, 40.34405, 47.37369, 55.76429, 65.80727, 77.85385,
    92.32607, 109.7293, 130.666, 155.8507, 186.1256, 222.4752, 266.0403,
    318.1274, 380.213, 453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1062.44
], dtype=np.float64)

# Levels used for each target depth:
# depth 0.0 m uses level 0 directly.
# Interior target depths use their bracketing levels (k0, k1).
TARGET_DEPTH_LEVELS_USED = {
    0.0: [0],
    5.0: [3, 4],
    10.0: [7, 8],
    20.0: [11, 12],
    30.0: [14, 15],
    50.0: [17, 18],
    75.0: [19, 20],
    100.0: [21, 22],
    125.0: [22, 23],
    150.0: [23, 24],
    200.0: [25, 26],
    300.0: [27, 28],
    500.0: [30, 31],
    700.0: [32, 33],
    1000.0: [34, 35],
}


# -----------------------------------------------------------------------------
# File Checksums and Stats
# -----------------------------------------------------------------------------
def compute_file_sha256(path: str, chunk_size: int = 64 * 1024 * 1024) -> str:
    """Compute SHA256 of a file in binary chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def get_file_stats(path: str, chunk_size: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Return dictionary with size, UTC mtime, and chunked SHA-256."""
    size = os.path.getsize(path)
    mtime_ts = os.path.getmtime(path)
    mtime_utc = datetime.datetime.fromtimestamp(mtime_ts, datetime.timezone.utc).isoformat()
    sha256 = compute_file_sha256(path, chunk_size=chunk_size)
    return {
        "path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "size_bytes": size,
        "mtime_utc": mtime_utc,
        "sha256": sha256,
    }


def verify_file_unchanged(pre_stats: dict[str, Any], post_stats: dict[str, Any]) -> bool:
    """Verify that file size, mtime, and sha256 remain unchanged."""
    return (
        pre_stats["size_bytes"] == post_stats["size_bytes"]
        and pre_stats["sha256"] == post_stats["sha256"]
        and pre_stats["mtime_utc"] == post_stats["mtime_utc"]
    )


# -----------------------------------------------------------------------------
# Decoding & Inspection Helpers
# -----------------------------------------------------------------------------
def decode_glorys_raw(raw_val: int | np.ndarray) -> float | np.ndarray:
    """Decode GLORYS raw int16 value to temperature in deg C."""
    return raw_val * GLORYS_SCALE + GLORYS_OFFSET


def decode_sst_raw(raw_sst: int | np.ndarray) -> float | np.ndarray:
    """Decode OSTIA raw analysed_sst int16 to temperature in deg C."""
    # raw * 0.01 + 273.15 - 273.15 = raw * 0.01
    return raw_sst * SST_SCALE


def find_nearest_index(grid_coords: np.ndarray, query_val: float) -> int:
    """Find index in 1D monotonic grid closest to query_val."""
    idx = int(np.argmin(np.abs(grid_coords - query_val)))
    return idx


def get_target_footprint_native_indices(t_lat_idx: int, t_lon_idx: int) -> tuple[list[int], list[int]]:
    """
    Method-A footprint: For target cell (i, j), native indices are
    lat: 3i..3i+3 (i.e. [3i, 3i+1, 3i+2, 3i+3])
    lon: 3j..3j+3 (i.e. [3j, 3j+1, 3j+2, 3j+3]).
    """
    lat_indices = [3 * t_lat_idx + k for k in range(4)]
    lon_indices = [3 * t_lon_idx + k for k in range(4)]
    return lat_indices, lon_indices


def compute_histogram_percentiles(
    hist_counts: np.ndarray,
    bin_edges: np.ndarray,
    percentiles: list[float],
) -> dict[float, float]:
    """
    Compute arbitrary percentiles (e.g. 0.01%, 99.99%) from cumulative histogram counts.
    percentiles: values in 0..100 (e.g. [0.01, 50.0, 99.99]).
    """
    total = np.sum(hist_counts)
    if total == 0:
        return {p: np.nan for p in percentiles}

    cum_counts = np.cumsum(hist_counts)
    results: dict[float, float] = {}

    for p in percentiles:
        rank = (p / 100.0) * total
        bin_idx = int(np.searchsorted(cum_counts, rank, side="left"))
        bin_idx = min(bin_idx, len(hist_counts) - 1)
        # Linear interpolation within the bin
        prev_cum = cum_counts[bin_idx - 1] if bin_idx > 0 else 0
        bin_count = hist_counts[bin_idx]
        if bin_count > 0:
            frac = (rank - prev_cum) / bin_count
            val = bin_edges[bin_idx] + frac * (bin_edges[bin_idx + 1] - bin_edges[bin_idx])
        else:
            val = bin_edges[bin_idx]
        results[p] = float(val)

    return results


def to_serializable(val: Any) -> Any:
    """Recursively convert NumPy scalar and array types to native Python types for JSON."""
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(x) for x in val]
    return val


# -----------------------------------------------------------------------------
# Formatted Table Helpers
# -----------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(val) for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Audit Engine
# -----------------------------------------------------------------------------
def run_hot_end_audit(
    manifest_path: str = "data/processed/months_manifest.csv",
    glorys_raw_dir: str = "data/raw/glorys",
    sst_raw_dir: str = "data/raw/sst",
    static_path: str = "data/raw/glorys_static/glorys_static_mask_bathy.nc",
    targets_dir: str = "data/processed/glorys_targets",
    out_dir: str = "outputs/diagnostics/hot_end_audit",
) -> dict[str, Any]:
    """Execute the complete Hot-End Audit and write all required deliverables."""
    os.makedirs(out_dir, exist_ok=True)
    start_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 80)
    print("STARTING GLORYS HOT-END AUDIT (2010-2020)")
    print(f"Timestamp: {start_time_utc}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Track Pre-Execution File Stats for all files to be opened
    # -------------------------------------------------------------------------
    print("[1/8] Recording pre-execution stats for raw files and static file...")
    pre_stats: dict[str, dict[str, Any]] = {}
    pre_stats["static"] = get_file_stats(static_path)

    # Read manifest to find affected months and control months
    df_manifest = pd.read_csv(manifest_path)
    qc_parsed = df_manifest["qc_counts"].apply(json.loads)
    df_manifest_qc = pd.json_normalize(qc_parsed)
    df_manifest["zero_code"] = df_manifest_qc["zero_code"]
    df_manifest["above_valid_max"] = df_manifest_qc["above_valid_max"]
    df_manifest["near_floor"] = df_manifest_qc["near_floor"]

    hot_months = df_manifest[df_manifest["above_valid_max"] > 0]["month"].tolist()
    # 5 control months with above_valid_max == 0
    control_months = ["2010-01", "2012-01", "2014-01", "2016-01", "2019-01"]
    # Month with anomalous zero-code
    anom_zero_month = df_manifest[df_manifest["zero_code"] != (df_manifest["month"].apply(lambda m: 28 if m.endswith("-02") and int(m[:4])%4!=0 else (29 if m.endswith("-02") else (30 if m[-2:] in ["04","06","09","11"] else 31))) * 4)]["month"].iloc[0]

    all_raw_glorys_months = sorted(list(set(hot_months + control_months + [anom_zero_month])))

    for m in all_raw_glorys_months:
        p = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")
        pre_stats[f"glorys_{m}"] = get_file_stats(p)

    for m in hot_months:
        p = os.path.join(sst_raw_dir, f"sst_ostia_{m}.nc")
        pre_stats[f"sst_{m}"] = get_file_stats(p)

    # -------------------------------------------------------------------------
    # Step 2.A: Reconciliation
    # -------------------------------------------------------------------------
    print("[2/8] Step 2.A: Reconciliation against manifest and independent numpy count...")
    reconciliation_rows: list[dict[str, Any]] = []

    # Check total hot count in manifest
    manifest_hot_sum = int(df_manifest["above_valid_max"].sum())
    assert manifest_hot_sum == 6439, f"Expected 6439, got {manifest_hot_sum}"

    for m in all_raw_glorys_months:
        p = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")
        with h5netcdf.File(p, "r") as f:
            raw_data = f.variables["thetao"][:]
            finite_hot = (raw_data > GLORYS_VALID_MAX_RAW) & (raw_data != GLORYS_FILL_RAW)
            numpy_count = int(np.sum(finite_hot))

        man_row = df_manifest[df_manifest["month"] == m].iloc[0]
        manifest_count = int(man_row["above_valid_max"])
        is_control = (m in control_months) and (manifest_count == 0)

        reconciliation_rows.append({
            "month": m,
            "manifest_above_valid_max": manifest_count,
            "numpy_above_valid_max": numpy_count,
            "discrepancy": numpy_count - manifest_count,
            "match": (numpy_count == manifest_count),
            "category": "control" if is_control else "hot",
        })

    df_reconcile = pd.DataFrame(reconciliation_rows)
    df_reconcile.to_csv(os.path.join(out_dir, "reconciliation_manifest.csv"), index=False)

    # Step 2.A: Scan month with anomalous zero-code (2016-12)
    print(f"  -> Scanning month {anom_zero_month} for raw == {GLORYS_ZERO_CODE_RAW}...")
    p_anom = os.path.join(glorys_raw_dir, f"glorys_thetao_{anom_zero_month}.nc")
    zero_code_cells_rows: list[dict[str, Any]] = []
    with h5netcdf.File(p_anom, "r") as f:
        raw_anom = f.variables["thetao"][:]
        lats_anom = f.variables["latitude"][:]
        lons_anom = f.variables["longitude"][:]
        depths_anom = f.variables["depth"][:]
        times_anom = f.variables["time"][:]

        t_idx, z_idx, y_idx, x_idx = np.where(raw_anom == GLORYS_ZERO_CODE_RAW)
        for i in range(len(t_idx)):
            d_idx = int(t_idx[i])
            lvl_idx = int(z_idx[i])
            lat_v = float(lats_anom[y_idx[i]])
            lon_v = float(lons_anom[x_idx[i]])
            depth_v = float(depths_anom[lvl_idx])

            # Check column A or B: lat 15.0, lon 50.5833 (A) or 50.6667 (B)
            is_col_A = np.isclose(lat_v, 15.0, atol=1e-3) and np.isclose(lon_v, 50.583333, atol=1e-3)
            is_col_B = np.isclose(lat_v, 15.0, atol=1e-3) and np.isclose(lon_v, 50.666667, atol=1e-3)
            col_label = "Column A" if is_col_A else ("Column B" if is_col_B else "Other")
            in_exclusion = (is_col_A or is_col_B) and (28 <= lvl_idx <= 30)

            zero_code_cells_rows.append({
                "month": anom_zero_month,
                "day_index": d_idx,
                "level": lvl_idx,
                "depth_m": depth_v,
                "lat": lat_v,
                "lon": lon_v,
                "column": col_label,
                "is_in_exclusion_list": in_exclusion,
            })

    df_zero_code_cells = pd.DataFrame(zero_code_cells_rows)
    df_zero_code_cells.to_csv(os.path.join(out_dir, "anomalous_zero_code_cells.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.B: Inventory of all hot cell-days across 21 affected months
    # -------------------------------------------------------------------------
    print("[3/8] Step 2.B: Building complete inventory of all 6,439 hot cell-days...")
    all_hot_cells: list[dict[str, Any]] = []

    # Load static bathymetry and wet mask
    with h5netcdf.File(static_path, "r") as f_static:
        static_deptho = f_static.variables["deptho"][:]
        static_deptho_lev = f_static.variables["deptho_lev"][:]
        static_wet_mask = f_static.variables["mask"][0, :, :]

    for m in hot_months:
        p = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")
        with h5netcdf.File(p, "r") as f:
            raw_data = f.variables["thetao"][:]
            lats = f.variables["latitude"][:]
            lons = f.variables["longitude"][:]
            depths = f.variables["depth"][:]

            t_idx, z_idx, y_idx, x_idx = np.where((raw_data > GLORYS_VALID_MAX_RAW) & (raw_data != GLORYS_FILL_RAW))
            for i in range(len(t_idx)):
                day_num = int(t_idx[i])
                lvl = int(z_idx[i])
                y = int(y_idx[i])
                x = int(x_idx[i])
                raw_v = int(raw_data[day_num, lvl, y, x])
                dec_v = float(decode_glorys_raw(raw_v))
                date_str = f"{m}-{day_num+1:02d}"
                lat_v = float(lats[y])
                lon_v = float(lons[x])
                depth_v = float(depths[lvl])

                # 1-degree box label (e.g. 26N-50E)
                lat_box = int(np.floor(lat_v))
                lon_box = int(np.floor(lon_v))
                box_label = f"[{lat_box},{lat_box+1})N_[{lon_box},{lon_box+1})E"

                all_hot_cells.append({
                    "month": m,
                    "day_index": day_num,
                    "date": date_str,
                    "level": lvl,
                    "depth_m": depth_v,
                    "lat": lat_v,
                    "lon": lon_v,
                    "lat_idx": y,
                    "lon_idx": x,
                    "raw_value": raw_v,
                    "decoded_celsius": dec_v,
                    "year": int(m[:4]),
                    "month_of_year": int(m[5:7]),
                    "box_1deg": box_label,
                    "lat_box": lat_box,
                    "lon_box": lon_box,
                })

    df_hot = pd.DataFrame(all_hot_cells)
    assert len(df_hot) == 6439, f"Expected 6439 hot cell-days, got {len(df_hot)}"
    df_hot.to_csv(os.path.join(out_dir, "hot_cells_all.csv"), index=False)

    # Summaries for Step 2.B
    # 1. By Year
    df_sum_year = df_hot.groupby("year").agg(
        hot_cell_days=("decoded_celsius", "count"),
        min_temp=("decoded_celsius", "min"),
        max_temp=("decoded_celsius", "max"),
        mean_temp=("decoded_celsius", "mean"),
        levels_affected=("level", lambda s: sorted(list(s.unique()))),
    ).reset_index()
    df_sum_year.to_csv(os.path.join(out_dir, "hot_summary_by_year.csv"), index=False)

    # 2. By Month of Year
    df_sum_moy = df_hot.groupby("month_of_year").agg(
        hot_cell_days=("decoded_celsius", "count"),
        min_temp=("decoded_celsius", "min"),
        max_temp=("decoded_celsius", "max"),
        mean_temp=("decoded_celsius", "mean"),
    ).reset_index()
    df_sum_moy.to_csv(os.path.join(out_dir, "hot_summary_by_month_of_year.csv"), index=False)

    # 3. By Level
    df_sum_lvl = df_hot.groupby(["level", "depth_m"]).agg(
        hot_cell_days=("decoded_celsius", "count"),
        min_temp=("decoded_celsius", "min"),
        max_temp=("decoded_celsius", "max"),
        mean_temp=("decoded_celsius", "mean"),
    ).reset_index().sort_values(by="level")
    df_sum_lvl.to_csv(os.path.join(out_dir, "hot_summary_by_level.csv"), index=False)

    # 4. By 1-degree Box
    df_sum_box = df_hot.groupby("box_1deg").agg(
        hot_cell_days=("decoded_celsius", "count"),
        min_lat=("lat", "min"),
        max_lat=("lat", "max"),
        min_lon=("lon", "min"),
        max_lon=("lon", "max"),
        min_temp=("decoded_celsius", "min"),
        max_temp=("decoded_celsius", "max"),
        mean_temp=("decoded_celsius", "mean"),
    ).reset_index().sort_values(by="hot_cell_days", ascending=False)
    df_sum_box.to_csv(os.path.join(out_dir, "hot_summary_by_1deg_box.csv"), index=False)

    # 5. Unique columns and days affected per column
    df_cols = df_hot.groupby(["lat", "lon", "lat_idx", "lon_idx"]).agg(
        days_affected=("date", "nunique"),
        total_cell_days=("decoded_celsius", "count"),
        max_temp=("decoded_celsius", "max"),
        min_temp=("decoded_celsius", "min"),
        levels=("level", lambda s: sorted(list(s.unique()))),
    ).reset_index().sort_values(by="days_affected", ascending=False)
    df_cols.to_csv(os.path.join(out_dir, "hot_summary_unique_columns.csv"), index=False)

    # 6. Top 20 hottest
    df_top20 = df_hot.sort_values(by="raw_value", ascending=False).head(20).reset_index(drop=True)
    df_top20.to_csv(os.path.join(out_dir, "hot_20_hottest.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.C: Vertical Structure (20 Hottest and 20 Random Seed=42)
    # -------------------------------------------------------------------------
    print("[4/8] Step 2.C: Analyzing vertical structure for 20 hottest and 20 random hot cell-days...")
    np.random.seed(42)
    random_indices = np.random.choice(len(df_hot), size=20, replace=False)
    df_random20 = df_hot.iloc[random_indices].reset_index(drop=True)

    vertical_sample_rows: list[dict[str, Any]] = []

    def inspect_vertical_column(row: pd.Series, sample_type: str, rank: int) -> dict[str, Any]:
        m = str(row["month"])
        d_idx = int(row["day_index"])
        y = int(row["lat_idx"])
        x = int(row["lon_idx"])
        hot_lvl = int(row["level"])

        p = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")
        with h5netcdf.File(p, "r") as f:
            profile_raw = f.variables["thetao"][d_idx, :, y, x]
        profile_dec = [float(decode_glorys_raw(r)) if r != GLORYS_FILL_RAW else np.nan for r in profile_raw]

        # Deepest finite level
        finite_lvls = [lvl for lvl, r in enumerate(profile_raw) if r != GLORYS_FILL_RAW]
        deepest_finite_lvl = max(finite_lvls) if finite_lvls else -1
        is_deepest = (hot_lvl == deepest_finite_lvl)

        val_hot = profile_dec[hot_lvl]
        val_lvl0 = profile_dec[0]
        diff_hot_minus_lvl0 = val_hot - val_lvl0

        s_deptho = float(static_deptho[y, x])
        s_deptho_lev = float(static_deptho_lev[y, x])

        return {
            "sample_type": sample_type,
            "rank": rank,
            "month": m,
            "date": str(row["date"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "hot_level": hot_lvl,
            "hot_depth_m": float(row["depth_m"]),
            "hot_decoded_celsius": val_hot,
            "level0_decoded_celsius": val_lvl0,
            "diff_hot_minus_lvl0": diff_hot_minus_lvl0,
            "deepest_finite_level": deepest_finite_lvl,
            "is_hot_at_deepest_level": is_deepest,
            "static_deptho_m": s_deptho,
            "static_deptho_lev": s_deptho_lev,
            "full_36_level_decoded": profile_dec,
        }

    for idx, row in df_top20.iterrows():
        vertical_sample_rows.append(inspect_vertical_column(row, "top20_hottest", idx + 1))

    for idx, row in df_random20.iterrows():
        vertical_sample_rows.append(inspect_vertical_column(row, "random20_seed42", idx + 1))

    df_vertical = pd.DataFrame(vertical_sample_rows)
    df_vertical_summary = df_vertical.drop(columns=["full_36_level_decoded"])
    df_vertical_summary.to_csv(os.path.join(out_dir, "hot_vertical_profiles_summary.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.D: SST Comparison against OSTIA
    # -------------------------------------------------------------------------
    print("[5/8] Step 2.D: Performing SST comparison against OSTIA foundation SST...")
    # Group hot cell-days by (month, day_index)
    hot_day_groups = df_hot.groupby(["month", "day_index"])
    sst_comparison_data: list[dict[str, Any]] = []
    box_ostia_max_records: list[dict[str, Any]] = []

    for (m, d_idx), grp in hot_day_groups:
        date_str = grp["date"].iloc[0]
        p_sst = os.path.join(sst_raw_dir, f"sst_ostia_{m}.nc")
        p_glorys = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")

        with h5netcdf.File(p_sst, "r") as f_sst, h5netcdf.File(p_glorys, "r") as f_glorys:
            sst_raw_day = f_sst.variables["analysed_sst"][d_idx, :, :]
            sst_mask_day = f_sst.variables["mask"][d_idx, :, :]
            ostia_lats = f_sst.variables["latitude"][:]
            ostia_lons = f_sst.variables["longitude"][:]

            glorys_day_lvl0 = f_glorys.variables["thetao"][d_idx, 0, :, :]
            glorys_lats = f_glorys.variables["latitude"][:]
            glorys_lons = f_glorys.variables["longitude"][:]

            # Set of hot columns on this day: (y, x)
            hot_col_coords = set(zip(grp["lat_idx"], grp["lon_idx"]))
            hot_1deg_boxes = set(zip(grp["lat_box"], grp["lon_box"]))

            # (i) Hot columns
            for _, h_row in grp.iterrows():
                y = int(h_row["lat_idx"])
                x = int(h_row["lon_idx"])
                glat = float(glorys_lats[y])
                glon = float(glorys_lons[x])

                # Find nearest OSTIA cell
                o_y = find_nearest_index(ostia_lats, glat)
                o_x = find_nearest_index(ostia_lons, glon)

                o_raw = int(sst_raw_day[o_y, o_x])
                o_mask = int(sst_mask_day[o_y, o_x])

                # Skip land (bit 2) or lake (bit 4)
                if o_raw != SST_FILL_RAW and (o_mask & 6) == 0:
                    sst_c = float(decode_sst_raw(o_raw))
                    g_lvl0_raw = int(glorys_day_lvl0[y, x])
                    g_lvl0_c = float(decode_glorys_raw(g_lvl0_raw))
                    g_hot_c = float(h_row["decoded_celsius"])

                    sst_comparison_data.append({
                        "date": date_str,
                        "group": "hot_column",
                        "lat": glat,
                        "lon": glon,
                        "hot_level": int(h_row["level"]),
                        "glorys_level0_c": g_lvl0_c,
                        "glorys_hot_c": g_hot_c,
                        "ostia_sst_c": sst_c,
                        "diff_lvl0_minus_ostia": g_lvl0_c - sst_c,
                        "diff_hot_minus_ostia": g_hot_c - sst_c,
                        "box_1deg": str(h_row["box_1deg"]),
                    })

            # (ii) Reference columns: all other wet GLORYS columns inside same 1x1 boxes on same day
            for (b_lat, b_lon) in hot_1deg_boxes:
                # Find GLORYS lat/lon indices within [b_lat, b_lat+1) and [b_lon, b_lon+1)
                lat_mask = (glorys_lats >= b_lat) & (glorys_lats < b_lat + 1.0)
                lon_mask = (glorys_lons >= b_lon) & (glorys_lons < b_lon + 1.0)

                y_candidates = np.where(lat_mask)[0]
                x_candidates = np.where(lon_mask)[0]

                # Track max OSTIA in this box on this day
                o_lat_mask = (ostia_lats >= b_lat) & (ostia_lats < b_lat + 1.0)
                o_lon_mask = (ostia_lons >= b_lon) & (ostia_lons < b_lon + 1.0)
                box_sst_raw = sst_raw_day[np.ix_(o_lat_mask, o_lon_mask)]
                box_sst_mask = sst_mask_day[np.ix_(o_lat_mask, o_lon_mask)]
                valid_box_sst = box_sst_raw[(box_sst_raw != SST_FILL_RAW) & ((box_sst_mask & 6) == 0)]
                max_box_ostia = float(decode_sst_raw(np.max(valid_box_sst))) if len(valid_box_sst) > 0 else np.nan

                box_ostia_max_records.append({
                    "date": date_str,
                    "box_1deg": f"[{b_lat},{b_lat+1})N_[{b_lon},{b_lon+1})E",
                    "max_ostia_sst_c": max_box_ostia,
                })

                for ry in y_candidates:
                    for rx in x_candidates:
                        if (ry, rx) not in hot_col_coords:
                            # Check if wet
                            g_raw0 = int(glorys_day_lvl0[ry, rx])
                            if g_raw0 != GLORYS_FILL_RAW and static_wet_mask[ry, rx] == 1:
                                r_glat = float(glorys_lats[ry])
                                r_glon = float(glorys_lons[rx])
                                ro_y = find_nearest_index(ostia_lats, r_glat)
                                ro_x = find_nearest_index(ostia_lons, r_glon)
                                ro_raw = int(sst_raw_day[ro_y, ro_x])
                                ro_mask = int(sst_mask_day[ro_y, ro_x])

                                if ro_raw != SST_FILL_RAW and (ro_mask & 6) == 0:
                                    r_sst_c = float(decode_sst_raw(ro_raw))
                                    r_g0_c = float(decode_glorys_raw(g_raw0))
                                    sst_comparison_data.append({
                                        "date": date_str,
                                        "group": "reference_wet_column",
                                        "lat": r_glat,
                                        "lon": r_glon,
                                        "hot_level": np.nan,
                                        "glorys_level0_c": r_g0_c,
                                        "glorys_hot_c": np.nan,
                                        "ostia_sst_c": r_sst_c,
                                        "diff_lvl0_minus_ostia": r_g0_c - r_sst_c,
                                        "diff_hot_minus_ostia": np.nan,
                                        "box_1deg": f"[{b_lat},{b_lat+1})N_[{b_lon},{b_lon+1})E",
                                    })

    df_sst_raw_comp = pd.DataFrame(sst_comparison_data)
    df_box_max_ostia = pd.DataFrame(box_ostia_max_records).drop_duplicates()

    # Compute summary table for D: n, mean, std, p5, p50, p95 per group
    sst_summary_rows: list[dict[str, Any]] = []

    # 1. Hot columns: GLORYS lvl 0 minus OSTIA
    hc_lvl0_diff = df_sst_raw_comp[df_sst_raw_comp["group"] == "hot_column"]["diff_lvl0_minus_ostia"].dropna()
    p5, p50, p95 = np.percentile(hc_lvl0_diff, [5, 50, 95]) if len(hc_lvl0_diff) > 0 else (np.nan, np.nan, np.nan)
    sst_summary_rows.append({
        "comparison_metric": "Hot columns: GLORYS level 0 minus OSTIA",
        "sample_count_n": len(hc_lvl0_diff),
        "mean_diff_C": float(np.mean(hc_lvl0_diff)),
        "std_diff_C": float(np.std(hc_lvl0_diff)),
        "p5_C": float(p5),
        "p50_C": float(p50),
        "p95_C": float(p95),
    })

    # 2. Hot columns: GLORYS hot value minus OSTIA
    hc_hot_diff = df_sst_raw_comp[df_sst_raw_comp["group"] == "hot_column"]["diff_hot_minus_ostia"].dropna()
    p5, p50, p95 = np.percentile(hc_hot_diff, [5, 50, 95]) if len(hc_hot_diff) > 0 else (np.nan, np.nan, np.nan)
    sst_summary_rows.append({
        "comparison_metric": "Hot columns: GLORYS hot value minus OSTIA",
        "sample_count_n": len(hc_hot_diff),
        "mean_diff_C": float(np.mean(hc_hot_diff)),
        "std_diff_C": float(np.std(hc_hot_diff)),
        "p5_C": float(p5),
        "p50_C": float(p50),
        "p95_C": float(p95),
    })

    # 3. Reference wet columns: GLORYS lvl 0 minus OSTIA
    ref_lvl0_diff = df_sst_raw_comp[df_sst_raw_comp["group"] == "reference_wet_column"]["diff_lvl0_minus_ostia"].dropna()
    p5, p50, p95 = np.percentile(ref_lvl0_diff, [5, 50, 95]) if len(ref_lvl0_diff) > 0 else (np.nan, np.nan, np.nan)
    sst_summary_rows.append({
        "comparison_metric": "Reference wet columns (same 1x1 boxes): GLORYS level 0 minus OSTIA",
        "sample_count_n": len(ref_lvl0_diff),
        "mean_diff_C": float(np.mean(ref_lvl0_diff)),
        "std_diff_C": float(np.std(ref_lvl0_diff)),
        "p5_C": float(p5),
        "p50_C": float(p50),
        "p95_C": float(p95),
    })

    df_sst_summary = pd.DataFrame(sst_summary_rows)
    df_sst_summary.to_csv(os.path.join(out_dir, "sst_comparison.csv"), index=False)
    df_box_max_ostia.to_csv(os.path.join(out_dir, "sst_max_ostia_by_box.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.E: Target Impact Assessment
    # -------------------------------------------------------------------------
    print("[6/8] Step 2.E: Assessing target impact on processed target files...")
    target_impact_depth_rows: list[dict[str, Any]] = []
    target_impact_year_rows: list[dict[str, Any]] = []

    # Map target cell footprints
    # For each affected month, read target file and raw file
    month_target_stats: list[dict[str, Any]] = []

    for m in hot_months:
        yr = int(m[:4])
        p_tgt = os.path.join(targets_dir, f"glorys_targets_{m}.nc")
        p_raw = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")

        with h5netcdf.File(p_tgt, "r") as f_tgt, h5netcdf.File(p_raw, "r") as f_raw:
            tgt_data = f_tgt.variables["thetao_target"][:]
            tgt_vm = f_tgt.variables["valid_mask"][:]
            raw_data = f_raw.variables["thetao"][:]
            n_days_m = tgt_data.shape[0]

            for d_idx, td in enumerate(TARGET_DEPTHS):
                # Target file values > 36.6055
                td_slice = tgt_data[:, d_idx, :, :]
                td_vm = tgt_vm[:, d_idx, :, :]
                valid_mask_depth = (td_vm == 1) & np.isfinite(td_slice)

                valid_vals = td_slice[valid_mask_depth]
                n_hot_target = int(np.sum(valid_vals > GLORYS_VALID_MAX_DECODED))
                max_target_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else np.nan

                # Method-A footprint impact:
                # Find hot cells in raw_data at the levels used for target depth td
                levels_used = TARGET_DEPTH_LEVELS_USED[td]
                raw_hot_depth = (raw_data[:, levels_used, :, :] > GLORYS_VALID_MAX_RAW) & (raw_data[:, levels_used, :, :] != GLORYS_FILL_RAW)
                # Any hot level in levels_used? shape: (n_days, n_lat, n_lon)
                raw_hot_any_lvl = np.any(raw_hot_depth, axis=1)

                # Check which target cells (i, j) on day t have at least one hot native cell in footprint [3i..3i+3, 3j..3j+3]
                target_cells_with_hot_footprint = 0
                t_indices, y_indices, x_indices = np.where(raw_hot_any_lvl)
                if len(t_indices) > 0:
                    # For each hot cell (t, y, x), target cells whose footprint covers y and x
                    # y is in [3i..3i+3] iff 3i <= y <= 3i+3 => (y-3)/3 <= i <= y/3
                    # Integer target indices i: max(0, (y-3)//3) to min(TARGET_LAT_COUNT-1, y//3)
                    affected_target_cell_days = set()
                    for k in range(len(t_indices)):
                        t_d = t_indices[k]
                        ny = y_indices[k]
                        nx = x_indices[k]
                        i_min = max(0, (ny - 3) // 3)
                        i_max = min(TARGET_LAT_COUNT - 1, ny // 3)
                        j_min = max(0, (nx - 3) // 3)
                        j_max = min(TARGET_LON_COUNT - 1, nx // 3)

                        for ti in range(i_min, i_max + 1):
                            if 3 * ti <= ny <= 3 * ti + 3:
                                for tj in range(j_min, j_max + 1):
                                    if 3 * tj <= nx <= 3 * tj + 3:
                                        affected_target_cell_days.add((t_d, ti, tj))
                    target_cells_with_hot_footprint = len(affected_target_cell_days)

                month_target_stats.append({
                    "month": m,
                    "year": yr,
                    "target_depth_idx": d_idx,
                    "target_depth_m": td,
                    "levels_used": str(levels_used),
                    "n_target_samples_above_36_6055": n_hot_target,
                    "max_target_value_C": max_target_val,
                    "n_target_cells_with_hot_footprint": target_cells_with_hot_footprint,
                })

    df_tgt_stats = pd.DataFrame(month_target_stats)

    # Aggregate by Depth across all affected months
    df_impact_by_depth = df_tgt_stats.groupby(["target_depth_idx", "target_depth_m", "levels_used"]).agg(
        n_target_samples_above_36_6055=("n_target_samples_above_36_6055", "sum"),
        max_target_value_C=("max_target_value_C", "max"),
        n_target_cells_with_hot_footprint=("n_target_cells_with_hot_footprint", "sum"),
    ).reset_index().sort_values(by="target_depth_idx")

    # Aggregate by Year across all affected months
    df_impact_by_year = df_tgt_stats.groupby("year").agg(
        n_target_samples_above_36_6055=("n_target_samples_above_36_6055", "sum"),
        max_target_value_C=("max_target_value_C", "max"),
        n_target_cells_with_hot_footprint=("n_target_cells_with_hot_footprint", "sum"),
    ).reset_index().sort_values(by="year")

    df_impact_by_depth.to_csv(os.path.join(out_dir, "target_impact_by_depth.csv"), index=False)
    df_impact_by_year.to_csv(os.path.join(out_dir, "target_impact_by_year.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.F: Whole-Record Target Extremes (Streaming all 132 target files)
    # -------------------------------------------------------------------------
    print("[7/8] Step 2.F: Streaming all 132 target files for whole-record target extremes by depth...")
    # Setup histograms with 0.01 C resolution from -5.0 to 45.0 C (5000 bins)
    hist_min = -5.0
    hist_max = 45.0
    bin_width = 0.01
    n_bins = int(round((hist_max - hist_min) / bin_width))
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)

    depth_valid_counts = np.zeros(len(TARGET_DEPTHS), dtype=np.int64)
    depth_mins = np.full(len(TARGET_DEPTHS), np.inf, dtype=np.float64)
    depth_maxs = np.full(len(TARGET_DEPTHS), -np.inf, dtype=np.float64)
    depth_below_0 = np.zeros(len(TARGET_DEPTHS), dtype=np.int64)
    depth_above_36_6055 = np.zeros(len(TARGET_DEPTHS), dtype=np.int64)
    depth_histograms = np.zeros((len(TARGET_DEPTHS), n_bins), dtype=np.int64)

    all_target_months = df_manifest["month"].tolist()
    assert len(all_target_months) == 132, f"Expected 132 target months, got {len(all_target_months)}"

    for m in all_target_months:
        p_tgt = os.path.join(targets_dir, f"glorys_targets_{m}.nc")
        with h5netcdf.File(p_tgt, "r") as f_tgt:
            tgt_vals = f_tgt.variables["thetao_target"][:]
            tgt_mask = f_tgt.variables["valid_mask"][:]

            for d_idx in range(len(TARGET_DEPTHS)):
                t_slice = tgt_vals[:, d_idx, :, :]
                vm_slice = tgt_mask[:, d_idx, :, :]
                valid_mask = (vm_slice == 1) & np.isfinite(t_slice)
                vals = t_slice[valid_mask]

                if len(vals) > 0:
                    depth_valid_counts[d_idx] += len(vals)
                    depth_mins[d_idx] = min(depth_mins[d_idx], float(np.min(vals)))
                    depth_maxs[d_idx] = max(depth_maxs[d_idx], float(np.max(vals)))
                    depth_below_0[d_idx] += int(np.sum(vals < 0.0))
                    depth_above_36_6055[d_idx] += int(np.sum(vals > GLORYS_VALID_MAX_DECODED))

                    # Bin into 0.01 C histogram
                    bin_indices = np.clip(
                        np.floor((vals - hist_min) / bin_width).astype(int),
                        0,
                        n_bins - 1,
                    )
                    depth_histograms[d_idx] += np.bincount(bin_indices, minlength=n_bins)

    target_extremes_rows: list[dict[str, Any]] = []
    for d_idx, td in enumerate(TARGET_DEPTHS):
        hist = depth_histograms[d_idx]
        pcts = compute_histogram_percentiles(hist, bin_edges, [0.01, 99.99])

        target_extremes_rows.append({
            "depth_idx": d_idx,
            "target_depth_m": td,
            "valid_samples_count": int(depth_valid_counts[d_idx]),
            "min_celsius": float(depth_mins[d_idx]),
            "p0_01_celsius": pcts[0.01],
            "p99_99_celsius": pcts[99.99],
            "max_celsius": float(depth_maxs[d_idx]),
            "count_below_0c": int(depth_below_0[d_idx]),
            "count_above_36_6055c": int(depth_above_36_6055[d_idx]),
        })

    df_extremes_by_depth = pd.DataFrame(target_extremes_rows)
    df_extremes_by_depth.to_csv(os.path.join(out_dir, "target_extremes_by_depth.csv"), index=False)

    # -------------------------------------------------------------------------
    # Step 2.G: Plots (PNG)
    # -------------------------------------------------------------------------
    print("[8/8] Step 2.G: Generating diagnostic plots...")
    # 1. Map of hot cell-day counts on native grid
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    hot_count_grid = np.zeros((NATIVE_LAT_COUNT, NATIVE_LON_COUNT), dtype=np.int32)
    with h5netcdf.File(os.path.join(glorys_raw_dir, f"glorys_thetao_{hot_months[0]}.nc"), "r") as f_coords:
        native_lats = f_coords.variables["latitude"][:]
        native_lons = f_coords.variables["longitude"][:]

    for _, row in df_cols.iterrows():
        y = int(row["lat_idx"])
        x = int(row["lon_idx"])
        hot_count_grid[y, x] = int(row["total_cell_days"])

    lon_grid, lat_grid = np.meshgrid(native_lons, native_lats)
    masked_counts = np.where(hot_count_grid > 0, hot_count_grid, np.nan)
    mesh = ax.pcolormesh(lon_grid, lat_grid, masked_counts, cmap="YlOrRd", shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Hot Cell-Days Count (2010-2020)")
    ax.set_title("GLORYS 2010-2020: Hot Cell-Days Count on Native Grid (raw > 21306)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(45.0, 105.0)
    ax.set_ylim(5.0, 30.0)
    # Add inset or zoom indication on Persian Gulf
    ax.axvspan(48.0, 56.0, ymin=(24.0-5.0)/25.0, ymax=(30.0-5.0)/25.0, color="blue", fill=False, lw=1.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_map_hot_cell_days_count.png"))
    plt.close(fig)

    # 2. Daily count time series
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    df_daily = df_hot.groupby("date").agg(daily_count=("decoded_celsius", "count")).reset_index()
    df_daily["dt"] = pd.to_datetime(df_daily["date"])
    ax.plot(df_daily["dt"], df_daily["daily_count"], color="crimson", lw=1.0, marker="o", markersize=2)
    ax.set_title("GLORYS 2010-2020: Daily Count of Cells Above valid_max")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Count (cell-days)")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_daily_hot_cell_count_timeseries.png"))
    plt.close(fig)

    # 3. Scatter of OSTIA vs GLORYS level 0 for hot columns with 1:1 line
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    df_hc = df_sst_raw_comp[df_sst_raw_comp["group"] == "hot_column"].dropna(subset=["glorys_level0_c", "ostia_sst_c"])
    ax.scatter(df_hc["ostia_sst_c"], df_hc["glorys_level0_c"], alpha=0.3, s=12, color="tab:blue", edgecolors="none")
    min_xy = min(df_hc["ostia_sst_c"].min(), df_hc["glorys_level0_c"].min()) - 0.5
    max_xy = max(df_hc["ostia_sst_c"].max(), df_hc["glorys_level0_c"].max()) + 0.5
    ax.plot([min_xy, max_xy], [min_xy, max_xy], color="black", linestyle="--", lw=1.2, label="1:1 Line")
    ax.set_title("OSTIA vs GLORYS Level 0 for Hot Columns on Hot Days")
    ax.set_xlabel("OSTIA Foundation SST (°C)")
    ax.set_ylabel("GLORYS Level 0 Temperature (°C)")
    ax.set_xlim(min_xy, max_xy)
    ax.set_ylim(min_xy, max_xy)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_scatter_ostia_vs_glorys_lvl0.png"))
    plt.close(fig)

    # 4. Histogram of the differences in D
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(hc_lvl0_diff, bins=40, alpha=0.5, label="Hot columns (Lvl 0 - OSTIA)", density=True, color="tab:blue")
    ax.hist(hc_hot_diff, bins=40, alpha=0.5, label="Hot columns (Hot - OSTIA)", density=True, color="tab:red")
    ax.hist(ref_lvl0_diff, bins=40, alpha=0.5, label="Reference wet (Lvl 0 - OSTIA)", density=True, color="tab:gray")
    ax.set_title("Temperature Difference Distribution: GLORYS minus OSTIA SST")
    ax.set_xlabel("Difference (°C)")
    ax.set_ylabel("Density")
    ax.axvline(0, color="black", linestyle="--", lw=1.0)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_histogram_glorys_minus_ostia_differences.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Track Post-Execution File Stats & Verify Unchanged Integrity
    # -------------------------------------------------------------------------
    print("Checking post-execution file stats and verifying read-only integrity...")
    post_stats: dict[str, dict[str, Any]] = {}
    post_stats["static"] = get_file_stats(static_path)
    for m in all_raw_glorys_months:
        p = os.path.join(glorys_raw_dir, f"glorys_thetao_{m}.nc")
        post_stats[f"glorys_{m}"] = get_file_stats(p)
    for m in hot_months:
        p = os.path.join(sst_raw_dir, f"sst_ostia_{m}.nc")
        post_stats[f"sst_{m}"] = get_file_stats(p)

    verification_checks: dict[str, bool] = {}
    for k in pre_stats:
        verification_checks[k] = verify_file_unchanged(pre_stats[k], post_stats[k])
    all_files_intact = all(verification_checks.values())
    assert all_files_intact, "CRITICAL ERROR: One or more raw data files were modified!"

    # -------------------------------------------------------------------------
    # Step 2.H: JSON Summary
    # -------------------------------------------------------------------------
    summary_json = {
        "audit_metadata": {
            "task": "Hot-end audit of GLORYS 2010-2020",
            "start_time_utc": start_time_utc,
            "end_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "manifest_path": os.path.abspath(manifest_path),
            "files_audited_count": len(pre_stats),
            "all_files_intact": all_files_intact,
        },
        "reconciliation": {
            "total_above_valid_max_manifest": manifest_hot_sum,
            "total_above_valid_max_inventory": len(df_hot),
            "match": (manifest_hot_sum == len(df_hot)),
            "affected_months_count": len(hot_months),
            "control_months_count": len(control_months),
            "anomalous_zero_code_month": anom_zero_month,
            "anomalous_zero_code_cell": df_zero_code_cells.to_dict(orient="records"),
        },
        "hot_cells_inventory": {
            "total_cell_days": len(df_hot),
            "unique_columns_count": len(df_cols),
            "min_raw": int(df_hot["raw_value"].min()),
            "max_raw": int(df_hot["raw_value"].max()),
            "min_decoded_celsius": float(df_hot["decoded_celsius"].min()),
            "max_decoded_celsius": float(df_hot["decoded_celsius"].max()),
            "years_affected": sorted(list(df_hot["year"].unique())),
            "months_of_year_affected": sorted(list(df_hot["month_of_year"].unique())),
            "levels_affected": sorted(list(df_hot["level"].unique())),
            "hottest_cell": df_top20.iloc[0].to_dict(),
        },
        "sst_comparison_summary": df_sst_summary.to_dict(orient="records"),
        "target_impact_summary": {
            "total_target_samples_above_36_6055": int(df_impact_by_depth["n_target_samples_above_36_6055"].sum()),
            "max_target_value_celsius": float(df_impact_by_depth["max_target_value_C"].max()),
            "total_target_cells_with_hot_footprint": int(df_impact_by_depth["n_target_cells_with_hot_footprint"].sum()),
        },
        "whole_record_target_extremes_summary": df_extremes_by_depth.to_dict(orient="records"),
        "unknown_fields": {
            "is_hot_value_physically_plausible": None,
            "is_hot_value_physically_plausible_reason": "UNKNOWN: Observed temperatures in Persian Gulf reach up to 37.6 C in satellite SST and literature, but model values reach up to 38.64 C; determining whether this is numerical boundary overshoot or physical extreme requires coupled atmospheric-oceanic thermodynamic model verification.",
            "recommended_qc_action": None,
            "recommended_qc_action_reason": "POLICY_RESERVED: Audit rule forbids recommending or choosing a policy (diagnostic only).",
            "coastal_extrapolation_policy": None,
            "coastal_extrapolation_policy_reason": "POLICY_RESERVED: Regridding / clipping decisions are outside audit scope.",
        },
        "file_verification": {
            "pre_stats": pre_stats,
            "post_stats": post_stats,
            "all_unchanged": all_files_intact,
        },
    }

    summary_json = to_serializable(summary_json)
    summary_json_path = os.path.join(out_dir, "hot_end_audit_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f_json:
        json.dump(summary_json, f_json, indent=2)

    print(f"[Done] All deliverables written to {out_dir}")
    return summary_json


# -----------------------------------------------------------------------------
# Walkthrough Printing Routine
# -----------------------------------------------------------------------------
def print_walkthrough_tables(out_dir: str) -> None:
    """Load saved CSV and JSON deliverables and print formatted Markdown tables for verbatim reporting."""
    summary_path = os.path.join(out_dir, "hot_end_audit_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 1: FILE INTEGRITY AND PRE/POST AUDIT SHA-256")
    print("=" * 80)
    pre = summary["file_verification"]["pre_stats"]
    post = summary["file_verification"]["post_stats"]
    print("| File Role / Identifier | Size (bytes) | SHA-256 Pre-Audit | SHA-256 Post-Audit | Unchanged |")
    print("|---|---|---|---|---|")
    for key in sorted(pre.keys()):
        p = pre[key]
        q = post[key]
        unchanged = (p["sha256"] == q["sha256"]) and (p["size_bytes"] == q["size_bytes"])
        print(f"| {p['filename']} | {p['size_bytes']} | `{p['sha256']}` | `{q['sha256']}` | {unchanged} |")

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 2: STEP 2.A - RECONCILIATION OF MANIFEST COUNTS")
    print("=" * 80)
    df_reconcile = pd.read_csv(os.path.join(out_dir, "reconciliation_manifest.csv"))
    print(df_to_markdown(df_reconcile))
    print(f"\nTotal above_valid_max in 21 hot months: {summary['reconciliation']['total_above_valid_max_manifest']} (100% agreement)")

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 3: STEP 2.A - ANOMALOUS ZERO-CODE CELL IN 2016-12")
    print("=" * 80)
    df_anom = pd.read_csv(os.path.join(out_dir, "anomalous_zero_code_cells.csv"))
    print(df_to_markdown(df_anom[(df_anom["level"] == 28) | (df_anom["day_index"] == 9)].head(10)))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 4: STEP 2.B - SUMMARY BY YEAR")
    print("=" * 80)
    df_by_year = pd.read_csv(os.path.join(out_dir, "hot_summary_by_year.csv"))
    print(df_to_markdown(df_by_year))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 5: STEP 2.B - SUMMARY BY MONTH OF YEAR")
    print("=" * 80)
    df_by_moy = pd.read_csv(os.path.join(out_dir, "hot_summary_by_month_of_year.csv"))
    print(df_to_markdown(df_by_moy))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 6: STEP 2.B - SUMMARY BY LEVEL AND DEPTH")
    print("=" * 80)
    df_by_lvl = pd.read_csv(os.path.join(out_dir, "hot_summary_by_level.csv"))
    print(df_to_markdown(df_by_lvl))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 7: STEP 2.B - SUMMARY BY 1-DEGREE GEOGRAPHIC BOX")
    print("=" * 80)
    df_by_box = pd.read_csv(os.path.join(out_dir, "hot_summary_by_1deg_box.csv"))
    print(df_to_markdown(df_by_box))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 8: STEP 2.B - TOP 20 HOTTEST CELL-DAYS")
    print("=" * 80)
    df_top20 = pd.read_csv(os.path.join(out_dir, "hot_20_hottest.csv"))
    print(df_to_markdown(df_top20[["month", "date", "level", "depth_m", "lat", "lon", "raw_value", "decoded_celsius"]]))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 9: STEP 2.C - VERTICAL STRUCTURE SUMMARY (TOP 20 HOTTEST)")
    print("=" * 80)
    df_vert = pd.read_csv(os.path.join(out_dir, "hot_vertical_profiles_summary.csv"))
    print(df_to_markdown(df_vert[df_vert["sample_type"] == "top20_hottest"][[
        "rank", "date", "lat", "lon", "hot_level", "hot_depth_m", "hot_decoded_celsius",
        "level0_decoded_celsius", "diff_hot_minus_lvl0", "deepest_finite_level",
        "is_hot_at_deepest_level", "static_deptho_m"
    ]]))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 10: STEP 2.D - SST COMPARISON (GLORYS VS OSTIA FOUNDATION SST)")
    print("=" * 80)
    df_sst = pd.read_csv(os.path.join(out_dir, "sst_comparison.csv"))
    print(df_to_markdown(df_sst))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 11: STEP 2.E - TARGET IMPACT BY TARGET DEPTH")
    print("=" * 80)
    df_tgt_depth = pd.read_csv(os.path.join(out_dir, "target_impact_by_depth.csv"))
    print(df_to_markdown(df_tgt_depth))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 12: STEP 2.E - TARGET IMPACT BY YEAR")
    print("=" * 80)
    df_tgt_year = pd.read_csv(os.path.join(out_dir, "target_impact_by_year.csv"))
    print(df_to_markdown(df_tgt_year))

    print("\n" + "=" * 80)
    print("WALKTHROUGH TABLE 13: STEP 2.F - WHOLE-RECORD TARGET EXTREMES BY DEPTH (132 MONTHS)")
    print("=" * 80)
    df_tgt_ext = pd.read_csv(os.path.join(out_dir, "target_extremes_by_depth.csv"))
    print(df_to_markdown(df_tgt_ext))

    print("\n" + "=" * 80)
    print("WALKTHROUGH: STEP 2.H - hot_end_audit_summary.json (VERBATIM IN FULL)")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GLORYS Hot-End Audit (2010-2020).")
    parser.add_argument("--out-dir", default="outputs/diagnostics/hot_end_audit", help="Output directory")
    parser.add_argument("--report-only", action="store_true", help="Print walkthrough tables from existing outputs")

    args = parser.parse_args()

    if args.report_only:
        print_walkthrough_tables(args.out_dir)
    else:
        run_hot_end_audit(out_dir=args.out_dir)
        print_walkthrough_tables(args.out_dir)


if __name__ == "__main__":
    main()

