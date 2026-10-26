#!/usr/bin/env python3
"""
GLORYS Vertical Interpolation Leave-One-Level-Out (LOO) Accuracy Test
(Diagnostic Only; Comparative Evidence Gathering; Read-Only on Raw Data;
 Do Not Recommend a Method)

This script conducts a leave-one-level-out (LOO) diagnostic experiment across
all 31 days of January 2019 GLORYS data to compare the predictive accuracy
of Method V1 (Linear) and Method V2 (PCHIP) vertical interpolation.

Test Design:
  For each interior level j = 1..34, each target column, and each day:
    - truth: Method A area-weighted value at level j (valid if f_T(j) >= tauT).
    - V1-LOO: linear interpolation between levels j-1 and j+1 (valid only if
      both levels j-1 and j+1 have f_T >= tauT).
    - V2-LOO: PCHIP (monotone cubic Hermite) through the contiguous valid prefix
      of levels with level j REMOVED from the nodes; defined only if level j+1
      is in the contiguous valid prefix.
    - error: prediction - truth.

Outputs generated in outputs/diagnostics/glorys_vertical_loo/:
  - loo_by_level.csv
  - loo_by_gradient_tercile.csv
  - loo_summary.json
  - rmse_vs_level_depth.png
  - rmse_ratio_vs_level_depth.png
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

# Ensure repository root is in sys.path
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
    PROJECT_ROOT, "outputs", "diagnostics", "glorys_vertical_loo"
)

BAND_90_330_MIN_M = 90.0
BAND_90_330_MAX_M = 330.0


# =============================================================================
# File Integrity & Metadata Helpers
# =============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file without loading entire file into RAM."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """Retrieve SHA256, size, and modification timestamps for a file."""
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
    """Strictly verify SHA256 hashes of raw input data before and after execution."""
    meta_thetao = get_file_metadata(thetao_path)
    meta_static = get_file_metadata(static_path)

    if meta_thetao["sha256"] != EXPECTED_SHA256_THETAO:
        raise ValueError(
            f"Integrity check failed: thetao SHA256 {meta_thetao['sha256']} != expected {EXPECTED_SHA256_THETAO}"
        )
    if meta_static["sha256"] != EXPECTED_SHA256_STATIC:
        raise ValueError(
            f"Integrity check failed: static SHA256 {meta_static['sha256']} != expected {EXPECTED_SHA256_STATIC}"
        )

    return meta_thetao, meta_static


# =============================================================================
# Core LOO Prediction Functions (Step 1)
# =============================================================================

def find_contiguous_valid_prefix_k(valid_levels: np.ndarray) -> np.ndarray:
    """Compute K, the last level of contiguous valid prefix (0..K) for each column.

    Parameters:
        valid_levels: bool array of shape (nz, ...) indicating valid levels (f_T >= tauT).

    Returns:
        K: int64 array of shape (...) where K is -1 if level 0 is invalid,
           otherwise K is the maximum index such that levels 0..K are all valid.
    """
    cum_valid = np.cumprod(valid_levels.astype(np.int8), axis=0)
    prefix_length = np.sum(cum_valid, axis=0)
    return prefix_length - 1


def compute_v1_loo_at_level(
    vals: np.ndarray,
    valid_mask: np.ndarray,
    level_depths: np.ndarray,
    j: int,
) -> np.ndarray:
    """Compute V1-LOO (linear interpolation between j-1 and j+1) at level j.

    Valid only if both level j-1 and level j+1 are valid (valid_mask is True).
    Evaluated at depth level_depths[j].

    Parameters:
        vals: float array of shape (nz, ...) with level values.
        valid_mask: bool array of shape (nz, ...) indicating valid levels.
        level_depths: 1D array of shape (nz,) with monotonic depths.
        j: interior level index (1 <= j <= nz - 2).

    Returns:
        v1_pred: float array of shape (...) containing V1-LOO predictions (NaN where invalid).
    """
    nz = vals.shape[0]
    if j < 1 or j > nz - 2:
        raise ValueError(f"Level index j={j} must be an interior level (1 <= j <= {nz - 2}).")

    z_prev = float(level_depths[j - 1])
    z_j = float(level_depths[j])
    z_next = float(level_depths[j + 1])

    w = (z_j - z_prev) / (z_next - z_prev)
    pred = (1.0 - w) * vals[j - 1] + w * vals[j + 1]
    valid = valid_mask[j - 1] & valid_mask[j + 1]

    return np.where(valid, pred, np.nan)


def compute_all_v1_loo(
    vals: np.ndarray,
    valid_mask: np.ndarray,
    level_depths: np.ndarray,
) -> np.ndarray:
    """Compute V1-LOO for all interior levels j = 1..nz-2.

    Parameters:
        vals: float array of shape (nz, ...) with level values.
        valid_mask: bool array of shape (nz, ...) indicating valid levels.
        level_depths: 1D array of shape (nz,) with monotonic depths.

    Returns:
        v1_all: float array of shape (nz, ...) with V1-LOO predictions at interior
                levels j = 1..nz-2, and NaN at boundary levels j = 0 and j = nz-1.
    """
    orig_shape = vals.shape
    nz = orig_shape[0]
    level_depths = np.asarray(level_depths, dtype=np.float64)

    out = np.full(orig_shape, np.nan, dtype=np.float64)
    for j in range(1, nz - 1):
        out[j] = compute_v1_loo_at_level(vals, valid_mask, level_depths, j)
    return out


def compute_v2_loo_column(
    vals: np.ndarray,
    valid_mask: np.ndarray,
    level_depths: np.ndarray,
    j: int,
) -> float:
    """Compute V2-LOO for a single 1D column profile at interior level j.

    PCHIP through the contiguous valid prefix of levels with level j REMOVED;
    defined only if level j+1 is in the contiguous valid prefix (K >= j+1).

    Parameters:
        vals: 1D float array of shape (nz,).
        valid_mask: 1D bool array of shape (nz,).
        level_depths: 1D float array of shape (nz,).
        j: interior level index (1 <= j <= nz - 2).

    Returns:
        float prediction, or NaN if undefined.
    """
    nz = len(vals)
    if j < 1 or j > nz - 2:
        return np.nan

    cum_valid = np.cumprod(valid_mask.astype(np.int8))
    k_val = int(np.sum(cum_valid) - 1)

    # Defined only if level j+1 is in the prefix
    if k_val < j + 1:
        return np.nan

    # Prefix nodes: 0..k_val with level j removed
    active_depths = np.delete(level_depths[: k_val + 1], j)
    active_vals = np.delete(vals[: k_val + 1], j)

    pchip = PchipInterpolator(active_depths, active_vals)
    return float(pchip(level_depths[j]))


def compute_all_v2_loo(
    vals: np.ndarray,
    valid_mask: np.ndarray,
    level_depths: np.ndarray,
) -> np.ndarray:
    """Compute V2-LOO for all interior levels j = 1..nz-2 vectorized across columns.

    PCHIP through the contiguous valid prefix (0..K) with level j REMOVED from the nodes;
    defined only if level j+1 is in the prefix (K >= j+1).
    Columns are grouped by prefix length K to vectorize PchipInterpolator across axis=0.

    Parameters:
        vals: float array of shape (nz, ...) with level values.
        valid_mask: bool array of shape (nz, ...) indicating valid levels (f_T >= tauT).
        level_depths: 1D float array of shape (nz,) with monotonic depths.

    Returns:
        v2_all: float array of shape (nz, ...) with V2-LOO predictions at interior
                levels j = 1..nz-2, and NaN at boundary levels or where undefined.
    """
    orig_shape = vals.shape
    nz = orig_shape[0]
    level_depths = np.asarray(level_depths, dtype=np.float64)

    # Flatten spatial dimensions to (nz, n_cols)
    vals_2d = vals.reshape(nz, -1)
    valid_2d = valid_mask.reshape(nz, -1)
    n_cols = vals_2d.shape[1]

    # Find contiguous valid prefix length K for each column
    K_flat = find_contiguous_valid_prefix_k(valid_2d)

    out = np.full((nz, n_cols), np.nan, dtype=np.float64)
    unique_K = np.unique(K_flat)

    for k_val in unique_K:
        # Need at least K >= 2 for interior level j >= 1 with j+1 <= K
        if k_val < 2:
            continue

        col_indices = np.where(K_flat == k_val)[0]
        if len(col_indices) == 0:
            continue

        # Interior levels j that satisfy j + 1 <= k_val
        max_j = min(int(k_val) - 1, nz - 2)
        if max_j < 1:
            continue

        vals_prefix = vals_2d[: k_val + 1, col_indices]  # shape: (k_val + 1, n_group)
        depths_prefix = level_depths[: k_val + 1]       # shape: (k_val + 1,)

        for j in range(1, max_j + 1):
            # Remove level j from the prefix nodes
            active_depths = np.delete(depths_prefix, j)
            active_vals = np.delete(vals_prefix, j, axis=0)  # shape: (k_val, n_group)

            pchip = PchipInterpolator(active_depths, active_vals, axis=0)
            eval_res = pchip(level_depths[j])  # shape: (n_group,)
            out[j, col_indices] = eval_res

    return out.reshape(orig_shape)


# =============================================================================
# Statistical Accumulation & Terciles (Step 1 & Step 3)
# =============================================================================

def compute_loo_statistics(
    truth: np.ndarray,
    pred_v1: np.ndarray,
    pred_v2: np.ndarray,
) -> Dict[str, Any]:
    """Compute LOO error statistics over cells where both methods and truth are defined.

    error = prediction - truth.
    """
    valid = np.isfinite(truth) & np.isfinite(pred_v1) & np.isfinite(pred_v2)
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n": 0,
            "bias_v1": np.nan,
            "bias_v2": np.nan,
            "rmse_v1": np.nan,
            "rmse_v2": np.nan,
            "mae_v1": np.nan,
            "mae_v2": np.nan,
            "p99_abs_err_v1": np.nan,
            "p99_abs_err_v2": np.nan,
            "max_abs_err_v1": np.nan,
            "max_abs_err_v2": np.nan,
            "ratio_rmse_v2_v1": np.nan,
            "frac_v2_better": np.nan,
        }

    err1 = pred_v1[valid] - truth[valid]
    err2 = pred_v2[valid] - truth[valid]
    abs_err1 = np.abs(err1)
    abs_err2 = np.abs(err2)

    rmse1 = float(np.sqrt(np.mean(err1 ** 2)))
    rmse2 = float(np.sqrt(np.mean(err2 ** 2)))
    ratio = float(rmse2 / rmse1) if rmse1 > 0 else np.nan
    frac_better = float(np.mean(abs_err2 < abs_err1))

    return {
        "n": n,
        "bias_v1": float(np.mean(err1)),
        "bias_v2": float(np.mean(err2)),
        "rmse_v1": rmse1,
        "rmse_v2": rmse2,
        "mae_v1": float(np.mean(abs_err1)),
        "mae_v2": float(np.mean(abs_err2)),
        "p99_abs_err_v1": float(np.percentile(abs_err1, 99)),
        "p99_abs_err_v2": float(np.percentile(abs_err2, 99)),
        "max_abs_err_v1": float(np.max(abs_err1)),
        "max_abs_err_v2": float(np.max(abs_err2)),
        "ratio_rmse_v2_v1": ratio,
        "frac_v2_better": frac_better,
    }


def compute_gradient_terciles(
    err1: np.ndarray,
    err2: np.ndarray,
    grad: np.ndarray,
) -> List[Dict[str, Any]]:
    """Split cells into terciles of local gradient and report RMSE for each tercile.

    Local gradient: g = |T(j+1) - T(j-1)| / (z_{j+1} - z_{j-1}).
    """
    n = len(grad)
    if n < 3:
        return []

    order = np.argsort(grad)
    t1_idx = order[: n // 3]
    t2_idx = order[n // 3 : 2 * (n // 3)]
    t3_idx = order[2 * (n // 3) :]

    terciles_data = [
        (1, "low", t1_idx),
        (2, "medium", t2_idx),
        (3, "high", t3_idx),
    ]

    records = []
    for t_num, t_label, idx in terciles_data:
        n_t = len(idx)
        if n_t == 0:
            continue
        g_t = grad[idx]
        e1_t = err1[idx]
        e2_t = err2[idx]

        rmse1 = float(np.sqrt(np.mean(e1_t ** 2)))
        rmse2 = float(np.sqrt(np.mean(e2_t ** 2)))
        ratio = float(rmse2 / rmse1) if rmse1 > 0 else np.nan
        frac_better = float(np.mean(np.abs(e2_t) < np.abs(e1_t)))

        records.append({
            "tercile": t_num,
            "tercile_label": t_label,
            "n": n_t,
            "grad_min_deg_per_m": float(np.min(g_t)),
            "grad_max_deg_per_m": float(np.max(g_t)),
            "rmse_v1": rmse1,
            "rmse_v2": rmse2,
            "ratio_rmse_v2_v1": ratio,
            "frac_v2_better": frac_better,
        })

    return records


# =============================================================================
# Plotting Helpers (Step 3d)
# =============================================================================

def generate_loo_plots(
    df_level: pd.DataFrame,
    output_dir: str,
) -> Tuple[str, str]:
    """Generate diagnostic plots for LOO accuracy versus depth for tauT = 1.0."""
    os.makedirs(output_dir, exist_ok=True)
    df_1 = df_level[df_level["tau_T"] == 1.0].sort_values("level_j").copy()

    depths = df_1["depth_j_m"].values
    rmse_v1 = df_1["rmse_v1"].values
    rmse_v2 = df_1["rmse_v2"].values
    ratios = df_1["ratio_rmse_v2_v1"].values

    # Plot 1: RMSE vs Level Depth
    plot1_path = os.path.join(output_dir, "rmse_vs_level_depth.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(depths, rmse_v1, marker="o", color="#1f77b4", linewidth=2.0, label="V1-LOO (Linear)")
    ax.plot(depths, rmse_v2, marker="s", color="#ff7f0e", linewidth=2.0, label="V2-LOO (PCHIP)")
    ax.axvspan(BAND_90_330_MIN_M, BAND_90_330_MAX_M, color="#2ca02c", alpha=0.15, label="Target Band (90-330 m)")
    ax.set_title("GLORYS Vertical LOO Accuracy: RMSE vs Level Depth (tauT = 1.0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("GLORYS Level Depth (m)", fontsize=11)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(plot1_path)
    plt.close(fig)

    # Plot 2: Ratio RMSE(V2) / RMSE(V1) vs Level Depth
    plot2_path = os.path.join(output_dir, "rmse_ratio_vs_level_depth.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(depths, ratios, marker="^", color="#d62728", linewidth=2.0, label="Ratio RMSE_V2 / RMSE_V1")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Parity (V2 = V1)")
    ax.axvspan(BAND_90_330_MIN_M, BAND_90_330_MAX_M, color="#2ca02c", alpha=0.15, label="Target Band (90-330 m)")
    ax.set_title("GLORYS Vertical LOO Accuracy: Ratio RMSE(V2)/RMSE(V1) vs Level Depth (tauT = 1.0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("GLORYS Level Depth (m)", fontsize=11)
    ax.set_ylabel("Ratio RMSE_V2 / RMSE_V1", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(plot2_path)
    plt.close(fig)

    return plot1_path, plot2_path


# =============================================================================
# Full 31-Day Real-Data Execution Loop (Step 3)
# =============================================================================

def run_vertical_loo_experiment(
    thetao_path: str = DEFAULT_THETAO_PATH,
    static_path: str = DEFAULT_STATIC_PATH,
    policy_path: str = DEFAULT_POLICY_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute the full 31-day vertical leave-one-level-out experiment."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 78)
    print("GLORYS VERTICAL LEAVE-ONE-LEVEL-OUT (LOO) ACCURACY TEST")
    print("  Diagnostic Only; Read-Only on Raw Data; Ranking Evidence Gathering")
    print("=" * 78)

    # 1. Pre-execution metadata capture & strict integrity check
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
    print(f"    mtime: {meta_thetao_before['mtime_ns']} ({meta_thetao_before['mtime']})")
    print(f"  static: {meta_static_before['path']}")
    print(f"    Size: {meta_static_before['size_bytes']} bytes, SHA256: {meta_static_before['sha256']}")
    print(f"    mtime: {meta_static_before['mtime_ns']} ({meta_static_before['mtime']})")

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
    nz = len(level_depths)
    interior_levels = list(range(1, nz - 1))  # 1..34

    # Precompute 2D horizontal overlap weights (Method A)
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

    # 3. Accumulators for errors and gradients across all 31 days
    # Structure: accum[tau_str][j] -> {"err1": [], "err2": [], "grad": []}
    accum: Dict[str, Dict[int, Dict[str, List[np.ndarray]]]] = {
        "1.0": {j: {"err1": [], "err2": [], "grad": []} for j in interior_levels},
        "0.5": {j: {"err1": [], "err2": [], "grad": []} for j in interior_levels},
    }

    print("\nProcessing 31 days of GLORYS thetao (January 2019)...")

    for d in range(31):
        print(f"  Day {d:02d}/30...", end="", flush=True)

        # a. Read raw slice
        raw_day = read_raw_day(thetao_path, day_index=d, policy=policy)

        # b. Apply QC
        qc_res = apply_qc(
            raw_day, static_wet, exclusion_mask, policy,
            lats=lats, lons=lons, depths=level_depths
        )

        # c. Decode valid cells manually after masking fill
        thetao_decoded = np.where(
            qc_res.valid,
            raw_day.astype(np.float64) * scale_factor + add_offset,
            np.nan,
        )

        # d. Horizontal Method A for all 36 levels
        val_A, f = regrid_method_A(thetao_decoded, qc_res.valid, weights_2d=w_2d)

        # Process both validity thresholds: primary tauT = 1.0, secondary tauT = 0.5
        for tau_str, tau_val in [("1.0", 1.0), ("0.5", 0.5)]:
            valid_mask = (f >= tau_val - 1e-12)

            # Compute LOO predictions for all interior levels
            v1_all = compute_all_v1_loo(val_A, valid_mask, level_depths)
            v2_all = compute_all_v2_loo(val_A, valid_mask, level_depths)

            for j in interior_levels:
                # Truth is val_A[j]
                # Defined where truth, v1, and v2 are all finite
                truth_j = val_A[j]
                v1_j = v1_all[j]
                v2_j = v2_all[j]

                mask_both = np.isfinite(truth_j) & np.isfinite(v1_j) & np.isfinite(v2_j)
                if not np.any(mask_both):
                    continue

                e1 = (v1_j[mask_both] - truth_j[mask_both]).astype(np.float64)
                e2 = (v2_j[mask_both] - truth_j[mask_both]).astype(np.float64)

                # Local gradient: |T(j+1) - T(j-1)| / (z_{j+1} - z_{j-1})
                t_prev = val_A[j - 1, mask_both]
                t_next = val_A[j + 1, mask_both]
                dz_span = float(level_depths[j + 1] - level_depths[j - 1])
                g = (np.abs(t_next - t_prev) / dz_span).astype(np.float64)

                accum[tau_str][j]["err1"].append(e1)
                accum[tau_str][j]["err2"].append(e2)
                accum[tau_str][j]["grad"].append(g)

        print(" done.")

    # 4. Aggregate statistics across all 31 days
    print("\nAggregating LOO statistics across all 31 days...")

    records_level: List[Dict[str, Any]] = []
    records_tercile: List[Dict[str, Any]] = []

    summary_by_tau: Dict[str, Dict[str, Any]] = {}

    for tau_str, tau_val in [("1.0", 1.0), ("0.5", 0.5)]:
        n_better_total = 0
        n_band_total = 0
        n_band_better = 0
        max_ratio = -1.0
        max_ratio_j = -1
        max_ratio_depth = np.nan

        for j in interior_levels:
            z_prev = float(level_depths[j - 1])
            z_j = float(level_depths[j])
            z_next = float(level_depths[j + 1])
            pos_span = (z_j - z_prev) / (z_next - z_prev)

            e1_parts = accum[tau_str][j]["err1"]
            e2_parts = accum[tau_str][j]["err2"]
            g_parts = accum[tau_str][j]["grad"]

            if len(e1_parts) > 0:
                e1 = np.concatenate(e1_parts)
                e2 = np.concatenate(e2_parts)
                g = np.concatenate(g_parts)

                n_cells = len(e1)
                abs_e1 = np.abs(e1)
                abs_e2 = np.abs(e2)

                rmse1 = float(np.sqrt(np.mean(e1 ** 2)))
                rmse2 = float(np.sqrt(np.mean(e2 ** 2)))
                ratio = float(rmse2 / rmse1) if rmse1 > 0 else np.nan
                frac_better = float(np.mean(abs_e2 < abs_e1))

                rec_lvl = {
                    "tau_T": tau_val,
                    "level_j": j,
                    "depth_prev_m": z_prev,
                    "depth_j_m": z_j,
                    "depth_next_m": z_next,
                    "pos_in_span": pos_span,
                    "n": n_cells,
                    "bias_v1": float(np.mean(e1)),
                    "bias_v2": float(np.mean(e2)),
                    "rmse_v1": rmse1,
                    "rmse_v2": rmse2,
                    "mae_v1": float(np.mean(abs_e1)),
                    "mae_v2": float(np.mean(abs_e2)),
                    "p99_abs_err_v1": float(np.percentile(abs_e1, 99)),
                    "p99_abs_err_v2": float(np.percentile(abs_e2, 99)),
                    "max_abs_err_v1": float(np.max(abs_e1)),
                    "max_abs_err_v2": float(np.max(abs_e2)),
                    "ratio_rmse_v2_v1": ratio,
                    "frac_v2_better": frac_better,
                }
                records_level.append(rec_lvl)

                # Summary tracking
                if rmse2 < rmse1:
                    n_better_total += 1

                in_band = (BAND_90_330_MIN_M <= z_j <= BAND_90_330_MAX_M)
                if in_band:
                    n_band_total += 1
                    if rmse2 < rmse1:
                        n_band_better += 1

                if ratio > max_ratio:
                    max_ratio = ratio
                    max_ratio_j = j
                    max_ratio_depth = z_j

                # Gradient terciles
                t_recs = compute_gradient_terciles(e1, e2, g)
                for tr in t_recs:
                    records_tercile.append({
                        "tau_T": tau_val,
                        "level_j": j,
                        "depth_j_m": z_j,
                        **tr,
                    })

        summary_by_tau[tau_str] = {
            "tau_T": tau_val,
            "n_interior_levels_total": len(interior_levels),
            "n_levels_rmse_v2_lt_v1": n_better_total,
            "n_band_90_330_total": n_band_total,
            "n_band_90_330_rmse_v2_lt_v1": n_band_better,
            "max_ratio_rmse_v2_v1": max_ratio,
            "max_ratio_level_j": max_ratio_j,
            "max_ratio_depth_m": max_ratio_depth,
        }

    # 5. Save CSV files
    df_level = pd.DataFrame(records_level)
    df_tercile = pd.DataFrame(records_tercile)

    csv_level_path = os.path.join(output_dir, "loo_by_level.csv")
    csv_tercile_path = os.path.join(output_dir, "loo_by_gradient_tercile.csv")

    df_level.to_csv(csv_level_path, index=False)
    df_tercile.to_csv(csv_tercile_path, index=False)
    print(f"  Saved: {csv_level_path}")
    print(f"  Saved: {csv_tercile_path}")

    # 6. Generate diagnostic plots
    plot1_path, plot2_path = generate_loo_plots(df_level, output_dir)
    print(f"  Saved: {plot1_path}")
    print(f"  Saved: {plot2_path}")

    # 7. Post-execution metadata capture & strict integrity check
    meta_thetao_after = get_file_metadata(thetao_path)
    meta_static_after = get_file_metadata(static_path)

    if meta_thetao_after["sha256"] != EXPECTED_SHA256_THETAO:
        raise ValueError(
            f"Post-check failed: thetao SHA256 {meta_thetao_after['sha256']} != expected {EXPECTED_SHA256_THETAO}"
        )
    if meta_static_after["sha256"] != EXPECTED_SHA256_STATIC:
        raise ValueError(
            f"Post-check failed: static SHA256 {meta_static_after['sha256']} != expected {EXPECTED_SHA256_STATIC}"
        )
    if meta_thetao_after["size_bytes"] != meta_thetao_before["size_bytes"]:
        raise ValueError("thetao size changed!")
    if meta_static_after["size_bytes"] != meta_static_before["size_bytes"]:
        raise ValueError("static size changed!")
    if meta_thetao_after["mtime_ns"] != meta_thetao_before["mtime_ns"]:
        raise ValueError("thetao mtime changed!")
    if meta_static_after["mtime_ns"] != meta_static_before["mtime_ns"]:
        raise ValueError("static mtime changed!")

    print("\nPost-Execution File Verification (All Match Strictly):")
    print(f"  thetao: SHA256 {meta_thetao_after['sha256']}, Size {meta_thetao_after['size_bytes']}, mtime {meta_thetao_after['mtime_ns']}")
    print(f"  static: SHA256 {meta_static_after['sha256']}, Size {meta_static_after['size_bytes']}, mtime {meta_static_after['mtime_ns']}")

    # 8. Identify 90-330 m band level indices
    band_levels = [
        int(j) for j in interior_levels
        if BAND_90_330_MIN_M <= float(level_depths[j]) <= BAND_90_330_MAX_M
    ]

    # 9. Build summary JSON
    summary_data = {
        "files_before": {
            "thetao": meta_thetao_before,
            "static": meta_static_before,
        },
        "files_after": {
            "thetao": meta_thetao_after,
            "static": meta_static_after,
        },
        "band_90_330_m_levels": band_levels,
        "results_by_tauT": summary_by_tau,
        "unknown_fields": {
            "recommended_method": {
                "value": None,
                "reason": (
                    "DIAGNOSTIC ONLY; ranking purpose only; leave-one-level-out "
                    "doubles grid spacing and overestimates error; method recommendation "
                    "prohibited by test protocol."
                ),
            },
            "argo_insitu_comparison": {
                "value": None,
                "reason": "No Argo or external data used per task constraints.",
            },
        },
    }

    summary_json_path = os.path.join(output_dir, "loo_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved: {summary_json_path}")

    return {
        "df_level": df_level,
        "df_tercile": df_tercile,
        "summary": summary_data,
        "csv_level_path": csv_level_path,
        "csv_tercile_path": csv_tercile_path,
        "plot1_path": plot1_path,
        "plot2_path": plot2_path,
        "summary_json_path": summary_json_path,
    }


# =============================================================================
# Formatted Walkthrough Printers (Step 5 of Task)
# =============================================================================

def print_walkthrough_data(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Print the exact walkthrough tables and JSON directly from saved artifacts."""
    csv_level_path = os.path.join(output_dir, "loo_by_level.csv")
    csv_tercile_path = os.path.join(output_dir, "loo_by_gradient_tercile.csv")
    summary_json_path = os.path.join(output_dir, "loo_summary.json")

    if not os.path.exists(csv_level_path):
        raise FileNotFoundError(f"Missing {csv_level_path}; run experiment first.")

    df_level = pd.read_csv(csv_level_path)
    df_tercile = pd.read_csv(csv_tercile_path)
    with open(summary_json_path, "r") as f:
        summary = json.load(f)

    # 1. loo_by_level for all interior levels at tauT = 1.0
    print("=" * 120)
    print("TABLE 1: LOO ACCURACY BY LEVEL (tauT = 1.0; ALL 34 INTERIOR LEVELS)")
    print("=" * 120)
    df_l1 = df_level[df_level["tau_T"] == 1.0].sort_values("level_j")
    cols_l1 = [
        "level_j", "depth_j_m", "depth_prev_m", "depth_next_m", "pos_in_span",
        "n", "bias_v1", "bias_v2", "rmse_v1", "rmse_v2",
        "ratio_rmse_v2_v1", "mae_v1", "mae_v2",
        "p99_abs_err_v1", "p99_abs_err_v2", "max_abs_err_v1", "max_abs_err_v2",
        "frac_v2_better",
    ]
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_rows", 50)
    print(df_l1[cols_l1].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # 2. Gradient terciles for levels in 90-330 m band at tauT = 1.0
    print("\n" + "=" * 120)
    print("TABLE 2: GRADIENT TERCILES FOR LEVELS IN 90-330 m BAND (tauT = 1.0)")
    print("=" * 120)
    df_t1 = df_tercile[
        (df_tercile["tau_T"] == 1.0) &
        (df_tercile["depth_j_m"] >= BAND_90_330_MIN_M) &
        (df_tercile["depth_j_m"] <= BAND_90_330_MAX_M)
    ].sort_values(["level_j", "tercile"])
    cols_t1 = [
        "level_j", "depth_j_m", "tercile", "tercile_label", "n",
        "grad_min_deg_per_m", "grad_max_deg_per_m",
        "rmse_v1", "rmse_v2", "ratio_rmse_v2_v1", "frac_v2_better",
    ]
    print(df_t1[cols_t1].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # 3. Summary counts
    print("\n" + "=" * 120)
    print("SUMMARY COUNTS")
    print("=" * 120)
    for tau_str in ["1.0", "0.5"]:
        res = summary["results_by_tauT"][tau_str]
        print(f"--- tauT = {tau_str} ---")
        print(f"  Interior levels where RMSE_V2 < RMSE_V1: {res['n_levels_rmse_v2_lt_v1']} / {res['n_interior_levels_total']}")
        print(f"  Levels in 90-330 m band where RMSE_V2 < RMSE_V1: {res['n_band_90_330_rmse_v2_lt_v1']} / {res['n_band_90_330_total']}")
        print(f"  Largest ratio RMSE_V2 / RMSE_V1: {res['max_ratio_rmse_v2_v1']:.6f} at Level {res['max_ratio_level_j']} (depth {res['max_ratio_depth_m']:.2f} m)")

    # 4. JSON in full
    print("\n" + "=" * 120)
    print("loo_summary.json (FULL VERBATIM)")
    print("=" * 120)
    print(json.dumps(summary, indent=2))


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GLORYS Vertical Interpolation LOO Accuracy Test")
    parser.add_argument("--data-thetao", default=DEFAULT_THETAO_PATH, help="Path to raw GLORYS thetao NetCDF")
    parser.add_argument("--data-static", default=DEFAULT_STATIC_PATH, help="Path to static bathy NetCDF")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to QC policy JSON")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--print-walkthrough", action="store_true", help="Print walkthrough tables from saved files")

    args = parser.parse_args()

    if args.print_walkthrough:
        print_walkthrough_data(args.output_dir)
    else:
        run_vertical_loo_experiment(
            thetao_path=args.data_thetao,
            static_path=args.data_static,
            policy_path=args.policy,
            output_dir=args.output_dir,
        )
        print_walkthrough_data(args.output_dir)


if __name__ == "__main__":
    main()
