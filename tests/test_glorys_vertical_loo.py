"""
Unit and synthetic tests for GLORYS Vertical Interpolation LOO Accuracy Test.

Verifies according to STEP 2:
 1. A profile linear in depth gives zero LOO error for V1 and V2.
 2. For a quadratic profile T = a z^2 on non-uniform nodes, the V1-LOO error at
    interior node j equals a (z_j - z_{j-1}) (z_{j+1} - z_j) exactly.
 3. V2-LOO matches a direct scipy PchipInterpolator call with node j removed.
 4. NaN handling: V1-LOO is NaN if level j-1 or j+1 is invalid; V2-LOO is NaN if
    level j+1 is not in the prefix.
 5. Raw data files exist and their SHA256 hashes are strictly unchanged.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.diagnostics.glorys_vertical_loo import (
    compute_all_v1_loo,
    compute_all_v2_loo,
    compute_file_sha256,
    compute_gradient_terciles,
    compute_loo_statistics,
    compute_v1_loo_at_level,
    compute_v2_loo_column,
    find_contiguous_valid_prefix_k,
    verify_raw_data_integrity,
    DEFAULT_THETAO_PATH,
    DEFAULT_STATIC_PATH,
    EXPECTED_SHA256_THETAO,
    EXPECTED_SHA256_STATIC,
)


# =============================================================================
# Test 1: Linear Profile Identity (Zero LOO Error for V1 and V2)
# =============================================================================

def test_linear_profile_zero_loo_error():
    """Verify that a linear vertical profile gives zero LOO error for both V1 and V2.

    For T(z) = m * z + c on arbitrary non-uniform nodes:
      V1-LOO at interior node j interpolates linearly between z_{j-1} and z_{j+1},
      reproducing T(z_j) with zero error.
      V2-LOO uses PCHIP with node j removed; PCHIP preserves linear functions exactly,
      reproducing T(z_j) with zero error.
    """
    level_depths = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0])
    nz = len(level_depths)

    m = -0.04
    c = 28.0
    true_vals = m * level_depths + c

    valid_mask_1d = np.ones(nz, dtype=bool)

    # Test 1D column
    v1_1d = compute_all_v1_loo(true_vals, valid_mask_1d, level_depths)
    v2_1d = compute_all_v2_loo(true_vals, valid_mask_1d, level_depths)

    for j in range(1, nz - 1):
        err_v1 = abs(v1_1d[j] - true_vals[j])
        err_v2 = abs(v2_1d[j] - true_vals[j])
        assert err_v1 < 1e-12, f"V1-LOO error {err_v1} at node {j} exceeds 1e-12"
        assert err_v2 < 1e-12, f"V2-LOO error {err_v2} at node {j} exceeds 1e-12"
        assert abs(v1_1d[j] - v2_1d[j]) < 1e-12

    # Boundaries must be NaN
    assert np.isnan(v1_1d[0]) and np.isnan(v1_1d[-1])
    assert np.isnan(v2_1d[0]) and np.isnan(v2_1d[-1])

    # Test 3D grid: shape (nz, 4, 3)
    ny, nx = 4, 3
    vals_3d = np.tile(true_vals[:, np.newaxis, np.newaxis], (1, ny, nx))
    valid_3d = np.ones((nz, ny, nx), dtype=bool)

    v1_3d = compute_all_v1_loo(vals_3d, valid_3d, level_depths)
    v2_3d = compute_all_v2_loo(vals_3d, valid_3d, level_depths)

    for j in range(1, nz - 1):
        assert np.all(np.abs(v1_3d[j] - true_vals[j]) < 1e-12)
        assert np.all(np.abs(v2_3d[j] - true_vals[j]) < 1e-12)


# =============================================================================
# Test 2: Quadratic Profile V1-LOO Exact Analytic Error
# =============================================================================

def test_quadratic_profile_v1_exact_analytic_error():
    """Verify that for a quadratic profile T = a * z^2 on non-uniform nodes,
    the V1-LOO error at interior node j equals a * (z_j - z_{j-1}) * (z_{j+1} - z_j) exactly.

    Proof:
      Let h0 = z_j - z_{j-1}, h1 = z_{j+1} - z_j, so h0 + h1 = z_{j+1} - z_{j-1}.
      V1 prediction at z_j is:
        (h1 * T_{j-1} + h0 * T_{j+1}) / (h0 + h1)
      Substituting T(z) = a * z^2 gives:
        pred - T(z_j) = a * h0 * h1 = a * (z_j - z_{j-1}) * (z_{j+1} - z_j).
    """
    level_depths = np.array([0.0, 5.0, 12.0, 25.0, 45.0, 80.0, 150.0])
    nz = len(level_depths)
    valid_mask = np.ones(nz, dtype=bool)

    for a in [0.002, -0.005, 1.25]:
        vals = a * (level_depths ** 2)

        v1_all = compute_all_v1_loo(vals, valid_mask, level_depths)

        for j in range(1, nz - 1):
            z_prev = level_depths[j - 1]
            z_j = level_depths[j]
            z_next = level_depths[j + 1]

            expected_err = a * (z_j - z_prev) * (z_next - z_j)
            actual_err = v1_all[j] - vals[j]

            # Single-level function check
            single_val = compute_v1_loo_at_level(vals, valid_mask, level_depths, j)
            single_err = single_val - vals[j]

            assert np.isclose(actual_err, expected_err, atol=1e-12), (
                f"V1-LOO error mismatch at level {j}: actual {actual_err} != expected {expected_err}"
            )
            assert np.isclose(single_err, expected_err, atol=1e-12)


# =============================================================================
# Test 3: V2-LOO Matches Direct Scipy PchipInterpolator
# =============================================================================

def test_v2_loo_matches_direct_scipy_pchip():
    """Verify that V2-LOO matches a direct scipy PchipInterpolator call with node j removed.

    Tested both for a single column and across a multi-column grid with differing prefix lengths.
    """
    level_depths = np.array([0.49, 2.65, 5.08, 9.57, 15.81, 25.21, 40.34, 65.81, 109.73])
    nz = len(level_depths)

    # Complex non-polynomial synthetic profile
    t_prof = 25.0 * np.exp(-level_depths / 50.0) + 3.0 * np.sin(level_depths / 20.0)

    # 1. Single column direct comparison
    valid_1d = np.ones(nz, dtype=bool)
    v2_1d = compute_all_v2_loo(t_prof, valid_1d, level_depths)

    for j in range(1, nz - 1):
        # Direct scipy call with node j removed
        active_z = np.delete(level_depths, j)
        active_t = np.delete(t_prof, j)
        pchip = PchipInterpolator(active_z, active_t)
        expected = float(pchip(level_depths[j]))

        # Check compute_v2_loo_column
        col_res = compute_v2_loo_column(t_prof, valid_1d, level_depths, j)
        assert np.isclose(col_res, expected, atol=1e-12)

        # Check compute_all_v2_loo
        assert np.isclose(v2_1d[j], expected, atol=1e-12)

    # 2. Multi-column grid with different prefix lengths K
    # Col 0: prefix K = 8 (full depth)
    # Col 1: prefix K = 5 (levels 0..5 valid, 6..8 invalid)
    # Col 2: prefix K = 3 (levels 0..3 valid, 4..8 invalid)
    vals_grid = np.tile(t_prof[:, np.newaxis], (1, 3))
    # Add column-specific variations to ensure independent evaluation
    vals_grid[:, 1] += 1.5
    vals_grid[:, 2] -= 0.8

    valid_grid = np.ones((nz, 3), dtype=bool)
    valid_grid[6:, 1] = False  # Col 1: K = 5
    valid_grid[4:, 2] = False  # Col 2: K = 3

    v2_grid = compute_all_v2_loo(vals_grid, valid_grid, level_depths)

    # Col 0 (K=8): all interior j=1..7 should match direct pchip
    for j in range(1, 8):
        active_z = np.delete(level_depths, j)
        active_v = np.delete(vals_grid[:, 0], j)
        expected = float(PchipInterpolator(active_z, active_v)(level_depths[j]))
        assert np.isclose(v2_grid[j, 0], expected, atol=1e-12)

    # Col 1 (K=5): interior j=1..4 should match direct pchip; j >= 5 must be NaN
    for j in range(1, 5):
        active_z = np.delete(level_depths[:6], j)
        active_v = np.delete(vals_grid[:6, 1], j)
        expected = float(PchipInterpolator(active_z, active_v)(level_depths[j]))
        assert np.isclose(v2_grid[j, 1], expected, atol=1e-12)
    assert np.isnan(v2_grid[5, 1]), "j=5 has j+1=6 > K=5, must be NaN"
    assert np.isnan(v2_grid[6, 1]), "j=6 has j+1=7 > K=5, must be NaN"

    # Col 2 (K=3): interior j=1..2 should match direct pchip; j >= 3 must be NaN
    for j in range(1, 3):
        active_z = np.delete(level_depths[:4], j)
        active_v = np.delete(vals_grid[:4, 2], j)
        expected = float(PchipInterpolator(active_z, active_v)(level_depths[j]))
        assert np.isclose(v2_grid[j, 2], expected, atol=1e-12)
    assert np.isnan(v2_grid[3, 2]), "j=3 has j+1=4 > K=3, must be NaN"


# =============================================================================
# Test 4: NaN Handling for V1 and V2
# =============================================================================

def test_nan_handling_v1_and_v2():
    """Verify NaN handling rules:
      - V1-LOO is NaN if level j-1 or j+1 is invalid.
      - V2-LOO is NaN if level j+1 is not in the contiguous valid prefix.
    """
    level_depths = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    nz = len(level_depths)
    vals = 25.0 - 0.1 * level_depths

    # Case A: Level j-1 invalid -> V1-LOO at j must be NaN
    mask_a = np.ones(nz, dtype=bool)
    mask_a[1] = False  # level 1 invalid
    v1_a = compute_all_v1_loo(vals, mask_a, level_depths)
    assert np.isnan(v1_a[2]), "V1-LOO at j=2 must be NaN when level 1 (j-1) is invalid"

    # Case B: Level j+1 invalid -> V1-LOO at j must be NaN
    mask_b = np.ones(nz, dtype=bool)
    mask_b[3] = False  # level 3 invalid
    v1_b = compute_all_v1_loo(vals, mask_b, level_depths)
    assert np.isnan(v1_b[2]), "V1-LOO at j=2 must be NaN when level 3 (j+1) is invalid"

    # Case C: Level j invalid, but j-1 and j+1 valid
    # V1 prediction itself is defined (from j-1 and j+1), but LOO error against truth is NaN
    mask_c = np.ones(nz, dtype=bool)
    mask_c[2] = False  # level 2 invalid
    v1_c = compute_all_v1_loo(vals, mask_c, level_depths)
    assert np.isfinite(v1_c[2]), "V1-LOO prediction only requires j-1 and j+1 to be valid"
    # When evaluated against truth, truth is invalid, so error statistics handle this:
    truth_c = np.where(mask_c, vals, np.nan)
    stats_c = compute_loo_statistics(truth_c[2:3], v1_c[2:3], np.array([np.nan]))
    assert stats_c["n"] == 0, "Cell with invalid truth must be excluded from error statistics"

    # Case D: V2-LOO prefix logic
    # 1. Prefix stops before j+1: level 3 is invalid, so K = 2.
    # Level j=2 has j+1 = 3 > K -> V2-LOO at j=2 must be NaN.
    mask_d1 = np.ones(nz, dtype=bool)
    mask_d1[3] = False
    v2_d1 = compute_all_v2_loo(vals, mask_d1, level_depths)
    assert np.isnan(v2_d1[2]), "V2-LOO at j=2 must be NaN when prefix K=2 < j+1=3"
    assert np.isfinite(v2_d1[1]), "V2-LOO at j=1 has j+1=2 <= K=2, so must be finite"

    # 2. Surface gap: level 0 is invalid -> K = -1. All V2-LOO must be NaN.
    mask_d2 = np.ones(nz, dtype=bool)
    mask_d2[0] = False
    v2_d2 = compute_all_v2_loo(vals, mask_d2, level_depths)
    assert np.all(np.isnan(v2_d2)), "All V2-LOO must be NaN when level 0 is invalid (K=-1)"

    # 3. Intermediate gap: level 1 is invalid, but levels 2..6 are valid.
    # The contiguous valid prefix stops at level 0 (K = 0).
    # Even though levels 2, 3, etc. are valid, they are not in the contiguous prefix!
    mask_d3 = np.ones(nz, dtype=bool)
    mask_d3[1] = False
    v2_d3 = compute_all_v2_loo(vals, mask_d3, level_depths)
    assert np.all(np.isnan(v2_d3[1:])), "V2-LOO must be NaN for all j when prefix stops at level 0"


# =============================================================================
# Test 5: Gradient Terciles & Statistics Computation
# =============================================================================

def test_gradient_terciles_and_statistics():
    """Verify that compute_gradient_terciles splits data into 3 equal parts
    and computes RMSE correctly.
    """
    n = 300
    np.random.seed(42)
    grad = np.linspace(0.001, 0.1, n)
    err1 = np.random.normal(0, 0.1, n)
    err2 = np.random.normal(0, 0.05, n)

    terciles = compute_gradient_terciles(err1, err2, grad)
    assert len(terciles) == 3

    # Terciles must have equal sizes (100 each)
    for t in terciles:
        assert t["n"] == 100
        assert t["rmse_v1"] > 0
        assert t["rmse_v2"] > 0
        assert t["ratio_rmse_v2_v1"] > 0

    assert terciles[0]["grad_max_deg_per_m"] <= terciles[1]["grad_min_deg_per_m"]
    assert terciles[1]["grad_max_deg_per_m"] <= terciles[2]["grad_min_deg_per_m"]


# =============================================================================
# Test 6: Raw Data Files Exist and SHA256 Hashes Are Strictly Unchanged
# =============================================================================

def test_raw_data_files_sha256_unmodified():
    """Verify that both raw data files exist and have exact expected SHA256 checksums."""
    meta_thetao, meta_static = verify_raw_data_integrity(
        DEFAULT_THETAO_PATH, DEFAULT_STATIC_PATH
    )

    assert meta_thetao["sha256"] == EXPECTED_SHA256_THETAO, (
        f"thetao SHA256 {meta_thetao['sha256']} != expected {EXPECTED_SHA256_THETAO}"
    )
    assert meta_static["sha256"] == EXPECTED_SHA256_STATIC, (
        f"static SHA256 {meta_static['sha256']} != expected {EXPECTED_SHA256_STATIC}"
    )
    assert meta_thetao["size_bytes"] == 484418616
    assert meta_static["size_bytes"] == 12608300
