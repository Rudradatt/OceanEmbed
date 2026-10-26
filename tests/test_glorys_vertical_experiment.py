"""
Unit and synthetic tests for GLORYS Vertical Interpolation Consequence Experiment.

Verifies:
1. A profile linear in depth gives exactly the same result with V1 and V2.
2. A quadratic profile gives different V1 and V2 results, and V2 follows the
   analytic curve more closely (verified with exact known numbers).
3. V1 is invalid (NaN) when either bracketing level is invalid.
4. V2 uses only the contiguous valid prefix and is NaN when the prefix does
   not reach the upper bracket.
5. A target depth equal to a level depth returns that level's value (exact equality).
6. Depth 0 handling (V0a and V0b) on a known profile with known linear slope.
7. Raw data files exist and their SHA256 hashes are strictly unchanged.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.diagnostics.glorys_vertical_experiment import (
    bracket_search,
    build_bracket_table,
    compute_file_sha256,
    find_contiguous_valid_prefix_k,
    interpolate_v0a,
    interpolate_v0b,
    interpolate_v1,
    interpolate_v2_pchip_vectorized,
    DEFAULT_THETAO_PATH,
    DEFAULT_STATIC_PATH,
    EXPECTED_SHA256_THETAO,
    EXPECTED_SHA256_STATIC,
)


# =============================================================================
# Test 1: Linear Profile Identity (V1 == V2)
# =============================================================================

def test_linear_profile_identity_v1_v2():
    """Verify that a linear vertical profile yields identical results for V1 and V2."""
    level_depths = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    nz = len(level_depths)

    # Linear profile: T(z) = -0.05 * z + 26.0
    a = -0.05
    b = 26.0
    true_vals = a * level_depths + b

    # Create synthetic 3D field of shape (nz, 2, 2)
    ny, nx = 2, 2
    vals_3d = np.tile(true_vals[:, np.newaxis, np.newaxis], (1, ny, nx))
    valid_levels_3d = np.ones((nz, ny, nx), dtype=bool)

    target_depths = [1.0, 3.5, 7.5, 15.0, 35.0, 75.0]
    bracket_dict = {z_t: bracket_search(level_depths, z_t) for z_t in target_depths}

    # Evaluate V2 via vectorized PCHIP
    v2_dict = interpolate_v2_pchip_vectorized(
        level_depths, vals_3d, valid_levels_3d, target_depths, bracket_dict
    )

    # Evaluate V1 via linear interpolation
    for z_t in target_depths:
        br = bracket_dict[z_t]
        k0, k1 = br["k0"], br["k1"]
        w = br["weight_k1"]
        v1_val = interpolate_v1(
            vals_3d[k0], vals_3d[k1], valid_levels_3d[k0], valid_levels_3d[k1], w
        )
        v2_val = v2_dict[z_t]
        expected_val = a * z_t + b

        # Assert V1 and V2 are equal to machine precision
        assert np.allclose(v1_val, v2_val, atol=1e-12), f"V1 != V2 at depth {z_t}"
        assert np.allclose(v1_val, expected_val, atol=1e-12), f"V1 != analytic at depth {z_t}"


# =============================================================================
# Test 2: Quadratic Profile Comparison & Known Numbers
# =============================================================================

def test_quadratic_profile_v1_v2_analytic_comparison():
    """Verify that a quadratic profile produces different V1 and V2 results,
    and V2 follows the analytic curve more closely with exact known numbers.
    
    Analytic curve: T(z) = 30.0 - 0.1 * z - 0.001 * z^2
    Levels: [0.0, 20.0, 50.0, 100.0, 150.0]
    Target depth: z_t = 35.0 (bracketed by levels 1 [20.0 m] and 2 [50.0 m])
    
    Known numbers:
      T(20) = 27.600000
      T(50) = 22.500000
      Analytic T(35) = 25.275000
      V1(35) = 0.5 * (27.6 + 22.5) = 25.050000 (error = 0.225000)
      V2(35) = 25.275479 (error = 0.000479)
      V2 error < V1 error by ~470x
    """
    level_depths = np.array([0.0, 20.0, 50.0, 100.0, 150.0])
    nz = len(level_depths)

    def T(depth):
        return 30.0 - 0.1 * depth - 0.001 * depth ** 2

    vals_1d = T(level_depths)
    vals_3d = vals_1d[:, np.newaxis, np.newaxis]  # shape (5, 1, 1)
    valid_3d = np.ones((nz, 1, 1), dtype=bool)

    z_t = 35.0
    bracket_dict = {z_t: bracket_search(level_depths, z_t)}
    br = bracket_dict[z_t]

    assert br["k0"] == 1
    assert br["k1"] == 2
    assert np.isclose(br["weight_k1"], 0.5, atol=1e-12)

    # Compute V1
    v1_res = interpolate_v1(
        vals_3d[br["k0"]], vals_3d[br["k1"]],
        valid_3d[br["k0"]], valid_3d[br["k1"]],
        br["weight_k1"]
    )
    v1_val = float(v1_res[0, 0])

    # Compute V2
    v2_dict = interpolate_v2_pchip_vectorized(
        level_depths, vals_3d, valid_3d, [z_t], bracket_dict
    )
    v2_val = float(v2_dict[z_t][0, 0])

    analytic_val = T(z_t)
    err1 = abs(v1_val - analytic_val)
    err2 = abs(v2_val - analytic_val)

    # Verify exact known numbers
    assert np.isclose(analytic_val, 25.275000, atol=1e-6)
    assert np.isclose(v1_val, 25.050000, atol=1e-6)
    assert np.isclose(v2_val, 25.275479, atol=1e-5)
    assert np.isclose(err1, 0.225000, atol=1e-6)
    assert np.isclose(err2, 0.000479, atol=1e-5)

    # V2 follows analytic curve much more closely than V1
    assert err2 < err1
    assert err1 > 400 * err2


# =============================================================================
# Test 3: V1 Invalid (NaN) When Either Bracketing Level is Invalid
# =============================================================================

def test_v1_nan_when_either_bracket_invalid():
    """Verify that V1 produces NaN if either or both bracketing levels are invalid."""
    vals_k0 = np.array([[20.0]])
    vals_k1 = np.array([[10.0]])
    w = 0.4

    # Both valid -> valid result
    res_both_valid = interpolate_v1(vals_k0, vals_k1, np.array([[True]]), np.array([[True]]), w)
    assert not np.isnan(res_both_valid[0, 0])
    assert np.isclose(res_both_valid[0, 0], 16.0)

    # Shallower (k0) invalid -> NaN
    res_k0_invalid = interpolate_v1(vals_k0, vals_k1, np.array([[False]]), np.array([[True]]), w)
    assert np.isnan(res_k0_invalid[0, 0])

    # Deeper (k1) invalid -> NaN
    res_k1_invalid = interpolate_v1(vals_k0, vals_k1, np.array([[True]]), np.array([[False]]), w)
    assert np.isnan(res_k1_invalid[0, 0])

    # Both invalid -> NaN
    res_both_invalid = interpolate_v1(vals_k0, vals_k1, np.array([[False]]), np.array([[False]]), w)
    assert np.isnan(res_both_invalid[0, 0])


# =============================================================================
# Test 4: V2 Contiguous Valid Prefix and Prefix Stopping Rules
# =============================================================================

def test_v2_contiguous_valid_prefix_and_nan_when_prefix_short():
    """Verify that V2 uses only the contiguous valid prefix (0..K) and produces NaN
    when the prefix does not reach the upper bracketing level k1."""
    level_depths = np.array([0.5, 2.0, 5.0, 10.0, 20.0])  # 5 levels (0..4)
    nz = len(level_depths)
    vals_3d = np.full((nz, 4, 1), 20.0, dtype=np.float64)

    # Create 4 columns with different validity patterns:
    # Col 0: all 5 levels valid -> K = 4
    # Col 1: levels 0, 1, 2 valid, level 3 invalid -> K = 2
    # Col 2: level 0 valid, level 1 invalid, levels 2, 3 valid -> K = 0 (breaks at 1)
    # Col 3: level 0 invalid -> K = -1
    valid_mask = np.zeros((nz, 4, 1), dtype=bool)
    valid_mask[:, 0, 0] = [True, True, True, True, True]
    valid_mask[:, 1, 0] = [True, True, True, False, False]
    valid_mask[:, 2, 0] = [True, False, True, True, True]
    valid_mask[:, 3, 0] = [False, True, True, True, True]

    K = find_contiguous_valid_prefix_k(valid_mask)
    assert K[0, 0] == 4
    assert K[1, 0] == 2
    assert K[2, 0] == 0
    assert K[3, 0] == -1

    # Target depth z_t = 7.0 (between level 2 [5.0 m] and level 3 [10.0 m])
    # Bracketing levels: k0 = 2, k1 = 3
    z_t = 7.0
    bracket_dict = {z_t: bracket_search(level_depths, z_t)}
    assert bracket_dict[z_t]["k0"] == 2
    assert bracket_dict[z_t]["k1"] == 3

    v2_dict = interpolate_v2_pchip_vectorized(
        level_depths, vals_3d, valid_mask, [z_t], bracket_dict
    )
    v2_col = v2_dict[z_t][:, 0]

    # Col 0: K = 4 >= 3 -> valid
    assert not np.isnan(v2_col[0])

    # Col 1: K = 2 < 3 -> prefix does not reach upper bracket k1=3 -> NaN
    assert np.isnan(v2_col[1])

    # Col 2: K = 0 < 3 -> prefix broke at level 1 -> NaN
    assert np.isnan(v2_col[2])

    # Col 3: K = -1 -> surface level 0 invalid -> NaN
    assert np.isnan(v2_col[3])


# =============================================================================
# Test 5: Exact Level Depth Equality
# =============================================================================

def test_exact_level_depth_equality():
    """Verify that a target depth exactly equal to a level depth returns that level's value."""
    level_depths = np.array([0.494, 5.0, 10.0, 20.0, 50.0])
    target_depth = 10.0  # matches index 2 exactly

    b = bracket_search(level_depths, target_depth)
    assert b["is_exact"] is True
    assert b["k0"] == 2
    assert b["k1"] == 2
    assert b["dist_nearest_m"] == 0.0

    vals_3d = np.array([15.0, 14.0, 13.25, 12.0, 10.0]).reshape(5, 1, 1)
    valid_3d = np.ones((5, 1, 1), dtype=bool)

    # Test V1 with is_exact
    v1_res = interpolate_v1(
        vals_3d[b["k0"]], vals_3d[b["k1"]],
        valid_3d[b["k0"]], valid_3d[b["k1"]],
        b["weight_k1"], is_exact=True
    )
    assert np.isclose(v1_res[0, 0], 13.25)

    # Test V2 with is_exact
    v2_dict = interpolate_v2_pchip_vectorized(
        level_depths, vals_3d, valid_3d, [target_depth], {target_depth: b}
    )
    assert np.isclose(v2_dict[target_depth][0, 0], 13.25)


# =============================================================================
# Test 6: Depth 0 Handling (V0a and V0b) on Known Profile
# =============================================================================

def test_depth0_v0a_v0b_known_profile():
    """Verify Depth 0 rules V0a and V0b on a profile with known surface slope.
    
    Levels: z0 = 0.5 m, z1 = 1.5 m
    Temperatures: T(z0) = 28.0 °C, T(z1) = 27.0 °C
    Linear slope: m = (27.0 - 28.0) / (1.5 - 0.5) = -1.0 °C / m
    V0a = T(z0) = 28.0 °C
    V0b = linear extrap to 0 m = 28.0 - (-1.0) * 0.5 = 28.5 °C
    Difference: V0b - V0a = +0.5 °C
    """
    z0 = 0.5
    z1 = 1.5
    val0 = np.array([[28.0]])
    val1 = np.array([[27.0]])

    # Both valid
    v_true = np.array([[True]])
    v0a = interpolate_v0a(val0, v_true)
    v0b = interpolate_v0b(val0, val1, v_true, v_true, z0, z1)

    assert np.isclose(v0a[0, 0], 28.0)
    assert np.isclose(v0b[0, 0], 28.5)
    assert np.isclose(v0b[0, 0] - v0a[0, 0], 0.5)

    # Level 0 invalid -> both NaN
    v_false = np.array([[False]])
    v0a_inv = interpolate_v0a(val0, v_false)
    v0b_inv = interpolate_v0b(val0, val1, v_false, v_true, z0, z1)
    assert np.isnan(v0a_inv[0, 0])
    assert np.isnan(v0b_inv[0, 0])

    # Level 0 valid, level 1 invalid -> V0a valid, V0b NaN
    v0b_l1_inv = interpolate_v0b(val0, val1, v_true, v_false, z0, z1)
    assert np.isnan(v0b_l1_inv[0, 0])


# =============================================================================
# Test 7: Raw Data Files Exist and SHA256 Unchanged
# =============================================================================

def test_raw_data_sha256_unmodified():
    """Verify raw GLORYS NetCDF files exist and their SHA256 checksums match expected."""
    assert os.path.exists(DEFAULT_THETAO_PATH), f"Missing: {DEFAULT_THETAO_PATH}"
    assert os.path.exists(DEFAULT_STATIC_PATH), f"Missing: {DEFAULT_STATIC_PATH}"

    sha_thetao = compute_file_sha256(DEFAULT_THETAO_PATH)
    sha_static = compute_file_sha256(DEFAULT_STATIC_PATH)

    assert sha_thetao == EXPECTED_SHA256_THETAO, (
        f"thetao SHA256 mismatch: got {sha_thetao}, expected {EXPECTED_SHA256_THETAO}"
    )
    assert sha_static == EXPECTED_SHA256_STATIC, (
        f"static SHA256 mismatch: got {sha_static}, expected {EXPECTED_SHA256_STATIC}"
    )
