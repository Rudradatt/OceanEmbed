"""
Unit and synthetic tests for GLORYS Horizontal Regridding Consequence Experiment.

Verifies:
1. Overlap weights sum to 1.0 and equal [1/6, 1/3, 1/3, 1/6] per axis.
2. A constant field gives exactly the constant for both Method L and Method A.
3. A linear field in lat and lon gives the target-center value for BOTH methods (diff = 0).
4. A quadratic field gives different results for L and A, and A equals a brute-force
   average on a fine grid built inside the test.
5. NaN handling: L is NaN if any of its 4 center cells is invalid; A uses only valid
   cells and f is correct; f = 0 gives NaN.
6. Edge indices at the domain boundary (i=99 uses 300; j=239 uses 720; out-of-domain not counted).
7. Raw data files exist and their SHA256 hashes are strictly unchanged.
"""

import os
import sys
import numpy as np
import pytest

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.diagnostics.glorys_regrid_experiment import (
    compute_file_sha256,
    compute_overlap_weights_1d,
    compute_overlap_weights_2d,
    compute_valid_fraction_f,
    regrid_method_A,
    regrid_method_L,
    DEFAULT_THETAO_PATH,
    DEFAULT_STATIC_PATH,
    EXPECTED_SHA256_THETAO,
    EXPECTED_SHA256_STATIC,
    TARGET_LAT_START,
    TARGET_LON_START,
    TARGET_N_LAT,
    TARGET_N_LON,
    TARGET_STEP,
    NATIVE_STEP,
)


# =============================================================================
# Test 1: Overlap Weights Calculation and Verification
# =============================================================================

def test_weights_sum_to_one_and_match_canonical():
    """Verify 1D overlap weights equal [1/6, 1/3, 1/3, 1/6] and 2D outer product sums to 1."""
    # Test on full domain coordinates
    lats = TARGET_LAT_START + np.arange(301) * NATIVE_STEP
    lons = TARGET_LON_START + np.arange(721) * NATIVE_STEP

    w_lat_1d, lat_indices, lat_weights = compute_overlap_weights_1d(
        native_centers=lats,
        target_start=TARGET_LAT_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LAT,
        native_step=NATIVE_STEP,
    )
    w_lon_1d, lon_indices, lon_weights = compute_overlap_weights_1d(
        native_centers=lons,
        target_start=TARGET_LON_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LON,
        native_step=NATIVE_STEP,
    )

    expected_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    assert np.allclose(w_lat_1d, expected_1d, atol=1e-12)
    assert np.allclose(w_lon_1d, expected_1d, atol=1e-12)

    # Check each individual target cell
    for i in range(TARGET_N_LAT):
        assert np.isclose(np.sum(lat_weights[i]), 1.0, atol=1e-12)
        assert np.allclose(lat_weights[i], expected_1d, atol=1e-12)
        assert lat_indices[i] == [3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3]

    for j in range(TARGET_N_LON):
        assert np.isclose(np.sum(lon_weights[j]), 1.0, atol=1e-12)
        assert np.allclose(lon_weights[j], expected_1d, atol=1e-12)
        assert lon_indices[j] == [3 * j, 3 * j + 1, 3 * j + 2, 3 * j + 3]

    # 2D outer product weights
    w_2d = compute_overlap_weights_2d(w_lat_1d, w_lon_1d)
    assert w_2d.shape == (4, 4)
    assert np.isclose(np.sum(w_2d), 1.0, atol=1e-12)


# =============================================================================
# Test 2: Constant Field Regridding
# =============================================================================

def test_constant_field_regridding():
    """Verify that a constant field yields exactly the constant for both Method L and A."""
    # Synthetic grid: 2 target cells in lat (7 native), 3 target cells in lon (10 native)
    c_val = 24.375
    field = np.full((7, 10), c_val, dtype=np.float64)
    valid_mask = np.ones((7, 10), dtype=bool)

    val_L = regrid_method_L(field, valid_mask)
    val_A, f = regrid_method_A(field, valid_mask)

    assert val_L.shape == (2, 3)
    assert val_A.shape == (2, 3)
    assert f.shape == (2, 3)

    assert np.allclose(val_L, c_val, atol=1e-12)
    assert np.allclose(val_A, c_val, atol=1e-12)
    assert np.allclose(f, 1.0, atol=1e-12)
    assert np.allclose(val_A - val_L, 0.0, atol=1e-12)


# =============================================================================
# Test 3: Linear Field Regridding
# =============================================================================

def test_linear_field_gives_target_center_both_methods():
    """Verify that a field linear in lat and lon gives the target-center value for BOTH methods (diff = 0)."""
    # Create synthetic grid with 3 target cells in lat (10 native) and 4 in lon (13 native)
    n_target_lat = 3
    n_target_lon = 4
    ny = 3 * n_target_lat + 1  # 10
    nx = 3 * n_target_lon + 1  # 13

    native_lats = TARGET_LAT_START + np.arange(ny) * NATIVE_STEP
    native_lons = TARGET_LON_START + np.arange(nx) * NATIVE_STEP

    # 2D coordinate mesh
    mesh_lons, mesh_lats = np.meshgrid(native_lons, native_lats)

    # Linear field: T = c0 + a * lat + b * lon
    c0 = 12.0
    a = 2.4
    b = -1.5
    field = c0 + a * mesh_lats + b * mesh_lons
    valid_mask = np.ones((ny, nx), dtype=bool)

    val_L = regrid_method_L(field, valid_mask)
    val_A, f = regrid_method_A(field, valid_mask)

    # Target centers
    target_center_lats = TARGET_LAT_START + 0.125 + np.arange(n_target_lat) * TARGET_STEP
    target_center_lons = TARGET_LON_START + 0.125 + np.arange(n_target_lon) * TARGET_STEP
    t_lons, t_lats = np.meshgrid(target_center_lons, target_center_lats)
    expected_center_vals = c0 + a * t_lats + b * t_lons

    # Check Method L equals target center value
    assert np.allclose(val_L, expected_center_vals, atol=1e-12)

    # Check Method A equals target center value
    assert np.allclose(val_A, expected_center_vals, atol=1e-12)

    # Check difference between L and A is strictly zero
    assert np.allclose(val_A - val_L, 0.0, atol=1e-12)


# =============================================================================
# Test 4: Quadratic Field & Brute-Force Fine Grid Equivalence
# =============================================================================

def test_quadratic_field_differs_and_matches_brute_force_fine_grid():
    """Verify quadratic field gives different results for L and A, and A matches fine grid average."""
    # Single target cell: 1 target in lat (4 native), 1 target in lon (4 native)
    ny, nx = 4, 4
    native_lats = TARGET_LAT_START + np.arange(ny) * NATIVE_STEP
    native_lons = TARGET_LON_START + np.arange(nx) * NATIVE_STEP
    mesh_lons, mesh_lats = np.meshgrid(native_lons, native_lats)

    target_center_lat = TARGET_LAT_START + 0.125
    target_center_lon = TARGET_LON_START + 0.125

    # Pure quadratic field centered on target center: T = (lat - lat_c)^2 + 2*(lon - lon_c)^2
    field = (mesh_lats - target_center_lat) ** 2 + 2.0 * (mesh_lons - target_center_lon) ** 2
    valid_mask = np.ones((ny, nx), dtype=bool)

    val_L = regrid_method_L(field, valid_mask)
    val_A, f = regrid_method_A(field, valid_mask)

    # L and A must differ
    diff = float(np.abs(val_A[0, 0] - val_L[0, 0]))
    assert diff > 1e-4, f"Expected non-zero difference for quadratic field, got {diff}"

    # Build brute-force fine grid inside the target cell [lat_start, lat_end] x [lon_start, lon_end]
    n_fine = 120  # 120 x 120 points
    d_lat = TARGET_STEP / n_fine
    d_lon = TARGET_STEP / n_fine
    fine_lats = TARGET_LAT_START + (np.arange(n_fine) + 0.5) * d_lat
    fine_lons = TARGET_LON_START + (np.arange(n_fine) + 0.5) * d_lon

    fine_mesh_lons, fine_mesh_lats = np.meshgrid(fine_lons, fine_lats)

    # Map each fine point to the native cell index that contains it
    half_native = NATIVE_STEP / 2.0
    lat_indices = np.floor((fine_mesh_lats - (native_lats[0] - half_native)) / NATIVE_STEP).astype(int)
    lon_indices = np.floor((fine_mesh_lons - (native_lons[0] - half_native)) / NATIVE_STEP).astype(int)

    # Assign native field value to fine grid
    fine_field_vals = field[lat_indices, lon_indices]
    brute_force_avg = float(np.mean(fine_field_vals))

    # Assert Method A equals brute force fine grid average
    assert np.isclose(val_A[0, 0], brute_force_avg, atol=1e-12)


# =============================================================================
# Test 5: NaN and Validity Handling
# =============================================================================

def test_nan_handling_and_valid_fraction():
    """Verify NaN behavior: L requires all 4 cells; A uses valid cells; f=0 gives NaN."""
    # Single target cell (4x4 native)
    field = np.arange(16, dtype=np.float64).reshape((4, 4))
    valid_mask = np.ones((4, 4), dtype=bool)

    w_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    w_2d = np.outer(w_1d, w_1d)

    # Case 1: All valid
    val_L, f_L = regrid_method_L(field, valid_mask), compute_valid_fraction_f(valid_mask, w_2d)
    val_A, f_A = regrid_method_A(field, valid_mask, w_2d)
    assert not np.isnan(val_L[0, 0])
    assert not np.isnan(val_A[0, 0])
    assert np.isclose(f_A[0, 0], 1.0, atol=1e-12)

    # Case 2: Invalidate ONE of the 4 center cells (1, 1)
    mask_partial = valid_mask.copy()
    mask_partial[1, 1] = False

    val_L_p = regrid_method_L(field, mask_partial)
    val_A_p, f_p = regrid_method_A(field, mask_partial, w_2d)

    # L must be NaN because one of its 4 bracket cells is invalid
    assert np.isnan(val_L_p[0, 0])

    # A must be finite and normalized by f
    expected_f = 1.0 - w_2d[1, 1]
    assert np.isclose(f_p[0, 0], expected_f, atol=1e-12)
    assert not np.isnan(val_A_p[0, 0])

    # Expected value for A: sum(w * x over valid) / sum(w over valid)
    expected_sum_wx = np.sum(w_2d * field) - (w_2d[1, 1] * field[1, 1])
    expected_val_A = expected_sum_wx / expected_f
    assert np.isclose(val_A_p[0, 0], expected_val_A, atol=1e-12)

    # Case 3: Invalidate ALL 4 center cells, keeping only perimeter cells valid
    mask_no_center = valid_mask.copy()
    mask_no_center[1:3, 1:3] = False

    val_L_nc = regrid_method_L(field, mask_no_center)
    val_A_nc, f_nc = regrid_method_A(field, mask_no_center, w_2d)

    assert np.isnan(val_L_nc[0, 0])
    assert not np.isnan(val_A_nc[0, 0])
    expected_f_nc = 1.0 - np.sum(w_2d[1:3, 1:3])  # 1.0 - 4/9 = 5/9
    assert np.isclose(f_nc[0, 0], expected_f_nc, atol=1e-12)

    # Case 4: Invalidate ALL 16 cells in footprint -> f = 0, both methods give NaN
    mask_none = np.zeros((4, 4), dtype=bool)
    val_L_none = regrid_method_L(field, mask_none)
    val_A_none, f_none = regrid_method_A(field, mask_none, w_2d)

    assert np.isnan(val_L_none[0, 0])
    assert np.isnan(val_A_none[0, 0])
    assert np.isclose(f_none[0, 0], 0.0, atol=1e-12)


# =============================================================================
# Test 6: Edge Indices at Domain Boundary
# =============================================================================

def test_edge_indices_at_domain_boundary():
    """Verify that domain boundary target cells use exact boundary native indices and no out-of-domain cells."""
    lats = TARGET_LAT_START + np.arange(301) * NATIVE_STEP
    lons = TARGET_LON_START + np.arange(721) * NATIVE_STEP

    _, lat_indices, _ = compute_overlap_weights_1d(
        native_centers=lats,
        target_start=TARGET_LAT_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LAT,
        native_step=NATIVE_STEP,
    )
    _, lon_indices, _ = compute_overlap_weights_1d(
        native_centers=lons,
        target_start=TARGET_LON_START,
        target_step=TARGET_STEP,
        n_target=TARGET_N_LON,
        native_step=NATIVE_STEP,
    )

    # Boundary target lat i = 99
    assert lat_indices[99] == [297, 298, 299, 300]
    assert max(lat_indices[99]) == 300
    assert min(lat_indices[0]) == 0

    # Boundary target lon j = 239
    assert lon_indices[239] == [717, 718, 719, 720]
    assert max(lon_indices[239]) == 720
    assert min(lon_indices[0]) == 0


# =============================================================================
# Test 7: Raw Data SHA256 Preservation
# =============================================================================

def test_raw_data_sha256_unmodified():
    """Verify raw data files exist and have expected SHA256 hashes (read-only contract)."""
    assert os.path.exists(DEFAULT_THETAO_PATH), f"Raw thetao file missing: {DEFAULT_THETAO_PATH}"
    assert os.path.exists(DEFAULT_STATIC_PATH), f"Raw static mask missing: {DEFAULT_STATIC_PATH}"

    sha_thetao = compute_file_sha256(DEFAULT_THETAO_PATH)
    sha_static = compute_file_sha256(DEFAULT_STATIC_PATH)

    assert sha_thetao == EXPECTED_SHA256_THETAO, f"thetao SHA256 mismatch: {sha_thetao} != {EXPECTED_SHA256_THETAO}"
    assert sha_static == EXPECTED_SHA256_STATIC, f"static SHA256 mismatch: {sha_static} != {EXPECTED_SHA256_STATIC}"
