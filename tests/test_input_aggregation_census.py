"""
Unit Tests for Input Aggregation and Common-Mask Census.

Contract:
- Block mean of a constant equals the constant.
- f is correct with planted invalid cells.
- A block with no valid cell gives NaN and f = 0.
- The 1 x 1 case reproduces the input exactly.
- Nesting check fails loudly on a mis-aligned grid.
- The SST rule excludes lake and land bits.
- SHA256 of all raw files unchanged.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.diagnostics.input_aggregation_census import (
    EXPECTED_SHA256,
    RAW_FILES,
    block_aggregate,
    compute_file_sha256,
    compute_sst_validity,
    verify_grid_nesting,
)


def test_block_mean_of_constant_equals_constant():
    """Verify block mean of a constant equals the constant across various block ratios."""
    # 5x5 ratio (like SST)
    c_val = 298.15
    native_field = np.full((100 * 5, 240 * 5), c_val, dtype=np.float64)
    valid_mask = np.ones((500, 1200), dtype=bool)

    mean_f, f = block_aggregate(native_field, valid_mask, ratio_y=5, ratio_x=5, n_target_y=100, n_target_x=240)
    assert mean_f.shape == (100, 240)
    assert f.shape == (100, 240)
    assert np.allclose(mean_f, c_val, atol=1e-12)
    assert np.allclose(f, 1.0, atol=1e-12)

    # 2x2 ratio (like SLA/SSS)
    c_sla = 0.42
    native_sla = np.full((100 * 2, 240 * 2), c_sla, dtype=np.float64)
    valid_sla = np.ones((200, 480), dtype=bool)

    mean_sla, f_sla = block_aggregate(native_sla, valid_sla, ratio_y=2, ratio_x=2, n_target_y=100, n_target_x=240)
    assert np.allclose(mean_sla, c_sla, atol=1e-12)
    assert np.allclose(f_sla, 1.0, atol=1e-12)


def test_f_correct_with_planted_invalid_cells():
    """Verify f is correct and mean only averages valid cells when invalid cells are planted."""
    ny, nx = 20, 20
    ratio = 2
    n_ty, n_tx = ny // ratio, nx // ratio

    native_field = np.ones((ny, nx), dtype=np.float64) * 10.0
    valid_mask = np.ones((ny, nx), dtype=bool)

    # In target cell (0, 0): plant 1 invalid cell out of 4 (native index (0, 0))
    valid_mask[0, 0] = False
    native_field[0, 0] = 999.0  # Planted outlier should be ignored

    # In target cell (0, 1): plant 2 invalid cells out of 4
    valid_mask[0, 2] = False
    valid_mask[0, 3] = False
    native_field[0, 2] = 888.0
    native_field[0, 3] = 777.0

    # In target cell (1, 0): plant 3 invalid cells out of 4
    valid_mask[2, 0] = False
    valid_mask[2, 1] = False
    valid_mask[3, 0] = False

    mean_f, f = block_aggregate(native_field, valid_mask, ratio_y=ratio, ratio_x=ratio, n_target_y=n_ty, n_target_x=n_tx)

    # Check cell (0, 0): 3 valid out of 4 -> f = 0.75, mean = 10.0
    assert abs(f[0, 0] - 0.75) < 1e-12
    assert abs(mean_f[0, 0] - 10.0) < 1e-12

    # Check cell (0, 1): 2 valid out of 4 -> f = 0.50, mean = 10.0
    assert abs(f[0, 1] - 0.50) < 1e-12
    assert abs(mean_f[0, 1] - 10.0) < 1e-12

    # Check cell (1, 0): 1 valid out of 4 -> f = 0.25, mean = 10.0
    assert abs(f[1, 0] - 0.25) < 1e-12
    assert abs(mean_f[1, 0] - 10.0) < 1e-12

    # Check cell (1, 1): all 4 valid -> f = 1.0, mean = 10.0
    assert abs(f[1, 1] - 1.0) < 1e-12
    assert abs(mean_f[1, 1] - 10.0) < 1e-12


def test_block_with_no_valid_cell_gives_nan_and_zero_f():
    """Verify that a block with no valid cell produces NaN for mean and 0.0 for f."""
    ratio = 5
    n_ty, n_tx = 2, 2
    native_field = np.full((n_ty * ratio, n_tx * ratio), 15.0, dtype=np.float64)
    valid_mask = np.ones_like(native_field, dtype=bool)

    # Empty target block (0, 1) entirely
    valid_mask[0:5, 5:10] = False
    native_field[0:5, 5:10] = -9999.0

    mean_f, f = block_aggregate(native_field, valid_mask, ratio_y=ratio, ratio_x=ratio, n_target_y=n_ty, n_target_x=n_tx)

    assert f[0, 1] == 0.0
    assert np.isnan(mean_f[0, 1])

    # Other blocks remain valid
    assert f[0, 0] == 1.0
    assert mean_f[0, 0] == 15.0


def test_one_by_one_case_reproduces_input_exactly():
    """Verify the 1 x 1 case reproduces the input exactly."""
    n_ty, n_tx = 100, 240
    rng = np.random.default_rng(42)
    native_field = rng.normal(loc=0.0, scale=1.0, size=(n_ty, n_tx))
    valid_mask = rng.choice([True, False], size=(n_ty, n_tx), p=[0.7, 0.3])

    mean_f, f = block_aggregate(native_field, valid_mask, ratio_y=1, ratio_x=1, n_target_y=n_ty, n_target_x=n_tx)

    # Where valid: mean_f equals native_field and f == 1.0
    assert np.allclose(mean_f[valid_mask], native_field[valid_mask], atol=1e-14)
    assert np.all(f[valid_mask] == 1.0)

    # Where invalid: mean_f is NaN and f == 0.0
    assert np.all(np.isnan(mean_f[~valid_mask]))
    assert np.all(f[~valid_mask] == 0.0)


def test_nesting_check_fails_loudly_on_misaligned_grid():
    """Verify nesting check raises ValueError on misaligned, shifted, or non-integer grids."""
    # 1. GLORYS grid: centers at [5.0, 30.0] with step 1/12 (straddles target edges)
    glorys_lats = np.linspace(5.0, 30.0, 301)
    with pytest.raises(ValueError, match="Start edges do not coincide"):
        verify_grid_nesting(glorys_lats, target_start=5.0, target_step=0.25, n_target=100)

    # 2. Non-integer ratio grid (e.g. native step 0.1 deg -> ratio = 2.5)
    bad_step_lats = np.linspace(5.05, 29.95, 250)
    with pytest.raises(ValueError, match="is not an integer"):
        verify_grid_nesting(bad_step_lats, target_start=5.0, target_step=0.25, n_target=100)

    # 3. Shifted start edge
    shifted_lats = np.linspace(5.05, 30.0, 500)
    with pytest.raises(ValueError, match="Start edges do not coincide"):
        verify_grid_nesting(shifted_lats, target_start=5.0, target_step=0.25, n_target=100)

    # 4. Truncated count (fewer native cells than domain)
    truncated_lats = np.linspace(5.025, 25.0, 400)
    with pytest.raises(ValueError, match="End edges do not coincide"):
        verify_grid_nesting(truncated_lats, target_start=5.0, target_step=0.25, n_target=100)


def test_sst_rule_excludes_lake_and_land_bits():
    """Verify the SST rule (analysed_sst != -32768) & ((mask & 6) == 0) correctly excludes lake and land."""
    # Synthetic values
    sst_raw = np.array([
        2800,    # water (mask 1) -> valid
        2800,    # land (mask 2) -> invalid (bit 1 set)
        2800,    # lake (mask 4) -> invalid (bit 2 set)
        2800,    # land+lake (mask 6) -> invalid (bits 1 & 2 set)
        2800,    # land+lake+ice (mask 14) -> invalid
        2800,    # ice only (mask 8) -> valid
        2800,    # river only (mask 16) -> valid
        -32768,  # fill value with mask 1 -> invalid (non-fill failed)
        -32768,  # fill value with mask 2 -> invalid
    ], dtype=np.int16)

    mask_raw = np.array([1, 2, 4, 6, 14, 8, 16, 1, 2], dtype=np.int8)

    valid_mask, umasks, lake_removed = compute_sst_validity(sst_raw, mask_raw)

    expected_valid = np.array([
        True,   # 1: water
        False,  # 2: land
        False,  # 4: lake
        False,  # 6: land+lake
        False,  # 14: land+lake+ice
        True,   # 8: ice
        True,   # 16: river
        False,  # fill
        False,  # fill
    ], dtype=bool)

    assert np.array_equal(valid_mask, expected_valid)
    # Non-fill cells are indices 0..6 (count = 7).
    # Removed among non-fill: indices 1, 2, 3, 4 (count = 4).
    assert lake_removed == 4
    assert np.array_equal(umasks, np.array([1, 2, 4, 6, 8, 14, 16], dtype=np.int8))


def test_sha256_raw_files_unchanged():
    """Verify all six raw input files match their expected SHA256 checksums."""
    for key, path in RAW_FILES.items():
        assert os.path.exists(path), f"Raw file {key} not found at {path}"
        actual_sha = compute_file_sha256(path)
        expected_sha = EXPECTED_SHA256[key]
        assert actual_sha == expected_sha, (
            f"Raw file {key} checksum mismatch! Expected: {expected_sha}, Got: {actual_sha}"
        )
