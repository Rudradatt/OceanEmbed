"""
Unit tests for GLORYS Hot-End Audit module.
Uses synthetic fixtures to test all modular audit logic before any real-data execution.
"""

import os
import tempfile
import numpy as np
import pytest

from scripts.diagnostics.hot_end_audit import (
    GLORYS_FILL_RAW,
    GLORYS_SCALE,
    GLORYS_OFFSET,
    GLORYS_VALID_MAX_RAW,
    GLORYS_VALID_MAX_DECODED,
    GLORYS_ZERO_CODE_RAW,
    TARGET_DEPTH_LEVELS_USED,
    compute_file_sha256,
    compute_histogram_percentiles,
    decode_glorys_raw,
    decode_sst_raw,
    find_nearest_index,
    get_file_stats,
    get_target_footprint_native_indices,
    verify_file_unchanged,
)


def test_sha256_and_file_integrity():
    """Verify SHA-256 calculation and file integrity verification."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"GLORYS Hot-End Audit Test Fixture Content")
        tmp_path = tmp.name

    try:
        pre_stats = get_file_stats(tmp_path, chunk_size=1024)
        assert pre_stats["size_bytes"] == 41
        assert len(pre_stats["sha256"]) == 64
        assert verify_file_unchanged(pre_stats, pre_stats) is True

        # Modify file and verify detection
        with open(tmp_path, "wb") as f:
            f.write(b"Modified File Content")
        post_stats = get_file_stats(tmp_path, chunk_size=1024)
        assert verify_file_unchanged(pre_stats, post_stats) is False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_planted_hot_cells_detection():
    """Test detection of planted hot cells exceeding declared valid_max."""
    # Create synthetic array (time=1, depth=3, lat=4, lon=4)
    raw = np.full((1, 3, 4, 4), 10000, dtype=np.int16)
    raw[0, 0, 0, 0] = GLORYS_FILL_RAW

    # Plant 2 hot cells: 21307 and 24000
    raw[0, 1, 2, 3] = 21307
    raw[0, 2, 1, 1] = 24000

    hot_mask = (raw > GLORYS_VALID_MAX_RAW) & (raw != GLORYS_FILL_RAW)
    assert np.sum(hot_mask) == 2

    # Check decoded values
    dec1 = decode_glorys_raw(21307)
    dec2 = decode_glorys_raw(24000)
    assert dec1 > GLORYS_VALID_MAX_DECODED
    assert np.isclose(dec1, 21307 * GLORYS_SCALE + GLORYS_OFFSET)
    assert np.isclose(dec2, 24000 * GLORYS_SCALE + GLORYS_OFFSET)


def test_planted_zero_codes():
    """Test detection and exclusion classification of planted zero codes."""
    # Zero code value: -28671
    assert GLORYS_ZERO_CODE_RAW == -28671

    # Standard exclusion columns:
    # Column A: lat 15.0, lon 50.5833 (levels 28-30)
    # Column B: lat 15.0, lon 50.6667 (levels 28-30)
    def check_exclusion(lat: float, lon: float, lvl: int) -> tuple[str, bool]:
        is_a = np.isclose(lat, 15.0, atol=1e-3) and np.isclose(lon, 50.583333, atol=1e-3)
        is_b = np.isclose(lat, 15.0, atol=1e-3) and np.isclose(lon, 50.666667, atol=1e-3)
        col = "Column A" if is_a else ("Column B" if is_b else "Other")
        in_excl = (is_a or is_b) and (28 <= lvl <= 30)
        return col, in_excl

    col_a, excl_a = check_exclusion(15.0, 50.5833, 29)
    assert col_a == "Column A"
    assert bool(excl_a) is True

    col_b, excl_b = check_exclusion(15.0, 50.6667, 30)
    assert col_b == "Column B"
    assert bool(excl_b) is True

    # Anomalous cell (Dec 2016 Day 9): Column B at level 28
    col_b28, excl_b28 = check_exclusion(15.0, 50.6667, 28)
    assert col_b28 == "Column B"
    assert bool(excl_b28) is True  # level 28 is within 28..30!

    # Outside exclusion
    col_other, excl_other = check_exclusion(20.0, 60.0, 29)
    assert col_other == "Other"
    assert bool(excl_other) is False


def test_nearest_neighbour_lookup():
    """Test nearest-neighbour index lookup on 1D coordinate axes."""
    lats = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
    assert find_nearest_index(lats, 5.0) == 0
    assert find_nearest_index(lats, 5.2) == 0
    assert find_nearest_index(lats, 5.3) == 1
    assert find_nearest_index(lats, 7.2) == 4
    assert find_nearest_index(lats, 4.0) == 0


def test_footprint_indexing_3i_3i_plus_3():
    """Test Method-A footprint indexing (3i..3i+3) and overlap mapping."""
    # Target cell (0, 0)
    lat_idx, lon_idx = get_target_footprint_native_indices(0, 0)
    assert lat_idx == [0, 1, 2, 3]
    assert lon_idx == [0, 1, 2, 3]

    # Target cell (1, 2)
    lat_idx1, lon_idx1 = get_target_footprint_native_indices(1, 2)
    assert lat_idx1 == [3, 4, 5, 6]
    assert lon_idx1 == [6, 7, 8, 9]

    # Check which target cells cover a native cell at (ny=3, nx=6)
    # ny=3 is covered by target cell i=0 ([0,1,2,3]) and target cell i=1 ([3,4,5,6])
    # nx=6 is covered by target cell j=1 ([3,4,5,6]) and target cell j=2 ([6,7,8,9])
    target_cells = []
    for ti in range(5):
        if 3 * ti <= 3 <= 3 * ti + 3:
            for tj in range(5):
                if 3 * tj <= 6 <= 3 * tj + 3:
                    target_cells.append((ti, tj))

    assert (0, 1) in target_cells
    assert (0, 2) in target_cells
    assert (1, 1) in target_cells
    assert (1, 2) in target_cells
    assert len(target_cells) == 4


def test_target_depth_levels_used():
    """Verify target depth to native levels used mapping."""
    # Depth 0.0m uses level 0 directly
    assert TARGET_DEPTH_LEVELS_USED[0.0] == [0]
    # Depth 5.0m uses bracketing levels 3 and 4
    assert TARGET_DEPTH_LEVELS_USED[5.0] == [3, 4]
    # Depth 10.0m uses bracketing levels 7 and 8
    assert TARGET_DEPTH_LEVELS_USED[10.0] == [7, 8]
    # All 15 target depths mapped
    assert len(TARGET_DEPTH_LEVELS_USED) == 15


def test_histogram_percentiles():
    """Test percentile extraction from 0.01 C resolution histogram."""
    # Construct synthetic data: uniform values between 10.00 and 20.00 C
    data = np.linspace(10.0, 20.0, 100000)
    bin_min = 0.0
    bin_max = 40.0
    bin_width = 0.01
    n_bins = int(round((bin_max - bin_min) / bin_width))
    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)

    bin_indices = np.clip(np.floor((data - bin_min) / bin_width).astype(int), 0, n_bins - 1)
    hist = np.bincount(bin_indices, minlength=n_bins)

    pcts = compute_histogram_percentiles(hist, bin_edges, [0.01, 50.0, 99.99])
    # Expected: 0.01% is approx 10.001 C, 50% is 15.00 C, 99.99% is approx 19.999 C
    assert np.isclose(pcts[0.01], 10.001, atol=0.02)
    assert np.isclose(pcts[50.0], 15.00, atol=0.02)
    assert np.isclose(pcts[99.99], 19.999, atol=0.02)
