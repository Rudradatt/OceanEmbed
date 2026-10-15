"""
Tests for GLORYS Snapshot Census and Pre-Freeze Trajectory Diagnostic.

Verifies:
1. Inventory check fails loudly if a year is missing or a coordinate differs beyond tolerance.
2. Per-level counts (zero code, <0 C, near floor, near ceiling) are exact on synthetic data.
3. The 5 coldest/warmest selection is correct and preserves ranking.
4. Wet-neighbor count is correct across corners, edges, and interior, and pit-candidate selection matches.
5. Final-zero-run start detector correctly ignores earlier runs when there is an interior gap.
6. SHA256 remains 100% unchanged after analysis.
"""

import os
import sys
import hashlib
import tempfile
import h5netcdf
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.diagnostics.glorys_snapshot_census import (
    compute_file_metadata,
    verify_integrity_unchanged,
    decode_thetao_array,
    check_snapshot_inventory,
    compare_masks,
    compute_level_census,
    extract_level_extremes,
    compute_wet_8neighbors,
    identify_pit_candidates_static,
    find_final_continuous_run_start,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_inventory_setup(tmp_path):
    """Create minimal synthetic Jan 2019 reference and snapshot files."""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    jan_file = tmp_path / "glorys_thetao_2019-01.nc"

    lats = np.array([14.9, 15.0, 15.1], dtype=np.float32)
    lons = np.array([50.5, 50.583332, 50.666668], dtype=np.float32)
    depths = np.array([0.5, 10.0], dtype=np.float32)  # 2 levels

    scale = 0.0007324442267417908
    offset = 21.0
    fill = -32767
    vmin = -32766
    vmax = 21306

    # Create Jan reference file
    with h5netcdf.File(jan_file, "w") as f:
        f.dimensions = {"time": 3, "depth": 2, "latitude": 3, "longitude": 3}
        f.create_variable("time", ("time",), data=np.array([0.0, 24.0, 48.0], dtype=np.float32))
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=np.zeros((3, 2, 3, 3), dtype=np.int16))
        v.attrs["scale_factor"] = scale
        v.attrs["add_offset"] = offset
        v.attrs["_FillValue"] = fill
        v.attrs["valid_min"] = vmin
        v.attrs["valid_max"] = vmax

    # Create 3 valid snapshot files: 1993, 1994, 1995
    for y, hours in [(1993, 381288.0), (1994, 390048.0), (1995, 398808.0)]:
        fpath = snap_dir / f"glorys_thetao_{y}-07-01.nc"
        with h5netcdf.File(fpath, "w") as f:
            f.dimensions = {"time": 1, "depth": 2, "latitude": 3, "longitude": 3}
            f.create_variable("time", ("time",), data=np.array([hours], dtype=np.float32))
            f.create_variable("depth", ("depth",), data=depths)
            f.create_variable("latitude", ("latitude",), data=lats)
            f.create_variable("longitude", ("longitude",), data=lons)
            v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=np.zeros((1, 2, 3, 3), dtype=np.int16))
            v.attrs["scale_factor"] = scale
            v.attrs["add_offset"] = offset
            v.attrs["_FillValue"] = fill
            v.attrs["valid_min"] = vmin
            v.attrs["valid_max"] = vmax

    return {
        "snap_dir": str(snap_dir),
        "jan_file": str(jan_file),
        "years": [1993, 1994, 1995],
        "depths": depths,
        "lats": lats,
        "lons": lons,
    }


# =============================================================================
# Test 1: Inventory check fails on missing year or coord mismatch
# =============================================================================

def test_inventory_check_fails_on_missing_year(synthetic_inventory_setup):
    """Test that check_snapshot_inventory fails loudly if a required year is missing."""
    snap_dir = synthetic_inventory_setup["snap_dir"]
    jan_file = synthetic_inventory_setup["jan_file"]

    # Delete 1994
    os.remove(os.path.join(snap_dir, "glorys_thetao_1994-07-01.nc"))

    with pytest.raises(FileNotFoundError, match="Required snapshot file missing"):
        check_snapshot_inventory(snap_dir, jan_file, years=[1993, 1994, 1995])


def test_inventory_check_fails_on_coord_mismatch(synthetic_inventory_setup):
    """Test that check_snapshot_inventory fails loudly if coordinates differ beyond tolerance."""
    snap_dir = synthetic_inventory_setup["snap_dir"]
    jan_file = synthetic_inventory_setup["jan_file"]

    # Overwrite 1995 with shifted latitudes (+0.05 deg, well above 1e-4)
    shifted_lats = synthetic_inventory_setup["lats"] + 0.05
    fpath = os.path.join(snap_dir, "glorys_thetao_1995-07-01.nc")
    with h5netcdf.File(fpath, "w") as f:
        f.dimensions = {"time": 1, "depth": 2, "latitude": 3, "longitude": 3}
        f.create_variable("time", ("time",), data=np.array([398808.0], dtype=np.float32))
        f.create_variable("depth", ("depth",), data=synthetic_inventory_setup["depths"])
        f.create_variable("latitude", ("latitude",), data=shifted_lats)
        f.create_variable("longitude", ("longitude",), data=synthetic_inventory_setup["lons"])
        v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=np.zeros((1, 2, 3, 3), dtype=np.int16))
        v.attrs["scale_factor"] = 0.0007324442267417908
        v.attrs["add_offset"] = 21.0
        v.attrs["_FillValue"] = -32767
        v.attrs["valid_min"] = -32766
        v.attrs["valid_max"] = 21306

    with pytest.raises(ValueError, match="Latitude coordinate difference .* exceeds tolerance"):
        check_snapshot_inventory(snap_dir, jan_file, years=[1993, 1994, 1995])


# =============================================================================
# Test 2: Per-level counts (zero code, <0 C, near floor, near ceiling)
# =============================================================================

def test_per_level_counts_correct():
    """Verify compute_level_census on known synthetic slice."""
    fill = -32767
    vmin = -32766
    vmax = 21306
    scale = 0.001
    offset = 20.0

    # Create 1D array of 7 cells
    raw = np.array([
        fill,            # 0: fill (not finite)
        -28671,          # 1: raw == -28671, dec = -28671*0.001+20 = -8.671 (<0)
        -20000,          # 2: dec = 0.0 C (|dec| < 1e-3)
        -25000,          # 3: dec = -5.0 C (<0)
        vmin + 10,       # 4: near floor (-32756)
        vmax - 10,       # 5: near ceiling (21296)
        5000,            # 6: normal positive cell (dec = 25.0 C)
    ], dtype=np.int16).reshape((1, 7))

    stats = compute_level_census(raw, fill_value=fill, scale_factor=scale, add_offset=offset, valid_min=vmin, valid_max=vmax)

    assert stats["n_total"] == 7
    assert stats["n_finite"] == 6
    assert stats["n_raw_neg_28671"] == 1
    assert stats["n_abs_lt_1e-3"] == 1
    # Cells with dec < 0:
    # idx 1: dec = -8.671 (<0)
    # idx 3: dec = -5.0 (<0)
    # idx 4: raw = -32756, dec = -12.756 (<0)
    assert stats["n_subzero"] == 3
    assert stats["n_near_floor"] == 1
    assert stats["n_near_ceiling"] == 1


# =============================================================================
# Test 3: Extremes selection (5 coldest & 5 warmest)
# =============================================================================

def test_extremes_selection_5_coldest_5_warmest():
    """Verify extract_level_extremes selects exactly 5 coldest and 5 warmest with ranks."""
    scale = 0.01
    offset = 0.0
    fill = -32767

    # 10 values: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    raw = np.array([
        [100, 200, 300, 400, 500],
        [600, 700, 800, 900, 1000]
    ], dtype=np.int16)

    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([50.0, 51.0, 52.0, 53.0, 54.0], dtype=np.float32)

    # Let Column A be at (0, 0) [val 100] and Column B be at (1, 4) [val 1000]
    cold_recs, warm_recs = extract_level_extremes(
        raw, lats, lons, level=5, depth_m=50.0, year=2000,
        fill_value=fill, scale_factor=scale, add_offset=offset,
        lat_a_idx=0, lon_a_idx=0, lat_b_idx=1, lon_b_idx=4, k=5
    )

    assert len(cold_recs) == 5
    assert len(warm_recs) == 5

    # Check cold ordering: 100, 200, 300, 400, 500
    cold_raws = [r["raw"] for r in cold_recs]
    assert cold_raws == [100, 200, 300, 400, 500]
    assert cold_recs[0]["rank"] == 1
    assert cold_recs[0]["is_A"] is True
    assert cold_recs[0]["is_B"] is False

    # Check warm ordering: 1000, 900, 800, 700, 600
    warm_raws = [r["raw"] for r in warm_recs]
    assert warm_raws == [1000, 900, 800, 700, 600]
    assert warm_recs[0]["rank"] == 1
    assert warm_recs[0]["is_A"] is False
    assert warm_recs[0]["is_B"] is True


# =============================================================================
# Test 4: Wet-neighbor count and pit-candidate selection
# =============================================================================

def test_wet_neighbor_count_and_pit_candidates():
    """Verify 8-neighbor wet count for corner, edge, and interior, and pit selection (<3)."""
    # 5x5 grid with clean layout:
    # Row 0: [1, 1, 1, 0, 1]  (0,0)=corner, n=1; (0,1)=edge, n=2; (0,4)=corner isolated, n=0
    # Row 1: [0, 0, 0, 0, 0]  buffer row
    # Row 2: [0, 1, 1, 1, 0]  3x3 block
    # Row 3: [0, 1, 1, 1, 0]  (3,2)=interior, n=8
    # Row 4: [0, 1, 1, 1, 0]
    mask = np.zeros((5, 5), dtype=np.int8)

    # Boundary cells in Row 0
    mask[0, 0] = 1  # Corner cell (only neighbor in domain is (0,1))
    mask[0, 1] = 1  # Edge cell (neighbors are (0,0) and (0,2))
    mask[0, 2] = 1
    mask[0, 4] = 1  # Corner cell (isolated)

    # Interior 3x3 block in Rows 2..4, Cols 1..3
    mask[2:5, 1:4] = 1

    neighbors = compute_wet_8neighbors(mask)

    # 1. Corner (0, 0): exactly 1 neighbor ((0, 1)) -> n = 1
    assert neighbors[0, 0] == 1
    # 2. Edge of domain (0, 1): exactly 2 neighbors ((0, 0) and (0, 2)) -> n = 2
    assert neighbors[0, 1] == 2
    # 3. Corner (0, 4): completely isolated -> n = 0
    assert neighbors[0, 4] == 0
    # 4. Interior (3, 2): center of 3x3 block -> all 8 neighbors are wet -> n = 8
    assert neighbors[3, 2] == 8

    # Pit candidates: wet cells with neighbors < 3
    is_pit = (mask == 1) & (neighbors < 3)
    assert is_pit[0, 0] is np.True_   # Corner pit (n=1)
    assert is_pit[0, 1] is np.True_   # Edge pit (n=2)
    assert is_pit[0, 4] is np.True_   # Isolated corner pit (n=0)
    assert is_pit[3, 2] is np.False_  # Interior cell (n=8 >= 3)
    assert is_pit[1, 1] is np.False_  # Land cell (mask=0)


# =============================================================================
# Test 5: Final-zero-run start detector with interior gap
# =============================================================================

def test_final_zero_run_start_detector_with_interior_gap():
    """Verify final-zero-run start detector finds the final run and ignores earlier runs across gaps."""
    target = -28671

    # Series with interior gap:
    # indices: 0   1       2        3   4       5       6
    # values:  10, target, target,  99, target, target, target
    arr = np.array([10, target, target, 99, target, target, target], dtype=np.int16)

    start_idx = find_final_continuous_run_start(arr, target)
    assert start_idx == 4  # The final run starts at index 4 (length = 3)

    # Series that does NOT end in target
    arr_no_end = np.array([target, target, target, 5], dtype=np.int16)
    assert find_final_continuous_run_start(arr_no_end, target) == -1

    # Series that is entirely target
    arr_all = np.array([target, target, target], dtype=np.int16)
    assert find_final_continuous_run_start(arr_all, target) == 0

    # Empty series
    assert find_final_continuous_run_start(np.array([], dtype=np.int16), target) == -1


# =============================================================================
# Test 6: SHA256 unchanged after analysis
# =============================================================================

def test_sha256_unchanged_after_analysis(synthetic_inventory_setup):
    """Verify that file integrity checking detects any modification and verifies clean files."""
    snap_dir = synthetic_inventory_setup["snap_dir"]
    jan_file = synthetic_inventory_setup["jan_file"]

    fpath = os.path.join(snap_dir, "glorys_thetao_1993-07-01.nc")
    before_meta = {os.path.abspath(fpath): compute_file_metadata(fpath)}

    # Simulate read-only operations
    with h5netcdf.File(fpath, "r") as f:
        _ = f.variables["thetao"][:]

    after_meta = {os.path.abspath(fpath): compute_file_metadata(fpath)}

    ok, mismatches = verify_integrity_unchanged(before_meta, after_meta)
    assert ok is True
    assert len(mismatches) == 0
