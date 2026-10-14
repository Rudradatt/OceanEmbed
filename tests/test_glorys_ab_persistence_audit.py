"""
Tests for GLORYS A/B Window Persistence Audit (1993-2020).

Verifies:
1. Longest-run function handles start, middle, end, and uniform sequences correctly.
2. Zero-code (-28671) day counts per cell are exact.
3. Wet-mask change detector flags planted mask changes.
4. Cross-download comparison detects and flags planted slice mismatches.
5. Coordinate lookup fails loudly when target coordinate exceeds tolerance.
6. SHA256 of raw files remains strictly unchanged after analysis.
"""

import hashlib
import os
import datetime
import h5netcdf
import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.glorys_ab_persistence_audit import (
    compute_file_metadata,
    lookup_coord_index,
    compute_longest_run,
    audit_wet_mask_history,
    census_window_wide,
    compare_cross_download_reproducibility,
)


def test_longest_run_function():
    """Test 1: Verify longest-run function across boundaries (start, middle, end, uniform)."""
    # Run at beginning
    run_len, val = compute_longest_run(np.array([5, 5, 5, 1, 2]))
    assert run_len == 3
    assert val == 5

    # Run in middle
    run_len, val = compute_longest_run(np.array([1, 2, 9, 9, 9, 9, 3]))
    assert run_len == 4
    assert val == 9

    # Run at end
    run_len, val = compute_longest_run(np.array([1, 2, 7, 7, 7]))
    assert run_len == 3
    assert val == 7

    # All identical
    run_len, val = compute_longest_run(np.array([4, 4, 4, 4]))
    assert run_len == 4
    assert val == 4

    # All different
    run_len, val = compute_longest_run(np.array([1, 2, 3]))
    assert run_len == 1

    # Single element
    run_len, val = compute_longest_run(np.array([8]))
    assert run_len == 1
    assert val == 8

    # Empty
    run_len, val = compute_longest_run(np.array([]))
    assert run_len == 0


@pytest.fixture
def synthetic_audit_nc(tmp_path):
    """Create a synthetic multi-day NetCDF file with controlled anomalies."""
    nc_path = tmp_path / "synthetic_audit.nc"

    # 4 days: 2019-01-01 to 2019-01-04
    base_dt = datetime.datetime(1950, 1, 1)
    start_dt = datetime.datetime(2019, 1, 1)
    times = np.array([(start_dt + datetime.timedelta(days=i) - base_dt).total_seconds() / 3600.0 for i in range(4)], dtype=np.float32)

    depths = np.array([0.5, 50.0], dtype=np.float32)  # 2 levels
    lats = np.array([14.9, 15.0], dtype=np.float32)    # 2 lats
    lons = np.array([50.583332, 50.666668], dtype=np.float32)  # 2 lons

    fill_val = -32767
    scale_factor = 0.001
    add_offset = 20.0

    # Base thetao: shape (4, 2, 2, 2)
    data = np.full((4, 2, 2, 2), 1000, dtype=np.int16)

    # Plant -28671 in cell (k=1, y=1, x=0) on days 0, 1, 2 (3 days)
    data[0:3, 1, 1, 0] = -28671

    with h5netcdf.File(str(nc_path), "w") as f:
        f.dimensions = {"time": len(times), "depth": len(depths), "latitude": len(lats), "longitude": len(lons)}
        f.create_variable("time", ("time",), data=times)
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        vt = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=data)
        vt.attrs["scale_factor"] = scale_factor
        vt.attrs["add_offset"] = add_offset
        vt.attrs["_FillValue"] = fill_val

    return {
        "nc_path": str(nc_path),
        "fill_val": fill_val,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
    }


def test_zero_code_day_counts(synthetic_audit_nc):
    """Test 2: Verify zero-code (-28671) day counts per cell are exact."""
    res = census_window_wide(synthetic_audit_nc["nc_path"])
    df = res["zero_cells_df"]

    assert len(df) == 1
    row = df.iloc[0]
    assert row["level"] == 1
    assert row["y_idx"] == 1
    assert row["x_idx"] == 0
    assert row["days_raw_28671"] == 3


def test_mask_change_detector_flags_planted_change(tmp_path):
    """Test 3: Verify wet-mask change detector flags planted mask change."""
    nc_path = tmp_path / "mask_change.nc"

    times = np.array([100.0, 124.0, 148.0], dtype=np.float32)  # 3 days
    depths = np.array([0.5, 10.0], dtype=np.float32)
    lats = np.array([15.0], dtype=np.float32)
    lons = np.array([50.5], dtype=np.float32)

    fill_val = -32767
    data = np.full((3, 2, 1, 1), 1000, dtype=np.int16)

    # Plant mask change: on day 2, level 1 becomes fill!
    data[2, 1, 0, 0] = fill_val

    with h5netcdf.File(str(nc_path), "w") as f:
        f.dimensions = {"time": len(times), "depth": len(depths), "latitude": len(lats), "longitude": len(lons)}
        f.create_variable("time", ("time",), data=times)
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        vt = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=data)
        vt.attrs["_FillValue"] = fill_val

    res = audit_wet_mask_history(str(nc_path))
    assert res["is_completely_time_invariant"] is False
    assert res["total_differing_points"] == 1
    assert len(res["differing_points_sample"]) == 1
    diff = res["differing_points_sample"][0]
    assert diff["day_index"] == 2
    assert diff["level"] == 1


def test_cross_download_flags_planted_mismatch(tmp_path):
    """Test 4: Verify cross-download comparison flags a planted mismatch."""
    p1 = tmp_path / "w1.nc"
    p2 = tmp_path / "w2.nc"

    base_dt = datetime.datetime(1950, 1, 1)
    start_dt = datetime.datetime(2019, 1, 1)
    times = np.array([(start_dt + datetime.timedelta(days=i) - base_dt).total_seconds() / 3600.0 for i in range(2)], dtype=np.float32)
    depths = np.array([0.5], dtype=np.float32)
    lats = np.array([15.0], dtype=np.float32)
    lons = np.array([50.5], dtype=np.float32)

    data1 = np.full((2, 1, 1, 1), 1000, dtype=np.int16)
    data2 = np.full((2, 1, 1, 1), 1000, dtype=np.int16)
    # Plant mismatch in data2 on day 1
    data2[1, 0, 0, 0] = 9999

    for p, d in [(p1, data1), (p2, data2)]:
        with h5netcdf.File(str(p), "w") as f:
            f.dimensions = {"time": len(times), "depth": len(depths), "latitude": len(lats), "longitude": len(lons)}
            f.create_variable("time", ("time",), data=times)
            f.create_variable("depth", ("depth",), data=depths)
            f.create_variable("latitude", ("latitude",), data=lats)
            f.create_variable("longitude", ("longitude",), data=lons)
            f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=d)

    res = compare_cross_download_reproducibility(str(p1), str(p2), year=2019, month=1)
    assert res["is_exact_match"] is False
    assert res["mismatch_count"] == 1
    assert len(res["first_20_mismatches"]) == 1
    mis = res["first_20_mismatches"][0]
    assert mis["window_raw_code"] == 1000
    assert mis["full_raw_code"] == 9999


def test_coordinate_lookup_fails_loudly_outside_tolerance():
    """Test 5: Verify coordinate lookup raises ValueError when target exceeds tolerance."""
    coords = np.array([14.8, 14.9, 15.0, 15.1], dtype=np.float32)

    # Valid lookup
    idx = lookup_coord_index(coords, 15.0, "latitude", tolerance=1e-3)
    assert idx == 2

    # Beyond tolerance: target = 16.0
    with pytest.raises(ValueError, match="exceeds tolerance"):
        lookup_coord_index(coords, 16.0, "latitude", tolerance=1e-3)


def test_sha256_unchanged_after_analysis(synthetic_audit_nc):
    """Test 6: Verify SHA256 of raw NetCDF file is strictly unchanged after analysis execution."""
    p = synthetic_audit_nc["nc_path"]

    meta_before = compute_file_metadata(p)

    # Run audit functions
    _ = census_window_wide(p)
    _ = audit_wet_mask_history(p)

    meta_after = compute_file_metadata(p)

    assert meta_before["sha256"] == meta_after["sha256"]
    assert meta_before["size_bytes"] == meta_after["size_bytes"]
    assert meta_before["mtime_utc"] == meta_after["mtime_utc"]
