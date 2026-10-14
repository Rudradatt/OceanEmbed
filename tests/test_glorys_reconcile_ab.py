"""
Tests for GLORYS A/B Raw Values Reconciliation.

Verifies:
1. Two independent readers (h5netcdf & xarray) agree on raw values.
2. Comparison function detects and flags deliberately planted artifact mismatches.
3. Recomputation functions return exact known counts for finite, -28671, and frozen cells.
4. SHA256 remains strictly unchanged after analysis execution.
"""

import hashlib
import os
import h5netcdf
import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.glorys_reconcile_ab import (
    compute_file_metadata,
    independent_raw_read,
    compare_against_earlier_artifacts,
    recompute_headline_numbers,
)


@pytest.fixture
def synthetic_thetao_nc(tmp_path):
    """Create a small synthetic NetCDF file with controlled int16 raw values."""
    nc_path = tmp_path / "synthetic_thetao.nc"

    times = np.array([0, 1, 2], dtype=np.float32)  # 3 days
    depths = np.array([0.5, 10.0, 50.0], dtype=np.float32)  # 3 levels
    # Latitudes including 15.0 at index 2
    lats = np.array([14.8, 14.9, 15.0, 15.1], dtype=np.float32)
    # Longitudes including 50.583332 at index 1 and 50.666668 at index 2
    lons = np.array([50.5, 50.583332, 50.666668, 50.8], dtype=np.float32)

    nt = len(times)
    nz = len(depths)
    ny = len(lats)
    nx = len(lons)

    fill_val = -32767
    scale_factor = 0.001
    add_offset = 20.0

    # Base data
    data = np.full((nt, nz, ny, nx), 1000, dtype=np.int16)

    # Make cell (0, 0, 0, 0) fill
    data[:, :, 0, 0] = fill_val

    # Plant target code -28671 in 2 cells across all 3 days -> 2 * 3 = 6 cells total
    # Cell 1: at (k=1, y=2, x=1)
    # Cell 2: at (k=1, y=2, x=2)
    data[:, 1, 2, 1] = -28671
    data[:, 1, 2, 2] = -28671

    # In cell (k=0, y=2, x=1) vary across days:
    for t in range(nt):
        data[t, 0, 2, 1] = 1000 + t

    with h5netcdf.File(str(nc_path), "w") as f:
        f.attrs["title"] = "Synthetic Reconciliation Test NetCDF"
        f.dimensions = {"time": nt, "depth": nz, "latitude": ny, "longitude": nx}
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
        "nt": nt,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "fill_val": fill_val,
        "col_a_idx": (2, 1),
        "col_b_idx": (2, 2),
    }


def test_independent_readers_agree(synthetic_thetao_nc):
    """Test 1: Verify h5netcdf and xarray readers return identical raw data."""
    res = independent_raw_read(
        nc_path=synthetic_thetao_nc["nc_path"],
        col_a_idx=synthetic_thetao_nc["col_a_idx"],
        col_b_idx=synthetic_thetao_nc["col_b_idx"],
    )

    assert res["methods_agree_A"] is True
    assert res["methods_agree_B"] is True
    assert res["n_times"] == synthetic_thetao_nc["nt"]
    assert res["n_depths"] == synthetic_thetao_nc["nz"]
    assert len(res["independent_df"]) == 2 * synthetic_thetao_nc["nt"] * synthetic_thetao_nc["nz"]


def test_comparison_flags_planted_mismatch(tmp_path):
    """Test 2: Verify comparison flags deliberately planted artifact mismatches."""
    # Create synthetic independent_df
    indep_df = pd.DataFrame([
        {"column": "A", "day": 0, "level": 0, "depth_m": 0.5, "raw_code": 1000, "is_fill": False, "decoded_C": 21.0},
        {"column": "A", "day": 1, "level": 0, "depth_m": 0.5, "raw_code": 1001, "is_fill": False, "decoded_C": 21.001},
        {"column": "B", "day": 0, "level": 0, "depth_m": 0.5, "raw_code": 2000, "is_fill": False, "decoded_C": 22.0},
    ])

    # Artifact with 1 correct row and 1 planted mismatch (day 1 has 9999 instead of 1001)
    art_path = tmp_path / "planted_art.csv"
    pd.DataFrame([
        {"day": 0, "column": "A", "level_index": 0, "raw_code": 1000},
        {"day": 1, "column": "A", "level_index": 0, "raw_code": 9999},  # Planted mismatch!
    ]).to_csv(art_path, index=False)

    comp_res = compare_against_earlier_artifacts(
        independent_df=indep_df,
        artifact_paths={"ab_profiles.csv": str(art_path)},
    )

    assert comp_res["total_rows_compared"] == 2
    assert comp_res["total_mismatches"] == 1
    mdf = comp_res["mismatches_df"]
    assert len(mdf) == 1
    row = mdf.iloc[0]
    assert row["column"] == "A"
    assert row["day"] == 1
    assert row["level"] == 0
    assert row["artifact_raw_code"] == 9999
    assert row["ground_truth_raw_code"] == 1001
    assert row["diff"] == 9999 - 1001


def test_recomputation_headline_numbers(synthetic_thetao_nc):
    """Test 3: Verify headline numbers recomputation on known synthetic grid."""
    res = recompute_headline_numbers(synthetic_thetao_nc["nc_path"])

    # Total grid: 3 depth x 4 lat x 4 lon = 48 cells per day.
    # Cell (0, 0) is fill on all 3 depth levels -> 3 fill cells per day.
    # Finite cells per day: 48 - 3 = 45.
    assert res["finite_cells_per_day"]["counts"] == [45, 45, 45]
    assert res["finite_cells_per_day"]["is_time_invariant"] is True

    # Raw == -28671: planted in 2 cells across 3 days -> 6 cells total
    assert res["raw_code_28671_census"]["total_4d_count"] == 6
    assert res["raw_code_28671_census"]["per_day_counts"] == [2, 2, 2]

    # Temporally constant finite cells:
    # Most cells are constant 1000 across 3 days!
    # Specifically, out of 45 finite cells, only cell (k=0, y=2, x=1) varies across days.
    # So 44 cells are temporally constant.
    assert res["temporally_frozen_finite_cells"]["total_frozen_count"] == 44


def test_sha256_unchanged_after_analysis(synthetic_thetao_nc):
    """Test 4: Verify SHA256 of raw file is strictly unchanged after analysis execution."""
    p = synthetic_thetao_nc["nc_path"]

    meta_before = compute_file_metadata(p)

    # Perform reads
    _ = independent_raw_read(p, col_a_idx=(2, 1), col_b_idx=(2, 2))
    _ = recompute_headline_numbers(p)

    meta_after = compute_file_metadata(p)

    assert meta_before["sha256"] == meta_after["sha256"]
    assert meta_before["size_bytes"] == meta_after["size_bytes"]
    assert meta_before["mtime_utc"] == meta_after["mtime_utc"]
