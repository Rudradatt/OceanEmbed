"""
Tests for independent min/max contract reconciliation script.
Validates:
1. Planted min/max values on synthetic NetCDF data (raw min/max, decoded min/max, lat/lon of min/max cells)
2. Planted mismatch detection across comparison sources
3. Strict SHA256 preservation and immutability (READ-ONLY verification)
"""

import os
import sys
import hashlib
import pytest
import numpy as np
import pandas as pd
import h5netcdf

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "scripts", "diagnostics"))

from reconcile_contract_minmax import (
    compute_independent_minmax,
    compare_minmax_sources,
)


def _compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def synthetic_reconcile_dataset(tmp_path):
    """Create a synthetic NetCDF dataset with known planted min/max."""
    nc_path = str(tmp_path / "synthetic_test.nc")
    lats = np.array([10.0, 15.0, 20.0], dtype=np.float32)
    lons = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    times = np.array([0.0, 24.0], dtype=np.float32)

    # Day 0:
    # row 0: [100, -9999, 300]
    # row 1: [250, 500,   150]
    # row 2: [400, 50,    350]
    # Non-fill min = 50 at (lat=20.0, lon=60.0), max = 500 at (lat=15.0, lon=60.0)
    # Day 1:
    # row 0: [120, -9999, 310]
    # row 1: [260, 600,   160]   <- max = 600
    # row 2: [410, 40,    360]   <- min = 40
    # Scale = 0.1, Offset = 5.0
    day0 = np.array([
        [100, -9999, 300],
        [250, 500,   150],
        [400, 50,    350]
    ], dtype=np.int16)

    day1 = np.array([
        [120, -9999, 310],
        [260, 600,   160],
        [410, 40,    360]
    ], dtype=np.int16)

    raw_3d = np.stack([day0, day1], axis=0)

    with h5netcdf.File(nc_path, "w") as f:
        f.dimensions = {"time": 2, "latitude": 3, "longitude": 3}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = times
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"

        v_data = f.create_variable("sst", ("time", "latitude", "longitude"), dtype=np.int16)
        v_data[:] = raw_3d
        v_data.attrs["_FillValue"] = np.int16(-9999)
        v_data.attrs["scale_factor"] = np.float64(0.1)
        v_data.attrs["add_offset"] = np.float64(5.0)

    return nc_path


def test_planted_minmax_independent_recompute(synthetic_reconcile_dataset):
    """Test that compute_independent_minmax accurately recomputes planted values."""
    nc_path = synthetic_reconcile_dataset
    df = compute_independent_minmax(
        sst_file=nc_path,
        sla_file=nc_path,
        sss_file=nc_path,
        cur_file=nc_path
    )

    # Filter to sst variable
    row = df[(df["source"] == "OSTIA_SST") & (df["variable"] == "sst")].iloc[0]

    # Day 0: raw min=50, max=500; decoded min=50*0.1+5=10.0, max=500*0.1+5=55.0
    assert row["raw_min_day0"] == 50
    assert row["raw_max_day0"] == 500
    assert np.isclose(row["decoded_min_day0"], 10.0)
    assert np.isclose(row["decoded_max_day0"], 55.0)

    # Coordinates of min cell: row 2 (lat=20.0), col 1 (lon=60.0)
    assert np.isclose(row["min_lat_day0"], 20.0)
    assert np.isclose(row["min_lon_day0"], 60.0)

    # Coordinates of max cell: row 1 (lat=15.0), col 1 (lon=60.0)
    assert np.isclose(row["max_lat_day0"], 15.0)
    assert np.isclose(row["max_lon_day0"], 60.0)

    # All days combined: raw min=40, max=600; decoded min=40*0.1+5=9.0, max=600*0.1+5=65.0
    assert row["raw_min_all31"] == 40
    assert row["raw_max_all31"] == 600
    assert np.isclose(row["decoded_min_all31"], 9.0)
    assert np.isclose(row["decoded_max_all31"], 65.0)


def test_planted_mismatch_detected(tmp_path):
    """Test that compare_minmax_sources flags mismatches between independent and audit CSVs."""
    ind_csv = str(tmp_path / "independent.csv")
    v1_csv = str(tmp_path / "v1.csv")
    v2_csv = str(tmp_path / "v2.csv")

    df_ind = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "depth_m": np.nan,
        "decoded_min_day0": 273.87,
        "decoded_max_day0": 302.95,
        "decoded_min_all31": 273.14,
        "decoded_max_all31": 304.09,
    }])
    df_ind.to_csv(ind_csv, index=False)

    # v1 matches
    df_v1 = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "depth_m": np.nan,
        "day_index": 0,
        "decoded_min": 273.87,
        "decoded_max": 302.95,
    }])
    df_v1.to_csv(v1_csv, index=False)

    # v2 has planted mismatch
    df_v2 = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "depth_m": np.nan,
        "day_index": 0,
        "decoded_min": 288.75,
        "decoded_max": 305.80,
    }])
    df_v2.to_csv(v2_csv, index=False)

    df_comp = compare_minmax_sources(ind_csv, v1_csv, v2_csv)
    row = df_comp.iloc[0]

    assert row["v1_matches_independent_d0"] == True
    assert row["v2_matches_independent_d0"] == False


def test_file_sha256_preservation(synthetic_reconcile_dataset):
    """Test that running independent recompute does NOT modify files."""
    nc_path = synthetic_reconcile_dataset
    sha_before = _compute_sha256(nc_path)
    size_before = os.path.getsize(nc_path)

    # Run compute
    _ = compute_independent_minmax(
        sst_file=nc_path,
        sla_file=nc_path,
        sss_file=nc_path,
        cur_file=nc_path
    )

    sha_after = _compute_sha256(nc_path)
    size_after = os.path.getsize(nc_path)

    assert sha_before == sha_after
    assert size_before == size_after
