"""
Unit tests for SST and SLA pilot data contract audit.
Tests on small synthetic NetCDF datasets with known packing, fill,
descending latitude axis, 0-360 longitudes, and 12:00 timestamps.
"""

import os
import sys
import pytest
import numpy as np
import h5netcdf

# Ensure scripts directory is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "scripts", "diagnostics"))

from input_contract_audit import (
    compute_file_sha256,
    dump_header_report,
    audit_coordinates_and_time,
    audit_variable_raw_contract,
    audit_coverage_vs_glorys,
    classify_variable,
)


@pytest.fixture
def synthetic_contract_dataset(tmp_path):
    """
    Create a synthetic NetCDF file with:
    - descending latitude axis: [30.0, 25.0, 20.0, 15.0, 10.0, 5.0]
    - 0-360 longitudes: [190.0, 200.0, 210.0, 220.0, 230.0] (max > 180)
    - 12:00 timestamps: hours since 2019-01-01 00:00:00 with values [12.0, 36.0, 60.0]
    - packed int16 data variable with scale_factor=0.01, add_offset=20.0, _FillValue=-999
    - valid_min=-100, valid_max=500, with intentional violations and decoded exact zero
    """
    nc_path = str(tmp_path / "synthetic_contract.nc")

    lats = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 5.0], dtype=np.float32)  # descending
    lons = np.array([190.0, 200.0, 210.0, 220.0, 230.0], dtype=np.float32)  # 0..360 convention
    times = np.array([12.0, 36.0, 60.0], dtype=np.float32)  # 12:00 UTC on 2019-01-01, 01-02, 01-03

    n_time = len(times)
    n_lat = len(lats)
    n_lon = len(lons)

    # Base raw array
    raw_data = np.full((n_time, n_lat, n_lon), 100, dtype=np.int16)

    # 1. Fill cells at (lat=0, lon=0) on all days
    raw_data[:, 0, 0] = -999

    # 2. Frozen non-fill cell at (lat=1, lon=1) with raw value 150 across all days
    raw_data[:, 1, 1] = 150

    # 3. Time-varying cell at (lat=2, lon=2): 100, 200, 300
    raw_data[0, 2, 2] = 100
    raw_data[1, 2, 2] = 200
    raw_data[2, 2, 2] = 300

    # 4. Decoded exact 0.0 cell: raw * 0.5 + 20.0 = 0.0 -> raw = -40 (inside [-100, 500])
    raw_data[0, 3, 3] = -40

    # 5. valid_min violation: raw = -200 (< valid_min -100) on day 0
    raw_data[0, 4, 4] = -200

    # 6. valid_max violation: raw = 600 (> valid_max 500) on day 1
    raw_data[1, 4, 4] = 600

    with h5netcdf.File(nc_path, "w") as f:
        # Dimensions
        f.dimensions = {"time": n_time, "latitude": n_lat, "longitude": n_lon}

        # Coordinates
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lat.attrs["units"] = "degrees_north"
        v_lat.attrs["axis"] = "Y"
        v_lat.attrs["standard_name"] = "latitude"

        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_lon.attrs["units"] = "degrees_east"
        v_lon.attrs["axis"] = "X"
        v_lon.attrs["standard_name"] = "longitude"

        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = times
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"
        v_time.attrs["calendar"] = "gregorian"
        v_time.attrs["axis"] = "T"
        v_time.attrs["standard_name"] = "time"

        # Data variable
        v_data = f.create_variable("test_packed_sst", ("time", "latitude", "longitude"), dtype=np.int16)
        v_data[:] = raw_data
        v_data.attrs["_FillValue"] = np.int16(-999)
        v_data.attrs["scale_factor"] = np.float32(0.5)
        v_data.attrs["add_offset"] = np.float32(20.0)
        v_data.attrs["valid_min"] = np.int16(-100)
        v_data.attrs["valid_max"] = np.int16(500)
        v_data.attrs["units"] = "degree_Celsius"
        v_data.attrs["standard_name"] = "sea_surface_temperature"

        # Flag variable
        v_flag = f.create_variable("test_mask", ("time", "latitude", "longitude"), dtype=np.int8)
        mask_data = np.ones((n_time, n_lat, n_lon), dtype=np.int8)
        mask_data[:, 0, 0] = 2  # land
        v_flag[:] = mask_data
        v_flag.attrs["flag_values"] = np.array([1, 2], dtype=np.int8)
        v_flag.attrs["flag_meanings"] = "water land"
        v_flag.attrs["standard_name"] = "status_flag"

    return nc_path


def test_header_attribute_presence_absence_detection(synthetic_contract_dataset):
    """Verify that dump_header_report correctly detects present vs absent attributes."""
    report_text, summary = dump_header_report(synthetic_contract_dataset)

    assert "test_packed_sst" in summary["variables"]
    var_meta = summary["variables"]["test_packed_sst"]

    # Present attributes
    assert var_meta["contract_attributes"]["_FillValue"] == "-999"
    assert var_meta["contract_attributes"]["valid_min"] == "-100"
    assert var_meta["contract_attributes"]["valid_max"] == "500"
    assert "degree_Celsius" in var_meta["contract_attributes"]["units"]

    # Absent attributes
    assert var_meta["contract_attributes"]["missing_value"] is None
    assert var_meta["contract_attributes"]["flag_values"] is None
    assert var_meta["contract_attributes"]["flag_masks"] is None

    # Variable classification
    assert var_meta["classification"] == "data"
    assert summary["variables"]["test_mask"]["classification"] == "flag/mask"
    assert summary["variables"]["latitude"]["classification"] == "coordinate"


def test_latitude_orientation_and_longitude_convention(synthetic_contract_dataset):
    """Verify descending latitude direction and 0..360 longitude convention detection."""
    coords = audit_coordinates_and_time(synthetic_contract_dataset)

    # Latitude
    assert coords["latitude"]["direction"] == "descending"
    assert coords["latitude"]["monotonic"] is True
    assert coords["latitude"]["step"] == -5.0

    # Longitude
    assert coords["longitude"]["direction"] == "ascending"
    assert coords["longitude"]["convention"] == "0..360"
    assert coords["longitude"]["min"] == 190.0
    assert coords["longitude"]["max"] == 230.0


def test_hour_of_day_detection(synthetic_contract_dataset):
    """Verify 12:00 timestamp and daily step gap detection."""
    coords = audit_coordinates_and_time(synthetic_contract_dataset)

    time_info = coords["time"]
    assert time_info["hours_of_day"] == [12]
    assert time_info["count"] == 3
    assert time_info["has_gaps"] is False
    assert time_info["has_duplicates"] is False
    assert time_info["first_timestamp"] == "2019-01-01T12:00:00+00:00"
    assert time_info["last_timestamp"] == "2019-01-03T12:00:00+00:00"


def test_grid_alignment_offset_computation(synthetic_contract_dataset):
    """Verify grid alignment offset against project target centres."""
    coords = audit_coordinates_and_time(synthetic_contract_dataset)
    grid_align = coords["target_grid_alignment"]

    # Target centres are in 5.125..29.875. Synthetic lats are [30, 25, 20, 15, 10, 5].
    # Distance to target centre 5.125 from 5.0 is 0.125 deg = 0.025 cell units (0.125 / 5.0).
    assert grid_align["lat_exact_within_1e-4"] is False
    assert grid_align["lon_exact_within_1e-4"] is False
    assert grid_align["lat_offset_cells"] > 0.0


def test_frozen_cells_and_valid_violations(synthetic_contract_dataset):
    """Verify frozen cell detection and valid_min/max violations counting."""
    df_contract, df_extremes, df_frozen, flag_summary = audit_variable_raw_contract(
        synthetic_contract_dataset, "TEST_SRC"
    )

    sst_contract = df_contract[df_contract["variable"] == "test_packed_sst"]
    assert len(sst_contract) == 3

    # All days fill mask identical
    assert bool(sst_contract["fill_mask_identical_all_days"].iloc[0]) is True
    assert sst_contract["fill_cells"].iloc[0] == 1  # (0, 0) is fill

    # Day 0: valid_min violation (-300 < -100)
    day0 = sst_contract[sst_contract["day_index"] == 0].iloc[0]
    assert day0["valid_min_violations"] == 1
    assert day0["valid_max_violations"] == 0
    assert day0["decoded_exact_zeros"] == 1  # cell with raw -2000 -> 0.0

    # Day 1: valid_max violation (700 > 500)
    day1 = sst_contract[sst_contract["day_index"] == 1].iloc[0]
    assert day1["valid_min_violations"] == 0
    assert day1["valid_max_violations"] == 1

    # Frozen cells
    frozen_for_var = df_frozen[df_frozen["variable"] == "test_packed_sst"]
    assert len(frozen_for_var) > 0
    # Cell (1, 1) was set to 150 on all days
    frozen_cell_11 = frozen_for_var[(frozen_for_var["lat_idx"] == 1) & (frozen_for_var["lon_idx"] == 1)]
    assert len(frozen_cell_11) == 1
    assert frozen_cell_11.iloc[0]["raw_value"] == 150


def test_coverage_counts_against_known_mask(tmp_path):
    """Verify coverage categorization against a known baseline mask."""
    # Create synthetic GLORYS-like file
    glo_path = str(tmp_path / "synthetic_glorys.nc")
    bathy_path = str(tmp_path / "synthetic_bathy.nc")
    src_path = str(tmp_path / "synthetic_source.nc")

    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([50.0, 51.0], dtype=np.float32)

    # GLORYS: cell (0, 0) wet (finite), (0, 1) wet, (1, 0) dry (fill), (1, 1) dry
    glo_raw = np.array([[[[100, 100], [-32767, -32767]]]], dtype=np.int16)  # shape: (1, 1, 2, 2)

    with h5netcdf.File(glo_path, "w") as f:
        f.dimensions = {"time": 1, "depth": 1, "latitude": 2, "longitude": 2}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_th = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), dtype=np.int16)
        v_th[:] = glo_raw
        v_th.attrs["_FillValue"] = np.int16(-32767)

    with h5netcdf.File(bathy_path, "w") as f:
        f.dimensions = {"latitude": 2, "longitude": 2}
        v_dp = f.create_variable("deptho", ("latitude", "longitude"), dtype=np.float32)
        v_dp[:] = np.array([[50.0, 100.0], [0.0, 0.0]], dtype=np.float32)

    # Source: cell (0, 0) finite, (0, 1) fill, (1, 0) finite, (1, 1) fill
    # Expected categories vs GLORYS:
    # (0, 0): GLORYS wet & Source finite -> 1
    # (0, 1): GLORYS wet & Source fill   -> 1
    # (1, 0): GLORYS dry & Source finite -> 1
    # (1, 1): GLORYS dry & Source fill   -> 1
    src_raw = np.array([[[500, -999], [500, -999]]], dtype=np.int16)  # shape: (1, 2, 2)
    with h5netcdf.File(src_path, "w") as f:
        f.dimensions = {"time": 1, "latitude": 2, "longitude": 2}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = [0.0]
        v_time.attrs["units"] = "days since 2019-01-01 00:00:00"
        v_var = f.create_variable("sst", ("time", "latitude", "longitude"), dtype=np.int16)
        v_var[:] = src_raw
        v_var.attrs["_FillValue"] = np.int16(-999)

    df_cov, df_miss = audit_coverage_vs_glorys(
        src_path, "TEST_SRC", ["sst"], glo_path, bathy_path
    )

    assert len(df_cov) == 1
    row = df_cov.iloc[0]
    assert row["glorys_finite_count"] == 2
    assert row["glorys_finite_source_finite"] == 1
    assert row["glorys_finite_source_fill"] == 1
    assert row["glorys_dry_source_finite"] == 1
    assert row["glorys_dry_source_fill"] == 1

    # Check missing cell listing
    assert len(df_miss) == 1
    miss = df_miss.iloc[0]
    assert miss["glorys_lat_idx"] == 0
    assert miss["glorys_lon_idx"] == 1
    assert miss["static_deptho_m"] == 100.0


def test_immutability_and_sha256_preservation(synthetic_contract_dataset):
    """Verify that auditing operations do not mutate synthetic files and SHA256 remains identical."""
    sha_before = compute_file_sha256(synthetic_contract_dataset)
    size_before = os.path.getsize(synthetic_contract_dataset)

    # Perform all audit queries
    dump_header_report(synthetic_contract_dataset)
    audit_coordinates_and_time(synthetic_contract_dataset)
    audit_variable_raw_contract(synthetic_contract_dataset, "TEST_SRC")

    sha_after = compute_file_sha256(synthetic_contract_dataset)
    size_after = os.path.getsize(synthetic_contract_dataset)

    assert sha_before == sha_after
    assert size_before == size_after
