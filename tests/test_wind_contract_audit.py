"""
Unit tests for wind contract audit using synthetic fixtures.
Tests all modular audit functions without accessing raw 6.4GB data.
"""

import datetime
import os
import tempfile
import h5netcdf
import numpy as np
import pytest

from scripts.diagnostics.wind_contract_audit import (
    AUDITED_VARIABLES,
    TARGET_LAT_CENTRES,
    TARGET_LON_CENTRES,
    aggregate_2x2_block,
    check_grid_nesting,
    compare_coverage_with_glorys,
    compute_daily_mean_and_std,
    compute_file_sha256,
    count_valid_hours,
    decode_raw_array,
    decode_time_axis,
    dump_header_metadata,
    get_file_stats,
    parse_time_units,
    verify_file_unchanged,
)


def test_sha256_and_unchanged_verification():
    """Verify SHA-256 calculation and file integrity verification."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"OceanEmbed Wind Test 1234567890")
        tmp_path = tmp.name

    try:
        pre_stats = get_file_stats(tmp_path, chunk_size=1024)
        assert pre_stats["size_bytes"] == 31
        assert len(pre_stats["sha256"]) == 64
        assert verify_file_unchanged(pre_stats, pre_stats) is True

        # Modify file and verify detection
        with open(tmp_path, "wb") as f:
            f.write(b"Modified Content")
        post_stats = get_file_stats(tmp_path, chunk_size=1024)
        assert verify_file_unchanged(pre_stats, post_stats) is False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_parse_time_units():
    """Test parsing of CF convention time units."""
    unit, dt = parse_time_units("seconds since 1990-01-01")
    assert unit == "seconds"
    assert dt == datetime.datetime(1990, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    unit2, dt2 = parse_time_units("hours since 2010-05-15 12:00:00")
    assert unit2 == "hours"
    assert dt2 == datetime.datetime(2010, 5, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

    with pytest.raises(ValueError):
        parse_time_units("invalid time unit string")


def test_decode_time_axis():
    """Test decoding of hourly time values over a synthetic month."""
    # Base: 1990-01-01. Seconds from 1990-01-01 to 2019-01-01T00:00:00
    base = datetime.datetime(1990, 1, 1, tzinfo=datetime.timezone.utc)
    t0_dt = datetime.datetime(2019, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    t0_sec = int((t0_dt - base).total_seconds())

    # 744 hours
    times_744 = np.array([t0_sec + i * 3600 for i in range(744)], dtype=np.int64)
    res = decode_time_axis(times_744, "seconds since 1990-01-01")

    assert res["count"] == 744
    assert res["first_dt"] == "2019-01-01T00:00:00+00:00"
    assert res["last_dt"] == "2019-01-31T23:00:00+00:00"
    assert res["step_set"] == [3600]
    assert res["is_exact_hourly_month"] is True

    # Test irregular or short times
    times_bad = np.array([t0_sec, t0_sec + 3600, t0_sec + 7200, t0_sec + 10800 + 100])
    res_bad = decode_time_axis(times_bad, "seconds since 1990-01-01")
    assert res_bad["is_exact_hourly_month"] is False


def test_check_grid_nesting():
    """Test 2x2 grid nesting verification."""
    # Exact native grid: 200 x 480
    native_lats = np.linspace(5.0625, 29.9375, 200)
    native_lons = np.linspace(45.0625, 104.9375, 480)
    res = check_grid_nesting(native_lats, native_lons, TARGET_LAT_CENTRES, TARGET_LON_CENTRES)

    assert res["pass"] is True
    assert res["ratio_is_2x2"] is True
    assert res["edges_nest_exact"] is True
    assert res["centers_nest_exact"] is True
    assert res["max_lat_edge_diff"] < 1e-4
    assert res["max_lon_edge_diff"] < 1e-4

    # Test inverted latitude (fail)
    res_inverted = check_grid_nesting(native_lats[::-1], native_lons, TARGET_LAT_CENTRES, TARGET_LON_CENTRES)
    assert res_inverted["pass"] is False
    assert res_inverted["lat_increasing"] is False

    # Test shifted native grid (fail)
    res_shifted = check_grid_nesting(native_lats + 0.05, native_lons, TARGET_LAT_CENTRES, TARGET_LON_CENTRES)
    assert res_shifted["pass"] is False
    assert res_shifted["edges_nest_exact"] is False


def test_decode_raw_array_and_valid_hours():
    """Test decode_raw_array and count_valid_hours on synthetic data."""
    fill_val = -32767
    scale = 0.01
    offset = 0.0

    raw = np.array([
        [100, fill_val],
        [fill_val, 200],
    ], dtype=np.int16)

    decoded = decode_raw_array(raw, fill_val, scale, offset)
    assert np.isclose(decoded[0, 0], 1.0)
    assert np.isnan(decoded[0, 1])
    assert np.isnan(decoded[1, 0])
    assert np.isclose(decoded[1, 1], 2.0)


def test_compute_daily_mean_and_std():
    """Test daily aggregation and within-day std dev on 24-hour synthetic chunk."""
    fill_val = -2147483647
    scale = 0.01
    offset = 0.0

    # 24 hours, 2x2 spatial grid
    chunk = np.full((24, 2, 2), fill_val, dtype=np.int32)

    # Cell (0, 0): all 24 hours valid, values 100 to 2400 (scaled: 1.0 to 24.0)
    for h in range(24):
        chunk[h, 0, 0] = (h + 1) * 100

    # Cell (0, 1): 18 hours valid (all equal to 500, scaled: 5.0)
    for h in range(18):
        chunk[h, 0, 1] = 500

    # Cell (1, 0): 1 hour valid (scaled: 3.0)
    chunk[0, 1, 0] = 300

    # Cell (1, 1): 0 hours valid (all fill)

    valid_hours, daily_mean, within_day_std = compute_daily_mean_and_std(chunk, fill_val, scale, offset)

    # Check valid hours
    assert valid_hours[0, 0] == 24
    assert valid_hours[0, 1] == 18
    assert valid_hours[1, 0] == 1
    assert valid_hours[1, 1] == 0

    # Check daily mean
    expected_mean_00 = np.mean(np.arange(1.0, 25.0))
    assert np.isclose(daily_mean[0, 0], expected_mean_00)
    assert np.isclose(daily_mean[0, 1], 5.0)
    assert np.isclose(daily_mean[1, 0], 3.0)
    assert np.isnan(daily_mean[1, 1])

    # Check within-day std (population ddof=0)
    expected_std_00 = np.std(np.arange(1.0, 25.0), ddof=0)
    assert np.isclose(within_day_std[0, 0], expected_std_00)
    assert np.isclose(within_day_std[0, 1], 0.0)  # constant values
    assert np.isclose(within_day_std[1, 0], 0.0)  # single sample
    assert np.isnan(within_day_std[1, 1])


def test_frozen_cells_detection_logic():
    """Test frozen cell detection: cell_min == cell_max with valid hours >= 2."""
    cell_valid_hours = np.array([24, 24, 0, 1])
    cell_min_raw = np.array([50, 50, 0, 10])
    cell_max_raw = np.array([50, 100, 0, 10])  # cell 0 is constant, cell 1 varies

    is_frozen = (cell_valid_hours >= 2) & (cell_min_raw == cell_max_raw)
    is_stationary_fill = (cell_valid_hours == 0)

    assert is_frozen[0] == True
    assert is_frozen[1] == False
    assert is_frozen[2] == False
    assert is_frozen[3] == False

    assert is_stationary_fill[2] == True


def test_aggregate_2x2_block():
    """Test 2x2 native to 1x1 target aggregation with valid fraction f."""
    # 4x4 native -> 2x2 target
    daily_mean = np.array([
        [1.0, 2.0,  5.0, np.nan],
        [3.0, 4.0,  np.nan, 7.0],
        [np.nan, np.nan, np.nan, np.nan],
        [9.0, np.nan, np.nan, np.nan],
    ])
    valid_hours = np.array([
        [24, 24, 24, 0],
        [24, 24, 0, 24],
        [0, 0, 0, 0],
        [24, 0, 0, 0],
    ])

    target_values, f_grid = aggregate_2x2_block(daily_mean, valid_hours)

    assert target_values.shape == (2, 2)
    assert f_grid.shape == (2, 2)

    # Block (0, 0): all 4 valid -> f = 1.0, mean = (1+2+3+4)/4 = 2.5
    assert f_grid[0, 0] == 1.0
    assert np.isclose(target_values[0, 0], 2.5)

    # Block (0, 1): 2 valid -> f = 0.5, mean = (5+7)/2 = 6.0
    assert f_grid[0, 1] == 0.5
    assert np.isclose(target_values[0, 1], 6.0)

    # Block (1, 0): 1 valid -> f = 0.25, mean = 9.0
    assert f_grid[1, 0] == 0.25
    assert np.isclose(target_values[1, 0], 9.0)

    # Block (1, 1): 0 valid -> f = 0.0, mean = NaN
    assert f_grid[1, 1] == 0.0
    assert np.isnan(target_values[1, 1])


def test_compare_coverage_with_glorys():
    """Test coverage comparison against GLORYS depth 0 mask."""
    f_grid = np.array([
        [1.0, 0.5],
        [0.0, 0.75],
    ])
    glorys_valid_d0 = np.array([
        [True, True],
        [True, False],
    ])
    lats = np.array([10.0, 20.0])
    lons = np.array([50.0, 60.0])

    cov = compare_coverage_with_glorys(f_grid, glorys_valid_d0, lats, lons)

    assert cov["total_target_cells"] == 4
    assert cov["count_f_eq_1"] == 1
    assert cov["count_f_ge_0_5"] == 3
    assert cov["count_f_gt_0"] == 3
    assert cov["count_f_eq_0"] == 1
    assert cov["glorys_wet_cells"] == 3
    # GLORYS wet & f == 0: cell (1, 0)
    assert cov["glorys_wet_with_f_0"] == 1
    # GLORYS wet & f < 1: cell (0, 1) [f=0.5] and cell (1, 0) [f=0.0]
    assert cov["glorys_wet_with_f_lt_1"] == 2
    assert len(cov["mismatch_samples"]) == 2


def test_dump_header_metadata_toy():
    """Test header metadata extraction on toy NetCDF file."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_nc:
        nc_path = tmp_nc.name

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_txt:
        txt_path = tmp_txt.name

    try:
        with h5netcdf.File(nc_path, "w") as f:
            f.attrs["title"] = "Toy Wind Test"
            f.attrs["institution"] = "OceanEmbed Test"
            f.dimensions = {"time": 2, "latitude": 2, "longitude": 2}

            # Coordinates
            t = f.create_variable("time", ("time",), dtype=np.int32)
            t.attrs["units"] = "seconds since 1990-01-01"
            lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
            lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)

            # Audited variables
            for v in AUDITED_VARIABLES:
                var = f.create_variable(v, ("time", "latitude", "longitude"), dtype=np.int32)
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
                var.attrs["_FillValue"] = -2147483647
                var.attrs["valid_min"] = -5000
                var.attrs["valid_max"] = 5000
                var.attrs["units"] = "N m-2"

            # Unaudited variable
            uvar = f.create_variable("air_density", ("time", "latitude", "longitude"), dtype=np.float32)
            uvar.attrs["units"] = "kg m-3"

        info = dump_header_metadata(nc_path, txt_path)
        assert info["total_variables_count"] == 3 + len(AUDITED_VARIABLES) + 1
        assert "air_density" in info["unaudited_data_variables"]
        assert "eastward_stress" in info["audited_variables"]
        assert info["audited_contract"]["eastward_stress"]["scale_factor"] == 0.01

        assert os.path.exists(txt_path)
        with open(txt_path, "r", encoding="utf-8") as ft:
            content = ft.read()
            assert "Toy Wind Test" in content
            assert "[AUDITED]" in content
            assert "[METADATA-ONLY]" in content
    finally:
        if os.path.exists(nc_path):
            os.remove(nc_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)

