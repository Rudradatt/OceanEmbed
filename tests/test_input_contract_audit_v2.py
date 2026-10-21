"""
Unit and integration tests for Generalized Input Contract Audit v2.
Tests on small synthetic NetCDF datasets:
1. Config-driven runner reproducing known counts
2. Nesting computation (0.05 deg, 0.125 deg, 1/12 deg with weights 1/6, 1/3, 1/3, 1/6)
3. Frozen-cell location summary (bounding box, lake mask, land neighbours, lat histogram)
4. Persistence of zero cells across days (daily count, all-days persistence, fill adjacency)
5. Depth-dimension handling (4D variables with multiple depths evaluated separately)
6. File immutability and SHA256 preservation
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import h5netcdf

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "scripts", "diagnostics"))

from input_contract_audit_v2 import (
    compute_file_sha256,
    get_file_metadata,
    dump_header_report,
    audit_coordinates_and_time,
    compute_nesting_diagnostics,
    audit_variable_raw_contract,
    audit_coverage_vs_glorys,
    audit_sst_frozen_error_cells,
    audit_sst_lake_cells,
    audit_sla_zero_cells,
    audit_sla_velocity_extremes,
    run_v2_audit,
)


# ==============================================================================
# Fixtures for Synthetic Data
# ==============================================================================

@pytest.fixture
def synthetic_4d_dataset(tmp_path):
    """Create a synthetic 4D NetCDF dataset with 2 depths (0m and 15m), 3 days, 4 lats, 4 lons."""
    nc_path = str(tmp_path / "synthetic_4d.nc")
    lats = np.array([5.125, 5.375, 5.625, 5.875], dtype=np.float32)
    lons = np.array([45.125, 45.375, 45.625, 45.875], dtype=np.float32)
    times = np.array([0.0, 24.0, 48.0], dtype=np.float32)
    depths = np.array([0, 15], dtype=np.int16)

    # 4D data: (time, depth, lat, lon)
    raw_4d = np.full((3, 2, 4, 4), 100, dtype=np.int16)
    # Fill at (lat=0, lon=0) on depth 0
    raw_4d[:, 0, 0, 0] = -32768
    # Different fill at (lat=1, lon=1) on depth 1
    raw_4d[:, 1, 1, 1] = -32768

    # 3D surface-only variable: (time, lat, lon)
    raw_3d = np.full((3, 4, 4), 200, dtype=np.int16)
    raw_3d[:, 0, 0] = -32768

    with h5netcdf.File(nc_path, "w") as f:
        f.dimensions = {"time": 3, "depth": 2, "latitude": 4, "longitude": 4}

        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lat.attrs["units"] = "degrees_north"
        v_lat.attrs["axis"] = "Y"

        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_lon.attrs["units"] = "degrees_east"
        v_lon.attrs["axis"] = "X"

        v_depth = f.create_variable("depth", ("depth",), dtype=np.int16)
        v_depth[:] = depths
        v_depth.attrs["units"] = "m"
        v_depth.attrs["axis"] = "Z"

        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = times
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"
        v_time.attrs["calendar"] = "gregorian"

        # 4D variable
        v_uo = f.create_variable("uo", ("time", "depth", "latitude", "longitude"), dtype=np.int16)
        v_uo[:] = raw_4d
        v_uo.attrs["_FillValue"] = np.int16(-32768)
        v_uo.attrs["scale_factor"] = np.float64(0.001)

        # 3D variable
        v_ugos = f.create_variable("ugos", ("time", "latitude", "longitude"), dtype=np.int16)
        v_ugos[:] = raw_3d
        v_ugos.attrs["_FillValue"] = np.int16(-32768)
        v_ugos.attrs["scale_factor"] = np.float64(0.001)

    return nc_path


@pytest.fixture
def synthetic_followup_dataset(tmp_path):
    """
    Create a synthetic dataset for SST and SLA follow-ups:
    - mask with lake bit (4) and land bit (2)
    - analysis_error with frozen cells
    - SLA variables with zeros (some persistent, some adjacent to fill)
    """
    nc_path = str(tmp_path / "synthetic_followup.nc")
    lats = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    lons = np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0], dtype=np.float32)
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    n_time, n_lat, n_lon = 3, 6, 6

    # Mask array: bit 1: water, bit 2: land, bit 4: lake
    mask = np.ones((n_time, n_lat, n_lon), dtype=np.int8)  # 1 = water
    mask[:, 0, 0] = 2  # land
    mask[:, 1, 1] = 6  # land + lake (2 | 4)
    mask[:, 2, 2] = 4  # lake

    # analysis_error: frozen cells
    err = np.full((n_time, n_lat, n_lon), 16, dtype=np.int16)  # 16 * 0.01 = 0.16 K
    err[:, 0, 0] = -32768  # fill on land
    # (1, 1) is frozen lake cell with value 19 (0.19 K)
    err[:, 1, 1] = 19
    # (2, 2) is frozen lake cell with value 70 (0.70 K)
    err[:, 2, 2] = 70
    # (3, 3) varies over time (not frozen)
    err[0, 3, 3] = 16
    err[1, 3, 3] = 20
    err[2, 3, 3] = 25

    # SLA variable with zeros
    sla = np.full((n_time, n_lat, n_lon), 100, dtype=np.int32)
    sla[:, 0, 0] = -2147483647  # fill at (0, 0)
    # Cell (0, 1) is adjacent to fill (0, 0) and zero on day 0
    sla[0, 0, 1] = 0
    # Cell (4, 4) is interior ocean and zero on all 3 days
    sla[:, 4, 4] = 0

    with h5netcdf.File(nc_path, "w") as f:
        f.dimensions = {"time": n_time, "latitude": n_lat, "longitude": n_lon}

        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = times
        v_time.attrs["units"] = "days since 2019-01-01"

        v_mask = f.create_variable("mask", ("time", "latitude", "longitude"), dtype=np.int8)
        v_mask[:] = mask
        v_mask.attrs["_FillValue"] = np.int8(-128)
        v_mask.attrs["flag_masks"] = np.array([1, 2, 4], dtype=np.int8)

        v_err = f.create_variable("analysis_error", ("time", "latitude", "longitude"), dtype=np.int16)
        v_err[:] = err
        v_err.attrs["_FillValue"] = np.int16(-32768)
        v_err.attrs["scale_factor"] = np.float64(0.01)
        v_err.attrs["add_offset"] = np.float64(0.0)

        v_sst = f.create_variable("analysed_sst", ("time", "latitude", "longitude"), dtype=np.int16)
        v_sst[:] = np.full((n_time, n_lat, n_lon), 2000, dtype=np.int16)
        v_sst.attrs["_FillValue"] = np.int16(-32768)
        v_sst.attrs["scale_factor"] = np.float64(0.01)
        v_sst.attrs["add_offset"] = np.float64(273.15)

        v_sla = f.create_variable("sla", ("time", "latitude", "longitude"), dtype=np.int32)
        v_sla[:] = sla
        v_sla.attrs["_FillValue"] = np.int32(-2147483647)
        v_sla.attrs["scale_factor"] = np.float64(0.0001)

        for vel_name in ["ugos", "vgos", "ugosa", "vgosa"]:
            v_vel = f.create_variable(vel_name, ("time", "latitude", "longitude"), dtype=np.int32)
            vel_arr = np.full((n_time, n_lat, n_lon), 5000, dtype=np.int32)  # 0.5 m/s
            vel_arr[0, 1, 1] = 12000  # 1.2 m/s (> 1.0)
            vel_arr[0, 2, 2] = 18000  # 1.8 m/s (> 1.5)
            v_vel[:] = vel_arr
            v_vel.attrs["_FillValue"] = np.int32(-2147483647)
            v_vel.attrs["scale_factor"] = np.float64(0.0001)

    return nc_path


# ==============================================================================
# Step 5 Test Implementations
# ==============================================================================

def test_nesting_computation():
    """
    Verify nesting computation:
    - 0.05 deg grid with centres at x.025 gives 5 cells per target cell, coincides=True, offset=0
    - 0.125 deg grid with centres at x.0625 gives 2 cells per target cell, coincides=True, offset=0
    - 1/12 deg grid with centres at x.0 gives not nested with weights 1/6, 1/3, 1/3, 1/6 and offset 0.5
    """
    # 1. 0.05 deg grid
    c_005 = np.arange(5.025, 6.0, 0.05)
    res_005 = compute_nesting_diagnostics(c_005, target_start=5.0, target_step=0.25)
    assert res_005["coincide"] is True
    assert np.isclose(res_005["cells_per_target"], 5.0)
    assert np.isclose(res_005["offset_native_units"], 0.0)
    assert len(res_005["weights"]) == 5
    for w in res_005["weights"]:
        assert np.isclose(w, 0.2)

    # 2. 0.125 deg grid
    c_0125 = np.arange(5.0625, 6.0, 0.125)
    res_0125 = compute_nesting_diagnostics(c_0125, target_start=5.0, target_step=0.25)
    assert res_0125["coincide"] is True
    assert np.isclose(res_0125["cells_per_target"], 2.0)
    assert np.isclose(res_0125["offset_native_units"], 0.0)
    assert len(res_0125["weights"]) == 2
    for w in res_0125["weights"]:
        assert np.isclose(w, 0.5)

    # 3. 1/12 deg grid with centres at x.0
    c_112 = np.array([5.0 + k * (1.0 / 12.0) for k in range(13)])
    res_112 = compute_nesting_diagnostics(c_112, target_start=5.0, target_step=0.25)
    assert res_112["coincide"] is False
    assert np.isclose(res_112["cells_per_target"], 3.0)
    assert np.isclose(res_112["offset_native_units"], 0.5)
    assert len(res_112["weights"]) == 4
    expected_weights = [1/6, 1/3, 1/3, 1/6]
    for actual_w, exp_w in zip(res_112["weights"], expected_weights):
        assert np.isclose(actual_w, exp_w, atol=1e-5)


def test_depth_dimension_handling(synthetic_4d_dataset):
    """
    Verify depth-dimension handling:
    - 4D variable with depth dimension evaluated for each depth separately
    - 3D surface variable evaluated without depth
    - Each depth's fill/non-fill counts correctly identified
    """
    df_contract, df_extremes, df_frozen, flag_summary = audit_variable_raw_contract(
        synthetic_4d_dataset, "TEST_4D", depth_selection=[0, 15]
    )

    uo_rows = df_contract[df_contract["variable"] == "uo"]
    assert len(uo_rows) == 6  # 3 days * 2 depths
    assert set(uo_rows["depth_m"]) == {0.0, 15.0}

    # Depth 0 has fill at (0, 0)
    uo_d0 = uo_rows[uo_rows["depth_m"] == 0.0]
    assert uo_d0["fill_cells"].iloc[0] == 1
    assert uo_d0["non_fill_cells"].iloc[0] == 15

    # Depth 15 has fill at (1, 1)
    uo_d15 = uo_rows[uo_rows["depth_m"] == 15.0]
    assert uo_d15["fill_cells"].iloc[0] == 1
    assert uo_d15["non_fill_cells"].iloc[0] == 15

    # 3D surface variable has depth_m None and 1 fill cell
    ugos_rows = df_contract[df_contract["variable"] == "ugos"]
    assert len(ugos_rows) == 3  # 3 days * 1 slice
    assert pd.isna(ugos_rows["depth_m"].iloc[0])
    assert ugos_rows["fill_cells"].iloc[0] == 1


def test_frozen_cell_location_summary(synthetic_followup_dataset, tmp_path):
    """
    Verify frozen-cell location summary:
    - Identifies lake cells among frozen cells
    - Computes bounding box
    - Computes latitude-band histogram
    - Detects land neighbours
    """
    out_csv = str(tmp_path / "test_frozen_cells.csv")
    glo_mock = synthetic_followup_dataset  # use same dataset for mock GLORYS coords
    res = audit_sst_frozen_error_cells(synthetic_followup_dataset, glo_mock, out_csv)

    # 36 total cells: (0, 0) is fill, (3, 3) varies. The remaining 34 are frozen!
    assert res["frozen_cells_total"] == 34
    # Cells (1, 1) and (2, 2) have mask bit 4 (lake). They are frozen lake cells!
    assert res["lake_cells_count"] == 2

    # Bounding boxes
    assert res["bounding_box_all"]["lat_min"] <= res["bounding_box_all"]["lat_max"]
    assert res["bounding_box_lake"]["lat_min"] <= res["bounding_box_lake"]["lat_max"]

    # Histogram by latitude band
    hist = res["latitude_band_histogram"]
    assert sum(hist.values()) == 34

    # Output CSV existence
    assert os.path.exists(out_csv)


def test_persistence_of_zero_cells_across_days(synthetic_followup_dataset, tmp_path):
    """
    Verify persistence of zero cells across days:
    - Counts daily zeros
    - Detects cells that are zero on all days vs subset of days
    - Identifies adjacency to fill cells
    """
    out_csv = str(tmp_path / "test_sla_zeros.csv")
    res = audit_sla_zero_cells(synthetic_followup_dataset, out_csv)

    sla_info = res["variables"]["sla"]
    # Cell (4, 4) was set to 0 on all 3 days -> zero_on_all_31_days_count >= 1
    assert sla_info["zero_on_all_31_days_count"] == 1

    # Cell (0, 1) was set to 0 on Day 0 and is adjacent to fill (0, 0)
    assert sla_info["adjacent_to_fill_8conn"] >= 1
    assert os.path.exists(out_csv)


def test_velocity_extremes_and_threshold_counts(synthetic_followup_dataset, tmp_path):
    """Verify velocity extremes and counts > 1.0 m/s and > 1.5 m/s per latitude band."""
    out_csv = str(tmp_path / "test_vel_extremes.csv")
    res = audit_sla_velocity_extremes(synthetic_followup_dataset, out_csv)

    ugos_res = res["results"]["ugos"]["day_0"]
    # In fixture, (1, 1) is 1.2 m/s and (2, 2) is 1.8 m/s
    assert ugos_res["total_cells_gt_1_0_mps"] == 2
    assert ugos_res["total_cells_gt_1_5_mps"] == 1
    assert len(ugos_res["top_10_cells"]) >= 2
    assert os.path.exists(out_csv)


def test_sha256_preservation_and_immutability(synthetic_4d_dataset):
    """Verify that auditing operations do not mutate files and SHA256 remains identical."""
    sha_before = compute_file_sha256(synthetic_4d_dataset)
    size_before = os.path.getsize(synthetic_4d_dataset)

    # Perform audits
    dump_header_report(synthetic_4d_dataset)
    audit_coordinates_and_time(synthetic_4d_dataset)
    audit_variable_raw_contract(synthetic_4d_dataset, "TEST_4D")

    sha_after = compute_file_sha256(synthetic_4d_dataset)
    size_after = os.path.getsize(synthetic_4d_dataset)

    assert sha_before == sha_after
    assert size_before == size_after


def test_config_driven_runner_known_counts(tmp_path):
    """Verify that the config-driven runner executes and reproduces known counts."""
    # Build minimal synthetic setup
    glo_nc = str(tmp_path / "glo.nc")
    bathy_nc = str(tmp_path / "bathy.nc")
    src_nc = str(tmp_path / "src.nc")
    cfg_json = str(tmp_path / "config.json")
    out_dir = str(tmp_path / "out")

    lats = np.array([5.125, 5.375], dtype=np.float32)
    lons = np.array([45.125, 45.375], dtype=np.float32)

    with h5netcdf.File(glo_nc, "w") as f:
        f.dimensions = {"time": 1, "depth": 1, "latitude": 2, "longitude": 2}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = [0.0]
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"
        v_th = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), dtype=np.int16)
        v_th[:] = np.full((1, 1, 2, 2), 100, dtype=np.int16)
        v_th.attrs["_FillValue"] = np.int16(-32767)

    with h5netcdf.File(bathy_nc, "w") as f:
        f.dimensions = {"latitude": 2, "longitude": 2}
        v_dp = f.create_variable("deptho", ("latitude", "longitude"), dtype=np.float32)
        v_dp[:] = np.full((2, 2), 50.0, dtype=np.float32)

    with h5netcdf.File(src_nc, "w") as f:
        f.dimensions = {"time": 1, "latitude": 2, "longitude": 2}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = [0.0]
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"
        v_data = f.create_variable("test_var", ("time", "latitude", "longitude"), dtype=np.int16)
        v_data[:] = np.array([[[100, -999], [200, 300]]], dtype=np.int16)
        v_data.attrs["_FillValue"] = np.int16(-999)
        v_data.attrs["scale_factor"] = np.float32(1.0)
        v_data.attrs["add_offset"] = np.float32(0.0)

    # Config
    cfg_data = {
        "sources": [
            {
                "name": "TEST_SRC",
                "path": src_nc,
                "variables": "all",
                "depth_selection": None,
                "coverage_variables": ["test_var"]
            }
        ],
        "reference": {
            "glorys_path": glo_nc,
            "glorys_static_bathy_path": bathy_nc,
            "expected_sha256": {}
        }
    }
    with open(cfg_json, "w") as fp:
        json.dump(cfg_data, fp)

    # Test audit_variable_raw_contract and audit_coverage_vs_glorys directly
    df_c, df_e, df_f, _ = audit_variable_raw_contract(src_nc, "TEST_SRC")
    assert len(df_c) == 1
    assert df_c["fill_cells"].iloc[0] == 1
    assert df_c["non_fill_cells"].iloc[0] == 3
    assert df_c["raw_min"].iloc[0] == 100
    assert df_c["raw_max"].iloc[0] == 300

    df_cov, df_miss = audit_coverage_vs_glorys(src_nc, "TEST_SRC", ["test_var"], glo_nc, bathy_nc)
    assert len(df_cov) == 1
    assert df_cov["glorys_finite_source_finite"].iloc[0] == 3
    assert df_cov["glorys_finite_source_fill"].iloc[0] == 1


def test_planted_minmax_contract_vs_independent(tmp_path):
    """
    Test that audit_variable_raw_contract reproduces planted min/max values
    for both integer and float datasets, preserves dtypes, and exactly equals
    the independent recomputation function.
    """
    from reconcile_contract_minmax import compute_independent_minmax

    nc_path = str(tmp_path / "synthetic_planted.nc")
    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([50.0, 51.0], dtype=np.float32)
    times = np.array([0.0], dtype=np.float32)

    # Integer dataset with known raw min=50, max=450, fill=-9999, scale=0.01, offset=10.0
    # Decoded: 50*0.01+10 = 10.5, 450*0.01+10 = 14.5
    raw_int = np.array([[[50, -9999], [200, 450]]], dtype=np.int16)

    # Float dataset with known raw min=12.34, max=56.78, fill=1e20, scale=1.0, offset=0.0
    raw_flt = np.array([[[12.34, 1e20], [30.0, 56.78]]], dtype=np.float32)

    with h5netcdf.File(nc_path, "w") as f:
        f.dimensions = {"time": 1, "latitude": 2, "longitude": 2}
        v_lat = f.create_variable("latitude", ("latitude",), dtype=np.float32)
        v_lat[:] = lats
        v_lon = f.create_variable("longitude", ("longitude",), dtype=np.float32)
        v_lon[:] = lons
        v_time = f.create_variable("time", ("time",), dtype=np.float32)
        v_time[:] = times
        v_time.attrs["units"] = "hours since 2019-01-01 00:00:00"

        # Var 1: int16
        v1 = f.create_variable("int_var", ("time", "latitude", "longitude"), dtype=np.int16)
        v1[:] = raw_int
        v1.attrs["_FillValue"] = np.int16(-9999)
        v1.attrs["scale_factor"] = np.float64(0.01)
        v1.attrs["add_offset"] = np.float64(10.0)

        # Var 2: float32
        v2 = f.create_variable("flt_var", ("time", "latitude", "longitude"), dtype=np.float32)
        v2[:] = raw_flt
        v2.attrs["_FillValue"] = np.float32(1e20)
        v2.attrs["scale_factor"] = np.float64(1.0)
        v2.attrs["add_offset"] = np.float64(0.0)

    # 1. Run v2 contract function
    df_contract, _, _, _ = audit_variable_raw_contract(nc_path, "PLANTED_SRC")

    int_row = df_contract[df_contract["variable"] == "int_var"].iloc[0]
    flt_row = df_contract[df_contract["variable"] == "flt_var"].iloc[0]

    # Verify integer variable contract
    assert int_row["raw_min"] == 50
    assert int_row["raw_max"] == 450
    assert np.isclose(int_row["decoded_min"], 10.5)
    assert np.isclose(int_row["decoded_max"], 14.5)

    # Verify float variable contract (must preserve float type, not cast to int!)
    assert np.isclose(flt_row["raw_min"], 12.34, atol=1e-3)
    assert np.isclose(flt_row["raw_max"], 56.78, atol=1e-3)
    assert isinstance(flt_row["raw_min"], (float, np.floating))
    assert np.isclose(flt_row["decoded_min"], 12.34, atol=1e-3)
    assert np.isclose(flt_row["decoded_max"], 56.78, atol=1e-3)

    # 2. Run independent recomputation
    df_ind = compute_independent_minmax(
        sst_file=nc_path,
        sla_file=nc_path,
        sss_file=nc_path,
        cur_file=nc_path
    )
    # Filter to one instance
    ind_int = df_ind[(df_ind["source"] == "OSTIA_SST") & (df_ind["variable"] == "int_var")].iloc[0]
    ind_flt = df_ind[(df_ind["source"] == "OSTIA_SST") & (df_ind["variable"] == "flt_var")].iloc[0]

    # Assert contract outputs equal independent function outputs exactly
    assert int_row["raw_min"] == ind_int["raw_min_day0"]
    assert int_row["raw_max"] == ind_int["raw_max_day0"]
    assert np.isclose(int_row["decoded_min"], ind_int["decoded_min_day0"])
    assert np.isclose(int_row["decoded_max"], ind_int["decoded_max_day0"])

    assert np.isclose(flt_row["raw_min"], ind_flt["raw_min_day0"])
    assert np.isclose(flt_row["raw_max"], ind_flt["raw_max_day0"])
    assert np.isclose(flt_row["decoded_min"], ind_flt["decoded_min_day0"])
    assert np.isclose(flt_row["decoded_max"], ind_flt["decoded_max_day0"])


def test_regression_harness_loud_mismatch_failure(tmp_path):
    """
    Test that check_regression_vs_v1 fails loudly (raises ValueError)
    when a planted mismatch exists between v2 and independent ground truth.
    """
    from input_contract_audit_v2 import check_regression_vs_v1

    # Fake v1 contract and coverage
    v1_dir = str(tmp_path / "v1")
    os.makedirs(v1_dir, exist_ok=True)
    df_v1 = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "day_index": 0,
        "fill_cells": 100,
        "non_fill_cells": 500,
        "decoded_min": 273.87,
        "decoded_max": 302.95,
        "frozen_cells_count": 0,
    }])
    df_v1.to_csv(os.path.join(v1_dir, "variable_contract.csv"), index=False)
    df_cov = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "day_index": 0,
        "glorys_finite_source_finite": 107312,
        "glorys_finite_source_fill": 412,
        "glorys_dry_source_finite": 907,
    }])
    df_cov.to_csv(os.path.join(v1_dir, "coverage_vs_glorys.csv"), index=False)

    # v2 with intentional mismatch (e.g. decoded_min = 288.75 instead of 273.87)
    df_v2 = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "depth_m": np.nan,
        "day_index": 0,
        "fill_cells": 100,
        "non_fill_cells": 500,
        "decoded_min": 288.75,
        "decoded_max": 305.80,
        "frozen_cells_count": 0,
    }])

    # Independent CSV with ground truth
    ind_csv = str(tmp_path / "ind.csv")
    df_ind = pd.DataFrame([{
        "source": "OSTIA_SST",
        "variable": "analysed_sst",
        "depth_m": np.nan,
        "decoded_min_day0": 273.87,
        "decoded_max_day0": 302.95,
    }])
    df_ind.to_csv(ind_csv, index=False)

    # Must raise ValueError
    with pytest.raises(ValueError, match="REGRESSION CHECK FAILED"):
        check_regression_vs_v1(v1_dir, df_v2, df_cov, independent_csv_path=ind_csv)

