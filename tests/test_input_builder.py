"""
Tests for OceanEmbed Input Builder v1.
Validates synthetic toy grid aggregation, validity masking, decoders, nesting checks,
date validation, file roundtrips, and determinism.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import tempfile
from typing import Dict

import h5netcdf
import numpy as np
import pytest
import xarray as xr

from scripts.preprocessing.input_builder import (
    PROJECT_ROOT,
    block_aggregate,
    build_day_currents,
    build_day_sla,
    build_day_sss,
    build_day_sst,
    compute_currents_validity,
    compute_file_sha256,
    compute_sla_validity,
    compute_sss_validity,
    compute_sst_validity,
    decode_currents,
    decode_sla,
    decode_sss,
    decode_sst,
    decode_sst_degc,
    decode_timestamps,
    nesting_check,
    parse_time_units,
    verify_time_match,
)


# =============================================================================
# 1. Block Mean of Constant, Planted Invalid Cells, All-Invalid Block
# =============================================================================

def test_block_mean_constant_and_planted_invalid():
    # 2x2 blocks -> target shape 2x3, native shape 4x6
    c = 15.5
    values = np.full((4, 6), c, dtype=np.float64)
    valid = np.ones((4, 6), dtype=bool)

    # Block (0, 0): all 4 valid -> mean = c, frac = 1.0
    # Block (0, 1): 2 valid, 2 invalid -> mean = c, frac = 0.5
    valid[0, 2] = False
    valid[1, 3] = False

    # Block (1, 2): 0 valid, 4 invalid -> mean = NaN, frac = 0.0
    valid[2:4, 4:6] = False

    mean_val, frac = block_aggregate(values, valid, ratio=2, n_target=(2, 3))

    assert mean_val.shape == (2, 3)
    assert frac.shape == (2, 3)

    # Block (0, 0)
    assert np.isclose(mean_val[0, 0], c)
    assert np.isclose(frac[0, 0], 1.0)

    # Block (0, 1)
    assert np.isclose(mean_val[0, 1], c)
    assert np.isclose(frac[0, 1], 0.5)

    # Block (1, 2)
    assert np.isnan(mean_val[1, 2])
    assert np.isclose(frac[1, 2], 0.0)


def test_block_mean_varying_values():
    # 2x2 ratio, target 1x1
    values = np.array([[10.0, 20.0], [30.0, 40.0]])
    valid = np.array([[True, False], [True, True]])
    # Valid values: 10, 30, 40 -> mean = 80 / 3 = 26.666667, frac = 0.75
    mean_val, frac = block_aggregate(values, valid, ratio=2)
    assert mean_val.shape == (1, 1)
    assert np.isclose(mean_val[0, 0], 80.0 / 3.0)
    assert np.isclose(frac[0, 0], 0.75)


# =============================================================================
# 2. 1x1 Case Reproduces Input Exactly
# =============================================================================

def test_block_aggregate_1x1_reproduces_input_exactly():
    rng = np.random.default_rng(42)
    native_field = rng.standard_normal((10, 15))
    valid_mask = rng.choice([True, False], size=(10, 15), p=[0.7, 0.3])

    mean_val, frac = block_aggregate(native_field, valid_mask, ratio=1)

    assert mean_val.shape == (10, 15)
    assert frac.shape == (10, 15)

    # Where valid: mean_val equals native_field exactly, frac equals 1.0
    assert np.all(mean_val[valid_mask] == native_field[valid_mask])
    assert np.all(frac[valid_mask] == 1.0)

    # Where invalid: mean_val is NaN, frac equals 0.0
    assert np.all(np.isnan(mean_val[~valid_mask]))
    assert np.all(frac[~valid_mask] == 0.0)


# =============================================================================
# 3. Nesting Check Fails Loudly on Mis-aligned Grids
# =============================================================================

def test_nesting_check_success():
    # Target grid: 100 cells, start 5.0, step 0.25, end 30.0
    # SST: 500 cells, start edge 5.0, step 0.05, end edge 30.0 -> ratio 5
    native_sst_lats = 5.025 + np.arange(500) * 0.05
    ratio_sst = nesting_check(native_sst_lats, target_start=5.0, target_step=0.25, n_target=100)
    assert ratio_sst == 5

    # SLA/SSS: 200 cells, step 0.125 -> ratio 2
    native_sla_lats = 5.0625 + np.arange(200) * 0.125
    ratio_sla = nesting_check(native_sla_lats, target_start=5.0, target_step=0.25, n_target=100)
    assert ratio_sla == 2

    # Currents: 100 cells, step 0.25 -> ratio 1
    native_cur_lats = 5.125 + np.arange(100) * 0.25
    ratio_cur = nesting_check(native_cur_lats, target_start=5.0, target_step=0.25, n_target=100)
    assert ratio_cur == 1


def test_nesting_check_misaligned_start():
    # Native start edge offset by 0.01 deg
    native_lats = 5.035 + np.arange(500) * 0.05
    with pytest.raises(ValueError, match="Start edges do not coincide"):
        nesting_check(native_lats, target_start=5.0, target_step=0.25, n_target=100)


def test_nesting_check_misaligned_end():
    # Only 499 cells (truncated)
    native_lats = 5.025 + np.arange(499) * 0.05
    with pytest.raises(ValueError, match="End edges do not coincide"):
        nesting_check(native_lats, target_start=5.0, target_step=0.25, n_target=100)


def test_nesting_check_non_integer_ratio():
    # Native step 0.2 -> ratio 0.25 / 0.2 = 1.25 (not an integer)
    native_lats = 5.1 + np.arange(125) * 0.2
    with pytest.raises(ValueError, match="not an integer"):
        nesting_check(native_lats, target_start=5.0, target_step=0.25, n_target=100)


def test_nesting_check_non_uniform_spacing():
    native_lats = 5.025 + np.arange(500) * 0.05
    native_lats[10] += 0.005  # perturb one center
    with pytest.raises(ValueError, match="not uniform"):
        nesting_check(native_lats, target_start=5.0, target_step=0.25, n_target=100)


# =============================================================================
# 4. SST Rule Excludes Land (bit 2) and Lake (bit 4)
# =============================================================================

def test_sst_rule_excludes_land_and_lake_bits():
    # mask & 6 == 0:
    # 0 -> valid (0 & 6 == 0)
    # 1 -> valid (1 & 6 == 0, ocean flag in OSTIA)
    # 2 -> land bit set (2 & 6 == 2 != 0) -> invalid
    # 4 -> lake bit set (4 & 6 == 4 != 0) -> invalid
    # 6 -> land + lake (6 & 6 == 6 != 0) -> invalid
    # 8 -> other bit (8 & 6 == 0) -> valid
    # 14 -> (14 & 6 == 6 != 0) -> invalid

    masks = np.array([0, 1, 2, 4, 6, 8, 14], dtype=np.int8)
    sst_raw = np.full_like(masks, 2500, dtype=np.int16)

    valid = compute_sst_validity(sst_raw, masks, fill_value=-32768)
    expected = np.array([True, True, False, False, False, True, False])
    assert np.array_equal(valid, expected)

    # Check fill value exclusion
    sst_raw[1] = -32768  # fill value on an ocean cell (mask=1)
    valid_fill = compute_sst_validity(sst_raw, masks, fill_value=-32768)
    assert not valid_fill[1]


# =============================================================================
# 5. Decoding Reproduces Known Values
# =============================================================================

def test_decoding_known_values():
    # SST: raw=2500 -> Kelvin: 2500 * 0.01 + 273.15 = 298.15; degC: 25.0
    raw_sst = np.array([2500], dtype=np.int16)
    k = decode_sst(raw_sst)
    c = decode_sst_degc(raw_sst)
    assert np.isclose(k[0], 298.15, atol=1e-4)
    assert np.isclose(c[0], 25.0, atol=1e-4)

    # SLA: raw=1500 -> 1500 * 0.0001 = 0.15 m
    raw_sla = np.array([1500], dtype=np.int32)
    m = decode_sla(raw_sla)
    assert np.isclose(m[0], 0.15, atol=1e-6)

    # SSS: raw=34.5 -> 34.5
    raw_sss = np.array([34.5], dtype=np.float32)
    s = decode_sss(raw_sss)
    assert np.isclose(s[0], 34.5, atol=1e-6)

    # Currents: raw=450 -> 450 * 0.001 = 0.45 m/s
    raw_cur = np.array([450], dtype=np.int16)
    vel = decode_currents(raw_cur)
    assert np.isclose(vel[0], 0.45, atol=1e-6)


def test_validity_functions():
    assert compute_sla_validity(np.array([-2147483647, 100]))[0] is np.bool_(False)
    assert compute_sla_validity(np.array([-2147483647, 100]))[1] is np.bool_(True)

    assert compute_sss_validity(np.array([-32767.0, 35.0]))[0] is np.bool_(False)
    assert compute_sss_validity(np.array([-32767.0, 35.0]))[1] is np.bool_(True)

    assert compute_currents_validity(np.array([32767, 150]))[0] is np.bool_(False)
    assert compute_currents_validity(np.array([32767, 150]))[1] is np.bool_(True)


# =============================================================================
# 6. Source With Mismatched Date Fails Loudly
# =============================================================================

def test_date_mismatch_fails_loudly():
    ref_times = np.array([0, 24, 48], dtype=np.float32)  # 2019-01-01, 01-02, 01-03
    ref_units = "hours since 2019-01-01 00:00:00"

    # Mismatched date (2019-01-04 instead of 2019-01-03)
    src_times_mismatch = np.array([0, 24, 72], dtype=np.float32)
    with pytest.raises(ValueError, match="date mismatch"):
        verify_time_match(src_times_mismatch, ref_units, ref_times, ref_units, "TestSource")

    # Non-00:00:00 timestamp (e.g. 12:00:00)
    src_times_hour = np.array([0, 24, 60], dtype=np.float32)
    with pytest.raises(ValueError, match="not 00:00:00 UTC"):
        verify_time_match(src_times_hour, ref_units, ref_times, ref_units, "TestSource")

    # Count mismatch
    src_times_count = np.array([0, 24], dtype=np.float32)
    with pytest.raises(ValueError, match="time count"):
        verify_time_match(src_times_count, ref_units, ref_times, ref_units, "TestSource")


# =============================================================================
# 7. Output File Roundtrip (Shape, Units, Attributes, NaN Pattern)
# =============================================================================

def test_output_roundtrip(tmp_path):
    out_file = str(tmp_path / "test_roundtrip.nc")

    n_time = 2
    n_lat = 4
    n_lon = 6

    lats = np.linspace(10.0, 20.0, n_lat, dtype=np.float32)
    lons = np.linspace(50.0, 60.0, n_lon, dtype=np.float32)
    times = np.array([0, 24], dtype=np.float32)

    rng = np.random.default_rng(123)
    sst_val = rng.uniform(20.0, 30.0, size=(n_time, n_lat, n_lon)).astype(np.float32)
    sst_frac = rng.uniform(0.0, 1.0, size=(n_time, n_lat, n_lon)).astype(np.float32)

    # Plant zeros in frac and NaNs in value
    sst_frac[0, 1, 1] = 0.0
    sst_val[0, 1, 1] = np.nan
    sst_frac[1, 2, 3] = 0.0
    sst_val[1, 2, 3] = np.nan

    ds = xr.Dataset(
        data_vars={
            "sst_c": (("time", "latitude", "longitude"), sst_val, {"units": "degC"}),
            "sst_frac": (("time", "latitude", "longitude"), sst_frac, {"units": "1"}),
        },
        coords={
            "time": ("time", times, {"units": "hours since 2019-01-01", "calendar": "gregorian"}),
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
        },
        attrs={
            "policy_name": "test_policy",
            "policy_version": "v1",
        },
    )

    encoding = {
        "sst_c": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan), "dtype": "float32"},
        "sst_frac": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan), "dtype": "float32"},
        "time": {"dtype": "float32"},
    }
    ds.to_netcdf(out_file, engine="h5netcdf", encoding=encoding)
    ds.close()

    # Read back and verify
    with xr.open_dataset(out_file, engine="h5netcdf") as ds_in:
        assert ds_in["sst_c"].shape == (n_time, n_lat, n_lon)
        assert ds_in["sst_frac"].shape == (n_time, n_lat, n_lon)
        assert ds_in["sst_c"].attrs["units"] == "degC"
        assert ds_in["sst_frac"].attrs["units"] == "1"
        assert ds_in.attrs["policy_name"] == "test_policy"

        # Check NaN pattern matches frac == 0
        val_in = ds_in["sst_c"].values
        frac_in = ds_in["sst_frac"].values

        assert np.isnan(val_in[0, 1, 1])
        assert frac_in[0, 1, 1] == 0.0
        assert np.isnan(val_in[1, 2, 3])
        assert frac_in[1, 2, 3] == 0.0

        # Where frac > 0, value should match original within float32 tolerance
        mask_valid = frac_in > 0
        np.testing.assert_allclose(val_in[mask_valid], sst_val[mask_valid], rtol=1e-6)


# =============================================================================
# 8. Determinism: Two Builds Give Identical Arrays
# =============================================================================

def test_determinism_two_builds_identical():
    rng = np.random.default_rng(999)
    raw = rng.integers(1000, 3000, size=(10, 20), dtype=np.int16)
    mask = rng.choice([1, 2, 4, 6], size=(10, 20))

    val1, frac1 = build_day_sst(raw, mask, ratio=(5, 5), n_target=(2, 4))
    val2, frac2 = build_day_sst(raw, mask, ratio=(5, 5), n_target=(2, 4))

    assert np.array_equal(frac1, frac2)
    # equal_nan check
    np.testing.assert_allclose(val1, val2, equal_nan=True)


# =============================================================================
# 9. Raw Test Inputs SHA256 Unchanged
# =============================================================================

def test_raw_files_sha256_unchanged():
    raw_files = [
        "data/raw/sst/sst_ostia_2019-01.nc",
        "data/raw/sla/sla_duacs_2019-01.nc",
        "data/raw/sss/sss_multiobs_2019-01.nc",
        "data/raw/currents/currents_multiobs_2019-01.nc",
        "data/raw/glorys/glorys_thetao_2019-01.nc",
        "data/raw/glorys_static/glorys_static_mask_bathy.nc",
        "data/processed/glorys_targets/glorys_targets_2019-01.nc",
    ]

    expected_hashes = {
        "data/raw/sst/sst_ostia_2019-01.nc": "9e733b5956f4d9bba81ec1c7a4bcbfa3b456e61af969d9ae13c57ef6fcc72265",
        "data/raw/sla/sla_duacs_2019-01.nc": "36b558a5642feaff3d8563aea4dac76f70f850d1b555431a8893474b7eccd692",
        "data/raw/sss/sss_multiobs_2019-01.nc": "74dc119b9d34a1b0330e1e9066e067db658dfaa8ff44dedb6edf03041b6ff043",
        "data/raw/currents/currents_multiobs_2019-01.nc": "ad98b7684663b42a2eaa4ced7d5b52ec795fc711d050cdd87cd0b70eaf0ca4a7",
        "data/raw/glorys/glorys_thetao_2019-01.nc": "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f",
        "data/raw/glorys_static/glorys_static_mask_bathy.nc": "fdb0c5fdc282180c0b5b58fa689753df79c22f69f4dc6dac64314a8cb82a79a5",
        "data/processed/glorys_targets/glorys_targets_2019-01.nc": "11cb8d749a7678331df318b6f9115c646cce2d45094f57fcae8c8ad0cfad456d",
    }

    for p in raw_files:
        full_path = os.path.join(PROJECT_ROOT, p)
        assert os.path.exists(full_path), f"File {p} does not exist"
        h = compute_file_sha256(full_path)
        assert h == expected_hashes[p], f"Hash mismatch for {p}: {h} != {expected_hashes[p]}"
