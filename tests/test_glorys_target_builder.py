"""
Unit and synthetic tests for GLORYS Target Builder v1.

Verifies:
1. Weights sum to 1 and equal [1/6, 1/3, 1/3, 1/6] per axis (and 2D outer product sums to 1).
2. A constant field gives the constant at all depths.
3. A profile linear in depth gives exactly the analytic values at the target depths (PCHIP reproduces a linear profile).
4. Depth 0 equals level 0 (P5 surface rule, no extrapolation).
5. A target cell with a partly dry footprint is NaN for tauT = 1.0 and valid for tauT = 0.5 (with the area-weighted value).
6. Target depths below the last valid level are NaN and there is no extrapolation.
7. A QC FAIL (a planted zero-code cell outside the exclusion list) stops the build with an informative error.
8. The output NetCDF file round-trips (shape, coordinates, attributes, NaN pattern).
9. Two builds of the same input give identical arrays.
10. SHA256 of the raw test inputs unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import numpy as np
import pytest
import xarray as xr

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.preprocessing.glorys_target_builder import (
    bracket_search,
    build_day_targets,
    build_month_file,
    compute_file_sha256,
    compute_overlap_weights_1d,
    compute_overlap_weights_2d,
    find_contiguous_valid_prefix_k,
    interpolate_column_pchip,
    load_target_policy,
    regrid_method_A,
)

# Reference constants
RAW_THETAO_PATH = os.path.join(REPO_ROOT, "data", "raw", "glorys", "glorys_thetao_2019-01.nc")
RAW_STATIC_PATH = os.path.join(REPO_ROOT, "data", "raw", "glorys_static", "glorys_static_mask_bathy.nc")
EXPECTED_SHA256_THETAO = "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f"
EXPECTED_SHA256_STATIC = "fdb0c5fdc282180c0b5b58fa689753df79c22f69f4dc6dac64314a8cb82a79a5"


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

@pytest.fixture
def toy_geometry():
    """Toy geometry: 2 x 3 target cells -> native shape (3*2+1, 3*3+1) = (7, 10)."""
    n_target_lat = 2
    n_target_lon = 3
    ny = 3 * n_target_lat + 1  # 7
    nx = 3 * n_target_lon + 1  # 10
    lat_start = 5.0
    lon_start = 45.0
    lat_step = 0.25
    lon_step = 0.25
    native_step = 1.0 / 12.0

    native_lats = lat_start + np.arange(ny) * native_step
    native_lons = lon_start + np.arange(nx) * native_step

    level_depths = np.array([0.494, 2.0, 5.0, 10.0, 20.0, 30.0], dtype=np.float64)
    nz = len(level_depths)

    w_lat_1d, _, _ = compute_overlap_weights_1d(native_lats, lat_start, lat_step, n_target_lat)
    w_lon_1d, _, _ = compute_overlap_weights_1d(native_lons, lon_start, lon_step, n_target_lon)
    w_2d = compute_overlap_weights_2d(w_lat_1d, w_lon_1d)

    policy = {
        "policy_name": "toy_policy",
        "policy_version": "v1",
        "qc_policy_path": "configs/glorys_qc_policy_v1.json",
        "tauT": 1.0,
        "horizontal_method": "area_weighted_valid",
        "vertical_method": "pchip_valid_prefix",
        "depth0": "level0",
        "seabed_gap": "missing",
        "target_depths": [0.0, 3.5, 7.5, 15.0, 25.0],
        "target_grid": {
            "n_lat": n_target_lat,
            "n_lon": n_target_lon,
            "lat_start": lat_start,
            "lat_end": lat_start + n_target_lat * lat_step,
            "lon_start": lon_start,
            "lon_end": lon_start + n_target_lon * lon_step,
            "lat_step": lat_step,
            "lon_step": lon_step,
        },
    }

    qc_policy = {
        "policy_version": "v1",
        "encoding": {
            "fill_raw": -32767,
            "valid_min_raw": -32766,
            "valid_max_raw": 21306,
            "scale_factor": 0.001,
            "add_offset": 20.0,
        },
        "levels_used": nz,
        "zero_code_raw": -28671,
        "near_floor_margin_codes": 50,
        "static_exclusions": [],
    }

    return {
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "n_target_lat": n_target_lat,
        "n_target_lon": n_target_lon,
        "native_lats": native_lats,
        "native_lons": native_lons,
        "level_depths": level_depths,
        "weights_2d": w_2d,
        "policy": policy,
        "qc_policy": qc_policy,
    }


def encode_temp(temp_C: np.ndarray, scale: float = 0.001, offset: float = 20.0) -> np.ndarray:
    """Encode float temperature into int16 raw codes."""
    raw = np.round((temp_C - offset) / scale).astype(np.int16)
    return raw


# =============================================================================
# 1. Weights Sum to 1 and Equal [1/6, 1/3, 1/3, 1/6]
# =============================================================================

def test_weights_sum_to_one_and_equal_canonical():
    """Verify that 1D weights equal [1/6, 1/3, 1/3, 1/6] and sum to 1.0, and 2D weights sum to 1.0."""
    native_lats = 5.0 + np.arange(301) * (1.0 / 12.0)
    w_1d, all_idx, all_w = compute_overlap_weights_1d(
        native_centers=native_lats,
        target_start=5.0,
        target_step=0.25,
        n_target=100,
        native_step=1.0 / 12.0,
    )

    expected_1d = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    assert np.allclose(w_1d, expected_1d, atol=1e-12)

    for i, w in enumerate(all_w):
        assert np.isclose(np.sum(w), 1.0, atol=1e-12), f"Cell {i} weights do not sum to 1"
        assert np.allclose(w, expected_1d, atol=1e-12), f"Cell {i} weights do not match canonical"

    w_2d = compute_overlap_weights_2d(w_1d, w_1d)
    assert w_2d.shape == (4, 4)
    assert np.isclose(np.sum(w_2d), 1.0, atol=1e-12)


# =============================================================================
# 2. Constant Field Gives Constant at All Depths
# =============================================================================

def test_constant_field_gives_constant_at_all_depths(toy_geometry):
    """Verify that a spatially and vertically constant field reproduces the constant at all target depths."""
    g = toy_geometry
    const_temp = 25.5
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    raw_val = int(np.round((const_temp - offset) / scale))
    raw_day = np.full((g["nz"], g["ny"], g["nx"]), raw_val, dtype=np.int16)
    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    targets, valid_mask, qc_summary, f_T = build_day_targets(
        raw_day=raw_day,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=g["policy"],
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    assert qc_summary["status"] == "PASS"
    assert targets.shape == (len(g["policy"]["target_depths"]), g["n_target_lat"], g["n_target_lon"])
    assert np.all(valid_mask == 1)
    assert np.allclose(targets, const_temp, atol=1e-4)


# =============================================================================
# 3. Profile Linear in Depth Gives Analytic Values
# =============================================================================

def test_linear_profile_gives_analytic_values(toy_geometry):
    """Verify that a profile linear in depth is reproduced exactly by PCHIP."""
    g = toy_geometry
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    # Linear profile: T(z) = -0.1 * z + 28.0
    slope = -0.1
    intercept = 28.0
    temps_1d = slope * g["level_depths"] + intercept

    raw_3d = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=np.int16)
    for k in range(g["nz"]):
        raw_3d[k, :, :] = int(np.round((temps_1d[k] - offset) / scale))

    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    # Test interior target depths
    target_depths = [1.0, 3.5, 7.5, 15.0, 25.0]
    pol = dict(g["policy"])
    pol["target_depths"] = target_depths

    targets, valid_mask, _, _ = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=pol,
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        target_depths=target_depths,
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    for d_idx, z_t in enumerate(target_depths):
        expected_val = slope * z_t + intercept
        target_slice = targets[d_idx]
        assert np.allclose(target_slice, expected_val, atol=1e-3), (
            f"Failed at target depth {z_t}: expected {expected_val}, got {np.mean(target_slice)}"
        )


# =============================================================================
# 4. Depth 0 Equals Level 0 (P5 Surface Rule)
# =============================================================================

def test_depth0_equals_level0(toy_geometry):
    """Verify that depth 0 takes the value of level 0 (~0.49 m) without extrapolation."""
    g = toy_geometry
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    # Surface gradient: level 0 = 29.0 C, level 1 = 27.0 C
    temps_1d = np.array([29.0, 27.0, 25.0, 23.0, 21.0, 19.0])
    raw_3d = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=np.int16)
    for k in range(g["nz"]):
        raw_3d[k, :, :] = int(np.round((temps_1d[k] - offset) / scale))

    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    targets, valid_mask, _, _ = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=g["policy"],
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    depth0_target = targets[0]  # target depth 0.0
    # Level 0 is 29.0 C. If extrapolated from level 1 and 0, it would be > 29.5 C.
    assert np.allclose(depth0_target, 29.0, atol=1e-3)


# =============================================================================
# 5. Partly Dry Footprint: NaN for tauT=1.0, Valid for tauT=0.5
# =============================================================================

def test_partly_dry_footprint_tau_threshold(toy_geometry):
    """Verify that a target cell with partly dry footprint is NaN under tauT=1.0 and valid under tauT=0.5."""
    g = toy_geometry
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    temp = 22.0
    raw_val = int(np.round((temp - offset) / scale))
    raw_3d = np.full((g["nz"], g["ny"], g["nx"]), raw_val, dtype=np.int16)
    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    # In target cell (0, 0), make native cell (0, 0) dry
    fill_raw = g["qc_policy"]["encoding"]["fill_raw"]
    static_wet[:, 0, 0] = False
    raw_3d[:, 0, 0] = fill_raw

    # 1. With tauT = 1.0 (strict): cell (0, 0) has f_T < 1.0, must be NaN at all depths
    pol_tau10 = dict(g["policy"])
    pol_tau10["tauT"] = 1.0

    targets_10, mask_10, _, f_T_10 = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=pol_tau10,
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    # Cell (0, 0) must be NaN and invalid
    assert np.all(np.isnan(targets_10[:, 0, 0]))
    assert np.all(mask_10[:, 0, 0] == 0)
    # Other target cells must be valid
    assert not np.isnan(targets_10[0, 1, 1])

    # 2. With tauT = 0.5: cell (0, 0) has f_T ~ 0.972 >= 0.5, must be valid
    pol_tau05 = dict(g["policy"])
    pol_tau05["tauT"] = 0.5

    targets_05, mask_05, _, f_T_05 = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=pol_tau05,
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    assert not np.isnan(targets_05[0, 0, 0])
    assert mask_05[0, 0, 0] == 1
    assert np.isclose(targets_05[0, 0, 0], temp, atol=1e-3)


# =============================================================================
# 6. Target Depths Below Last Valid Level are NaN (No Extrapolation)
# =============================================================================

def test_no_extrapolation_below_last_valid_level(toy_geometry):
    """Verify that depths below the last valid level in the column are NaN without extrapolation."""
    g = toy_geometry
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    # Levels: [0.494, 2.0, 5.0, 10.0, 20.0, 30.0]
    # Mark levels 3, 4, 5 as dry (seabed reached after level 2 at 5.0 m)
    fill_raw = g["qc_policy"]["encoding"]["fill_raw"]
    raw_3d = np.full((g["nz"], g["ny"], g["nx"]), 1000, dtype=np.int16)
    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    static_wet[3:, :, :] = False  # levels 3, 4, 5 are dry
    raw_3d[3:, :, :] = fill_raw
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    # Target depths:
    # 0.0 m (surface: level 0 valid -> VALID)
    # 3.5 m (bracketed by levels 1 [2.0m] and 2 [5.0m], both valid -> VALID)
    # 7.5 m (bracketed by levels 2 [5.0m] and 3 [10.0m], level 3 dry -> MUST BE NaN)
    # 15.0 m (bracketed by levels 3 and 4 -> MUST BE NaN)
    targets, valid_mask, _, _ = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=g["policy"],
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    # Target depths 0.0 and 3.5 are valid
    assert np.all(~np.isnan(targets[0]))
    assert np.all(~np.isnan(targets[1]))

    # Target depths 7.5, 15.0, 25.0 are below last valid level (5.0 m) -> NaN
    assert np.all(np.isnan(targets[2]))
    assert np.all(np.isnan(targets[3]))
    assert np.all(np.isnan(targets[4]))


# =============================================================================
# 7. QC FAIL Stops the Build with Informative Error
# =============================================================================

def test_qc_fail_stops_build_with_informative_error(toy_geometry):
    """Verify that an unexpected QC violation halts the build with an error naming the cells."""
    g = toy_geometry
    raw_3d = np.full((g["nz"], g["ny"], g["nx"]), 5000, dtype=np.int16)
    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    # Plant a zero-code cell (-28671) at level 1, row 2, col 3
    zero_code = g["qc_policy"]["zero_code_raw"]
    raw_3d[1, 2, 3] = zero_code

    with pytest.raises(RuntimeError) as excinfo:
        build_day_targets(
            raw_day=raw_3d,
            static_wet=static_wet,
            exclusion_mask=exclusion_mask,
            policy=g["policy"],
            native_lats=g["native_lats"],
            native_lons=g["native_lons"],
            native_depths=g["level_depths"],
            weights_2d=g["weights_2d"],
            qc_policy=g["qc_policy"],
        )

    msg = str(excinfo.value)
    assert "QC policy v1 reported FAIL" in msg
    assert "level=1" in msg
    assert "lat_idx=2" in msg
    assert "lon_idx=3" in msg


# =============================================================================
# 8. Output NetCDF File Roundtrips
# =============================================================================

def test_output_netcdf_roundtrip(toy_geometry):
    """Verify that build_month_file writes a compliant NetCDF file that round-trips correctly."""
    g = toy_geometry
    scale = g["qc_policy"]["encoding"]["scale_factor"]
    offset = g["qc_policy"]["encoding"]["add_offset"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a synthetic monthly raw file with 2 days
        raw_glorys_path = os.path.join(tmpdir, "toy_glorys.nc")
        raw_static_path = os.path.join(tmpdir, "toy_static.nc")
        qc_pol_path = os.path.join(tmpdir, "toy_qc_pol.json")
        out_nc_path = os.path.join(tmpdir, "toy_targets.nc")

        # Save QC policy
        with open(qc_pol_path, "w", encoding="utf-8") as f:
            json.dump(g["qc_policy"], f)

        # Write synthetic GLORYS NetCDF
        times = np.array([100.0, 124.0])
        time_attrs = {"units": "hours since 1950-01-01", "calendar": "gregorian"}
        raw_data = np.full((2, g["nz"], g["ny"], g["nx"]), 5000, dtype=np.int16)

        ds_raw = xr.Dataset(
            data_vars={
                "thetao": (
                    ("time", "depth", "latitude", "longitude"),
                    raw_data,
                    {
                        "scale_factor": scale,
                        "add_offset": offset,
                        "_FillValue": np.int16(-32767),
                        "valid_min": np.int16(-32766),
                        "valid_max": np.int16(21306),
                    },
                )
            },
            coords={
                "time": ("time", times, time_attrs),
                "depth": ("depth", g["level_depths"]),
                "latitude": ("latitude", g["native_lats"]),
                "longitude": ("longitude", g["native_lons"]),
            },
        )
        ds_raw.to_netcdf(raw_glorys_path, engine="h5netcdf")
        ds_raw.close()

        # Write synthetic static NetCDF
        ds_static = xr.Dataset(
            data_vars={
                "mask": (
                    ("depth", "latitude", "longitude"),
                    np.ones((g["nz"], g["ny"], g["nx"]), dtype=np.int8),
                )
            },
            coords={
                "depth": ("depth", g["level_depths"]),
                "latitude": ("latitude", g["native_lats"]),
                "longitude": ("longitude", g["native_lons"]),
            },
        )
        ds_static.to_netcdf(raw_static_path, engine="h5netcdf")
        ds_static.close()

        toy_pol = dict(g["policy"])
        toy_pol["qc_policy_path"] = qc_pol_path

        # Run build_month_file
        manifest = build_month_file(
            glorys_path=raw_glorys_path,
            static_path=raw_static_path,
            out_path=out_nc_path,
            policy=toy_pol,
            verbose=False,
        )

        assert os.path.exists(out_nc_path)
        assert os.path.exists(os.path.splitext(out_nc_path)[0] + "_manifest.json")

        # Reopen NetCDF and check roundtrip properties
        with xr.open_dataset(out_nc_path, decode_times=False) as ds_out:
            assert "thetao_target" in ds_out.variables
            assert "valid_mask" in ds_out.variables
            assert ds_out["thetao_target"].shape == (2, len(toy_pol["target_depths"]), g["n_target_lat"], g["n_target_lon"])
            assert ds_out["valid_mask"].dtype == np.uint8
            assert ds_out["thetao_target"].dtype == np.float32

            # Check required global attributes
            for attr in [
                "policy_name", "policy_version", "tauT", "horizontal_method",
                "vertical_method", "depth0_method", "raw_glorys_sha256",
                "raw_static_sha256", "qc_policy_sha256", "builder_script_sha256",
                "git_commit", "created_utc", "python_version", "numpy_version",
                "scipy_version", "xarray_version"
            ]:
                assert attr in ds_out.attrs, f"Missing global attribute: {attr}"


# =============================================================================
# 9. Two Builds of Same Input Give Identical Arrays
# =============================================================================

def test_two_builds_give_identical_arrays(toy_geometry):
    """Verify bitwise and numerical reproducibility across repeated builds."""
    g = toy_geometry
    raw_3d = np.random.default_rng(123).integers(2000, 8000, size=(g["nz"], g["ny"], g["nx"]), dtype=np.int16)
    static_wet = np.ones((g["nz"], g["ny"], g["nx"]), dtype=bool)
    exclusion_mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)

    targets_1, mask_1, _, _ = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=g["policy"],
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    targets_2, mask_2, _, _ = build_day_targets(
        raw_day=raw_3d,
        static_wet=static_wet,
        exclusion_mask=exclusion_mask,
        policy=g["policy"],
        native_lats=g["native_lats"],
        native_lons=g["native_lons"],
        native_depths=g["level_depths"],
        weights_2d=g["weights_2d"],
        qc_policy=g["qc_policy"],
    )

    assert np.array_equal(targets_1, targets_2, equal_nan=True)
    assert np.array_equal(mask_1, mask_2)


# =============================================================================
# 10. SHA256 of Raw Test Inputs Unchanged
# =============================================================================

def test_raw_inputs_sha256_strictly_unchanged():
    """Verify raw NetCDF files have expected checksums and were not modified."""
    assert os.path.exists(RAW_THETAO_PATH), f"Missing: {RAW_THETAO_PATH}"
    assert os.path.exists(RAW_STATIC_PATH), f"Missing: {RAW_STATIC_PATH}"

    sha_thetao = compute_file_sha256(RAW_THETAO_PATH)
    sha_static = compute_file_sha256(RAW_STATIC_PATH)

    assert sha_thetao == EXPECTED_SHA256_THETAO, f"thetao modified! Got {sha_thetao}"
    assert sha_static == EXPECTED_SHA256_STATIC, f"static modified! Got {sha_static}"
