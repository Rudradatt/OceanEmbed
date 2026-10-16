"""
Tests for GLORYS Quality-Control (QC) Policy Module v1.

Verifies:
1. Each flag bit is set exactly where expected, and fill is never flagged NEAR_FLOOR.
2. A zero-code cell INSIDE the exclusion list gives PASS, and OUTSIDE gives FAIL with cell listed.
3. A near-floor cell outside the exclusion list gives FAIL.
4. A cell above valid_max gives WARN and stays valid.
5. A finite cell in a static-dry location gives FAIL.
6. Static-wet-but-fill cells are not valid and are counted.
7. A mask differing from the reference gives FAIL.
8. Exclusion lookup fails loudly if coordinates are not found within tolerance.
9. A policy/file encoding mismatch fails loudly.
10. Inputs are not mutated, and SHA256 of test files is unchanged.
"""

import hashlib
import json
import os
import sys
import tempfile
import h5netcdf
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.preprocessing.glorys_qc import (
    FLAG_FILL_IN_STATIC_WET,
    FLAG_STATIC_EXCLUDED,
    FLAG_ZERO_CODE,
    FLAG_NEAR_FLOOR,
    FLAG_ABOVE_VALID_MAX,
    FLAG_FINITE_IN_STATIC_DRY,
    FLAG_MASK_DIFFERS_FROM_REF,
    load_policy,
    read_raw_day,
    load_coordinates,
    load_static_wet_mask,
    locate_exclusions,
    apply_qc,
)


# =============================================================================
# Synthetic Fixture
# =============================================================================

@pytest.fixture
def base_policy():
    """Return a baseline policy dictionary for testing."""
    return {
        "policy_version": "v1",
        "encoding": {
            "fill_raw": -32767,
            "valid_min_raw": -32766,
            "valid_max_raw": 21306,
            "scale_factor": 0.0007324442267417908,
            "add_offset": 21.0,
        },
        "levels_used": 3,
        "zero_code_raw": -28671,
        "near_floor_margin_codes": 50,
        "coordinate_tolerance_deg": 0.001,
        "static_exclusions": [
            {
                "name": "Col_A",
                "lat": 15.0,
                "lon": 50.58333,
                "levels": [1, 2],
                "reason": "Synthetic test exclusion A",
            },
            {
                "name": "Col_B",
                "lat": 15.0,
                "lon": 50.66667,
                "levels": [1, 2],
                "reason": "Synthetic test exclusion B",
            },
        ],
    }


@pytest.fixture
def synthetic_files(tmp_path, base_policy):
    """Create minimal synthetic NetCDF files for I/O and integrity tests."""
    lats = np.array([14.9, 15.0, 15.1], dtype=np.float64)  # ny=3
    lons = np.array([50.5, 50.583332, 50.666668], dtype=np.float64)  # nx=3
    depths = np.array([0.5, 10.0, 50.0], dtype=np.float64)  # nz=3

    raw_file = tmp_path / "glorys_thetao_test.nc"
    static_file = tmp_path / "glorys_static_test.nc"
    policy_file = tmp_path / "glorys_qc_policy_test.json"

    with open(policy_file, "w") as fp:
        json.dump(base_policy, fp)

    # 1. Raw thetao file
    with h5netcdf.File(raw_file, "w") as f:
        f.dimensions = {"time": 2, "depth": 3, "latitude": 3, "longitude": 3}
        f.create_variable("time", ("time",), data=np.array([0.0, 24.0], dtype=np.float32))
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)

        raw_data = np.full((2, 3, 3, 3), base_policy["encoding"]["fill_raw"], dtype=np.int16)
        raw_data[:, :, 1:, 1:] = 10000  # finite ocean
        v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=raw_data)
        v.attrs["scale_factor"] = base_policy["encoding"]["scale_factor"]
        v.attrs["add_offset"] = base_policy["encoding"]["add_offset"]
        v.attrs["_FillValue"] = base_policy["encoding"]["fill_raw"]
        v.attrs["valid_min"] = base_policy["encoding"]["valid_min_raw"]
        v.attrs["valid_max"] = base_policy["encoding"]["valid_max_raw"]

    # 2. Static mask file
    with h5netcdf.File(static_file, "w") as f:
        f.dimensions = {"depth": 3, "latitude": 3, "longitude": 3}
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)

        mask_data = np.zeros((3, 3, 3), dtype=np.int8)
        mask_data[:, 1:, 1:] = 1  # ocean
        f.create_variable("mask", ("depth", "latitude", "longitude"), data=mask_data)

    return {
        "raw_file": str(raw_file),
        "static_file": str(static_file),
        "policy_file": str(policy_file),
        "lats": lats,
        "lons": lons,
        "depths": depths,
    }


# =============================================================================
# Test 1: Flag Bits & Fill Never Flagged Near Floor
# =============================================================================

def test_flag_bits_exact_and_fill_not_near_floor(base_policy):
    """Verify that each flag bit is set exactly where expected, and fill is never near floor."""
    nz, ny, nx = 3, 3, 3
    fill = base_policy["encoding"]["fill_raw"]      # -32767
    vmin = base_policy["encoding"]["valid_min_raw"] # -32766
    vmax = base_policy["encoding"]["valid_max_raw"] # 21306

    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # 1. Fill in static wet: (0, 0, 0)
    raw[0, 0, 0] = fill
    static_wet[0, 0, 0] = True

    # 2. Static excluded: (0, 0, 1)
    exclusion_mask[0, 0, 1] = True

    # 3. Zero code: (0, 0, 2)
    raw[0, 0, 2] = -28671

    # 4. Near floor: (0, 1, 0)
    raw[0, 1, 0] = vmin + 10  # -32756

    # 5. Above valid max: (0, 1, 1)
    raw[0, 1, 1] = vmax + 50  # 21356

    # 6. Finite in static dry: (0, 1, 2)
    raw[0, 1, 2] = 12000
    static_wet[0, 1, 2] = False

    # 7. Mask differs from reference: (0, 2, 0)
    ref_finite = np.ones((nz, ny, nx), dtype=bool)
    raw[0, 2, 0] = fill  # curr is fill (False), ref is finite (True)

    # CRITICAL: Verify cell (0, 2, 1) holds fill value -32767 and is NOT flagged NEAR_FLOOR
    raw[0, 2, 1] = fill

    res = apply_qc(raw, static_wet, exclusion_mask, base_policy, reference_finite=ref_finite)

    flags = res.flags
    # Check bit 0: FILL_IN_STATIC_WET (1)
    assert bool(flags[0, 0, 0] & FLAG_FILL_IN_STATIC_WET) is True

    # Check bit 1: STATIC_EXCLUDED (2)
    assert bool(flags[0, 0, 1] & FLAG_STATIC_EXCLUDED) is True

    # Check bit 2: ZERO_CODE (4)
    assert bool(flags[0, 0, 2] & FLAG_ZERO_CODE) is True

    # Check bit 3: NEAR_FLOOR (8)
    assert bool(flags[0, 1, 0] & FLAG_NEAR_FLOOR) is True

    # Check bit 4: ABOVE_VALID_MAX (16)
    assert bool(flags[0, 1, 1] & FLAG_ABOVE_VALID_MAX) is True

    # Check bit 5: FINITE_IN_STATIC_DRY (32)
    assert bool(flags[0, 1, 2] & FLAG_FINITE_IN_STATIC_DRY) is True

    # Check bit 6: MASK_DIFFERS_FROM_REF (64)
    assert bool(flags[0, 2, 0] & FLAG_MASK_DIFFERS_FROM_REF) is True

    # VERIFY: fill value (-32767) is NEVER flagged NEAR_FLOOR
    assert bool(flags[0, 2, 1] & FLAG_NEAR_FLOOR) is False
    assert bool(flags[0, 0, 0] & FLAG_NEAR_FLOOR) is False


# =============================================================================
# Test 2: Zero Code Inside vs Outside Exclusion
# =============================================================================

def test_zero_code_inside_vs_outside_exclusion(base_policy):
    """Verify zero code INSIDE exclusion gives PASS, while OUTSIDE gives FAIL."""
    nz, ny, nx = 3, 3, 3
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # Place zero code at (1, 1, 1) and mark it EXCLUDED
    raw[1, 1, 1] = -28671
    exclusion_mask[1, 1, 1] = True

    res_inside = apply_qc(raw, static_wet, exclusion_mask, base_policy)
    assert res_inside.status == "PASS"
    assert len(res_inside.unexpected) == 0
    assert res_inside.counts["zero_code"] == 1
    assert res_inside.counts["zero_code_unexcluded"] == 0

    # Place another zero code at (1, 2, 2) which is OUTSIDE exclusion
    raw[1, 2, 2] = -28671
    res_outside = apply_qc(raw, static_wet, exclusion_mask, base_policy)
    assert res_outside.status == "FAIL"
    assert len(res_outside.unexpected) == 1
    assert res_outside.unexpected[0]["level"] == 1
    assert res_outside.unexpected[0]["lat_idx"] == 2
    assert res_outside.unexpected[0]["lon_idx"] == 2
    assert res_outside.unexpected[0]["raw"] == -28671
    assert res_outside.unexpected[0]["rule"] == "ZERO_CODE_OUTSIDE_EXCLUSION"


# =============================================================================
# Test 3: Near-Floor Cell Outside Exclusion Fails
# =============================================================================

def test_near_floor_outside_exclusion_fails(base_policy):
    """Verify near-floor cell outside exclusion list causes FAIL."""
    nz, ny, nx = 3, 3, 3
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # Near floor cell at (2, 0, 1) with raw = -32750
    raw[2, 0, 1] = -32750
    res = apply_qc(raw, static_wet, exclusion_mask, base_policy)

    assert res.status == "FAIL"
    assert len(res.unexpected) == 1
    assert res.unexpected[0]["level"] == 2
    assert res.unexpected[0]["raw"] == -32750
    assert res.unexpected[0]["rule"] == "NEAR_FLOOR_OUTSIDE_EXCLUSION"


# =============================================================================
# Test 4: Cell Above Valid Max Gives WARN and Stays Valid
# =============================================================================

def test_above_valid_max_warns_and_stays_valid(base_policy):
    """Verify raw > 21306 gives WARN (not FAIL), unexpected is empty, and cell is valid."""
    nz, ny, nx = 3, 3, 3
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # Set hot cell at (1, 1, 1) with raw = 22024
    raw[1, 1, 1] = 22024

    res = apply_qc(raw, static_wet, exclusion_mask, base_policy)

    assert res.status == "WARN"
    assert len(res.unexpected) == 0
    assert res.counts["above_valid_max"] == 1
    # Cell MUST stay valid!
    assert res.valid[1, 1, 1] is True or res.valid[1, 1, 1] == 1


# =============================================================================
# Test 5: Finite in Static Dry Gives FAIL
# =============================================================================

def test_finite_in_static_dry_fails(base_policy):
    """Verify finite value on a land/dry cell gives FAIL."""
    nz, ny, nx = 3, 3, 3
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # Mark (0, 1, 1) dry while holding finite value 10000
    static_wet[0, 1, 1] = False

    res = apply_qc(raw, static_wet, exclusion_mask, base_policy)

    assert res.status == "FAIL"
    assert len(res.unexpected) == 1
    assert res.unexpected[0]["level"] == 0
    assert res.unexpected[0]["lat_idx"] == 1
    assert res.unexpected[0]["lon_idx"] == 1
    assert res.unexpected[0]["rule"] == "FINITE_IN_STATIC_DRY"
    # Dry cell must NOT be valid
    assert res.valid[0, 1, 1] == False


# =============================================================================
# Test 6: Static Wet But Fill Not Valid and Counted
# =============================================================================

def test_static_wet_fill_not_valid_and_counted(base_policy):
    """Verify static-wet fill cells are flagged, counted, and marked invalid."""
    nz, ny, nx = 3, 3, 3
    fill = base_policy["encoding"]["fill_raw"]
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    # Make 5 cells fill in static wet
    raw[0, 0, 0] = fill
    raw[0, 0, 1] = fill
    raw[1, 2, 2] = fill
    raw[2, 1, 0] = fill
    raw[2, 1, 1] = fill

    res = apply_qc(raw, static_wet, exclusion_mask, base_policy)

    assert res.counts["fill_in_static_wet"] == 5
    assert res.valid[0, 0, 0] == False
    assert res.valid[0, 0, 1] == False
    assert res.valid[1, 2, 2] == False
    assert res.valid[2, 1, 0] == False
    assert res.valid[2, 1, 1] == False


# =============================================================================
# Test 7: Mask Differing From Reference Fails
# =============================================================================

def test_mask_differing_from_reference_fails(base_policy):
    """Verify discrepancy against reference finite mask gives FAIL."""
    nz, ny, nx = 3, 3, 3
    fill = base_policy["encoding"]["fill_raw"]
    raw = np.full((nz, ny, nx), 10000, dtype=np.int16)
    static_wet = np.ones((nz, ny, nx), dtype=bool)
    exclusion_mask = np.zeros((nz, ny, nx), dtype=bool)

    ref_finite = np.full((nz, ny, nx), True, dtype=bool)
    # Cell (1, 0, 0) was finite in reference, but is fill in current
    raw[1, 0, 0] = fill

    res = apply_qc(raw, static_wet, exclusion_mask, base_policy, reference_finite=ref_finite)

    assert res.status == "FAIL"
    assert res.counts["mask_differs_from_reference"] == 1
    assert any(u["rule"] == "MASK_DIFFERS_FROM_REFERENCE" for u in res.unexpected)


# =============================================================================
# Test 8: Exclusion Lookup Fails Loudly on Missing Coordinates
# =============================================================================

def test_exclusion_lookup_fails_on_missing_coords(base_policy):
    """Verify locate_exclusions raises ValueError if exclusion coordinate is not found within tolerance."""
    lats = np.array([20.0, 21.0, 22.0], dtype=np.float64)  # 15.0 is far away!
    lons = np.array([50.0, 51.0, 52.0], dtype=np.float64)

    with pytest.raises(ValueError, match="not found within tolerance"):
        locate_exclusions(lats, lons, base_policy)


# =============================================================================
# Test 9: Policy / File Encoding Mismatch Fails Loudly
# =============================================================================

def test_file_encoding_mismatch_fails_loudly(synthetic_files, base_policy):
    """Verify read_raw_day raises ValueError if NetCDF encoding differs from policy."""
    raw_file = synthetic_files["raw_file"]

    # Alter policy valid_max_raw to trigger mismatch
    tampered_policy = dict(base_policy)
    tampered_policy["encoding"] = dict(base_policy["encoding"])
    tampered_policy["encoding"]["valid_max_raw"] = 99999

    with pytest.raises(ValueError, match="File encoding verification failed"):
        read_raw_day(raw_file, day_index=0, policy=tampered_policy)


# =============================================================================
# Test 10: Inputs Not Mutated & File SHA256 Unchanged
# =============================================================================

def test_inputs_not_mutated_and_file_sha256_unchanged(synthetic_files, base_policy):
    """Verify apply_qc does not mutate arrays in-place, and NetCDF file SHA256 is unchanged."""
    raw_file = synthetic_files["raw_file"]
    static_file = synthetic_files["static_file"]

    # Compute SHA256 before
    def _hash_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    sha_raw_before = _hash_file(raw_file)
    sha_stat_before = _hash_file(static_file)

    lats, lons, depths = load_coordinates(raw_file)
    raw = read_raw_day(raw_file, day_index=0, policy=base_policy)
    static_wet = load_static_wet_mask(static_file, ref_lats=lats, ref_lons=lons, ref_depths=depths, levels_used=3)
    exclusion_mask = locate_exclusions(lats, lons, base_policy)

    # Save exact copies of input arrays to verify no mutation
    raw_copy = raw.copy()
    static_wet_copy = static_wet.copy()
    exclusion_copy = exclusion_mask.copy()

    # Apply QC
    res = apply_qc(raw, static_wet, exclusion_mask, base_policy, lats=lats, lons=lons, depths=depths)

    # Verify input arrays were NOT mutated in-place
    assert np.array_equal(raw, raw_copy)
    assert np.array_equal(static_wet, static_wet_copy)
    assert np.array_equal(exclusion_mask, exclusion_copy)

    # Verify files on disk were NOT modified
    sha_raw_after = _hash_file(raw_file)
    sha_stat_after = _hash_file(static_file)

    assert sha_raw_before == sha_raw_after
    assert sha_stat_before == sha_stat_after
