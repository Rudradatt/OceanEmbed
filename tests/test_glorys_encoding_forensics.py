"""
Tests for GLORYS thetao Forensic Encoding Analysis.

Validates the forensic diagnostics on small synthetic NetCDF datasets
before running on real data.
"""

import os
import tempfile
import numpy as np
import pytest
import h5netcdf

from scripts.diagnostics.glorys_encoding_forensics import (
    compute_file_metadata,
    compute_encoding_arithmetic,
    lookup_nearest_index,
    run_raw_code_census,
    check_mask_time_invariance,
    RawNetCDFReader,
    generate_header_dump,
)


@pytest.fixture
def synthetic_netcdf(tmp_path):
    """Create a small synthetic NetCDF file with known parameters.

    Dimensions:
      time: 2 (days)
      depth: 3 levels (50.0 m, 318.1 m, 454.0 m)
      latitude: 4 points (14.9, 15.0, 15.1, 15.2)
      longitude: 4 points (50.5, 50.583332, 50.666668, 50.8)

    Attributes:
      scale_factor = 0.0007324442267417908
      add_offset = 21.0
      _FillValue = -32767
      valid_min = -32766
      valid_max = 21306
    """
    nc_path = tmp_path / "synthetic_glorys.nc"

    scale_factor = 0.0007324442267417908
    add_offset = 21.0
    fill_value = -32767

    times = np.array([0, 1], dtype=np.int32)
    depths = np.array([50.0, 318.1274, 453.9377], dtype=np.float32)
    lats = np.array([14.9, 15.0, 15.1, 15.2], dtype=np.float32)
    lons = np.array([50.5, 50.583332, 50.666668, 50.8], dtype=np.float32)

    # Initialize data with non-fill constant code 1000
    # Shape: (2, 3, 4, 4) = 96 cells
    data = np.full((2, 3, 4, 4), 1000, dtype=np.int16)

    # Level 0: 6 fill cells on day 0 and day 1 (time-invariant)
    data[:, 0, 0, :3] = fill_value  # 3 cells per day * 2 days = 6 fill cells

    # Level 1 (approx 318m):
    # Set known cell whose raw code equals packed code of 0.0 (-28671)
    # Put it at day 0 and day 1 at col (lat index 1, lon index 1 -> 15.0, 50.583332)
    data[0, 1, 1, 1] = -28671
    data[1, 1, 1, 1] = -28671

    # Put a code -28670 at day 0, level 1, col (1, 2)
    data[0, 1, 1, 2] = -28670
    data[1, 1, 1, 2] = -28670

    # Put 4 fill cells at level 1
    data[:, 1, 3, :2] = fill_value

    # Level 2 (approx 454m):
    # Set -28671 at day 0 and 1, col (1, 2 -> 15.0, 50.666668)
    data[:, 2, 1, 2] = -28671
    # 8 fill cells at level 2
    data[:, 2, 2:, :] = fill_value

    with h5netcdf.File(str(nc_path), "w") as f:
        f.attrs["title"] = "Synthetic Test NetCDF"
        f.attrs["Conventions"] = "CF-1.4"
        f.attrs["source"] = "Synthetic Test Suite"
        f.dimensions = {
            "time": len(times),
            "depth": len(depths),
            "latitude": len(lats),
            "longitude": len(lons),
        }

        v_time = f.create_variable("time", ("time",), data=times)
        v_depth = f.create_variable("depth", ("depth",), data=depths)
        v_lat = f.create_variable("latitude", ("latitude",), data=lats)
        v_lon = f.create_variable("longitude", ("longitude",), data=lons)

        v_thetao = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=data)
        v_thetao.attrs["scale_factor"] = scale_factor
        v_thetao.attrs["add_offset"] = add_offset
        v_thetao.attrs["_FillValue"] = fill_value
        v_thetao.attrs["valid_min"] = np.int16(-32766)
        v_thetao.attrs["valid_max"] = np.int16(21306)

    return str(nc_path)


def test_manual_decode_equals_expected():
    """Verify manual decode logic (raw * scale + offset) matches expected float64 values."""
    scale = 0.002
    offset = 10.0
    raw_code = -5000
    expected = -5000 * 0.002 + 10.0  # -10.0 + 10.0 = 0.0
    decoded = float(np.float64(raw_code) * np.float64(scale) + np.float64(offset))
    assert np.isclose(decoded, expected)
    assert np.isclose(decoded, 0.0)

    # GLORYS parameters test
    glorys_scale = 0.0007324442267417908
    glorys_offset = 21.0
    dec_neg28671 = float(np.float64(-28671) * np.float64(glorys_scale) + np.float64(glorys_offset))
    assert np.isclose(dec_neg28671, 9.1575086e-05, atol=1e-10)


def test_zero_code_computation():
    """Verify raw code computation for float 0.0 C."""
    scale = 0.0007324442267417908
    offset = 21.0
    arith = compute_encoding_arithmetic(
        scale_factor=scale,
        add_offset=offset,
        fill_value=-32767,
        valid_min=-32766,
        valid_max=21306,
        valid_range=None,
    )
    assert arith["rounded_code_equals_neg28671"] is True
    assert arith["rounded_zero_code_int"] == -28671
    assert np.isclose(arith["exact_zero_code_float"], -28671.125026702066)
    assert arith["valid_range_checks"]["neg28671_inside_raw_valid_range"] is True
    assert arith["valid_range_checks"]["neg28671_inside_decoded_valid_range"] is True


def test_raw_code_census_counts(synthetic_netcdf):
    """Verify that raw-code census counts fill, finite, -28671, and near-zero cells correctly."""
    with RawNetCDFReader(synthetic_netcdf) as reader:
        attrs = reader.get_var_attrs("thetao")
        census_df, summary = run_raw_code_census(
            reader=reader,
            fill_value=int(attrs["_FillValue"]),
            scale_factor=float(attrs["scale_factor"]),
            add_offset=float(attrs["add_offset"]),
        )

    # Total cells: 2 * 3 * 4 * 4 = 96
    assert summary["total_cells_evaluated"] == 96

    # Level 0 has 6 fill cells (3 per day * 2 days), 26 finite cells
    assert census_df.loc[census_df["level_index"] == 0, "fill_count"].iloc[0] == 6
    assert census_df.loc[census_df["level_index"] == 0, "finite_count"].iloc[0] == 26
    assert census_df.loc[census_df["level_index"] == 0, "code_neg28671_count"].iloc[0] == 0

    # Level 1 has 2 instances of -28671 (day 0 and day 1)
    assert census_df.loc[census_df["level_index"] == 1, "code_neg28671_count"].iloc[0] == 2
    # Level 2 has 2 instances of -28671 (day 0 and day 1)
    assert census_df.loc[census_df["level_index"] == 2, "code_neg28671_count"].iloc[0] == 2

    # Total -28671 cells: 4
    assert summary["total_neg28671_cells"] == 4
    assert summary["neg28671_unique_level_indices"] == [1, 2]
    # Unique columns with -28671: (15.0, 50.583332) and (15.0, 50.666668)
    assert len(summary["neg28671_unique_columns"]) == 2

    # Near zero (|x| < 1e-3 C):
    # -28671 decodes to ~9.16e-5 C (< 1e-3 C) -> 4 cells
    # -28670 decodes to ~8.24e-4 C (< 1e-3 C) -> 2 cells
    # Total near zero = 6 cells
    assert summary["total_near_zero_cells"] == 6


def test_mask_time_invariance_detector(synthetic_netcdf, tmp_path):
    """Verify mask time-invariance detector passes on invariant data and detects altered cells."""
    with RawNetCDFReader(synthetic_netcdf) as reader:
        inv_result = check_mask_time_invariance(reader, fill_value=-32767)
    assert inv_result["is_time_invariant"] is True
    assert inv_result["total_differing_points"] == 0

    # Create mutated file where day 1 mask differs from day 0
    mutated_nc = tmp_path / "mutated_mask.nc"
    # Copy data from synthetic_netcdf
    with h5netcdf.File(synthetic_netcdf, "r") as src, h5netcdf.File(str(mutated_nc), "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for dim, sz in src.dimensions.items():
            dst.dimensions[dim] = len(sz)
        for var in ["time", "depth", "latitude", "longitude"]:
            dst.create_variable(var, (var,), data=src[var][:])
        tdata = src["thetao"][:].copy()
        # Mutate single cell: on day 1, set finite cell (1, 0, 1, 1) to fill
        tdata[1, 0, 1, 1] = -32767
        vt = dst.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=tdata)
        for ak, av in src["thetao"].attrs.items():
            vt.attrs[ak] = av

    with RawNetCDFReader(str(mutated_nc)) as reader:
        mutated_result = check_mask_time_invariance(reader, fill_value=-32767)
    assert mutated_result["is_time_invariant"] is False
    assert mutated_result["total_differing_points"] == 1
    assert mutated_result["per_day_diff_counts"][1] == 1


def test_file_sha256_unchanged_after_analysis(synthetic_netcdf, tmp_path):
    """Verify that computing metadata and running read-only operations leaves SHA256 unchanged."""
    meta_before = compute_file_metadata(synthetic_netcdf)
    out_dump = tmp_path / "dump.txt"

    with RawNetCDFReader(synthetic_netcdf) as reader:
        generate_header_dump(reader, synthetic_netcdf, str(out_dump))
        _ = run_raw_code_census(reader, fill_value=-32767, scale_factor=0.00073, add_offset=21.0)
        _ = check_mask_time_invariance(reader, fill_value=-32767)

    meta_after = compute_file_metadata(synthetic_netcdf)

    assert meta_before["sha256"] == meta_after["sha256"]
    assert meta_before["size_bytes"] == meta_after["size_bytes"]
    assert meta_before["mtime_utc"] == meta_after["mtime_utc"]


def test_nearest_lookup_fail_loudly_outside_tolerance():
    """Verify nearest coordinate/level lookup succeeds within tolerance and fails loudly outside."""
    lats = np.array([10.0, 15.0, 20.0])

    # Within tolerance
    idx, val = lookup_nearest_index(lats, 15.0001, "latitude", tolerance=1e-3)
    assert idx == 1
    assert np.isclose(val, 15.0)

    # Outside tolerance -> raises ValueError
    with pytest.raises(ValueError, match="exceeds tolerance"):
        lookup_nearest_index(lats, 15.05, "latitude", tolerance=1e-3)

    # Depth lookup outside tolerance
    depths = np.array([50.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="exceeds tolerance"):
        lookup_nearest_index(depths, 318.0, "depth", tolerance=5.0)
