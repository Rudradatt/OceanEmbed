"""
Tests for GLORYS Hot-End Audit and Ceiling-Count Reconciliation Diagnostic.

Verifies:
1. Cells above valid_max are counted correctly and fill is never counted.
2. The independent maximum-raw function is correct and ignores fill.
3. The ceiling rule is applied with explicit, tested comparison operators.
4. SHA256 remains 100% unchanged after analysis.
5. Boundary handling for 3x3 neighbourhood on edge/corner cells.
6. Multi-year tracking and percentile rank computations on synthetic distributions.
"""

import hashlib
import os
import sys
import tempfile
import h5netcdf
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.diagnostics.glorys_hot_end_audit import (
    compute_file_metadata,
    verify_integrity_unchanged,
    decode_thetao_scalar,
    decode_thetao_array,
    compute_hot_end_metrics,
    find_above_valid_max_cells,
    extract_hottest_cells_context,
    track_hot_cells_multi_year,
    summarize_hot_end_by_level_year,
)


# =============================================================================
# Fixture: Synthetic Small NetCDF Dataset
# =============================================================================

@pytest.fixture
def synthetic_hot_end_setup(tmp_path):
    """Create minimal synthetic NetCDF files with known encoding and values."""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    jan_file = tmp_path / "glorys_thetao_2019-01.nc"
    static_file = tmp_path / "glorys_static_mask_bathy.nc"

    lats = np.array([29.8, 29.9, 30.0], dtype=np.float32)  # 3 rows (ny=3)
    lons = np.array([48.2, 48.3, 48.4, 48.5], dtype=np.float32)  # 4 cols (nx=4)
    depths = np.array([0.5, 6.44, 9.57], dtype=np.float32)  # 3 levels (nz=3)

    scale = 0.0007324442267417908
    offset = 21.0
    fill = -32767
    vmin = -32766
    vmax = 21306

    # 1. Static file
    with h5netcdf.File(static_file, "w") as f:
        f.dimensions = {"latitude": 3, "longitude": 4, "depth": 3}
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        f.create_variable("depth", ("depth",), data=depths)

        deptho = np.array([
            [15.0, 18.0, 22.0, 25.0],
            [12.0, 14.0, 16.0, 18.0],
            [8.0, 10.0, 11.0, 13.0],
        ], dtype=np.float32)
        deptho_lev = np.array([
            [8.0, 9.0, 10.0, 11.0],
            [7.0, 7.0, 8.0, 9.0],
            [5.0, 6.0, 6.0, 7.0],
        ], dtype=np.float32)

        v_dp = f.create_variable("deptho", ("latitude", "longitude"), data=deptho)
        v_dp.attrs["_FillValue"] = np.float32(9.96921e+36)
        v_dl = f.create_variable("deptho_lev", ("latitude", "longitude"), data=deptho_lev)
        v_dl.attrs["_FillValue"] = np.float32(9.96921e+36)
        v_m = f.create_variable("mask", ("depth", "latitude", "longitude"), data=np.ones((3, 3, 4), dtype=np.int8))

    # 2. Jan 2019 file (2 time steps)
    with h5netcdf.File(jan_file, "w") as f:
        f.dimensions = {"time": 2, "depth": 3, "latitude": 3, "longitude": 4}
        f.create_variable("time", ("time",), data=np.array([604848.0, 604872.0], dtype=np.float32))
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)

        # Day 0: normal values
        # Day 1: one out-of-range value at (lvl=1, r=1, c=1) with raw = 21500
        jan_data = np.full((2, 3, 3, 4), fill, dtype=np.int16)
        jan_data[0, :, 1:, 1:] = 12000  # finite normal
        jan_data[1, :, 1:, 1:] = 12000
        jan_data[1, 1, 1, 1] = 21500  # raw > vmax

        v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=jan_data)
        v.attrs["scale_factor"] = scale
        v.attrs["add_offset"] = offset
        v.attrs["_FillValue"] = fill
        v.attrs["valid_min"] = vmin
        v.attrs["valid_max"] = vmax

    # 3. Snapshot files: 1996 (normal), 1997 (has 2 cells > vmax), 1998 (normal)
    years_data = {
        1996: {
            "max_val": 20000,
            "hot_cells": [],
        },
        1997: {
            "max_val": 20000,
            "hot_cells": [
                (2, 2, 2, 22024),  # lvl=2 (depth=9.57m), r=2 (lat=30.0), c=2 (lon=48.4), raw=22024
                (1, 1, 3, 21400),  # lvl=1 (depth=6.44m), r=1 (lat=29.9), c=3 (lon=48.5), raw=21400
            ],
        },
        1998: {
            "max_val": 19500,
            "hot_cells": [],
        },
    }

    for y, y_info in years_data.items():
        snap_path = snap_dir / f"glorys_thetao_{y}-07-01.nc"
        s_data = np.full((1, 3, 3, 4), fill, dtype=np.int16)
        s_data[0, :, :, :] = y_info["max_val"]
        s_data[0, :, 0, 0] = fill  # keep corner as fill
        for k, r, c, val in y_info["hot_cells"]:
            s_data[0, k, r, c] = val

        with h5netcdf.File(snap_path, "w") as f:
            f.dimensions = {"time": 1, "depth": 3, "latitude": 3, "longitude": 4}
            f.create_variable("time", ("time",), data=np.array([400000.0], dtype=np.float32))
            f.create_variable("depth", ("depth",), data=depths)
            f.create_variable("latitude", ("latitude",), data=lats)
            f.create_variable("longitude", ("longitude",), data=lons)
            v = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=s_data)
            v.attrs["scale_factor"] = scale
            v.attrs["add_offset"] = offset
            v.attrs["_FillValue"] = fill
            v.attrs["valid_min"] = vmin
            v.attrs["valid_max"] = vmax

    return {
        "snap_dir": str(snap_dir),
        "jan_file": str(jan_file),
        "static_file": str(static_file),
        "years": [1996, 1997, 1998],
        "scale": scale,
        "offset": offset,
        "fill": fill,
        "vmin": vmin,
        "vmax": vmax,
        "lats": lats,
        "lons": lons,
        "depths": depths,
    }


# =============================================================================
# Test 1: Cells above valid_max are counted correctly and fill is never counted
# =============================================================================

def test_cells_above_valid_max_counted_and_fill_ignored(synthetic_hot_end_setup):
    """Verify that find_above_valid_max_cells counts cells > 21306 and ignores fill."""
    cfg = synthetic_hot_end_setup

    df_above = find_above_valid_max_cells(
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
        years=cfg["years"],
        valid_max=cfg["vmax"],
        fill_value=cfg["fill"],
        scale_factor=cfg["scale"],
        add_offset=cfg["offset"],
    )

    # In our fixture:
    # 1997 snapshot has 2 cells: raw 22024 and 21400
    # Jan 2019 has 1 cell: day 1 raw 21500
    # Total = 3 cells above vmax!
    assert len(df_above) == 3

    # All returned cells must strictly have raw > 21306
    assert (df_above["raw"] > cfg["vmax"]).all()

    # Verify no fill value was counted
    assert (df_above["raw"] != cfg["fill"]).all()

    # Verify sorting: highest raw/decoded first
    assert df_above.iloc[0]["raw"] == 22024
    assert df_above.iloc[1]["raw"] == 21500
    assert df_above.iloc[2]["raw"] == 21400

    # Verify coordinates of top cell
    top_cell = df_above.iloc[0]
    assert top_cell["date"] == "1997-07-01"
    assert top_cell["level"] == 2
    assert top_cell["lat_idx"] == 2
    assert top_cell["lon_idx"] == 2
    assert np.isclose(top_cell["decoded_C"], 22024 * cfg["scale"] + cfg["offset"])
    assert top_cell["static_deptho"] == 11.0
    assert top_cell["static_deptho_lev"] == 6.0


# =============================================================================
# Test 2: Independent maximum-raw function is correct
# =============================================================================

def test_independent_maximum_raw_function():
    """Verify compute_hot_end_metrics finds correct max raw, level, lat, lon and ignores fill."""
    fill = -32767
    scale = 0.001
    offset = 20.0
    vmax = 21306
    ceiling_thresh = 21256

    lats = np.array([10.0, 11.0], dtype=np.float64)
    lons = np.array([50.0, 51.0, 52.0], dtype=np.float64)
    depths = np.array([5.0, 15.0], dtype=np.float64)

    # 3D shape (2 levels, 2 lats, 3 lons)
    raw_3d = np.full((2, 2, 3), fill, dtype=np.int16)
    # Set finite cells
    raw_3d[0, 0, 0] = 10000
    raw_3d[0, 0, 1] = 21250
    raw_3d[0, 1, 0] = 21256  # exactly at ceiling threshold
    raw_3d[1, 1, 2] = 22024  # maximum raw at (lvl=1, r=1, c=2)

    metrics = compute_hot_end_metrics(
        raw_3d, lats, lons, depths,
        fill_value=fill,
        ceiling_thresh=ceiling_thresh,
        valid_max=vmax,
        scale_factor=scale,
        add_offset=offset,
    )

    assert metrics["max_raw"] == 22024
    assert metrics["max_raw_level"] == 1
    assert metrics["max_raw_depth_m"] == 15.0
    assert metrics["max_raw_lat_idx"] == 1
    assert metrics["max_raw_lon_idx"] == 2
    assert metrics["max_raw_lat"] == 11.0
    assert metrics["max_raw_lon"] == 52.0
    assert np.isclose(metrics["max_decoded_C"], 22024 * 0.001 + 20.0)
    assert metrics["n_finite"] == 4
    # Cells >= 21256: 21256, 22024 -> 2 cells
    assert metrics["n_ge_ceiling_thresh"] == 2
    # Cells > 21306: 22024 -> 1 cell
    assert metrics["n_gt_valid_max"] == 1
    # Cells in [21256, 21306]: 21256 -> 1 cell
    assert metrics["n_between_thresh_and_vmax"] == 1


# =============================================================================
# Test 3: Ceiling rule applied with explicit, tested comparison operators
# =============================================================================

def test_ceiling_rule_comparison_operators():
    """Verify behavior of the three comparison operators on a known array.
    
    Demonstrates the exact root cause of the contradiction:
    - Operator 1 (stated rule): raw >= 21256
    - Operator 2 (snapshot census code): (raw >= 21256) & (raw <= 21306)
    - Operator 3 (out of declared range): raw > 21306
    """
    fill = -32767
    valid_max = 21306
    ceiling_thresh = 21256

    test_raw = np.array([
        fill,            # Fill value: must be excluded by finite mask
        15000,           # Within range, below threshold
        21255,           # Just below threshold (21256 - 1)
        21256,           # Exactly threshold
        21300,           # In range [21256, 21306]
        21306,           # Exactly valid_max
        21307,           # Just above valid_max
        22024,           # Warmest extreme in 1997 snapshot
    ], dtype=np.int16)

    finite = (test_raw != fill)
    raw_fin = test_raw[finite]

    # Operator 1: Stated rule "raw >= 21256"
    op1_mask = (raw_fin >= ceiling_thresh)
    # Operator 2: Census code "(raw <= valid_max) & (raw >= valid_max - 50)"
    op2_mask = (raw_fin <= valid_max) & (raw_fin >= ceiling_thresh)
    # Operator 3: Out-of-range "raw > valid_max"
    op3_mask = (raw_fin > valid_max)

    count_op1 = int(np.sum(op1_mask))
    count_op2 = int(np.sum(op2_mask))
    count_op3 = int(np.sum(op3_mask))

    # Cells satisfying Operator 1: 21256, 21300, 21306, 21307, 22024 (5 cells)
    assert count_op1 == 5
    # Cells satisfying Operator 2: 21256, 21300, 21306 (3 cells)
    assert count_op2 == 3
    # Cells satisfying Operator 3: 21307, 22024 (2 cells)
    assert count_op3 == 2

    # Operator 1 equals Operator 2 + Operator 3
    assert count_op1 == count_op2 + count_op3

    # CRUCIAL DEMONSTRATION FOR 1997:
    # If the finite dataset only has cells with raw > 21306 (like 22024):
    only_hot = np.array([22024, 22017, 21972], dtype=np.int16)
    hot_op1 = int(np.sum(only_hot >= ceiling_thresh))
    hot_op2 = int(np.sum((only_hot <= valid_max) & (only_hot >= ceiling_thresh)))
    hot_op3 = int(np.sum(only_hot > valid_max))

    assert hot_op1 == 3
    assert hot_op2 == 0  # <--- ZERO! This is why total_near_ceiling = 0 in census!
    assert hot_op3 == 3


# =============================================================================
# Test 4: SHA256 unchanged after analysis
# =============================================================================

def test_sha256_unchanged_after_analysis(synthetic_hot_end_setup):
    """Verify that file SHA256 remains 100% unchanged before and after analysis."""
    cfg = synthetic_hot_end_setup

    files_to_track = [cfg["jan_file"], cfg["static_file"]]
    for y in cfg["years"]:
        files_to_track.append(os.path.join(cfg["snap_dir"], f"glorys_thetao_{y}-07-01.nc"))

    # Compute before metadata
    before_meta = {f: compute_file_metadata(f) for f in files_to_track}

    # Run analysis functions
    df_above = find_above_valid_max_cells(
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
        years=cfg["years"],
    )
    df_ctx = extract_hottest_cells_context(
        df_above_vmax=df_above,
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
    )
    df_my = track_hot_cells_multi_year(
        df_hottest=df_ctx,
        snapshot_dir=cfg["snap_dir"],
        years=cfg["years"],
    )
    df_ly = summarize_hot_end_by_level_year(
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        years=cfg["years"],
    )

    # Compute after metadata
    after_meta = {f: compute_file_metadata(f) for f in files_to_track}

    all_unchanged, mismatches = verify_integrity_unchanged(before_meta, after_meta)
    assert all_unchanged is True
    assert len(mismatches) == 0


# =============================================================================
# Test 5: Boundary handling for 3x3 neighbourhood
# =============================================================================

def test_neighbourhood_boundary_handling(synthetic_hot_end_setup):
    """Verify that 3x3 neighbourhood properly marks OUT_OF_GRID and FILL without crashing."""
    cfg = synthetic_hot_end_setup

    # Cell at corner (lvl=2, r=2, c=2) in a 3-row grid:
    # r=2 is the bottom edge (dr=+1 is OUT_OF_GRID)
    df_above = pd.DataFrame([{
        "date": "1997-07-01",
        "level": 2,
        "depth_m": 9.57,
        "lat_idx": 2,
        "lon_idx": 2,
        "latitude": 30.0,
        "longitude": 48.4,
        "raw": 22024,
        "decoded_C": 37.13,
        "static_deptho": 11.0,
        "static_deptho_lev": 6.0,
    }])

    df_ctx = extract_hottest_cells_context(
        df_above_vmax=df_above,
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
        top_n=1,
    )

    assert len(df_ctx) == 1
    row = df_ctx.iloc[0]

    # Row +1 must be OUT_OF_GRID because ny=3 (valid rows are 0, 1, 2)
    assert row["nb_r+1_c-1"] == "OUT_OF_GRID"
    assert row["nb_r+1_c0"] == "OUT_OF_GRID"
    assert row["nb_r+1_c+1"] == "OUT_OF_GRID"

    # Center cell is the target
    assert np.isclose(float(row["nb_r0_c0"]), 37.13, atol=0.01)

    # Profile has 3 levels
    assert row["profile_lev00"] != "FILL"
    assert row["profile_lev01"] != "FILL"
    assert row["profile_lev02"] != "FILL"


# =============================================================================
# Test 6: Multi-year tracking and percentile calculations
# =============================================================================

def test_multi_year_tracking_and_percentiles(synthetic_hot_end_setup):
    """Verify track_hot_cells_multi_year computes percentiles and rank correctly."""
    cfg = synthetic_hot_end_setup

    df_above = find_above_valid_max_cells(
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
        years=cfg["years"],
    )
    df_ctx = extract_hottest_cells_context(
        df_above_vmax=df_above,
        snapshot_dir=cfg["snap_dir"],
        jan2019_file=cfg["jan_file"],
        static_file=cfg["static_file"],
        top_n=2,
    )

    df_my = track_hot_cells_multi_year(
        df_hottest=df_ctx,
        snapshot_dir=cfg["snap_dir"],
        years=cfg["years"],
    )

    # 2 cells x 3 years = 6 rows
    assert len(df_my) == 6

    # In 1997, rank 1 cell has raw 22024, which is the highest in the domain
    r1_1997 = df_my[(df_my["cell_rank"] == 1) & (df_my["year"] == 1997)].iloc[0]
    assert r1_1997["cell_raw"] == 22024
    # Since it's the maximum value in the domain, its weak percentile rank is 100.0
    assert np.isclose(r1_1997["cell_percentile_rank"], 100.0)
    assert r1_1997["cell_decoded_C"] >= r1_1997["domain_p99_9_decoded_C"]
