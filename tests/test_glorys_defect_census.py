"""
Tests for GLORYS Defect Census.

Verifies:
- Frozen cell detection (constant vs time-varying vs fill cell),
- Planted outlier residual detection,
- Unevaluable cell detection (<3 finite neighbours),
- Halting if finite mask is time-varying,
- Immutability of file SHA256.
"""

import os
import tempfile
import numpy as np
import pytest
import h5netcdf

from scripts.diagnostics.glorys_defect_census import (
    census_frozen_cells,
    census_local_consistency,
    run_defect_census,
)
from scripts.diagnostics.glorys_encoding_forensics import (
    compute_file_metadata,
    RawNetCDFReader,
    check_mask_time_invariance,
)


@pytest.fixture
def synthetic_defect_nc(tmp_path):
    """Create a synthetic NetCDF file with controlled cells.

    Dimensions:
      time: 31
      depth: 2 (level 0: 50.0m, level 1: 380.0m)
      latitude: 5 (14.8, 14.9, 15.0, 15.1, 15.2)
      longitude: 5 (50.4, 50.5, 50.583332, 50.666668, 50.8)

    Cells planted at level 0:
      (2, 2) -> Center cell, constant code 1000 across all 31 days (FROZEN)
      (2, 3) -> Varying cell (1000 + t) across 31 days (VARYING)
      (0, 0) -> Fill cell (-32767) (FILL)
      (0, 1) -> Finite cell, but surrounded by fill so has only 2 finite neighbors (UNEVALUABLE)
      (3, 3) -> Planted outlier: decoded value ~0.0 C while neighbors are ~20.0 C
    """
    nc_path = tmp_path / "synthetic_defect.nc"

    scale_factor = 0.0007324442267417908
    add_offset = 21.0
    fill_value = -32767

    times = np.arange(31, dtype=np.int32)
    depths = np.array([50.0, 380.213], dtype=np.float32)
    lats = np.array([14.8, 14.9, 15.0, 15.1, 15.2], dtype=np.float32)
    lons = np.array([50.4, 50.5, 50.583332, 50.666668, 50.8], dtype=np.float32)

    # Base: non-fill code ~ -1365 (decodes to ~20.0 C)
    # -1365 * scale + 21 = 20.0002
    data = np.full((31, 2, 5, 5), -1365, dtype=np.int16)

    # Varying across time for (2, 3):
    for t in range(31):
        data[t, 0, 2, 3] = -1365 + t

    # Frozen cell at (2, 2): stays exactly 1000 on all 31 days
    data[:, 0, 2, 2] = 1000

    # Fill cell at (0, 0)
    data[:, 0, 0, 0] = fill_value

    # Make cell (0, 1) have only 2 finite neighbours:
    # Neighbors of (0, 1) are: (0, 0) [fill], (0, 2), (1, 0), (1, 1), (1, 2).
    # If we set (1, 0), (1, 1), and (1, 2) to fill, then (0, 1) has only (0, 2) as finite neighbor!
    data[:, 0, 1, 0] = fill_value
    data[:, 0, 1, 1] = fill_value
    data[:, 0, 1, 2] = fill_value

    # Planted spatial outlier at (3, 3):
    # Set to -28671 (~0.0 C) while surrounding are ~20.0 C
    data[:, 0, 3, 3] = -28671

    with h5netcdf.File(str(nc_path), "w") as f:
        f.attrs["title"] = "Synthetic Defect Test NetCDF"
        f.attrs["Conventions"] = "CF-1.4"
        f.dimensions = {
            "time": len(times),
            "depth": len(depths),
            "latitude": len(lats),
            "longitude": len(lons),
        }

        f.create_variable("time", ("time",), data=times)
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)

        vt = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=data)
        vt.attrs["scale_factor"] = scale_factor
        vt.attrs["add_offset"] = add_offset
        vt.attrs["_FillValue"] = fill_value
        vt.attrs["valid_min"] = np.int16(-32766)
        vt.attrs["valid_max"] = np.int16(21306)

    return str(nc_path)


def test_frozen_cell_detection(synthetic_defect_nc):
    """Verify constant cell is flagged frozen; varying cell and fill cell are not."""
    with RawNetCDFReader(synthetic_defect_nc) as reader:
        by_level_df, frozen_list_df, summary = census_frozen_cells(
            reader=reader,
            scale_factor=0.0007324442267417908,
            add_offset=21.0,
            fill_value=-32767,
        )

    # Check level 0 frozen cells
    lvl0_frozen = frozen_list_df[frozen_list_df["level_index"] == 0]
    frozen_coords = list(zip(lvl0_frozen["lat_index"], lvl0_frozen["lon_index"]))

    # Cell (2, 2) is frozen with code 1000
    assert (2, 2) in frozen_coords
    # Cell (3, 3) is planted with constant code -28671 -> also frozen
    assert (3, 3) in frozen_coords

    # Varying cell (2, 3) must NOT be frozen
    assert (2, 3) not in frozen_coords

    # Fill cell (0, 0) must NOT be frozen
    assert (0, 0) not in frozen_coords


def test_planted_outlier_residual(synthetic_defect_nc):
    """Verify planted spatial outlier gets a large residual and ranks top."""
    with RawNetCDFReader(synthetic_defect_nc) as reader:
        consistency_df, top10_df, summary, _ = census_local_consistency(
            reader=reader,
            scale_factor=0.0007324442267417908,
            add_offset=21.0,
            fill_value=-32767,
            lat_A_idx=2,
            lon_A_idx=2,
            lat_B_idx=2,
            lon_B_idx=3,
        )

    lvl0_top = top10_df[top10_df["level_index"] == 0]
    assert len(lvl0_top) > 0

    # Planted outlier at (3, 3) should have rank 1 or large residual (~ -20 C)
    row_outlier = lvl0_top[(lvl0_top["lat_index"] == 3) & (lvl0_top["lon_index"] == 3)]
    assert len(row_outlier) == 1
    res = row_outlier["residual_C"].iloc[0]
    assert res < -15.0  # Decoded value ~0 C vs neighbor median ~20 C -> residual ~ -20 C


def test_unevaluable_cell(synthetic_defect_nc):
    """Verify finite cell with fewer than 3 finite neighbours is flagged UNEVALUABLE."""
    with RawNetCDFReader(synthetic_defect_nc) as reader:
        consistency_df, top10_df, summary, _ = census_local_consistency(
            reader=reader,
            scale_factor=0.0007324442267417908,
            add_offset=21.0,
            fill_value=-32767,
            lat_A_idx=2,
            lon_A_idx=2,
            lat_B_idx=2,
            lon_B_idx=3,
        )

    # In level 0, cell (0, 1) has < 3 finite neighbours
    lvl0_consistency = consistency_df[consistency_df["level_index"] == 0].iloc[0]
    assert lvl0_consistency["unevaluable_finite_cells"] >= 1

    # Cell (0, 1) should not be in top 10 evaluable residuals
    lvl0_top = top10_df[top10_df["level_index"] == 0]
    uneval_in_top = lvl0_top[(lvl0_top["lat_index"] == 0) & (lvl0_top["lon_index"] == 1)]
    assert len(uneval_in_top) == 0


def test_time_varying_mask_stops(synthetic_defect_nc, tmp_path):
    """Verify execution stops if the finite mask changes across days."""
    mutated_nc = tmp_path / "mutated_mask.nc"

    with h5netcdf.File(synthetic_defect_nc, "r") as src, h5netcdf.File(str(mutated_nc), "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for dim, sz in src.dimensions.items():
            dst.dimensions[dim] = len(sz)
        for var in ["time", "depth", "latitude", "longitude"]:
            dst.create_variable(var, (var,), data=src[var][:])
        tdata = src["thetao"][:].copy()
        # Mutate cell (2, 2) on Day 5: make it fill where Day 0 was finite
        tdata[5, 0, 2, 2] = -32767
        vt = dst.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=tdata)
        for ak, av in src["thetao"].attrs.items():
            vt.attrs[ak] = av

    with RawNetCDFReader(str(mutated_nc)) as reader:
        inv = check_mask_time_invariance(reader, fill_value=-32767)
    assert inv["is_time_invariant"] is False
    assert inv["total_differing_points"] == 1


def test_sha256_unchanged_after_analysis(synthetic_defect_nc, tmp_path):
    """Verify that running analysis leaves input file checksums completely unchanged."""
    meta_before = compute_file_metadata(synthetic_defect_nc)

    with RawNetCDFReader(synthetic_defect_nc) as reader:
        _ = census_frozen_cells(reader, 0.00073, 21.0, -32767)
        _ = census_local_consistency(reader, 0.00073, 21.0, -32767, 2, 2, 2, 3)

    meta_after = compute_file_metadata(synthetic_defect_nc)

    assert meta_before["sha256"] == meta_after["sha256"]
    assert meta_before["size_bytes"] == meta_after["size_bytes"]
    assert meta_before["mtime_utc"] == meta_after["mtime_utc"]
