"""
Tests for GLORYS Static Mask and Bathymetry Cross-Check.

Verifies:
1. The 2x2 confusion counts are correct.
2. A planted (mask==0 & finite) cell is listed with correct coordinates/value.
3. The level-index mismatch histograms for both conventions (d1, d0) are correct.
4. The grid alignment check fails loudly beyond tolerance (lat/lon and depth).
5. A 2D mask is handled without crashing and flagged appropriately.
6. SHA256 of files remains unchanged after analysis.
"""

import os
import hashlib
import h5netcdf
import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.glorys_static_mask_crosscheck import (
    compute_file_metadata,
    RawNetCDFReader,
    check_grid_alignment,
    evaluate_mask_vs_thetao,
    evaluate_bathymetry_consistency,
    run_static_crosscheck,
)


@pytest.fixture
def synthetic_pair_nc(tmp_path):
    """Create a pair of small synthetic NetCDF files (thetao and static) with known values."""
    thetao_path = tmp_path / "synthetic_thetao.nc"
    static_path = tmp_path / "synthetic_static.nc"

    # Coordinates
    lats = np.array([14.8, 14.9, 15.0, 15.1, 15.2], dtype=np.float32)
    lons = np.array([50.4, 50.5, 50.583332, 50.666668, 50.8], dtype=np.float32)
    depths = np.array([0.5, 10.0, 50.0, 100.0], dtype=np.float32)  # 4 levels

    times = np.array([0, 1, 2], dtype=np.float32)

    scale_factor = 0.001
    add_offset = 20.0
    fill_thetao = -32767
    fill_static = 9.96921e36

    ny = len(lats)
    nx = len(lons)
    nz = len(depths)

    # Base thetao: shape (3, 4, 5, 5)
    # Let's populate thetao day 0:
    # Most cells are finite (code = 1000 => 21.0 C)
    thetao_data = np.full((3, nz, ny, nx), 1000, dtype=np.int16)

    # Base 3D mask: shape (4, 5, 5), default 1 (sea)
    mask_data = np.ones((nz, ny, nx), dtype=np.int8)

    # Depth bathymetry
    # Level at seafloor (deptho_lev): default 4.0 (sea bottom at level 4 -> index 3)
    deptho_lev_data = np.full((ny, nx), 4.0, dtype=np.float32)
    deptho_data = np.full((ny, nx), 100.0, dtype=np.float32)

    # 1) At level 0, create known 2x2 cells:
    # Cell (0, 0): mask=1, thetao=finite (default)
    # Cell (0, 1): mask=1, thetao=fill
    thetao_data[:, 0, 0, 1] = fill_thetao

    # Cell (0, 2): mask=0, thetao=finite (PLANTED cell)
    mask_data[0, 0, 2] = 0
    thetao_data[:, 0, 0, 2] = 5000  # finite

    # Cell (0, 3): mask=0, thetao=fill (land cell)
    mask_data[0, 0, 3] = 0
    thetao_data[:, 0, 0, 3] = fill_thetao
    deptho_lev_data[0, 3] = fill_static
    deptho_data[0, 3] = fill_static

    # 2) For bathymetry consistency:
    # Cell (1, 1): deepest finite level = 2 (index 2).
    # deptho_lev = 3.0.
    # d1 = 2 - (3 - 1) = 0.
    # d0 = 2 - 3 = -1.
    thetao_data[:, 3, 1, 1] = fill_thetao  # level 3 is fill, so deepest is 2
    deptho_lev_data[1, 1] = 3.0
    deptho_data[1, 1] = 50.0

    # Cell (1, 2): deepest finite level = 1 (index 1).
    # deptho_lev = 3.0.
    # d1 = 1 - (3 - 1) = -1.
    # d0 = 1 - 3 = -2.
    thetao_data[:, 2:, 1, 2] = fill_thetao  # levels 2 and 3 are fill, deepest is 1
    deptho_lev_data[1, 2] = 3.0
    deptho_data[1, 2] = 50.0

    # Cell (1, 3): deptho_lev = 10.0 (> nz=4).
    # All 4 levels are finite, so deepest is 3 (nz - 1).
    deptho_lev_data[1, 3] = 10.0
    deptho_data[1, 3] = 500.0

    # Write synthetic thetao NetCDF
    with h5netcdf.File(str(thetao_path), "w") as f:
        f.attrs["title"] = "Synthetic Thetao"
        f.dimensions = {"time": 3, "depth": nz, "latitude": ny, "longitude": nx}
        f.create_variable("time", ("time",), data=times)
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        vt = f.create_variable("thetao", ("time", "depth", "latitude", "longitude"), data=thetao_data)
        vt.attrs["scale_factor"] = scale_factor
        vt.attrs["add_offset"] = add_offset
        vt.attrs["_FillValue"] = fill_thetao

    # Write synthetic static NetCDF
    with h5netcdf.File(str(static_path), "w") as f:
        f.attrs["title"] = "Synthetic Static"
        f.dimensions = {"depth": nz, "latitude": ny, "longitude": nx}
        f.create_variable("depth", ("depth",), data=depths)
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        vm = f.create_variable("mask", ("depth", "latitude", "longitude"), data=mask_data)
        vm.attrs["long_name"] = "Land-sea mask"
        vd = f.create_variable("deptho", ("latitude", "longitude"), data=deptho_data)
        vd.attrs["_FillValue"] = fill_static
        vdl = f.create_variable("deptho_lev", ("latitude", "longitude"), data=deptho_lev_data)
        vdl.attrs["_FillValue"] = fill_static

    return {
        "thetao_path": str(thetao_path),
        "static_path": str(static_path),
        "lats": lats,
        "lons": lons,
        "depths": depths,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "fill_thetao": fill_thetao,
        "fill_static": fill_static,
    }


def test_2x2_confusion_counts(synthetic_pair_nc):
    """Test 1: Verify 2x2 confusion counts (mask1_fin, mask1_fil, m0_fin, m0_fil)."""
    with RawNetCDFReader(synthetic_pair_nc["static_path"]) as s_reader, \
         RawNetCDFReader(synthetic_pair_nc["thetao_path"]) as t_reader:
        mask = s_reader.get_raw_variable("mask")
        thetao0 = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))

    res = evaluate_mask_vs_thetao(
        mask=mask,
        thetao_day0=thetao0,
        fill_value=synthetic_pair_nc["fill_thetao"],
        lats=synthetic_pair_nc["lats"],
        lons=synthetic_pair_nc["lons"],
        depths=synthetic_pair_nc["depths"],
        scale_factor=synthetic_pair_nc["scale_factor"],
        add_offset=synthetic_pair_nc["add_offset"],
    )

    df = res["by_level_df"]
    # At level 0:
    # Total cells = 25
    # Cell (0, 1): mask=1, fill -> mask1_fill = 1
    # Cell (0, 2): mask=0, finite -> mask0_finite = 1
    # Cell (0, 3): mask=0, fill -> mask0_fill = 1
    # Remaining 22 cells: mask=1, finite -> mask1_finite = 22
    row0 = df[df["level"] == 0].iloc[0]
    assert row0["mask1_finite"] == 22
    assert row0["mask1_fill"] == 1
    assert row0["mask0_finite"] == 1
    assert row0["mask0_fill"] == 1
    assert row0["total_cells"] == 25


def test_planted_mask0_finite_cell_listed(synthetic_pair_nc):
    """Test 2: Verify planted (mask==0 & finite) cell is listed with exact coordinates and value."""
    with RawNetCDFReader(synthetic_pair_nc["static_path"]) as s_reader, \
         RawNetCDFReader(synthetic_pair_nc["thetao_path"]) as t_reader:
        mask = s_reader.get_raw_variable("mask")
        thetao0 = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))

    res = evaluate_mask_vs_thetao(
        mask=mask,
        thetao_day0=thetao0,
        fill_value=synthetic_pair_nc["fill_thetao"],
        lats=synthetic_pair_nc["lats"],
        lons=synthetic_pair_nc["lons"],
        depths=synthetic_pair_nc["depths"],
        scale_factor=synthetic_pair_nc["scale_factor"],
        add_offset=synthetic_pair_nc["add_offset"],
    )

    # Cell planted was at level 0, y=0, x=2, raw_code=5000
    # Decoded = 5000 * 0.001 + 20.0 = 25.0
    planted_matches = [
        c for c in res["m0_finite_cells"]
        if c["level"] == 0 and c["y_idx"] == 0 and c["x_idx"] == 2
    ]
    assert len(planted_matches) == 1
    cell = planted_matches[0]
    assert cell["lat"] == pytest.approx(synthetic_pair_nc["lats"][0])
    assert cell["lon"] == pytest.approx(synthetic_pair_nc["lons"][2])
    assert cell["raw_code"] == 5000
    assert cell["decoded_thetao"] == pytest.approx(25.0)


def test_level_index_mismatch_histograms(synthetic_pair_nc):
    """Test 3: Verify bathymetry level mismatch histograms for conventions d1 and d0."""
    with RawNetCDFReader(synthetic_pair_nc["static_path"]) as s_reader, \
         RawNetCDFReader(synthetic_pair_nc["thetao_path"]) as t_reader:
        deptho = s_reader.get_raw_variable("deptho")
        deptho_lev = s_reader.get_raw_variable("deptho_lev")
        thetao0 = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))

    res = evaluate_bathymetry_consistency(
        thetao_day0=thetao0,
        fill_value=synthetic_pair_nc["fill_thetao"],
        deptho=deptho,
        fill_deptho=synthetic_pair_nc["fill_static"],
        deptho_lev=deptho_lev,
        fill_deptho_lev=synthetic_pair_nc["fill_static"],
        lats=synthetic_pair_nc["lats"],
        lons=synthetic_pair_nc["lons"],
        depths=synthetic_pair_nc["depths"],
    )

    # In our synthetic grid:
    # Total cells: 25.
    # Land cell (0, 3): fill_deptho_lev -> excluded. Valid columns: 24.
    # Columns with deptho_lev > 4: cell (1, 3) has deptho_lev=10.0 -> count_gt = 1.
    # Columns with deptho_lev <= 4: count_le = 23.
    # Among these 23:
    # Cell (1, 1): deepest = 2, deptho_lev = 3 -> d1 = 2 - (3-1) = 0; d0 = 2 - 3 = -1.
    # Cell (1, 2): deepest = 1, deptho_lev = 3 -> d1 = 1 - (3-1) = -1; d0 = 1 - 3 = -2.
    # The remaining 21 columns have deepest = 3 (index 3) and deptho_lev = 4.0:
    #   d1 = 3 - (4-1) = 0.
    #   d0 = 3 - 4 = -1.
    # So for d1: exactly 22 columns have d1 == 0, and 1 column has d1 == -1.
    # For d0: exactly 22 columns have d0 == -1, and 1 column has d0 == -2.
    assert res["count_deptho_lev_le_36"] == 23
    assert res["count_deptho_lev_gt_36"] == 1
    assert res["hist_d1"].get(0.0) == 22
    assert res["hist_d1"].get(-1.0) == 1
    assert res["exact_match_d1_count"] == 22
    assert res["exact_match_d1_fraction"] == pytest.approx(22 / 23)

    assert res["hist_d0"].get(-1.0) == 22
    assert res["hist_d0"].get(-2.0) == 1
    assert res["exact_match_d0_count"] == 0
    assert res["exact_match_d0_fraction"] == 0.0

    # Column with deptho_lev > 4 has deepest == 3 (max level index 4 - 1 = 3)
    assert res["deepest_at_max_count_gt_36"] == 1
    assert "1-based" in res["better_convention_inference"]


def test_grid_alignment_fails_loudly_beyond_tolerance(tmp_path):
    """Test 4: Verify grid alignment raises ValueError when tolerance is exceeded."""
    p_good = tmp_path / "good.nc"
    p_bad_lat = tmp_path / "bad_lat.nc"
    p_bad_depth = tmp_path / "bad_depth.nc"

    lats = np.array([10.0, 11.0], dtype=np.float32)
    lons = np.array([50.0, 51.0], dtype=np.float32)
    depths = np.array([1.0, 10.0], dtype=np.float32)

    with h5netcdf.File(str(p_good), "w") as f:
        f.dimensions = {"depth": 2, "latitude": 2, "longitude": 2}
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        f.create_variable("depth", ("depth",), data=depths)

    # Bad lat: diff = 0.01 > 1e-4
    with h5netcdf.File(str(p_bad_lat), "w") as f:
        f.dimensions = {"depth": 2, "latitude": 2, "longitude": 2}
        f.create_variable("latitude", ("latitude",), data=lats + 0.01)
        f.create_variable("longitude", ("longitude",), data=lons)
        f.create_variable("depth", ("depth",), data=depths)

    # Bad depth: diff = 0.01 > 1e-3
    with h5netcdf.File(str(p_bad_depth), "w") as f:
        f.dimensions = {"depth": 2, "latitude": 2, "longitude": 2}
        f.create_variable("latitude", ("latitude",), data=lats)
        f.create_variable("longitude", ("longitude",), data=lons)
        f.create_variable("depth", ("depth",), data=depths + 0.01)

    with RawNetCDFReader(str(p_good)) as r_good, RawNetCDFReader(str(p_bad_lat)) as r_bad_lat:
        with pytest.raises(ValueError, match="Latitude max difference"):
            check_grid_alignment(r_bad_lat, r_good)

    with RawNetCDFReader(str(p_good)) as r_good, RawNetCDFReader(str(p_bad_depth)) as r_bad_depth:
        with pytest.raises(ValueError, match="Depth max difference"):
            check_grid_alignment(r_bad_depth, r_good)


def test_2d_mask_handled_without_crashing_and_flagged(synthetic_pair_nc):
    """Test 5: Verify 2D mask is handled safely, compares level 0, and flags 3D impossible."""
    with RawNetCDFReader(synthetic_pair_nc["static_path"]) as s_reader, \
         RawNetCDFReader(synthetic_pair_nc["thetao_path"]) as t_reader:
        mask_3d = s_reader.get_raw_variable("mask")
        thetao0 = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))

    # Slice mask to 2D
    mask_2d = mask_3d[0]
    assert mask_2d.ndim == 2

    res = evaluate_mask_vs_thetao(
        mask=mask_2d,
        thetao_day0=thetao0,
        fill_value=synthetic_pair_nc["fill_thetao"],
        lats=synthetic_pair_nc["lats"],
        lons=synthetic_pair_nc["lons"],
        depths=synthetic_pair_nc["depths"],
        scale_factor=synthetic_pair_nc["scale_factor"],
        add_offset=synthetic_pair_nc["add_offset"],
    )

    assert res["is_2d"] is True
    assert res["is_3d_possible"] is False
    df = res["by_level_df"]
    assert len(df) == 4  # 4 levels total
    row0 = df[df["level"] == 0].iloc[0]
    assert row0["mask1_finite"] is not None
    # Level 1 has note
    row1 = df[df["level"] == 1].iloc[0]
    assert pd.isna(row1["mask1_finite"])
    assert "3D comparison could not be done" in row1["note"]


def test_sha256_unchanged_after_analysis(synthetic_pair_nc, tmp_path):
    """Test 6: Verify file SHA256 is strictly unchanged after analysis execution."""
    t_path = synthetic_pair_nc["thetao_path"]
    s_path = synthetic_pair_nc["static_path"]

    def get_sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while c := f.read(65536):
                h.update(c)
        return h.hexdigest().lower()

    sha_t_before = get_sha256(t_path)
    sha_s_before = get_sha256(s_path)

    out_dir = tmp_path / "test_output"

    # We can run the functions or run_static_crosscheck
    # To run run_static_crosscheck, expected thetao sha check can be mocked or we verify functions
    # Let's test the functions directly reading both files:
    with RawNetCDFReader(s_path) as s_reader, RawNetCDFReader(t_path) as t_reader:
        _ = check_grid_alignment(s_reader, t_reader)
        m = s_reader.get_raw_variable("mask")
        th = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))
        _ = evaluate_mask_vs_thetao(
            mask=m,
            thetao_day0=th,
            fill_value=synthetic_pair_nc["fill_thetao"],
            lats=synthetic_pair_nc["lats"],
            lons=synthetic_pair_nc["lons"],
            depths=synthetic_pair_nc["depths"],
            scale_factor=synthetic_pair_nc["scale_factor"],
            add_offset=synthetic_pair_nc["add_offset"],
        )

    sha_t_after = get_sha256(t_path)
    sha_s_after = get_sha256(s_path)

    assert sha_t_before == sha_t_after
    assert sha_s_before == sha_s_after
