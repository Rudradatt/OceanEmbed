"""
Tests for Pipeline Month Driver (scripts/pipeline/month_pipeline.py)
"""

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import h5netcdf
import numpy as np
import pandas as pd
import pytest

from scripts.pipeline.month_pipeline import (
    MANIFEST_COLUMNS,
    append_manifest_row,
    compute_valid_and_usable_counts,
    download_source_month,
    format_month,
    generate_month_range,
    get_days_in_month,
    get_raw_filepath,
    is_month_completed,
    load_sources_config,
    parse_month,
    process_month,
    read_manifest,
    setup_logger,
    verify_all_raw_files,
    verify_raw_file,
)


@pytest.fixture
def mock_sources_cfg():
    return {
        "version": "v1",
        "spatial_subset": {
            "lat_min": 5.0,
            "lat_max": 30.0,
            "lon_min": 45.0,
            "lon_max": 105.0,
        },
        "sources": {
            "glorys": {
                "name": "GLORYS12V1 Daily Reanalysis",
                "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                "enabled": True,
                "raw_dir": "data/raw/glorys",
                "filename_template": "glorys_thetao_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
            },
            "static": {
                "name": "GLORYS12V1 Static Grid & Bathymetry",
                "dataset_id": "cmems_mod_glo_phy_my_0.083deg_static",
                "enabled": True,
                "raw_dir": "data/raw/glorys_static",
                "filename_template": "glorys_static_mask_bathy.nc",
                "download_frequency": "once",
            },
            "sst": {
                "name": "OSTIA SST",
                "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
                "enabled": True,
                "raw_dir": "data/raw/sst",
                "filename_template": "sst_ostia_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
            },
            "sla": {
                "name": "DUACS SLA",
                "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
                "enabled": True,
                "raw_dir": "data/raw/sla",
                "filename_template": "sla_duacs_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
            },
            "sss": {
                "name": "MULTIOBS SSS",
                "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
                "enabled": True,
                "raw_dir": "data/raw/sss",
                "filename_template": "sss_multiobs_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
            },
            "cur": {
                "name": "MULTIOBS Currents",
                "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
                "enabled": True,
                "raw_dir": "data/raw/currents",
                "filename_template": "currents_multiobs_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
            },
            "wind": {
                "name": "Wind L4",
                "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
                "enabled": False,
                "raw_dir": "data/raw/wind",
                "filename_template": "wind_hourly_{year:04d}-{month:02d}.nc",
                "download_frequency": "monthly",
                "notice": "DISABLED: Wind dataset not yet audited.",
            },
        },
    }


# =============================================================================
# 1. Month Iteration & Parsing Tests
# =============================================================================

def test_month_parsing_and_formatting():
    assert parse_month("2019-01") == (2019, 1)
    assert parse_month("2020-12") == (2020, 12)
    assert format_month(2019, 1) == "2019-01"

    with pytest.raises(ValueError):
        parse_month("2019/01")
    with pytest.raises(ValueError):
        parse_month("2019-13")
    with pytest.raises(ValueError):
        parse_month("invalid")


def test_days_in_month():
    assert get_days_in_month(2019, 1) == 31
    assert get_days_in_month(2019, 2) == 28
    assert get_days_in_month(2020, 2) == 29  # Leap year
    assert get_days_in_month(2019, 4) == 30


def test_month_range_generation():
    seq = generate_month_range("2019-01", "2019-04")
    assert seq == ["2019-01", "2019-02", "2019-03", "2019-04"]

    seq_cross_year = generate_month_range("2018-11", "2019-02")
    assert seq_cross_year == ["2018-11", "2018-12", "2019-01", "2019-02"]

    single = generate_month_range("2019-01", "2019-01")
    assert single == ["2019-01"]

    with pytest.raises(ValueError):
        generate_month_range("2019-05", "2019-01")


# =============================================================================
# 2. Resumability & Skip Tests
# =============================================================================

def test_is_month_completed(tmp_path):
    manifest_csv = tmp_path / "months_manifest.csv"
    output_root = tmp_path / "processed"
    output_root.mkdir()

    # Empty manifest
    assert not is_month_completed(str(manifest_csv), "2019-01", str(output_root))

    # Output files not created yet
    row = {"month": "2019-01", "status": "OK"}
    append_manifest_row(str(manifest_csv), row)
    assert not is_month_completed(str(manifest_csv), "2019-01", str(output_root))

    # Create dummy output files
    tgt_dir = output_root / "glorys_targets"
    tgt_dir.mkdir()
    (tgt_dir / "glorys_targets_2019-01.nc").write_text("dummy")

    inp_dir = output_root / "inputs"
    inp_dir.mkdir()
    (inp_dir / "inputs_2019-01.nc").write_text("dummy")

    # Now should be completed
    assert is_month_completed(str(manifest_csv), "2019-01", str(output_root))

    # Different month
    assert not is_month_completed(str(manifest_csv), "2019-02", str(output_root))


def test_process_month_skips_when_completed(tmp_path, mock_sources_cfg):
    manifest_csv = tmp_path / "months_manifest.csv"
    output_root = tmp_path / "processed"
    output_root.mkdir()

    row = {"month": "2019-01", "status": "OK"}
    append_manifest_row(str(manifest_csv), row)

    tgt_dir = output_root / "glorys_targets"
    tgt_dir.mkdir()
    (tgt_dir / "glorys_targets_2019-01.nc").write_text("dummy")
    inp_dir = output_root / "inputs"
    inp_dir.mkdir()
    (inp_dir / "inputs_2019-01.nc").write_text("dummy")

    res = process_month(
        month="2019-01",
        sources_cfg=mock_sources_cfg,
        output_root=str(output_root),
        manifest_csv_path=str(manifest_csv),
        force=False,
    )
    assert res["status"] == "SKIPPED"


# =============================================================================
# 3. Download Logic Tests (Mocks, Temp Rename, Cleanup, Retries)
# =============================================================================

def test_download_source_month_disabled_skipped(tmp_path, mock_sources_cfg):
    # wind is disabled
    res = download_source_month(
        source="wind",
        year=2019,
        month=1,
        sources_cfg=mock_sources_cfg,
        project_root=str(tmp_path),
    )
    assert res is None


def test_download_atomic_rename_and_cleanup_on_failure(tmp_path, mock_sources_cfg):
    call_count = 0

    def failing_downloader(source, dest_path, cfg):
        nonlocal call_count
        call_count += 1
        # Create temp file then fail
        with open(dest_path, "w") as f:
            f.write("partial")
        if call_count < 3:
            raise IOError("Simulated network drop")
        # On 3rd attempt succeed
        with open(dest_path, "w") as f:
            f.write("full_content")

    out_file = download_source_month(
        source="sst",
        year=2019,
        month=1,
        sources_cfg=mock_sources_cfg,
        project_root=str(tmp_path),
        downloader_fn=failing_downloader,
        max_retries=2,
    )

    assert out_file is not None
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        assert f.read() == "full_content"
    assert call_count == 3

    # Check no leftover .tmp files
    raw_dir = os.path.dirname(out_file)
    tmp_files = [f for f in os.listdir(raw_dir) if ".tmp." in f]
    assert len(tmp_files) == 0


def test_download_exceeds_retries_raises(tmp_path, mock_sources_cfg):
    def always_fail(source, dest_path, cfg):
        raise IOError("Permanent network failure")

    with pytest.raises(RuntimeError) as exc_info:
        download_source_month(
            source="sst",
            year=2019,
            month=1,
            sources_cfg=mock_sources_cfg,
            project_root=str(tmp_path),
            downloader_fn=always_fail,
            max_retries=2,
        )
    assert "Download failed for 'sst'" in str(exc_info.value)


# =============================================================================
# 4. Raw File Verification Tests (Missing, Failed Verify)
# =============================================================================

def test_verify_raw_file_missing_raises(tmp_path, mock_sources_cfg):
    p = str(tmp_path / "nonexistent.nc")
    with pytest.raises(FileNotFoundError):
        verify_raw_file("sst", p, 2019, 1, mock_sources_cfg)


def test_verify_raw_file_wrong_days_raises(tmp_path, mock_sources_cfg):
    nc_path = tmp_path / "sst_wrong_days.nc"
    with h5netcdf.File(str(nc_path), "w") as f:
        f.dimensions = {"time": 10, "lat": 2, "lon": 2}
        v = f.create_variable("analysed_sst", ("time", "lat", "lon"), dtype=np.int16)
        v[:] = 0
        m = f.create_variable("mask", ("time", "lat", "lon"), dtype=np.int8)
        m[:] = 1
        t = f.create_variable("time", ("time",), dtype=np.float32)
        t[:] = np.arange(10)
        t.attrs["units"] = "seconds since 1981-01-01 00:00:00"

    with pytest.raises(ValueError) as exc:
        verify_raw_file("sst", str(nc_path), 2019, 1, mock_sources_cfg)
    assert "10 time steps; expected 31 days" in str(exc.value)


def test_verify_all_raw_files_missing_status(tmp_path, mock_sources_cfg):
    ok, status, sizes, hashes, paths = verify_all_raw_files(
        year=2019,
        month=1,
        sources_cfg=mock_sources_cfg,
        project_root=str(tmp_path),
    )
    assert not ok
    assert status == "MISSING_RAW"


# =============================================================================
# 5. Raw GLORYS Deletion Safety Tests
# =============================================================================

def test_delete_raw_glorys_safety(tmp_path, mock_sources_cfg):
    # Setup dummy directory structure
    raw_glorys_dir = tmp_path / "data" / "raw" / "glorys"
    raw_glorys_dir.mkdir(parents=True)
    raw_glorys_file = raw_glorys_dir / "glorys_thetao_2019-01.nc"
    raw_glorys_file.write_text("raw_glorys_data")

    raw_sst_dir = tmp_path / "data" / "raw" / "sst"
    raw_sst_dir.mkdir(parents=True)
    raw_sst_file = raw_sst_dir / "sst_ostia_2019-01.nc"
    raw_sst_file.write_text("raw_sst_data")

    manifest_csv = tmp_path / "months_manifest.csv"
    output_root = tmp_path / "processed"
    output_root.mkdir()

    # Create dummy outputs so is_month_completed returns True
    tgt_dir = output_root / "glorys_targets"
    tgt_dir.mkdir()
    (tgt_dir / "glorys_targets_2019-01.nc").write_text("tgt")
    inp_dir = output_root / "inputs"
    inp_dir.mkdir()
    (inp_dir / "inputs_2019-01.nc").write_text("inp")

    # Mock process_month components
    with patch("scripts.pipeline.month_pipeline.verify_all_raw_files") as mock_verify, \
         patch("scripts.pipeline.month_pipeline.run_glorys_qc_month") as mock_qc, \
         patch("scripts.pipeline.month_pipeline.build_glorys_targets_month") as mock_tgt, \
         patch("scripts.pipeline.month_pipeline.build_inputs_month") as mock_inp, \
         patch("scripts.pipeline.month_pipeline.compute_valid_and_usable_counts") as mock_counts:

        mock_verify.return_value = (
            True,
            "OK",
            {"glorys": 100, "sst": 100},
            {"glorys": "shaG", "sst": "shaS"},
            {
                "glorys": str(raw_glorys_file),
                "sst": str(raw_sst_file),
                "static": str(tmp_path / "static.nc"),
                "sla": str(tmp_path / "sla.nc"),
                "sss": str(tmp_path / "sss.nc"),
                "cur": str(tmp_path / "cur.nc"),
            },
        )
        mock_qc.return_value = {"status": "PASS", "counts": {}, "unexpected_count": 0}
        mock_counts.return_value = ([100] * 15, [80] * 15)

        res = process_month(
            month="2019-01",
            sources_cfg=mock_sources_cfg,
            output_root=str(output_root),
            manifest_csv_path=str(manifest_csv),
            delete_raw_glorys=True,
            force=True,
        )

        assert res["status"] == "OK"
        # Check raw GLORYS file was deleted
        assert not os.path.exists(raw_glorys_file)
        # Check raw SST file was NOT deleted!
        assert os.path.exists(raw_sst_file)


# =============================================================================
# 6. Manifest Completeness & Atomicity Tests
# =============================================================================

def test_manifest_schema_and_atomicity(tmp_path):
    manifest_csv = tmp_path / "months_manifest.csv"

    record_1 = {
        "month": "2019-01",
        "status": "OK",
        "wall_seconds": 12.34,
        "raw_sizes": {"glorys": 12345},
        "raw_sha256": {"glorys": "abcde"},
        "qc_status": "PASS",
        "qc_counts": {"zero_code": 0},
        "builder_sha256": "builder123",
        "policy_sha256": "policy123",
        "git_commit": "commit123",
        "git_dirty": "false",
        "valid_counts_by_depth": [1000] * 15,
        "usable_counts_by_depth": [800] * 15,
    }

    append_manifest_row(str(manifest_csv), record_1)
    df = read_manifest(str(manifest_csv))

    assert len(df) == 1
    assert list(df.columns) == MANIFEST_COLUMNS
    assert df.iloc[0]["month"] == "2019-01"
    assert df.iloc[0]["status"] == "OK"
    assert json.loads(df.iloc[0]["valid_counts_by_depth"]) == [1000] * 15

    # Update row for same month
    record_1_update = dict(record_1)
    record_1_update["wall_seconds"] = 15.67
    append_manifest_row(str(manifest_csv), record_1_update)

    df2 = read_manifest(str(manifest_csv))
    assert len(df2) == 1  # Still 1 row
    assert float(df2.iloc[0]["wall_seconds"]) == 15.67


# =============================================================================
# 7. No Credentials Test
# =============================================================================

def test_no_credentials_in_logs_and_manifest(tmp_path):
    logger, log_path = setup_logger(str(tmp_path / "logs"))
    logger.info("Executing pipeline run with standard configuration.")

    manifest_csv = tmp_path / "months_manifest.csv"
    record = {
        "month": "2019-01",
        "status": "OK",
        "wall_seconds": 1.0,
        "raw_sizes": {},
        "raw_sha256": {},
        "qc_status": "PASS",
        "qc_counts": {},
        "builder_sha256": "hash",
        "policy_sha256": "hash",
        "git_commit": "hash",
        "git_dirty": "false",
        "valid_counts_by_depth": [],
        "usable_counts_by_depth": [],
    }
    append_manifest_row(str(manifest_csv), record)

    # Check log contents
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read().lower()
    for sensitive_word in ["password", "token", "secret", "bearer", "authorization"]:
        assert sensitive_word not in log_content

    # Check manifest contents
    with open(manifest_csv, "r", encoding="utf-8") as f:
        manifest_content = f.read().lower()
    for sensitive_word in ["password", "token", "secret", "bearer", "authorization"]:
        assert sensitive_word not in manifest_content

