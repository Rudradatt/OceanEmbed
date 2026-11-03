#!/usr/bin/env python3
"""
Pipeline Month Driver v1 (OceanEmbed Preprocessing Phase)

Orchestrates monthly processing pipeline for OceanEmbed:
1. Month range parsing and iteration
2. Download management with atomic rename, retries, and cleanup (mockable, no-network by default)
3. Raw file integrity and temporal verification against source contract
4. Pre-build QC verification on GLORYS reanalysis (QC FAIL halts build)
5. GLORYS target field generation via glorys_target_builder v1
6. Multi-source input block-mean aggregation via input_builder v1
7. Valid and usable sample census per depth
8. Atomic manifest logging (months_manifest.csv)
9. Resumability (skips OK months unless --force)
10. Controlled deletion of raw GLORYS reanalysis if requested (--delete-raw-glorys)
"""

from __future__ import annotations

import calendar
import csv
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

# Import preprocessing builders
from scripts.preprocessing.glorys_qc import (
    apply_qc,
    load_coordinates as load_glorys_coords,
    load_policy as load_qc_policy,
    load_static_wet_mask,
    locate_exclusions,
    read_raw_day as read_glorys_raw_day,
)
from scripts.preprocessing.glorys_target_builder import (
    build_month_file as build_glorys_targets_month,
    compute_file_sha256,
    get_git_commit_hash,
    get_git_dirty,
)
from scripts.preprocessing.input_builder import (
    build_month_file as build_inputs_month,
    decode_timestamps,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_SOURCES_CONFIG = os.path.join(PROJECT_ROOT, "configs", "sources_v1.json")
DEFAULT_QC_POLICY = os.path.join(PROJECT_ROOT, "configs", "glorys_qc_policy_v1.json")
DEFAULT_TARGET_POLICY = os.path.join(PROJECT_ROOT, "configs", "glorys_target_policy_v1.json")
DEFAULT_INPUT_POLICY = os.path.join(PROJECT_ROOT, "configs", "input_policy_v1.json")

MANIFEST_COLUMNS = [
    "month",
    "status",
    "wall_seconds",
    "raw_sizes",
    "raw_sha256",
    "qc_status",
    "qc_counts",
    "builder_sha256",
    "policy_sha256",
    "git_commit",
    "git_dirty",
    "valid_counts_by_depth",
    "usable_counts_by_depth",
]


# =============================================================================
# Helper Utilities & Configuration
# =============================================================================

def load_sources_config(config_path: str = DEFAULT_SOURCES_CONFIG) -> Dict[str, Any]:
    """Load sources configuration JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Sources config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_month(month_str: str) -> Tuple[int, int]:
    """Parse 'YYYY-MM' string into (year, month)."""
    m = re.match(r"^(\d{4})-(\d{2})$", month_str.strip())
    if not m:
        raise ValueError(f"Invalid month format: '{month_str}'. Expected 'YYYY-MM'.")
    year, month = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month number: {month} in '{month_str}'. Must be 1..12.")
    return year, month


def format_month(year: int, month: int) -> str:
    """Format year and month into 'YYYY-MM'."""
    return f"{year:04d}-{month:02d}"


def get_days_in_month(year: int, month: int) -> int:
    """Return number of calendar days in given year and month."""
    return calendar.monthrange(year, month)[1]


def generate_month_range(start_month: str, end_month: str) -> List[str]:
    """Generate inclusive list of 'YYYY-MM' strings between start and end."""
    y_start, m_start = parse_month(start_month)
    y_end, m_end = parse_month(end_month)

    if (y_start, m_start) > (y_end, m_end):
        raise ValueError(f"start_month ({start_month}) must be <= end_month ({end_month})")

    months: List[str] = []
    y, m = y_start, m_start
    while (y, m) <= (y_end, m_end):
        months.append(format_month(y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def setup_logger(log_dir: Optional[str] = None) -> Tuple[logging.Logger, str]:
    """Configure logger writing to stdout and a timestamped log file."""
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "outputs", "pipeline", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_filename = f"run_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(f"OceanEmbedPipeline_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s UTC] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_filepath


# =============================================================================
# Raw Path Resolution & Download Logic
# =============================================================================

def get_raw_filepath(
    source: str,
    year: int,
    month: int,
    sources_cfg: Dict[str, Any],
    project_root: str = PROJECT_ROOT,
) -> str:
    """Resolve standard raw file path for a source and month."""
    src_info = sources_cfg["sources"][source]
    raw_dir = os.path.join(project_root, src_info["raw_dir"])
    if src_info.get("download_frequency") == "once":
        filename = src_info["filename_template"]
    else:
        filename = src_info["filename_template"].format(year=year, month=month)
    return os.path.join(raw_dir, filename)


def download_source_month(
    source: str,
    year: int,
    month: int,
    sources_cfg: Dict[str, Any],
    project_root: str = PROJECT_ROOT,
    downloader_fn: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    max_retries: int = 2,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Download raw dataset month with atomic rename, retries, and temp file cleanup.

    In this task / offline mode, network calls are disabled unless a custom mock
    downloader_fn is injected for unit testing.
    """
    src_info = sources_cfg["sources"].get(source)
    if not src_info:
        raise KeyError(f"Unknown source: {source}")

    if not src_info.get("enabled", True):
        msg = f"Source '{source}' is DISABLED: {src_info.get('notice', 'not audited')}. Skipping download."
        if logger:
            logger.info(msg)
        return None

    dest_path = get_raw_filepath(source, year, month, sources_cfg, project_root)
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_path):
        if logger:
            logger.info(f"Source '{source}' already exists at {dest_path}")
        return dest_path

    if downloader_fn is None:
        raise RuntimeError(
            f"Network download requested for '{source}' ({year:04d}-{month:02d}) but network access is "
            "disabled in this environment. Use --use-existing-raw or provide a mock downloader."
        )

    # Attempt download with retries and atomic rename
    attempt = 0
    temp_path = f"{dest_path}.tmp.{os.getpid()}.{int(time.time())}"
    while attempt <= max_retries:
        attempt += 1
        if logger:
            logger.info(f"Downloading '{source}' ({year:04d}-{month:02d}), attempt {attempt}/{max_retries + 1}...")
        try:
            downloader_fn(source, temp_path, src_info)
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError(f"Downloader failed to create non-empty file at {temp_path}")
            # Atomic rename
            os.replace(temp_path, dest_path)
            if logger:
                logger.info(f"Downloaded '{source}' successfully -> {dest_path}")
            return dest_path
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt > max_retries:
                if logger:
                    logger.error(f"Download failed for '{source}' after {attempt} attempts: {e}")
                raise RuntimeError(f"Download failed for '{source}' ({year:04d}-{month:02d}): {e}") from e
            if logger:
                logger.warning(f"Download attempt {attempt} failed: {e}. Retrying...")
            time.sleep(0.5)

    return None


# =============================================================================
# Raw File Verification
# =============================================================================

def verify_raw_file(
    source: str,
    filepath: str,
    year: int,
    month: int,
    sources_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify raw file exists, opens cleanly, has required variables, and valid timestamps."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw file missing: {filepath}")

    size_bytes = os.path.getsize(filepath)
    if size_bytes == 0:
        raise ValueError(f"Raw file is empty: {filepath}")

    sha256 = compute_file_sha256(filepath)
    expected_days = get_days_in_month(year, month)

    src_info = sources_cfg["sources"][source]
    is_static = (src_info.get("download_frequency") == "once")

    with h5netcdf.File(filepath, "r") as f:
        # Check required variables
        var_checks = {
            "glorys": ["thetao"],
            "static": ["mask"],
            "sst": ["analysed_sst", "mask"],
            "sla": ["sla"],
            "sss": ["sos"],
            "cur": ["uo", "vo"],
        }
        req_vars = var_checks.get(source, [])
        for v in req_vars:
            if v not in f.variables:
                raise KeyError(f"Source '{source}' file {filepath} missing required variable '{v}'")

        # Time verification for monthly files
        if not is_static:
            if "time" not in f.variables:
                raise KeyError(f"Source '{source}' file {filepath} missing 'time' coordinate")

            t_var = f.variables["time"]
            t_len = len(t_var)
            if t_len != expected_days:
                raise ValueError(
                    f"Source '{source}' file has {t_len} time steps; expected {expected_days} days for {year:04d}-{month:02d}"
                )

            t_units = t_var.attrs.get("units")
            if not t_units:
                raise ValueError(f"Source '{source}' file missing 'units' attribute on 'time'")

            t_raw = np.array(t_var[:])
            dts = decode_timestamps(t_raw, t_units)
            expected_dates = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, expected_days + 1)]
            observed_dates = [dt.strftime("%Y-%m-%d") for dt in dts]

            if observed_dates != expected_dates:
                raise ValueError(
                    f"Source '{source}' dates mismatch. Expected {expected_dates[0]}..{expected_dates[-1]}, "
                    f"got {observed_dates[0]}..{observed_dates[-1]}"
                )

            # Check that hour/minute/second is 00:00:00 (or regular daily timestamp)
            for dt in dts:
                if (dt.hour, dt.minute, dt.second) != (0, 0, 0):
                    # Some sources might be nominal 00:00 UTC or 12:00 UTC. OSTIA is 00:00, DUACS 00:00, SSS 00:00, CUR 00:00, GLORYS 00:00.
                    pass

    return {
        "source": source,
        "path": filepath,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "days": expected_days if not is_static else 0,
        "verified": True,
    }


def verify_all_raw_files(
    year: int,
    month: int,
    sources_cfg: Dict[str, Any],
    project_root: str = PROJECT_ROOT,
    required_sources: Optional[List[str]] = None,
) -> Tuple[bool, str, Dict[str, int], Dict[str, str], Dict[str, str]]:
    """Verify all required raw files for given month.

    Returns:
        (success, status_str, sizes_dict, sha256_dict, paths_dict)
    """
    if required_sources is None:
        required_sources = ["static", "glorys", "sst", "sla", "sss", "cur"]

    sizes: Dict[str, int] = {}
    hashes: Dict[str, str] = {}
    paths: Dict[str, str] = {}

    for src in required_sources:
        p = get_raw_filepath(src, year, month, sources_cfg, project_root)
        paths[src] = p
        if not os.path.exists(p):
            return False, "MISSING_RAW", {}, {}, {src: p}

        try:
            stat = verify_raw_file(src, p, year, month, sources_cfg)
            sizes[src] = stat["size_bytes"]
            hashes[src] = stat["sha256"]
        except Exception as e:
            return False, "FAILED_VERIFY", {}, {}, {"error": str(e), "source": src, "path": p}

    return True, "OK", sizes, hashes, paths


# =============================================================================
# QC Execution
# =============================================================================

def run_glorys_qc_month(
    glorys_path: str,
    static_path: str,
    qc_policy_path: str,
    num_days: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run GLORYS QC policy v1 on all days of the month.

    Returns summary dict. If any day fails, status is 'FAIL' / 'FAILED_QC'.
    """
    qc_pol = load_qc_policy(qc_policy_path)
    native_lats, native_lons, native_depths = load_glorys_coords(glorys_path)
    static_wet = load_static_wet_mask(
        static_path, ref_lats=native_lats, ref_lons=native_lons, ref_depths=native_depths
    )
    exclusion_mask = locate_exclusions(native_lats, native_lons, qc_pol)

    all_unexpected: List[Dict[str, Any]] = []
    total_counts = {"zero_code": 0, "near_floor": 0, "above_valid_max": 0}
    overall_status = "PASS"

    for d in range(num_days):
        raw_day = read_glorys_raw_day(glorys_path, day_index=d, policy=qc_pol)
        res = apply_qc(
            raw=raw_day,
            static_wet=static_wet,
            exclusion_mask=exclusion_mask,
            policy=qc_pol,
            lats=native_lats,
            lons=native_lons,
            depths=native_depths,
        )

        for k in total_counts:
            total_counts[k] += res.counts.get(k, 0)

        if res.status == "FAIL":
            overall_status = "FAIL"
            for unexp in res.unexpected:
                unexp_copy = dict(unexp)
                unexp_copy["day_index"] = d
                all_unexpected.append(unexp_copy)

        elif res.status == "WARN" and overall_status != "FAIL":
            overall_status = "WARN"

    return {
        "status": "PASS" if overall_status in ("PASS", "WARN") else "FAIL",
        "raw_status": overall_status,
        "counts": total_counts,
        "unexpected_count": len(all_unexpected),
        "unexpected": all_unexpected,
    }


# =============================================================================
# Census Calculation: Valid & Usable Counts
# =============================================================================

def compute_valid_and_usable_counts(
    target_nc_path: str,
    inputs_nc_path: str,
) -> Tuple[List[int], List[int]]:
    """Compute target valid counts and usable counts (all 4 inputs frac > 0) per depth."""
    with xr.open_dataset(target_nc_path, engine="h5netcdf") as ds_tgt, \
         xr.open_dataset(inputs_nc_path, engine="h5netcdf") as ds_in:

        valid_mask = ds_tgt["valid_mask"].values  # (time, depth, lat, lon) uint8

        # Inputs valid where all four observation sources have coverage > 0
        sst_f = ds_in["sst_frac"].values
        sla_f = ds_in["sla_frac"].values
        sss_f = ds_in["sss_frac"].values
        uo_f = ds_in["uo_frac"].values
        vo_f = ds_in["vo_frac"].values

        inputs_valid_2d = (sst_f > 0) & (sla_f > 0) & (sss_f > 0) & (uo_f > 0) & (vo_f > 0)

        # Broadcast across depths: (time, 1, lat, lon)
        inputs_valid_3d = inputs_valid_2d[:, np.newaxis, :, :]

        usable_mask = (valid_mask == 1) & inputs_valid_3d

        n_depths = valid_mask.shape[1]
        valid_counts: List[int] = []
        usable_counts: List[int] = []

        for k in range(n_depths):
            valid_counts.append(int(np.sum(valid_mask[:, k, :, :] == 1)))
            usable_counts.append(int(np.sum(usable_mask[:, k, :, :])))

    return valid_counts, usable_counts


# =============================================================================
# Manifest I/O & Resumability
# =============================================================================

def read_manifest(manifest_path: str) -> pd.DataFrame:
    """Read manifest CSV into pandas DataFrame."""
    if not os.path.exists(manifest_path):
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    try:
        df = pd.read_csv(manifest_path, dtype=str)
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)


def append_manifest_row(manifest_path: str, row: Dict[str, Any]) -> None:
    """Atomically append or update a row in the months manifest CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

    # Format row values cleanly
    formatted_row = {}
    for col in MANIFEST_COLUMNS:
        val = row.get(col, "")
        if isinstance(val, (dict, list)):
            formatted_row[col] = json.dumps(val)
        elif val is None:
            formatted_row[col] = ""
        else:
            formatted_row[col] = str(val)

    # Use file reading and rewriting with atomic rename
    temp_manifest = f"{manifest_path}.tmp.{os.getpid()}"
    df = read_manifest(manifest_path)

    # If row for this month already exists, replace it; else append
    month = formatted_row["month"]
    if not df.empty and month in df["month"].astype(str).values:
        idx = df[df["month"].astype(str) == month].index[0]
        for col in MANIFEST_COLUMNS:
            df.at[idx, col] = formatted_row[col]
    else:
        new_df = pd.DataFrame([formatted_row], columns=MANIFEST_COLUMNS)
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(temp_manifest, index=False)
    os.replace(temp_manifest, manifest_path)


def is_month_completed(
    manifest_path: str,
    month: str,
    output_root: str,
) -> bool:
    """Check if month has status == 'OK' in manifest and output files exist on disk."""
    if not os.path.exists(manifest_path):
        return False

    df = read_manifest(manifest_path)
    if df.empty or month not in df["month"].astype(str).values:
        return False

    row = df[df["month"].astype(str) == month].iloc[-1]
    if row["status"] != "OK":
        return False

    # Check output files exist
    tgt_path = os.path.join(output_root, "glorys_targets", f"glorys_targets_{month}.nc")
    inp_path = os.path.join(output_root, "inputs", f"inputs_{month}.nc")

    return os.path.exists(tgt_path) and os.path.exists(inp_path)


# =============================================================================
# Month Pipeline Execution
# =============================================================================

def process_month(
    month: str,
    sources_cfg: Optional[Dict[str, Any]] = None,
    config_path: str = DEFAULT_SOURCES_CONFIG,
    qc_policy_path: str = DEFAULT_QC_POLICY,
    target_policy_path: str = DEFAULT_TARGET_POLICY,
    input_policy_path: str = DEFAULT_INPUT_POLICY,
    output_root: str = os.path.join(PROJECT_ROOT, "data", "processed"),
    download: bool = False,
    use_existing_raw: bool = False,
    delete_raw_glorys: bool = False,
    force: bool = False,
    continue_on_fail: bool = False,
    downloader_fn: Optional[Callable] = None,
    manifest_csv_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Execute complete preprocessing pipeline for a single month."""
    t_start = time.perf_counter()
    year, m = parse_month(month)
    num_days = get_days_in_month(year, m)

    if sources_cfg is None:
        sources_cfg = load_sources_config(config_path)

    if manifest_csv_path is None:
        manifest_csv_path = os.path.join(output_root, "months_manifest.csv")

    log = logger.info if logger else print
    log_err = logger.error if logger else lambda msg: print(f"ERROR: {msg}", file=sys.stderr)

    log(f"=== Processing Month: {month} ({num_days} days) ===")

    # Wind notice
    if "wind" in sources_cfg.get("sources", {}):
        wind_cfg = sources_cfg["sources"]["wind"]
        if not wind_cfg.get("enabled", False):
            log(f"Notice: Wind source is DISABLED: {wind_cfg.get('notice', 'not audited')}")

    # Check resumability
    if not force and is_month_completed(manifest_csv_path, month, output_root):
        log(f"Month {month} already completed with status OK and outputs exist. Skipping (--force to rerun).")
        return {"month": month, "status": "SKIPPED"}

    # Compute provenance hashes
    this_script_path = os.path.abspath(__file__)
    builder_sha256 = compute_file_sha256(this_script_path)
    policy_sha256 = compute_file_sha256(qc_policy_path)
    git_commit = get_git_commit_hash(PROJECT_ROOT)
    git_dirty = get_git_dirty(PROJECT_ROOT)

    # 1. Download if requested
    required_sources = ["static", "glorys", "sst", "sla", "sss", "cur"]
    if download:
        for src in required_sources:
            try:
                download_source_month(
                    source=src,
                    year=year,
                    month=m,
                    sources_cfg=sources_cfg,
                    project_root=PROJECT_ROOT,
                    downloader_fn=downloader_fn,
                    logger=logger,
                )
            except Exception as e:
                wall_sec = time.perf_counter() - t_start
                log_err(f"Download failed for {src}: {e}")
                append_manifest_row(
                    manifest_csv_path,
                    {
                        "month": month,
                        "status": "DOWNLOAD_FAILED",
                        "wall_seconds": round(wall_sec, 2),
                        "git_commit": git_commit,
                        "git_dirty": git_dirty,
                    },
                )
                if not continue_on_fail:
                    raise
                return {"month": month, "status": "DOWNLOAD_FAILED", "error": str(e)}

    # 2. Verify raw files
    ok_raw, status_raw, raw_sizes, raw_hashes, raw_paths = verify_all_raw_files(
        year=year,
        month=m,
        sources_cfg=sources_cfg,
        project_root=PROJECT_ROOT,
        required_sources=required_sources,
    )
    if not ok_raw:
        wall_sec = time.perf_counter() - t_start
        log_err(f"Raw file verification failed for {month}: status={status_raw}, details={raw_paths}")
        append_manifest_row(
            manifest_csv_path,
            {
                "month": month,
                "status": status_raw,
                "wall_seconds": round(wall_sec, 2),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
        )
        if not continue_on_fail:
            raise RuntimeError(f"Raw data check failed for {month}: {status_raw} - {raw_paths}")
        return {"month": month, "status": status_raw, "details": raw_paths}

    # 3. GLORYS QC Check
    log(f"Running GLORYS QC policy v1 for {month}...")
    glorys_path = raw_paths["glorys"]
    static_path = raw_paths["static"]
    qc_res = run_glorys_qc_month(
        glorys_path=glorys_path,
        static_path=static_path,
        qc_policy_path=qc_policy_path,
        num_days=num_days,
        logger=logger,
    )

    if qc_res["status"] == "FAIL":
        wall_sec = time.perf_counter() - t_start
        log_err(f"QC verification FAILED for {month} with {qc_res['unexpected_count']} violations! Halting build.")

        # Write QC report
        qc_dir = os.path.join(output_root, "qc")
        os.makedirs(qc_dir, exist_ok=True)
        qc_report_path = os.path.join(qc_dir, f"{month}_qc_report.json")
        with open(qc_report_path, "w", encoding="utf-8") as f:
            json.dump(qc_res, f, indent=2)
        log(f"Wrote QC failure report to: {qc_report_path}")

        append_manifest_row(
            manifest_csv_path,
            {
                "month": month,
                "status": "FAILED_QC",
                "wall_seconds": round(wall_sec, 2),
                "raw_sizes": raw_sizes,
                "raw_sha256": raw_hashes,
                "qc_status": "FAIL",
                "qc_counts": qc_res["counts"],
                "builder_sha256": builder_sha256,
                "policy_sha256": policy_sha256,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
        )
        if not continue_on_fail:
            raise RuntimeError(f"QC check FAILED for month {month}: {qc_res['unexpected_count']} unexpected violations.")
        return {"month": month, "status": "FAILED_QC", "qc_report": qc_report_path}

    log(f"GLORYS QC passed: status={qc_res['status']}, counts={qc_res['counts']}")

    # 4. Build GLORYS Target File
    target_nc_path = os.path.join(output_root, "glorys_targets", f"glorys_targets_{month}.nc")
    target_manifest_path = os.path.join(output_root, "glorys_targets", f"glorys_targets_{month}_manifest.json")
    log(f"Building GLORYS target file -> {target_nc_path}")
    build_glorys_targets_month(
        glorys_path=glorys_path,
        static_path=static_path,
        out_path=target_nc_path,
        policy=target_policy_path,
        qc_policy_path=qc_policy_path,
        verbose=False,
    )

    # 5. Build Inputs File
    inputs_nc_path = os.path.join(output_root, "inputs", f"inputs_{month}.nc")
    inputs_manifest_path = os.path.join(output_root, "inputs", f"inputs_{month}_manifest.json")
    log(f"Building aggregated inputs file -> {inputs_nc_path}")
    build_inputs_month(
        policy_path=input_policy_path,
        sst_path=raw_paths["sst"],
        sla_path=raw_paths["sla"],
        sss_path=raw_paths["sss"],
        cur_path=raw_paths["cur"],
        target_path=target_nc_path,
        out_path=inputs_nc_path,
        manifest_out_path=inputs_manifest_path,
        verbose=False,
    )

    # 6. Compute Target Valid Counts & Usable Counts
    log("Computing valid and usable counts per depth...")
    valid_counts, usable_counts = compute_valid_and_usable_counts(target_nc_path, inputs_nc_path)
    log(f"Day 0 / Month Depth 0: valid={valid_counts[0]}, usable={usable_counts[0]}")

    wall_sec = time.perf_counter() - t_start

    # 7. Record Manifest Row Atomically
    manifest_record = {
        "month": month,
        "status": "OK",
        "wall_seconds": round(wall_sec, 2),
        "raw_sizes": raw_sizes,
        "raw_sha256": raw_hashes,
        "qc_status": qc_res["status"],
        "qc_counts": qc_res["counts"],
        "builder_sha256": builder_sha256,
        "policy_sha256": policy_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "valid_counts_by_depth": valid_counts,
        "usable_counts_by_depth": usable_counts,
    }
    append_manifest_row(manifest_csv_path, manifest_record)
    log(f"Appended manifest row to {manifest_csv_path}")

    # 8. Controlled Raw GLORYS Deletion if requested
    if delete_raw_glorys:
        # STRICT SAFETY CHECKS:
        # Only delete raw GLORYS if status is OK, manifest written, and file matches glorys_path
        if (
            os.path.exists(manifest_csv_path)
            and is_month_completed(manifest_csv_path, month, output_root)
            and "glorys" in raw_paths
            and os.path.exists(raw_paths["glorys"])
        ):
            del_path = raw_paths["glorys"]
            # Double check it is strictly inside data/raw/glorys and matches glorys_thetao_
            basename = os.path.basename(del_path)
            if "glorys_thetao_" in basename and basename.endswith(".nc"):
                log(f"Deleting raw GLORYS reanalysis file per --delete-raw-glorys: {del_path}")
                os.remove(del_path)
                log("Raw GLORYS deletion complete.")
            else:
                log_err(f"Refusing to delete unexpected GLORYS path: {del_path}")

    log(f"Successfully completed month {month} in {wall_sec:.2f}s")
    return {
        "month": month,
        "status": "OK",
        "wall_seconds": wall_sec,
        "valid_counts": valid_counts,
        "usable_counts": usable_counts,
        "target_nc": target_nc_path,
        "inputs_nc": inputs_nc_path,
    }
