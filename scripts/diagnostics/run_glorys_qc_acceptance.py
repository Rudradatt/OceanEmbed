#!/usr/bin/env python3
"""
GLORYS QC Policy v1 Acceptance Runner
(Read-Only Forensic Acceptance Diagnostic)

Executes acceptance validation of GLORYS QC Policy v1 across:
1. 31 daily slices of January 2019 (glorys_thetao_2019-01.nc)
2. 28 annual snapshots 1993-2020 (glorys_thetao_<YYYY>-07-01.nc)
3. In-memory fault injection on a real daily slice (4 test vectors: zero code, near floor, above valid max, dry finite)
4. Bit-for-bit file integrity verification (SHA256, size, mtime before and after)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5netcdf
import numpy as np
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.preprocessing.glorys_qc import (
    load_policy,
    read_raw_day,
    load_coordinates,
    load_static_wet_mask,
    locate_exclusions,
    apply_qc,
    QCResult,
)


# =============================================================================
# Helper: File Integrity Check
# =============================================================================

def compute_file_metadata(filepath: str) -> Dict[str, Any]:
    """Compute SHA256, size (bytes), and UTC mtime for a file without modifying it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    stat = os.stat(filepath)
    mtime_dt = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)

    return {
        "filepath": os.path.abspath(filepath),
        "filename": os.path.basename(filepath),
        "sha256": sha256.hexdigest().lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": mtime_dt.isoformat(),
    }


def verify_integrity_unchanged(
    before_meta: Dict[str, Dict[str, Any]],
    after_meta: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Verify that all files have identical SHA256, size, and mtime before and after."""
    mismatches = []
    for path, b_info in before_meta.items():
        if path not in after_meta:
            mismatches.append(f"Missing in after_meta: {path}")
            continue
        a_info = after_meta[path]
        if b_info["sha256"] != a_info["sha256"]:
            mismatches.append(f"SHA256 changed for {path}: {b_info['sha256']} -> {a_info['sha256']}")
        if b_info["size_bytes"] != a_info["size_bytes"]:
            mismatches.append(f"Size changed for {path}: {b_info['size_bytes']} -> {a_info['size_bytes']}")
        if b_info["mtime_utc"] != a_info["mtime_utc"]:
            mismatches.append(f"mtime changed for {path}: {b_info['mtime_utc']} -> {a_info['mtime_utc']}")

    return (len(mismatches) == 0, mismatches)


# =============================================================================
# Main Acceptance Runner
# =============================================================================

def run_acceptance(
    policy_path: str = "configs/glorys_qc_policy_v1.json",
    jan2019_file: str = "data/raw/glorys/glorys_thetao_2019-01.nc",
    snapshot_dir: str = "data/raw/glorys_snapshots",
    static_file: str = "data/raw/glorys_static/glorys_static_mask_bathy.nc",
    output_dir: str = "outputs/diagnostics/glorys_qc_acceptance",
) -> Dict[str, Any]:
    """Execute complete QC policy v1 acceptance run."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GLORYS Quality-Control Policy v1 Acceptance Validation")
    print(f"Timestamp UTC: {start_time.isoformat()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 0: Integrity Check Before Analysis
    # -------------------------------------------------------------------------
    print("\n--- Integrity Check: Pre-Analysis Metadata ---")
    years = list(range(1993, 2021))
    files_to_track = [os.path.abspath(jan2019_file), os.path.abspath(static_file)]
    for y in years:
        files_to_track.append(os.path.abspath(os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")))

    before_metadata = {}
    for fpath in files_to_track:
        meta = compute_file_metadata(fpath)
        before_metadata[fpath] = meta
        print(f"  {meta['filename']}: sha256={meta['sha256'][:16]}... size={meta['size_bytes']}")

    # -------------------------------------------------------------------------
    # Step 1: Load Policy & Static Mask
    # -------------------------------------------------------------------------
    print("\n--- Loading Policy & Static Wet Mask ---")
    policy = load_policy(policy_path)
    print(f"Loaded policy version: {policy['policy_version']}")

    lats, lons, depths = load_coordinates(jan2019_file)
    print(f"Grid dimensions: nz={len(depths)}, ny={len(lats)}, nx={len(lons)}")

    levels_used = policy.get("levels_used", 36)
    static_wet = load_static_wet_mask(
        static_file,
        ref_lats=lats,
        ref_lons=lons,
        ref_depths=depths,
        levels_used=levels_used,
    )
    print(f"Static wet mask loaded (levels 0..{levels_used - 1}): {np.sum(static_wet):,} ocean cells")

    exclusion_mask = locate_exclusions(lats, lons, policy)
    print(f"Exclusion mask located: {np.sum(exclusion_mask)} excluded cells across Columns A & B")

    # Reference finite mask: January 2019 day 0 finite mask
    print("Loading baseline reference finite mask (January 2019 Day 0)...")
    raw_day0 = read_raw_day(jan2019_file, day_index=0, policy=policy)
    fill_raw = int(policy["encoding"]["fill_raw"])
    ref_finite = (raw_day0 != fill_raw)
    print(f"Baseline finite mask: {np.sum(ref_finite):,} finite cells")

    # -------------------------------------------------------------------------
    # Step 2: Evaluation on January 2019 (31 Days) and Snapshots (28 Years)
    # -------------------------------------------------------------------------
    print("\n--- Running QC Policy on 31 Days of Jan 2019 & 28 July-1 Snapshots ---")

    by_file_rows = []
    checks_rows = []
    extremes_rows = []

    check_counter = 0

    def _record_check(file_type: str, identifier: str, date_str: str, metric: str, observed: Any, expected: Any):
        nonlocal check_counter
        check_counter += 1
        is_pass = (observed == expected)
        result_str = "PASS" if is_pass else "FAIL"
        checks_rows.append({
            "check_id": check_counter,
            "file_type": file_type,
            "identifier": identifier,
            "date": date_str,
            "metric": metric,
            "observed_value": observed,
            "expected_value": expected,
            "result": result_str,
        })
        if not is_pass:
            print(f"  [FAIL] {file_type} {identifier} ({date_str}): {metric} observed={observed}, expected={expected}")
        return is_pass

    # A. 31 Days of January 2019
    with h5netcdf.File(jan2019_file, mode="r") as fj:
        n_days = fj.variables["thetao"].shape[0]

    for d in range(n_days):
        date_str = f"2019-01-{d + 1:02d}"
        ident = f"day_{d:02d}"
        raw_d = read_raw_day(jan2019_file, day_index=d, policy=policy)

        res = apply_qc(
            raw=raw_d,
            static_wet=static_wet,
            exclusion_mask=exclusion_mask,
            policy=policy,
            reference_finite=ref_finite,
            lats=lats,
            lons=lons,
            depths=depths,
        )

        # Expected values for January 2019:
        # fill_in_static_wet: 129
        # excluded_cells: 6
        # zero_code: 4 (all inside exclusion list, unexcluded = 0)
        # near_floor: 0
        # above_valid_max: 0
        # finite_in_static_dry: 0
        # mask differences: 0
        # valid cells: 3,437,651
        # status: PASS
        # unexpected: empty
        c1 = _record_check("jan2019", ident, date_str, "fill_in_static_wet", res.counts["fill_in_static_wet"], 129)
        c2 = _record_check("jan2019", ident, date_str, "excluded_cells", int(np.sum(exclusion_mask)), 6)
        c3 = _record_check("jan2019", ident, date_str, "zero_code", res.counts["zero_code"], 4)
        c4 = _record_check("jan2019", ident, date_str, "zero_code_unexcluded", res.counts["zero_code_unexcluded"], 0)
        c5 = _record_check("jan2019", ident, date_str, "near_floor", res.counts["near_floor"], 0)
        c6 = _record_check("jan2019", ident, date_str, "above_valid_max", res.counts["above_valid_max"], 0)
        c7 = _record_check("jan2019", ident, date_str, "finite_in_static_dry", res.counts["finite_in_static_dry"], 0)
        c8 = _record_check("jan2019", ident, date_str, "mask_differences", res.counts["mask_differs_from_reference"], 0)
        c9 = _record_check("jan2019", ident, date_str, "valid_cells", res.counts["valid_cells"], 3437651)
        c10 = _record_check("jan2019", ident, date_str, "unexpected_count", len(res.unexpected), 0)
        c11 = _record_check("jan2019", ident, date_str, "status", res.status, "PASS")

        all_day_pass = all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

        by_file_rows.append({
            "file_type": "jan2019_day",
            "identifier": ident,
            "date": date_str,
            "fill_in_static_wet": res.counts["fill_in_static_wet"],
            "excluded_cells": int(np.sum(exclusion_mask)),
            "zero_code": res.counts["zero_code"],
            "zero_code_unexcluded": res.counts["zero_code_unexcluded"],
            "near_floor": res.counts["near_floor"],
            "above_valid_max": res.counts["above_valid_max"],
            "finite_in_static_dry": res.counts["finite_in_static_dry"],
            "mask_differences": res.counts["mask_differs_from_reference"],
            "valid_cells": res.counts["valid_cells"],
            "unexpected_count": len(res.unexpected),
            "status": res.status,
            "expected_status": "PASS",
            "all_checks_passed": all_day_pass,
        })

        for ext in res.extremes_by_level:
            extremes_rows.append({
                "file_type": "jan2019_day",
                "identifier": ident,
                "date": date_str,
                "level": ext["level"],
                "depth_m": ext["depth_m"],
                "n_valid": ext["n_valid"],
                "max_raw": ext["max_raw"],
                "max_decoded_C": ext["max_decoded_C"],
                "max_lat_idx": ext["max_lat_idx"],
                "max_lon_idx": ext["max_lon_idx"],
                "max_latitude": ext["max_latitude"],
                "max_longitude": ext["max_longitude"],
                "min_raw": ext["min_raw"],
                "min_decoded_C": ext["min_decoded_C"],
                "min_lat_idx": ext["min_lat_idx"],
                "min_lon_idx": ext["min_lon_idx"],
                "min_latitude": ext["min_latitude"],
                "min_longitude": ext["min_longitude"],
            })

    # B. 28 July-1 Snapshots
    for y in years:
        snap_path = os.path.join(snapshot_dir, f"glorys_thetao_{y}-07-01.nc")
        date_str = f"{y}-07-01"
        ident = str(y)

        raw_y = read_raw_day(snap_path, day_index=0, policy=policy)

        res = apply_qc(
            raw=raw_y,
            static_wet=static_wet,
            exclusion_mask=exclusion_mask,
            policy=policy,
            reference_finite=ref_finite,
            lats=lats,
            lons=lons,
            depths=depths,
        )

        expected_zc = 0 if y < 2003 else 4
        expected_above_vmax = 8 if y == 1997 else 0
        expected_status = "WARN" if y == 1997 else "PASS"

        c1 = _record_check("snapshot", ident, date_str, "fill_in_static_wet", res.counts["fill_in_static_wet"], 129)
        c2 = _record_check("snapshot", ident, date_str, "excluded_cells", int(np.sum(exclusion_mask)), 6)
        c3 = _record_check("snapshot", ident, date_str, "zero_code", res.counts["zero_code"], expected_zc)
        c4 = _record_check("snapshot", ident, date_str, "zero_code_unexcluded", res.counts["zero_code_unexcluded"], 0)
        c5 = _record_check("snapshot", ident, date_str, "near_floor", res.counts["near_floor"], 0)
        c6 = _record_check("snapshot", ident, date_str, "above_valid_max", res.counts["above_valid_max"], expected_above_vmax)
        c7 = _record_check("snapshot", ident, date_str, "finite_in_static_dry", res.counts["finite_in_static_dry"], 0)
        c8 = _record_check("snapshot", ident, date_str, "mask_differences", res.counts["mask_differs_from_reference"], 0)
        c9 = _record_check("snapshot", ident, date_str, "valid_cells", res.counts["valid_cells"], 3437651)
        c10 = _record_check("snapshot", ident, date_str, "unexpected_count", len(res.unexpected), 0)
        c11 = _record_check("snapshot", ident, date_str, "status", res.status, expected_status)

        all_snap_pass = all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

        by_file_rows.append({
            "file_type": "snapshot",
            "identifier": ident,
            "date": date_str,
            "fill_in_static_wet": res.counts["fill_in_static_wet"],
            "excluded_cells": int(np.sum(exclusion_mask)),
            "zero_code": res.counts["zero_code"],
            "zero_code_unexcluded": res.counts["zero_code_unexcluded"],
            "near_floor": res.counts["near_floor"],
            "above_valid_max": res.counts["above_valid_max"],
            "finite_in_static_dry": res.counts["finite_in_static_dry"],
            "mask_differences": res.counts["mask_differs_from_reference"],
            "valid_cells": res.counts["valid_cells"],
            "unexpected_count": len(res.unexpected),
            "status": res.status,
            "expected_status": expected_status,
            "all_checks_passed": all_snap_pass,
        })

        for ext in res.extremes_by_level:
            extremes_rows.append({
                "file_type": "snapshot",
                "identifier": ident,
                "date": date_str,
                "level": ext["level"],
                "depth_m": ext["depth_m"],
                "n_valid": ext["n_valid"],
                "max_raw": ext["max_raw"],
                "max_decoded_C": ext["max_decoded_C"],
                "max_lat_idx": ext["max_lat_idx"],
                "max_lon_idx": ext["max_lon_idx"],
                "max_latitude": ext["max_latitude"],
                "max_longitude": ext["max_longitude"],
                "min_raw": ext["min_raw"],
                "min_decoded_C": ext["min_decoded_C"],
                "min_lat_idx": ext["min_lat_idx"],
                "min_lon_idx": ext["min_lon_idx"],
                "min_latitude": ext["min_latitude"],
                "min_longitude": ext["min_longitude"],
            })

    # -------------------------------------------------------------------------
    # Step 3: In-Memory Fault Injection on Real Day 0
    # -------------------------------------------------------------------------
    print("\n--- Running In-Memory Fault Injection on January 2019 Day 0 ---")
    fault_results = {}

    # Fault (a): Set a non-excluded cell to raw -28671 -> expect FAIL listing it
    raw_fa = raw_day0.copy()
    # Pick a finite, non-excluded ocean cell (lvl=10, r=100, c=100)
    raw_fa[10, 100, 100] = -28671
    res_fa = apply_qc(raw_fa, static_wet, exclusion_mask, policy, reference_finite=ref_finite, lats=lats, lons=lons, depths=depths)
    pass_fa = (
        res_fa.status == "FAIL" and
        len(res_fa.unexpected) == 1 and
        res_fa.unexpected[0]["level"] == 10 and
        res_fa.unexpected[0]["raw"] == -28671 and
        res_fa.unexpected[0]["rule"] == "ZERO_CODE_OUTSIDE_EXCLUSION"
    )
    fault_results["fault_a_unexcluded_zero_code"] = {
        "description": "Set non-excluded cell (lvl=10, r=100, c=100) to raw -28671",
        "expected_status": "FAIL",
        "observed_status": res_fa.status,
        "unexpected_count": len(res_fa.unexpected),
        "listed_violation": res_fa.unexpected[0] if res_fa.unexpected else None,
        "outcome": "PASS" if pass_fa else "FAIL",
    }
    print(f"  Fault (a) unexcluded zero-code: status={res_fa.status}, unexpected={len(res_fa.unexpected)} -> {'PASS' if pass_fa else 'FAIL'}")

    # Fault (b): Set another non-excluded cell to raw -32766 (near floor) -> expect FAIL listing it
    raw_fb = raw_day0.copy()
    raw_fb[12, 110, 110] = -32766
    res_fb = apply_qc(raw_fb, static_wet, exclusion_mask, policy, reference_finite=ref_finite, lats=lats, lons=lons, depths=depths)
    pass_fb = (
        res_fb.status == "FAIL" and
        len(res_fb.unexpected) == 1 and
        res_fb.unexpected[0]["level"] == 12 and
        res_fb.unexpected[0]["raw"] == -32766 and
        res_fb.unexpected[0]["rule"] == "NEAR_FLOOR_OUTSIDE_EXCLUSION"
    )
    fault_results["fault_b_near_floor"] = {
        "description": "Set non-excluded cell (lvl=12, r=110, c=110) to raw -32766",
        "expected_status": "FAIL",
        "observed_status": res_fb.status,
        "unexpected_count": len(res_fb.unexpected),
        "listed_violation": res_fb.unexpected[0] if res_fb.unexpected else None,
        "outcome": "PASS" if pass_fb else "FAIL",
    }
    print(f"  Fault (b) near-floor raw=-32766: status={res_fb.status}, unexpected={len(res_fb.unexpected)} -> {'PASS' if pass_fb else 'FAIL'}")

    # Fault (c): Set another cell to raw 21307 -> expect WARN and valid == True
    raw_fc = raw_day0.copy()
    raw_fc[14, 120, 120] = 21307
    res_fc = apply_qc(raw_fc, static_wet, exclusion_mask, policy, reference_finite=ref_finite, lats=lats, lons=lons, depths=depths)
    pass_fc = (
        res_fc.status == "WARN" and
        len(res_fc.unexpected) == 0 and
        res_fc.counts["above_valid_max"] == 1 and
        bool(res_fc.valid[14, 120, 120]) is True
    )
    fault_results["fault_c_above_valid_max"] = {
        "description": "Set cell (lvl=14, r=120, c=120) to raw 21307",
        "expected_status": "WARN",
        "observed_status": res_fc.status,
        "unexpected_count": len(res_fc.unexpected),
        "cell_is_valid": bool(res_fc.valid[14, 120, 120]),
        "outcome": "PASS" if pass_fc else "FAIL",
    }
    print(f"  Fault (c) above valid_max raw=21307: status={res_fc.status}, cell_valid={res_fc.valid[14, 120, 120]} -> {'PASS' if pass_fc else 'FAIL'}")

    # Fault (d): Make one static-dry cell finite -> expect FAIL listing it
    raw_fd = raw_day0.copy()
    # Find a land/dry cell at level 0 (static_wet == False)
    dry_coords = np.where(~static_wet[0])
    dry_r = int(dry_coords[0][0])
    dry_c = int(dry_coords[1][0])
    raw_fd[0, dry_r, dry_c] = 15000
    res_fd = apply_qc(raw_fd, static_wet, exclusion_mask, policy, reference_finite=ref_finite, lats=lats, lons=lons, depths=depths)
    pass_fd = (
        res_fd.status == "FAIL" and
        any(u["rule"] == "FINITE_IN_STATIC_DRY" and u["level"] == 0 and u["lat_idx"] == dry_r and u["lon_idx"] == dry_c for u in res_fd.unexpected)
    )
    fault_results["fault_d_finite_in_dry"] = {
        "description": f"Set static-dry cell (lvl=0, r={dry_r}, c={dry_c}) to finite raw 15000",
        "expected_status": "FAIL",
        "observed_status": res_fd.status,
        "unexpected_count": len(res_fd.unexpected),
        "listed_violation": [u for u in res_fd.unexpected if u["rule"] == "FINITE_IN_STATIC_DRY"][0] if res_fd.unexpected else None,
        "outcome": "PASS" if pass_fd else "FAIL",
    }
    print(f"  Fault (d) finite in static dry: status={res_fd.status}, unexpected={len(res_fd.unexpected)} -> {'PASS' if pass_fd else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Step 4: Save Acceptance Outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving Output CSVs and Artifacts ---")
    df_by_file = pd.DataFrame(by_file_rows)
    by_file_csv = os.path.join(output_dir, "qc_acceptance_by_file.csv")
    df_by_file.to_csv(by_file_csv, index=False)
    print(f"Saved: {by_file_csv} ({len(df_by_file)} rows)")

    df_checks = pd.DataFrame(checks_rows)
    checks_csv = os.path.join(output_dir, "qc_acceptance_checks.csv")
    df_checks.to_csv(checks_csv, index=False)
    print(f"Saved: {checks_csv} ({len(df_checks)} checks)")

    df_extremes = pd.DataFrame(extremes_rows)
    extremes_csv = os.path.join(output_dir, "qc_extremes_by_day_level.csv")
    df_extremes.to_csv(extremes_csv, index=False)
    print(f"Saved: {extremes_csv} ({len(df_extremes)} rows)")

    # Copy policy JSON into output directory
    policy_copy_path = os.path.join(output_dir, "glorys_qc_policy_v1.json")
    shutil.copyfile(policy_path, policy_copy_path)
    print(f"Saved copy of policy: {policy_copy_path}")

    # -------------------------------------------------------------------------
    # Step 5: Post-Analysis File Integrity Verification
    # -------------------------------------------------------------------------
    print("\n--- Integrity Check: Post-Analysis Verification ---")
    after_metadata = {}
    files_integrity_dict = {}
    for fpath in files_to_track:
        meta = compute_file_metadata(fpath)
        after_metadata[fpath] = meta
        b_meta = before_metadata[fpath]
        match = (
            meta["sha256"] == b_meta["sha256"] and
            meta["size_bytes"] == b_meta["size_bytes"] and
            meta["mtime_utc"] == b_meta["mtime_utc"]
        )
        files_integrity_dict[fpath] = {
            "filename": meta["filename"],
            "sha256_before": b_meta["sha256"],
            "sha256_after": meta["sha256"],
            "size_bytes": meta["size_bytes"],
            "mtime_before": b_meta["mtime_utc"],
            "mtime_after": meta["mtime_utc"],
            "match": match,
        }

    all_unchanged, mismatches = verify_integrity_unchanged(before_metadata, after_metadata)
    print(f"All {len(files_to_track)} input files unchanged: {all_unchanged}")
    if not all_unchanged:
        print(f"Mismatches detected: {mismatches}")

    # -------------------------------------------------------------------------
    # Step 6: Master JSON Summary
    # -------------------------------------------------------------------------
    total_checks = len(df_checks)
    failed_checks = int(np.sum(df_checks["result"] == "FAIL"))
    passed_checks = int(np.sum(df_checks["result"] == "PASS"))
    all_acceptance_passed = (failed_checks == 0)

    results_json = {
        "analysis_name": "glorys_qc_acceptance",
        "policy_version": policy.get("policy_version", "v1"),
        "timestamp_utc": start_time.isoformat(),
        "integrity": {
            "all_files_unchanged": all_unchanged,
            "total_files_checked": len(files_to_track),
            "files": files_integrity_dict,
        },
        "acceptance_summary": {
            "total_targets_evaluated": len(df_by_file),
            "jan2019_days_evaluated": n_days,
            "snapshots_evaluated": len(years),
            "total_checks_evaluated": total_checks,
            "total_checks_passed": passed_checks,
            "total_checks_failed": failed_checks,
            "all_acceptance_checks_passed": all_acceptance_passed,
        },
        "jan2019_acceptance": {
            "days_evaluated": n_days,
            "all_days_status_pass": bool((df_by_file[df_by_file["file_type"] == "jan2019_day"]["status"] == "PASS").all()),
            "fill_in_static_wet_constant": int(df_by_file[df_by_file["file_type"] == "jan2019_day"]["fill_in_static_wet"].iloc[0]),
            "excluded_cells_constant": int(df_by_file[df_by_file["file_type"] == "jan2019_day"]["excluded_cells"].iloc[0]),
            "zero_code_constant": int(df_by_file[df_by_file["file_type"] == "jan2019_day"]["zero_code"].iloc[0]),
            "valid_cells_constant": int(df_by_file[df_by_file["file_type"] == "jan2019_day"]["valid_cells"].iloc[0]),
            "unexpected_cells_all_days": int(df_by_file[df_by_file["file_type"] == "jan2019_day"]["unexpected_count"].sum()),
        },
        "snapshots_acceptance": {
            "snapshots_evaluated": len(years),
            "warn_years": [int(r["identifier"]) for _, r in df_by_file[(df_by_file["file_type"] == "snapshot") & (df_by_file["status"] == "WARN")].iterrows()],
            "pass_years": [int(r["identifier"]) for _, r in df_by_file[(df_by_file["file_type"] == "snapshot") & (df_by_file["status"] == "PASS")].iterrows()],
            "fail_years": [int(r["identifier"]) for _, r in df_by_file[(df_by_file["file_type"] == "snapshot") & (df_by_file["status"] == "FAIL")].iterrows()],
            "zero_code_1993_2002_all_zero": bool((df_by_file[df_by_file["file_type"] == "snapshot"].loc[lambda d: d["identifier"].astype(int) < 2003]["zero_code"] == 0).all()),
            "zero_code_2003_2020_all_four": bool((df_by_file[df_by_file["file_type"] == "snapshot"].loc[lambda d: d["identifier"].astype(int) >= 2003]["zero_code"] == 4).all()),
            "above_valid_max_1997_count": int(df_by_file[(df_by_file["file_type"] == "snapshot") & (df_by_file["identifier"] == "1997")]["above_valid_max"].iloc[0]),
            "valid_cells_constant": int(df_by_file[df_by_file["file_type"] == "snapshot"]["valid_cells"].iloc[0]),
            "unexpected_cells_all_snapshots": int(df_by_file[df_by_file["file_type"] == "snapshot"]["unexpected_count"].sum()),
        },
        "fault_injection_results": fault_results,
        "unknown_fields": {
            "physical_origin_of_pit_formation": None,
            "reason_for_physical_origin_unknown": "Columns A & B lock into an isolated 2-cell numerical pit starting in 2003. Identifying whether this stems from bathymetric discretization, steep slope advection, or lateral mixing parameterization requires NEMO source-level investigation; labeled INFERENCE.",
            "copernicus_valid_max_clamp_intention": None,
            "reason_for_valid_max_clamp_intention_unknown": "Copernicus declares valid_max = 21306 (36.605 C), but 1997 snapshot values reach 37.13 C in the Persian Gulf. Whether Copernicus intended valid_max as an absolute quality threshold or an open-ocean descriptive metadata value is unknown without producer documentation.",
        },
    }

    json_path = os.path.join(output_dir, "glorys_qc_acceptance.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(results_json, fj, indent=2)
    print(f"Saved master JSON summary: {json_path}")
    print("=" * 80)

    print(f"\nFinal Acceptance Status: {'ALL CHECKS PASSED' if all_acceptance_passed else 'SOME CHECKS FAILED'}")
    print(f"Total Checks: {total_checks}, Passed: {passed_checks}, Failed: {failed_checks}")

    return results_json


def main():
    parser = argparse.ArgumentParser(description="GLORYS QC Policy v1 Acceptance Runner")
    parser.add_argument("--policy", default="configs/glorys_qc_policy_v1.json", help="Path to QC policy JSON")
    parser.add_argument("--jan2019-file", default="data/raw/glorys/glorys_thetao_2019-01.nc", help="Path to Jan 2019 file")
    parser.add_argument("--snapshot-dir", default="data/raw/glorys_snapshots", help="Path to snapshot directory")
    parser.add_argument("--static-file", default="data/raw/glorys_static/glorys_static_mask_bathy.nc", help="Path to static mask/bathy file")
    parser.add_argument("--output-dir", default="outputs/diagnostics/glorys_qc_acceptance", help="Output directory")
    args = parser.parse_args()

    run_acceptance(
        policy_path=args.policy,
        jan2019_file=args.jan2019_file,
        snapshot_dir=args.snapshot_dir,
        static_file=args.static_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
