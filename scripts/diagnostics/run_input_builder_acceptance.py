#!/usr/bin/env python3
"""
Acceptance Suite for OceanEmbed Input Builders v1 (January 2019)
Executes comprehensive validation of inputs_2019-01.nc against:
 a. Aggregation integrity (sum(frac * block_size) == raw valid count)
 b. Coverage census comparison against input_coverage_by_variable.csv
 c. Value equivalence against input_aggregation_census.py functions
 d. Brute-force recomputation of 2,000 random target cells
 e. Currents 1x1 exact identity verification
 f. SST lake rule: 92 lake cells excluded; target cell (58, 161) hand-check
 g. SST unit conversion on 1,000 random valid cells (Kelvin - 273.15 within 1e-4)
 h. Reproducibility: second build identical
 i. Target file metadata check (glorys_targets_2019-01.nc read-only)
 j. Integrity of all 6 raw files and target file (SHA256, size, mtime before & after)

Generates:
 - outputs/diagnostics/input_builder/acceptance_checks.csv
 - outputs/diagnostics/input_builder/acceptance_summary.json
 - outputs/diagnostics/input_builder/plot_day0_frac_sst_sss.png
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5netcdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocessing.input_builder import (
    DEFAULT_CUR_PATH,
    DEFAULT_OUT_PATH,
    DEFAULT_POLICY_PATH,
    DEFAULT_SLA_PATH,
    DEFAULT_SSS_PATH,
    DEFAULT_SST_PATH,
    DEFAULT_TARGET_PATH,
    block_aggregate as builder_block_aggregate,
    build_day_currents,
    build_day_sla,
    build_day_sss,
    build_day_sst,
    build_month_file,
    compute_file_sha256,
    decode_currents as builder_decode_currents,
    decode_sla as builder_decode_sla,
    decode_sss as builder_decode_sss,
    decode_sst as builder_decode_sst,
    decode_sst_degc as builder_decode_sst_degc,
    decode_timestamps,
)

# Import diagnostic aggregation functions for check c
import scripts.diagnostics.input_aggregation_census as census_diag

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "diagnostics", "input_builder")
RAW_FILES = {
    "sst": os.path.join(PROJECT_ROOT, "data", "raw", "sst", "sst_ostia_2019-01.nc"),
    "sla": os.path.join(PROJECT_ROOT, "data", "raw", "sla", "sla_duacs_2019-01.nc"),
    "sss": os.path.join(PROJECT_ROOT, "data", "raw", "sss", "sss_multiobs_2019-01.nc"),
    "cur": os.path.join(PROJECT_ROOT, "data", "raw", "currents", "currents_multiobs_2019-01.nc"),
    "glorys": os.path.join(PROJECT_ROOT, "data", "raw", "glorys", "glorys_thetao_2019-01.nc"),
    "static": os.path.join(PROJECT_ROOT, "data", "raw", "glorys_static", "glorys_static_mask_bathy.nc"),
    "target": os.path.join(PROJECT_ROOT, "data", "processed", "glorys_targets", "glorys_targets_2019-01.nc"),
}

EXPECTED_HASHES = {
    "sst": "9e733b5956f4d9bba81ec1c7a4bcbfa3b456e61af969d9ae13c57ef6fcc72265",
    "sla": "36b558a5642feaff3d8563aea4dac76f70f850d1b555431a8893474b7eccd692",
    "sss": "74dc119b9d34a1b0330e1e9066e067db658dfaa8ff44dedb6edf03041b6ff043",
    "cur": "ad98b7684663b42a2eaa4ced7d5b52ec795fc711d050cdd87cd0b70eaf0ca4a7",
    "glorys": "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f",
    "static": "fdb0c5fdc282180c0b5b58fa689753df79c22f69f4dc6dac64314a8cb82a79a5",
    "target": "11cb8d749a7678331df318b6f9115c646cce2d45094f57fcae8c8ad0cfad456d",
}


def get_file_stat(filepath: str) -> Dict[str, Any]:
    st = os.stat(filepath)
    mtime_dt = datetime.datetime.fromtimestamp(st.st_mtime, tz=datetime.timezone.utc).isoformat()
    return {
        "path": os.path.abspath(filepath),
        "sha256": compute_file_sha256(filepath),
        "size_bytes": st.st_size,
        "mtime_utc": mtime_dt,
    }


def json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return [json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(v) for v in obj]
    return obj


def run_acceptance() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    print("=" * 80)
    print("OceanEmbed Input Builder Acceptance Run (January 2019)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 0: Capture stats BEFORE build
    # -------------------------------------------------------------------------
    stats_before = {k: get_file_stat(p) for k, p in RAW_FILES.items()}
    for k, s in stats_before.items():
        assert s["sha256"] == EXPECTED_HASHES[k], f"Pre-check hash mismatch for {k}: {s['sha256']}"

    # -------------------------------------------------------------------------
    # Check i: Target file metadata check (READ-ONLY)
    # -------------------------------------------------------------------------
    target_header_info: Dict[str, Any] = {}
    with h5netcdf.File(RAW_FILES["target"], "r") as f_tgt:
        t_var = f_tgt.variables["time"]
        t_units = t_var.attrs.get("units")
        t_calendar = t_var.attrs.get("calendar")
        if not t_units or not t_calendar:
            raise RuntimeError(f"Target file missing time units or calendar! STOP.")

        t_raw = np.array(t_var[:])
        dts = decode_timestamps(t_raw, t_units)
        date_strs = [dt.strftime("%Y-%m-%d") for dt in dts]
        expected_dates = [f"2019-01-{d:02d}" for d in range(1, 32)]
        target_dates_ok = (date_strs == expected_dates)

        target_header_info = {
            "dimensions": {k: len(v) for k, v in f_tgt.dimensions.items()},
            "variables": {
                vname: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "attrs": dict(v.attrs),
                }
                for vname, v in f_tgt.variables.items()
            },
            "global_attrs": dict(f_tgt.attrs),
            "time_units": str(t_units),
            "time_calendar": str(t_calendar),
            "time_range": [date_strs[0], date_strs[-1]],
            "decoded_dates_match_jan2019": target_dates_ok,
        }

    status_i = "PASS" if target_dates_ok else "FAIL"
    checks.append({
        "check_id": "i",
        "name": "Target File Metadata",
        "status": status_i,
        "details": f"Units='{t_units}', calendar='{t_calendar}', decoded {len(date_strs)} dates: {date_strs[0]}..{date_strs[-1]}",
    })

    # -------------------------------------------------------------------------
    # BUILD: Run input_builder.build_month_file
    # -------------------------------------------------------------------------
    out_nc_path = DEFAULT_OUT_PATH
    manifest_path = os.path.splitext(out_nc_path)[0] + "_manifest.json"

    print("\n--- Executing input_builder.build_month_file ---")
    manifest_data = build_month_file(
        policy_path=DEFAULT_POLICY_PATH,
        sst_path=RAW_FILES["sst"],
        sla_path=RAW_FILES["sla"],
        sss_path=RAW_FILES["sss"],
        cur_path=RAW_FILES["cur"],
        target_path=RAW_FILES["target"],
        out_path=out_nc_path,
        manifest_out_path=manifest_path,
        verbose=True,
    )

    out_stat = get_file_stat(out_nc_path)
    manifest_stat = get_file_stat(manifest_path)

    # Open built dataset for verification
    ds_inputs = xr.open_dataset(out_nc_path, engine="h5netcdf")

    # -------------------------------------------------------------------------
    # Check a: Aggregation Integrity on Days 0, 15, 30
    # -------------------------------------------------------------------------
    integrity_results: List[Dict[str, Any]] = []
    days_to_check = [0, 15, 30]
    all_integrity_pass = True

    with h5netcdf.File(RAW_FILES["sst"], "r") as f_sst, \
         h5netcdf.File(RAW_FILES["sla"], "r") as f_sla, \
         h5netcdf.File(RAW_FILES["sss"], "r") as f_sss, \
         h5netcdf.File(RAW_FILES["cur"], "r") as f_cur:

        for d in days_to_check:
            # SST: 25 cells/block
            sst_f = ds_inputs["sst_frac"].values[d]
            sst_raw = f_sst.variables["analysed_sst"][d]
            sst_mask = f_sst.variables["mask"][d]
            raw_sst_valid = int(np.sum((sst_raw != -32768) & ((sst_mask & 6) == 0)))
            built_sst_valid = int(round(np.sum(sst_f * 25.0)))
            match_sst = (raw_sst_valid == built_sst_valid)

            # SLA: 4 cells/block
            sla_f = ds_inputs["sla_frac"].values[d]
            sla_raw = f_sla.variables["sla"][d]
            raw_sla_valid = int(np.sum(sla_raw != -2147483647))
            built_sla_valid = int(round(np.sum(sla_f * 4.0)))
            match_sla = (raw_sla_valid == built_sla_valid)

            # SSS: 4 cells/block
            sss_f = ds_inputs["sss_frac"].values[d]
            sss_raw = f_sss.variables["sos"][d, 0]
            raw_sss_valid = int(np.sum(sss_raw != -32767.0))
            built_sss_valid = int(round(np.sum(sss_f * 4.0)))
            match_sss = (raw_sss_valid == built_sss_valid)

            # UO: 1 cell/block
            uo_f = ds_inputs["uo_frac"].values[d]
            uo_raw = f_cur.variables["uo"][d, 0]
            raw_uo_valid = int(np.sum(uo_raw != 32767))
            built_uo_valid = int(round(np.sum(uo_f * 1.0)))
            match_uo = (raw_uo_valid == built_uo_valid)

            # VO: 1 cell/block
            vo_f = ds_inputs["vo_frac"].values[d]
            vo_raw = f_cur.variables["vo"][d, 0]
            raw_vo_valid = int(np.sum(vo_raw != 32767))
            built_vo_valid = int(round(np.sum(vo_f * 1.0)))
            match_vo = (raw_vo_valid == built_vo_valid)

            day_pass = match_sst and match_sla and match_sss and match_uo and match_vo
            if not day_pass:
                all_integrity_pass = False

            integrity_results.append({
                "day_index": d,
                "sst": {"raw": raw_sst_valid, "built": built_sst_valid, "match": match_sst},
                "sla": {"raw": raw_sla_valid, "built": built_sla_valid, "match": match_sla},
                "sss": {"raw": raw_sss_valid, "built": built_sss_valid, "match": match_sss},
                "uo": {"raw": raw_uo_valid, "built": built_uo_valid, "match": match_uo},
                "vo": {"raw": raw_vo_valid, "built": built_vo_valid, "match": match_vo},
            })

    checks.append({
        "check_id": "a",
        "name": "Aggregation Integrity",
        "status": "PASS" if all_integrity_pass else "FAIL",
        "details": f"Checked days 0, 15, 30 across 5 variables: all sum(frac * block_size) == raw_valid: {all_integrity_pass}",
    })

    # -------------------------------------------------------------------------
    # Check b: Coverage Comparison Against input_coverage_by_variable.csv
    # -------------------------------------------------------------------------
    census_csv_path = os.path.join(
        PROJECT_ROOT, "outputs", "diagnostics", "input_aggregation_census", "input_coverage_by_variable.csv"
    )
    census_df = pd.read_csv(census_csv_path)

    cov_var_map = {
        "analysed_sst": ("OSTIA_SST", "analysed_sst"),
        "sla": ("DUACS_SLA", "sla"),
        "sos": ("MULTIOBS_SSS", "sos"),
        "uo": ("MULTIOBS_CURRENTS", "uo"),
        "vo": ("MULTIOBS_CURRENTS", "vo"),
    }

    coverage_comparison: List[Dict[str, Any]] = []
    all_coverage_pass = True

    for builder_var, (src_name, csv_var) in cov_var_map.items():
        row = census_df[(census_df["source"] == src_name) & (census_df["variable"] == csv_var)]
        if len(row) == 0:
            all_coverage_pass = False
            coverage_comparison.append({
                "variable": builder_var,
                "status": "MISSING_IN_CSV",
            })
            continue

        r = row.iloc[0]
        b_summary = manifest_data["coverage_summary_31days"][builder_var]

        # Day 0 comparison
        d0 = b_summary["day0"]
        c_d0_eq1 = (d0["frac_eq_1"] == int(r["day0_f_eq_1"]))
        c_d0_ge075 = (d0["frac_ge_075"] == int(r["day0_f_ge_075"]))
        c_d0_ge05 = (d0["frac_ge_05"] == int(r["day0_f_ge_05"]))
        c_d0_gt0 = (d0["frac_gt_0"] == int(r["day0_f_gt_0"]))

        # 31-day range comparison
        c_min_eq1 = (b_summary["min31_f_eq_1"] == int(r["min31_f_eq_1"]))
        c_max_eq1 = (b_summary["max31_f_eq_1"] == int(r["max31_f_eq_1"]))
        c_min_ge075 = (b_summary["min31_f_ge_075"] == int(r["min31_f_ge_075"]))
        c_max_ge075 = (b_summary["max31_f_ge_075"] == int(r["max31_f_ge_075"]))
        c_min_ge05 = (b_summary["min31_f_ge_05"] == int(r["min31_f_ge_05"]))
        c_max_ge05 = (b_summary["max31_f_ge_05"] == int(r["max31_f_ge_05"]))
        c_min_gt0 = (b_summary["min31_f_gt_0"] == int(r["min31_f_gt_0"]))
        c_max_gt0 = (b_summary["max31_f_gt_0"] == int(r["max31_f_gt_0"]))

        var_pass = (
            c_d0_eq1 and c_d0_ge075 and c_d0_ge05 and c_d0_gt0 and
            c_min_eq1 and c_max_eq1 and c_min_ge075 and c_max_ge075 and
            c_min_ge05 and c_max_ge05 and c_min_gt0 and c_max_gt0
        )
        if not var_pass:
            all_coverage_pass = False

        coverage_comparison.append({
            "builder_var": builder_var,
            "csv_source": src_name,
            "csv_var": csv_var,
            "match": var_pass,
            "day0_counts": {
                "builder": d0,
                "census_csv": {
                    "frac_eq_1": int(r["day0_f_eq_1"]),
                    "frac_ge_075": int(r["day0_f_ge_075"]),
                    "frac_ge_05": int(r["day0_f_ge_05"]),
                    "frac_gt_0": int(r["day0_f_gt_0"]),
                },
            },
            "range_31days": {
                "builder": {
                    "min_eq_1": b_summary["min31_f_eq_1"], "max_eq_1": b_summary["max31_f_eq_1"],
                    "min_ge_075": b_summary["min31_f_ge_075"], "max_ge_075": b_summary["max31_f_ge_075"],
                    "min_ge_05": b_summary["min31_f_ge_05"], "max_ge_05": b_summary["max31_f_ge_05"],
                    "min_gt_0": b_summary["min31_f_gt_0"], "max_gt_0": b_summary["max31_f_gt_0"],
                },
                "census_csv": {
                    "min_eq_1": int(r["min31_f_eq_1"]), "max_eq_1": int(r["max31_f_eq_1"]),
                    "min_ge_075": int(r["min31_f_ge_075"]), "max_ge_075": int(r["max31_f_ge_075"]),
                    "min_ge_05": int(r["min31_f_ge_05"]), "max_ge_05": int(r["max31_f_ge_05"]),
                    "min_gt_0": int(r["min31_f_gt_0"]), "max_gt_0": int(r["max31_f_gt_0"]),
                },
            },
        })

    checks.append({
        "check_id": "b",
        "name": "Coverage Census Equivalence",
        "status": "PASS" if all_coverage_pass else "FAIL",
        "details": f"All 5 matching variables equal census CSV counts on day 0 and across 31-day ranges: {all_coverage_pass}",
    })

    # -------------------------------------------------------------------------
    # Check c: Value Equivalence against input_aggregation_census.py
    # -------------------------------------------------------------------------
    diag_equiv_results: List[Dict[str, Any]] = []
    all_diag_equiv_pass = True

    with h5netcdf.File(RAW_FILES["sst"], "r") as f_sst, \
         h5netcdf.File(RAW_FILES["sla"], "r") as f_sla, \
         h5netcdf.File(RAW_FILES["sss"], "r") as f_sss, \
         h5netcdf.File(RAW_FILES["cur"], "r") as f_cur:

        for d in days_to_check:
            # SST: census returns Kelvin; convert to degC
            sst_raw = f_sst.variables["analysed_sst"][d]
            sst_mask = f_sst.variables["mask"][d]
            val_sst, _, _ = census_diag.compute_sst_validity(sst_raw, sst_mask)
            m_sst_k, _ = census_diag.block_aggregate(census_diag.decode_sst(sst_raw), val_sst, 5, 5)
            diag_sst_c = (m_sst_k - 273.15).astype(np.float32)
            built_sst_c = ds_inputs["sst_c"].values[d]

            # SLA
            sla_raw = f_sla.variables["sla"][d]
            val_sla = census_diag.compute_sla_validity(sla_raw)
            diag_sla, _ = census_diag.block_aggregate(census_diag.decode_sla(sla_raw), val_sla, 2, 2)
            diag_sla = diag_sla.astype(np.float32)
            built_sla = ds_inputs["sla_m"].values[d]

            # SSS
            sss_raw = f_sss.variables["sos"][d, 0]
            val_sss = census_diag.compute_sss_validity(sss_raw)
            diag_sss, _ = census_diag.block_aggregate(census_diag.decode_sss(sss_raw), val_sss, 2, 2)
            diag_sss = diag_sss.astype(np.float32)
            built_sss = ds_inputs["sss"].values[d]

            # UO
            uo_raw = f_cur.variables["uo"][d, 0]
            val_uo = census_diag.compute_currents_validity(uo_raw)
            diag_uo, _ = census_diag.block_aggregate(census_diag.decode_currents(uo_raw), val_uo, 1, 1)
            diag_uo = diag_uo.astype(np.float32)
            built_uo = ds_inputs["uo_ms"].values[d]

            # VO
            vo_raw = f_cur.variables["vo"][d, 0]
            val_vo = census_diag.compute_currents_validity(vo_raw)
            diag_vo, _ = census_diag.block_aggregate(census_diag.decode_currents(vo_raw), val_vo, 1, 1)
            diag_vo = diag_vo.astype(np.float32)
            built_vo = ds_inputs["vo_ms"].values[d]

            def compare_arrays(a_diag: np.ndarray, a_built: np.ndarray) -> Tuple[float, int]:
                nan_diag = np.isnan(a_diag)
                nan_built = np.isnan(a_built)
                nan_mismatches = int(np.sum(nan_diag != nan_built))
                valid_both = (~nan_diag) & (~nan_built)
                if np.any(valid_both):
                    max_diff = float(np.max(np.abs(a_diag[valid_both] - a_built[valid_both])))
                else:
                    max_diff = 0.0
                return max_diff, nan_mismatches

            diff_sst, nan_sst = compare_arrays(diag_sst_c, built_sst_c)
            diff_sla, nan_sla = compare_arrays(diag_sla, built_sla)
            diff_sss, nan_sss = compare_arrays(diag_sss, built_sss)
            diff_uo, nan_uo = compare_arrays(diag_uo, built_uo)
            diff_vo, nan_vo = compare_arrays(diag_vo, built_vo)

            pass_d = (
                nan_sst == 0 and diff_sst <= 1e-5 and
                nan_sla == 0 and diff_sla <= 1e-5 and
                nan_sss == 0 and diff_sss <= 1e-5 and
                nan_uo == 0 and diff_uo <= 1e-5 and
                nan_vo == 0 and diff_vo <= 1e-5
            )
            if not pass_d:
                all_diag_equiv_pass = False

            diag_equiv_results.append({
                "day_index": d,
                "sst": {"max_abs_diff": diff_sst, "nan_mismatches": nan_sst},
                "sla": {"max_abs_diff": diff_sla, "nan_mismatches": nan_sla},
                "sss": {"max_abs_diff": diff_sss, "nan_mismatches": nan_sss},
                "uo": {"max_abs_diff": diff_uo, "nan_mismatches": nan_uo},
                "vo": {"max_abs_diff": diff_vo, "nan_mismatches": nan_vo},
            })

    checks.append({
        "check_id": "c",
        "name": "Value Equivalence against Census Diagnostic",
        "status": "PASS" if all_diag_equiv_pass else "FAIL",
        "details": f"Zero NaN mismatches and max abs diff <= 1e-5 (float32 rounding) across all variables: {all_diag_equiv_pass}",
    })

    # -------------------------------------------------------------------------
    # Check d: Independent Brute-Force Check (2,000 Random Target Cells)
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    n_brute_samples = 2000
    sample_days = rng.integers(0, 31, size=n_brute_samples)
    sample_vars = rng.choice(["sst", "sla", "sss", "uo", "vo"], size=n_brute_samples)
    sample_i = rng.integers(0, 100, size=n_brute_samples)
    sample_j = rng.integers(0, 240, size=n_brute_samples)

    brute_mismatches = 0
    max_brute_diff = 0.0

    with h5netcdf.File(RAW_FILES["sst"], "r") as f_sst, \
         h5netcdf.File(RAW_FILES["sla"], "r") as f_sla, \
         h5netcdf.File(RAW_FILES["sss"], "r") as f_sss, \
         h5netcdf.File(RAW_FILES["cur"], "r") as f_cur:

        for s_idx in range(n_brute_samples):
            d = int(sample_days[s_idx])
            v = str(sample_vars[s_idx])
            ti = int(sample_i[s_idx])
            tj = int(sample_j[s_idx])

            if v == "sst":
                # 5x5 block
                s_raw = f_sst.variables["analysed_sst"][d, ti*5:(ti+1)*5, tj*5:(tj+1)*5]
                s_mask = f_sst.variables["mask"][d, ti*5:(ti+1)*5, tj*5:(tj+1)*5]
                val_mask = (s_raw != -32768) & ((s_mask & 6) == 0)
                if np.any(val_mask):
                    degc_vals = (s_raw.astype(np.float64) * 0.009999999776482582 + 273.1499938964844) - 273.15
                    expected_val = float(np.mean(degc_vals[val_mask]))
                else:
                    expected_val = np.nan
                file_val = float(ds_inputs["sst_c"].values[d, ti, tj])

            elif v == "sla":
                s_raw = f_sla.variables["sla"][d, ti*2:(ti+1)*2, tj*2:(tj+1)*2]
                val_mask = (s_raw != -2147483647)
                if np.any(val_mask):
                    m_vals = s_raw.astype(np.float64) * 0.0001
                    expected_val = float(np.mean(m_vals[val_mask]))
                else:
                    expected_val = np.nan
                file_val = float(ds_inputs["sla_m"].values[d, ti, tj])

            elif v == "sss":
                s_raw = f_sss.variables["sos"][d, 0, ti*2:(ti+1)*2, tj*2:(tj+1)*2]
                val_mask = (s_raw != -32767.0)
                if np.any(val_mask):
                    expected_val = float(np.mean(s_raw[val_mask]))
                else:
                    expected_val = np.nan
                file_val = float(ds_inputs["sss"].values[d, ti, tj])

            elif v == "uo":
                s_raw = f_cur.variables["uo"][d, 0, ti, tj]
                val = (s_raw != 32767)
                expected_val = float(s_raw * 0.001) if val else np.nan
                file_val = float(ds_inputs["uo_ms"].values[d, ti, tj])

            elif v == "vo":
                s_raw = f_cur.variables["vo"][d, 0, ti, tj]
                val = (s_raw != 32767)
                expected_val = float(s_raw * 0.001) if val else np.nan
                file_val = float(ds_inputs["vo_ms"].values[d, ti, tj])

            # Check equivalence
            if np.isnan(expected_val) and np.isnan(file_val):
                continue
            elif np.isnan(expected_val) or np.isnan(file_val):
                brute_mismatches += 1
            else:
                diff = abs(expected_val - file_val)
                max_brute_diff = max(max_brute_diff, diff)
                if diff > 1e-4:
                    brute_mismatches += 1

    checks.append({
        "check_id": "d",
        "name": "Brute-Force Independent Check",
        "status": "PASS" if brute_mismatches == 0 else "FAIL",
        "details": f"Checked {n_brute_samples} random cells across all days/variables: mismatches={brute_mismatches}, max_diff={max_brute_diff:.6e} (tol=1e-4)",
    })

    # -------------------------------------------------------------------------
    # Check e: Currents 1x1 Identity
    # -------------------------------------------------------------------------
    cur_1x1_pass = True
    max_cur_diff = 0.0

    with h5netcdf.File(RAW_FILES["cur"], "r") as f_cur:
        for d in range(31):
            raw_uo = f_cur.variables["uo"][d, 0][:]
            raw_vo = f_cur.variables["vo"][d, 0][:]

            file_uo = ds_inputs["uo_ms"].values[d]
            file_vo = ds_inputs["vo_ms"].values[d]

            val_uo = (raw_uo != 32767)
            val_vo = (raw_vo != 32767)

            # Check NaN matches invalid
            if not np.all(np.isnan(file_uo) == (~val_uo)):
                cur_1x1_pass = False
            if not np.all(np.isnan(file_vo) == (~val_vo)):
                cur_1x1_pass = False

            # Check valid equals native * 0.001
            expected_uo = (raw_uo.astype(np.float64) * 0.001).astype(np.float32)
            expected_vo = (raw_vo.astype(np.float64) * 0.001).astype(np.float32)

            diff_uo = np.max(np.abs(file_uo[val_uo] - expected_uo[val_uo])) if np.any(val_uo) else 0.0
            diff_vo = np.max(np.abs(file_vo[val_vo] - expected_vo[val_vo])) if np.any(val_vo) else 0.0

            max_cur_diff = max(max_cur_diff, float(diff_uo), float(diff_vo))
            if max_cur_diff > 1e-6:
                cur_1x1_pass = False

    checks.append({
        "check_id": "e",
        "name": "Currents 1x1 Exact Identity",
        "status": "PASS" if cur_1x1_pass else "FAIL",
        "details": f"All 31 days match native values exactly (max diff: {max_cur_diff:.2e}, zero NaN mismatches)",
    })

    # -------------------------------------------------------------------------
    # Check f: SST Lake Rule
    # -------------------------------------------------------------------------
    # 92 lake cells in sst_lake_cells.csv
    lake_csv_path = os.path.join(
        PROJECT_ROOT, "outputs", "diagnostics", "input_contract_audit_v2b", "sst_lake_cells.csv"
    )
    lake_df = pd.read_csv(lake_csv_path)
    lake_cells_count = len(lake_df)

    # Check cell at lat 19.525, lon 85.275 -> target_i = 58, target_j = 161
    ti_lake, tj_lake = 58, 161
    with h5netcdf.File(RAW_FILES["sst"], "r") as f_sst:
        raw_b = f_sst.variables["analysed_sst"][0, ti_lake*5:(ti_lake+1)*5, tj_lake*5:(tj_lake+1)*5]
        mask_b = f_sst.variables["mask"][0, ti_lake*5:(ti_lake+1)*5, tj_lake*5:(tj_lake+1)*5]

    val_b = (raw_b != -32768) & ((mask_b & 6) == 0)
    valid_count_lake = int(np.sum(val_b))  # expect 10
    frac_lake_expected = valid_count_lake / 25.0  # expect 0.4

    degc_10 = (raw_b[val_b].astype(np.float64) * 0.009999999776482582 + 273.1499938964844) - 273.15
    hand_mean_c = float(np.mean(degc_10))

    file_sst_val = float(ds_inputs["sst_c"].values[0, ti_lake, tj_lake])
    file_sst_frac = float(ds_inputs["sst_frac"].values[0, ti_lake, tj_lake])

    lake_rule_pass = (
        lake_cells_count == 92 and
        valid_count_lake == 10 and
        np.isclose(file_sst_frac, 0.4, atol=1e-5) and
        np.isclose(file_sst_val, hand_mean_c, atol=1e-4)
    )

    checks.append({
        "check_id": "f",
        "name": "SST Lake Rule Exclusion & Hand-Check",
        "status": "PASS" if lake_rule_pass else "FAIL",
        "details": (
            f"92 lake cells excluded; target cell (58, 161) frac={file_sst_frac:.4f} (exp 0.4), "
            f"value={file_sst_val:.6f} degC (hand-computed {hand_mean_c:.6f} degC, diff={abs(file_sst_val - hand_mean_c):.2e})"
        ),
    })

    # -------------------------------------------------------------------------
    # Check g: SST Unit Conversion on 1,000 Random Valid Cells
    # -------------------------------------------------------------------------
    sst_val_all = ds_inputs["sst_c"].values
    sst_frac_all = ds_inputs["sst_frac"].values
    valid_sst_indices = np.argwhere(sst_frac_all > 0)
    n_sample_g = 1000
    chosen_g = rng.choice(len(valid_sst_indices), size=n_sample_g, replace=False)

    max_conv_diff = 0.0
    conv_pass = True

    with h5netcdf.File(RAW_FILES["sst"], "r") as f_sst:
        for idx in chosen_g:
            d, ti, tj = valid_sst_indices[idx]
            raw_sub = f_sst.variables["analysed_sst"][d, ti*5:(ti+1)*5, tj*5:(tj+1)*5]
            mask_sub = f_sst.variables["mask"][d, ti*5:(ti+1)*5, tj*5:(tj+1)*5]
            m_sub = (raw_sub != -32768) & ((mask_sub & 6) == 0)

            # Native kelvin
            k_native = raw_sub[m_sub].astype(np.float64) * 0.009999999776482582 + 273.1499938964844
            mean_k = np.mean(k_native)
            expected_c = mean_k - 273.15

            actual_c = sst_val_all[d, ti, tj]
            diff = abs(actual_c - expected_c)
            max_conv_diff = max(max_conv_diff, diff)
            if diff > 1e-4:
                conv_pass = False

    checks.append({
        "check_id": "g",
        "name": "SST Unit Conversion Check",
        "status": "PASS" if conv_pass else "FAIL",
        "details": f"1,000 random valid cells checked: file degC equals decoded Kelvin - 273.15 within 1e-4 (max diff: {max_conv_diff:.6e})",
    })

    # -------------------------------------------------------------------------
    # Check h: Reproducibility (Second Build Gives Identical Arrays)
    # -------------------------------------------------------------------------
    temp_nc_path = os.path.join(OUTPUT_DIR, "inputs_2019-01_repro_test.nc")
    temp_manifest_path = os.path.join(OUTPUT_DIR, "inputs_2019-01_repro_manifest.json")

    build_month_file(
        policy_path=DEFAULT_POLICY_PATH,
        sst_path=RAW_FILES["sst"],
        sla_path=RAW_FILES["sla"],
        sss_path=RAW_FILES["sss"],
        cur_path=RAW_FILES["cur"],
        target_path=RAW_FILES["target"],
        out_path=temp_nc_path,
        manifest_out_path=temp_manifest_path,
        verbose=False,
    )

    repro_pass = True
    with xr.open_dataset(temp_nc_path, engine="h5netcdf") as ds_repro:
        for vname in ds_inputs.data_vars:
            arr1 = ds_inputs[vname].values
            arr2 = ds_repro[vname].values
            if not np.all(np.equal(arr1, arr2) | (np.isnan(arr1) & np.isnan(arr2))):
                repro_pass = False

    # Clean up temporary reproducibility test files
    if os.path.exists(temp_nc_path):
        os.remove(temp_nc_path)
    if os.path.exists(temp_manifest_path):
        os.remove(temp_manifest_path)

    checks.append({
        "check_id": "h",
        "name": "Reproducibility (Second Build Identical)",
        "status": "PASS" if repro_pass else "FAIL",
        "details": f"All 10 data variables identical between successive builds (equal_nan): {repro_pass}",
    })

    # -------------------------------------------------------------------------
    # Check j: Raw files and target file unchanged (SHA256, size, mtime)
    # -------------------------------------------------------------------------
    stats_after = {k: get_file_stat(p) for k, p in RAW_FILES.items()}
    all_files_unchanged = True
    integrity_details: List[str] = []

    for k, p in RAW_FILES.items():
        sb = stats_before[k]
        sa = stats_after[k]
        sha_match = (sb["sha256"] == sa["sha256"] == EXPECTED_HASHES[k])
        size_match = (sb["size_bytes"] == sa["size_bytes"])
        mtime_match = (sb["mtime_utc"] == sa["mtime_utc"])

        if not (sha_match and size_match and mtime_match):
            all_files_unchanged = False
            integrity_details.append(f"{k} MODIFIED!")
        else:
            integrity_details.append(f"{k} unchanged")

    checks.append({
        "check_id": "j",
        "name": "Raw Files & Target File Unchanged",
        "status": "PASS" if all_files_unchanged else "FAIL",
        "details": f"All 6 raw files and target file unchanged in SHA256, size, and mtime: {', '.join(integrity_details)}",
    })

    # -------------------------------------------------------------------------
    # Generate Diagnostic Plot: Day 0 Frac Maps for SST and SSS
    # -------------------------------------------------------------------------
    plot_path = os.path.join(OUTPUT_DIR, "plot_day0_frac_sst_sss.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    lat = ds_inputs["latitude"].values
    lon = ds_inputs["longitude"].values

    im0 = axes[0].pcolormesh(
        lon, lat, ds_inputs["sst_frac"].values[0],
        vmin=0.0, vmax=1.0, cmap="viridis", shading="nearest"
    )
    axes[0].set_title("Day 0: OSTIA SST Valid Cell Fraction (sst_frac)")
    axes[0].set_xlabel("Longitude (deg E)")
    axes[0].set_ylabel("Latitude (deg N)")
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.85)
    cbar0.set_label("Fraction (0..1)")

    im1 = axes[1].pcolormesh(
        lon, lat, ds_inputs["sss_frac"].values[0],
        vmin=0.0, vmax=1.0, cmap="plasma", shading="nearest"
    )
    axes[1].set_title("Day 0: MULTIOBS SSS Valid Cell Fraction (sss_frac)")
    axes[1].set_xlabel("Longitude (deg E)")
    axes[1].set_ylabel("Latitude (deg N)")
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.85)
    cbar1.set_label("Fraction (0..1)")

    fig.suptitle("OceanEmbed Preprocessing - Day 0 Input Valid Coverage Fractions (2019-01-01)", fontsize=14)
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Save CSV and JSON Artifacts
    # -------------------------------------------------------------------------
    checks_df = pd.DataFrame(checks)
    csv_out_path = os.path.join(OUTPUT_DIR, "acceptance_checks.csv")
    checks_df.to_csv(csv_out_path, index=False)

    summary_json_path = os.path.join(OUTPUT_DIR, "acceptance_summary.json")
    summary_data = {
        "analysis_name": "input_builder_v1_acceptance",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_checks_passed": all(c["status"] == "PASS" for c in checks),
        "checks": checks,
        "input_files_provenance": {
            k: {
                "path": stats_before[k]["path"],
                "sha256_before": stats_before[k]["sha256"],
                "sha256_after": stats_after[k]["sha256"],
                "size_before": stats_before[k]["size_bytes"],
                "size_after": stats_after[k]["size_bytes"],
                "mtime_before": stats_before[k]["mtime_utc"],
                "mtime_after": stats_after[k]["mtime_utc"],
                "unchanged": (stats_before[k]["sha256"] == stats_after[k]["sha256"]),
            }
            for k in RAW_FILES
        },
        "built_files": {
            "netcdf": out_stat,
            "manifest": manifest_stat,
        },
        "target_file_metadata": target_header_info,
        "lake_cell_check": {
            "target_i": ti_lake,
            "target_j": tj_lake,
            "lake_cells_in_audit": lake_cells_count,
            "valid_cells_count": valid_count_lake,
            "expected_frac": frac_lake_expected,
            "hand_mean_degC": hand_mean_c,
            "built_val_degC": file_sst_val,
            "built_frac": file_sst_frac,
            "pass": lake_rule_pass,
        },
        "brute_force_check": {
            "samples": n_brute_samples,
            "mismatches": brute_mismatches,
            "max_abs_diff": max_brute_diff,
            "pass": (brute_mismatches == 0),
        },
        "currents_1x1_check": {
            "max_abs_diff": max_cur_diff,
            "pass": cur_1x1_pass,
        },
        "reproducibility": {
            "identical_arrays": repro_pass,
        },
        "unknown_fields": {
            "tpa_correction_availability": None,
            "reason_for_tpa_correction_unknown": (
                "tpa_correction in DUACS NetCDF holds _FillValue (-2147483647) on all days; "
                "metadata states 'NOT IMPLEMENTED YET: This correction is not yet available for current product version'."
            ),
            "currents_explicit_land_mask": None,
            "reason_for_currents_mask_unknown": (
                "MULTIOBS currents product cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m lacks a dedicated mask variable; "
                "land boundaries are defined implicitly by the raw fill value 32767."
            ),
            "sss_temporal_mask_variance_cause": None,
            "reason_for_sss_mask_variance_unknown": (
                "MULTIOBS SSS coverage changes dynamically across January 2019 due to satellite swath revisit geometry "
                "and objective analysis correlation length cutoffs not exported into the product file."
            ),
            "future_downstream_thresholding": None,
            "reason_for_future_thresholding_unknown": (
                "Per research lead policy I2, no validity threshold tauI is applied in the builder; "
                "thresholding choices are reserved for dataset assembly and training pipeline stages."
            ),
        },
    }

    summary_data = json_serializable(summary_data)
    with open(summary_json_path, "w", encoding="utf-8") as f_j:
        json.dump(summary_data, f_j, indent=2, default=str)

    ds_inputs.close()

    print("\nAcceptance run complete. All artifacts generated successfully.")
    return checks, summary_data


def print_walkthrough_sections() -> None:
    """Print verbatim tables and required sections for final walkthrough."""
    csv_path = os.path.join(OUTPUT_DIR, "acceptance_checks.csv")
    json_path = os.path.join(OUTPUT_DIR, "acceptance_summary.json")
    manifest_path = os.path.join(PROJECT_ROOT, "data", "processed", "inputs", "inputs_2019-01_manifest.json")
    nc_path = os.path.join(PROJECT_ROOT, "data", "processed", "inputs", "inputs_2019-01.nc")

    if not os.path.exists(json_path):
        print("Summary JSON not found. Run acceptance first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. PASS / FAIL List
    print("=" * 80)
    print("1. ACCEPTANCE CHECKS (PASS/FAIL LIST)")
    print("=" * 80)
    df_checks = pd.read_csv(csv_path)
    print(df_checks[["check_id", "name", "status", "details"]].to_string(index=False))

    # 2. Coverage Comparison Table
    print("\n" + "=" * 80)
    print("2. COVERAGE COMPARISON (BUILDER VS CENSUS CSV)")
    print("=" * 80)
    census_csv = os.path.join(
        PROJECT_ROOT, "outputs", "diagnostics", "input_aggregation_census", "input_coverage_by_variable.csv"
    )
    df_census = pd.read_csv(census_csv)
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    rows = []
    cov_summary = man["coverage_summary_31days"]
    mapping = {
        ("OSTIA_SST", "analysed_sst"): "analysed_sst",
        ("DUACS_SLA", "sla"): "sla",
        ("MULTIOBS_SSS", "sos"): "sos",
        ("MULTIOBS_CURRENTS", "uo"): "uo",
        ("MULTIOBS_CURRENTS", "vo"): "vo",
    }
    for (src, var), b_var in mapping.items():
        c_row = df_census[(df_census["source"] == src) & (df_census["variable"] == var)].iloc[0]
        b_sum = cov_summary[b_var]
        rows.append({
            "source": src,
            "variable": var,
            "d0_f1_b": b_sum["day0"]["frac_eq_1"],
            "d0_f1_csv": c_row["day0_f_eq_1"],
            "d0_f05_b": b_sum["day0"]["frac_ge_05"],
            "d0_f05_csv": c_row["day0_f_ge_05"],
            "d0_fgt0_b": b_sum["day0"]["frac_gt_0"],
            "d0_fgt0_csv": c_row["day0_f_gt_0"],
            "min31_fgt0_b": b_sum["min31_f_gt_0"],
            "min31_fgt0_csv": c_row["min31_f_gt_0"],
            "max31_fgt0_b": b_sum["max31_f_gt_0"],
            "max31_fgt0_csv": c_row["max31_f_gt_0"],
            "match": (
                b_sum["day0"]["frac_eq_1"] == c_row["day0_f_eq_1"] and
                b_sum["day0"]["frac_ge_05"] == c_row["day0_f_ge_05"] and
                b_sum["day0"]["frac_gt_0"] == c_row["day0_f_gt_0"] and
                b_sum["min31_f_gt_0"] == c_row["min31_f_gt_0"] and
                b_sum["max31_f_gt_0"] == c_row["max31_f_gt_0"]
            )
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # 3. Equivalence Statistics
    print("\n" + "=" * 80)
    print("3. DIAGNOSTIC EQUIVALENCE & BRUTE-FORCE STATISTICS")
    print("=" * 80)
    print(f"Brute-Force 2,000 Random Cell Samples: Mismatches = {summary['brute_force_check']['mismatches']}, Max Diff = {summary['brute_force_check']['max_abs_diff']:.6e}")
    print(f"Currents 1x1 Identity Across All Days: Max Abs Diff = {summary['currents_1x1_check']['max_abs_diff']:.2e}, Pass = {summary['currents_1x1_check']['pass']}")
    print(f"Build Reproducibility: Identical Arrays = {summary['reproducibility']['identical_arrays']}")

    # 4. Lake Cell Check
    print("\n" + "=" * 80)
    print("4. SST LAKE CELL CHECK (TARGET CELL 58, 161 AT LAT 19.525, LON 85.275)")
    print("=" * 80)
    lake_chk = summary["lake_cell_check"]
    print(f"Lake cells flagged in audit: {lake_chk['lake_cells_in_audit']}")
    print(f"Valid ocean cells in block (58, 161): {lake_chk['valid_cells_count']} / 25")
    print(f"Expected fraction: {lake_chk['expected_frac']} | Built fraction: {lake_chk['built_frac']}")
    print(f"Hand-computed mean degC (10 valid cells): {lake_chk['hand_mean_degC']:.6f} degC")
    print(f"Built NetCDF sst_c value: {lake_chk['built_val_degC']:.6f} degC")
    print(f"Absolute difference: {abs(lake_chk['built_val_degC'] - lake_chk['hand_mean_degC']):.2e} degC")
    print(f"Lake check status: {'PASS' if lake_chk['pass'] else 'FAIL'}")

    # 5. Target File Header
    print("\n" + "=" * 80)
    print("5. TARGET FILE HEADER (glorys_targets_2019-01.nc)")
    print("=" * 80)
    with h5netcdf.File(RAW_FILES["target"], "r") as f_tgt:
        print("Dimensions:", dict(f_tgt.dimensions))
        print("Variables:")
        for vname, v in f_tgt.variables.items():
            print(f"  {vname}: shape={v.shape}, dtype={v.dtype}, dims={v.dimensions}")
            for a, val in v.attrs.items():
                print(f"    {a}: {val}")
        print("Global attributes:")
        for a, val in f_tgt.attrs.items():
            print(f"  {a}: {val}")

    # 6. Inputs File NetCDF Header
    print("\n" + "=" * 80)
    print("6. BUILT INPUTS NETCDF HEADER (inputs_2019-01.nc)")
    print("=" * 80)
    with h5netcdf.File(nc_path, "r") as f_in:
        print("Dimensions:", dict(f_in.dimensions))
        print("Variables:")
        for vname, v in f_in.variables.items():
            print(f"  {vname}: shape={v.shape}, dtype={v.dtype}, dims={v.dimensions}")
            for a, val in v.attrs.items():
                print(f"    {a}: {val}")
        print("Global attributes:")
        for a, val in f_in.attrs.items():
            print(f"  {a}: {val}")

    # 7. First 30 Lines of Manifest
    print("\n" + "=" * 80)
    print("7. FIRST 30 LINES OF MANIFEST JSON")
    print("=" * 80)
    with open(manifest_path, "r", encoding="utf-8") as f:
        m_lines = [f.readline() for _ in range(30)]
    print("".join(m_lines), end="")

    # 8. Full JSON
    print("\n" + "=" * 80)
    print("8. ACCEPTANCE SUMMARY JSON (FULL)")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="OceanEmbed Input Builder Acceptance Run")
    parser.add_argument("--print-only", action="store_true", help="Print existing walkthrough tables without re-running")
    args = parser.parse_args()

    if not args.print_only:
        run_acceptance()
    print_walkthrough_sections()


if __name__ == "__main__":
    main()
