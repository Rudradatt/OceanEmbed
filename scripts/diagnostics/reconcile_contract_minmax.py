"""
Independent recompute of contract min/max values and side-by-side reconciliation
between v1, v2, v2b, and independent ground truth.
Strictly READ-ONLY on raw data.
Imports NO helper functions from input_contract_audit.py or input_contract_audit_v2.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import h5netcdf

# Raw data file paths
DEFAULT_SST_FILE = "data/raw/sst/sst_ostia_2019-01.nc"
DEFAULT_SLA_FILE = "data/raw/sla/sla_duacs_2019-01.nc"
DEFAULT_SSS_FILE = "data/raw/sss/sss_multiobs_2019-01.nc"
DEFAULT_CUR_FILE = "data/raw/currents/currents_multiobs_2019-01.nc"

# Conflict table values reported in prompt for Day 0
CONFLICT_TABLE = {
    ("OSTIA_SST", "analysed_sst"): {
        "v1_min": 273.870, "v1_max": 302.950,
        "conflict_v2_min": 288.750, "conflict_v2_max": 305.800
    },
    ("OSTIA_SST", "analysis_error"): {
        "v1_min": 0.120, "v1_max": 2.060,
        "conflict_v2_min": 0.090, "conflict_v2_max": 1.830
    },
    ("DUACS_SLA", "sla"): {
        "v1_min": -0.255, "v1_max": 0.452,
        "conflict_v2_min": -1.7487, "conflict_v2_max": 1.5833
    },
    ("DUACS_SLA", "adt"): {
        "v1_min": 0.375, "v1_max": 1.562,
        "conflict_v2_min": -1.3323, "conflict_v2_max": 2.1287
    },
    ("DUACS_SLA", "ugos"): {
        "v1_min": -1.376, "v1_max": 1.131,
        "conflict_v2_min": -1.3763, "conflict_v2_max": 1.1578
    },
    ("DUACS_SLA", "vgos"): {
        "v1_min": -1.078, "v1_max": 2.097,
        "conflict_v2_min": -2.0967, "conflict_v2_max": 1.6366
    },
}


def compute_independent_minmax(
    sst_file: str = DEFAULT_SST_FILE,
    sla_file: str = DEFAULT_SLA_FILE,
    sss_file: str = DEFAULT_SSS_FILE,
    cur_file: str = DEFAULT_CUR_FILE,
    out_csv: str = None
) -> pd.DataFrame:
    """
    Independent recompute (plain numpy + h5netcdf) for time index 0 AND all 31 days.
    For each data variable:
    - raw min/max over non-fill
    - manual decoded min/max (raw * scale + offset)
    - on Day 0: lat/lon of min and max cells
    """
    sources = [
        ("OSTIA_SST", sst_file),
        ("DUACS_SLA", sla_file),
        ("MULTIOBS_SSS", sss_file),
        ("MULTIOBS_CURRENTS", cur_file),
    ]

    rows = []

    for source_name, file_path in sources:
        with h5netcdf.File(file_path, mode="r") as f:
            lat_name = "latitude" if "latitude" in f.variables else "lat"
            lon_name = "longitude" if "longitude" in f.variables else "lon"
            lats = np.array(f.variables[lat_name][:])
            lons = np.array(f.variables[lon_name][:])

            depth_vals = None
            if "depth" in f.variables:
                depth_vals = [float(x) for x in np.array(f.variables["depth"][:])]

            for vname, var in f.variables.items():
                if vname in [lat_name, lon_name, "time", "depth", "crs"]:
                    continue

                shape = var.shape
                # Skip 1D variables without spatial dimensions
                if len(shape) < 3:
                    continue

                v_attrs = dict(var.attrs)
                raw_full = np.array(var[:])
                fill_val = v_attrs.get("_FillValue", None)
                scale = float(v_attrs.get("scale_factor", 1.0))
                offset = float(v_attrs.get("add_offset", 0.0))

                # Handle depth dimension
                has_depth = (len(shape) == 4)
                if has_depth:
                    n_depths = shape[1]
                    depth_slices = [(depth_vals[d] if depth_vals else float(d), raw_full[:, d, :, :]) for d in range(n_depths)]
                else:
                    depth_slices = [(np.nan, raw_full)]

                for depth_m, raw_3d in depth_slices:
                    n_time, n_lat, n_lon = raw_3d.shape

                    # 1. Day 0 computation
                    day0_raw = raw_3d[0]
                    if fill_val is not None:
                        day0_valid = (day0_raw != fill_val)
                    else:
                        day0_valid = np.ones_like(day0_raw, dtype=bool)

                    day0_non_fill_count = int(np.sum(day0_valid))
                    day0_fill_count = int(np.sum(~day0_valid))

                    if day0_non_fill_count > 0:
                        day0_valid_vals = day0_raw[day0_valid]
                        is_int = np.issubdtype(day0_raw.dtype, np.integer)
                        
                        sel_min = np.argmin(np.where(day0_valid, day0_raw, np.inf))
                        min_coords = np.unravel_index(sel_min, day0_raw.shape)
                        raw_min_d0 = int(day0_raw[min_coords]) if is_int else float(day0_raw[min_coords])
                        min_lat_d0 = float(lats[min_coords[0]])
                        min_lon_d0 = float(lons[min_coords[1]])

                        sel_max = np.argmax(np.where(day0_valid, day0_raw, -np.inf))
                        max_coords = np.unravel_index(sel_max, day0_raw.shape)
                        raw_max_d0 = int(day0_raw[max_coords]) if is_int else float(day0_raw[max_coords])
                        max_lat_d0 = float(lats[max_coords[0]])
                        max_lon_d0 = float(lons[max_coords[1]])

                        dec_min_d0 = float(raw_min_d0 * scale + offset)
                        dec_max_d0 = float(raw_max_d0 * scale + offset)
                    else:
                        raw_min_d0, raw_max_d0 = None, None
                        dec_min_d0, dec_max_d0 = None, None
                        min_lat_d0, min_lon_d0, max_lat_d0, max_lon_d0 = None, None, None, None

                    # 2. All 31 days computation
                    if fill_val is not None:
                        all31_valid = (raw_3d != fill_val)
                    else:
                        all31_valid = np.ones_like(raw_3d, dtype=bool)

                    all31_non_fill_count = int(np.sum(all31_valid))
                    all31_fill_count = int(np.sum(~all31_valid))

                    if all31_non_fill_count > 0:
                        all31_valid_vals = raw_3d[all31_valid]
                        is_int = np.issubdtype(raw_3d.dtype, np.integer)
                        raw_min_all31 = int(np.min(all31_valid_vals)) if is_int else float(np.min(all31_valid_vals))
                        raw_max_all31 = int(np.max(all31_valid_vals)) if is_int else float(np.max(all31_valid_vals))
                        dec_min_all31 = float(raw_min_all31 * scale + offset)
                        dec_max_all31 = float(raw_max_all31 * scale + offset)
                    else:
                        raw_min_all31, raw_max_all31 = None, None
                        dec_min_all31, dec_max_all31 = None, None

                    rows.append({
                        "source": source_name,
                        "variable": vname,
                        "depth_m": depth_m,
                        "dtype": str(var.dtype),
                        "scale_factor": scale,
                        "add_offset": offset,
                        "fill_value": fill_val if fill_val is not None else np.nan,
                        "day0_non_fill_cells": day0_non_fill_count,
                        "day0_fill_cells": day0_fill_count,
                        "raw_min_day0": raw_min_d0,
                        "raw_max_day0": raw_max_d0,
                        "decoded_min_day0": dec_min_d0,
                        "decoded_max_day0": dec_max_d0,
                        "min_lat_day0": min_lat_d0,
                        "min_lon_day0": min_lon_d0,
                        "max_lat_day0": max_lat_d0,
                        "max_lon_day0": max_lon_d0,
                        "all31_non_fill_cells": all31_non_fill_count,
                        "all31_fill_cells": all31_fill_count,
                        "raw_min_all31": raw_min_all31,
                        "raw_max_all31": raw_max_all31,
                        "decoded_min_all31": dec_min_all31,
                        "decoded_max_all31": dec_max_all31,
                    })

    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def compare_minmax_sources(
    independent_csv: str,
    v1_csv: str,
    v2_csv: str,
    out_csv: str = None
) -> pd.DataFrame:
    """
    Compare independent recompute against v1, v2, and the user-reported CONFLICT table.
    Classify whether each CSV's min/max corresponds to Day 0, All Days, or other.
    """
    df_ind = pd.read_csv(independent_csv)
    df_v1 = pd.read_csv(v1_csv) if os.path.exists(v1_csv) else None
    df_v2 = pd.read_csv(v2_csv) if os.path.exists(v2_csv) else None

    comparison_rows = []

    for _, r_ind in df_ind.iterrows():
        src = r_ind["source"]
        vn = r_ind["variable"]
        depth = r_ind["depth_m"]

        # 1. Independent ground truth
        ind_d0_min = r_ind["decoded_min_day0"]
        ind_d0_max = r_ind["decoded_max_day0"]
        ind_all_min = r_ind["decoded_min_all31"]
        ind_all_max = r_ind["decoded_max_all31"]

        # 2. v1 values (if variable in v1)
        v1_d0_min, v1_d0_max = None, None
        v1_all_min, v1_all_max = None, None
        v1_slice_match = "N/A"
        if df_v1 is not None:
            sub_v1 = df_v1[(df_v1["source"] == src) & (df_v1["variable"] == vn)]
            if len(sub_v1) > 0:
                v1_d0_row = sub_v1[sub_v1["day_index"] == 0]
                if len(v1_d0_row) > 0:
                    v1_d0_min = float(v1_d0_row["decoded_min"].iloc[0])
                    v1_d0_max = float(v1_d0_row["decoded_max"].iloc[0])
                v1_all_min = float(sub_v1["decoded_min"].min())
                v1_all_max = float(sub_v1["decoded_max"].max())

                # Check which slice v1 day 0 matches
                if np.isclose(v1_d0_min, ind_d0_min, atol=1e-4) and np.isclose(v1_d0_max, ind_d0_max, atol=1e-4):
                    v1_slice_match = "Day 0"
                elif np.isclose(v1_d0_min, ind_all_min, atol=1e-4) and np.isclose(v1_d0_max, ind_all_max, atol=1e-4):
                    v1_slice_match = "All 31 Days"
                else:
                    v1_slice_match = "Mismatch"

        # 3. v2 values
        v2_d0_min, v2_d0_max = None, None
        v2_all_min, v2_all_max = None, None
        v2_slice_match = "N/A"
        if df_v2 is not None:
            if pd.isna(depth):
                sub_v2 = df_v2[(df_v2["source"] == src) & (df_v2["variable"] == vn) & (df_v2["depth_m"].isna())]
            else:
                sub_v2 = df_v2[(df_v2["source"] == src) & (df_v2["variable"] == vn) & (df_v2["depth_m"] == depth)]

            if len(sub_v2) > 0:
                v2_d0_row = sub_v2[sub_v2["day_index"] == 0]
                if len(v2_d0_row) > 0:
                    v2_d0_min = float(v2_d0_row["decoded_min"].iloc[0])
                    v2_d0_max = float(v2_d0_row["decoded_max"].iloc[0])
                v2_all_min = float(sub_v2["decoded_min"].min())
                v2_all_max = float(sub_v2["decoded_max"].max())

                if np.isclose(v2_d0_min, ind_d0_min, atol=1e-4) and np.isclose(v2_d0_max, ind_d0_max, atol=1e-4):
                    v2_slice_match = "Day 0"
                elif np.isclose(v2_d0_min, ind_all_min, atol=1e-4) and np.isclose(v2_d0_max, ind_all_max, atol=1e-4):
                    v2_slice_match = "All 31 Days"
                else:
                    v2_slice_match = "Mismatch"

        # 4. Check against CONFLICT table from prompt
        conflict_entry = CONFLICT_TABLE.get((src, vn), None)
        conflict_v1_min = conflict_entry["v1_min"] if conflict_entry else None
        conflict_v1_max = conflict_entry["v1_max"] if conflict_entry else None
        conflict_v2_min = conflict_entry["conflict_v2_min"] if conflict_entry else None
        conflict_v2_max = conflict_entry["conflict_v2_max"] if conflict_entry else None

        # Does v2 CSV match independent Day 0?
        v2_matches_independent_day0 = (
            np.isclose(v2_d0_min, ind_d0_min, atol=1e-4) and
            np.isclose(v2_d0_max, ind_d0_max, atol=1e-4)
        ) if (v2_d0_min is not None and ind_d0_min is not None) else None

        # Does v1 CSV match independent Day 0?
        v1_matches_independent_day0 = (
            np.isclose(v1_d0_min, ind_d0_min, atol=1e-4) and
            np.isclose(v1_d0_max, ind_d0_max, atol=1e-4)
        ) if (v1_d0_min is not None and ind_d0_min is not None) else None

        # Does conflict v2 table match v2 CSV?
        conflict_v2_matches_v2_csv = (
            np.isclose(conflict_v2_min, v2_d0_min, atol=1e-3) and
            np.isclose(conflict_v2_max, v2_d0_max, atol=1e-3)
        ) if (conflict_v2_min is not None and v2_d0_min is not None) else None

        comparison_rows.append({
            "source": src,
            "variable": vn,
            "depth_m": depth,
            "independent_day0_min": ind_d0_min,
            "independent_day0_max": ind_d0_max,
            "independent_all31_min": ind_all_min,
            "independent_all31_max": ind_all_max,
            "v1_day0_min": v1_d0_min,
            "v1_day0_max": v1_d0_max,
            "v1_slice_classification": v1_slice_match,
            "v1_matches_independent_d0": v1_matches_independent_day0,
            "v2_day0_min": v2_d0_min,
            "v2_day0_max": v2_d0_max,
            "v2_slice_classification": v2_slice_match,
            "v2_matches_independent_d0": v2_matches_independent_day0,
            "conflict_v1_table_min": conflict_v1_min,
            "conflict_v1_table_max": conflict_v1_max,
            "conflict_v2_table_min": conflict_v2_min,
            "conflict_v2_table_max": conflict_v2_max,
            "conflict_v2_matches_v2_csv": conflict_v2_matches_v2_csv,
        })

    df_comp = pd.DataFrame(comparison_rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_comp.to_csv(out_csv, index=False)
    return df_comp


if __name__ == "__main__":
    out_dir = "outputs/diagnostics/contract_minmax_reconcile"
    os.makedirs(out_dir, exist_ok=True)

    ind_csv = os.path.join(out_dir, "independent_minmax.csv")
    comp_csv = os.path.join(out_dir, "minmax_comparison.csv")

    v1_csv = "outputs/diagnostics/input_contract_audit/variable_contract.csv"
    v2_csv = "outputs/diagnostics/input_contract_audit_v2/variable_contract.csv"
    v2b_csv = "outputs/diagnostics/input_contract_audit_v2b/variable_contract.csv"

    print("Running independent recompute...")
    df_ind = compute_independent_minmax(out_csv=ind_csv)
    print(f"Wrote independent min/max: {ind_csv} ({len(df_ind)} series)")

    print("Running side-by-side comparison (v1 vs v2)...")
    df_comp = compare_minmax_sources(ind_csv, v1_csv, v2_csv, out_csv=comp_csv)
    print(f"Wrote comparison: {comp_csv}")

    if os.path.exists(v2b_csv):
        comp_v2b_csv = os.path.join(out_dir, "minmax_comparison_v2b.csv")
        print("Running side-by-side comparison (v1 vs v2b)...")
        df_comp_v2b = compare_minmax_sources(ind_csv, v1_csv, v2b_csv, out_csv=comp_v2b_csv)
        print(f"Wrote comparison vs v2b: {comp_v2b_csv}")
