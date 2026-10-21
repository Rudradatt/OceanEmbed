"""
Generalized Input Contract Audit v2.
Regression on SST/SLA, follow-ups on flagged items, and pilot audit of SSS and currents.
READ-ONLY on raw data. No regridding. No interpolation decisions.
"""

import os
import sys
import re
import json
import datetime
import hashlib
import numpy as np
import pandas as pd
import h5netcdf
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Target Grid Specification (cell-centered 0.25 deg)
TARGET_LAT_CENTRES = np.linspace(5.125, 29.875, 100)
TARGET_LON_CENTRES = np.linspace(45.125, 104.875, 240)
DOMAIN_LAT_BOUNDS = (5.0, 30.0)
DOMAIN_LON_BOUNDS = (45.0, 105.0)


def compute_file_sha256(path: str) -> str:
    """Compute SHA256 hash of a file in 8MB chunks without loading entire file in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_file_metadata(path: str) -> dict:
    """Return dictionary of file sha256, size_bytes, and mtime (ISO UTC and raw)."""
    sz = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    mtime_iso = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()
    sha = compute_file_sha256(path)
    return {
        "path": os.path.abspath(path),
        "sha256": sha,
        "size_bytes": sz,
        "mtime_raw": mtime,
        "mtime_utc": mtime_iso,
    }


def parse_time_units(units_str: str) -> tuple[str, datetime.datetime]:
    """Parse CF time units string (e.g. 'seconds since 1981-01-01 00:00:00')."""
    match = re.match(
        r"(seconds|minutes|hours|days)\s+since\s+([\d]{4}-[\d]{2}-[\d]{2}(?:[T\s][\d]{2}:[\d]{2}:[\d]{2}(?:\.[\d]+)?)?)",
        units_str.strip(),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized time unit format: '{units_str}'")
    unit_type = match.group(1).lower()
    base_str = match.group(2).replace("T", " ")
    if len(base_str) == 10:
        base_dt = datetime.datetime.strptime(base_str, "%Y-%m-%d")
    else:
        base_dt = datetime.datetime.strptime(base_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
    base_dt = base_dt.replace(tzinfo=datetime.timezone.utc)
    return unit_type, base_dt


def decode_timestamps(time_array: np.ndarray, units_str: str) -> list[datetime.datetime]:
    """Decode raw numeric time array into list of timezone-aware UTC datetimes."""
    unit_type, base_dt = parse_time_units(units_str)
    decoded = []
    for val in time_array:
        v = float(val)
        if unit_type == "seconds":
            dt = base_dt + datetime.timedelta(seconds=v)
        elif unit_type == "minutes":
            dt = base_dt + datetime.timedelta(minutes=v)
        elif unit_type == "hours":
            dt = base_dt + datetime.timedelta(hours=v)
        elif unit_type == "days":
            dt = base_dt + datetime.timedelta(days=v)
        else:
            raise ValueError(f"Unsupported time unit: {unit_type}")
        decoded.append(dt)
    return decoded


def classify_variable(name: str, attrs: dict, dims: tuple) -> tuple[str, str]:
    """
    Classify variable as 'coordinate', 'flag/mask', 'uncertainty/error', or 'data',
    quoting attribute evidence.
    """
    std_name = str(attrs.get("standard_name", "")).lower()
    name_lower = name.lower()

    if name_lower in ["latitude", "longitude", "time", "lat", "lon", "depth", "deptho", "deptho_lev", "crs"] or "axis" in attrs:
        evidence = f"coordinate axis '{attrs.get('axis', name)}' with standard_name='{attrs.get('standard_name', '')}'"
        return "coordinate", evidence

    if "flag_values" in attrs or "flag_masks" in attrs or "flag_meanings" in attrs or "status_flag" in std_name or "mask" in name_lower or "flag_" in name_lower:
        evidence = f"flag/mask metadata present (flag_values/masks/meanings: '{attrs.get('flag_meanings', '')}')"
        return "flag/mask", evidence

    if "error" in name_lower or "err_" in name_lower or "standard_error" in std_name or "standard deviation error" in str(attrs.get("comment", "")).lower():
        evidence = f"error/uncertainty naming and standard_name='{attrs.get('standard_name', '')}'"
        return "uncertainty/error", evidence

    evidence = f"physical data field with units='{attrs.get('units', '')}' and standard_name='{attrs.get('standard_name', '')}'"
    return "data", evidence


def dump_header_report(file_path: str, output_txt_path: str = None) -> tuple[str, dict]:
    """
    Extract verbatim global attributes, dimensions, and per-variable metadata,
    checking specifically for required contract attributes.
    """
    lines = []
    summary_dict = {"global_attributes": {}, "dimensions": {}, "variables": {}}

    with h5netcdf.File(file_path, "r") as f:
        lines.append("=" * 80)
        lines.append(f"HEADER DUMP: {os.path.basename(file_path)}")
        lines.append(f"Full Path: {os.path.abspath(file_path)}")
        lines.append("=" * 80)

        lines.append("\n--- Global Attributes ---")
        for k, v in f.attrs.items():
            val_str = repr(v)
            summary_dict["global_attributes"][k] = str(v)
            lines.append(f"  {k}: {val_str}")

        lines.append("\n--- Dimensions ---")
        for dname, dval in f.dimensions.items():
            sz = len(dval) if hasattr(dval, "__len__") else dval.size
            summary_dict["dimensions"][dname] = sz
            lines.append(f"  {dname}: size={sz}")

        lines.append("\n--- Variables ---")
        contract_attrs = [
            "scale_factor", "add_offset", "_FillValue", "missing_value",
            "valid_min", "valid_max", "valid_range",
            "flag_values", "flag_masks", "flag_meanings",
            "units", "standard_name"
        ]

        for vname, var in f.variables.items():
            v_attrs = dict(var.attrs)
            cls, evidence = classify_variable(vname, v_attrs, var.dimensions)
            lines.append(f"\nVariable: {vname}")
            lines.append(f"  dtype: {var.dtype}")
            lines.append(f"  shape: {var.shape}")
            lines.append(f"  dimensions: {var.dimensions}")
            lines.append(f"  Classification: {cls} (Evidence: {evidence})")
            lines.append("  Contract Attributes Existence & Values:")

            var_info = {
                "dtype": str(var.dtype),
                "shape": list(var.shape),
                "dimensions": list(var.dimensions),
                "classification": cls,
                "evidence": evidence,
                "attributes": {},
                "contract_attributes": {}
            }

            for ca in contract_attrs:
                if ca in v_attrs:
                    cval = v_attrs[ca]
                    if isinstance(cval, np.ndarray):
                        cval_str = repr(cval.tolist())
                    else:
                        cval_str = repr(cval)
                    lines.append(f"    - {ca}: PRESENT = {cval_str}")
                    var_info["contract_attributes"][ca] = str(cval)
                else:
                    lines.append(f"    - {ca}: ABSENT")
                    var_info["contract_attributes"][ca] = None

            lines.append("  All Attributes:")
            for ak, av in v_attrs.items():
                if isinstance(av, np.ndarray):
                    av_str = repr(av.tolist())
                else:
                    av_str = repr(av)
                lines.append(f"    {ak}: {av_str}")
                var_info["attributes"][ak] = str(av)

            summary_dict["variables"][vname] = var_info

    report_text = "\n".join(lines)
    if output_txt_path:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, "w", encoding="utf-8") as fp:
            fp.write(report_text)

    return report_text, summary_dict


def audit_coordinates_and_time(file_path: str, reference_dates: list[str] = None) -> dict:
    """
    Audit coordinates (lat, lon, time, depth) and calculate grid alignment against project target.
    """
    with h5netcdf.File(file_path, "r") as f:
        lat_name = "latitude" if "latitude" in f.variables else "lat"
        lon_name = "longitude" if "longitude" in f.variables else "lon"

        lat = np.array(f.variables[lat_name][:])
        lon = np.array(f.variables[lon_name][:])
        time_raw = np.array(f.variables["time"][:])
        time_attrs = dict(f.variables["time"].attrs)

        depth_info = None
        if "depth" in f.variables:
            depth_arr = np.array(f.variables["depth"][:])
            depth_attrs = dict(f.variables["depth"].attrs)
            depth_info = {
                "values": [float(x) for x in depth_arr],
                "count": len(depth_arr),
                "units": str(depth_attrs.get("units", "m")),
                "positive": str(depth_attrs.get("positive", "down")),
            }

    # Latitude metrics
    lat_count = len(lat)
    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    lat_diffs = np.diff(lat)
    lat_step = float(np.mean(lat_diffs))
    lat_monotonic = bool(np.all(lat_diffs > 0) or np.all(lat_diffs < 0))
    lat_direction = "ascending" if np.all(lat_diffs > 0) else ("descending" if np.all(lat_diffs < 0) else "non-monotonic")

    # Longitude metrics
    lon_count = len(lon)
    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    lon_diffs = np.diff(lon)
    lon_step = float(np.mean(lon_diffs))
    lon_monotonic = bool(np.all(lon_diffs > 0) or np.all(lon_diffs < 0))
    lon_direction = "ascending" if np.all(lon_diffs > 0) else ("descending" if np.all(lon_diffs < 0) else "non-monotonic")
    lon_convention = "0..360" if lon_max > 180.0 else "-180..180"

    # Time metrics
    time_units = str(time_attrs.get("units", ""))
    time_calendar = str(time_attrs.get("calendar", "standard"))
    time_count = len(time_raw)
    decoded_dts = decode_timestamps(time_raw, time_units)
    iso_timestamps = [dt.isoformat() for dt in decoded_dts]
    date_strs = [dt.strftime("%Y-%m-%d") for dt in decoded_dts]
    hours_of_day = [dt.hour for dt in decoded_dts]
    unique_hours = list(sorted(set(hours_of_day)))
    has_duplicate_dates = len(set(date_strs)) != len(date_strs)

    # Gaps check (daily step)
    time_diffs = [decoded_dts[i+1] - decoded_dts[i] for i in range(len(decoded_dts)-1)]
    has_gaps = any(td != datetime.timedelta(days=1) for td in time_diffs)

    # Date equality against reference
    dates_equal_reference = None
    if reference_dates is not None:
        dates_equal_reference = (date_strs == reference_dates)

    # Grid alignment against Target Grid (100 x 240, centres 5.125..29.875, 45.125..104.875)
    d_lat_abs = abs(lat_step)
    d_lon_abs = abs(lon_step)

    lat_dist_to_target = np.min(np.abs(lat[:, None] - TARGET_LAT_CENTRES[None, :]), axis=0)
    lat_exact = bool(np.all(lat_dist_to_target < 1e-4))
    lat_offset_cells = float(np.mean(lat_dist_to_target / d_lat_abs))

    lon_dist_to_target = np.min(np.abs(lon[:, None] - TARGET_LON_CENTRES[None, :]), axis=0)
    lon_exact = bool(np.all(lon_dist_to_target < 1e-4))
    lon_offset_cells = float(np.mean(lon_dist_to_target / d_lon_abs))

    lat_in_domain = int(np.sum((lat >= DOMAIN_LAT_BOUNDS[0] - 1e-5) & (lat <= DOMAIN_LAT_BOUNDS[1] + 1e-5)))
    lon_in_domain = int(np.sum((lon >= DOMAIN_LON_BOUNDS[0] - 1e-5) & (lon <= DOMAIN_LON_BOUNDS[1] + 1e-5)))
    native_cells_in_domain = lat_in_domain * lon_in_domain

    result = {
        "latitude": {
            "name": lat_name,
            "count": lat_count,
            "min": lat_min,
            "max": lat_max,
            "step": lat_step,
            "monotonic": lat_monotonic,
            "direction": lat_direction,
        },
        "longitude": {
            "name": lon_name,
            "count": lon_count,
            "min": lon_min,
            "max": lon_max,
            "step": lon_step,
            "monotonic": lon_monotonic,
            "direction": lon_direction,
            "convention": lon_convention,
            "range": [lon_min, lon_max],
        },
        "time": {
            "units": time_units,
            "calendar": time_calendar,
            "count": time_count,
            "raw_min": float(np.min(time_raw)),
            "raw_max": float(np.max(time_raw)),
            "first_timestamp": iso_timestamps[0] if iso_timestamps else None,
            "last_timestamp": iso_timestamps[-1] if iso_timestamps else None,
            "hours_of_day": unique_hours,
            "has_gaps": has_gaps,
            "has_duplicates": has_duplicate_dates,
            "dates_equal_reference": dates_equal_reference,
            "date_strings": date_strs,
        },
        "target_grid_alignment": {
            "lat_exact_within_1e-4": lat_exact,
            "lat_offset_cells": lat_offset_cells,
            "lon_exact_within_1e-4": lon_exact,
            "lon_offset_cells": lon_offset_cells,
            "cells_in_domain_lat": lat_in_domain,
            "cells_in_domain_lon": lon_in_domain,
            "native_cells_inside_target_domain": native_cells_in_domain,
        },
    }
    if depth_info is not None:
        result["depth"] = depth_info
    return result


def compute_nesting_diagnostics(
    centres: np.ndarray,
    target_start: float,
    target_step: float = 0.25,
    tol: float = 1e-4
) -> dict:
    """
    Compute nesting metrics for a 1D regular axis against target cell edges:
    - Target cell edges are at target_start + k * target_step.
    - Native cell edges around centre c are [c - step/2, c + step/2].
    - Determines if native edges coincide with target edges.
    - Computes native cells per target cell (ratio).
    - Computes edge offset in native-cell units.
    - Computes 1D area overlap weights of native cells within one complete target cell.
    """
    centres = np.asarray(centres, dtype=np.float64)
    step = float(np.mean(np.diff(centres)))
    half_step = step / 2.0
    native_edges = np.concatenate([[centres[0] - half_step], centres + half_step])

    t_min = target_start
    t_max = target_start + target_step * int(round((centres[-1] + half_step - target_start) / target_step))
    target_edges = np.arange(t_min, t_max + 0.5 * target_step, target_step)

    dists = np.min(np.abs(target_edges[:, None] - native_edges[None, :]), axis=1)
    coincide = bool(np.all(dists < tol))
    max_dist = float(np.max(dists))
    offset_native_units = max_dist / step
    cells_per_target = target_step / step

    # Calculate 1D overlap weights for the first complete target cell [target_start, target_start + target_step]
    c_start = target_start
    c_end = target_start + target_step

    cell_lefts = centres - half_step
    cell_rights = centres + half_step

    intersect_mask = (cell_rights > c_start + tol) & (cell_lefts < c_end - tol)
    int_lefts = np.maximum(cell_lefts[intersect_mask], c_start)
    int_rights = np.minimum(cell_rights[intersect_mask], c_end)
    widths = int_rights - int_lefts
    weights = widths / target_step

    return {
        "step": step,
        "cells_per_target": cells_per_target,
        "coincide": coincide,
        "offset_native_units": round(float(offset_native_units), 6),
        "weights": [round(float(w), 6) for w in weights],
    }


def audit_variable_raw_contract(
    file_path: str,
    source_name: str,
    target_var_names: list[str] = None,
    depth_selection: list[float] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Audit raw data variables across all days and depths:
    - raw dtype
    - fill cells per day & time-invariance of fill mask
    - raw min/max over non-fill
    - manual decoded min, max, mean (raw * scale + offset)
    - valid_min / valid_max violation count
    - decoded exact 0.0 count
    - frozen cells count and listing
    - lowest 5 and highest 5 decoded values on days 0, 15, and 30
    """
    contract_rows = []
    extremes_rows = []
    frozen_rows = []
    flag_mask_summary = {}

    with h5netcdf.File(file_path, "r") as f:
        lat_name = "latitude" if "latitude" in f.variables else "lat"
        lon_name = "longitude" if "longitude" in f.variables else "lon"
        lat = np.array(f.variables[lat_name][:])
        lon = np.array(f.variables[lon_name][:])
        time_raw = np.array(f.variables["time"][:])
        time_units = str(f.variables["time"].attrs.get("units", ""))
        decoded_dts = decode_timestamps(time_raw, time_units)
        date_strs = [dt.strftime("%Y-%m-%d") for dt in decoded_dts]

        depth_values = None
        if "depth" in f.variables:
            depth_values = [float(x) for x in np.array(f.variables["depth"][:])]

        all_vars = list(f.variables.keys()) if target_var_names is None else target_var_names

        for vname in all_vars:
            if vname in [lat_name, lon_name, "time", "depth", "crs"]:
                continue

            var = f.variables[vname]
            v_attrs = dict(var.attrs)
            dims = var.dimensions
            shape = var.shape

            # Skip 1D variables without spatial dimensions
            if len(shape) < 3:
                continue

            raw_full = np.array(var[:])
            cls, _ = classify_variable(vname, v_attrs, dims)

            # Determine depth slices to evaluate
            # Check if variable has depth dimension
            has_depth = "depth" in dims or (len(shape) == 4)

            slices_to_eval = []
            if has_depth and len(shape) == 4:
                n_depths = shape[1]
                for d_idx in range(n_depths):
                    d_val = depth_values[d_idx] if depth_values is not None else float(d_idx)
                    if depth_selection is not None and d_val not in depth_selection:
                        continue
                    slices_to_eval.append((d_val, raw_full[:, d_idx, :, :]))
            else:
                slices_to_eval.append((None, raw_full))

            fill_val = v_attrs.get("_FillValue", None)
            scale = float(v_attrs.get("scale_factor", 1.0))
            offset = float(v_attrs.get("add_offset", 0.0))
            vmin = v_attrs.get("valid_min", None)
            vmax = v_attrs.get("valid_max", None)

            for depth_m, raw_data in slices_to_eval:
                n_time, n_lat, n_lon = raw_data.shape
                total_spatial_cells = n_lat * n_lon

                # Flag mask daily summary
                if cls == "flag/mask":
                    flag_daily = {}
                    for t in range(n_time):
                        u_vals, u_counts = np.unique(raw_data[t], return_counts=True)
                        flag_daily[date_strs[t]] = dict(zip([int(x) for x in u_vals], [int(x) for x in u_counts]))
                    flag_mask_summary[vname] = flag_daily

                # Fill mask analysis
                if fill_val is not None:
                    fill_mask = (raw_data == fill_val)
                else:
                    fill_mask = np.zeros_like(raw_data, dtype=bool)

                non_fill_mask = ~fill_mask

                # Check time-invariance of fill mask
                fill_mask_day0 = fill_mask[0]
                fill_mask_identical = True
                for t in range(1, n_time):
                    if not np.array_equal(fill_mask[t], fill_mask_day0):
                        fill_mask_identical = False
                        break

                # Frozen cell analysis: non-fill on all days AND raw value identical on all days
                all_days_non_fill = np.all(non_fill_mask, axis=0)
                raw_min_time = np.min(raw_data, axis=0)
                raw_max_time = np.max(raw_data, axis=0)
                frozen_mask = all_days_non_fill & (raw_min_time == raw_max_time)
                frozen_count = int(np.sum(frozen_mask))

                # Record first 20 frozen cells
                if frozen_count > 0:
                    frozen_indices = np.argwhere(frozen_mask)[:20]
                    for idx_count, (r_idx, c_idx) in enumerate(frozen_indices):
                        r_val = int(raw_data[0, r_idx, c_idx])
                        d_val = float(r_val * scale + offset)
                        frozen_rows.append({
                            "source": source_name,
                            "variable": vname,
                            "depth_m": depth_m,
                            "frozen_rank": idx_count + 1,
                            "lat_idx": int(r_idx),
                            "lon_idx": int(c_idx),
                            "latitude": float(lat[r_idx]),
                            "longitude": float(lon[c_idx]),
                            "raw_value": r_val,
                            "decoded_value": d_val,
                        })

                # Daily analysis
                for t in range(n_time):
                    date_str = date_strs[t]
                    day_raw = raw_data[t]
                    day_non_fill = non_fill_mask[t]
                    n_fill = int(np.sum(fill_mask[t]))
                    n_non_fill = int(np.sum(day_non_fill))

                    if n_non_fill > 0:
                        valid_raw_vals = day_raw[day_non_fill]
                        raw_min = int(np.min(valid_raw_vals))
                        raw_max = int(np.max(valid_raw_vals))

                        # Manual decode: raw * scale + offset
                        valid_decoded = valid_raw_vals.astype(np.float64) * scale + offset
                        dec_min = float(np.min(valid_decoded))
                        dec_max = float(np.max(valid_decoded))
                        dec_mean = float(np.mean(valid_decoded))

                        # Valid min / max violations
                        n_vmin_viol = int(np.sum(valid_raw_vals < vmin)) if vmin is not None else 0
                        n_vmax_viol = int(np.sum(valid_raw_vals > vmax)) if vmax is not None else 0

                        # Decoded exact 0.0 cells
                        n_exact_zeros = int(np.sum(valid_decoded == 0.0))
                    else:
                        raw_min, raw_max = None, None
                        dec_min, dec_max, dec_mean = None, None, None
                        n_vmin_viol, n_vmax_viol = 0, 0
                        n_exact_zeros = 0

                    contract_rows.append({
                        "source": source_name,
                        "variable": vname,
                        "depth_m": depth_m,
                        "day_index": t,
                        "date": date_str,
                        "raw_dtype": str(var.dtype),
                        "total_cells": total_spatial_cells,
                        "fill_cells": n_fill,
                        "non_fill_cells": n_non_fill,
                        "fill_mask_identical_all_days": fill_mask_identical,
                        "raw_min": raw_min,
                        "raw_max": raw_max,
                        "decoded_min": dec_min,
                        "decoded_max": dec_max,
                        "decoded_mean": dec_mean,
                        "valid_min_violations": n_vmin_viol,
                        "valid_max_violations": n_vmax_viol,
                        "decoded_exact_zeros": n_exact_zeros,
                        "frozen_cells_count": frozen_count,
                    })

                    # Extremes on Day 0, Day 15, Day 30
                    if t in [0, 15, 30] and n_non_fill >= 5:
                        day_dec_2d = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
                        day_dec_2d[day_non_fill] = day_raw[day_non_fill].astype(np.float64) * scale + offset

                        r_coords, c_coords = np.where(day_non_fill)
                        dec_vals = day_dec_2d[day_non_fill]
                        raw_vals = day_raw[day_non_fill]

                        # Lowest 5
                        sort_asc = np.argsort(dec_vals)
                        for rank_idx in range(min(5, len(dec_vals))):
                            sel = sort_asc[rank_idx]
                            extremes_rows.append({
                                "source": source_name,
                                "variable": vname,
                                "depth_m": depth_m,
                                "day_index": t,
                                "date": date_str,
                                "extreme_type": "lowest_5",
                                "rank": rank_idx + 1,
                                "lat_idx": int(r_coords[sel]),
                                "lon_idx": int(c_coords[sel]),
                                "latitude": float(lat[r_coords[sel]]),
                                "longitude": float(lon[c_coords[sel]]),
                                "raw_value": int(raw_vals[sel]),
                                "decoded_value": float(dec_vals[sel]),
                            })

                        # Highest 5
                        sort_desc = np.argsort(-dec_vals)
                        for rank_idx in range(min(5, len(dec_vals))):
                            sel = sort_desc[rank_idx]
                            extremes_rows.append({
                                "source": source_name,
                                "variable": vname,
                                "depth_m": depth_m,
                                "day_index": t,
                                "date": date_str,
                                "extreme_type": "highest_5",
                                "rank": rank_idx + 1,
                                "lat_idx": int(r_coords[sel]),
                                "lon_idx": int(c_coords[sel]),
                                "latitude": float(lat[r_coords[sel]]),
                                "longitude": float(lon[c_coords[sel]]),
                                "raw_value": int(raw_vals[sel]),
                                "decoded_value": float(dec_vals[sel]),
                            })

    df_contract = pd.DataFrame(contract_rows)
    df_extremes = pd.DataFrame(extremes_rows)
    df_frozen = pd.DataFrame(frozen_rows)
    return df_contract, df_extremes, df_frozen, flag_mask_summary


def audit_coverage_vs_glorys(
    source_path: str,
    source_name: str,
    var_names: list[str],
    glorys_path: str,
    static_bathy_path: str,
    target_depth: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample source variables at GLORYS native cell centres by nearest-neighbour lookup (diagnostic only).
    Compute coverage categories:
    - GLORYS level-0 finite and source finite
    - GLORYS finite and source fill
    - GLORYS dry and source finite
    - GLORYS dry and source fill
    List first 30 GLORYS-finite/source-fill cells on Day 0 with lat, lon, static deptho.
    """
    coverage_rows = []
    missing_cells_rows = []

    # 1. Load GLORYS level 0 finite mask and coordinates
    with h5netcdf.File(glorys_path, "r") as f_glo:
        glo_lat = np.array(f_glo.variables["latitude"][:])
        glo_lon = np.array(f_glo.variables["longitude"][:])
        glo_raw_t0 = np.array(f_glo.variables["thetao"][0, 0, :, :])
        glo_fill = f_glo.variables["thetao"].attrs.get("_FillValue", -32767)
        glo_finite_mask = (glo_raw_t0 != glo_fill)
        n_glo_lat, n_glo_lon = glo_finite_mask.shape
        total_glorys_cells = n_glo_lat * n_glo_lon

    # 2. Load static deptho
    with h5netcdf.File(static_bathy_path, "r") as f_bathy:
        deptho_2d = np.array(f_bathy.variables["deptho"][:])

    # 3. Open source dataset and perform nearest neighbour mapping
    with h5netcdf.File(source_path, "r") as f_src:
        src_lat_name = "latitude" if "latitude" in f_src.variables else "lat"
        src_lon_name = "longitude" if "longitude" in f_src.variables else "lon"
        src_lat = np.array(f_src.variables[src_lat_name][:])
        src_lon = np.array(f_src.variables[src_lon_name][:])
        time_raw = np.array(f_src.variables["time"][:])
        time_units = str(f_src.variables["time"].attrs.get("units", ""))
        decoded_dts = decode_timestamps(time_raw, time_units)
        date_strs = [dt.strftime("%Y-%m-%d") for dt in decoded_dts]

        # Depth selection
        depth_idx = 0
        if "depth" in f_src.variables:
            depth_vals = [float(x) for x in np.array(f_src.variables["depth"][:])]
            if target_depth in depth_vals:
                depth_idx = depth_vals.index(target_depth)

        # Precompute nearest neighbour coordinate indices for GLORYS grid centres
        lat_nn_indices = np.argmin(np.abs(src_lat[:, None] - glo_lat[None, :]), axis=0)
        lon_nn_indices = np.argmin(np.abs(src_lon[:, None] - glo_lon[None, :]), axis=0)

        for vname in var_names:
            if vname not in f_src.variables:
                continue
            var = f_src.variables[vname]
            v_fill = var.attrs.get("_FillValue", None)
            raw_full = np.array(var[:])
            n_time = raw_full.shape[0]

            for t in range(n_time):
                date_str = date_strs[t]
                if raw_full.ndim == 4:
                    day_raw = raw_full[t, depth_idx, :, :]
                else:
                    day_raw = raw_full[t, :, :]

                if v_fill is not None:
                    src_finite_2d = (day_raw != v_fill)
                else:
                    src_finite_2d = np.ones_like(day_raw, dtype=bool)

                # Sample source finite mask at GLORYS grid centres
                src_finite_at_glorys = src_finite_2d[lat_nn_indices[:, None], lon_nn_indices[None, :]]

                # Categorize
                fin_fin = int(np.sum(glo_finite_mask & src_finite_at_glorys))
                fin_fill = int(np.sum(glo_finite_mask & ~src_finite_at_glorys))
                dry_fin = int(np.sum(~glo_finite_mask & src_finite_at_glorys))
                dry_fill = int(np.sum(~glo_finite_mask & ~src_finite_at_glorys))

                coverage_rows.append({
                    "source": source_name,
                    "variable": vname,
                    "day_index": t,
                    "date": date_str,
                    "total_glorys_cells": total_glorys_cells,
                    "glorys_finite_count": int(np.sum(glo_finite_mask)),
                    "glorys_finite_source_finite": fin_fin,
                    "glorys_finite_source_fill": fin_fill,
                    "glorys_dry_source_finite": dry_fin,
                    "glorys_dry_source_fill": dry_fill,
                })

                # On Day 0, capture up to 30 missing cells (GLORYS finite and source fill)
                if t == 0:
                    missing_coords = np.argwhere(glo_finite_mask & ~src_finite_at_glorys)[:30]
                    for m_idx, (r_idx, c_idx) in enumerate(missing_coords):
                        missing_cells_rows.append({
                            "source": source_name,
                            "variable": vname,
                            "rank": m_idx + 1,
                            "glorys_lat_idx": int(r_idx),
                            "glorys_lon_idx": int(c_idx),
                            "latitude": float(glo_lat[r_idx]),
                            "longitude": float(glo_lon[c_idx]),
                            "static_deptho_m": float(deptho_2d[r_idx, c_idx]),
                        })

    df_coverage = pd.DataFrame(coverage_rows)
    df_missing = pd.DataFrame(missing_cells_rows)
    return df_coverage, df_missing


# ==============================================================================
# Step 3 Follow-Up Diagnostics (SST and SLA)
# ==============================================================================

def audit_sst_frozen_error_cells(sst_path: str, glorys_path: str, out_csv: str) -> dict:
    """
    Follow-up 3a: SST analysis_error frozen cells (4,227).
    - how many are lake cells (mask bit for lake)
    - bounding boxes
    - histogram by 5-degree latitude band
    - how many have at least one land neighbour
    - distribution of values
    - 20 examples saved to sst_frozen_error_cells.csv
    """
    with h5netcdf.File(sst_path, "r") as f_sst:
        err = np.array(f_sst.variables["analysis_error"][:])
        mask_day0 = np.array(f_sst.variables["mask"][0])
        lat = np.array(f_sst.variables["latitude"][:])
        lon = np.array(f_sst.variables["longitude"][:])
        scale = float(f_sst.variables["analysis_error"].attrs["scale_factor"])
        offset = float(f_sst.variables["analysis_error"].attrs["add_offset"])
        fill = f_sst.variables["analysis_error"].attrs["_FillValue"]

    non_fill = (err != fill)
    all_non_fill = np.all(non_fill, axis=0)
    raw_min_time = np.min(err, axis=0)
    raw_max_time = np.max(err, axis=0)
    frozen_mask = all_non_fill & (raw_min_time == raw_max_time)
    frozen_count = int(np.sum(frozen_mask))

    # Lake cells (mask bit 4)
    is_lake = (mask_day0 & 4) != 0
    frozen_lake_mask = frozen_mask & is_lake
    frozen_lake_count = int(np.sum(frozen_lake_mask))

    # Bounding boxes
    r_idx, c_idx = np.where(frozen_mask)
    f_lats = lat[r_idx]
    f_lons = lon[c_idx]
    bbox_all = {
        "lat_min": float(f_lats.min()),
        "lat_max": float(f_lats.max()),
        "lon_min": float(f_lons.min()),
        "lon_max": float(f_lons.max()),
    }

    bbox_lake = None
    if frozen_lake_count > 0:
        r_lk, c_lk = np.where(frozen_lake_mask)
        bbox_lake = {
            "lat_min": float(lat[r_lk].min()),
            "lat_max": float(lat[r_lk].max()),
            "lon_min": float(lon[c_lk].min()),
            "lon_max": float(lon[c_lk].max()),
        }

    # Histogram by 5-degree latitude band
    bins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.001]
    bin_labels = ["[5.0, 10.0)", "[10.0, 15.0)", "[15.0, 20.0)", "[20.0, 25.0)", "[25.0, 30.0]"]
    counts, _ = np.histogram(f_lats, bins=bins)
    lat_band_hist = dict(zip(bin_labels, [int(c) for c in counts]))

    # Land neighbour analysis (mask bit 2 is land)
    is_land = (mask_day0 & 2) != 0
    struct_8 = np.ones((3, 3), dtype=bool)
    struct_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    land_adj_8 = ndimage.binary_dilation(is_land, structure=struct_8) & ~is_land
    land_adj_4 = ndimage.binary_dilation(is_land, structure=struct_4) & ~is_land

    n_with_land_8 = int(np.sum(frozen_mask & land_adj_8))
    n_with_land_4 = int(np.sum(frozen_mask & land_adj_4))
    n_are_land = int(np.sum(frozen_mask & is_land))
    n_land_inclusive_8 = int(np.sum(frozen_mask & ndimage.binary_dilation(is_land, structure=struct_8)))

    # Values distribution
    raw_vals = err[0, r_idx, c_idx]
    dec_vals = np.round(raw_vals.astype(np.float64) * scale + offset, 4)
    u_vals, u_counts = np.unique(dec_vals, return_counts=True)
    val_dist = {f"{float(uv):.4f} K": int(uc) for uv, uc in zip(u_vals, u_counts)}

    # Save CSV with 20 examples plus overall metrics
    csv_rows = []
    for i in range(min(20, len(r_idx))):
        r, c = r_idx[i], c_idx[i]
        csv_rows.append({
            "example_rank": i + 1,
            "lat_idx": int(r),
            "lon_idx": int(c),
            "latitude": float(lat[r]),
            "longitude": float(lon[c]),
            "mask_value": int(mask_day0[r, c]),
            "is_lake": bool(is_lake[r, c]),
            "is_land": bool(is_land[r, c]),
            "has_8conn_land_neighbour": bool(land_adj_8[r, c]),
            "has_4conn_land_neighbour": bool(land_adj_4[r, c]),
            "raw_analysis_error": int(raw_vals[i]),
            "decoded_analysis_error_kelvin": float(dec_vals[i]),
        })
    df_out = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    return {
        "frozen_cells_total": frozen_count,
        "lake_cells_count": frozen_lake_count,
        "bounding_box_all": bbox_all,
        "bounding_box_lake": bbox_lake,
        "latitude_band_histogram": lat_band_hist,
        "cells_with_8conn_land_neighbour": n_with_land_8,
        "cells_with_4conn_land_neighbour": n_with_land_4,
        "cells_with_land_or_land_neighbour_8conn": n_land_inclusive_8,
        "cells_that_are_land_mask": n_are_land,
        "values_distribution": val_dist,
        "unique_values_count": len(u_vals),
        "mean_decoded_value": float(np.mean(dec_vals)),
        "min_decoded_value": float(np.min(dec_vals)),
        "max_decoded_value": float(np.max(dec_vals)),
        "csv_path": out_csv,
    }


def audit_sst_lake_cells(sst_path: str, glorys_path: str, out_csv: str) -> dict:
    """
    Follow-up 3b: SST lake cells (92).
    - list lat, lon, analysed_sst min/max over the month
    - whether any lie inside GLORYS level-0 wet set (state count)
    - output: sst_lake_cells.csv
    """
    with h5netcdf.File(sst_path, "r") as f_sst:
        mask_day0 = np.array(f_sst.variables["mask"][0])
        sst = np.array(f_sst.variables["analysed_sst"][:])
        lat_sst = np.array(f_sst.variables["latitude"][:])
        lon_sst = np.array(f_sst.variables["longitude"][:])
        fill_sst = f_sst.variables["analysed_sst"].attrs["_FillValue"]
        scale = float(f_sst.variables["analysed_sst"].attrs["scale_factor"])
        offset = float(f_sst.variables["analysed_sst"].attrs["add_offset"])

    with h5netcdf.File(glorys_path, "r") as f_glo:
        lat_glo = np.array(f_glo.variables["latitude"][:])
        lon_glo = np.array(f_glo.variables["longitude"][:])
        glo_wet = (np.array(f_glo.variables["thetao"][0, 0]) != -32767)

    lake_mask = (mask_day0 & 4) != 0
    r_idx, c_idx = np.where(lake_mask)
    n_lake = len(r_idx)

    lat_nn = np.argmin(np.abs(lat_glo[:, None] - lat_sst[r_idx][None, :]), axis=0)
    lon_nn = np.argmin(np.abs(lon_glo[:, None] - lon_sst[c_idx][None, :]), axis=0)
    glo_wet_at_lake = glo_wet[lat_nn, lon_nn]
    n_inside_glorys_wet = int(np.sum(glo_wet_at_lake))

    rows = []
    for i in range(n_lake):
        r, c = r_idx[i], c_idx[i]
        vals = sst[:, r, c]
        val_non_fill = vals[vals != fill_sst]
        if len(val_non_fill) > 0:
            dec_vals = val_non_fill.astype(np.float64) * scale + offset
            dec_min = float(np.min(dec_vals))
            dec_max = float(np.max(dec_vals))
        else:
            dec_min, dec_max = None, None

        rows.append({
            "lake_cell_index": i + 1,
            "lat_idx": int(r),
            "lon_idx": int(c),
            "latitude": float(lat_sst[r]),
            "longitude": float(lon_sst[c]),
            "analysed_sst_min_kelvin": dec_min,
            "analysed_sst_max_kelvin": dec_max,
            "analysed_sst_min_degC": round(dec_min - 273.15, 3) if dec_min is not None else None,
            "analysed_sst_max_degC": round(dec_max - 273.15, 3) if dec_max is not None else None,
            "in_glorys_level0_wet": bool(glo_wet_at_lake[i]),
            "glorys_nearest_lat": float(lat_glo[lat_nn[i]]),
            "glorys_nearest_lon": float(lon_glo[lon_nn[i]]),
        })

    df_lake = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_lake.to_csv(out_csv, index=False)

    return {
        "lake_cells_total": n_lake,
        "inside_glorys_level0_wet_count": n_inside_glorys_wet,
        "overall_min_sst_degC": float(df_lake["analysed_sst_min_degC"].min()),
        "overall_max_sst_degC": float(df_lake["analysed_sst_max_degC"].max()),
        "csv_path": out_csv,
    }


def audit_sla_zero_cells(sla_path: str, out_csv: str) -> dict:
    """
    Follow-up 3c: SLA exact zeros for sla, ugos, vgos, ugosa, vgosa.
    - per day count
    - how many cells are zero on all 31 days
    - latitude-band histogram
    - how many are adjacent to fill cells
    - 20 examples saved to sla_zero_cells.csv
    """
    target_vars = ["sla", "ugos", "vgos", "ugosa", "vgosa"]
    bins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.001]
    bin_labels = ["[5.0, 10.0)", "[10.0, 15.0)", "[15.0, 20.0)", "[20.0, 25.0)", "[25.0, 30.0]"]

    struct_8 = np.ones((3, 3), dtype=bool)
    struct_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    per_var_summary = {}
    example_rows = []

    with h5netcdf.File(sla_path, "r") as f_sla:
        lat = np.array(f_sla.variables["latitude"][:])
        lon = np.array(f_sla.variables["longitude"][:])
        time_raw = np.array(f_sla.variables["time"][:])
        time_units = str(f_sla.variables["time"].attrs.get("units", ""))
        decoded_dts = decode_timestamps(time_raw, time_units)
        date_strs = [dt.strftime("%Y-%m-%d") for dt in decoded_dts]

        for vn in target_vars:
            arr = np.array(f_sla.variables[vn][:])
            fill_val = f_sla.variables[vn].attrs["_FillValue"]
            fill_mask = (arr[0] == fill_val)

            fill_adj_8 = ndimage.binary_dilation(fill_mask, structure=struct_8) & ~fill_mask
            fill_adj_4 = ndimage.binary_dilation(fill_mask, structure=struct_4) & ~fill_mask

            zeros_3d = (arr == 0)
            daily_counts = [int(x) for x in np.sum(zeros_3d, axis=(1, 2))]
            zero_all_31_days = int(np.sum(np.all(zeros_3d, axis=0)))

            total_zeros = int(np.sum(zeros_3d))
            zeros_adj_8 = int(np.sum(zeros_3d & fill_adj_8[None, :, :]))
            zeros_adj_4 = int(np.sum(zeros_3d & fill_adj_4[None, :, :]))

            # Latitude band histogram
            t_idx, r_idx, c_idx = np.where(zeros_3d)
            z_lats = lat[r_idx]
            counts, _ = np.histogram(z_lats, bins=bins)
            lat_hist = dict(zip(bin_labels, [int(c) for c in counts]))

            per_var_summary[vn] = {
                "daily_counts": dict(zip(date_strs, daily_counts)),
                "min_daily_zeros": min(daily_counts),
                "max_daily_zeros": max(daily_counts),
                "total_zero_instances": total_zeros,
                "zero_on_all_31_days_count": zero_all_31_days,
                "adjacent_to_fill_8conn": zeros_adj_8,
                "adjacent_to_fill_4conn": zeros_adj_4,
                "percent_adjacent_8conn": round(zeros_adj_8 / total_zeros * 100, 2) if total_zeros > 0 else 0.0,
                "latitude_band_histogram": lat_hist,
            }

            # Take up to 4 examples per variable to make 20 total examples
            for ex_i in range(min(4, len(t_idx))):
                t, r, c = t_idx[ex_i], r_idx[ex_i], c_idx[ex_i]
                example_rows.append({
                    "variable": vn,
                    "example_index": len(example_rows) + 1,
                    "day_index": int(t),
                    "date": date_strs[t],
                    "lat_idx": int(r),
                    "lon_idx": int(c),
                    "latitude": float(lat[r]),
                    "longitude": float(lon[c]),
                    "raw_value": 0,
                    "decoded_value": 0.0,
                    "adjacent_to_fill_8conn": bool(fill_adj_8[r, c]),
                    "adjacent_to_fill_4conn": bool(fill_adj_4[r, c]),
                })

    df_examples = pd.DataFrame(example_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_examples.to_csv(out_csv, index=False)

    return {
        "variables": per_var_summary,
        "csv_path": out_csv,
    }


def audit_sla_velocity_extremes(sla_path: str, out_csv: str) -> dict:
    """
    Follow-up 3d: SLA velocity extremes for ugos, vgos, ugosa, vgosa on days 0, 15, 30.
    - 10 largest |value| cells (lat, lon, value)
    - number of cells with |value| > 1.0 and > 1.5 m/s per 5-degree latitude band
    - output: sla_velocity_extremes.csv
    """
    target_vars = ["ugos", "vgos", "ugosa", "vgosa"]
    bins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.001]
    bin_labels = ["[5.0, 10.0)", "[10.0, 15.0)", "[15.0, 20.0)", "[20.0, 25.0)", "[25.0, 30.0]"]

    csv_rows = []
    summary_results = {}

    with h5netcdf.File(sla_path, "r") as f_sla:
        lat = np.array(f_sla.variables["latitude"][:])
        lon = np.array(f_sla.variables["longitude"][:])
        time_raw = np.array(f_sla.variables["time"][:])
        time_units = str(f_sla.variables["time"].attrs.get("units", ""))
        decoded_dts = decode_timestamps(time_raw, time_units)
        date_strs = [dt.strftime("%Y-%m-%d") for dt in decoded_dts]

        for vn in target_vars:
            var = f_sla.variables[vn]
            fill = var.attrs["_FillValue"]
            scale = float(var.attrs["scale_factor"])
            raw = np.array(var[:])

            var_days_summary = {}

            target_days = [d for d in [0, 15, 30] if d < len(date_strs)]
            for t in target_days:
                date_str = date_strs[t]
                day_raw = raw[t]
                valid = (day_raw != fill)
                r_coords, c_coords = np.where(valid)

                dec_vals = day_raw[valid].astype(np.float64) * scale
                abs_vals = np.abs(dec_vals)

                # 10 largest |value|
                sort_desc = np.argsort(-abs_vals)
                top10_list = []
                for rank_idx in range(min(10, len(abs_vals))):
                    sel = sort_desc[rank_idx]
                    r, c = r_coords[sel], c_coords[sel]
                    v_val = float(dec_vals[sel])
                    top10_list.append({
                        "rank": rank_idx + 1,
                        "latitude": float(lat[r]),
                        "longitude": float(lon[c]),
                        "velocity_mps": v_val,
                        "abs_velocity_mps": float(abs_vals[sel]),
                    })
                    csv_rows.append({
                        "variable": vn,
                        "day_index": t,
                        "date": date_str,
                        "rank": rank_idx + 1,
                        "lat_idx": int(r),
                        "lon_idx": int(c),
                        "latitude": float(lat[r]),
                        "longitude": float(lon[c]),
                        "velocity_mps": v_val,
                        "abs_velocity_mps": float(abs_vals[sel]),
                    })

                # Latitude band counts for > 1.0 and > 1.5 m/s
                v_lats = lat[r_coords]
                gt_1_0_mask = abs_vals > 1.0
                gt_1_5_mask = abs_vals > 1.5

                c_1_0, _ = np.histogram(v_lats[gt_1_0_mask], bins=bins)
                c_1_5, _ = np.histogram(v_lats[gt_1_5_mask], bins=bins)

                var_days_summary[f"day_{t}"] = {
                    "date": date_str,
                    "max_abs_velocity_mps": float(np.max(abs_vals)),
                    "total_cells_gt_1_0_mps": int(np.sum(gt_1_0_mask)),
                    "total_cells_gt_1_5_mps": int(np.sum(gt_1_5_mask)),
                    "histogram_gt_1_0_mps": dict(zip(bin_labels, [int(x) for x in c_1_0])),
                    "histogram_gt_1_5_mps": dict(zip(bin_labels, [int(x) for x in c_1_5])),
                    "top_10_cells": top10_list,
                }

            summary_results[vn] = var_days_summary

    df_extremes = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_extremes.to_csv(out_csv, index=False)

    return {
        "results": summary_results,
        "csv_path": out_csv,
    }


# ==============================================================================
# Diagnostic Plotting
# ==============================================================================

def generate_audit_v2_plots(
    sss_path: str,
    currents_path: str,
    glorys_path: str,
    out_dir: str
) -> dict:
    """
    Generate the 3 required diagnostic PNGs:
    1. Day-0 map of SSS main variable (sos)
    2. Day-0 map of total surface current speed (sqrt(uo^2 + vo^2) at depth 0)
    3. Map of nesting / coverage mismatch for currents vs GLORYS
    """
    os.makedirs(out_dir, exist_ok=True)
    generated = {}

    # 1. Day-0 SSS map
    with h5netcdf.File(sss_path, "r") as f_sss:
        sss_lat = np.array(f_sss.variables["latitude"][:])
        sss_lon = np.array(f_sss.variables["longitude"][:])
        sos_raw = np.array(f_sss.variables["sos"][0, 0, :, :])
        sos_fill = f_sss.variables["sos"].attrs["_FillValue"]
        sos_scale = float(f_sss.variables["sos"].attrs.get("scale_factor", 1.0))
        sos_offset = float(f_sss.variables["sos"].attrs.get("add_offset", 0.0))
        sos_psu = np.where(sos_raw != sos_fill, sos_raw.astype(np.float64) * sos_scale + sos_offset, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    mesh = ax.pcolormesh(sss_lon, sss_lat, sos_psu, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="Sea Surface Salinity (PSU)")
    ax.set_title("Multi-Obs L4 Reprocessed SSS — Day 0 (2019-01-01)\nDomain: 5–30°N, 45–105°E (0.125° native)", fontsize=11)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "plot_sss_sos_day0.png")
    fig.savefig(p1)
    plt.close(fig)
    generated["plot_sss_sos_day0"] = p1

    # 2. Day-0 Surface Current Speed map
    with h5netcdf.File(currents_path, "r") as f_cur:
        cur_lat = np.array(f_cur.variables["latitude"][:])
        cur_lon = np.array(f_cur.variables["longitude"][:])
        uo_raw = np.array(f_cur.variables["uo"][0, 0, :, :])
        vo_raw = np.array(f_cur.variables["vo"][0, 0, :, :])
        uo_fill = f_cur.variables["uo"].attrs["_FillValue"]
        vo_fill = f_cur.variables["vo"].attrs["_FillValue"]
        uo_scale = float(f_cur.variables["uo"].attrs.get("scale_factor", 1.0))
        vo_scale = float(f_cur.variables["vo"].attrs.get("scale_factor", 1.0))

        valid_cur = (uo_raw != uo_fill) & (vo_raw != vo_fill)
        uo_val = np.where(valid_cur, uo_raw.astype(np.float64) * uo_scale, np.nan)
        vo_val = np.where(valid_cur, vo_raw.astype(np.float64) * vo_scale, np.nan)
        speed_val = np.sqrt(uo_val**2 + vo_val**2)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    mesh = ax.pcolormesh(cur_lon, cur_lat, speed_val, cmap="plasma", shading="auto")
    fig.colorbar(mesh, ax=ax, label="Surface Current Speed (m/s)")
    ax.set_title("Multi-Obs Total Surface Current Speed — Day 0 (2019-01-01, depth=0 m)\nDomain: 5–30°N, 45–105°E (0.25° native)", fontsize=11)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "plot_currents_speed_surface_day0.png")
    fig.savefig(p2)
    plt.close(fig)
    generated["plot_currents_speed_surface_day0"] = p2

    # 3. Currents Nesting / Coverage Mismatch vs GLORYS Level 0
    with h5netcdf.File(glorys_path, "r") as f_glo:
        glo_lat = np.array(f_glo.variables["latitude"][:])
        glo_lon = np.array(f_glo.variables["longitude"][:])
        glo_finite = (np.array(f_glo.variables["thetao"][0, 0, :, :]) != -32767)

    with h5netcdf.File(currents_path, "r") as f_cur:
        cur_fin = (np.array(f_cur.variables["uo"][0, 0, :, :]) != f_cur.variables["uo"].attrs["_FillValue"])
        lat_nn = np.argmin(np.abs(cur_lat[:, None] - glo_lat[None, :]), axis=0)
        lon_nn = np.argmin(np.abs(cur_lon[:, None] - glo_lon[None, :]), axis=0)
        cur_at_glo = cur_fin[lat_nn[:, None], lon_nn[None, :]]

        glo_wet_cur_fill = glo_finite & ~cur_at_glo
        glo_dry_cur_fin = ~glo_finite & cur_at_glo

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)

    r_wf, c_wf = np.where(glo_wet_cur_fill)
    axes[0].scatter(glo_lon[c_wf], glo_lat[r_wf], s=6, color="crimson", alpha=0.7)
    axes[0].set_title(f"GLORYS Ocean but Currents Missing (N={len(r_wf):,})\nNearest-neighbour sampled", fontsize=10)
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_xlim(44.5, 105.5)
    axes[0].set_ylim(4.5, 30.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    r_df, c_df = np.where(glo_dry_cur_fin)
    axes[1].scatter(glo_lon[c_df], glo_lat[r_df], s=6, color="navy", alpha=0.7)
    axes[1].set_title(f"GLORYS Land/Dry but Currents Finite (N={len(r_df):,})\nNearest-neighbour sampled", fontsize=10)
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_xlim(44.5, 105.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p3 = os.path.join(out_dir, "plot_currents_coverage_mismatch_vs_glorys.png")
    fig.savefig(p3)
    plt.close(fig)
    generated["plot_currents_coverage_mismatch_vs_glorys"] = p3

    return generated


# ==============================================================================
# Environment Snapshot Recording
# ==============================================================================

def record_environment_snapshot(out_path: str) -> dict:
    """Record interpreter path, version, and pip freeze in environment_snapshot.txt."""
    import subprocess
    py_exec = sys.executable
    py_ver = sys.version
    import platform
    plat_str = platform.platform()

    # Run pip freeze
    res = subprocess.run([py_exec, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    freeze_text = res.stdout.strip()

    prev_ver = "Python 3.13.0"
    differs = not py_ver.startswith("3.13.0")

    lines = [
        "ENVIRONMENT SNAPSHOT v2",
        f"Python Executable: {py_exec}",
        f"Python Version: {py_ver}",
        f"Platform / OS: {plat_str}",
        "Interpreter Comparison Note:",
        f"  Active interpreter: {py_exec}",
        f"  Comparison vs previous run ({prev_ver}): {'DIFFERS' if differs else 'IDENTICAL (Python 3.13.0)'}",
        "",
        "PIP FREEZE:",
        freeze_text,
    ]
    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(report_text)

    return {
        "executable": py_exec,
        "version": py_ver,
        "platform": plat_str,
        "differs_from_previous": differs,
    }


# ==============================================================================
# Regression Verification Against v1
# ==============================================================================

def check_regression_vs_v1(v1_dir: str, df_v2_contract: pd.DataFrame, df_v2_coverage: pd.DataFrame) -> dict:
    """
    Compare v2 outputs on SST and SLA against v1 expectations and saved CSVs:
    - Per-variable fill counts per day
    - Non-fill counts per day
    - Decoded min/max
    - Frozen-cell counts
    - Coverage counts vs GLORYS level 0:
        SST analysed_sst: 107,312 / 412 / 907
        SLA sla: 107,664 / 60 / 2,383
        SLA adt, ugos, vgos: 107,393 / 331 / 1,832
    """
    results = {}
    v1_contract_path = os.path.join(v1_dir, "variable_contract.csv")
    v1_cov_path = os.path.join(v1_dir, "coverage_vs_glorys.csv")

    if not os.path.exists(v1_contract_path) or not os.path.exists(v1_cov_path):
        results["overall_regression_status"] = "FAIL: v1 reference files not found"
        return results

    df_v1_c = pd.read_csv(v1_contract_path)
    df_v1_cov = pd.read_csv(v1_cov_path)

    # Filter v2 contract to OSTIA_SST and DUACS_SLA
    v2_sub_c = df_v2_contract[df_v2_contract["source"].isin(["OSTIA_SST", "DUACS_SLA"])].copy()

    # 1. Fill count comparison
    fill_match = True
    non_fill_match = True
    dec_min_match = True
    dec_max_match = True
    frozen_match = True

    for _, r1 in df_v1_c.iterrows():
        match_rows = v2_sub_c[
            (v2_sub_c["source"] == r1["source"]) &
            (v2_sub_c["variable"] == r1["variable"]) &
            (v2_sub_c["day_index"] == r1["day_index"])
        ]
        if len(match_rows) == 0:
            fill_match = False
            continue
        r2 = match_rows.iloc[0]
        if r1["fill_cells"] != r2["fill_cells"]:
            fill_match = False
        if r1["non_fill_cells"] != r2["non_fill_cells"]:
            non_fill_match = False
        if not np.isclose(r1["decoded_min"], r2["decoded_min"], atol=1e-3, equal_nan=True):
            dec_min_match = False
        if not np.isclose(r1["decoded_max"], r2["decoded_max"], atol=1e-3, equal_nan=True):
            dec_max_match = False
        if r1["frozen_cells_count"] != r2["frozen_cells_count"]:
            frozen_match = False

    results["check_sst_sla_fill_counts_per_day"] = "PASS" if fill_match else "FAIL"
    results["check_sst_sla_non_fill_counts_per_day"] = "PASS" if non_fill_match else "FAIL"
    results["check_sst_sla_decoded_min"] = "PASS" if dec_min_match else "FAIL"
    results["check_sst_sla_decoded_max"] = "PASS" if dec_max_match else "FAIL"
    results["check_sst_sla_frozen_cells_count"] = "PASS" if frozen_match else "FAIL"

    # 2. Coverage counts vs GLORYS level 0
    # Expected:
    # SST analysed_sst: 107312 / 412 / 907
    # SLA sla: 107664 / 60 / 2383
    # SLA adt, ugos, vgos: 107393 / 331 / 1832
    expected_cov = {
        ("OSTIA_SST", "analysed_sst"): (107312, 412, 907),
        ("DUACS_SLA", "sla"): (107664, 60, 2383),
        ("DUACS_SLA", "adt"): (107393, 331, 1832),
        ("DUACS_SLA", "ugos"): (107393, 331, 1832),
        ("DUACS_SLA", "vgos"): (107393, 331, 1832),
    }

    cov_checks = {}
    all_cov_pass = True
    for (src, vn), (exp_fin_fin, exp_fin_fill, exp_dry_fin) in expected_cov.items():
        sub = df_v2_coverage[
            (df_v2_coverage["source"] == src) &
            (df_v2_coverage["variable"] == vn) &
            (df_v2_coverage["day_index"] == 0)
        ]
        if len(sub) == 0:
            cov_checks[f"coverage_{src}_{vn}"] = "FAIL: variable not found in v2 coverage"
            all_cov_pass = False
            continue
        row = sub.iloc[0]
        actual = (int(row["glorys_finite_source_finite"]), int(row["glorys_finite_source_fill"]), int(row["glorys_dry_source_finite"]))
        if actual == (exp_fin_fin, exp_fin_fill, exp_dry_fin):
            cov_checks[f"coverage_{src}_{vn}"] = f"PASS: {actual[0]} / {actual[1]} / {actual[2]}"
        else:
            cov_checks[f"coverage_{src}_{vn}"] = f"FAIL: expected {(exp_fin_fin, exp_fin_fill, exp_dry_fin)}, got {actual}"
            all_cov_pass = False

    results["coverage_checks"] = cov_checks
    results["all_coverage_checks_passed"] = all_cov_pass

    all_pass = (fill_match and non_fill_match and dec_min_match and dec_max_match and frozen_match and all_cov_pass)
    results["overall_regression_status"] = "PASS" if all_pass else "FAIL"

    return results


# ==============================================================================
# Master Pipeline Runner
# ==============================================================================

def run_v2_audit(config_path: str, output_dir: str) -> dict:
    """Execute complete generalized input contract audit v2 pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    glo_file = config["reference"]["glorys_path"]
    bathy_file = config["reference"]["glorys_static_bathy_path"]
    expected_sha_map = config["reference"]["expected_sha256"]

    # All six raw files
    raw_files = [
        "data/raw/sst/sst_ostia_2019-01.nc",
        "data/raw/sla/sla_duacs_2019-01.nc",
        "data/raw/sss/sss_multiobs_2019-01.nc",
        "data/raw/currents/currents_multiobs_2019-01.nc",
        glo_file,
        bathy_file,
    ]

    # Pre-audit integrity
    pre_integrity = {f: get_file_metadata(f) for f in raw_files}

    # Step 0: Record environment snapshot
    env_snapshot_path = os.path.join(output_dir, "environment_snapshot.txt")
    env_info = record_environment_snapshot(env_snapshot_path)

    # Reference dates from GLORYS
    glo_coords = audit_coordinates_and_time(glo_file)
    reference_dates = glo_coords["time"]["date_strings"]

    # Section A & B: Headers and coordinates for all sources
    header_reports = {}
    coords_reports = {"glorys": glo_coords}

    for src in config["sources"]:
        s_name = src["name"]
        s_path = src["path"]
        hdr_txt_name = f"{s_name.lower()}_header_dump.txt"
        hdr_txt_path = os.path.join(output_dir, hdr_txt_name)
        _, hdr_dict = dump_header_report(s_path, hdr_txt_path)
        header_reports[s_name] = hdr_dict
        coords_reports[s_name] = audit_coordinates_and_time(s_path, reference_dates=reference_dates)

    # Section C: Variable raw contract audit per source
    all_contract_dfs = []
    all_extremes_dfs = []
    all_frozen_dfs = []
    flag_mask_summaries = {}

    for src in config["sources"]:
        s_name = src["name"]
        s_path = src["path"]
        depth_sel = src.get("depth_selection", None)
        df_c, df_e, df_f, flags = audit_variable_raw_contract(
            s_path, s_name, depth_selection=depth_sel
        )
        all_contract_dfs.append(df_c)
        all_extremes_dfs.append(df_e)
        all_frozen_dfs.append(df_f)
        flag_mask_summaries[s_name] = flags

    df_contract = pd.concat(all_contract_dfs, ignore_index=True)
    df_extremes = pd.concat(all_extremes_dfs, ignore_index=True)
    df_frozen = pd.concat(all_frozen_dfs, ignore_index=True)

    contract_csv = os.path.join(output_dir, "variable_contract.csv")
    extremes_csv = os.path.join(output_dir, "variable_extremes.csv")
    frozen_csv = os.path.join(output_dir, "frozen_cells.csv")

    df_contract.to_csv(contract_csv, index=False)
    df_extremes.to_csv(extremes_csv, index=False)
    df_frozen.to_csv(frozen_csv, index=False)

    # Section D: Coverage against GLORYS level 0
    all_cov_dfs = []
    all_miss_dfs = []

    for src in config["sources"]:
        s_name = src["name"]
        s_path = src["path"]
        cov_vars = src.get("coverage_variables", [])
        if not cov_vars:
            continue
        df_cov, df_miss = audit_coverage_vs_glorys(
            s_path, s_name, cov_vars, glo_file, bathy_file
        )
        all_cov_dfs.append(df_cov)
        all_miss_dfs.append(df_miss)

    df_coverage = pd.concat(all_cov_dfs, ignore_index=True)
    df_missing = pd.concat(all_miss_dfs, ignore_index=True)

    coverage_csv = os.path.join(output_dir, "coverage_vs_glorys.csv")
    missing_csv = os.path.join(output_dir, "coverage_missing_cells.csv")

    df_coverage.to_csv(coverage_csv, index=False)
    df_missing.to_csv(missing_csv, index=False)

    # Section E: Grid Nesting Diagnostics (Step 4)
    nesting_rows = []
    nesting_sources = [
        ("OSTIA_SST", "data/raw/sst/sst_ostia_2019-01.nc"),
        ("DUACS_SLA", "data/raw/sla/sla_duacs_2019-01.nc"),
        ("MULTIOBS_SSS", "data/raw/sss/sss_multiobs_2019-01.nc"),
        ("MULTIOBS_CURRENTS", "data/raw/currents/currents_multiobs_2019-01.nc"),
        ("GLORYS_NATIVE", glo_file),
    ]

    glorys_1d_weights = None
    for s_name, s_path in nesting_sources:
        with h5netcdf.File(s_path, "r") as f:
            lat_n = "latitude" if "latitude" in f.variables else "lat"
            lon_n = "longitude" if "longitude" in f.variables else "lon"
            lat_c = np.array(f.variables[lat_n][:])
            lon_c = np.array(f.variables[lon_n][:])

        res_lat = compute_nesting_diagnostics(lat_c, target_start=5.0, target_step=0.25)
        res_lon = compute_nesting_diagnostics(lon_c, target_start=45.0, target_step=0.25)

        if s_name == "GLORYS_NATIVE":
            glorys_1d_weights = res_lat["weights"]

        nesting_rows.append({
            "source": s_name,
            "axis": "latitude",
            "native_min_centre": float(lat_c.min()),
            "native_max_centre": float(lat_c.max()),
            "native_step_deg": res_lat["step"],
            "target_start_edge": 5.0,
            "target_step_deg": 0.25,
            "edges_coincide": res_lat["coincide"],
            "native_cells_per_target_cell": res_lat["cells_per_target"],
            "edge_offset_native_units": res_lat["offset_native_units"],
            "1d_overlap_weights": json.dumps(res_lat["weights"]),
            "diagnostic_note": "Diagnostic metric only; not a regridding decision.",
        })
        nesting_rows.append({
            "source": s_name,
            "axis": "longitude",
            "native_min_centre": float(lon_c.min()),
            "native_max_centre": float(lon_c.max()),
            "native_step_deg": res_lon["step"],
            "target_start_edge": 45.0,
            "target_step_deg": 0.25,
            "edges_coincide": res_lon["coincide"],
            "native_cells_per_target_cell": res_lon["cells_per_target"],
            "edge_offset_native_units": res_lon["offset_native_units"],
            "1d_overlap_weights": json.dumps(res_lon["weights"]),
            "diagnostic_note": "Diagnostic metric only; not a regridding decision.",
        })

    df_nesting = pd.DataFrame(nesting_rows)
    nesting_csv = os.path.join(output_dir, "nesting_diagnostics.csv")
    df_nesting.to_csv(nesting_csv, index=False)

    # 2D area overlap weights for GLORYS (outer product of 1D weights)
    w1d = np.array(glorys_1d_weights if glorys_1d_weights is not None else [1/6, 1/3, 1/3, 1/6])
    w2d = np.outer(w1d, w1d)
    w2d_list = [[round(float(val), 6) for val in row] for row in w2d]

    # Section F: Step 3 Follow-up items
    sst_file = "data/raw/sst/sst_ostia_2019-01.nc"
    sla_file = "data/raw/sla/sla_duacs_2019-01.nc"
    step3a = audit_sst_frozen_error_cells(sst_file, glo_file, os.path.join(output_dir, "sst_frozen_error_cells.csv"))
    step3b = audit_sst_lake_cells(sst_file, glo_file, os.path.join(output_dir, "sst_lake_cells.csv"))
    step3c = audit_sla_zero_cells(sla_file, os.path.join(output_dir, "sla_zero_cells.csv"))
    step3d = audit_sla_velocity_extremes(sla_file, os.path.join(output_dir, "sla_velocity_extremes.csv"))

    # Section G: Regression acceptance vs v1
    v1_output_dir = "outputs/diagnostics/input_contract_audit"
    regression_results = check_regression_vs_v1(v1_output_dir, df_contract, df_coverage)

    # Section H: Diagnostic Plots (3 PNGs)
    sss_file = "data/raw/sss/sss_multiobs_2019-01.nc"
    currents_file = "data/raw/currents/currents_multiobs_2019-01.nc"
    plots = generate_audit_v2_plots(sss_file, currents_file, glo_file, output_dir)

    # Post-audit integrity verification
    post_integrity = {f: get_file_metadata(f) for f in raw_files}
    all_files_unchanged = True
    integrity_report = {}

    for f in raw_files:
        pre = pre_integrity[f]
        post = post_integrity[f]
        match = (pre["sha256"] == post["sha256"] and pre["size_bytes"] == post["size_bytes"] and pre["mtime_raw"] == post["mtime_raw"])
        if not match:
            all_files_unchanged = False
        integrity_report[f] = {
            "filename": os.path.basename(f),
            "sha256_before": pre["sha256"],
            "sha256_after": post["sha256"],
            "size_before": pre["size_bytes"],
            "size_after": post["size_bytes"],
            "mtime_before": pre["mtime_utc"],
            "mtime_after": post["mtime_utc"],
            "match": match,
        }

    # Summary JSON structure
    summary_json = {
        "analysis_name": "generalized_input_contract_audit_v2",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "integrity": {
            "all_files_unchanged": all_files_unchanged,
            "total_files_checked": len(raw_files),
            "files": integrity_report,
        },
        "environment": env_info,
        "coordinates_and_time": coords_reports,
        "nesting_diagnostics": {
            "sources": nesting_rows,
            "glorys_area_overlap_weights_1d": [round(float(w), 6) for w in w1d],
            "glorys_area_overlap_weights_2d_matrix": w2d_list,
            "disclaimer": "All nesting and overlap metrics are diagnostic observations only; no regridding decisions or weights application are performed."
        },
        "regression_acceptance_vs_v1": regression_results,
        "step3_followups": {
            "sst_frozen_error_cells": step3a,
            "sst_lake_cells": step3b,
            "sla_zero_cells": step3c,
            "sla_velocity_extremes": step3d,
        },
        "variable_contract_summary": {
            "total_variables_evaluated": len(df_contract["variable"].unique()),
            "variables": list(df_contract["variable"].unique()),
            "variables_with_depths": list(df_contract.groupby(["source", "variable", "depth_m"]).groups.keys()),
            "all_fill_masks_identical_over_time": bool(df_contract["fill_mask_identical_all_days"].all()),
            "total_valid_min_violations": int(df_contract["valid_min_violations"].sum()),
            "total_valid_max_violations": int(df_contract["valid_max_violations"].sum()),
            "total_decoded_exact_zeros": int(df_contract["decoded_exact_zeros"].sum()),
            "flag_masks": flag_mask_summaries,
        },
        "coverage_summary_day0": {
            "glorys_ocean_cells": int(df_coverage[df_coverage["day_index"] == 0]["glorys_finite_count"].iloc[0]),
            "by_variable": {
                f"{r['source']}_{r['variable']}": {
                    "glorys_finite_source_finite": int(r["glorys_finite_source_finite"]),
                    "glorys_finite_source_fill": int(r["glorys_finite_source_fill"]),
                    "glorys_dry_source_finite": int(r["glorys_dry_source_finite"]),
                    "glorys_dry_source_fill": int(r["glorys_dry_source_fill"]),
                }
                for _, r in df_coverage[df_coverage["day_index"] == 0].iterrows()
            }
        },
        "diagnostic_plots": plots,
        "unknown_fields": {
            "tpa_correction_availability": None,
            "reason_for_tpa_correction_unknown": "tpa_correction variable in DUACS NetCDF holds _FillValue (-2147483647) on all 31 days and metadata states 'NOT IMPLEMENTED YET: This correction is not yet available for the current product version'.",
            "future_regridding_interpolation_weights": None,
            "reason_for_future_regridding_weights_unknown": "Task scope is strictly READ-ONLY data contract audit without defining or executing interpolation schemes; overlap weights are diagnostic observations only.",
            "currents_mask_definition_coastal_rule": None,
            "reason_for_currents_mask_rule_unknown": "Dataset cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m lacks an explicit mask variable; land/coast boundaries must be inferred from _FillValue patterns, and no mask rule is defined in this diagnostic phase.",
            "sss_temporal_mask_variance_cause": None,
            "reason_for_sss_mask_variance_unknown": "SSS multiobs fill count varies dynamically across January 2019 (48,049 to 49,456 fill cells per day); underlying satellite swaths and optimal interpolation quality flag cutoffs are not stored in the product file."
        }
    }

    summary_json_path = os.path.join(output_dir, "audit_v2_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as fp:
        json.dump(summary_json, fp, indent=2, default=str)

    # Terminal output
    print("\n" + "=" * 80)
    print("GENERALIZED INPUT CONTRACT AUDIT V2 COMPLETE")
    print("=" * 80)
    print(f"Summary JSON: {summary_json_path}")
    print(f"Regression acceptance: {regression_results['overall_regression_status']}")
    print(f"All 6 files unchanged: {all_files_unchanged}")
    for f, rep in integrity_report.items():
        print(f"  {rep['filename']}: sha256={rep['sha256_before']}, match={rep['match']}")

    return summary_json


if __name__ == "__main__":
    cfg = "configs/input_sources_pilot.json"
    out = "outputs/diagnostics/input_contract_audit_v2"
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    run_v2_audit(cfg, out)
