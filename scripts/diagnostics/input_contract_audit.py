"""
GLORYS, OSTIA SST, and DUACS SLA Input Data Contract Audit (January 2019).
READ-ONLY on raw data. No regridding. No interpolation decisions.
"""

import os
import sys
import re
import datetime
import hashlib
import numpy as np
import pandas as pd
import h5netcdf
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
    long_name = str(attrs.get("long_name", "")).lower()
    name_lower = name.lower()

    if name_lower in ["latitude", "longitude", "time", "lat", "lon", "depth", "deptho", "deptho_lev"] or "axis" in attrs:
        evidence = f"coordinate axis '{attrs.get('axis', name)}' with standard_name='{attrs.get('standard_name', '')}'"
        return "coordinate", evidence

    if "flag_values" in attrs or "flag_masks" in attrs or "flag_meanings" in attrs or "status_flag" in std_name or "mask" in name_lower:
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
    Audit coordinates (lat, lon, time) and calculate grid alignment against project target.
    """
    with h5netcdf.File(file_path, "r") as f:
        # Determine latitude variable name
        lat_name = "latitude" if "latitude" in f.variables else "lat"
        lon_name = "longitude" if "longitude" in f.variables else "lon"

        lat = np.array(f.variables[lat_name][:])
        lon = np.array(f.variables[lon_name][:])
        time_raw = np.array(f.variables["time"][:])
        time_attrs = dict(f.variables["time"].attrs)

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

    # Distance from target centres to nearest native coordinate
    lat_dist_to_target = np.min(np.abs(lat[:, None] - TARGET_LAT_CENTRES[None, :]), axis=0)
    lat_exact = bool(np.all(lat_dist_to_target < 1e-4))
    lat_offset_cells = float(np.mean(lat_dist_to_target / d_lat_abs))

    lon_dist_to_target = np.min(np.abs(lon[:, None] - TARGET_LON_CENTRES[None, :]), axis=0)
    lon_exact = bool(np.all(lon_dist_to_target < 1e-4))
    lon_offset_cells = float(np.mean(lon_dist_to_target / d_lon_abs))

    # Native cells inside target domain (5-30N, 45-105E)
    lat_in_domain = int(np.sum((lat >= DOMAIN_LAT_BOUNDS[0]) & (lat <= DOMAIN_LAT_BOUNDS[1])))
    lon_in_domain = int(np.sum((lon >= DOMAIN_LON_BOUNDS[0]) & (lon <= DOMAIN_LON_BOUNDS[1])))
    native_cells_in_domain = lat_in_domain * lon_in_domain

    return {
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


def audit_variable_raw_contract(
    file_path: str,
    source_name: str,
    target_var_names: list[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Audit raw data variables across all days:
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

        all_vars = list(f.variables.keys()) if target_var_names is None else target_var_names

        for vname in all_vars:
            if vname in [lat_name, lon_name, "time"]:
                continue

            var = f.variables[vname]
            v_attrs = dict(var.attrs)
            dims = var.dimensions
            shape = var.shape

            # Skip 1D variables without spatial dimensions (e.g. tpa_correction) from spatial contract
            if len(shape) < 3:
                # Capture summary for 1D variables
                raw_1d = np.array(var[:])
                fill_val = v_attrs.get("_FillValue", None)
                continue

            raw_data = np.array(var[:])  # shape: (n_time, n_lat, n_lon)
            n_time, n_lat, n_lon = shape
            total_spatial_cells = n_lat * n_lon

            fill_val = v_attrs.get("_FillValue", None)
            scale = float(v_attrs.get("scale_factor", 1.0))
            offset = float(v_attrs.get("add_offset", 0.0))
            vmin = v_attrs.get("valid_min", None)
            vmax = v_attrs.get("valid_max", None)

            # Check if variable is flag/mask
            cls, _ = classify_variable(vname, v_attrs, dims)
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
                    # Construct decoded 2D array with NaNs in fill
                    day_dec_2d = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
                    day_dec_2d[day_non_fill] = day_raw[day_non_fill].astype(np.float64) * scale + offset

                    # Flatten valid coordinates and values
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

        # Precompute nearest neighbour coordinate indices for GLORYS grid centres
        lat_nn_indices = np.argmin(np.abs(src_lat[:, None] - glo_lat[None, :]), axis=0)
        lon_nn_indices = np.argmin(np.abs(src_lon[:, None] - glo_lon[None, :]), axis=0)

        for vname in var_names:
            if vname not in f_src.variables:
                continue
            var = f_src.variables[vname]
            v_fill = var.attrs.get("_FillValue", None)
            raw_3d = np.array(var[:])
            n_time = raw_3d.shape[0]

            for t in range(n_time):
                date_str = date_strs[t]
                day_raw = raw_3d[t]

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


def generate_audit_plots(
    sst_path: str,
    sla_path: str,
    glorys_path: str,
    out_dir: str
) -> list[str]:
    """Generate titled, labeled diagnostic PNG plots."""
    os.makedirs(out_dir, exist_ok=True)
    generated_plots = []

    # 1. Day-0 Maps for Main Data Variables
    # SST analysed_sst (convert Kelvin to Celsius for readability)
    with h5netcdf.File(sst_path, "r") as f_sst:
        sst_lat = np.array(f_sst.variables["latitude"][:])
        sst_lon = np.array(f_sst.variables["longitude"][:])
        sst_raw = np.array(f_sst.variables["analysed_sst"][0, :, :])
        sst_fill = f_sst.variables["analysed_sst"].attrs["_FillValue"]
        sst_scale = float(f_sst.variables["analysed_sst"].attrs["scale_factor"])
        sst_offset = float(f_sst.variables["analysed_sst"].attrs["add_offset"])
        sst_degC = np.where(sst_raw != sst_fill, sst_raw.astype(np.float64) * sst_scale + sst_offset - 273.15, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    mesh = ax.pcolormesh(sst_lon, sst_lat, sst_degC, cmap="turbo", shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, label="Analysed SST (°C)")
    ax.set_title("OSTIA L4 Reprocessed SST — Day 0 (2019-01-01)\nDomain: 5–30°N, 45–105°E (0.05° native)", fontsize=11)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path_sst = os.path.join(out_dir, "plot_sst_analysed_sst_day0.png")
    fig.savefig(plot_path_sst)
    plt.close(fig)
    generated_plots.append(plot_path_sst)

    # SLA main variables: sla, adt, ugos, vgos
    with h5netcdf.File(sla_path, "r") as f_sla:
        sla_lat = np.array(f_sla.variables["latitude"][:])
        sla_lon = np.array(f_sla.variables["longitude"][:])

        variables_to_plot = [
            ("sla", "Sea Level Anomaly (m)", "RdBu_r"),
            ("adt", "Absolute Dynamic Topography (m)", "viridis"),
            ("ugos", "Zonal Geostrophic Velocity (m/s)", "coolwarm"),
            ("vgos", "Meridional Geostrophic Velocity (m/s)", "coolwarm"),
        ]

        for vname, vlabel, cmap in variables_to_plot:
            var = f_sla.variables[vname]
            raw = np.array(var[0, :, :])
            fill = var.attrs["_FillValue"]
            scale = float(var.attrs["scale_factor"])
            dec = np.where(raw != fill, raw.astype(np.float64) * scale, np.nan)

            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
            mesh = ax.pcolormesh(sla_lon, sla_lat, dec, cmap=cmap, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax, label=vlabel)
            ax.set_title(f"DUACS L4 Sea Level: {vname.upper()} — Day 0 (2019-01-01)\nDomain: 5–30°N, 45–105°E (0.125° native)", fontsize=11)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            p_path = os.path.join(out_dir, f"plot_sla_{vname}_day0.png")
            fig.savefig(p_path)
            plt.close(fig)
            generated_plots.append(p_path)

    # 2. Per-Day Domain Mean Time Series (January 2019)
    with h5netcdf.File(sst_path, "r") as f_sst:
        sst_all = np.array(f_sst.variables["analysed_sst"][:])
        fill_sst = f_sst.variables["analysed_sst"].attrs["_FillValue"]
        sc_sst = float(f_sst.variables["analysed_sst"].attrs["scale_factor"])
        off_sst = float(f_sst.variables["analysed_sst"].attrs["add_offset"])
        sst_means_degC = []
        for t in range(31):
            valid = (sst_all[t] != fill_sst)
            sst_means_degC.append(float(np.mean(sst_all[t, valid].astype(np.float64) * sc_sst + off_sst - 273.15)))

    with h5netcdf.File(sla_path, "r") as f_sla:
        sla_means = {}
        for vn in ["sla", "adt", "ugos", "vgos"]:
            arr = np.array(f_sla.variables[vn][:])
            fl = f_sla.variables[vn].attrs["_FillValue"]
            sc = float(f_sla.variables[vn].attrs["scale_factor"])
            means = []
            for t in range(31):
                valid = (arr[t] != fl)
                means.append(float(np.mean(arr[t, valid].astype(np.float64) * sc)))
            sla_means[vn] = means

    days = np.arange(1, 32)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=150)

    # SST mean
    axes[0].plot(days, sst_means_degC, marker="o", color="tab:red", linewidth=1.5, label="Analysed SST (°C)")
    axes[0].set_ylabel("SST (°C)")
    axes[0].set_title("OSTIA Reprocessed SST: Domain Mean (January 2019)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # SLA / ADT mean
    axes[1].plot(days, sla_means["sla"], marker="s", color="tab:blue", linewidth=1.5, label="SLA (m)")
    axes[1].plot(days, sla_means["adt"], marker="^", color="tab:green", linewidth=1.5, label="ADT (m)")
    axes[1].set_ylabel("Height (m)")
    axes[1].set_title("DUACS Sea Level: Domain Mean SLA and ADT (January 2019)", fontsize=10)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Velocities mean
    axes[2].plot(days, sla_means["ugos"], marker="v", color="tab:purple", linewidth=1.5, label="ugos (zonal m/s)")
    axes[2].plot(days, sla_means["vgos"], marker="d", color="tab:orange", linewidth=1.5, label="vgos (meridional m/s)")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_xlabel("Day of Month (January 2019)")
    axes[2].set_title("DUACS Geostrophic Velocities: Domain Mean ugos and vgos", fontsize=10)
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_ts_path = os.path.join(out_dir, "plot_domain_means_timeseries.png")
    fig.savefig(plot_ts_path)
    plt.close(fig)
    generated_plots.append(plot_ts_path)

    # 3. Map of GLORYS-Finite / Source-Fill Cells (Missing Ocean Points)
    with h5netcdf.File(glorys_path, "r") as f_glo:
        glo_lat = np.array(f_glo.variables["latitude"][:])
        glo_lon = np.array(f_glo.variables["longitude"][:])
        glo_finite = (np.array(f_glo.variables["thetao"][0, 0, :, :]) != -32767)

    with h5netcdf.File(sst_path, "r") as f_sst:
        sst_lat = np.array(f_sst.variables["latitude"][:])
        sst_lon = np.array(f_sst.variables["longitude"][:])
        sst_fin = (np.array(f_sst.variables["analysed_sst"][0, :, :]) != f_sst.variables["analysed_sst"].attrs["_FillValue"])
        lat_nn_sst = np.argmin(np.abs(sst_lat[:, None] - glo_lat[None, :]), axis=0)
        lon_nn_sst = np.argmin(np.abs(sst_lon[:, None] - glo_lon[None, :]), axis=0)
        sst_at_glo = sst_fin[lat_nn_sst[:, None], lon_nn_sst[None, :]]
        sst_missing_on_glo = glo_finite & ~sst_at_glo

    with h5netcdf.File(sla_path, "r") as f_sla:
        sla_lat = np.array(f_sla.variables["latitude"][:])
        sla_lon = np.array(f_sla.variables["longitude"][:])
        sla_fin = (np.array(f_sla.variables["sla"][0, :, :]) != f_sla.variables["sla"].attrs["_FillValue"])
        lat_nn_sla = np.argmin(np.abs(sla_lat[:, None] - glo_lat[None, :]), axis=0)
        lon_nn_sla = np.argmin(np.abs(sla_lon[:, None] - glo_lon[None, :]), axis=0)
        sla_at_glo = sla_fin[lat_nn_sla[:, None], lon_nn_sla[None, :]]
        sla_missing_on_glo = glo_finite & ~sla_at_glo

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)

    # SST missing
    sst_miss_r, sst_miss_c = np.where(sst_missing_on_glo)
    axes[0].scatter(glo_lon[sst_miss_c], glo_lat[sst_miss_r], s=6, color="red", alpha=0.7)
    axes[0].set_title(f"OSTIA SST Fill over GLORYS Ocean (N={len(sst_miss_r)})\nNearest-neighbour sampled", fontsize=10)
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_xlim(44.5, 105.5)
    axes[0].set_ylim(4.5, 30.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # SLA missing
    sla_miss_r, sla_miss_c = np.where(sla_missing_on_glo)
    axes[1].scatter(glo_lon[sla_miss_c], glo_lat[sla_miss_r], s=6, color="blue", alpha=0.7)
    axes[1].set_title(f"DUACS SLA Fill over GLORYS Ocean (N={len(sla_miss_r)})\nNearest-neighbour sampled", fontsize=10)
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_xlim(44.5, 105.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_miss_path = os.path.join(out_dir, "plot_coverage_missing_vs_glorys.png")
    fig.savefig(plot_miss_path)
    plt.close(fig)
    generated_plots.append(plot_miss_path)

    return generated_plots


def run_full_audit(
    sst_path: str,
    sla_path: str,
    glorys_path: str,
    static_bathy_path: str,
    output_dir: str
) -> dict:
    """Run master audit pipeline on real data, preserving strict data immutability."""
    os.makedirs(output_dir, exist_ok=True)
    all_input_files = [sst_path, sla_path, glorys_path, static_bathy_path]

    # Pre-audit integrity
    pre_integrity = {f: get_file_metadata(f) for f in all_input_files}

    # Section A: Header dumps
    sst_hdr_path = os.path.join(output_dir, "sst_header_dump.txt")
    sla_hdr_path = os.path.join(output_dir, "sla_header_dump.txt")
    _, sst_hdr_dict = dump_header_report(sst_path, sst_hdr_path)
    _, sla_hdr_dict = dump_header_report(sla_path, sla_hdr_path)

    # Section B: Coordinates and time
    # Establish reference dates from GLORYS
    glo_coords = audit_coordinates_and_time(glorys_path)
    reference_dates = glo_coords["time"]["date_strings"]

    sst_coords = audit_coordinates_and_time(sst_path, reference_dates=reference_dates)
    sla_coords = audit_coordinates_and_time(sla_path, reference_dates=reference_dates)

    # Section C: Raw-value contract per variable
    df_sst_contract, df_sst_extremes, df_sst_frozen, sst_flag_summary = audit_variable_raw_contract(
        sst_path, "OSTIA_SST"
    )
    df_sla_contract, df_sla_extremes, df_sla_frozen, sla_flag_summary = audit_variable_raw_contract(
        sla_path, "DUACS_SLA"
    )

    df_contract = pd.concat([df_sst_contract, df_sla_contract], ignore_index=True)
    df_extremes = pd.concat([df_sst_extremes, df_sla_extremes], ignore_index=True)
    df_frozen = pd.concat([df_sst_frozen, df_sla_frozen], ignore_index=True)

    contract_csv_path = os.path.join(output_dir, "variable_contract.csv")
    extremes_csv_path = os.path.join(output_dir, "variable_extremes.csv")
    frozen_csv_path = os.path.join(output_dir, "frozen_cells.csv")

    df_contract.to_csv(contract_csv_path, index=False)
    df_extremes.to_csv(extremes_csv_path, index=False)
    df_frozen.to_csv(frozen_csv_path, index=False)

    # Section D: Coverage against GLORYS level 0
    df_sst_cov, df_sst_miss = audit_coverage_vs_glorys(
        sst_path, "OSTIA_SST", ["analysed_sst"], glorys_path, static_bathy_path
    )
    df_sla_cov, df_sla_miss = audit_coverage_vs_glorys(
        sla_path, "DUACS_SLA", ["sla", "adt", "ugos", "vgos"], glorys_path, static_bathy_path
    )

    df_coverage = pd.concat([df_sst_cov, df_sla_cov], ignore_index=True)
    df_missing = pd.concat([df_sst_miss, df_sla_miss], ignore_index=True)

    coverage_csv_path = os.path.join(output_dir, "coverage_vs_glorys.csv")
    missing_csv_path = os.path.join(output_dir, "coverage_missing_cells.csv")

    df_coverage.to_csv(coverage_csv_path, index=False)
    df_missing.to_csv(missing_csv_path, index=False)

    # Section E: Diagnostic Plots
    plot_paths = generate_audit_plots(sst_path, sla_path, glorys_path, output_dir)

    # Post-audit integrity verification
    post_integrity = {f: get_file_metadata(f) for f in all_input_files}
    all_files_unchanged = True
    integrity_report = {}

    for f in all_input_files:
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

    # Section F: JSON summary
    summary_json = {
        "analysis_name": "sst_sla_pilot_contract_audit",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "integrity": {
            "all_files_unchanged": all_files_unchanged,
            "total_files_checked": len(all_input_files),
            "files": integrity_report,
        },
        "coordinates_and_time": {
            "glorys": glo_coords,
            "ostia_sst": sst_coords,
            "duacs_sla": sla_coords,
        },
        "variable_contract_summary": {
            "total_variables_evaluated": len(df_contract["variable"].unique()),
            "variables": list(df_contract["variable"].unique()),
            "all_fill_masks_identical_over_time": bool(df_contract["fill_mask_identical_all_days"].all()),
            "total_valid_min_violations": int(df_contract["valid_min_violations"].sum()),
            "total_valid_max_violations": int(df_contract["valid_max_violations"].sum()),
            "total_decoded_exact_zeros": int(df_contract["decoded_exact_zeros"].sum()),
            "variables_with_frozen_cells": {
                v: int(df_contract[df_contract["variable"] == v]["frozen_cells_count"].iloc[0])
                for v in df_contract["variable"].unique()
                if df_contract[df_contract["variable"] == v]["frozen_cells_count"].iloc[0] > 0
            },
            "flag_masks": {
                "sst": sst_flag_summary,
                "sla": sla_flag_summary,
            },
        },
        "coverage_summary_day0": {
            "glorys_ocean_cells": int(df_coverage[df_coverage["day_index"] == 0]["glorys_finite_count"].iloc[0]),
            "sst_analysed_sst": {
                "glorys_finite_source_finite": int(df_coverage[(df_coverage["variable"] == "analysed_sst") & (df_coverage["day_index"] == 0)]["glorys_finite_source_finite"].iloc[0]),
                "glorys_finite_source_fill": int(df_coverage[(df_coverage["variable"] == "analysed_sst") & (df_coverage["day_index"] == 0)]["glorys_finite_source_fill"].iloc[0]),
                "glorys_dry_source_finite": int(df_coverage[(df_coverage["variable"] == "analysed_sst") & (df_coverage["day_index"] == 0)]["glorys_dry_source_finite"].iloc[0]),
                "glorys_dry_source_fill": int(df_coverage[(df_coverage["variable"] == "analysed_sst") & (df_coverage["day_index"] == 0)]["glorys_dry_source_fill"].iloc[0]),
            },
            "sla_sla": {
                "glorys_finite_source_finite": int(df_coverage[(df_coverage["variable"] == "sla") & (df_coverage["day_index"] == 0)]["glorys_finite_source_finite"].iloc[0]),
                "glorys_finite_source_fill": int(df_coverage[(df_coverage["variable"] == "sla") & (df_coverage["day_index"] == 0)]["glorys_finite_source_fill"].iloc[0]),
                "glorys_dry_source_finite": int(df_coverage[(df_coverage["variable"] == "sla") & (df_coverage["day_index"] == 0)]["glorys_dry_source_finite"].iloc[0]),
                "glorys_dry_source_fill": int(df_coverage[(df_coverage["variable"] == "sla") & (df_coverage["day_index"] == 0)]["glorys_dry_source_fill"].iloc[0]),
            },
            "sla_adt": {
                "glorys_finite_source_finite": int(df_coverage[(df_coverage["variable"] == "adt") & (df_coverage["day_index"] == 0)]["glorys_finite_source_finite"].iloc[0]),
                "glorys_finite_source_fill": int(df_coverage[(df_coverage["variable"] == "adt") & (df_coverage["day_index"] == 0)]["glorys_finite_source_fill"].iloc[0]),
                "glorys_dry_source_finite": int(df_coverage[(df_coverage["variable"] == "adt") & (df_coverage["day_index"] == 0)]["glorys_dry_source_finite"].iloc[0]),
                "glorys_dry_source_fill": int(df_coverage[(df_coverage["variable"] == "adt") & (df_coverage["day_index"] == 0)]["glorys_dry_source_fill"].iloc[0]),
            },
        },
        "unknown_fields": {
            "tpa_correction_availability": None,
            "reason_for_tpa_correction_unknown": "tpa_correction variable in DUACS NetCDF holds _FillValue (-2147483647) on all 31 days and metadata states 'NOT IMPLEMENTED YET: This correction is not yet available for the current product version'.",
            "future_regridding_interpolation_weights": None,
            "reason_for_future_regridding_weights_unknown": "Task scope is strictly READ-ONLY data contract audit without defining or executing interpolation schemes; interpolation weights are reserved for future preprocessing steps."
        },
    }

    summary_json_path = os.path.join(output_dir, "audit_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as fp:
        import json
        json.dump(summary_json, fp, indent=2)

    # Terminal Report Output
    print("\n" + "=" * 80)
    print("SECTION A: Header Dumps")
    print("=" * 80)
    print(f"  Wrote: {sst_hdr_path}")
    print(f"  Wrote: {sla_hdr_path}")
    print(f"  OSTIA SST variables: {list(sst_hdr_dict['variables'].keys())}")
    print(f"  DUACS SLA variables: {list(sla_hdr_dict['variables'].keys())}")

    print("\n" + "=" * 80)
    print("SECTION B: Coordinates and Time Audit")
    print("=" * 80)
    for src_name, coords in [("GLORYS Reference", glo_coords), ("OSTIA SST", sst_coords), ("DUACS SLA", sla_coords)]:
        print(f"[{src_name}]")
        print(f"  Latitude:  min={coords['latitude']['min']:.4f}, max={coords['latitude']['max']:.4f}, step={coords['latitude']['step']:.4f}, count={coords['latitude']['count']}, dir={coords['latitude']['direction']}")
        print(f"  Longitude: min={coords['longitude']['min']:.4f}, max={coords['longitude']['max']:.4f}, step={coords['longitude']['step']:.4f}, count={coords['longitude']['count']}, dir={coords['longitude']['direction']}, conv={coords['longitude']['convention']}")
        print(f"  Time:      count={coords['time']['count']} days, units='{coords['time']['units']}', first='{coords['time']['first_timestamp']}', last='{coords['time']['last_timestamp']}', hours={coords['time']['hours_of_day']}")
        print(f"  Dates equal GLORYS Jan 2019: {coords['time']['dates_equal_reference']}")
        print(f"  Target Grid Alignment: exact_lat={coords['target_grid_alignment']['lat_exact_within_1e-4']} (offset={coords['target_grid_alignment']['lat_offset_cells']:.4f} cells), exact_lon={coords['target_grid_alignment']['lon_exact_within_1e-4']} (offset={coords['target_grid_alignment']['lon_offset_cells']:.4f} cells), cells_in_domain={coords['target_grid_alignment']['native_cells_inside_target_domain']}")

    print("\n" + "=" * 80)
    print("SECTION C: Raw-Value Contract Summary")
    print("=" * 80)
    print(f"  Evaluated {len(df_contract['variable'].unique())} variables across 31 days.")
    print(f"  All fill masks identical over time: {summary_json['variable_contract_summary']['all_fill_masks_identical_over_time']}")
    print(f"  Total valid_min violations: {summary_json['variable_contract_summary']['total_valid_min_violations']}")
    print(f"  Total valid_max violations: {summary_json['variable_contract_summary']['total_valid_max_violations']}")
    print(f"  Total decoded exact zeros: {summary_json['variable_contract_summary']['total_decoded_exact_zeros']}")
    print(f"  Variables with frozen non-fill cells: {summary_json['variable_contract_summary']['variables_with_frozen_cells']}")
    print(f"  Wrote: {contract_csv_path}")
    print(f"  Wrote: {extremes_csv_path}")
    print(f"  Wrote: {frozen_csv_path}")

    print("\n" + "=" * 80)
    print("SECTION D: Coverage Against GLORYS Level 0")
    print("=" * 80)
    print(f"  Total GLORYS Ocean cells (Level 0): {summary_json['coverage_summary_day0']['glorys_ocean_cells']}")
    for vname in ["analysed_sst", "sla", "adt"]:
        sub = df_coverage[(df_coverage["variable"] == vname) & (df_coverage["day_index"].isin([0, 15, 30]))]
        print(f"\n  Coverage: {vname}")
        for _, r in sub.iterrows():
            print(f"    Day {r['day_index']:02d} ({r['date']}): GLORYS-fin & src-fin: {r['glorys_finite_source_finite']:,} | GLORYS-fin & src-fill: {r['glorys_finite_source_fill']:,} | GLORYS-dry & src-fin: {r['glorys_dry_source_finite']:,}")
    print(f"  Wrote: {coverage_csv_path}")
    print(f"  Wrote: {missing_csv_path}")

    print("\n" + "=" * 80)
    print("SECTION E: Diagnostic Plots")
    print("=" * 80)
    for p in plot_paths:
        print(f"  Generated Plot: {p}")

    print("\n" + "=" * 80)
    print("INTEGRITY: SHA256 Before vs After")
    print("=" * 80)
    for f, rep in integrity_report.items():
        print(f"  {rep['filename']}:")
        print(f"    before: {rep['sha256_before']} (size={rep['size_before']})")
        print(f"    after:  {rep['sha256_after']} (size={rep['size_after']})")
        print(f"    match:  {rep['match']}")
    print(f"\nAll files unchanged: {all_files_unchanged}")
    print(f"Wrote master JSON: {summary_json_path}")

    return summary_json


if __name__ == "__main__":
    sst_file = "data/raw/sst/sst_ostia_2019-01.nc"
    sla_file = "data/raw/sla/sla_duacs_2019-01.nc"
    glo_file = "data/raw/glorys/glorys_thetao_2019-01.nc"
    bathy_file = "data/raw/glorys_static/glorys_static_mask_bathy.nc"
    out_dir = "outputs/diagnostics/input_contract_audit"

    print("================================================================================")
    print("GLORYS, OSTIA SST, and DUACS SLA Pilot Data Contract Audit (January 2019)")
    print("================================================================================")
    summary = run_full_audit(sst_file, sla_file, glo_file, bathy_file, out_dir)

