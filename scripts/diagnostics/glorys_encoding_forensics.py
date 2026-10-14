#!/usr/bin/env python3
"""
GLORYS thetao Forensic Encoding Analysis
Data-validation read-only forensic script for GLORYS NetCDF products.

This script gathers forensic evidence regarding raw integer encodings,
metadata attributes, mask time-invariance, suspect column profiles,
level-wise value context, and column depth anomalies without modifying
or rewriting any raw data files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Support both netCDF4 (if installed) and h5netcdf
try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    netCDF4 = None
    HAVE_NETCDF4 = False

import h5netcdf
import xarray as xr


# =============================================================================
# Helper: File Integrity (Read-Only Checksum and Metadata)
# =============================================================================

def compute_file_metadata(filepath: str) -> Dict[str, Any]:
    """Compute SHA256, size (bytes), and mtime for a file without modifying it."""
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
        "sha256": sha256.hexdigest().lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": mtime_dt.isoformat(),
    }


# =============================================================================
# Raw Dataset Wrapper
# =============================================================================

class RawNetCDFReader:
    """Read-only NetCDF reader returning raw integers and raw attributes."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._nc = None
        self._h5 = None
        self._mode = None

    def __enter__(self):
        if HAVE_NETCDF4:
            self._nc = netCDF4.Dataset(self.filepath, mode="r")
            self._nc.set_auto_maskandscale(False)
            self._mode = "netCDF4"
        else:
            self._h5 = h5netcdf.File(self.filepath, mode="r")
            self._mode = "h5netcdf"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._nc is not None:
            self._nc.close()
            self._nc = None
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def mode(self) -> str:
        return self._mode

    def get_global_attrs(self) -> Dict[str, Any]:
        if self._mode == "netCDF4":
            attrs = {}
            for k in self._nc.ncattrs():
                v = self._nc.getncattr(k)
                attrs[k] = v.item() if hasattr(v, "item") else v
            return attrs
        else:
            attrs = {}
            for k, v in self._h5.attrs.items():
                attrs[k] = v.item() if hasattr(v, "item") else v
            return attrs

    def get_dimensions(self) -> Dict[str, int]:
        if self._mode == "netCDF4":
            return {k: len(v) for k, v in self._nc.dimensions.items()}
        else:
            return {k: len(v) for k, v in self._h5.dimensions.items()}

    def get_variables(self) -> List[str]:
        if self._mode == "netCDF4":
            return list(self._nc.variables.keys())
        else:
            return list(self._h5.variables.keys())

    def get_var_attrs(self, varname: str) -> Dict[str, Any]:
        if self._mode == "netCDF4":
            var = self._nc.variables[varname]
            attrs = {}
            for k in var.ncattrs():
                v = var.getncattr(k)
                attrs[k] = v.item() if hasattr(v, "item") else v
            return attrs
        else:
            var = self._h5[varname]
            attrs = {}
            for k, v in var.attrs.items():
                attrs[k] = v.item() if hasattr(v, "item") else v
            return attrs

    def get_var_metadata(self, varname: str) -> Dict[str, Any]:
        if self._mode == "netCDF4":
            var = self._nc.variables[varname]
            dtype = str(var.dtype)
            shape = list(var.shape)
            dims = list(var.dimensions)
            chunking = var.chunking() if hasattr(var, "chunking") else None
            filters = var.filters() if hasattr(var, "filters") else None
            endian = var.endian() if hasattr(var, "endian") else None
        else:
            var = self._h5[varname]
            dtype = str(var.dtype)
            shape = list(var.shape)
            dims = list(var.dimensions)
            chunking = var.chunks
            filters = {"compression": var.compression, "compression_opts": var.compression_opts}
            endian = "little" if var.dtype.byteorder in ("<", "=") else "big"

        attrs = self.get_var_attrs(varname)
        return {
            "name": varname,
            "dtype": dtype,
            "shape": shape,
            "dimensions": dims,
            "chunking": chunking,
            "filters": filters,
            "endianness": endian,
            "attributes": attrs,
        }

    def get_raw_variable(self, varname: str) -> np.ndarray:
        """Read entire variable array raw without scaling or masking."""
        if self._mode == "netCDF4":
            return np.asarray(self._nc.variables[varname][:])
        else:
            return np.asarray(self._h5[varname][:])

    def get_raw_slice(self, varname: str, key: Any) -> np.ndarray:
        """Read a slice of variable raw without scaling or masking."""
        if self._mode == "netCDF4":
            return np.asarray(self._nc.variables[varname][key])
        else:
            return np.asarray(self._h5[varname][key])


# =============================================================================
# Lookup Functions with Strict Tolerance
# =============================================================================

def lookup_nearest_index(
    coords: np.ndarray,
    target_val: float,
    coord_name: str,
    tolerance: float,
) -> Tuple[int, float]:
    """Find index of nearest coordinate value; raise ValueError if beyond tolerance."""
    coords_arr = np.asarray(coords)
    diffs = np.abs(coords_arr - target_val)
    min_idx = int(np.argmin(diffs))
    min_diff = float(diffs[min_idx])
    nearest_val = float(coords_arr[min_idx])

    if min_diff > tolerance:
        raise ValueError(
            f"Nearest {coord_name} to target {target_val} is {nearest_val} "
            f"(diff={min_diff:.6f}), which exceeds tolerance {tolerance:.6f}."
        )
    return min_idx, nearest_val


# =============================================================================
# Section A: Header Dump
# =============================================================================

def generate_header_dump(reader: RawNetCDFReader, filepath: str, output_path: str) -> str:
    """Generate verbatim header dump text file."""
    meta = compute_file_metadata(filepath)
    global_attrs = reader.get_global_attrs()
    dims = reader.get_dimensions()
    vars_list = reader.get_variables()

    lines = []
    lines.append("=" * 80)
    lines.append("GLORYS THETAO ENCODING FORENSIC HEADER DUMP")
    lines.append("=" * 80)
    lines.append(f"File: {meta['filepath']}")
    lines.append(f"Size: {meta['size_bytes']} bytes")
    lines.append(f"SHA256: {meta['sha256']}")
    lines.append(f"MTime (UTC): {meta['mtime_utc']}")
    lines.append(f"Reader backend: {reader.mode}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("GLOBAL ATTRIBUTES (Verbatim)")
    lines.append("-" * 80)
    for k, v in global_attrs.items():
        lines.append(f"{k}: {repr(v)}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("DIMENSIONS")
    lines.append("-" * 80)
    for k, v in dims.items():
        lines.append(f"  {k} = {v}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("VARIABLES AND ATTRIBUTES")
    lines.append("-" * 80)
    for vname in vars_list:
        vmeta = reader.get_var_metadata(vname)
        lines.append(f"Variable: {vname}")
        lines.append(f"  dtype: {vmeta['dtype']}")
        lines.append(f"  shape: {vmeta['shape']}")
        lines.append(f"  dimensions: {vmeta['dimensions']}")
        lines.append(f"  chunking: {vmeta['chunking']}")
        lines.append(f"  filters: {vmeta['filters']}")
        lines.append(f"  endianness: {vmeta['endianness']}")
        lines.append("  Attributes:")
        for ak, av in vmeta["attributes"].items():
            lines.append(f"    {ak}: {repr(av)} (type: {type(av).__name__})")
        lines.append("")

    # Thetao attributes checklist
    lines.append("-" * 80)
    lines.append("THETAO ATTRIBUTES CHECKLIST")
    lines.append("-" * 80)
    thetao_attrs = reader.get_var_attrs("thetao")
    checklist_keys = [
        "scale_factor",
        "add_offset",
        "_FillValue",
        "missing_value",
        "valid_min",
        "valid_max",
        "valid_range",
        "flag_values",
        "flag_meanings",
    ]
    for key in checklist_keys:
        if key in thetao_attrs:
            val = thetao_attrs[key]
            lines.append(f"  {key}: EXISTS -> value = {repr(val)} (type: {type(val).__name__})")
        else:
            lines.append(f"  {key}: NOT PRESENT")
    lines.append("")

    # Ancillary/Mask variables checklist
    lines.append("-" * 80)
    lines.append("ANCILLARY / MASK / BATHYMETRY / MLOTST VARIABLES CHECK")
    lines.append("-" * 80)
    lines.append(f"Variables present in file: {vars_list}")
    suspect_vars = ["mask", "bathymetry", "deptho", "mlotst", "tmask", "topo"]
    found_ancillary = [v for v in vars_list if any(s in v.lower() for s in suspect_vars)]
    if found_ancillary:
        lines.append(f"  Found potential ancillary variables: {found_ancillary}")
    else:
        lines.append("  NO mask, bathymetry, deptho, mlotst, or topological variable exists in this file.")
    lines.append("")

    # Depth values with index
    lines.append("-" * 80)
    lines.append("DEPTH LEVELS (All 36 levels)")
    lines.append("-" * 80)
    depths = reader.get_raw_variable("depth")
    for idx, d in enumerate(depths):
        lines.append(f"  level[{idx:2d}]: {float(d):12.6f} m")
    lines.append("")

    dump_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dump_text)

    return dump_text


# =============================================================================
# Section B: Encoding Arithmetic
# =============================================================================

def compute_encoding_arithmetic(
    scale_factor: float,
    add_offset: float,
    fill_value: Optional[int],
    valid_min: Optional[int],
    valid_max: Optional[int],
    valid_range: Optional[List[int]],
) -> Dict[str, Any]:
    """Perform float64 encoding arithmetic for GLORYS packed codes."""
    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    exact_zero_code = (0.0 - offset) / scale
    rounded_zero_code = int(np.round(exact_zero_code))
    is_neg28671 = (rounded_zero_code == -28671)

    def decode(code: int) -> float:
        return float(np.float64(code) * scale + offset)

    decoded_neg28671 = decode(-28671)
    decoded_neg28672 = decode(-28672)
    decoded_neg28670 = decode(-28670)
    decoded_fill = decode(fill_value) if fill_value is not None else None

    # Range checks for -28671
    # Raw check
    inside_raw_valid_min = (valid_min is None) or (-28671 >= valid_min)
    inside_raw_valid_max = (valid_max is None) or (-28671 <= valid_max)
    inside_raw_valid_range = True
    if valid_range is not None and len(valid_range) == 2:
        inside_raw_valid_range = (valid_range[0] <= -28671 <= valid_range[1])

    is_inside_raw_valid = inside_raw_valid_min and inside_raw_valid_max and inside_raw_valid_range

    # Decoded check
    dec_valid_min = decode(valid_min) if valid_min is not None else None
    dec_valid_max = decode(valid_max) if valid_max is not None else None
    dec_valid_range = [decode(valid_range[0]), decode(valid_range[1])] if valid_range is not None and len(valid_range) == 2 else None

    inside_dec_min = (dec_valid_min is None) or (decoded_neg28671 >= dec_valid_min)
    inside_dec_max = (dec_valid_max is None) or (decoded_neg28671 <= dec_valid_max)
    is_inside_dec_valid = inside_dec_min and inside_dec_max

    return {
        "quantization_step": float(scale),
        "scale_factor": float(scale),
        "add_offset": float(offset),
        "exact_zero_code_float": float(exact_zero_code),
        "rounded_zero_code_int": rounded_zero_code,
        "rounded_code_equals_neg28671": bool(is_neg28671),
        "decoded_values": {
            "code_neg28672": decoded_neg28672,
            "code_neg28671": decoded_neg28671,
            "code_neg28670": decoded_neg28670,
            "fill_value_code": int(fill_value) if fill_value is not None else None,
            "decoded_fill_value": decoded_fill,
        },
        "valid_range_checks": {
            "raw_valid_min": int(valid_min) if valid_min is not None else None,
            "raw_valid_max": int(valid_max) if valid_max is not None else None,
            "raw_valid_range": [int(x) for x in valid_range] if valid_range is not None else None,
            "neg28671_inside_raw_valid_range": bool(is_inside_raw_valid),
            "decoded_valid_min": dec_valid_min,
            "decoded_valid_max": dec_valid_max,
            "decoded_valid_range": dec_valid_range,
            "neg28671_inside_decoded_valid_range": bool(is_inside_dec_valid),
        },
    }


def verify_manual_decode_against_xarray(
    filepath: str,
    scale_factor: float,
    add_offset: float,
    fill_value: Optional[int],
    sample_size: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify manual decode against xarray decoded values on a random finite sample."""
    # Open with xarray mask_and_scale=True
    ds_decoded = xr.open_dataset(filepath, engine="h5netcdf", mask_and_scale=True)
    # Open raw dataset
    ds_raw = xr.open_dataset(filepath, engine="h5netcdf", mask_and_scale=False)

    thetao_raw_da = ds_raw["thetao"]
    thetao_dec_da = ds_decoded["thetao"]

    # Sample finite points across time and levels
    # To keep memory footprint modest, pick sample indices randomly across dimensions
    rng = np.random.default_rng(seed)
    n_time, n_depth, n_lat, n_lon = thetao_raw_da.shape

    sample_manual = []
    sample_xr = []
    attempts = 0
    max_attempts = sample_size * 20

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    # Sample batch by batch to be fast
    batch_size = 2000
    while len(sample_manual) < sample_size and attempts < max_attempts:
        t_idx = rng.integers(0, n_time, size=batch_size)
        d_idx = rng.integers(0, n_depth, size=batch_size)
        y_idx = rng.integers(0, n_lat, size=batch_size)
        x_idx = rng.integers(0, n_lon, size=batch_size)

        # Slice points
        # Using numpy advanced indexing on sliced da
        raw_vals = thetao_raw_da.values[t_idx, d_idx, y_idx, x_idx]
        xr_vals = thetao_dec_da.values[t_idx, d_idx, y_idx, x_idx]

        mask = np.isfinite(xr_vals)
        if fill_value is not None:
            mask = mask & (raw_vals != fill_value)

        valid_raw = raw_vals[mask]
        valid_xr = xr_vals[mask]

        if len(valid_raw) > 0:
            dec_manual = valid_raw.astype(np.float64) * scale + offset
            sample_manual.extend(dec_manual.tolist())
            sample_xr.extend(valid_xr.tolist())

        attempts += batch_size

    ds_decoded.close()
    ds_raw.close()

    sample_manual = np.array(sample_manual[:sample_size], dtype=np.float64)
    sample_xr = np.array(sample_xr[:sample_size], dtype=np.float64)

    abs_diffs = np.abs(sample_manual - sample_xr)
    max_abs_diff = float(np.max(abs_diffs)) if len(abs_diffs) > 0 else 0.0
    mean_abs_diff = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0

    return {
        "sample_size": len(sample_manual),
        "seed": seed,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "consistent_with_float32_tolerance": bool(max_abs_diff < 1e-4),
    }


# =============================================================================
# Section C: Raw-Code Census
# =============================================================================

def run_raw_code_census(
    reader: RawNetCDFReader,
    fill_value: Optional[int],
    scale_factor: float,
    add_offset: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform census of raw codes per depth level and across the full file."""
    dims = reader.get_dimensions()
    n_depth = dims["depth"]
    depths = reader.get_raw_variable("depth")
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")
    times = reader.get_raw_variable("time")

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    rows = []
    cells_neg28671 = []
    histogram_codes = {code: 0 for code in range(-28675, -28666)}  # [-28675, -28667]

    total_fill = 0
    total_finite = 0
    total_neg28671 = 0
    total_near_zero = 0

    for k in range(n_depth):
        depth_m = float(depths[k])
        # Read slice for level k: shape (time, lat, lon)
        raw_slice = reader.get_raw_slice("thetao", (slice(None), k, slice(None), slice(None)))

        if fill_value is not None:
            is_fill = (raw_slice == fill_value)
            fill_count = int(np.sum(is_fill))
            finite_count = int(raw_slice.size - fill_count)
            finite_vals = raw_slice[~is_fill]
        else:
            is_fill = np.zeros_like(raw_slice, dtype=bool)
            fill_count = 0
            finite_count = int(raw_slice.size)
            finite_vals = raw_slice.flatten()

        total_fill += fill_count
        total_finite += finite_count

        # Check for -28671
        is_28671 = (raw_slice == -28671)
        count_28671 = int(np.sum(is_28671))
        total_neg28671 += count_28671

        if count_28671 > 0:
            t_indices, y_indices, x_indices = np.where(is_28671)
            for tidx, yidx, xidx in zip(t_indices, y_indices, x_indices):
                cells_neg28671.append({
                    "day_index": int(tidx),
                    "level_index": int(k),
                    "depth_m": depth_m,
                    "lat": float(lats[yidx]),
                    "lon": float(lons[xidx]),
                    "lat_index": int(yidx),
                    "lon_index": int(xidx),
                    "raw_code": -28671,
                })

        # Min/max of finite
        if finite_count > 0:
            raw_min = int(finite_vals.min())
            raw_max = int(finite_vals.max())
            # Near exact zero: |decoded| < 1e-3 C
            decoded_finite = finite_vals.astype(np.float64) * scale + offset
            near_zero_count = int(np.sum(np.abs(decoded_finite) < 1e-3))
        else:
            raw_min = None
            raw_max = None
            near_zero_count = 0

        total_near_zero += near_zero_count

        # Histogram slice in [-28675, -28667]
        for c in range(-28675, -28666):
            c_cnt = int(np.sum(raw_slice == c))
            histogram_codes[c] += c_cnt

        rows.append({
            "level_index": k,
            "depth_m": depth_m,
            "fill_count": fill_count,
            "finite_count": finite_count,
            "code_neg28671_count": count_28671,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "near_zero_count": near_zero_count,
        })

    census_df = pd.DataFrame(rows)

    # Unique columns (lat, lon) for -28671
    unique_cols_28671 = sorted(list({(c["lat"], c["lon"]) for c in cells_neg28671}))
    unique_levels_28671 = sorted(list({c["level_index"] for c in cells_neg28671}))

    summary = {
        "total_cells_evaluated": int(census_df["fill_count"].sum() + census_df["finite_count"].sum()),
        "total_fill_cells": total_fill,
        "total_finite_cells": total_finite,
        "total_neg28671_cells": total_neg28671,
        "total_near_zero_cells": total_near_zero,
        "histogram_near_neg28671": {str(k): v for k, v in histogram_codes.items()},
        "neg28671_occurrences_count": len(cells_neg28671),
        "neg28671_unique_columns": [{"lat": lat, "lon": lon} for lat, lon in unique_cols_28671],
        "neg28671_unique_level_indices": unique_levels_28671,
        "neg28671_cells_details": cells_neg28671,
    }

    return census_df, summary


# =============================================================================
# Section D: Mask Time-Invariance
# =============================================================================

def check_mask_time_invariance(
    reader: RawNetCDFReader,
    fill_value: Optional[int],
) -> Dict[str, Any]:
    """Compare finite/non-fill mask of every day to day 0 across all levels."""
    dims = reader.get_dimensions()
    n_time = dims["time"]
    n_depth = dims["depth"]

    if fill_value is None:
        return {
            "is_time_invariant": True,
            "differing_points_count": 0,
            "per_day_diff_counts": {},
            "note": "Fill value undefined; all points treated as finite.",
        }

    total_diff_points = 0
    per_day_diffs = {t: 0 for t in range(n_time)}

    for k in range(n_depth):
        # Read all times for level k
        raw_level = reader.get_raw_slice("thetao", (slice(None), k, slice(None), slice(None)))
        mask_day0 = (raw_level[0] != fill_value)

        for t in range(1, n_time):
            mask_t = (raw_level[t] != fill_value)
            diffs = int(np.sum(mask_t != mask_day0))
            if diffs > 0:
                per_day_diffs[t] += diffs
                total_diff_points += diffs

    is_invariant = (total_diff_points == 0)

    return {
        "is_time_invariant": bool(is_invariant),
        "total_differing_points": total_diff_points,
        "per_day_diff_counts": per_day_diffs,
    }


# =============================================================================
# Section E: Suspect-Column Profiles and Time Series
# =============================================================================

def extract_suspect_profiles_and_timeseries(
    reader: RawNetCDFReader,
    scale_factor: float,
    add_offset: float,
    fill_value: Optional[int],
    target_A: Tuple[float, float] = (15.000, 50.583),
    target_B: Tuple[float, float] = (15.000, 50.667),
    target_depths: Tuple[float, float, float] = (318.0, 380.0, 454.0),
    coord_tolerance: float = 1e-3,
    depth_tolerance: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Extract profile and time-series evidence for columns A, B and their 3x3 neighborhoods."""
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")
    depths = reader.get_raw_variable("depth")
    dims = reader.get_dimensions()
    n_time = dims["time"]
    n_depth = dims["depth"]

    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    # Locate A and B
    lat_A_idx, exact_lat_A = lookup_nearest_index(lats, target_A[0], "lat_A", coord_tolerance)
    lon_A_idx, exact_lon_A = lookup_nearest_index(lons, target_A[1], "lon_A", coord_tolerance)
    lat_B_idx, exact_lat_B = lookup_nearest_index(lats, target_B[0], "lat_B", coord_tolerance)
    lon_B_idx, exact_lon_B = lookup_nearest_index(lons, target_B[1], "lon_B", coord_tolerance)

    # Locate depths
    suspect_depth_meta = {}
    for td in target_depths:
        d_idx, exact_d = lookup_nearest_index(depths, td, f"depth_{td}m", depth_tolerance)
        suspect_depth_meta[td] = {"level_index": d_idx, "exact_depth_m": exact_d}

    # Days for profiles: day 0, day 15, last day (n_time - 1)
    profile_days = [0, 15, n_time - 1]

    # 3x3 neighborhood around A and B
    def get_3x3_indices(y_idx, x_idx):
        indices = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny = y_idx + dy
                nx = x_idx + dx
                if 0 <= ny < len(lats) and 0 <= nx < len(lons):
                    indices.append((ny, nx))
        return indices

    nbrs_A = set(get_3x3_indices(lat_A_idx, lon_A_idx))
    nbrs_B = set(get_3x3_indices(lat_B_idx, lon_B_idx))
    all_cols = sorted(list(nbrs_A.union(nbrs_B)))

    # Profile DataFrame
    profile_rows = []
    for day_idx in profile_days:
        day_raw = reader.get_raw_slice("thetao", (day_idx, slice(None), slice(None), slice(None)))
        for y_idx, x_idx in all_cols:
            col_lat = float(lats[y_idx])
            col_lon = float(lons[x_idx])
            is_col_A = (y_idx == lat_A_idx and x_idx == lon_A_idx)
            is_col_B = (y_idx == lat_B_idx and x_idx == lon_B_idx)
            tag = "A" if is_col_A else ("B" if is_col_B else "neighbor")

            for k in range(n_depth):
                depth_m = float(depths[k])
                raw_code = int(day_raw[k, y_idx, x_idx])
                is_fill = bool(raw_code == fill_value) if fill_value is not None else False
                decoded = float(np.float64(raw_code) * scale + offset) if not is_fill else np.nan

                profile_rows.append({
                    "day": day_idx,
                    "lat": col_lat,
                    "lon": col_lon,
                    "lat_index": y_idx,
                    "lon_index": x_idx,
                    "column_tag": tag,
                    "level_index": k,
                    "depth_m": depth_m,
                    "raw_code": raw_code,
                    "is_fill": is_fill,
                    "decoded_C": decoded,
                })

    profile_df = pd.DataFrame(profile_rows)

    # Time series at suspect levels for A and B
    ts_rows = []
    ts_stats = {}

    for col_label, (y_idx, x_idx) in [("A", (lat_A_idx, lon_A_idx)), ("B", (lat_B_idx, lon_B_idx))]:
        col_lat = float(lats[y_idx])
        col_lon = float(lons[x_idx])
        col_raw_all = reader.get_raw_slice("thetao", (slice(None), slice(None), y_idx, x_idx))  # shape (time, depth)

        for td, dmeta in suspect_depth_meta.items():
            lvl = dmeta["level_index"]
            edepth = dmeta["exact_depth_m"]

            series_raw = col_raw_all[:, lvl]
            series_is_fill = (series_raw == fill_value) if fill_value is not None else np.zeros(n_time, dtype=bool)
            series_dec = np.where(~series_is_fill, series_raw.astype(np.float64) * scale + offset, np.nan)

            for t in range(n_time):
                ts_rows.append({
                    "day": t,
                    "column": col_label,
                    "lat": col_lat,
                    "lon": col_lon,
                    "target_depth_m": td,
                    "exact_depth_m": edepth,
                    "level_index": lvl,
                    "raw_code": int(series_raw[t]),
                    "is_fill": bool(series_is_fill[t]),
                    "decoded_C": float(series_dec[t]) if not series_is_fill[t] else np.nan,
                })

            valid_dec = series_dec[~series_is_fill]
            stat_key = f"col_{col_label}_depth_{td}m"
            if len(valid_dec) > 0:
                unique_codes = sorted(list(set(series_raw.tolist())))
                abs_day_diffs = np.abs(np.diff(valid_dec)) if len(valid_dec) > 1 else np.array([0.0])
                ts_stats[stat_key] = {
                    "column": col_label,
                    "target_depth_m": td,
                    "exact_depth_m": edepth,
                    "level_index": lvl,
                    "unique_raw_codes_count": len(unique_codes),
                    "unique_raw_codes": unique_codes,
                    "min_decoded_C": float(np.min(valid_dec)),
                    "max_decoded_C": float(np.max(valid_dec)),
                    "std_decoded_C": float(np.std(valid_dec)),
                    "mean_abs_day_to_day_change_C": float(np.mean(abs_day_diffs)),
                }
            else:
                ts_stats[stat_key] = {
                    "column": col_label,
                    "target_depth_m": td,
                    "exact_depth_m": edepth,
                    "level_index": lvl,
                    "all_fill": True,
                }

    timeseries_df = pd.DataFrame(ts_rows)

    meta_summary = {
        "column_A": {
            "target": target_A,
            "exact": (exact_lat_A, exact_lon_A),
            "indices": (lat_A_idx, lon_A_idx),
        },
        "column_B": {
            "target": target_B,
            "exact": (exact_lat_B, exact_lon_B),
            "indices": (lat_B_idx, lon_B_idx),
        },
        "suspect_depths": suspect_depth_meta,
        "timeseries_stats": ts_stats,
    }

    return profile_df, timeseries_df, meta_summary


# =============================================================================
# Section F: Level-Wise Context for Suspect Levels
# =============================================================================

def compute_level_context(
    reader: RawNetCDFReader,
    suspect_depths: Dict[float, Dict[str, Any]],
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
    scale_factor: float,
    add_offset: float,
    fill_value: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute level-wise statistical context and percentile rankings."""
    dims = reader.get_dimensions()
    n_time = dims["time"]
    scale = np.float64(scale_factor)
    offset = np.float64(add_offset)

    percentile_levels = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9]
    context_rows = []
    summary_results = {}

    for td, dmeta in suspect_depths.items():
        lvl = dmeta["level_index"]
        exact_depth = dmeta["exact_depth_m"]

        # Read entire level across time: shape (time, lat, lon)
        raw_level = reader.get_raw_slice("thetao", (slice(None), lvl, slice(None), slice(None)))

        # Day 0 values for A and B
        val_A_day0 = float(raw_level[0, lat_A_idx, lon_A_idx] * scale + offset)
        val_B_day0 = float(raw_level[0, lat_B_idx, lon_B_idx] * scale + offset)

        # Nearest neighbors for A and B on Day 0
        def get_neighbor_vals(y_idx, x_idx):
            nbr_vals = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y_idx + dy, x_idx + dx
                if 0 <= ny < raw_level.shape[1] and 0 <= nx < raw_level.shape[2]:
                    code = raw_level[0, ny, nx]
                    if fill_value is None or code != fill_value:
                        nbr_vals.append(float(code * scale + offset))
            return nbr_vals[:2]

        nbrs_A = get_neighbor_vals(lat_A_idx, lon_A_idx)
        nbrs_B = get_neighbor_vals(lat_B_idx, lon_B_idx)

        # Pooled finite values
        if fill_value is not None:
            finite_mask = (raw_level != fill_value)
        else:
            finite_mask = np.ones_like(raw_level, dtype=bool)

        finite_raw_pooled = raw_level[finite_mask]
        finite_dec_pooled = finite_raw_pooled.astype(np.float64) * scale + offset

        # Day 0 finite
        finite_raw_day0 = raw_level[0][finite_mask[0]]
        finite_dec_day0 = finite_raw_day0.astype(np.float64) * scale + offset

        # Count < -0.5 C pooled and day 0
        below_neg_half_pooled = int(np.sum(finite_dec_pooled < -0.5))
        below_neg_half_day0 = int(np.sum(finite_dec_day0 < -0.5))

        # Percentiles
        p_day0 = np.percentile(finite_dec_day0, percentile_levels) if len(finite_dec_day0) > 0 else [np.nan] * len(percentile_levels)
        p_pooled = np.percentile(finite_dec_pooled, percentile_levels) if len(finite_dec_pooled) > 0 else [np.nan] * len(percentile_levels)

        # Percentile ranks
        rank_A_day0 = float(stats.percentileofscore(finite_dec_day0, val_A_day0, kind="weak")) if len(finite_dec_day0) > 0 else np.nan
        rank_B_day0 = float(stats.percentileofscore(finite_dec_day0, val_B_day0, kind="weak")) if len(finite_dec_day0) > 0 else np.nan
        rank_A_pooled = float(stats.percentileofscore(finite_dec_pooled, val_A_day0, kind="weak")) if len(finite_dec_pooled) > 0 else np.nan
        rank_B_pooled = float(stats.percentileofscore(finite_dec_pooled, val_B_day0, kind="weak")) if len(finite_dec_pooled) > 0 else np.nan

        # Per-day distributions for each day t
        for t in range(n_time):
            val_A_t = float(raw_level[t, lat_A_idx, lon_A_idx] * scale + offset)
            val_B_t = float(raw_level[t, lat_B_idx, lon_B_idx] * scale + offset)
            finite_raw_t = raw_level[t][finite_mask[t]]
            finite_dec_t = finite_raw_t.astype(np.float64) * scale + offset
            below_neg_half_t = int(np.sum(finite_dec_t < -0.5))
            p_t = np.percentile(finite_dec_t, percentile_levels) if len(finite_dec_t) > 0 else [np.nan] * len(percentile_levels)
            rank_A_t = float(stats.percentileofscore(finite_dec_t, val_A_t, kind="weak")) if len(finite_dec_t) > 0 else np.nan
            rank_B_t = float(stats.percentileofscore(finite_dec_t, val_B_t, kind="weak")) if len(finite_dec_t) > 0 else np.nan

            base_dict_t = {
                "target_depth_m": td,
                "exact_depth_m": exact_depth,
                "level_index": lvl,
                "scope": f"day_{t}",
                "finite_cells_count": len(finite_dec_t),
                "cells_below_neg_0_5_C": below_neg_half_t,
                "min_C": float(np.min(finite_dec_t)) if len(finite_dec_t) > 0 else np.nan,
                "max_C": float(np.max(finite_dec_t)) if len(finite_dec_t) > 0 else np.nan,
                "val_A": val_A_t,
                "rank_A": rank_A_t,
                "val_B": val_B_t,
                "rank_B": rank_B_t,
            }
            for plvl, val in zip(percentile_levels, p_t):
                base_dict_t[f"p_{str(plvl).replace('.', '_')}"] = float(val)
            context_rows.append(base_dict_t)

        # Pooled distribution across all days
        base_dict_pooled = {
            "target_depth_m": td,
            "exact_depth_m": exact_depth,
            "level_index": lvl,
            "scope": "pooled_31_days",
            "finite_cells_count": len(finite_dec_pooled),
            "cells_below_neg_0_5_C": below_neg_half_pooled,
            "min_C": float(np.min(finite_dec_pooled)) if len(finite_dec_pooled) > 0 else np.nan,
            "max_C": float(np.max(finite_dec_pooled)) if len(finite_dec_pooled) > 0 else np.nan,
            "val_A": val_A_day0,
            "rank_A": rank_A_pooled,
            "val_B": val_B_day0,
            "rank_B": rank_B_pooled,
        }
        for plvl, val in zip(percentile_levels, p_pooled):
            base_dict_pooled[f"p_{str(plvl).replace('.', '_')}"] = float(val)
        context_rows.append(base_dict_pooled)

        summary_results[f"depth_{td}m"] = {
            "target_depth_m": td,
            "exact_depth_m": exact_depth,
            "level_index": lvl,
            "val_A_day0": val_A_day0,
            "val_B_day0": val_B_day0,
            "rank_A_day0": rank_A_day0,
            "rank_B_day0": rank_B_day0,
            "rank_A_pooled": rank_A_pooled,
            "rank_B_pooled": rank_B_pooled,
            "nearest_neighbors_A_day0": nbrs_A,
            "nearest_neighbors_B_day0": nbrs_B,
            "finite_count_day0": len(finite_dec_day0),
            "finite_count_pooled": len(finite_dec_pooled),
            "count_below_neg_0_5_C_day0": below_neg_half_day0,
            "count_below_neg_0_5_C_pooled": below_neg_half_pooled,
            "percentiles_day0": {f"p_{p}": float(v) for p, v in zip(percentile_levels, p_day0)},
            "percentiles_pooled": {f"p_{p}": float(v) for p, v in zip(percentile_levels, p_pooled)},
        }

    context_df = pd.DataFrame(context_rows)
    return context_df, summary_results


# =============================================================================
# Section G: Uniqueness Check of the Deep-Structure Anomaly
# =============================================================================

def compute_deepest_finite_anomaly(
    reader: RawNetCDFReader,
    fill_value: Optional[int],
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute deepest finite level per column on Day 0 and difference to 8-neighbor median."""
    dims = reader.get_dimensions()
    n_depth = dims["depth"]
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")
    depths = reader.get_raw_variable("depth")

    # Read Day 0 raw array: shape (depth, lat, lon)
    day0_raw = reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))

    if fill_value is not None:
        finite_mask = (day0_raw != fill_value)  # shape (depth, lat, lon)
    else:
        finite_mask = np.ones_like(day0_raw, dtype=bool)

    # Compute deepest finite level index for each column
    # If a column has no finite cells, deepest = -1
    has_finite = finite_mask.any(axis=0)  # shape (lat, lon)
    level_indices = np.arange(n_depth)[:, None, None]
    masked_levels = np.where(finite_mask, level_indices, -1)
    deepest_level = masked_levels.max(axis=0)  # shape (lat, lon)

    n_lat, n_lon = deepest_level.shape
    columns_data = []

    for y in range(n_lat):
        for x in range(n_lon):
            c_deep = int(deepest_level[y, x])
            if c_deep < 0:
                continue  # Skip land / entirely fill columns

            # 8 neighbors
            nbr_levels = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < n_lat and 0 <= nx < n_lon:
                        n_deep = deepest_level[ny, nx]
                        if n_deep >= 0:
                            nbr_levels.append(int(n_deep))

            if len(nbr_levels) > 0:
                nbr_median = float(np.median(nbr_levels))
                diff = float(c_deep - nbr_median)
            else:
                nbr_median = np.nan
                diff = 0.0

            is_A = (y == lat_A_idx and x == lon_A_idx)
            is_B = (y == lat_B_idx and x == lon_B_idx)

            columns_data.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "lat_index": y,
                "lon_index": x,
                "deepest_level_index": c_deep,
                "deepest_depth_m": float(depths[c_deep]),
                "neighbor_median_level": nbr_median,
                "difference_to_median": diff,
                "is_col_A": is_A,
                "is_col_B": is_B,
            })

    # Sort descending by difference_to_median
    columns_data_sorted = sorted(columns_data, key=lambda c: c["difference_to_median"], reverse=True)

    # Rank calculation
    rank_A = None
    rank_B = None
    info_A = None
    info_B = None
    for idx, c in enumerate(columns_data_sorted):
        if c["is_col_A"]:
            rank_A = idx + 1
            info_A = c
        if c["is_col_B"]:
            rank_B = idx + 1
            info_B = c

    # Count of columns with diff > 3
    count_diff_gt_3 = sum(1 for c in columns_data if c["difference_to_median"] > 3.0)

    top_20 = columns_data_sorted[:20]

    summary = {
        "total_ocean_columns_evaluated": len(columns_data),
        "columns_with_deepest_level_diff_gt_3": count_diff_gt_3,
        "column_A_rank": rank_A,
        "column_A_info": info_A,
        "column_B_rank": rank_B,
        "column_B_info": info_B,
        "top_20_columns": top_20,
    }

    return top_20, summary


# =============================================================================
# Section H: Plots (Matplotlib)
# =============================================================================

def generate_plots(
    profile_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    census_summary: Dict[str, Any],
    reader: RawNetCDFReader,
    lat_A_idx: int,
    lon_A_idx: int,
    lat_B_idx: int,
    lon_B_idx: int,
    output_dir: str,
) -> List[str]:
    """Generate 4 forensic PNG plots."""
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []

    # 1. raw_code profile and decoded profile vs depth (day 0) for A, B and 3x3 neighbors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    day0_profiles = profile_df[profile_df["day"] == 0]

    # Plot neighbors first
    nbr_df = day0_profiles[day0_profiles["column_tag"] == "neighbor"]
    for (lat, lon), grp in nbr_df.groupby(["lat", "lon"]):
        grp_sorted = grp.sort_values("depth_m")
        ax1.plot(grp_sorted["raw_code"], grp_sorted["depth_m"], color="gray", alpha=0.35, linewidth=1.0, linestyle=":")
        ax2.plot(grp_sorted["decoded_C"], grp_sorted["depth_m"], color="gray", alpha=0.35, linewidth=1.0, linestyle=":")

    # Plot A and B
    df_A = day0_profiles[day0_profiles["column_tag"] == "A"].sort_values("depth_m")
    df_B = day0_profiles[day0_profiles["column_tag"] == "B"].sort_values("depth_m")

    ax1.plot(df_A["raw_code"], df_A["depth_m"], color="crimson", linewidth=2.5, linestyle="-", marker="o", markersize=4, label="Col A (15.000N, 50.583E)")
    ax1.plot(df_B["raw_code"], df_B["depth_m"], color="royalblue", linewidth=2.5, linestyle="--", marker="s", markersize=4, label="Col B (15.000N, 50.667E)")

    ax2.plot(df_A["decoded_C"], df_A["depth_m"], color="crimson", linewidth=2.5, linestyle="-", marker="o", markersize=4, label="Col A (15.000N, 50.583E)")
    ax2.plot(df_B["decoded_C"], df_B["depth_m"], color="royalblue", linewidth=2.5, linestyle="--", marker="s", markersize=4, label="Col B (15.000N, 50.667E)")

    # Highlight -28671 reference line
    ax1.axvline(-28671, color="orange", linestyle="--", alpha=0.7, label="Packed code 0°C (-28671)")
    ax2.axvline(0.0, color="orange", linestyle="--", alpha=0.7, label="0.0 °C")

    ax1.invert_yaxis()
    ax1.set_xlabel("Raw Code (int16)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Day 0: Raw Code Profile vs Depth")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    ax2.set_xlabel("Decoded Potential Temperature (°C)")
    ax2.set_title("Day 0: Decoded Temperature Profile vs Depth")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)

    plt.suptitle("GLORYS thetao Profiles for Suspect Columns A & B and 3x3 Neighborhoods", fontsize=13, weight="bold")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "glorys_suspect_profiles.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    generated_plots.append(p1)

    # 2. Time series at suspect levels for A and B
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    depth_targets = [318.0, 380.0, 454.0]

    for idx, td in enumerate(depth_targets):
        ax = axes[idx]
        ts_td = timeseries_df[timeseries_df["target_depth_m"] == td]
        ts_A = ts_td[ts_td["column"] == "A"].sort_values("day")
        ts_B = ts_td[ts_td["column"] == "B"].sort_values("day")

        ax.plot(ts_A["day"] + 1, ts_A["decoded_C"], color="crimson", linewidth=2.0, marker="o", markersize=4, label="Col A (decoded °C)")
        ax.plot(ts_B["day"] + 1, ts_B["decoded_C"], color="royalblue", linewidth=2.0, linestyle="--", marker="s", markersize=4, label="Col B (decoded °C)")

        edepth = ts_td["exact_depth_m"].iloc[0] if len(ts_td) > 0 else td
        ax.set_title(f"Target Depth ~{td:.0f} m (Exact: {edepth:.1f} m, Level {ts_td['level_index'].iloc[0]})")
        ax.set_ylabel("Decoded Temp (°C)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Day of Month (January 2019)")
    plt.suptitle("GLORYS thetao Time Series at Suspect Depths for Columns A and B", fontsize=13, weight="bold")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "glorys_suspect_timeseries.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    generated_plots.append(p2)

    # 3. Histogram of raw codes near -28671 (log y)
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_data = census_summary["histogram_near_neg28671"]
    codes = [int(k) for k in hist_data.keys()]
    counts = [hist_data[str(k)] for k in codes]

    bars = ax.bar(codes, counts, color="teal", edgecolor="black", alpha=0.85)
    for c, bar in zip(codes, bars):
        if c == -28671:
            bar.set_color("crimson")
            bar.set_edgecolor("black")

    ax.set_yscale("log")
    ax.set_xticks(codes)
    ax.set_xlabel("Raw Code (int16)")
    ax.set_ylabel("Whole-File Cell Count (Log Scale)")
    ax.set_title("Histogram of Raw Codes in Interval [-28675, -28667] (Whole File)")
    ax.grid(True, linestyle=":", alpha=0.6, which="both")

    for c, cnt in zip(codes, counts):
        if cnt > 0:
            ax.annotate(f"{cnt}", (c, cnt), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    p3 = os.path.join(output_dir, "glorys_raw_code_histogram.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    generated_plots.append(p3)

    # 4. Map (lat/lon) of deepest-finite-level index in a window of +/- 1 deg around A and B
    lats = reader.get_raw_variable("latitude")
    lons = reader.get_raw_variable("longitude")
    depths = reader.get_raw_variable("depth")
    dims = reader.get_dimensions()
    n_depth = dims["depth"]

    lat_center = float(lats[lat_A_idx])
    lon_center = float(0.5 * (lons[lon_A_idx] + lons[lon_B_idx]))

    lat_mask = (lats >= lat_center - 1.05) & (lats <= lat_center + 1.05)
    lon_mask = (lons >= lon_center - 1.05) & (lons <= lon_center + 1.05)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    lat_slice = slice(int(lat_indices[0]), int(lat_indices[-1]) + 1)
    lon_slice = slice(int(lon_indices[0]), int(lon_indices[-1]) + 1)

    sub_lats = lats[lat_slice]
    sub_lons = lons[lon_slice]

    day0_raw = reader.get_raw_slice("thetao", (0, slice(None), lat_slice, lon_slice))
    fill_value = census_summary.get("fill_value", -32767)
    finite_mask = (day0_raw != fill_value)

    level_idx_grid = np.arange(n_depth)[:, None, None]
    masked_levels = np.where(finite_mask, level_idx_grid, -1)
    deepest_sub = masked_levels.max(axis=0)  # shape (sub_lat, sub_lon)

    deepest_sub_float = deepest_sub.astype(float)
    deepest_sub_float[deepest_sub < 0] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    mesh = ax.pcolormesh(sub_lons, sub_lats, deepest_sub_float, cmap="viridis", shading="auto")
    cbar = plt.colorbar(mesh, ax=ax, label="Deepest Finite Level Index (0..35)")

    # Mark A and B
    ax.scatter([lons[lon_A_idx]], [lats[lat_A_idx]], color="red", s=100, marker="o", edgecolors="white", linewidth=1.5, label="Column A (15.000N, 50.583E)", zorder=5)
    ax.scatter([lons[lon_B_idx]], [lats[lat_B_idx]], color="cyan", s=100, marker="s", edgecolors="black", linewidth=1.5, label="Column B (15.000N, 50.667E)", zorder=5)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Deepest Finite Level Index (Day 0) within ±1° Window of A & B")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p4 = os.path.join(output_dir, "glorys_deepest_level_map.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    generated_plots.append(p4)

    return generated_plots


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_forensics(input_file: str, output_dir: str) -> Dict[str, Any]:
    """Execute complete GLORYS thetao encoding forensics."""
    # Compute initial metadata and SHA256
    print("\n" + "=" * 80)
    print("STEP 1: COMPUTING INITIAL INPUT FILE CHECKSUM AND METADATA (READ-ONLY)")
    print("=" * 80)
    meta_before = compute_file_metadata(input_file)
    print(f"File: {meta_before['filepath']}")
    print(f"Size: {meta_before['size_bytes']} bytes")
    print(f"MTime (UTC): {meta_before['mtime_utc']}")
    print(f"SHA256 (before): {meta_before['sha256']}")

    os.makedirs(output_dir, exist_ok=True)

    header_dump_path = os.path.join(output_dir, "glorys_header_dump.txt")
    census_csv_path = os.path.join(output_dir, "glorys_raw_code_census.csv")
    profile_csv_path = os.path.join(output_dir, "glorys_suspect_columns_raw_profile.csv")
    ts_csv_path = os.path.join(output_dir, "glorys_suspect_timeseries.csv")
    context_csv_path = os.path.join(output_dir, "glorys_level_context.csv")
    json_path = os.path.join(output_dir, "glorys_encoding_forensics.json")

    with RawNetCDFReader(input_file) as reader:
        # A. Header Dump
        print("\n" + "=" * 80)
        print("SECTION A: METADATA DUMP")
        print("=" * 80)
        generate_header_dump(reader, input_file, header_dump_path)
        print(f"Header dump written to: {header_dump_path}")

        # Extract variable attributes
        thetao_attrs = reader.get_var_attrs("thetao")
        scale_factor = float(thetao_attrs.get("scale_factor", 1.0))
        add_offset = float(thetao_attrs.get("add_offset", 0.0))
        fill_val_raw = thetao_attrs.get("_FillValue", None)
        fill_value = int(fill_val_raw) if fill_val_raw is not None else None
        valid_min = int(thetao_attrs["valid_min"]) if "valid_min" in thetao_attrs else None
        valid_max = int(thetao_attrs["valid_max"]) if "valid_max" in thetao_attrs else None
        valid_range = [int(x) for x in thetao_attrs["valid_range"]] if "valid_range" in thetao_attrs else None

        # B. Encoding Arithmetic
        print("\n" + "=" * 80)
        print("SECTION B: ENCODING ARITHMETIC")
        print("=" * 80)
        arithmetic = compute_encoding_arithmetic(
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill_value=fill_value,
            valid_min=valid_min,
            valid_max=valid_max,
            valid_range=valid_range,
        )
        print(f"Quantization step (scale_factor): {arithmetic['quantization_step']:.10e}")
        print(f"Add offset: {arithmetic['add_offset']}")
        print(f"Exact zero code float: {arithmetic['exact_zero_code_float']:.10f}")
        print(f"Rounded zero code int: {arithmetic['rounded_zero_code_int']}")
        print(f"Rounded code equals -28671: {arithmetic['rounded_code_equals_neg28671']}")
        print(f"Decoded code -28672: {arithmetic['decoded_values']['code_neg28672']:.10e} °C")
        print(f"Decoded code -28671: {arithmetic['decoded_values']['code_neg28671']:.10e} °C")
        print(f"Decoded code -28670: {arithmetic['decoded_values']['code_neg28670']:.10e} °C")
        print(f"Decoded _FillValue ({fill_value}): {arithmetic['decoded_values']['decoded_fill_value']} °C")
        print(f"-28671 inside raw valid range: {arithmetic['valid_range_checks']['neg28671_inside_raw_valid_range']}")
        print(f"-28671 inside decoded valid range: {arithmetic['valid_range_checks']['neg28671_inside_decoded_valid_range']}")

        print("\nVerifying manual decode against xarray on 10,000 points (seed 42)...")
        xr_verify = verify_manual_decode_against_xarray(
            filepath=input_file,
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill_value=fill_value,
            sample_size=10000,
            seed=42,
        )
        print(f"Sample size: {xr_verify['sample_size']}")
        print(f"Max abs diff: {xr_verify['max_abs_diff']:.10e}")
        print(f"Consistent with float32 tolerance: {xr_verify['consistent_with_float32_tolerance']}")

        # C. Raw-Code Census
        print("\n" + "=" * 80)
        print("SECTION C: RAW-CODE CENSUS OVER WHOLE FILE")
        print("=" * 80)
        census_df, census_summary = run_raw_code_census(
            reader=reader,
            fill_value=fill_value,
            scale_factor=scale_factor,
            add_offset=add_offset,
        )
        census_summary["fill_value"] = fill_value
        census_df.to_csv(census_csv_path, index=False)
        print(f"Census table written to: {census_csv_path}")
        print(f"Total cells evaluated: {census_summary['total_cells_evaluated']:,}")
        print(f"Total finite cells: {census_summary['total_finite_cells']:,}")
        print(f"Total fill cells: {census_summary['total_fill_cells']:,}")
        print(f"Total -28671 cells: {census_summary['total_neg28671_cells']}")
        print(f"Total near-zero (|x| < 1e-3 C) cells: {census_summary['total_near_zero_cells']}")
        print(f"Histogram near -28671: {census_summary['histogram_near_neg28671']}")
        print(f"Unique columns containing -28671: {census_summary['neg28671_unique_columns']}")
        print(f"Unique level indices containing -28671: {census_summary['neg28671_unique_level_indices']}")

        # D. Mask Time-Invariance
        print("\n" + "=" * 80)
        print("SECTION D: MASK TIME-INVARIANCE CHECK")
        print("=" * 80)
        mask_inv = check_mask_time_invariance(reader, fill_value)
        print(f"Is land/ocean mask strictly time-invariant: {mask_inv['is_time_invariant']}")
        print(f"Total differing points across days 1..30 vs day 0: {mask_inv['total_differing_points']}")

        # E. Suspect-Column Profiles and Time Series
        print("\n" + "=" * 80)
        print("SECTION E: SUSPECT-COLUMN PROFILES AND TIME SERIES")
        print("=" * 80)
        profile_df, ts_df, suspect_meta = extract_suspect_profiles_and_timeseries(
            reader=reader,
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill_value=fill_value,
            target_A=(15.000, 50.583),
            target_B=(15.000, 50.667),
            target_depths=(318.0, 380.0, 454.0),
            coord_tolerance=1e-3,
            depth_tolerance=5.0,
        )
        profile_df.to_csv(profile_csv_path, index=False)
        ts_df.to_csv(ts_csv_path, index=False)
        print(f"Profiles written to: {profile_csv_path}")
        print(f"Time series written to: {ts_csv_path}")
        print(f"Column A located at: lat={suspect_meta['column_A']['exact'][0]:.6f}, lon={suspect_meta['column_A']['exact'][1]:.6f}")
        print(f"Column B located at: lat={suspect_meta['column_B']['exact'][0]:.6f}, lon={suspect_meta['column_B']['exact'][1]:.6f}")
        for td, dmeta in suspect_meta["suspect_depths"].items():
            print(f"Target depth {td} m -> level {dmeta['level_index']}, exact depth = {dmeta['exact_depth_m']:.4f} m")

        print("\nTime Series Statistics:")
        for sk, sv in suspect_meta["timeseries_stats"].items():
            print(f"  {sk}: unique_raw_codes={sv.get('unique_raw_codes_count')}, "
                  f"min={sv.get('min_decoded_C'):.6f} C, max={sv.get('max_decoded_C'):.6f} C, "
                  f"std={sv.get('std_decoded_C'):.6e} C, mean_abs_change={sv.get('mean_abs_day_to_day_change_C'):.6e} C")

        # F. Level-Wise Context for Suspect Levels
        print("\n" + "=" * 80)
        print("SECTION F: LEVEL-WISE CONTEXT FOR SUSPECT LEVELS")
        print("=" * 80)
        lat_A_idx, lon_A_idx = suspect_meta["column_A"]["indices"]
        lat_B_idx, lon_B_idx = suspect_meta["column_B"]["indices"]
        context_df, context_summary = compute_level_context(
            reader=reader,
            suspect_depths=suspect_meta["suspect_depths"],
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill_value=fill_value,
        )
        context_df.to_csv(context_csv_path, index=False)
        print(f"Level context table written to: {context_csv_path}")
        for dk, dv in context_summary.items():
            print(f"  {dk} (Level {dv['level_index']}):")
            print(f"    Val A (Day 0): {dv['val_A_day0']:.6f} C (Day 0 rank: {dv['rank_A_day0']:.2f}%, Pooled rank: {dv['rank_A_pooled']:.2f}%)")
            print(f"    Val B (Day 0): {dv['val_B_day0']:.6f} C (Day 0 rank: {dv['rank_B_day0']:.2f}%, Pooled rank: {dv['rank_B_pooled']:.2f}%)")
            print(f"    Nearest neighbors A (Day 0): {dv['nearest_neighbors_A_day0']}")
            print(f"    Nearest neighbors B (Day 0): {dv['nearest_neighbors_B_day0']}")
            print(f"    Cells < -0.5 C (Day 0): {dv['count_below_neg_0_5_C_day0']} / {dv['finite_count_day0']}")
            print(f"    Cells < -0.5 C (Pooled): {dv['count_below_neg_0_5_C_pooled']} / {dv['finite_count_pooled']}")

        # G. Uniqueness Check of the Deep-Structure Anomaly
        print("\n" + "=" * 80)
        print("SECTION G: DEEP-STRUCTURE ANOMALY UNIQUENESS CHECK")
        print("=" * 80)
        top_20_cols, g_summary = compute_deepest_finite_anomaly(
            reader=reader,
            fill_value=fill_value,
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
        )
        print(f"Ocean columns evaluated: {g_summary['total_ocean_columns_evaluated']:,}")
        print(f"Columns with deepest level > 3 levels deeper than neighbor median: {g_summary['columns_with_deepest_level_diff_gt_3']}")
        print(f"Column A rank: {g_summary['column_A_rank']} (diff = {g_summary['column_A_info']['difference_to_median']})")
        print(f"Column B rank: {g_summary['column_B_rank']} (diff = {g_summary['column_B_info']['difference_to_median']})")
        print("\nTop 20 Columns by Difference to Neighbor Median Deepest Level:")
        print(f"{'Rank':<5} {'Lat':<8} {'Lon':<8} {'Deepest Lvl':<12} {'Deepest Depth (m)':<18} {'Neighbor Med':<14} {'Diff':<8} {'Tag'}")
        for idx, col in enumerate(top_20_cols):
            tag = "Col A" if col["is_col_A"] else ("Col B" if col["is_col_B"] else "")
            print(f"{idx+1:<5} {col['lat']:<8.3f} {col['lon']:<8.3f} {col['deepest_level_index']:<12} {col['deepest_depth_m']:<18.2f} {col['neighbor_median_level']:<14.1f} {col['difference_to_median']:<8.1f} {tag}")

        # H. Plots
        print("\n" + "=" * 80)
        print("SECTION H: GENERATING PLOTS")
        print("=" * 80)
        plot_paths = generate_plots(
            profile_df=profile_df,
            timeseries_df=ts_df,
            census_summary=census_summary,
            reader=reader,
            lat_A_idx=lat_A_idx,
            lon_A_idx=lon_A_idx,
            lat_B_idx=lat_B_idx,
            lon_B_idx=lon_B_idx,
            output_dir=output_dir,
        )
        for p in plot_paths:
            print(f"Generated plot: {p}")

    # Compute final metadata and verify SHA256 matches
    print("\n" + "=" * 80)
    print("STEP FINAL: VERIFYING INPUT FILE INTEGRITY (BEFORE VS AFTER)")
    print("=" * 80)
    meta_after = compute_file_metadata(input_file)
    print(f"SHA256 (before): {meta_before['sha256']}")
    print(f"SHA256 (after):  {meta_after['sha256']}")
    print(f"Size (before):   {meta_before['size_bytes']} bytes")
    print(f"Size (after):    {meta_after['size_bytes']} bytes")
    print(f"MTime (before):  {meta_before['mtime_utc']}")
    print(f"MTime (after):   {meta_after['mtime_utc']}")

    sha256_match = (meta_before["sha256"] == meta_after["sha256"])
    size_match = (meta_before["size_bytes"] == meta_after["size_bytes"])
    mtime_match = (meta_before["mtime_utc"] == meta_after["mtime_utc"])

    if not sha256_match or not size_match or not mtime_match:
        raise RuntimeError("FATAL: Input raw file was modified during analysis!")

    # Assemble JSON findings
    json_findings = {
        "analysis_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_file_integrity": {
            "filepath": meta_before["filepath"],
            "sha256_before": meta_before["sha256"],
            "sha256_after": meta_after["sha256"],
            "sha256_match": sha256_match,
            "size_bytes_before": meta_before["size_bytes"],
            "size_bytes_after": meta_after["size_bytes"],
            "size_bytes_match": size_match,
            "mtime_utc_before": meta_before["mtime_utc"],
            "mtime_utc_after": meta_after["mtime_utc"],
            "mtime_match": mtime_match,
        },
        "unknown_fields": {
            "nature_of_code_neg28671": None,
            "nature_of_code_neg28671_reason": "UNKNOWN: Data-validation forensic evidence only; physical nature, valid value status, fill status, or corruption origin cannot be verified from file alone.",
            "missing_value_attribute": None if "missing_value" not in thetao_attrs else thetao_attrs["missing_value"],
            "missing_value_attribute_reason": "Attribute missing_value is not defined on thetao in this NetCDF file." if "missing_value" not in thetao_attrs else None,
            "valid_range_attribute": None if "valid_range" not in thetao_attrs else thetao_attrs["valid_range"],
            "valid_range_attribute_reason": "Attribute valid_range is not defined on thetao in this NetCDF file." if "valid_range" not in thetao_attrs else None,
            "flag_values_attribute": None if "flag_values" not in thetao_attrs else thetao_attrs["flag_values"],
            "flag_values_attribute_reason": "Attribute flag_values is not defined on thetao in this NetCDF file." if "flag_values" not in thetao_attrs else None,
            "flag_meanings_attribute": None if "flag_meanings" not in thetao_attrs else thetao_attrs["flag_meanings"],
            "flag_meanings_attribute_reason": "Attribute flag_meanings is not defined on thetao in this NetCDF file." if "flag_meanings" not in thetao_attrs else None,
        },
        "section_B_encoding_arithmetic": arithmetic,
        "section_B_xarray_verification": xr_verify,
        "section_C_census_summary": {
            "total_cells_evaluated": census_summary["total_cells_evaluated"],
            "total_fill_cells": census_summary["total_fill_cells"],
            "total_finite_cells": census_summary["total_finite_cells"],
            "total_neg28671_cells": census_summary["total_neg28671_cells"],
            "total_near_zero_cells": census_summary["total_near_zero_cells"],
            "histogram_near_neg28671": census_summary["histogram_near_neg28671"],
            "neg28671_unique_columns": census_summary["neg28671_unique_columns"],
            "neg28671_unique_level_indices": census_summary["neg28671_unique_level_indices"],
            "neg28671_occurrences_count": census_summary["neg28671_occurrences_count"],
            "neg28671_cells_details": census_summary["neg28671_cells_details"],
        },
        "section_D_mask_time_invariance": mask_inv,
        "section_E_suspect_columns_summary": suspect_meta,
        "section_F_level_context_summary": context_summary,
        "section_G_deep_structure_anomaly": g_summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_findings, f, indent=2)
    print(f"\nForensic JSON findings written to: {json_path}")

    return json_findings


def main():
    parser = argparse.ArgumentParser(description="GLORYS thetao Forensic Encoding Analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/glorys/glorys_thetao_2019-01.nc",
        help="Path to GLORYS NetCDF input file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/diagnostics/glorys_encoding_forensics",
        help="Path to output directory for forensic artifacts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_forensics(args.input, args.output_dir)


if __name__ == "__main__":
    main()
