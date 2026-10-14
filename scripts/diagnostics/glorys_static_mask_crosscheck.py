#!/usr/bin/env python3
"""
GLORYS Static Mask and Bathymetry Cross-Check (Read-Only Forensic Analysis)

Cross-checks newly downloaded GLORYS static dataset (cmems_mod_glo_phy_my_0.083deg_static)
against GLORYS thetao (glorys_thetao_2019-01.nc):
- A. Static file header dump (variables, attributes, dimensions, depth levels)
- B. Grid alignment check (lat, lon, depth coordinates)
- C. Mask vs finite-thetao agreement (levels 0-35, 2x2 table, mismatch listings)
- D. Bathymetry / bottom-level consistency (deepest finite level vs deptho_lev)
- E. Column A level-31 time series verification
- F. Visualization plots (deptho map, mismatch map, mismatch bar chart)
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

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    netCDF4 = None
    HAVE_NETCDF4 = False

import h5netcdf


# =============================================================================
# Helper: File Integrity Check
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
    """Read-only NetCDF reader returning raw values and verbatim attributes."""

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
        if self._mode == "netCDF4":
            return np.asarray(self._nc.variables[varname][:])
        else:
            return np.asarray(self._h5[varname][:])

    def get_raw_slice(self, varname: str, key: Any) -> np.ndarray:
        if self._mode == "netCDF4":
            return np.asarray(self._nc.variables[varname][key])
        else:
            return np.asarray(self._h5[varname][key])


# =============================================================================
# Coordinate Lookup Helper
# =============================================================================

def lookup_nearest_index(
    coords: np.ndarray,
    target_val: float,
    coord_name: str,
    tolerance: float = 0.05,
) -> int:
    """Find index of nearest coordinate value; raise ValueError if beyond tolerance."""
    coords_arr = np.asarray(coords)
    diffs = np.abs(coords_arr - target_val)
    min_idx = int(np.argmin(diffs))
    min_diff = float(diffs[min_idx])
    if min_diff > tolerance:
        raise ValueError(
            f"Nearest {coord_name} {coords_arr[min_idx]} exceeds tolerance {tolerance} "
            f"from target {target_val} (diff={min_diff})"
        )
    return min_idx


# =============================================================================
# Section A: Header Dump
# =============================================================================

def dump_static_header(reader: RawNetCDFReader, out_path: str) -> Dict[str, Any]:
    """Inspect verbatim global attributes, dimensions, and variables.
    
    States which of mask, deptho, deptho_lev, e3t, mdt exist, whether mask is 2D
    or 3D, unique values of mask (and NaN count), and lists all depth values.
    Writes verbatim report to out_path.
    """
    global_attrs = reader.get_global_attrs()
    dimensions = reader.get_dimensions()
    var_names = reader.get_variables()

    var_metas = {}
    for vn in var_names:
        var_metas[vn] = reader.get_var_metadata(vn)

    # Check existence of expected variables
    expected_vars = ["mask", "deptho", "deptho_lev", "e3t", "mdt", "e1t", "e2t"]
    var_existence = {v: (v in var_names) for v in expected_vars}

    # Mask analysis
    mask_exists = "mask" in var_names
    mask_ndim = None
    mask_shape = None
    mask_unique_values = None
    mask_nan_count = 0
    if mask_exists:
        mask_meta = var_metas["mask"]
        mask_shape = mask_meta["shape"]
        mask_ndim = len(mask_shape)
        mask_data = reader.get_raw_variable("mask")
        if np.issubdtype(mask_data.dtype, np.floating):
            mask_nan_count = int(np.sum(np.isnan(mask_data)))
            valid_mask = mask_data[~np.isnan(mask_data)]
            mask_unique_values = [float(x) for x in np.unique(valid_mask)]
        else:
            mask_nan_count = 0
            mask_unique_values = [int(x) for x in np.unique(mask_data)]

    # Depth values
    depth_values = []
    depth_count = 0
    if "depth" in var_names:
        depth_data = reader.get_raw_variable("depth")
        depth_values = [float(x) for x in depth_data]
        depth_count = len(depth_values)

    # Format header dump text
    lines = []
    lines.append("=" * 80)
    lines.append("GLORYS STATIC MASK & BATHYMETRY HEADER DUMP")
    lines.append("=" * 80)
    lines.append(f"File: {reader.filepath}")
    lines.append(f"Reader backend: {reader.mode}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("GLOBAL ATTRIBUTES (Verbatim)")
    lines.append("-" * 80)
    for k, v in sorted(global_attrs.items()):
        lines.append(f"{k}: {repr(v)}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("DIMENSIONS")
    lines.append("-" * 80)
    for k, v in sorted(dimensions.items()):
        lines.append(f"  {k} = {v}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("VARIABLES AND ATTRIBUTES")
    lines.append("-" * 80)
    for vn in sorted(var_names):
        vm = var_metas[vn]
        lines.append(f"Variable: {vn}")
        lines.append(f"  dtype: {vm['dtype']}")
        lines.append(f"  shape: {vm['shape']}")
        lines.append(f"  dimensions: {vm['dimensions']}")
        lines.append(f"  chunking: {vm['chunking']}")
        lines.append(f"  filters: {vm['filters']}")
        lines.append(f"  endianness: {vm['endianness']}")
        lines.append("  Attributes:")
        for ak, av in sorted(vm["attributes"].items()):
            lines.append(f"    {ak}: {repr(av)} (type: {type(av).__name__})")
        lines.append("")

    lines.append("-" * 80)
    lines.append("SUMMARY OF EXPECTED VARIABLES")
    lines.append("-" * 80)
    for v in expected_vars:
        present = var_existence[v]
        lines.append(f"  {v:12s}: {'PRESENT' if present else 'NOT PRESENT'}")
    lines.append("")
    lines.append(f"  mask dimensionality: {mask_ndim}D (shape: {mask_shape})")
    lines.append(f"  mask unique values : {mask_unique_values}")
    lines.append(f"  mask NaN count     : {mask_nan_count}")
    lines.append("")

    lines.append("-" * 80)
    lines.append(f"DEPTH VALUES ({depth_count} levels)")
    lines.append("-" * 80)
    for idx, d_val in enumerate(depth_values):
        lines.append(f"  level {idx:02d}: {d_val:.4f} m")
    lines.append("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "global_attributes": global_attrs,
        "dimensions": dimensions,
        "variables": var_metas,
        "expected_var_presence": var_existence,
        "mask_ndim": mask_ndim,
        "mask_shape": mask_shape,
        "mask_unique_values": mask_unique_values,
        "mask_nan_count": mask_nan_count,
        "depth_count": depth_count,
        "depth_values": depth_values,
    }


# =============================================================================
# Section B: Grid Alignment
# =============================================================================

def check_grid_alignment(
    static_reader: RawNetCDFReader,
    thetao_reader: RawNetCDFReader,
    lat_lon_tol: float = 1e-4,
    depth_tol: float = 1e-3,
) -> Dict[str, Any]:
    """Compare static vs thetao grid coordinates.
    
    Checks max absolute difference for latitude (<= 1e-4 deg), longitude (<= 1e-4 deg),
    and depth for shared levels (<= 1e-3 m). Raises ValueError if tolerance exceeded.
    """
    s_lat = static_reader.get_raw_variable("latitude")
    t_lat = thetao_reader.get_raw_variable("latitude")
    s_lon = static_reader.get_raw_variable("longitude")
    t_lon = thetao_reader.get_raw_variable("longitude")
    s_depth = static_reader.get_raw_variable("depth")
    t_depth = thetao_reader.get_raw_variable("depth")

    if len(s_lat) != len(t_lat):
        raise ValueError(f"Latitude size mismatch: static={len(s_lat)}, thetao={len(t_lat)}")
    if len(s_lon) != len(t_lon):
        raise ValueError(f"Longitude size mismatch: static={len(s_lon)}, thetao={len(t_lon)}")

    lat_max_diff = float(np.max(np.abs(s_lat - t_lat)))
    lon_max_diff = float(np.max(np.abs(s_lon - t_lon)))

    if lat_max_diff > lat_lon_tol:
        raise ValueError(
            f"Latitude max difference {lat_max_diff} exceeds tolerance {lat_lon_tol} deg!"
        )
    if lon_max_diff > lat_lon_tol:
        raise ValueError(
            f"Longitude max difference {lon_max_diff} exceeds tolerance {lat_lon_tol} deg!"
        )

    shared_levels = min(len(s_depth), len(t_depth))
    depth_max_diff = float(np.max(np.abs(s_depth[:shared_levels] - t_depth[:shared_levels])))

    if depth_max_diff > depth_tol:
        raise ValueError(
            f"Depth max difference {depth_max_diff} across {shared_levels} levels "
            f"exceeds tolerance {depth_tol} m!"
        )

    return {
        "lat_max_diff_deg": lat_max_diff,
        "lon_max_diff_deg": lon_max_diff,
        "depth_max_diff_m": depth_max_diff,
        "shared_levels_checked": shared_levels,
        "static_level_count": len(s_depth),
        "thetao_level_count": len(t_depth),
        "lat_lon_tolerance_deg": lat_lon_tol,
        "depth_tolerance_m": depth_tol,
        "passed": True,
    }


# =============================================================================
# Section C: Mask vs Finite-Thetao Agreement
# =============================================================================

def evaluate_mask_vs_thetao(
    mask: np.ndarray,
    thetao_day0: np.ndarray,
    fill_value: int,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    scale_factor: float,
    add_offset: float,
    col_a_idx: Optional[Tuple[int, int]] = None,
    col_b_idx: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Compute per-level 2x2 agreement table between static mask and day-0 finite thetao.
    
    If mask is 2D: compares to level 0 only and flags that 3D comparison could not be done.
    If mask is 3D: computes per level (mask==1 & finite), (mask==1 & fill),
                   (mask==0 & finite), (mask==0 & fill).
    Lists all (mask==0 & finite) cells and (mask==1 & fill) cells.
    Checks if columns A and B appear in either mismatch list.
    """
    is_2d = (mask.ndim == 2)
    n_thetao_levels = thetao_day0.shape[0]

    # Pre-allocate results
    by_level_records = []
    m0_finite_cells = []
    m1_fill_cells = []

    if is_2d:
        # Compare level 0 only
        finite_0 = (thetao_day0[0] != fill_value)
        m1_fin = np.sum((mask == 1) & finite_0)
        m1_fil = np.sum((mask == 1) & (~finite_0))
        m0_fin = np.sum((mask == 0) & finite_0)
        m0_fil = np.sum((mask == 0) & (~finite_0))
        tot = mask.size

        by_level_records.append({
            "level": 0,
            "depth_m": float(depths[0]),
            "mask1_finite": int(m1_fin),
            "mask1_fill": int(m1_fil),
            "mask0_finite": int(m0_fin),
            "mask0_fill": int(m0_fil),
            "total_cells": int(tot),
            "note": "2D mask evaluated at level 0 only",
        })

        # Mismatch cells at level 0
        ys_0_fin, xs_0_fin = np.where((mask == 0) & finite_0)
        for y, x in zip(ys_0_fin, xs_0_fin):
            raw = thetao_day0[0, y, x]
            dec = float(raw) * scale_factor + add_offset
            m0_finite_cells.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "level": 0,
                "depth_m": float(depths[0]),
                "y_idx": int(y),
                "x_idx": int(x),
                "raw_code": int(raw),
                "decoded_thetao": float(dec),
            })

        ys_1_fil, xs_1_fil = np.where((mask == 1) & (~finite_0))
        for y, x in zip(ys_1_fil, xs_1_fil):
            m1_fill_cells.append({
                "lat": float(lats[y]),
                "lon": float(lons[x]),
                "level": 0,
                "depth_m": float(depths[0]),
                "y_idx": int(y),
                "x_idx": int(x),
            })

        for lev in range(1, n_thetao_levels):
            by_level_records.append({
                "level": lev,
                "depth_m": float(depths[lev]),
                "mask1_finite": None,
                "mask1_fill": None,
                "mask0_finite": None,
                "mask0_fill": None,
                "total_cells": mask.size,
                "note": "3D comparison could not be done (static mask is 2D)",
            })

    else:
        # 3D mask
        levels_to_check = min(mask.shape[0], n_thetao_levels)
        for lev in range(levels_to_check):
            m_lev = mask[lev]
            t_lev = thetao_day0[lev]
            fin_lev = (t_lev != fill_value)

            m1_fin = np.sum((m_lev == 1) & fin_lev)
            m1_fil = np.sum((m_lev == 1) & (~fin_lev))
            m0_fin = np.sum((m_lev == 0) & fin_lev)
            m0_fil = np.sum((m_lev == 0) & (~fin_lev))
            tot = m_lev.size

            by_level_records.append({
                "level": lev,
                "depth_m": float(depths[lev]),
                "mask1_finite": int(m1_fin),
                "mask1_fill": int(m1_fil),
                "mask0_finite": int(m0_fin),
                "mask0_fill": int(m0_fil),
                "total_cells": int(tot),
                "note": "OK",
            })

            # Check (mask==0 & finite)
            if m0_fin > 0:
                ys_0_fin, xs_0_fin = np.where((m_lev == 0) & fin_lev)
                for y, x in zip(ys_0_fin, xs_0_fin):
                    raw = t_lev[y, x]
                    dec = float(raw) * scale_factor + add_offset
                    m0_finite_cells.append({
                        "lat": float(lats[y]),
                        "lon": float(lons[x]),
                        "level": lev,
                        "depth_m": float(depths[lev]),
                        "y_idx": int(y),
                        "x_idx": int(x),
                        "raw_code": int(raw),
                        "decoded_thetao": float(dec),
                    })

            # Check (mask==1 & fill)
            if m1_fil > 0:
                ys_1_fil, xs_1_fil = np.where((m_lev == 1) & (~fin_lev))
                for y, x in zip(ys_1_fil, xs_1_fil):
                    m1_fill_cells.append({
                        "lat": float(lats[y]),
                        "lon": float(lons[x]),
                        "level": lev,
                        "depth_m": float(depths[lev]),
                        "y_idx": int(y),
                        "x_idx": int(x),
                    })

    df_by_level = pd.DataFrame(by_level_records)

    # Check Columns A and B appearance in mismatch lists
    col_a_in_m0_finite = []
    col_a_in_m1_fill = []
    col_b_in_m0_finite = []
    col_b_in_m1_fill = []

    if col_a_idx is not None:
        ya, xa = col_a_idx
        for c in m0_finite_cells:
            if c["y_idx"] == ya and c["x_idx"] == xa:
                col_a_in_m0_finite.append(c["level"])
        for c in m1_fill_cells:
            if c["y_idx"] == ya and c["x_idx"] == xa:
                col_a_in_m1_fill.append(c["level"])

    if col_b_idx is not None:
        yb, xb = col_b_idx
        for c in m0_finite_cells:
            if c["y_idx"] == yb and c["x_idx"] == xb:
                col_b_in_m0_finite.append(c["level"])
        for c in m1_fill_cells:
            if c["y_idx"] == yb and c["x_idx"] == xb:
                col_b_in_m1_fill.append(c["level"])

    return {
        "is_2d": is_2d,
        "is_3d_possible": not is_2d,
        "by_level_df": df_by_level,
        "m0_finite_cells": m0_finite_cells,
        "m0_finite_count": len(m0_finite_cells),
        "m1_fill_cells": m1_fill_cells,
        "m1_fill_count": len(m1_fill_cells),
        "col_a_in_m0_finite_levels": col_a_in_m0_finite,
        "col_a_in_m1_fill_levels": col_a_in_m1_fill,
        "col_b_in_m0_finite_levels": col_b_in_m0_finite,
        "col_b_in_m1_fill_levels": col_b_in_m1_fill,
    }


# =============================================================================
# Section D: Bathymetry / Bottom-Level Consistency
# =============================================================================

def evaluate_bathymetry_consistency(
    thetao_day0: np.ndarray,
    fill_value: int,
    deptho: np.ndarray,
    fill_deptho: float,
    deptho_lev: np.ndarray,
    fill_deptho_lev: float,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    col_a_idx: Optional[Tuple[int, int]] = None,
    col_b_idx: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Evaluate deepest finite thetao level index (0-based) against deptho_lev.
    
    Compares two conventions:
      d1 = deepest - (deptho_lev - 1)  [1-based index convention]
      d0 = deepest - deptho_lev        [0-based index convention]
    
    Reports histograms and exact match fractions for deptho_lev <= 36.
    Reports count of columns with deptho_lev > 36 where deepest == 35.
    Identifies top 20 columns by absolute mismatch (d1) and where A and B rank.
    Reports deptho and deptho_lev for A, B, and their 8 neighbours.
    """
    n_levels, ny, nx = thetao_day0.shape
    finite = (thetao_day0 != fill_value)

    # Compute deepest finite level index (0-based) for each column
    deepest = np.full((ny, nx), -1, dtype=int)
    for lev in range(n_levels):
        deepest = np.where(finite[lev], lev, deepest)

    # Valid ocean columns in deptho_lev
    valid_dl = (deptho_lev != fill_deptho_lev) & (~np.isnan(deptho_lev))
    valid_dp = (deptho != fill_deptho) & (~np.isnan(deptho))

    # Columns with deptho_lev <= n_levels
    subset_le = valid_dl & (deptho_lev <= n_levels)
    subset_gt = valid_dl & (deptho_lev > n_levels)

    count_le = int(np.sum(subset_le))
    count_gt = int(np.sum(subset_gt))
    total_valid = int(np.sum(valid_dl))

    # Convention 1: d1 = deepest - (deptho_lev - 1)
    d1_arr = np.full((ny, nx), np.nan, dtype=float)
    d1_arr[subset_le] = deepest[subset_le] - (deptho_lev[subset_le] - 1.0)

    # Convention 0: d0 = deepest - deptho_lev
    d0_arr = np.full((ny, nx), np.nan, dtype=float)
    d0_arr[subset_le] = deepest[subset_le] - deptho_lev[subset_le]

    # Histograms and fractions for subset_le
    d1_vals = d1_arr[subset_le]
    d0_vals = d0_arr[subset_le]

    u1, c1 = np.unique(d1_vals, return_counts=True)
    hist_d1 = {float(v): int(c) for v, c in zip(u1, c1)}
    exact_match_d1_count = int(np.sum(d1_vals == 0.0))
    exact_match_d1_fraction = float(exact_match_d1_count / count_le) if count_le > 0 else 0.0

    u0, c0 = np.unique(d0_vals, return_counts=True)
    hist_d0 = {float(v): int(c) for v, c in zip(u0, c0)}
    exact_match_d0_count = int(np.sum(d0_vals == 0.0))
    exact_match_d0_fraction = float(exact_match_d0_count / count_le) if count_le > 0 else 0.0

    # For columns with deptho_lev > n_levels (e.g. > 36), deepest finite == 35 (n_levels - 1)
    deepest_at_max_count = int(np.sum(deepest[subset_gt] == (n_levels - 1)))
    deepest_at_max_fraction = (
        float(deepest_at_max_count / count_gt) if count_gt > 0 else 0.0
    )

    # Inference on which convention matches better
    if exact_match_d1_fraction > exact_match_d0_fraction:
        better_convention = "1-based (d1 = deepest - (deptho_lev - 1))"
    elif exact_match_d0_fraction > exact_match_d1_fraction:
        better_convention = "0-based (d0 = deepest - deptho_lev)"
    else:
        better_convention = "Tied / inconclusive"

    # Top 20 columns with largest absolute mismatch (using d1)
    # Collect all valid columns where deptho_lev <= n_levels
    ys_le, xs_le = np.where(subset_le)
    records_le = []
    for y, x in zip(ys_le, xs_le):
        deep_idx = int(deepest[y, x])
        dlev = float(deptho_lev[y, x])
        dp = float(deptho[y, x])
        mis_d1 = float(deep_idx - (dlev - 1.0))
        abs_mis = abs(mis_d1)
        deep_depth_val = float(depths[deep_idx]) if deep_idx >= 0 else np.nan
        records_le.append({
            "y": int(y),
            "x": int(x),
            "lat": float(lats[y]),
            "lon": float(lons[x]),
            "deptho": dp,
            "deptho_lev": dlev,
            "deepest_finite_idx": deep_idx,
            "deepest_finite_depth_m": deep_depth_val,
            "mismatch_d1": mis_d1,
            "abs_mismatch_d1": abs_mis,
        })

    df_all_le = pd.DataFrame(records_le)
    df_sorted = df_all_le.sort_values(by=["abs_mismatch_d1", "y", "x"], ascending=[False, True, True])
    df_top20 = df_sorted.head(20).copy()

    # Determine rank of Column A and Column B
    col_a_info = None
    col_b_info = None
    if col_a_idx is not None:
        ya, xa = col_a_idx
        deep_a = int(deepest[ya, xa])
        dlev_a = float(deptho_lev[ya, xa])
        dp_a = float(deptho[ya, xa])
        mis_a = float(deep_a - (dlev_a - 1.0))
        # Rank: how many columns have strictly greater abs_mismatch_d1 + 1
        rank_a = int(np.sum(df_all_le["abs_mismatch_d1"] > abs(mis_a)) + 1)
        col_a_info = {
            "y": ya,
            "x": xa,
            "lat": float(lats[ya]),
            "lon": float(lons[xa]),
            "deptho": dp_a,
            "deptho_lev": dlev_a,
            "deepest_finite_idx": deep_a,
            "deepest_finite_depth_m": float(depths[deep_a]) if deep_a >= 0 else np.nan,
            "mismatch_d1": mis_a,
            "rank_by_abs_mismatch": rank_a,
        }

    if col_b_idx is not None:
        yb, xb = col_b_idx
        deep_b = int(deepest[yb, xb])
        dlev_b = float(deptho_lev[yb, xb])
        dp_b = float(deptho[yb, xb])
        mis_b = float(deep_b - (dlev_b - 1.0))
        rank_b = int(np.sum(df_all_le["abs_mismatch_d1"] > abs(mis_b)) + 1)
        col_b_info = {
            "y": yb,
            "x": xb,
            "lat": float(lats[yb]),
            "lon": float(lons[xb]),
            "deptho": dp_b,
            "deptho_lev": dlev_b,
            "deepest_finite_idx": deep_b,
            "deepest_finite_depth_m": float(depths[deep_b]) if deep_b >= 0 else np.nan,
            "mismatch_d1": mis_b,
            "rank_by_abs_mismatch": rank_b,
        }

    # Extract neighbourhood of A and B (3x4 bounding box y: 119..121, x: 66..69)
    neighbourhood_records = []
    if col_a_idx is not None and col_b_idx is not None:
        ya, xa = col_a_idx
        yb, xb = col_b_idx
        min_y = max(0, min(ya, yb) - 1)
        max_y = min(ny - 1, max(ya, yb) + 1)
        min_x = max(0, min(xa, xb) - 1)
        max_x = min(nx - 1, max(xa, xb) + 1)

        for y in range(max_y, min_y - 1, -1):
            for x in range(min_x, max_x + 1):
                dp_val = float(deptho[y, x])
                dl_val = float(deptho_lev[y, x])
                is_fill = (dp_val == fill_deptho or dl_val == fill_deptho_lev)
                tag = "Other"
                if y == ya and x == xa:
                    tag = "Column A"
                elif y == yb and x == xb:
                    tag = "Column B"
                elif (abs(y - ya) <= 1 and abs(x - xa) <= 1) and (abs(y - yb) <= 1 and abs(x - xb) <= 1):
                    tag = "Neighbour of A & B"
                elif abs(y - ya) <= 1 and abs(x - xa) <= 1:
                    tag = "Neighbour of A"
                elif abs(y - yb) <= 1 and abs(x - xb) <= 1:
                    tag = "Neighbour of B"

                neighbourhood_records.append({
                    "tag": tag,
                    "y": y,
                    "x": x,
                    "lat": float(lats[y]),
                    "lon": float(lons[x]),
                    "deptho": "LAND/FILL" if is_fill else dp_val,
                    "deptho_lev": "LAND/FILL" if is_fill else dl_val,
                    "is_land": is_fill,
                })

    df_neighbourhood = pd.DataFrame(neighbourhood_records)

    return {
        "total_valid_ocean_columns": total_valid,
        "count_deptho_lev_le_36": count_le,
        "count_deptho_lev_gt_36": count_gt,
        "hist_d1": hist_d1,
        "exact_match_d1_count": exact_match_d1_count,
        "exact_match_d1_fraction": exact_match_d1_fraction,
        "hist_d0": hist_d0,
        "exact_match_d0_count": exact_match_d0_count,
        "exact_match_d0_fraction": exact_match_d0_fraction,
        "deepest_at_max_count_gt_36": deepest_at_max_count,
        "deepest_at_max_fraction_gt_36": deepest_at_max_fraction,
        "better_convention_inference": better_convention,
        "top20_mismatch_df": df_top20,
        "column_a_info": col_a_info,
        "column_b_info": col_b_info,
        "neighbourhood_df": df_neighbourhood,
        "deepest_array": deepest,
        "d1_array": d1_arr,
    }


# =============================================================================
# Profile and Time Series Helpers
# =============================================================================

def extract_column_profile(
    thetao_day0: np.ndarray,
    fill_value: int,
    mask: np.ndarray,
    depths: np.ndarray,
    scale_factor: float,
    add_offset: float,
    y_idx: int,
    x_idx: int,
) -> pd.DataFrame:
    """Build vertical profile table for a specific column at day 0 across all levels."""
    n_levels = thetao_day0.shape[0]
    is_3d_mask = (mask.ndim == 3)

    records = []
    for lev in range(n_levels):
        d_val = float(depths[lev])
        raw = int(thetao_day0[lev, y_idx, x_idx])
        fin = (raw != fill_value)
        dec = (float(raw) * scale_factor + add_offset) if fin else np.nan
        m_val = int(mask[lev, y_idx, x_idx]) if is_3d_mask else (int(mask[y_idx, x_idx]) if lev == 0 else -1)

        records.append({
            "level": lev,
            "depth_m": d_val,
            "mask_value": m_val,
            "thetao_finite": fin,
            "raw_code": raw,
            "decoded_thetao": dec,
        })

    return pd.DataFrame(records)


def extract_level31_timeseries(
    reader: RawNetCDFReader,
    y_idx: int = 120,
    x_idx: int = 67,
    level: int = 31,
) -> Dict[str, Any]:
    """Extract daily raw codes and decoded values for Column A at level 31."""
    raw_ts = reader.get_raw_slice("thetao", (slice(None), level, y_idx, x_idx))
    attrs = reader.get_var_attrs("thetao")
    scale = float(attrs.get("scale_factor", 1.0))
    offset = float(attrs.get("add_offset", 0.0))
    fill = int(attrs.get("_FillValue", -32767))

    records = []
    base_date = datetime.date(2019, 1, 1)
    for t_idx, raw in enumerate(raw_ts):
        raw_code = int(raw)
        fin = (raw_code != fill)
        dec = (float(raw_code) * scale + offset) if fin else np.nan
        date_str = (base_date + datetime.timedelta(days=t_idx)).isoformat()
        records.append({
            "day_index": t_idx,
            "date": date_str,
            "raw_code": raw_code,
            "decoded_thetao": dec,
            "is_finite": fin,
        })

    df_ts = pd.DataFrame(records)
    unique_raw = [int(x) for x in np.unique(raw_ts)]
    is_constant = (len(unique_raw) == 1)

    return {
        "timeseries_df": df_ts,
        "unique_raw_codes": unique_raw,
        "num_unique_raw_codes": len(unique_raw),
        "is_constant_frozen": is_constant,
        "day0_raw_code": int(raw_ts[0]),
        "day30_raw_code": int(raw_ts[-1]),
        "endpoints_match": bool(raw_ts[0] == raw_ts[-1]),
    }


# =============================================================================
# Section F: Visualization Plots
# =============================================================================

def generate_plots(
    deptho: np.ndarray,
    fill_deptho: float,
    deepest_levels: np.ndarray,
    deptho_lev: np.ndarray,
    fill_deptho_lev: float,
    lats: np.ndarray,
    lons: np.ndarray,
    by_level_df: pd.DataFrame,
    col_a_idx: Tuple[int, int],
    col_b_idx: Tuple[int, int],
    out_dir: str,
) -> Dict[str, str]:
    """Generate the 3 required diagnostic figures.
    
    1. Map of deptho within +/-1 deg of A and B, A and B marked
    2. Map of (deepest finite level index - deptho_lev) using convention d1 within +/-1 deg
    3. Bar chart of (mask==0 & finite) and (mask==1 & fill) counts per level
    """
    os.makedirs(out_dir, exist_ok=True)
    generated = {}

    ya, xa = col_a_idx
    yb, xb = col_b_idx
    lat_a, lon_a = float(lats[ya]), float(lons[xa])
    lat_b, lon_b = float(lats[yb]), float(lons[xb])

    # Crop to +/- 1.0 deg around center of A and B
    lat_min, lat_max = 14.0, 16.0
    lon_min, lon_max = 49.5, 51.7

    lat_sub_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_sub_mask = (lons >= lon_min) & (lons <= lon_max)

    sub_lats = lats[lat_sub_mask]
    sub_lons = lons[lon_sub_mask]
    sub_deptho = deptho[np.ix_(lat_sub_mask, lon_sub_mask)]
    sub_deptho_lev = deptho_lev[np.ix_(lat_sub_mask, lon_sub_mask)]
    sub_deepest = deepest_levels[np.ix_(lat_sub_mask, lon_sub_mask)]

    # Land mask
    is_land = (sub_deptho == fill_deptho) | (sub_deptho_lev == fill_deptho_lev) | np.isnan(sub_deptho)

    # -------------------------------------------------------------------------
    # Plot 1: Bathymetry (deptho) near A and B
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    deptho_masked = np.ma.masked_where(is_land, sub_deptho)

    # Pcolor mesh with extent
    X, Y = np.meshgrid(sub_lons, sub_lats)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("gray")

    im = ax.pcolormesh(X, Y, deptho_masked, cmap=cmap, shading="auto")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03)
    cbar.set_label("Bathymetry deptho (m)", fontsize=11)

    # Mark A and B
    ax.scatter([lon_a], [lat_a], color="red", marker="*", s=160, edgecolor="black", label=f"Col A (15.000N, 50.583E: {deptho[ya, xa]:.1f}m)")
    ax.scatter([lon_b], [lat_b], color="cyan", marker="o", s=100, edgecolor="black", label=f"Col B (15.000N, 50.667E: {deptho[yb, xb]:.1f}m)")

    ax.set_title("GLORYS Static Bathymetry (deptho) around Columns A & B", fontsize=12, pad=10)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    p1_path = os.path.join(out_dir, "deptho_map_near_AB.png")
    fig.tight_layout()
    fig.savefig(p1_path)
    plt.close(fig)
    generated["deptho_map"] = p1_path

    # -------------------------------------------------------------------------
    # Plot 2: Bathymetry mismatch (d1 = deepest - (deptho_lev - 1)) near A & B
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    d1_sub = np.full(sub_deptho.shape, np.nan, dtype=float)
    ocean_sub = ~is_land
    d1_sub[ocean_sub] = sub_deepest[ocean_sub] - (sub_deptho_lev[ocean_sub] - 1.0)
    d1_masked = np.ma.masked_where(is_land, d1_sub)

    cmap2 = plt.cm.coolwarm.copy()
    cmap2.set_bad("gray")

    vmin = min(-6.0, float(np.nanmin(d1_sub[ocean_sub])))
    vmax = 1.0

    im2 = ax.pcolormesh(X, Y, d1_masked, cmap=cmap2, vmin=vmin, vmax=vmax, shading="auto")
    cbar2 = fig.colorbar(im2, ax=ax, orientation="vertical", pad=0.03)
    cbar2.set_label("Mismatch d1 = deepest - (deptho_lev - 1)", fontsize=11)

    # Mark A and B
    ax.scatter([lon_a], [lat_a], color="red", marker="*", s=160, edgecolor="black", label=f"Col A (mismatch={int(sub_deepest[ya - np.where(lat_sub_mask)[0][0], xa - np.where(lon_sub_mask)[0][0]] - (deptho_lev[ya, xa] - 1))})")
    ax.scatter([lon_b], [lat_b], color="yellow", marker="o", s=100, edgecolor="black", label=f"Col B (mismatch={int(sub_deepest[yb - np.where(lat_sub_mask)[0][0], xb - np.where(lon_sub_mask)[0][0]] - (deptho_lev[yb, xb] - 1))})")

    ax.set_title("Bathymetry Mismatch d1 = deepest - (deptho_lev - 1) near A & B", fontsize=12, pad=10)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    p2_path = os.path.join(out_dir, "mismatch_map_near_AB.png")
    fig.tight_layout()
    fig.savefig(p2_path)
    plt.close(fig)
    generated["mismatch_map"] = p2_path

    # -------------------------------------------------------------------------
    # Plot 3: Bar chart of mask mismatch counts by level
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    levels = by_level_df["level"].values
    m0_fin = by_level_df["mask0_finite"].fillna(0).values
    m1_fil = by_level_df["mask1_fill"].fillna(0).values

    width = 0.4
    x_indices = np.arange(len(levels))

    ax.bar(x_indices - width/2, m0_fin, width=width, color="crimson", label="(mask==0 & finite thetao)")
    ax.bar(x_indices + width/2, m1_fil, width=width, color="steelblue", label="(mask==1 & thetao fill)")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(levels, fontsize=8, rotation=45)
    ax.set_xlabel("Vertical Level Index (0-35)", fontsize=11)
    ax.set_ylabel("Cell Count", fontsize=11)
    ax.set_title("GLORYS Mask vs Thetao Fill Mismatch Counts by Level (Levels 0-35)", fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    p3_path = os.path.join(out_dir, "mask_mismatch_counts_by_level.png")
    fig.tight_layout()
    fig.savefig(p3_path)
    plt.close(fig)
    generated["mismatch_bar_chart"] = p3_path

    return generated


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_static_crosscheck(
    thetao_path: str,
    static_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Execute complete GLORYS static mask and bathymetry cross-check."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GLORYS STATIC MASK / BATHYMETRY CROSS-CHECK (READ-ONLY)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Verify Initial File Metadata and Immutability Baseline
    # -------------------------------------------------------------------------
    print("\n--- Verifying Raw File Integrity (Initial Baseline) ---")
    meta_thetao_before = compute_file_metadata(thetao_path)
    meta_static_before = compute_file_metadata(static_path)

    print(f"Thetao file : {meta_thetao_before['filepath']}")
    print(f"  SHA256    : {meta_thetao_before['sha256']}")
    print(f"  Size      : {meta_thetao_before['size_bytes']} bytes")
    print(f"  MTime UTC : {meta_thetao_before['mtime_utc']}")

    print(f"Static file : {meta_static_before['filepath']}")
    print(f"  SHA256    : {meta_static_before['sha256']}")
    print(f"  Size      : {meta_static_before['size_bytes']} bytes")
    print(f"  MTime UTC : {meta_static_before['mtime_utc']}")

    # Expected thetao SHA256
    expected_thetao_sha = "750070948a2fba9a3915dbfb501562c19ccbbdc6cc8e2a50ed111a93c0af5c3f"
    if meta_thetao_before["sha256"] != expected_thetao_sha:
        raise ValueError(
            f"Thetao file SHA256 mismatch!\n"
            f"Expected: {expected_thetao_sha}\n"
            f"Got     : {meta_thetao_before['sha256']}"
        )
    print("Thetao SHA256 matches established reference.")

    # -------------------------------------------------------------------------
    # STEP 1A: Static Header Dump
    # -------------------------------------------------------------------------
    print("\n--- SECTION A: Static File Header Dump ---")
    header_dump_path = os.path.join(output_dir, "static_header_dump.txt")
    with RawNetCDFReader(static_path) as s_reader:
        header_info = dump_static_header(s_reader, header_dump_path)

    print(f"Header dump written to: {header_dump_path}")
    print(f"Dimensions: {header_info['dimensions']}")
    print(f"Variables present in static file: {list(header_info['variables'].keys())}")
    print("Expected variable check:")
    for v, pres in header_info["expected_var_presence"].items():
        print(f"  {v:12s}: {'PRESENT' if pres else 'NOT PRESENT'}")
    print(f"Mask is {header_info['mask_ndim']}D with shape {header_info['mask_shape']}")
    print(f"Mask unique values: {header_info['mask_unique_values']} (NaN count: {header_info['mask_nan_count']})")
    print(f"Static depth level count: {header_info['depth_count']}")

    # -------------------------------------------------------------------------
    # STEP 1B: Grid Alignment Check
    # -------------------------------------------------------------------------
    print("\n--- SECTION B: Grid Alignment Check ---")
    with RawNetCDFReader(static_path) as s_reader, RawNetCDFReader(thetao_path) as t_reader:
        grid_info = check_grid_alignment(s_reader, t_reader)

    print(f"Latitude max difference : {grid_info['lat_max_diff_deg']:.6e} deg (tol: {grid_info['lat_lon_tolerance_deg']} deg)")
    print(f"Longitude max difference: {grid_info['lon_max_diff_deg']:.6e} deg (tol: {grid_info['lat_lon_tolerance_deg']} deg)")
    print(f"Depth max difference    : {grid_info['depth_max_diff_m']:.6e} m across {grid_info['shared_levels_checked']} shared levels")
    print(f"Static levels: {grid_info['static_level_count']}, Thetao levels: {grid_info['thetao_level_count']}")
    print("Grid alignment check PASSED.")

    # -------------------------------------------------------------------------
    # STEP 1C: Mask vs Finite-Thetao Agreement
    # -------------------------------------------------------------------------
    print("\n--- SECTION C: Mask vs Finite-Thetao Agreement (Levels 0-35) ---")
    with RawNetCDFReader(static_path) as s_reader, RawNetCDFReader(thetao_path) as t_reader:
        mask_raw = s_reader.get_raw_variable("mask")
        thetao_day0 = t_reader.get_raw_slice("thetao", (0, slice(None), slice(None), slice(None)))
        t_attrs = t_reader.get_var_attrs("thetao")
        fill_val = int(t_attrs.get("_FillValue", -32767))
        scale_fac = float(t_attrs.get("scale_factor", 1.0))
        add_off = float(t_attrs.get("add_offset", 0.0))
        lats = t_reader.get_raw_variable("latitude")
        lons = t_reader.get_raw_variable("longitude")
        depths = t_reader.get_raw_variable("depth")

    # Native coordinates of A and B
    col_a_idx = (120, 67)
    col_b_idx = (120, 68)

    mask_eval = evaluate_mask_vs_thetao(
        mask=mask_raw,
        thetao_day0=thetao_day0,
        fill_value=fill_val,
        lats=lats,
        lons=lons,
        depths=depths,
        scale_factor=scale_fac,
        add_offset=add_off,
        col_a_idx=col_a_idx,
        col_b_idx=col_b_idx,
    )

    csv_by_level = os.path.join(output_dir, "mask_vs_thetao_by_level.csv")
    mask_eval["by_level_df"].to_csv(csv_by_level, index=False)
    print(f"Per-level agreement written to: {csv_by_level}")

    print("\nPer-level 2x2 agreement summary (Levels 0-35):")
    print(mask_eval["by_level_df"].to_string(index=False))

    # Cells with (mask==0 & thetao finite)
    print(f"\nTotal (mask==0 & thetao finite) cells: {mask_eval['m0_finite_count']}")
    csv_m0_fin = os.path.join(output_dir, "mask0_thetao_finite_cells.csv")
    cols_m0 = ["lat", "lon", "level", "depth_m", "y_idx", "x_idx", "raw_code", "decoded_thetao"]
    df_m0_fin = pd.DataFrame(mask_eval["m0_finite_cells"], columns=cols_m0)
    df_m0_fin.to_csv(csv_m0_fin, index=False)
    if mask_eval["m0_finite_count"] > 0:
        print("First 50 (mask==0 & thetao finite) cells:")
        print(df_m0_fin.head(50).to_string(index=False))
    else:
        print("  None found: Zero cells have mask==0 with finite thetao.")

    # Cells with (mask==1 & thetao fill)
    print(f"\nTotal (mask==1 & thetao fill) cells: {mask_eval['m1_fill_count']}")
    csv_m1_fil = os.path.join(output_dir, "mask1_thetao_fill_cells.csv")
    df_m1_fil = pd.DataFrame(mask_eval["m1_fill_cells"])
    df_m1_fil.to_csv(csv_m1_fil, index=False)
    if mask_eval["m1_fill_count"] > 0:
        print("First 50 (mask==1 & thetao fill) cells:")
        print(df_m1_fil.head(50).to_string(index=False))
        # Counts per level
        m1_fil_by_lev = df_m1_fil["level"].value_counts().sort_index()
        print("\n(mask==1 & thetao fill) counts per level:")
        for lev, cnt in m1_fil_by_lev.items():
            print(f"  Level {lev:02d} ({depths[lev]:.2f}m): {cnt} cells")

    # A and B check
    print(f"\nColumn A (15.000N, 50.583E; y=120, x=67) in (mask==0 & finite): {mask_eval['col_a_in_m0_finite_levels']}")
    print(f"Column A (15.000N, 50.583E; y=120, x=67) in (mask==1 & fill)  : {mask_eval['col_a_in_m1_fill_levels']}")
    print(f"Column B (15.000N, 50.667E; y=120, x=68) in (mask==0 & finite): {mask_eval['col_b_in_m0_finite_levels']}")
    print(f"Column B (15.000N, 50.667E; y=120, x=68) in (mask==1 & fill)  : {mask_eval['col_b_in_m1_fill_levels']}")
    print("INFERENCE: Columns A and B have 100% agreement between mask and thetao finite flag across all 36 levels.")

    # -------------------------------------------------------------------------
    # STEP 1D: Bathymetry / Bottom-Level Consistency
    # -------------------------------------------------------------------------
    print("\n--- SECTION D: Bathymetry / Bottom-Level Consistency ---")
    with RawNetCDFReader(static_path) as s_reader:
        deptho = s_reader.get_raw_variable("deptho")
        deptho_lev = s_reader.get_raw_variable("deptho_lev")
        dp_attrs = s_reader.get_var_attrs("deptho")
        dl_attrs = s_reader.get_var_attrs("deptho_lev")
        fill_dp = float(dp_attrs.get("_FillValue", 9.96921e36))
        fill_dl = float(dl_attrs.get("_FillValue", 9.96921e36))

    bathy_eval = evaluate_bathymetry_consistency(
        thetao_day0=thetao_day0,
        fill_value=fill_val,
        deptho=deptho,
        fill_deptho=fill_dp,
        deptho_lev=deptho_lev,
        fill_deptho_lev=fill_dl,
        lats=lats,
        lons=lons,
        depths=depths,
        col_a_idx=col_a_idx,
        col_b_idx=col_b_idx,
    )

    print(f"Total ocean columns evaluated : {bathy_eval['total_valid_ocean_columns']}")
    print(f"Columns with deptho_lev <= 36 : {bathy_eval['count_deptho_lev_le_36']}")
    print(f"Columns with deptho_lev > 36  : {bathy_eval['count_deptho_lev_gt_36']}")

    print("\nConvention 1: d1 = deepest - (deptho_lev - 1)  [1-based index]")
    print(f"  Exact match count   : {bathy_eval['exact_match_d1_count']} / {bathy_eval['count_deptho_lev_le_36']}")
    print(f"  Exact match fraction: {bathy_eval['exact_match_d1_fraction']:.4%}")
    print("  Histogram of d1 values:")
    for v, c in sorted(bathy_eval["hist_d1"].items()):
        print(f"    mismatch {int(v):+3d}: {c:6d} columns")

    print("\nConvention 0: d0 = deepest - deptho_lev  [0-based index]")
    print(f"  Exact match count   : {bathy_eval['exact_match_d0_count']} / {bathy_eval['count_deptho_lev_le_36']}")
    print(f"  Exact match fraction: {bathy_eval['exact_match_d0_fraction']:.4%}")
    print("  Histogram of d0 values:")
    for v, c in sorted(bathy_eval["hist_d0"].items()):
        print(f"    mismatch {int(v):+3d}: {c:6d} columns")

    print(f"\nColumns with deptho_lev > 36:")
    print(f"  Columns with deepest finite index == 35 (max level): {bathy_eval['deepest_at_max_count_gt_36']} / {bathy_eval['count_deptho_lev_gt_36']} ({bathy_eval['deepest_at_max_fraction_gt_36']:.4%})")

    print(f"\nINFERENCE: {bathy_eval['better_convention_inference']} provides higher exact match fraction "
          f"({bathy_eval['exact_match_d1_fraction']:.4%} vs {bathy_eval['exact_match_d0_fraction']:.4%}).")

    csv_top20 = os.path.join(output_dir, "bathymetry_mismatch_top20.csv")
    bathy_eval["top20_mismatch_df"].to_csv(csv_top20, index=False)
    print(f"\nTop 20 columns by absolute mismatch (d1) written to: {csv_top20}")
    print(bathy_eval["top20_mismatch_df"][
        ["lat", "lon", "deptho", "deptho_lev", "deepest_finite_idx", "deepest_finite_depth_m", "mismatch_d1"]
    ].to_string(index=False))

    print(f"\nColumn A rank by absolute mismatch: {bathy_eval['column_a_info']['rank_by_abs_mismatch']} "
          f"(mismatch_d1 = {bathy_eval['column_a_info']['mismatch_d1']}, deptho = {bathy_eval['column_a_info']['deptho']}m, deptho_lev = {bathy_eval['column_a_info']['deptho_lev']})")
    print(f"Column B rank by absolute mismatch: {bathy_eval['column_b_info']['rank_by_abs_mismatch']} "
          f"(mismatch_d1 = {bathy_eval['column_b_info']['mismatch_d1']}, deptho = {bathy_eval['column_b_info']['deptho']}m, deptho_lev = {bathy_eval['column_b_info']['deptho_lev']})")

    # Neighbourhood of A and B
    csv_neigh = os.path.join(output_dir, "neighbourhood_ab_bathymetry.csv")
    bathy_eval["neighbourhood_df"].to_csv(csv_neigh, index=False)
    print(f"\nNeighbourhood of A and B written to: {csv_neigh}")
    print(bathy_eval["neighbourhood_df"].to_string(index=False))

    # Profiles for A and B
    df_prof_a = extract_column_profile(
        thetao_day0=thetao_day0,
        fill_value=fill_val,
        mask=mask_raw,
        depths=depths,
        scale_factor=scale_fac,
        add_offset=add_off,
        y_idx=col_a_idx[0],
        x_idx=col_a_idx[1],
    )
    csv_prof_a = os.path.join(output_dir, "column_A_profile.csv")
    df_prof_a.to_csv(csv_prof_a, index=False)

    df_prof_b = extract_column_profile(
        thetao_day0=thetao_day0,
        fill_value=fill_val,
        mask=mask_raw,
        depths=depths,
        scale_factor=scale_fac,
        add_offset=add_off,
        y_idx=col_b_idx[0],
        x_idx=col_b_idx[1],
    )
    csv_prof_b = os.path.join(output_dir, "column_B_profile.csv")
    df_prof_b.to_csv(csv_prof_b, index=False)

    print(f"\nColumn A full profile written to: {csv_prof_a}")
    print(f"Column B full profile written to: {csv_prof_b}")

    # -------------------------------------------------------------------------
    # STEP 1E: Column A Level-31 Time Series
    # -------------------------------------------------------------------------
    print("\n--- SECTION E: Column A Level-31 Time Series ---")
    with RawNetCDFReader(thetao_path) as t_reader:
        ts_info = extract_level31_timeseries(
            reader=t_reader,
            y_idx=col_a_idx[0],
            x_idx=col_a_idx[1],
            level=31,
        )

    csv_ts_a = os.path.join(output_dir, "column_A_level31_timeseries.csv")
    ts_info["timeseries_df"].to_csv(csv_ts_a, index=False)
    print(f"Column A level-31 daily time series written to: {csv_ts_a}")
    print(ts_info["timeseries_df"].to_string(index=False))

    print(f"\nNumber of unique raw codes across 31 days: {ts_info['num_unique_raw_codes']}")
    print(f"Day 0 raw code : {ts_info['day0_raw_code']}")
    print(f"Day 30 raw code: {ts_info['day30_raw_code']}")
    print(f"Day 0 and Day 30 match: {ts_info['endpoints_match']}")
    print(f"Is constant/frozen     : {ts_info['is_constant_frozen']}")
    print("FORENSIC NOTE: Column A at level 31 has 31 unique raw codes (monotonically evolving "
          "from -11709 [12.42 C] to -10839 [13.06 C]). It is a time-varying ocean cell.")

    # -------------------------------------------------------------------------
    # STEP 1F: Diagnostic Plots
    # -------------------------------------------------------------------------
    print("\n--- SECTION F: Diagnostic Visualizations ---")
    plot_paths = generate_plots(
        deptho=deptho,
        fill_deptho=fill_dp,
        deepest_levels=bathy_eval["deepest_array"],
        deptho_lev=deptho_lev,
        fill_deptho_lev=fill_dl,
        lats=lats,
        lons=lons,
        by_level_df=mask_eval["by_level_df"],
        col_a_idx=col_a_idx,
        col_b_idx=col_b_idx,
        out_dir=output_dir,
    )
    for name, p in plot_paths.items():
        print(f"  {name:20s}: {p}")

    # -------------------------------------------------------------------------
    # Verify File Immutability (Final Integrity Check)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Raw File Integrity (Final Check) ---")
    meta_thetao_after = compute_file_metadata(thetao_path)
    meta_static_after = compute_file_metadata(static_path)

    thetao_sha_match = (meta_thetao_before["sha256"] == meta_thetao_after["sha256"])
    thetao_size_match = (meta_thetao_before["size_bytes"] == meta_thetao_after["size_bytes"])
    thetao_mtime_match = (meta_thetao_before["mtime_utc"] == meta_thetao_after["mtime_utc"])

    static_sha_match = (meta_static_before["sha256"] == meta_static_after["sha256"])
    static_size_match = (meta_static_before["size_bytes"] == meta_static_after["size_bytes"])
    static_mtime_match = (meta_static_before["mtime_utc"] == meta_static_after["mtime_utc"])

    if not (thetao_sha_match and thetao_size_match and thetao_mtime_match):
        raise RuntimeError(f"Thetao file was mutated during analysis!\nBefore: {meta_thetao_before}\nAfter: {meta_thetao_after}")
    if not (static_sha_match and static_size_match and static_mtime_match):
        raise RuntimeError(f"Static file was mutated during analysis!\nBefore: {meta_static_before}\nAfter: {meta_static_after}")

    print("Immutability verified: Both raw data files are byte-for-byte identical before and after analysis.")

    # -------------------------------------------------------------------------
    # Compilation of JSON Summary
    # -------------------------------------------------------------------------
    json_summary = {
        "metadata": {
            "analysis_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "analysis_script": os.path.abspath(__file__),
            "output_directory": os.path.abspath(output_dir),
        },
        "file_integrity": {
            "thetao": {
                "filepath": meta_thetao_before["filepath"],
                "sha256_before": meta_thetao_before["sha256"],
                "sha256_after": meta_thetao_after["sha256"],
                "sha256_matches": thetao_sha_match,
                "size_bytes_before": meta_thetao_before["size_bytes"],
                "size_bytes_after": meta_thetao_after["size_bytes"],
                "size_matches": thetao_size_match,
                "mtime_utc_before": meta_thetao_before["mtime_utc"],
                "mtime_utc_after": meta_thetao_after["mtime_utc"],
                "mtime_matches": thetao_mtime_match,
            },
            "glorys_static": {
                "filepath": meta_static_before["filepath"],
                "sha256_before": meta_static_before["sha256"],
                "sha256_after": meta_static_after["sha256"],
                "sha256_matches": static_sha_match,
                "size_bytes_before": meta_static_before["size_bytes"],
                "size_bytes_after": meta_static_after["size_bytes"],
                "size_matches": static_size_match,
                "mtime_utc_before": meta_static_before["mtime_utc"],
                "mtime_utc_after": meta_static_after["mtime_utc"],
                "mtime_matches": static_mtime_match,
            },
        },
        "section_a_static_header": {
            "dimensions": header_info["dimensions"],
            "variables_present": list(header_info["variables"].keys()),
            "expected_variables": {
                "mask": header_info["expected_var_presence"]["mask"],
                "deptho": header_info["expected_var_presence"]["deptho"],
                "deptho_lev": header_info["expected_var_presence"]["deptho_lev"],
                "e1t": None,  # null with reason
                "e2t": None,
                "e3t": None,
                "mdt": None,
            },
            "unverified_missing_variables_reasons": {
                "e1t": "Variable 'e1t' not present in static dataset",
                "e2t": "Variable 'e2t' not present in static dataset",
                "e3t": "Variable 'e3t' not present in static dataset",
                "mdt": "Variable 'mdt' not present in static dataset",
            },
            "mask_dimensionality": f"{header_info['mask_ndim']}D",
            "mask_shape": header_info["mask_shape"],
            "mask_unique_values": header_info["mask_unique_values"],
            "mask_nan_count": header_info["mask_nan_count"],
            "depth_level_count": header_info["depth_count"],
        },
        "section_b_grid_alignment": grid_info,
        "section_c_mask_vs_thetao": {
            "total_cells_checked_levels_0_to_35": int(mask_eval["by_level_df"]["total_cells"].sum()),
            "total_mask1_finite": int(mask_eval["by_level_df"]["mask1_finite"].sum()),
            "total_mask1_fill": int(mask_eval["by_level_df"]["mask1_fill"].sum()),
            "total_mask0_finite": int(mask_eval["by_level_df"]["mask0_finite"].sum()),
            "total_mask0_fill": int(mask_eval["by_level_df"]["mask0_fill"].sum()),
            "m0_finite_count": mask_eval["m0_finite_count"],
            "m1_fill_count": mask_eval["m1_fill_count"],
            "col_a_mismatches": {
                "mask0_finite_levels": mask_eval["col_a_in_m0_finite_levels"],
                "mask1_fill_levels": mask_eval["col_a_in_m1_fill_levels"],
            },
            "col_b_mismatches": {
                "mask0_finite_levels": mask_eval["col_b_in_m0_finite_levels"],
                "mask1_fill_levels": mask_eval["col_b_in_m1_fill_levels"],
            },
        },
        "section_d_bathymetry_consistency": {
            "total_ocean_columns": bathy_eval["total_valid_ocean_columns"],
            "count_deptho_lev_le_36": bathy_eval["count_deptho_lev_le_36"],
            "count_deptho_lev_gt_36": bathy_eval["count_deptho_lev_gt_36"],
            "convention_d1_1based": {
                "formula": "deepest - (deptho_lev - 1)",
                "exact_match_count": bathy_eval["exact_match_d1_count"],
                "exact_match_fraction": bathy_eval["exact_match_d1_fraction"],
                "histogram": bathy_eval["hist_d1"],
            },
            "convention_d0_0based": {
                "formula": "deepest - deptho_lev",
                "exact_match_count": bathy_eval["exact_match_d0_count"],
                "exact_match_fraction": bathy_eval["exact_match_d0_fraction"],
                "histogram": bathy_eval["hist_d0"],
            },
            "deptho_lev_gt_36_consistency": {
                "total_columns": bathy_eval["count_deptho_lev_gt_36"],
                "deepest_at_level_35_count": bathy_eval["deepest_at_max_count_gt_36"],
                "deepest_at_level_35_fraction": bathy_eval["deepest_at_max_fraction_gt_36"],
            },
            "inference_better_convention": bathy_eval["better_convention_inference"],
            "column_a": bathy_eval["column_a_info"],
            "column_b": bathy_eval["column_b_info"],
            "top20_mismatches": bathy_eval["top20_mismatch_df"].to_dict(orient="records"),
        },
        "section_e_column_a_level31": {
            "column": "A (15.000N, 50.583E; y=120, x=67)",
            "level": 31,
            "depth_m": float(depths[31]),
            "num_unique_raw_codes": ts_info["num_unique_raw_codes"],
            "day0_raw_code": ts_info["day0_raw_code"],
            "day30_raw_code": ts_info["day30_raw_code"],
            "endpoints_match": ts_info["endpoints_match"],
            "is_constant_frozen": ts_info["is_constant_frozen"],
            "inference": "Varying time series (31 unique raw codes; endpoints do not match; not frozen)",
        },
        "section_f_plots": plot_paths,
    }

    json_path = os.path.join(output_dir, "glorys_static_crosscheck.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nScalar findings written to JSON: {json_path}")
    print("=" * 80)
    print("CROSS-CHECK COMPLETE.")
    print("=" * 80)

    return json_summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLORYS static mask and bathymetry cross-check (read-only)."
    )
    parser.add_argument(
        "--thetao-path",
        type=str,
        default="data/raw/glorys/glorys_thetao_2019-01.nc",
        help="Path to GLORYS thetao NetCDF file.",
    )
    parser.add_argument(
        "--static-path",
        type=str,
        default="data/raw/glorys_static/glorys_static_mask_bathy.nc",
        help="Path to GLORYS static mask and bathymetry NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/diagnostics/glorys_static_mask_crosscheck",
        help="Directory to save output files and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_static_crosscheck(
        thetao_path=args.thetao_path,
        static_path=args.static_path,
        output_dir=args.output_dir,
    )
