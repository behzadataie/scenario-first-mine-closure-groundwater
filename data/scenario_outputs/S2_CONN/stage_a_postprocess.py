#!/usr/bin/env python3
"""
Post-processing and quality-control plots for the Stage A deterministic benchmark.

This script expects the JSON manifest written by ``stage_a_basecase_flopy.py``.
It generates:
- hydrographs at key observation points,
- pit inflow time series,
- selected package-flow time series,
- plan-view head and drawdown maps,
- a cross-section through the pit-receptor corridor,
- CSV tables containing the extracted time series, stage summaries, and simple diagnostic checks.

The plotting emphasis is on clarity and technical readability rather than on
publication styling.  The outputs are therefore suitable for model checking,
appendices, and manuscript-figure drafting.  Stage A.2b keeps the Stage A.1
masking improvements and adds run-synchronised output folders, package-specific
pit inflow diagnostics that separate PITDRN_FLOOR and PITDRN_WALL, an extra
receptor-layer drawdown map, and clearer vector plotting.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import flopy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FloPy is required for this script. Install it first, for example with 'pip install flopy'."
    ) from exc


CELLBUDGET_GET_DATA_SUPPORTS_PAKNAM = "paknam" in inspect.signature(flopy.utils.CellBudgetFile.get_data).parameters



# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def read_manifest(workspace: Path) -> Dict[str, object]:
    manifest_path = workspace / "stage_a_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_output_dir(workspace: Path, manifest: Dict[str, object]) -> Path:
    run_id = str(manifest.get("run_id", "latest"))
    return workspace / "postprocess" / run_id


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def add_source_run_id(df: pd.DataFrame, run_id: object) -> pd.DataFrame:
    """Return a copy of *df* with a leading ``source_run_id`` column.

    Stage A.2c mirrors selected CSV outputs back to the workspace root. Once that
    happens, the file location alone is no longer enough to identify which run
    generated the table.  This helper stamps each exported table with the run id
    carried by the manifest so copied outputs remain traceable.
    """
    out = df.copy()
    run_id_text = str(run_id)
    if "source_run_id" in out.columns:
        out["source_run_id"] = run_id_text
        cols = ["source_run_id", *[c for c in out.columns if c != "source_run_id"]]
        return out.loc[:, cols]
    out.insert(0, "source_run_id", run_id_text)
    return out


def mirror_latest_outputs(workspace: Path, outputs: Path, filenames: Sequence[str]) -> None:
    """Copy selected run-specific outputs back to the workspace root.

    The run-specific ``postprocess/<run_id>/`` folder remains the authoritative
    archive.  These mirrored files are convenience copies only, intended to make
    quick inspection easier from the workspace root without having to browse into
    the run-specific folder. Missing files are skipped quietly so partial reruns
    do not fail during cleanup.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = outputs / name
        if not src.exists():
            continue
        dst = workspace / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_headfile(path: Path) -> flopy.utils.HeadFile:
    return flopy.utils.HeadFile(str(path), precision="double")


def safe_cbcfile(path: Path) -> flopy.utils.CellBudgetFile:
    return flopy.utils.CellBudgetFile(str(path), precision="double")


def last_head_array(path: Path) -> np.ndarray:
    hdobj = safe_headfile(path)
    last_kstpkper = hdobj.get_kstpkper()[-1]
    return hdobj.get_data(kstpkper=last_kstpkper)


def read_percent_discrepancy(listfile: Path) -> Optional[float]:
    """Extract the last reported percent discrepancy from a MODFLOW 6 list file.

    The manifest normally stores the correct list-file path, but this helper also
    falls back to common MF6 list-file names if that path is missing.
    """
    candidates = [listfile]
    if listfile.parent.exists():
        candidates.extend([listfile.parent / "mfsim.lst", *sorted(listfile.parent.glob("*.lst"))])

    text = None
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            break
    if text is None:
        return None

    matches = re.findall(r"PERCENT DISCREPANCY\s*=\s*([-+0-9.Ee]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def layer_tops_and_bottoms(config: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    botm = np.array(config["botm_elevs_m"], dtype=float)
    lay_tops = np.empty(botm.size, dtype=float)
    lay_tops[0] = float(config["top_elev_m"])
    lay_tops[1:] = botm[:-1]
    thickness = lay_tops - botm
    return lay_tops, botm, thickness


def x_y_centres(config: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    ncol = int(config["ncol"])
    nrow = int(config["nrow"])
    delr = float(config["delr_m"])
    delc = float(config["delc_m"])
    lx = float(config["length_x_m"])
    ly = float(config["length_y_m"])
    x = np.arange(0.5 * delr, lx, delr)
    y = ly - np.arange(0.5 * delc, ly, delc)
    if x.size != ncol or y.size != nrow:
        raise ValueError("Grid dimensions in manifest are inconsistent with delr/delc and domain size.")
    return x, y


def pit_outline(config: Dict[str, object], npts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, npts)
    cx = float(config["pit_center_x_m"])
    cy = float(config["pit_center_y_m"])
    a = 0.5 * float(config["pit_length_m"])
    b = 0.5 * float(config["pit_width_m"])
    return cx + a * np.cos(theta), cy + b * np.sin(theta)


def observation_points(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return manifest["observation_cells"]


def stage_table(manifest: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(manifest["stages"])


def load_model(stage_record: Dict[str, object]) -> object:
    ws = Path(stage_record["workspace"])
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    return sim.get_model(stage_record["model_name"])


def stage_idomain(stage_record: Dict[str, object]) -> np.ndarray:
    """Return the stage-specific idomain array from the saved model input."""
    gwf = load_model(stage_record)
    return np.array(gwf.dis.idomain.array, dtype=int)


def invalid_value_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return (~np.isfinite(arr)) | (np.abs(arr) >= 1.0e20)


def masked_model_array(arr: np.ndarray, idomain: Optional[np.ndarray] = None) -> np.ma.MaskedArray:
    arr = np.asarray(arr, dtype=float)
    mask = invalid_value_mask(arr)
    if idomain is not None:
        mask |= (np.asarray(idomain) <= 0)
    return np.ma.masked_where(mask, arr)


def masked_drawdown_array(
    baseline_head: np.ndarray,
    stressed_head: np.ndarray,
    baseline_idomain: Optional[np.ndarray] = None,
    stressed_idomain: Optional[np.ndarray] = None,
) -> np.ma.MaskedArray:
    baseline = np.asarray(baseline_head, dtype=float)
    stressed = np.asarray(stressed_head, dtype=float)
    mask = invalid_value_mask(baseline) | invalid_value_mask(stressed)
    if baseline_idomain is not None:
        mask |= (np.asarray(baseline_idomain) <= 0)
    if stressed_idomain is not None:
        mask |= (np.asarray(stressed_idomain) <= 0)
    dd = baseline - stressed
    return np.ma.masked_where(mask, dd)


def contour_levels_from_data(data: np.ma.MaskedArray, nlevels: int = 12, drawdown: bool = False) -> Optional[np.ndarray]:
    valid = np.asarray(data.compressed(), dtype=float) if np.ma.isMaskedArray(data) else np.asarray(data, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size < 2:
        return None
    vmin = 0.0 if drawdown else float(valid.min())
    vmax = float(valid.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return None
    return np.linspace(vmin, vmax, nlevels)


def _decode_header_value(value: object) -> str:
    """Normalise a binary-budget header label to uppercase stripped text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8', errors='ignore')
        except Exception:
            value = str(value)
    value = str(value)
    return value.replace("\x00", "").strip().upper()


def budget_header_table(cbc: flopy.utils.CellBudgetFile) -> Optional[pd.DataFrame]:
    """Return a normalised header table with record indices when available.

    FloPy versions differ in whether the metadata are exposed through ``headers``
    (newer versions) or ``recordarray`` (older versions).  This helper handles
    both so package-specific budget records can be matched by text and package
    name without depending on ``paknam`` filtering support alone.
    """
    source = None
    if hasattr(cbc, 'headers') and cbc.headers is not None:
        source = cbc.headers
    elif hasattr(cbc, 'recordarray') and cbc.recordarray is not None:
        source = cbc.recordarray
    if source is None:
        return None

    if isinstance(source, pd.DataFrame):
        headers = source.copy()
    else:
        try:
            headers = pd.DataFrame(source)
        except Exception:
            return None

    if headers.empty:
        return None

    headers = headers.reset_index(drop=True)
    headers['idx'] = np.arange(len(headers), dtype=int)
    for col in ['text', 'paknam', 'paknam2']:
        if col not in headers.columns:
            headers[col] = ''
        headers[col] = headers[col].map(_decode_header_value)
    if 'totim' not in headers.columns:
        headers['totim'] = np.nan
    headers['totim'] = pd.to_numeric(headers['totim'], errors='coerce')
    return headers


def budget_record_indices(
    cbc: flopy.utils.CellBudgetFile,
    totim: float,
    text: str,
    package_name: Optional[str] = None,
) -> List[int]:
    """Return matching budget-record indices for one time, text, and package.

    For MODFLOW 6 list-style boundary packages, the budget file commonly stores
    one record per package instance for a given text and time.  Using the header
    table avoids ambiguity about whether a helper such as ``paknam`` or
    ``paknam2`` is interpreted as the from-package or the named package.
    """
    headers = budget_header_table(cbc)
    if headers is None or headers.empty:
        return []

    text_norm = _decode_header_value(text)
    rows = headers[np.isclose(headers['totim'].to_numpy(dtype=float), float(totim), atol=0.0, rtol=1.0e-12)]
    rows = rows[rows['text'] == text_norm]
    if package_name is not None:
        pkg_norm = _decode_header_value(package_name)
        rows = rows[(rows['paknam'] == pkg_norm) | (rows['paknam2'] == pkg_norm)]
    return rows['idx'].astype(int).tolist()


def budget_sum_for_indices(cbc: flopy.utils.CellBudgetFile, indices: Sequence[int]) -> float:
    """Sum flow across an explicit list of budget record indices."""
    total = 0.0
    for idx in indices:
        try:
            data = cbc.get_data(idx=int(idx))
        except TypeError:
            try:
                data = [cbc.get_record(int(idx))]
            except Exception:
                data = None
        except Exception:
            data = None
        if data is None:
            continue
        if not isinstance(data, list):
            data = [data]
        for rec in data:
            if rec is None:
                continue
            if hasattr(rec, 'dtype') and rec.dtype.names and 'q' in rec.dtype.names:
                total += float(np.nansum(rec['q']))
            else:
                arr = np.asarray(rec)
                total += float(np.nansum(arr))
    return total


def budget_sum_for_named_package(
    cbc: flopy.utils.CellBudgetFile,
    text: str,
    totim: float,
    package_name: str,
) -> float:
    """Return the summed flux for one named package instance.

    Strategy order:
    1. use explicit header-index matching when available;
    2. fall back to ``paknam2`` filtering, which FloPy documents for named
       MODFLOW 6 packages;
    3. finally try ``paknam`` for older or non-standard cases.
    """
    indices = budget_record_indices(cbc, totim=totim, text=text, package_name=package_name)
    if indices:
        return budget_sum_for_indices(cbc, indices)

    if CELLBUDGET_GET_DATA_SUPPORTS_PAKNAM:
        for kwargs in ({'paknam2': package_name}, {'paknam': package_name}):
            try:
                data = cbc.get_data(text=text, totim=totim, **kwargs)
            except TypeError:
                data = None
            except Exception:
                data = None
            if data:
                total = 0.0
                for rec in data:
                    if rec is None:
                        continue
                    if hasattr(rec, 'dtype') and rec.dtype.names and 'q' in rec.dtype.names:
                        total += float(np.nansum(rec['q']))
                    else:
                        total += float(np.nansum(np.asarray(rec)))
                return total

    return float('nan')


# --------------------------------------------------------------------------------------
# Time-series extraction
# --------------------------------------------------------------------------------------


def extract_head_timeseries(manifest: Dict[str, object]) -> pd.DataFrame:
    """Combine stage-wise observation-point hydrographs into one continuous table."""
    obs = observation_points(manifest)
    rows: List[Dict[str, float]] = []

    for stage in manifest["stages"]:
        start_year = float(stage["start_year"])
        hdobj = safe_headfile(Path(stage["head_file"]))
        times_days = np.array(hdobj.get_times(), dtype=float)

        for obs_name, meta in obs.items():
            cellid = tuple(meta["cellid"])
            ts = hdobj.get_ts(cellid)
            # HeadFile.get_ts returns two columns: time and head.
            for t_day, head in ts:
                rows.append(
                    {
                        "run_id": manifest.get("run_id", "latest"),
                        "stage": stage["name"],
                        "obs_name": obs_name,
                        "time_days_stage": float(t_day),
                        "time_years_total": start_year + float(t_day) / 365.25,
                        "head_m": float(head),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["obs_name", "time_years_total"]).reset_index(drop=True)
    return df


def budget_sum_for_time(
    cbc: flopy.utils.CellBudgetFile,
    text: str,
    totim: float,
    paknam: Optional[str] = None,
    paknam2: Optional[str] = None,
) -> float:
    """Return the summed flux for one budget text and one simulation time.

    Stage A.2b note
    ---------------
    Aggregate extraction first uses the budget-header table when available so
    multi-package MODFLOW 6 budget records remain synchronised with the package-
    specific extraction logic.  If no header table is available, the function
    falls back to FloPy's ``get_data`` call.  Named-package extraction is handled
    separately by :func:`budget_sum_for_named_package`.
    """
    if paknam is not None or paknam2 is not None:
        package_name = paknam2 if paknam2 is not None else paknam
        return budget_sum_for_named_package(cbc, text=text, totim=totim, package_name=str(package_name))

    indices = budget_record_indices(cbc, totim=totim, text=text, package_name=None)
    if indices:
        return budget_sum_for_indices(cbc, indices)

    try:
        data = cbc.get_data(text=text, totim=totim)
    except Exception:
        return 0.0

    if data is None or len(data) == 0:
        return 0.0

    total = 0.0
    for rec in data:
        if rec is None:
            continue
        if hasattr(rec, 'dtype') and rec.dtype.names and 'q' in rec.dtype.names:
            total += float(np.nansum(rec['q']))
        else:
            total += float(np.nansum(np.asarray(rec)))
    return total



def extract_budget_timeseries(manifest: Dict[str, object]) -> pd.DataFrame:
    """Extract selected package flows from each stage budget file.

    Stage A.2b keeps the aggregate DRN / WEL series but also exports the named
    MODFLOW 6 package components directly in the CSV so ``PITDRN_FLOOR`` and
    ``PITDRN_WALL`` can be checked explicitly.  The canonical mine-inflow series
    is taken from the sum of components when that split is available and falls
    back to the aggregate DRN + WEL total otherwise.
    """
    rows: List[Dict[str, float]] = []
    aggregate_aliases = {
        'drn_total_raw_m3_per_d': ['DRN', 'PITDRN'],
        'wel_total_raw_m3_per_d': ['WEL', 'PITWEL'],
        'RIV': ['RIV'],
        'GHB': ['GHB'],
        'RCH': ['RCH', 'RCHA'],
    }

    for stage in manifest['stages']:
        start_year = float(stage['start_year'])
        cbc = safe_cbcfile(Path(stage['budget_file']))
        times = np.array(cbc.get_times(), dtype=float)

        floor_pkg = stage.get('pit_floor_package_name')
        wall_pkg = stage.get('pit_wall_package_name')
        well_pkg = stage.get('pit_well_package_name')

        for totim in times:
            row = {
                'run_id': manifest.get('run_id', 'latest'),
                'stage': stage['name'],
                'time_days_stage': float(totim),
                'time_years_total': start_year + float(totim) / 365.25,
            }
            for label, aliases in aggregate_aliases.items():
                value = 0.0
                for alias in aliases:
                    try:
                        value = budget_sum_for_time(cbc, alias, totim)
                    except Exception:
                        value = 0.0
                    if not np.isclose(value, 0.0):
                        break
                row[label] = value

            row['PITDRN_FLOOR_raw_m3_per_d'] = budget_sum_for_named_package(cbc, 'DRN', totim, floor_pkg) if floor_pkg else 0.0
            row['PITDRN_WALL_raw_m3_per_d'] = budget_sum_for_named_package(cbc, 'DRN', totim, wall_pkg) if wall_pkg else 0.0
            row['PITWEL_raw_m3_per_d'] = budget_sum_for_named_package(cbc, 'WEL', totim, well_pkg) if well_pkg else 0.0
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(['time_years_total', 'stage']).reset_index(drop=True)

    # Positive mine-inflow magnitudes from named package components.
    df['PITDRN_FLOOR_m3_per_d'] = -df['PITDRN_FLOOR_raw_m3_per_d']
    df['PITDRN_WALL_m3_per_d'] = -df['PITDRN_WALL_raw_m3_per_d']
    df['PITWEL_m3_per_d'] = -df['PITWEL_raw_m3_per_d']

    # Backward-compatible aliases used in earlier Stage A scripts and notes.
    df['mine_inflow_floor_m3_per_d'] = df['PITDRN_FLOOR_m3_per_d']
    df['mine_inflow_wall_m3_per_d'] = df['PITDRN_WALL_m3_per_d']
    df['mine_inflow_well_m3_per_d'] = df['PITWEL_m3_per_d']

    component_cols = ['PITDRN_FLOOR_m3_per_d', 'PITDRN_WALL_m3_per_d', 'PITWEL_m3_per_d']
    df['mine_inflow_components_sum_m3_per_d'] = df[component_cols].sum(axis=1, skipna=True)
    df['mine_inflow_total_from_aggregate_m3_per_d'] = -(df.get('drn_total_raw_m3_per_d', 0.0) + df.get('wel_total_raw_m3_per_d', 0.0))
    df['component_split_available'] = ~df[['PITDRN_FLOOR_raw_m3_per_d', 'PITDRN_WALL_raw_m3_per_d', 'PITWEL_raw_m3_per_d']].isna().any(axis=1)
    df['mine_inflow_total_m3_per_d'] = np.where(
        df['component_split_available'],
        df['mine_inflow_components_sum_m3_per_d'],
        df['mine_inflow_total_from_aggregate_m3_per_d'],
    )
    df['mine_inflow_component_mismatch_m3_per_d'] = np.where(
        df['component_split_available'],
        df['mine_inflow_components_sum_m3_per_d'] - df['mine_inflow_total_from_aggregate_m3_per_d'],
        np.nan,
    )

    # Backward-compatible alias used elsewhere in earlier Stage A scripts.
    df['mine_inflow_m3_per_d'] = df['mine_inflow_total_m3_per_d']
    return df


# --------------------------------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------------------------------


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )



def annotate_observation_points(ax: plt.Axes, manifest: Dict[str, object]) -> None:
    for name, meta in observation_points(manifest).items():
        x = float(meta["x_m"])
        y = float(meta["y_m"])
        ax.scatter(x, y, s=28, edgecolor="black", linewidth=0.5, zorder=5)
        ax.text(x + 90.0, y + 60.0, name, fontsize=8, va="bottom", ha="left")



def plot_plan_map(
    manifest: Dict[str, object],
    stage_record: Dict[str, object],
    head: np.ndarray,
    idomain: np.ndarray,
    outfile: Path,
    title: str,
    plot_layer: int,
    drawdown_from: Optional[np.ndarray] = None,
    drawdown_idomain: Optional[np.ndarray] = None,
) -> None:
    """Plot a plan-view head or drawdown map for one stage.

    Stage A.1 note
    ---------------
    MODFLOW 6 writes very large no-data values for inactive or dry cells.  If
    these are not masked explicitly, the resulting colour scales and contours can
    become meaningless.  This helper therefore masks both invalid values and
    inactive cells before any plotting is attempted.
    """
    config = manifest["config"]
    x, y = x_y_centres(config)
    xx, yy = np.meshgrid(x, y)

    if drawdown_from is None:
        arr3d = masked_model_array(head, idomain=idomain)
        data = arr3d[plot_layer, :, :]
        cbar_label = "Head (m)"
        levels = contour_levels_from_data(data, drawdown=False)
    else:
        arr3d = masked_drawdown_array(drawdown_from, head, baseline_idomain=drawdown_idomain, stressed_idomain=idomain)
        data = arr3d[plot_layer, :, :]
        cbar_label = "Drawdown (m)"
        levels = contour_levels_from_data(data, drawdown=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    mesh = ax.pcolormesh(xx, yy, data, shading="nearest")
    if levels is not None:
        cs = ax.contour(xx, yy, data, levels=levels, colors="black", linewidths=0.6, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", inline=True, fontsize=7)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.88)
    cbar.set_label(cbar_label)

    px, py = pit_outline(config)
    ax.plot(px, py, linewidth=1.4)
    ax.plot(
        [float(config["river_xmin_m"]), float(config["river_xmax_m"])],
        [float(config["river_center_y_m"]), float(config["river_center_y_m"])],
        linewidth=3.0,
        alpha=0.8,
    )

    annotate_observation_points(ax, manifest)

    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(config["length_x_m"]))
    ax.set_ylim(0.0, float(config["length_y_m"]))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.text(0.01, 0.01, f"Stage: {stage_record['name']} | Layer {plot_layer + 1}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_flow_vectors(
    manifest: Dict[str, object],
    stage_record: Dict[str, object],
    idomain: np.ndarray,
    outfile: Path,
    plot_layer: int,
    title: str,
    stride: int = 6,
) -> None:
    """Plot plan-view heads with specific-discharge vectors.

    Stage A.2 uses stronger thinning and clipped arrow lengths so the vector plot
    remains diagnostic rather than visually saturated near the pit.
    """
    config = manifest["config"]
    gwf = load_model(stage_record)
    hdobj = safe_headfile(Path(stage_record["head_file"]))
    head = hdobj.get_data(kstpkper=hdobj.get_kstpkper()[-1])
    head_ma = masked_model_array(head, idomain=idomain)
    cbc = safe_cbcfile(Path(stage_record["budget_file"]))
    spdis = cbc.get_data(text="DATA-SPDIS")[-1]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    x, y = x_y_centres(config)
    xx, yy = np.meshgrid(x, y)
    data = head_ma[plot_layer, :, :]
    levels = contour_levels_from_data(data, drawdown=False)

    qmask = np.ma.getmaskarray(data) | invalid_value_mask(qx[plot_layer, :, :]) | invalid_value_mask(qy[plot_layer, :, :])
    qx_layer = np.ma.masked_where(qmask, qx[plot_layer, :, :])
    qy_layer = np.ma.masked_where(qmask, qy[plot_layer, :, :])

    xs = xx[::stride, ::stride]
    ys = yy[::stride, ::stride]
    qx_s = qx_layer[::stride, ::stride]
    qy_s = qy_layer[::stride, ::stride]
    qmag = np.sqrt(np.square(qx_s) + np.square(qy_s))
    qvalid = np.asarray(qmag.compressed(), dtype=float)
    q95 = float(np.nanpercentile(qvalid, 95.0)) if qvalid.size > 0 else 1.0
    q95 = max(q95, 1.0e-12)
    rel = np.clip(np.asarray(qmag.filled(0.0), dtype=float) / q95, 0.0, 1.0)
    keep = (~np.ma.getmaskarray(qx_s)) & (~np.ma.getmaskarray(qy_s)) & (rel >= 0.05)
    qmag_safe = np.where(np.asarray(qmag.filled(0.0), dtype=float) > 1.0e-12, np.asarray(qmag.filled(0.0), dtype=float), 1.0)
    dirx = np.asarray(qx_s.filled(0.0), dtype=float) / qmag_safe
    diry = np.asarray(qy_s.filled(0.0), dtype=float) / qmag_safe
    max_arrow_len_m = 0.75 * stride * min(float(config["delr_m"]), float(config["delc_m"]))
    u = np.where(keep, dirx * rel * max_arrow_len_m, np.nan)
    v = np.where(keep, diry * rel * max_arrow_len_m, np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    mesh = ax.pcolormesh(xx, yy, data, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.88)
    cbar.set_label("Head (m)")
    if levels is not None:
        cs = ax.contour(xx, yy, data, levels=levels, colors="black", linewidths=0.6, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", inline=True, fontsize=7)
    ax.quiver(
        xs,
        ys,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        width=0.0018,
        alpha=0.85,
    )

    px, py = pit_outline(config)
    ax.plot(px, py, linewidth=1.4)
    ax.plot(
        [float(config["river_xmin_m"]), float(config["river_xmax_m"])],
        [float(config["river_center_y_m"]), float(config["river_center_y_m"])],
        linewidth=3.0,
        alpha=0.8,
    )
    annotate_observation_points(ax, manifest)

    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(config["length_x_m"]))
    ax.set_ylim(0.0, float(config["length_y_m"]))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



def get_config_value(config: Dict[str, object], *keys: str, default: Optional[object] = None) -> object:
    """Return the first matching config value from a list of possible keys.

    This keeps post-processing compatible with small key-name differences
    between Stage A script revisions.
    """
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    if default is not None:
        return default
    raise KeyError(keys[0] if keys else 'config key not found')

def plot_cross_section(
    manifest: Dict[str, object],
    stage_record: Dict[str, object],
    idomain: np.ndarray,
    outfile: Path,
    title: str,
    drawdown_from: Optional[np.ndarray] = None,
    drawdown_idomain: Optional[np.ndarray] = None,
) -> None:
    """Cross-section through the pit and downgradient receptor corridor.

    Stage A.2c overlays a simple design pit envelope so the intended excavation
    geometry can be compared visually with the stepped model-cell representation.
    """
    config = manifest["config"]
    gwf = load_model(stage_record)
    hdobj = safe_headfile(Path(stage_record["head_file"]))
    head = hdobj.get_data(kstpkper=hdobj.get_kstpkper()[-1])

    row = int(observation_points(manifest)["compliance"]["cellid"][1])
    if drawdown_from is None:
        arr = masked_model_array(head, idomain=idomain)
        label = "Head (m)"
        levels = contour_levels_from_data(arr[:, row, :], drawdown=False)
    else:
        arr = masked_drawdown_array(drawdown_from, head, baseline_idomain=drawdown_idomain, stressed_idomain=idomain)
        label = "Drawdown (m)"
        levels = contour_levels_from_data(arr[:, row, :], drawdown=True)

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    xsect = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"row": row})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Warning: converting a masked element to nan.",
            category=UserWarning,
        )
        pc = xsect.plot_array(arr, masked_values=[1.0e30, -1.0e30])
        xsect.plot_grid(linewidth=0.15, alpha=0.3)
        if levels is not None:
            try:
                cset = xsect.contour_array(arr, levels=levels, colors="black", linewidths=0.45)
                ax.clabel(cset, fmt="%.1f", inline=True, fontsize=7)
            except Exception:
                pass

    cbar = fig.colorbar(pc, ax=ax, shrink=0.88)
    cbar.set_label(label)

    # Overlay the design pit envelope for the current stage.  The underlying
    # model remains cell based, but the overlay makes the intended geometry
    # explicit and easier to compare with the stepped active-cell cavity.
    floor = stage_record.get("pit_floor_elev_m")
    if floor is not None:
        xc = float(get_config_value(config, "pit_center_x_m", "pit_center_x"))
        pit_length = float(get_config_value(config, "pit_length_m", "pit_length_x_m", "pit_length"))
        half_top = 0.5 * pit_length
        slope = max(float(config.get("pit_wall_slope_h_to_v", 0.35)), 1.0e-6)
        top_elev = float(get_config_value(config, "top_elev_m", "top_elev"))
        depth = max(top_elev - float(floor), 0.0)
        inset = min(half_top * 0.75, slope * depth)
        left_top = xc - half_top
        right_top = xc + half_top
        left_floor = xc - max(half_top - inset, 0.15 * half_top)
        right_floor = xc + max(half_top - inset, 0.15 * half_top)
        ax.plot([left_top, left_floor], [top_elev, float(floor)], linestyle='--', linewidth=1.2, color='white', alpha=0.95)
        ax.plot([left_floor, right_floor], [float(floor), float(floor)], linestyle='--', linewidth=1.2, color='white', alpha=0.95)
        ax.plot([right_floor, right_top], [float(floor), top_elev], linestyle='--', linewidth=1.2, color='white', alpha=0.95)
        ax.text(xc, float(floor) + 2.0, 'design pit envelope', color='white', fontsize=7, ha='center', va='bottom')

    for obs_name, meta in observation_points(manifest).items():
        ax.axvline(float(meta["x_m"]), linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(float(meta["x_m"]) + 35.0, float(config["top_elev_m"]) - 3.0, obs_name, rotation=90, va="top")

    ax.set_xlabel("x along section (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs(df_heads: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    for obs_name, group in df_heads.groupby("obs_name"):
        ax.plot(group["time_years_total"], group["head_m"], label=obs_name, linewidth=1.8)
    ax.axvspan(0.0, 20.0, alpha=0.12)
    for x in [5.0, 10.0, 15.0, 20.0]:
        ax.axvline(x, color="0.45", linewidth=0.7, linestyle="--", alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(10.0, ymin, "Operations", ha="center", va="bottom")
    ax.text(70.0, ymin, "Recovery", ha="center", va="bottom")
    ax.set_xlabel("Time since start of operations (years)")
    ax.set_ylabel("Head (m)")
    ax.set_title("Stage A hydrographs at key observation points")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



def plot_pit_inflow(df_budget: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.plot(df_budget['time_years_total'], df_budget['mine_inflow_total_m3_per_d'], linewidth=2.1, label='total')
    component_map = {
        'PITDRN_FLOOR_m3_per_d': 'PITDRN_FLOOR',
        'PITDRN_WALL_m3_per_d': 'PITDRN_WALL',
        'PITWEL_m3_per_d': 'PITWEL',
    }
    for col, label in component_map.items():
        if col in df_budget.columns:
            series = pd.to_numeric(df_budget[col], errors='coerce')
            values = series.to_numpy(dtype=float)
            if np.isfinite(values).any() and np.nanmax(np.abs(values)) > 0.0:
                ax.plot(df_budget['time_years_total'], series, linewidth=1.2, linestyle='--', label=label)
    ax.axvspan(0.0, 20.0, alpha=0.12)
    for x in [5.0, 10.0, 15.0, 20.0]:
        ax.axvline(x, color='0.45', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.set_xlabel('Time since start of operations (years)')
    ax.set_ylabel('Mine inflow magnitude (m$^3$/d)')
    ax.set_title('Stage A pit inflow by component')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)



def plot_package_flows(df_budget: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for col in ["RIV", "GHB", "RCH"]:
        if col in df_budget.columns:
            ax.plot(df_budget["time_years_total"], df_budget[col], label=col, linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvspan(0.0, 20.0, alpha=0.12)
    for x in [5.0, 10.0, 15.0, 20.0]:
        ax.axvline(x, color="0.45", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time since start of operations (years)")
    ax.set_ylabel("Package flux (m$^3$/d; MODFLOW sign convention)")
    ax.set_title("Selected package fluxes through time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Quality checks and summary tables
# --------------------------------------------------------------------------------------


def build_stage_summary(manifest: Dict[str, object], df_heads: pd.DataFrame, df_budget: pd.DataFrame) -> pd.DataFrame:
    """Return a compact stage-wise summary table for checking transitions."""
    stages = stage_table(manifest)[["name", "kind", "duration_years", "start_year", "end_year"]].rename(columns={"name": "stage"})

    head_end = (
        df_heads.sort_values(["stage", "obs_name", "time_days_stage"])
        .groupby(["stage", "obs_name"], as_index=False)
        .tail(1)
        .pivot(index="stage", columns="obs_name", values="head_m")
        .rename(columns=lambda c: f"end_head_{c}_m")
        .reset_index()
    )

    budget_aggs = {
        "mine_inflow_total_m3_per_d": ["mean", "min", "max"],
        "PITDRN_FLOOR_m3_per_d": ["mean"],
        "PITDRN_WALL_m3_per_d": ["mean"],
        "PITWEL_m3_per_d": ["mean"],
        "mine_inflow_floor_m3_per_d": ["mean"],
        "mine_inflow_wall_m3_per_d": ["mean"],
        "mine_inflow_well_m3_per_d": ["mean"],
    }
    for col in ["RIV", "GHB", "RCH", "mine_inflow_component_mismatch_m3_per_d"]:
        if col in df_budget.columns:
            budget_aggs[col] = ["mean"]
    budget_summary = df_budget.groupby("stage").agg(budget_aggs)
    budget_summary.columns = ["_".join([a for a in tup if a]).strip("_") for tup in budget_summary.columns.to_flat_index()]
    budget_summary = budget_summary.reset_index()

    summary = stages.merge(head_end, on="stage", how="left").merge(budget_summary, on="stage", how="left")
    return add_source_run_id(summary, manifest.get("run_id", "latest"))


def compute_checks(manifest: Dict[str, object], df_heads: pd.DataFrame, df_budget: pd.DataFrame) -> pd.DataFrame:
    """Assemble a compact QC table for the deterministic Stage A build."""
    obs = observation_points(manifest)
    config = manifest["config"]

    # Predevelopment and recovery heads at the compliance bore.
    compliance = df_heads[df_heads["obs_name"] == "compliance"].sort_values("time_years_total")
    receptor = df_heads[df_heads["obs_name"] == "receptor"].sort_values("time_years_total")

    predev_head = compliance.iloc[0]["head_m"]
    final_head = compliance.iloc[-1]["head_m"]
    recovery_gap = predev_head - final_head

    max_inflow = df_budget["mine_inflow_total_m3_per_d"].max() if "mine_inflow_total_m3_per_d" in df_budget else np.nan
    max_receptor_drawdown = receptor.iloc[0]["head_m"] - receptor["head_m"].min()

    stage_mean_inflow = df_budget.groupby("stage")["mine_inflow_total_m3_per_d"].mean() if "mine_inflow_total_m3_per_d" in df_budget else pd.Series(dtype=float)
    late_ops_ratio = np.nan
    if "ops_stage_3" in stage_mean_inflow.index and "ops_stage_4" in stage_mean_inflow.index and not np.isclose(stage_mean_inflow["ops_stage_3"], 0.0):
        late_ops_ratio = float(stage_mean_inflow["ops_stage_4"] / stage_mean_inflow["ops_stage_3"])

    component_mismatch_abs = np.nan
    if "mine_inflow_component_mismatch_m3_per_d" in df_budget.columns:
        mismatch = pd.to_numeric(df_budget["mine_inflow_component_mismatch_m3_per_d"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(mismatch).any():
            component_mismatch_abs = float(np.nanmax(np.abs(mismatch)))

    checks = []
    for stage in manifest["stages"]:
        pdsc = read_percent_discrepancy(Path(stage["list_file"]))
        checks.append(
            {
                "check": f"percent_discrepancy_{stage['name']}",
                "value": pdsc,
                "units": "%",
                "target": "Preferably |value| < 1%",
                "comment": "MODFLOW 6 list-file mass-balance discrepancy for the stage.",
            }
        )

    checks.extend(
        [
            {
                "check": "max_mine_inflow",
                "value": float(max_inflow),
                "units": "m3/d",
                "target": "> 0 during operations",
                "comment": "Peak Stage A inflow from drains and optional wells.",
            },
            {
                "check": "max_receptor_drawdown",
                "value": float(max_receptor_drawdown),
                "units": "m",
                "target": "Benchmark-specific; compare later against 2 m manuscript threshold",
                "comment": "Maximum simulated drawdown at the receptor observation cell.",
            },
            {
                "check": "max_component_split_mismatch_abs",
                "value": float(component_mismatch_abs) if np.isfinite(component_mismatch_abs) else np.nan,
                "units": "m3/d",
                "target": "Near zero when component-wise pit budget extraction is working correctly",
                "comment": "Maximum absolute difference between total pit inflow from components and total pit inflow from aggregate DRN + WEL series.",
            },
            {
                "check": "compliance_recovery_gap_after_100_years",
                "value": float(recovery_gap),
                "units": "m",
                "target": "Near zero for strong recovery support",
                "comment": "Remaining head difference from the predevelopment compliance head.",
            },
            {
                "check": "late_ops_mean_inflow_ratio_stage4_to_stage3",
                "value": float(late_ops_ratio),
                "units": "-",
                "target": "Context-specific; very small values can indicate an overly floor-dominated pit representation",
                "comment": "Mean operational inflow in years 16-20 divided by mean inflow in years 11-15.",
            },
            {
                "check": "regional_gradient_implied",
                "value": (float(config["west_boundary_head_m"]) - float(config["east_boundary_head_m"])) / float(config["length_x_m"]),
                "units": "m/m",
                "target": "Approximately 1e-3",
                "comment": "Imposed regional hydraulic gradient across the domain.",
            },
        ]
    )

    return pd.DataFrame(checks)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process the Stage A deterministic benchmark outputs.")
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Root workspace directory created by stage_a_basecase_flopy.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. By default Stage A.2 writes to <workspace>/postprocess/<run_id>.",
    )
    parser.add_argument(
        "--keep-existing-output-dir",
        action="store_true",
        help="Keep an existing output directory instead of deleting and recreating it.",
    )
    return parser



def main() -> None:
    apply_plot_style()
    parser = build_cli()
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    manifest = read_manifest(workspace)
    outputs = args.output_dir.resolve() if args.output_dir is not None else default_output_dir(workspace, manifest)
    if outputs.exists() and (not args.keep_existing_output_dir):
        shutil.rmtree(outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    write_json(outputs / "stage_a_manifest_used.json", manifest)

    df_heads = add_source_run_id(extract_head_timeseries(manifest), manifest.get("run_id", "latest"))
    df_budget = add_source_run_id(extract_budget_timeseries(manifest), manifest.get("run_id", "latest"))
    df_heads.to_csv(outputs / "stage_a_heads_timeseries.csv", index=False)
    df_budget.to_csv(outputs / "stage_a_budget_timeseries.csv", index=False)
    stage_summary = build_stage_summary(manifest, df_heads, df_budget)
    stage_summary.to_csv(outputs / "stage_a_stage_summary.csv", index=False)

    plot_hydrographs(df_heads, outputs / "hydrographs_stage_a.png")
    plot_pit_inflow(df_budget, outputs / "pit_inflow_stage_a.png")
    plot_package_flows(df_budget, outputs / "package_flows_stage_a.png")

    # Representative plan-view and cross-section figures.
    stage_lookup = {stage["name"]: stage for stage in manifest["stages"]}
    plot_layer = int(manifest["config"]["plan_plot_layer_zero_based"])

    pre_head = last_head_array(Path(stage_lookup["predevelopment"]["head_file"]))
    final_ops_head = last_head_array(Path(stage_lookup["ops_stage_4"]["head_file"]))
    recovery_head = last_head_array(Path(stage_lookup["recovery"]["head_file"]))
    pre_idomain = stage_idomain(stage_lookup["predevelopment"])
    ops_idomain = stage_idomain(stage_lookup["ops_stage_4"])
    recovery_idomain = stage_idomain(stage_lookup["recovery"])

    plot_plan_map(
        manifest,
        stage_lookup["predevelopment"],
        pre_head,
        pre_idomain,
        outputs / "plan_heads_predevelopment.png",
        "Predevelopment heads: Stage A deterministic base case",
        plot_layer,
    )
    plot_plan_map(
        manifest,
        stage_lookup["ops_stage_4"],
        final_ops_head,
        ops_idomain,
        outputs / "plan_drawdown_end_of_operations.png",
        "Drawdown at end of operations (year 20)",
        plot_layer,
        drawdown_from=pre_head,
        drawdown_idomain=pre_idomain,
    )
    receptor_layer = int(observation_points(manifest)["receptor"]["cellid"][0])
    plot_plan_map(
        manifest,
        stage_lookup["ops_stage_4"],
        final_ops_head,
        ops_idomain,
        outputs / "plan_drawdown_end_of_operations_receptor_layer.png",
        f"Drawdown at end of operations (year 20): receptor layer (Layer {receptor_layer + 1})",
        receptor_layer,
        drawdown_from=pre_head,
        drawdown_idomain=pre_idomain,
    )
    plot_plan_map(
        manifest,
        stage_lookup["recovery"],
        recovery_head,
        recovery_idomain,
        outputs / "plan_heads_end_of_recovery.png",
        "Heads at end of 100-year recovery",
        plot_layer,
    )
    plot_flow_vectors(
        manifest,
        stage_lookup["ops_stage_4"],
        ops_idomain,
        outputs / "plan_heads_vectors_end_of_operations.png",
        plot_layer,
        "Heads and specific-discharge vectors at end of operations",
    )
    plot_cross_section(
        manifest,
        stage_lookup["ops_stage_4"],
        ops_idomain,
        outputs / "cross_section_end_of_operations.png",
        "Cross-section through pit and receptor corridor at end of operations",
        drawdown_from=pre_head,
        drawdown_idomain=pre_idomain,
    )
    plot_cross_section(
        manifest,
        stage_lookup["recovery"],
        recovery_idomain,
        outputs / "cross_section_end_of_recovery.png",
        "Cross-section through pit and receptor corridor at end of recovery",
        drawdown_from=None,
    )

    checks = add_source_run_id(compute_checks(manifest, df_heads, df_budget), manifest.get("run_id", "latest"))
    checks.to_csv(outputs / "stage_a_checks.csv", index=False)

    output_manifest = {
        "run_id": manifest.get("run_id", "latest"),
        "workspace": str(workspace),
        "outputs": str(outputs),
        "files": {
            "heads_csv": str((outputs / "stage_a_heads_timeseries.csv").resolve()),
            "budget_csv": str((outputs / "stage_a_budget_timeseries.csv").resolve()),
            "stage_summary_csv": str((outputs / "stage_a_stage_summary.csv").resolve()),
            "checks_csv": str((outputs / "stage_a_checks.csv").resolve()),
            "hydrographs_png": str((outputs / "hydrographs_stage_a.png").resolve()),
            "pit_inflow_png": str((outputs / "pit_inflow_stage_a.png").resolve()),
            "package_flows_png": str((outputs / "package_flows_stage_a.png").resolve()),
            "plan_predevelopment_png": str((outputs / "plan_heads_predevelopment.png").resolve()),
            "plan_drawdown_ops_png": str((outputs / "plan_drawdown_end_of_operations.png").resolve()),
            "plan_drawdown_ops_receptor_layer_png": str((outputs / "plan_drawdown_end_of_operations_receptor_layer.png").resolve()),
            "plan_recovery_png": str((outputs / "plan_heads_end_of_recovery.png").resolve()),
            "flow_vectors_png": str((outputs / "plan_heads_vectors_end_of_operations.png").resolve()),
            "xsec_ops_png": str((outputs / "cross_section_end_of_operations.png").resolve()),
            "xsec_recovery_png": str((outputs / "cross_section_end_of_recovery.png").resolve()),
        },
    }
    write_json(outputs / "stage_a_postprocess_manifest.json", output_manifest)
    (workspace / "stage_a_latest_postprocess_dir.txt").write_text(str(outputs.resolve()) + "\n", encoding="utf-8")

    mirror_latest_outputs(
        workspace,
        outputs,
        [
            "stage_a_manifest_used.json",
            "stage_a_postprocess_manifest.json",
            "stage_a_heads_timeseries.csv",
            "stage_a_budget_timeseries.csv",
            "stage_a_stage_summary.csv",
            "stage_a_checks.csv",
            "hydrographs_stage_a.png",
            "pit_inflow_stage_a.png",
            "package_flows_stage_a.png",
            "plan_heads_predevelopment.png",
            "plan_drawdown_end_of_operations.png",
            "plan_drawdown_end_of_operations_receptor_layer.png",
            "plan_heads_end_of_recovery.png",
            "plan_heads_vectors_end_of_operations.png",
            "cross_section_end_of_operations.png",
            "cross_section_end_of_recovery.png",
        ],
    )

    print(f"Post-processing complete. Outputs written to: {outputs}")
    print(f"Latest CSV/manifest/PNG copies mirrored to workspace root: {workspace}")


if __name__ == "__main__":
    main()
