#!/usr/bin/env python3
"""
Stage A deterministic benchmark for the sparse-data mine-closure study.

Purpose
-------
Stage A is the first implementation step for the benchmark developed in the
manuscript.  It is intentionally deterministic and is meant to answer a very
practical question first: *is the base conceptual model numerically stable,
physically sensible, and easy to diagnose before any stochastic work starts?*

The script therefore builds a sequence of MODFLOW 6 models with FloPy:

1. Predevelopment steady state.
2. Four 5-year operational stages with progressively deeper pit drainage.
3. A 100-year recovery stage with the excavated cells reactivated as porous
   backfill.

Important design choices
------------------------
- The implementation uses a structured DIS grid (100 m x 100 m plan cells)
  rather than the final manuscript DISV grid.  This is deliberate.  The goal of
  Stage A is numerical verification and transparent debugging before moving to a
  more complex production grid.
- The pit is represented without MODFLOW 6 LAK.  During operation, the pit is
  approximated with stage-specific drain cells placed on the pit floor plus an
  optional set of wall / bench seepage drains in the cells immediately outside
  the excavated footprint.  During recovery, those drains are removed and the
  previously excavated cells are reactivated with backfill hydraulic properties.
- Geometry changes are handled through *separate sequential simulations* rather
  than through time-varying idomain or time-varying hydraulic-property packages.
  This is robust and easy to inspect.
- Stage A.2 keeps the Stage A.1 wall-drain analogue and adds stronger run
  bookkeeping so figures, CSV tables, and manifests are all tied to one unique
  run identifier.  Stage A.2 also splits floor-drain, wall-drain, and well
  support into separate budget components for cleaner diagnostics.

Units
-----
Length: metres (m)
Time: days (d)
Hydraulic conductivity: metres per day (m/d)
Specific storage: 1/m
Specific yield: dimensionless
Recharge: metres per day (m/d); input values are specified in mm/year and
          converted internally.

What this script writes
-----------------------
- A model workspace for each stage under the selected root folder.
- A JSON manifest describing the stage sequence and output files.
- A CSV summary of the main benchmark parameters.

The companion script ``stage_a_postprocess.py`` reads the manifest and produces
quality-control plots and summary metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import flopy
except ImportError as exc:  # pragma: no cover - import checked at runtime only
    raise ImportError(
        "FloPy is required for this script. Install it first, for example with "
        "'pip install flopy'."
    ) from exc


# --------------------------------------------------------------------------------------
# Data classes holding benchmark inputs
# --------------------------------------------------------------------------------------


@dataclass
class StageDefinition:
    """Definition of one sequential simulation stage."""

    name: str
    kind: str  # 'steady', 'operation', 'recovery'
    duration_years: int
    pit_floor_elev_m: Optional[float] = None
    use_backfill: bool = False
    notes: str = ""


@dataclass
class StageAConfig:
    """Master configuration for the Stage A deterministic benchmark."""

    # --- paths / executables ---
    sim_name: str = "stage_a_sparse_data_mine_closure"
    mf6_exe: str = "mf6"

    # --- global units and time conversions ---
    days_per_year: float = 365.25

    # --- model domain geometry ---
    length_x_m: float = 12_000.0
    length_y_m: float = 8_000.0
    top_elev_m: float = 60.0
    botm_elevs_m: Tuple[float, ...] = (40.0, 25.0, 10.0, -5.0, -25.0, -45.0, -55.0, -70.0, -85.0, -100.0)
    delr_m: float = 100.0
    delc_m: float = 100.0

    # --- conceptual pit geometry ---
    pit_center_x_m: float = 4_500.0
    pit_center_y_m: float = 4_000.0
    pit_length_m: float = 1_400.0
    pit_width_m: float = 650.0
    pit_stage_depths_bgl_m: Tuple[float, ...] = (25.0, 50.0, 80.0, 110.0)

    # --- observation and receptor coordinates ---
    compliance_x_m: float = 6_500.0
    compliance_y_m: float = 4_000.0
    landholder_x_m: float = 9_300.0
    landholder_y_m: float = 4_100.0
    receptor_x_m: float = 8_800.0
    receptor_y_m: float = 4_200.0
    pit_margin_obs_x_m: float = 5_300.0
    pit_margin_obs_y_m: float = 4_000.0

    # --- predevelopment regional support ---
    west_boundary_head_m: float = 52.0
    east_boundary_head_m: float = 40.0
    ghb_conductance_factor: float = 1.0
    base_recharge_mm_per_year: float = 5.0

    # --- river / receptor corridor ---
    river_xmin_m: float = 7_800.0
    river_xmax_m: float = 10_900.0
    river_center_y_m: float = 4_200.0
    river_halfwidth_m: float = 120.0
    river_stage_m: float = 41.5
    river_bottom_m: float = 40.5
    riverbed_k_m_per_d: float = 0.05
    riverbed_thickness_m: float = 1.0

    # --- deterministic hidden architecture for Stage A ---
    channel_xmin_m: float = 3_600.0
    channel_xmax_m: float = 9_500.0
    channel_base_y_m: float = 4_100.0
    channel_amplitude_m: float = 180.0
    channel_wavelength_m: float = 3_600.0
    channel_halfwidth_m: float = 140.0
    channel_layers_zero_based: Tuple[int, ...] = (2, 3, 4)

    # --- layer-scale hydraulic properties (deterministic base case) ---
    # Layer groups: 1 upper/weathered, 2-5 main aquifer, 6-7 lower low-K, 8 basal unit.
    kh_by_layer_m_per_d: Tuple[float, ...] = (2.0, 4.5, 6.0, 6.0, 4.5, 0.25, 0.20, 0.20, 0.01, 0.01)
    kvkh_by_layer: Tuple[float, ...] = (0.25, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05)
    sy_by_layer: Tuple[float, ...] = (0.12, 0.10, 0.08, 0.08, 0.08, 0.05, 0.04, 0.04, 0.02, 0.02)
    ss_by_layer_per_m: Tuple[float, ...] = (1.0e-5, 5.0e-6, 5.0e-6, 5.0e-6, 5.0e-6, 2.0e-6, 2.0e-6, 2.0e-6, 1.0e-6, 1.0e-6)

    # --- deterministic paleochannel properties ---
    channel_kh_m_per_d: float = 20.0
    channel_kvkh: float = 0.20

    # --- backfill properties used in the recovery stage ---
    backfill_kh_m_per_d: float = 0.30
    backfill_kvkh: float = 0.15
    backfill_sy: float = 0.06
    backfill_ss_per_m: float = 5.0e-6

    # --- pit drainage representation during operations ---
    use_perimeter_wells: bool = False
    total_perimeter_pumping_m3_per_d_by_stage: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    drain_cond_multiplier: float = 1.0  # scales K*A/L-style pit-floor-drain conductance
    use_wall_drains: bool = True
    wall_drain_cond_multiplier: float = 0.15  # moderate analogue for wall / bench seepage

    # Small vertical buffer used to keep head-dependent boundaries safely above
    # convertible-cell bottoms.  This avoids common MODFLOW 6 input errors and
    # precision-edge cases when a boundary elevation is intended to be at, or
    # just slightly above, a cell bottom.
    boundary_elevation_buffer_m: float = 0.05

    # --- output / QC selections ---
    plan_plot_layer_zero_based: int = 3
    hydrograph_layer_zero_based: int = 4
    receptor_layer_zero_based: int = 1
    pit_margin_layer_zero_based: int = 2

    # --- solver controls ---
    outer_dvclose: float = 1.0e-4
    inner_dvclose: float = 1.0e-5
    outer_maximum: int = 100
    inner_maximum: int = 300
    relaxation_factor: float = 0.97

    # --- numeric masks populated later (not user input) ---
    nlay: int = field(init=False)
    nrow: int = field(init=False)
    ncol: int = field(init=False)

    def __post_init__(self) -> None:
        self.nlay = len(self.botm_elevs_m)
        self.ncol = int(round(self.length_x_m / self.delr_m))
        self.nrow = int(round(self.length_y_m / self.delc_m))
        if not np.isclose(self.ncol * self.delr_m, self.length_x_m):
            raise ValueError("length_x_m must be divisible by delr_m for the structured Stage A grid.")
        if not np.isclose(self.nrow * self.delc_m, self.length_y_m):
            raise ValueError("length_y_m must be divisible by delc_m for the structured Stage A grid.")

    @property
    def stage_floor_elevations_m(self) -> Tuple[float, ...]:
        """Pit floor elevations for the four operational stages."""
        return tuple(self.top_elev_m - depth for depth in self.pit_stage_depths_bgl_m)

    @property
    def base_recharge_m_per_d(self) -> float:
        return (self.base_recharge_mm_per_year / 1000.0) / self.days_per_year

    @property
    def delr(self) -> np.ndarray:
        return np.full(self.ncol, self.delr_m, dtype=float)

    @property
    def delc(self) -> np.ndarray:
        return np.full(self.nrow, self.delc_m, dtype=float)

    @property
    def botm(self) -> np.ndarray:
        return np.array(self.botm_elevs_m, dtype=float)


# --------------------------------------------------------------------------------------
# Helper functions for grid geometry and deterministic architecture
# --------------------------------------------------------------------------------------


def build_stage_list(cfg: StageAConfig) -> List[StageDefinition]:
    """Return the ordered Stage A simulation sequence."""
    floors = cfg.stage_floor_elevations_m
    return [
        StageDefinition(
            name="predevelopment",
            kind="steady",
            duration_years=1,
            pit_floor_elev_m=None,
            use_backfill=False,
            notes="Pre-mining steady-state reference case.",
        ),
        StageDefinition(
            name="ops_stage_1",
            kind="operation",
            duration_years=5,
            pit_floor_elev_m=floors[0],
            use_backfill=False,
            notes="Operational years 1-5; pit floor at 25 m below ground.",
        ),
        StageDefinition(
            name="ops_stage_2",
            kind="operation",
            duration_years=5,
            pit_floor_elev_m=floors[1],
            use_backfill=False,
            notes="Operational years 6-10; pit floor at 50 m below ground.",
        ),
        StageDefinition(
            name="ops_stage_3",
            kind="operation",
            duration_years=5,
            pit_floor_elev_m=floors[2],
            use_backfill=False,
            notes="Operational years 11-15; pit floor at 80 m below ground.",
        ),
        StageDefinition(
            name="ops_stage_4",
            kind="operation",
            duration_years=5,
            pit_floor_elev_m=floors[3],
            use_backfill=False,
            notes="Operational years 16-20; pit floor at 110 m below ground.",
        ),
        StageDefinition(
            name="recovery",
            kind="recovery",
            duration_years=100,
            pit_floor_elev_m=floors[3],
            use_backfill=True,
            notes="Closure and 100-year recovery with porous backfill in excavated cells.",
        ),
    ]


def layer_top_bottom_thickness(cfg: StageAConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of layer tops, bottoms and thicknesses."""
    lay_tops = np.empty(cfg.nlay, dtype=float)
    lay_botm = cfg.botm.copy()
    lay_tops[0] = cfg.top_elev_m
    lay_tops[1:] = lay_botm[:-1]
    thickness = lay_tops - lay_botm
    return lay_tops, lay_botm, thickness


def clamp_above_bottom(value_m: float, cell_bottom_m: float, buffer_m: float) -> float:
    """Return a boundary elevation safely above a convertible-cell bottom.

    MODFLOW 6 checks several head-dependent boundaries against the cell bottom
    for convertible cells.  A small positive buffer keeps the intent of the
    boundary condition unchanged while avoiding precision-edge failures.
    """
    return max(value_m, cell_bottom_m + buffer_m)


def candidate_list_files(stage_workspace: Path, stage_name: str) -> List[Path]:
    """Return likely MODFLOW 6 list files for one stage workspace."""
    candidates = [stage_workspace / f"{stage_name}.lst", stage_workspace / "mfsim.lst"]
    candidates.extend(sorted(stage_workspace.glob("*.lst")))
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def tail_text_file(path: Path, nlines: int = 60) -> List[str]:
    """Return the last ``nlines`` lines from a text file if it exists."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return text[-nlines:]


def collect_failure_diagnostics(stage_workspace: Path, stage_name: str, buff: Sequence[str]) -> str:
    """Assemble a compact diagnostic message when MODFLOW 6 fails."""
    chunks: List[str] = []
    if buff:
        chunks.append("--- MODFLOW 6 stdout/stderr tail ---")
        chunks.extend(list(buff)[-40:])

    for path in candidate_list_files(stage_workspace, stage_name):
        tail = tail_text_file(path, nlines=60)
        if tail:
            chunks.append(f"--- tail of {path.name} ---")
            chunks.extend(tail)

    if not chunks:
        return "No additional MODFLOW 6 diagnostic text was found in the stage workspace."
    return "\n".join(chunks)


def cell_center_arrays(cfg: StageAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return x- and y-centre coordinate arrays for the structured grid.

    Note: row 0 is at the northern side of the domain, which is the standard
    MODFLOW row ordering.  The y-centres are therefore returned from north to
    south.
    """
    x = np.cumsum(cfg.delr) - 0.5 * cfg.delr
    y = cfg.length_y_m - (np.cumsum(cfg.delc) - 0.5 * cfg.delc)
    return x, y


def meshgrid_centres(cfg: StageAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2-D arrays of cell-centre coordinates."""
    x, y = cell_center_arrays(cfg)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def xy_to_rowcol(cfg: StageAConfig, x_m: float, y_m: float) -> Tuple[int, int]:
    """Map real-world coordinates to a structured-grid (row, col) index."""
    if not (0.0 <= x_m <= cfg.length_x_m and 0.0 <= y_m <= cfg.length_y_m):
        raise ValueError(f"Point ({x_m}, {y_m}) lies outside the model domain.")

    col = int(min(cfg.ncol - 1, max(0, math.floor(x_m / cfg.delr_m))))
    row = int(min(cfg.nrow - 1, max(0, math.floor((cfg.length_y_m - y_m) / cfg.delc_m))))
    return row, col


def ellipse_mask(cfg: StageAConfig, center_x_m: float, center_y_m: float, length_m: float, width_m: float) -> np.ndarray:
    """Boolean mask for an elliptical footprint on the structured grid."""
    xx, yy = meshgrid_centres(cfg)
    a = 0.5 * length_m
    b = 0.5 * width_m
    return (((xx - center_x_m) / a) ** 2 + ((yy - center_y_m) / b) ** 2) <= 1.0


def pit_mask(cfg: StageAConfig) -> np.ndarray:
    """Pit footprint mask on the structured grid."""
    return ellipse_mask(cfg, cfg.pit_center_x_m, cfg.pit_center_y_m, cfg.pit_length_m, cfg.pit_width_m)


def pit_shell_face_counts(cfg: StageAConfig) -> np.ndarray:
    """Return the number of orthogonal pit-adjacent faces for each shell cell.

    The returned array is zero everywhere except in cells immediately outside the
    pit footprint.  A value of 1 to 4 indicates how many north/south/east/west
    cell faces are directly adjacent to excavated pit cells.  This is used by the
    Stage A.1 wall-drain analogue to approximate seepage from exposed walls and
    benches without introducing a lake package or a more complicated geometric
    representation.
    """
    pit2d = pit_mask(cfg)
    counts = np.zeros_like(pit2d, dtype=int)

    counts[1:, :] += pit2d[:-1, :]
    counts[:-1, :] += pit2d[1:, :]
    counts[:, 1:] += pit2d[:, :-1]
    counts[:, :-1] += pit2d[:, 1:]

    counts[pit2d] = 0
    return counts


def paleochannel_mask(cfg: StageAConfig) -> np.ndarray:
    """Deterministic moderate-connectivity paleochannel corridor for Stage A.

    This is not the stochastic T-PROGS representation.  It is a single, fixed,
    geologically plausible transmissive corridor used only so that Stage A has a
    meaningful architecture to test against.
    """
    xx, yy = meshgrid_centres(cfg)
    within_x = (xx >= cfg.channel_xmin_m) & (xx <= cfg.channel_xmax_m)
    centerline_y = cfg.channel_base_y_m + cfg.channel_amplitude_m * np.sin(
        2.0 * np.pi * (xx - cfg.channel_xmin_m) / cfg.channel_wavelength_m
    )
    within_width = np.abs(yy - centerline_y) <= cfg.channel_halfwidth_m
    return within_x & within_width


def river_mask(cfg: StageAConfig) -> np.ndarray:
    """Approximate internal gaining-stream / wetland corridor mask."""
    xx, yy = meshgrid_centres(cfg)
    return (
        (xx >= cfg.river_xmin_m)
        & (xx <= cfg.river_xmax_m)
        & (np.abs(yy - cfg.river_center_y_m) <= cfg.river_halfwidth_m)
    )


# --------------------------------------------------------------------------------------
# Property arrays, excavation logic, and stage-specific stresses
# --------------------------------------------------------------------------------------


def build_base_property_arrays(cfg: StageAConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic K, K33, Sy and Ss arrays for the base benchmark.

    The deterministic Stage A model includes a moderate-conductivity paleochannel
    in selected middle layers, while the background system is represented only by
    the hydrostratigraphic layer properties.
    """
    kh = np.zeros((cfg.nlay, cfg.nrow, cfg.ncol), dtype=float)
    k33 = np.zeros_like(kh)
    sy = np.zeros_like(kh)
    ss = np.zeros_like(kh)

    # Fill layer-by-layer background properties first.
    for k in range(cfg.nlay):
        kh[k, :, :] = cfg.kh_by_layer_m_per_d[k]
        k33[k, :, :] = cfg.kh_by_layer_m_per_d[k] * cfg.kvkh_by_layer[k]
        sy[k, :, :] = cfg.sy_by_layer[k]
        ss[k, :, :] = cfg.ss_by_layer_per_m[k]

    # Superimpose a deterministic transmissive corridor in the main aquifer.
    channel = paleochannel_mask(cfg)
    for k in cfg.channel_layers_zero_based:
        kh[k, channel] = cfg.channel_kh_m_per_d
        k33[k, channel] = cfg.channel_kh_m_per_d * cfg.channel_kvkh

    return kh, k33, sy, ss


def excavated_cells_by_layer(cfg: StageAConfig, pit_floor_elev_m: float) -> np.ndarray:
    """Return a 3-D boolean array of cells fully excavated above the pit floor.

    Logic
    -----
    A cell is considered fully excavated if its *entire* thickness lies above the
    current pit floor elevation, i.e., if the cell bottom is higher than the pit
    floor.  The first cell whose bottom lies below the floor remains active and is
    drained by the stage-specific pit-floor drain.  This avoids the need for
    time-varying idomain inside a single simulation.
    """
    pit2d = pit_mask(cfg)
    _, lay_botm, _ = layer_top_bottom_thickness(cfg)
    excavated = np.zeros((cfg.nlay, cfg.nrow, cfg.ncol), dtype=bool)
    for k in range(cfg.nlay):
        if lay_botm[k] > pit_floor_elev_m:
            excavated[k, pit2d] = True
    return excavated


def build_stage_arrays(
    cfg: StageAConfig, stage: StageDefinition
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return idomain, kh, k33, sy and ss for one stage."""
    kh, k33, sy, ss = build_base_property_arrays(cfg)
    idomain = np.ones((cfg.nlay, cfg.nrow, cfg.ncol), dtype=int)

    if stage.kind == "operation" and stage.pit_floor_elev_m is not None:
        excavated = excavated_cells_by_layer(cfg, stage.pit_floor_elev_m)
        idomain[excavated] = 0

    if stage.kind == "recovery" and stage.use_backfill and stage.pit_floor_elev_m is not None:
        # Re-activate excavated cells as porous backfill.
        backfill_zone = excavated_cells_by_layer(cfg, stage.pit_floor_elev_m)
        idomain[backfill_zone] = 1
        kh[backfill_zone] = cfg.backfill_kh_m_per_d
        k33[backfill_zone] = cfg.backfill_kh_m_per_d * cfg.backfill_kvkh
        sy[backfill_zone] = cfg.backfill_sy
        ss[backfill_zone] = cfg.backfill_ss_per_m

    return idomain, kh, k33, sy, ss


def build_initial_heads(
    cfg: StageAConfig,
    idomain: np.ndarray,
    previous_head: Optional[np.ndarray],
    stage: StageDefinition,
) -> np.ndarray:
    """Create starting heads for a stage.

    - If ``previous_head`` is None, a simple west-to-east linear gradient is used.
    - If a previous stage exists, its final simulated heads are used wherever
      possible.
    - Cells that become newly active in the recovery stage (the backfill cells)
      receive the final pit-floor elevation as an initial water level.  This is a
      deliberate, transparent approximation for the first recovery model.
    """
    x, _ = cell_center_arrays(cfg)
    frac = x / cfg.length_x_m
    base_profile = cfg.west_boundary_head_m + frac * (cfg.east_boundary_head_m - cfg.west_boundary_head_m)
    linear_head = np.repeat(base_profile.reshape(1, 1, cfg.ncol), cfg.nlay, axis=0)
    linear_head = np.repeat(linear_head, cfg.nrow, axis=1)

    if previous_head is None:
        strt = linear_head.copy()
    else:
        strt = previous_head.copy()
        # Replace typical MF6 no-data values for inactive/dry cells.
        bad = (~np.isfinite(strt)) | (strt > 1.0e20) | (strt < -1.0e20)
        strt[bad] = np.nan

        # Fill any remaining NaNs with the regional linear gradient first.
        fill_mask = np.isnan(strt)
        strt[fill_mask] = linear_head[fill_mask]

    # Newly active backfill cells need a sensible closure-stage initial head.
    if stage.kind == "recovery" and stage.use_backfill and stage.pit_floor_elev_m is not None:
        backfill_zone = excavated_cells_by_layer(cfg, stage.pit_floor_elev_m)
        strt[backfill_zone] = stage.pit_floor_elev_m

    # Inactive cells may carry arbitrary values, but keeping them finite helps diagnostics.
    strt[idomain == 0] = 0.0
    return strt


def recharge_array_for_stage(cfg: StageAConfig, stage: StageDefinition) -> np.ndarray:
    """Return a 2-D recharge array for the stage.

    Recharge is uniform at the domain scale in this Stage A base case.  Recharge
    is set to zero over the pit footprint after mining begins.
    """
    rch = np.full((cfg.nrow, cfg.ncol), cfg.base_recharge_m_per_d, dtype=float)
    if stage.kind in {"operation", "recovery"}:
        rch[pit_mask(cfg)] = 0.0
    return rch


def build_ghb_spd(
    cfg: StageAConfig, idomain: np.ndarray, kh: np.ndarray
) -> List[Tuple[Tuple[int, int, int], float, float]]:
    """Build stress-period data for west and east general-head boundaries.

    A small head-above-bottom buffer is enforced for convertible cells.  This is
    numerically conservative and prevents a common MF6 input failure where a GHB
    head is specified at, or microscopically below, a cell bottom.
    """
    _, lay_botm, thick = layer_top_bottom_thickness(cfg)
    delr = cfg.delr_m
    delc = cfg.delc_m
    spd: List[Tuple[Tuple[int, int, int], float, float]] = []
    n_adjusted = 0

    for k in range(cfg.nlay):
        for i in range(cfg.nrow):
            # West boundary (column 0)
            if idomain[k, i, 0] > 0:
                cond = cfg.ghb_conductance_factor * kh[k, i, 0] * thick[k] * delc / (0.5 * delr)
                head = clamp_above_bottom(cfg.west_boundary_head_m, lay_botm[k], cfg.boundary_elevation_buffer_m)
                if not np.isclose(head, cfg.west_boundary_head_m):
                    n_adjusted += 1
                spd.append(((k, i, 0), head, cond))
            # East boundary (last column)
            if idomain[k, i, cfg.ncol - 1] > 0:
                cond = cfg.ghb_conductance_factor * kh[k, i, cfg.ncol - 1] * thick[k] * delc / (0.5 * delr)
                head = clamp_above_bottom(cfg.east_boundary_head_m, lay_botm[k], cfg.boundary_elevation_buffer_m)
                if not np.isclose(head, cfg.east_boundary_head_m):
                    n_adjusted += 1
                spd.append(((k, i, cfg.ncol - 1), head, cond))

    if n_adjusted > 0:
        print(
            f"INFO: adjusted {n_adjusted} GHB head values upward by {cfg.boundary_elevation_buffer_m:g} m "
            "to keep them above convertible-cell bottoms."
        )
    return spd


def build_riv_spd(cfg: StageAConfig, idomain: np.ndarray) -> List[Tuple[Tuple[int, int, int], float, float, float]]:
    """Build stress-period data for the internal river/receptor reach.

    The default benchmark uses the top layer for the receptor corridor.  MF6
    checks river-bottom and stage elevations against the host-cell bottom for
    convertible cells, so both are kept safely above the cell bottom.
    """
    stream = river_mask(cfg)
    spd: List[Tuple[Tuple[int, int, int], float, float, float]] = []

    # Use only the top layer for the receptor reach in Stage A.
    k = 0
    _, lay_botm, _ = layer_top_bottom_thickness(cfg)
    cell_bottom = lay_botm[k]
    streambed_cond = cfg.riverbed_k_m_per_d * (cfg.delr_m * cfg.delc_m) / cfg.riverbed_thickness_m
    rows, cols = np.where(stream)
    n_bottom_adjusted = 0
    n_stage_adjusted = 0
    for i, j in zip(rows, cols):
        if idomain[k, i, j] > 0:
            rbot = clamp_above_bottom(cfg.river_bottom_m, cell_bottom, cfg.boundary_elevation_buffer_m)
            stage = max(cfg.river_stage_m, rbot + cfg.boundary_elevation_buffer_m)
            if not np.isclose(rbot, cfg.river_bottom_m):
                n_bottom_adjusted += 1
            if not np.isclose(stage, cfg.river_stage_m):
                n_stage_adjusted += 1
            spd.append(((k, i, j), stage, streambed_cond, rbot))

    if n_bottom_adjusted > 0 or n_stage_adjusted > 0:
        print(
            f"INFO: adjusted {n_bottom_adjusted} RIV bottoms and {n_stage_adjusted} RIV stages "
            "to keep the river package geometrically valid in convertible cells."
        )
    return spd


def build_floor_drn_spd(
    cfg: StageAConfig,
    stage: StageDefinition,
    idomain: np.ndarray,
    kh: np.ndarray,
) -> List[Tuple[Tuple[int, int, int], float, float]]:
    """Build pit-floor drain stress-period data for one operational stage.

    The floor drain is placed in the shallowest active cell in each pit column.
    A small elevation buffer is enforced if the nominal pit-floor elevation lands
    exactly on, or slightly below, the bottom of the host cell.
    """
    if stage.kind != "operation" or stage.pit_floor_elev_m is None:
        return []

    pit2d = pit_mask(cfg)
    _, lay_botm, thick = layer_top_bottom_thickness(cfg)
    spd: List[Tuple[Tuple[int, int, int], float, float]] = []
    rows, cols = np.where(pit2d)
    area = cfg.delr_m * cfg.delc_m
    n_adjusted = 0

    for i, j in zip(rows, cols):
        active_layers = np.where(idomain[:, i, j] > 0)[0]
        if active_layers.size == 0:
            continue
        k = int(active_layers[0])
        vertical_distance = max(1.0, 0.5 * thick[k])
        cond = cfg.drain_cond_multiplier * kh[k, i, j] * area / vertical_distance
        elev = clamp_above_bottom(stage.pit_floor_elev_m, lay_botm[k], cfg.boundary_elevation_buffer_m)
        if not np.isclose(elev, stage.pit_floor_elev_m):
            n_adjusted += 1
        spd.append(((k, i, j), elev, cond))

    if n_adjusted > 0:
        print(
            f"INFO: adjusted {n_adjusted} PITDRN floor elevations upward by {cfg.boundary_elevation_buffer_m:g} m "
            "to keep them above host-cell bottoms."
        )
    return spd


def build_wall_drn_spd(
    cfg: StageAConfig,
    stage: StageDefinition,
    idomain: np.ndarray,
    kh: np.ndarray,
) -> List[Tuple[Tuple[int, int, int], float, float]]:
    """Build a moderate wall / bench seepage drain analogue for operations.

    Rationale
    ---------
    The original Stage A representation used pit-floor drains only.  That is a
    useful first debug model, but late operational behaviour can become too
    dependent on the deepest remaining active floor cells after the upper, more
    transmissive units have been excavated.  This helper adds drains in the cells
    immediately outside the pit footprint wherever a vertical wall interval is
    exposed.  The drain elevation is set at the base of the exposed interval, and
    the conductance is based on a simple K*A/L expression scaled by a dedicated
    wall-drain multiplier.

    The intent is not to claim a unique physical representation of pit-wall
    seepage.  The intent is to preserve the no-LAK Stage A structure while
    avoiding an unrealistically floor-dominated inflow response.
    """
    if stage.kind != "operation" or stage.pit_floor_elev_m is None or (not cfg.use_wall_drains):
        return []

    face_counts = pit_shell_face_counts(cfg)
    lay_tops, lay_botm, _ = layer_top_bottom_thickness(cfg)
    rows, cols = np.where(face_counts > 0)
    spd: List[Tuple[Tuple[int, int, int], float, float]] = []
    n_adjusted = 0

    for i, j in zip(rows, cols):
        nfaces = int(face_counts[i, j])
        if nfaces <= 0:
            continue

        # The Stage A grid is square in plan, so one representative half-cell
        # flow length is sufficient for this analogue.
        flow_distance = max(1.0, 0.5 * cfg.delr_m)
        lateral_face_length = nfaces * cfg.delr_m

        for k in range(cfg.nlay):
            if idomain[k, i, j] <= 0:
                continue

            exposed_base = max(lay_botm[k], stage.pit_floor_elev_m)
            exposed_height = max(0.0, lay_tops[k] - exposed_base)
            if exposed_height <= 0.0:
                continue

            exposed_area = lateral_face_length * exposed_height
            cond = cfg.wall_drain_cond_multiplier * kh[k, i, j] * exposed_area / flow_distance
            if cond <= 0.0:
                continue

            elev = clamp_above_bottom(exposed_base, lay_botm[k], cfg.boundary_elevation_buffer_m)
            if not np.isclose(elev, exposed_base):
                n_adjusted += 1
            spd.append(((k, i, j), elev, cond))

    if n_adjusted > 0:
        print(
            f"INFO: adjusted {n_adjusted} PITWALL elevations upward by {cfg.boundary_elevation_buffer_m:g} m "
            "to keep them above host-cell bottoms."
        )
    return spd


def build_drn_spd(
    cfg: StageAConfig,
    stage: StageDefinition,
    idomain: np.ndarray,
    kh: np.ndarray,
) -> List[Tuple[Tuple[int, int, int], float, float]]:
    """Return the combined floor- and wall-drain stress-period data."""
    return build_floor_drn_spd(cfg, stage, idomain, kh) + build_wall_drn_spd(cfg, stage, idomain, kh)


def build_wel_spd(
    cfg: StageAConfig,
    stage_index_zero_based: int,
    stage: StageDefinition,
    idomain: np.ndarray,
) -> List[Tuple[Tuple[int, int, int], float]]:
    """Optional perimeter wells around the pit.

    By default Stage A uses drains only.  The well support is kept as an option
    because some users prefer a perimeter-pumping analogue during the operational
    phase.  The default total pumping is zero in all stages.
    """
    if (not cfg.use_perimeter_wells) or stage.kind != "operation":
        return []

    total_q = cfg.total_perimeter_pumping_m3_per_d_by_stage[stage_index_zero_based]
    if np.isclose(total_q, 0.0):
        return []

    pit2d = pit_mask(cfg)
    perimeter = np.zeros_like(pit2d, dtype=bool)
    # Simple one-cell dilation ring around the pit footprint.
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.roll(np.roll(pit2d, di, axis=0), dj, axis=1)
        perimeter |= shifted
    perimeter &= ~pit2d

    rows, cols = np.where(perimeter)
    spd: List[Tuple[Tuple[int, int, int], float]] = []
    if rows.size == 0:
        return spd

    q_per_cell = -abs(total_q) / float(rows.size)  # negative sign = pumping out of aquifer
    k_active = cfg.hydrograph_layer_zero_based
    for i, j in zip(rows, cols):
        if idomain[k_active, i, j] > 0:
            spd.append(((k_active, i, j), q_per_cell))
    return spd


# --------------------------------------------------------------------------------------
# FloPy model build / write / run
# --------------------------------------------------------------------------------------


def build_perioddata(cfg: StageAConfig, stage: StageDefinition) -> Tuple[List[Tuple[float, int, float]], Dict[int, bool], Dict[int, bool]]:
    """Return period data plus STO flags for a stage."""
    if stage.kind == "steady":
        perioddata = [(1.0, 1, 1.0)]
        steady_state = {0: True}
        transient: Dict[int, bool] = {}
    else:
        perioddata = [(cfg.days_per_year, 1, 1.0) for _ in range(stage.duration_years)]
        steady_state = {}
        transient = {iper: True for iper in range(stage.duration_years)}
    return perioddata, steady_state, transient


def read_last_head(headfile: Path) -> np.ndarray:
    """Read the final simulated head array from a MODFLOW 6 head file."""
    hdobj = flopy.utils.HeadFile(str(headfile), precision="double")
    last_kstpkper = hdobj.get_kstpkper()[-1]
    return hdobj.get_data(kstpkper=last_kstpkper)


def build_and_run_stage(
    cfg: StageAConfig,
    stage: StageDefinition,
    stage_workspace: Path,
    previous_head: Optional[np.ndarray],
    stage_index_in_sequence: int,
    run_model: bool = True,
) -> Dict[str, object]:
    """Build, write and optionally run one stage model.

    Returns a manifest dictionary with the main file paths and metadata.
    """
    stage_workspace.mkdir(parents=True, exist_ok=True)

    idomain, kh, k33, sy, ss = build_stage_arrays(cfg, stage)
    strt = build_initial_heads(cfg, idomain, previous_head, stage)
    rch = recharge_array_for_stage(cfg, stage)
    ghb_spd = build_ghb_spd(cfg, idomain, kh)
    riv_spd = build_riv_spd(cfg, idomain)
    floor_drn_spd = build_floor_drn_spd(cfg, stage, idomain, kh)
    wall_drn_spd = build_wall_drn_spd(cfg, stage, idomain, kh)
    drn_spd = floor_drn_spd + wall_drn_spd

    # The stage index among operational stages is needed only for optional WEL support.
    op_stage_idx = max(0, stage_index_in_sequence - 1)
    wel_spd = build_wel_spd(cfg, op_stage_idx, stage, idomain)

    sim = flopy.mf6.MFSimulation(
        sim_name=stage.name,
        sim_ws=str(stage_workspace),
        exe_name=cfg.mf6_exe,
    )

    perioddata, steady_state, transient = build_perioddata(cfg, stage)
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(perioddata), perioddata=perioddata)

    flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_dvclose=cfg.outer_dvclose,
        inner_dvclose=cfg.inner_dvclose,
        outer_maximum=cfg.outer_maximum,
        inner_maximum=cfg.inner_maximum,
        relaxation_factor=cfg.relaxation_factor,
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=stage.name, save_flows=True, newtonoptions="NEWTON")

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=cfg.nlay,
        nrow=cfg.nrow,
        ncol=cfg.ncol,
        delr=cfg.delr,
        delc=cfg.delc,
        top=cfg.top_elev_m,
        botm=cfg.botm,
        idomain=idomain,
    )

    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=np.ones(cfg.nlay, dtype=int),
        k=kh,
        k33=k33,
        save_specific_discharge=True,
    )

    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=np.ones(cfg.nlay, dtype=int),
        ss=ss,
        sy=sy,
        steady_state=steady_state if steady_state else None,
        transient=transient if transient else None,
    )

    floor_drn_package_name = "PITDRN_FLOOR"
    wall_drn_package_name = "PITDRN_WALL"
    well_package_name = "PITWEL"

    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb_spd}, pname="GHB")
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: riv_spd}, pname="RIV")

    if floor_drn_spd:
        flopy.mf6.ModflowGwfdrn(
            gwf,
            stress_period_data={0: floor_drn_spd},
            pname=floor_drn_package_name,
            filename=f"{stage.name}.pitdrn_floor.drn",
        )
    if wall_drn_spd:
        flopy.mf6.ModflowGwfdrn(
            gwf,
            stress_period_data={0: wall_drn_spd},
            pname=wall_drn_package_name,
            filename=f"{stage.name}.pitdrn_wall.drn",
        )
    if wel_spd:
        flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data={0: wel_spd},
            pname=well_package_name,
            filename=f"{stage.name}.pitwel.wel",
        )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{stage.name}.hds",
        budget_filerecord=f"{stage.name}.cbc",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("BUDGET", "LAST")],
    )

    sim.write_simulation()

    success = None
    stdout = ""
    if run_model:
        success, buff = sim.run_simulation(silent=True)
        stdout = "\n".join(buff)
        if not success:
            diagnostics = collect_failure_diagnostics(stage_workspace, stage.name, buff)
            raise RuntimeError(
                f"MODFLOW 6 failed during stage '{stage.name}'. Review the stage workspace at {stage_workspace}.\n\n"
                f"{diagnostics}"
            )

    list_candidates = candidate_list_files(stage_workspace, stage.name)
    list_file = next((str(path.resolve()) for path in list_candidates if path.exists()), str((stage_workspace / f"{stage.name}.lst").resolve()))

    manifest = {
        "name": stage.name,
        "kind": stage.kind,
        "duration_years": stage.duration_years,
        "pit_floor_elev_m": stage.pit_floor_elev_m,
        "use_backfill": stage.use_backfill,
        "notes": stage.notes,
        "workspace": str(stage_workspace.resolve()),
        "model_name": stage.name,
        "n_pit_floor_drains": len(floor_drn_spd),
        "n_pit_wall_drains": len(wall_drn_spd),
        "n_pit_wells": len(wel_spd),
        "use_wall_drains": cfg.use_wall_drains,
        "pit_floor_package_name": floor_drn_package_name if floor_drn_spd else None,
        "pit_wall_package_name": wall_drn_package_name if wall_drn_spd else None,
        "pit_well_package_name": well_package_name if wel_spd else None,
        "ghb_package_name": "GHB",
        "riv_package_name": "RIV",
        "head_file": str((stage_workspace / f"{stage.name}.hds").resolve()),
        "budget_file": str((stage_workspace / f"{stage.name}.cbc").resolve()),
        "list_file": list_file,
        "success": success,
        "stdout_tail": stdout.splitlines()[-20:] if stdout else [],
    }
    return manifest


# --------------------------------------------------------------------------------------
# Parameter summary and manifest helpers
# --------------------------------------------------------------------------------------


def observation_cells(cfg: StageAConfig) -> Dict[str, Dict[str, object]]:
    """Return observation-point metadata with x/y coordinates and cell IDs."""
    def make_obs(name: str, x: float, y: float, layer: int) -> Dict[str, object]:
        row, col = xy_to_rowcol(cfg, x, y)
        return {
            "name": name,
            "x_m": x,
            "y_m": y,
            "cellid": [layer, row, col],
        }

    return {
        "compliance": make_obs("compliance", cfg.compliance_x_m, cfg.compliance_y_m, cfg.hydrograph_layer_zero_based),
        "landholder": make_obs("landholder", cfg.landholder_x_m, cfg.landholder_y_m, cfg.hydrograph_layer_zero_based),
        "receptor": make_obs("receptor", cfg.receptor_x_m, cfg.receptor_y_m, cfg.receptor_layer_zero_based),
        "pit_margin": make_obs("pit_margin", cfg.pit_margin_obs_x_m, cfg.pit_margin_obs_y_m, cfg.pit_margin_layer_zero_based),
    }


def parameter_summary_dataframe(cfg: StageAConfig) -> pd.DataFrame:
    """Create a tidy table summarizing the main Stage A parameters and units."""
    records = [
        ("domain", "length_x", cfg.length_x_m, "m", "East-west model extent"),
        ("domain", "length_y", cfg.length_y_m, "m", "North-south model extent"),
        ("domain", "top_elevation", cfg.top_elev_m, "m", "Ground-surface reference elevation"),
        ("domain", "layer_bottoms", "; ".join(f"{b:.1f}" for b in cfg.botm), "m", "Bottom elevation of each layer"),
        ("grid", "delr", cfg.delr_m, "m", "Column width in Stage A structured grid"),
        ("grid", "delc", cfg.delc_m, "m", "Row width in Stage A structured grid"),
        ("pit", "pit_center_x", cfg.pit_center_x_m, "m", "Pit-centre x coordinate"),
        ("pit", "pit_center_y", cfg.pit_center_y_m, "m", "Pit-centre y coordinate"),
        ("pit", "pit_length", cfg.pit_length_m, "m", "Pit major axis"),
        ("pit", "pit_width", cfg.pit_width_m, "m", "Pit minor axis"),
        (
            "pit",
            "pit_stage_floors",
            "; ".join(f"{z:.1f}" for z in cfg.stage_floor_elevations_m),
            "m",
            "Pit-floor elevations for the four operational stages",
        ),
        ("hydrology", "west_boundary_head", cfg.west_boundary_head_m, "m", "West GHB head"),
        ("hydrology", "east_boundary_head", cfg.east_boundary_head_m, "m", "East GHB head"),
        ("hydrology", "base_recharge", cfg.base_recharge_mm_per_year, "mm/yr", "Uniform diffuse recharge before pit masking"),
        ("pit", "pit_floor_drain_multiplier", cfg.drain_cond_multiplier, "-", "Multiplier on K*A/L pit-floor-drain conductance"),
        ("pit", "use_wall_drains", int(cfg.use_wall_drains), "flag", "1 = enable shell-cell wall / bench seepage drains"),
        ("pit", "wall_drain_multiplier", cfg.wall_drain_cond_multiplier, "-", "Multiplier on K*A/L wall-drain conductance analogue"),
        ("river", "river_stage", cfg.river_stage_m, "m", "Internal receptor-reach stage"),
        ("river", "river_bottom", cfg.river_bottom_m, "m", "Internal receptor-reach bed elevation"),
        ("recovery", "backfill_kh", cfg.backfill_kh_m_per_d, "m/d", "Base backfill horizontal conductivity"),
        ("recovery", "backfill_sy", cfg.backfill_sy, "-", "Base backfill specific yield"),
    ]
    return pd.DataFrame(records, columns=["group", "parameter", "value", "units", "note"])


def sanitize_run_tag(tag: str) -> str:
    """Return a filesystem-safe run tag."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
    if not safe:
        raise ValueError("The supplied --run-tag becomes empty after sanitization.")
    return safe


def build_run_id(user_tag: Optional[str] = None) -> str:
    """Return a unique run identifier for Stage A.2 bookkeeping."""
    if user_tag is not None:
        return sanitize_run_tag(user_tag)
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally run the Stage A FloPy / MODFLOW 6 benchmark.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("stage_a_workspace"),
        help="Root workspace directory for all sequential stage models.",
    )
    parser.add_argument(
        "--mf6",
        type=str,
        default=None,
        help="Path to the MODFLOW 6 executable. If omitted, the script uses 'mf6' from PATH.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Write model input files but do not run MODFLOW 6.",
    )
    parser.add_argument(
        "--enable-perimeter-wells",
        action="store_true",
        help="Enable the optional perimeter-well support during operations.",
    )
    parser.add_argument(
        "--disable-wall-drains",
        action="store_true",
        help="Disable the Stage A.1 wall / bench seepage drain analogue and keep floor drains only.",
    )
    parser.add_argument(
        "--wall-drain-cond-multiplier",
        type=float,
        default=None,
        help="Override the default wall-drain conductance multiplier (Stage A.1 cleanup).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional user-defined tag for this Stage A.2 run. If omitted, a UTC timestamp-based run id is created.",
    )
    parser.add_argument(
        "--clean-workspace-root",
        action="store_true",
        help="Delete the entire workspace root before building the new run. Use with care.",
    )
    parser.add_argument(
        "--clean-run-root",
        action="store_true",
        help="If the selected run folder already exists, delete that run folder before rebuilding it.",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    cfg = StageAConfig()
    if args.mf6 is not None:
        cfg.mf6_exe = args.mf6
    if args.enable_perimeter_wells:
        cfg.use_perimeter_wells = True
    if args.disable_wall_drains:
        cfg.use_wall_drains = False
    if args.wall_drain_cond_multiplier is not None:
        cfg.wall_drain_cond_multiplier = float(args.wall_drain_cond_multiplier)

    if (not args.build_only) and shutil.which(cfg.mf6_exe) is None and not Path(cfg.mf6_exe).exists():
        raise FileNotFoundError(
            f"MODFLOW 6 executable '{cfg.mf6_exe}' was not found. Provide --mf6 /path/to/mf6 or use --build-only."
        )

    workspace_root = args.workspace.resolve()
    if args.clean_workspace_root and workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(args.run_tag)
    run_root = workspace_root / "runs" / run_id
    if run_root.exists():
        if args.clean_run_root:
            shutil.rmtree(run_root)
        else:
            raise FileExistsError(
                f"Run root already exists: {run_root}. Use --clean-run-root or choose a different --run-tag."
            )
    run_root.mkdir(parents=True, exist_ok=False)

    parameter_df = parameter_summary_dataframe(cfg)
    parameter_df.to_csv(run_root / "stage_a_parameter_summary.csv", index=False)
    parameter_df.to_csv(workspace_root / "stage_a_parameter_summary.csv", index=False)
    parameter_df.to_csv(workspace_root / f"stage_a_parameter_summary_{run_id}.csv", index=False)

    stage_sequence = build_stage_list(cfg)
    manifests: List[Dict[str, object]] = []
    previous_head: Optional[np.ndarray] = None

    start_year = 0.0
    for idx, stage in enumerate(stage_sequence):
        stage_ws = run_root / stage.name
        manifest = build_and_run_stage(
            cfg=cfg,
            stage=stage,
            stage_workspace=stage_ws,
            previous_head=previous_head,
            stage_index_in_sequence=idx,
            run_model=not args.build_only,
        )
        manifest["run_id"] = run_id
        manifest["start_year"] = start_year
        manifest["end_year"] = start_year + stage.duration_years
        manifests.append(manifest)

        # Load end heads for chaining only if the model was run.
        if not args.build_only:
            previous_head = read_last_head(Path(manifest["head_file"]))
        else:
            previous_head = None
        start_year += stage.duration_years if stage.kind != "steady" else 0.0

    created_utc = datetime.now(timezone.utc).isoformat()
    top_manifest = {
        "stage_a_code_version": "Stage A.2c",
        "created_utc": created_utc,
        "run_id": run_id,
        "workspace_root": str(workspace_root.resolve()),
        "run_root": str(run_root.resolve()),
        "sim_name": cfg.sim_name,
        "mf6_exe": cfg.mf6_exe,
        "build_only": bool(args.build_only),
        "stage_a_uses_structured_grid": True,
        "vertical_layer_scheme": "pit_refined_10layer",
        "config": asdict(cfg),
        "observation_cells": observation_cells(cfg),
        "stages": manifests,
    }

    write_json(run_root / "stage_a_manifest.json", top_manifest)
    write_json(workspace_root / "stage_a_manifest.json", top_manifest)
    write_json(workspace_root / f"stage_a_manifest_{run_id}.json", top_manifest)
    (workspace_root / "stage_a_latest_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    (workspace_root / "stage_a_latest_run_root.txt").write_text(str(run_root.resolve()) + "\n", encoding="utf-8")

    print(f"Stage A.2c build complete. Root workspace: {workspace_root}")
    print(f"Stage A.2c run id: {run_id}")
    print(f"Run root: {run_root}")
    print(f"Latest manifest written to: {workspace_root / 'stage_a_manifest.json'}")
    print(
        "Next step: run 'stage_a_postprocess.py --workspace <root>' to create run-synchronised QC plots and summary tables."
    )


if __name__ == "__main__":
    main()
