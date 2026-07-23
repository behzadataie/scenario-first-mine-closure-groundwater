#!/usr/bin/env python3
"""Shared helpers for the Stage C continuous-K / pilot-point workflow."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PRIMARY_DEFAULT = ["S0_BASE", "S2_CONN", "S6_UPRISK"]
OPTIONAL_DEFAULT = ["S3_BUFF"]


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p.resolve()
    return None


def search_upwards(start: Path, names: Sequence[str], max_levels: int = 4) -> Optional[Path]:
    start = start.resolve()
    for base in [start] + list(start.parents)[:max_levels]:
        for name in names:
            cand = base / name
            if cand.exists():
                return cand.resolve()
    return None


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def scenario_library() -> Dict[str, Dict[str, object]]:
    return {
        "S0_BASE": {
            "title": "Base supportive backfill",
            "family": "reference",
            "description": "Frozen Stage A.2c baseline with moderate support, moderate transmissive architecture, and the supportive porous-backfill closure base case.",
            "modifiers": {},
        },
        "S1_DRY": {
            "title": "Dry-support case",
            "family": "regional support",
            "description": "Lower diffuse recharge, slightly lower boundary heads, and slightly lower receptor-stage support.",
            "modifiers": {
                "base_recharge_mm_per_year": 2.0,
                "west_boundary_head_m": 51.0,
                "east_boundary_head_m": 39.5,
                "river_stage_m": 41.0,
                "ghb_conductance_factor": 0.8,
            },
        },
        "S2_CONN": {
            "title": "Connected transmissive pathway",
            "family": "architecture",
            "description": "A stronger and slightly wider transmissive corridor that connects the pit toward the receptor reach more directly.",
            "modifiers": {
                "channel_kh_m_per_d": 35.0,
                "channel_halfwidth_m": 180.0,
                "channel_xmax_m": 10500.0,
                "channel_amplitude_m": 100.0,
                "channel_layers_zero_based": [2, 3, 4, 5],
                "riverbed_k_m_per_d": 0.10,
            },
        },
        "S3_BUFF": {
            "title": "Buffered / barrier pathway",
            "family": "architecture",
            "description": "A low-K barrier corridor intercepts the pit-to-receptor path and the receptor reach is slightly less hydraulically open.",
            "modifiers": {
                "barrier_enabled": True,
                "barrier_xmin_m": 7100.0,
                "barrier_xmax_m": 8300.0,
                "barrier_center_y_m": 4150.0,
                "barrier_halfwidth_m": 260.0,
                "barrier_layers_zero_based": [1, 2, 3, 4, 5],
                "barrier_kh_m_per_d": 0.02,
                "barrier_kvkh": 0.05,
                "riverbed_k_m_per_d": 0.02,
            },
        },
        "S4_LOWKBF": {
            "title": "Low-K backfill closure",
            "family": "closure",
            "description": "A lower-permeability, lower-Sy backfill intended to test slow hydraulic recovery after closure.",
            "modifiers": {
                "backfill_kh_m_per_d": 0.05,
                "backfill_kvkh": 0.10,
                "backfill_sy": 0.04,
                "backfill_ss_per_m": 2.0e-6,
            },
        },
        "S5_HIKBF": {
            "title": "Coarse / high-K backfill closure",
            "family": "closure",
            "description": "A more permeable backfill representing coarser spoil or a more conductive closure geometry than the base case.",
            "modifiers": {
                "backfill_kh_m_per_d": 1.50,
                "backfill_kvkh": 0.25,
                "backfill_sy": 0.10,
                "backfill_ss_per_m": 8.0e-6,
            },
        },
        "S6_UPRISK": {
            "title": "Upper-risk combined case",
            "family": "combined",
            "description": "Dry support + connected transmissive pathway + low-K backfill to test a combined upper-risk deterministic envelope.",
            "modifiers": {
                "base_recharge_mm_per_year": 2.0,
                "west_boundary_head_m": 51.0,
                "east_boundary_head_m": 39.5,
                "river_stage_m": 41.0,
                "ghb_conductance_factor": 0.8,
                "channel_kh_m_per_d": 35.0,
                "channel_halfwidth_m": 180.0,
                "channel_xmax_m": 10500.0,
                "channel_amplitude_m": 100.0,
                "channel_layers_zero_based": [2, 3, 4, 5],
                "riverbed_k_m_per_d": 0.10,
                "backfill_kh_m_per_d": 0.05,
                "backfill_kvkh": 0.10,
                "backfill_sy": 0.04,
                "backfill_ss_per_m": 2.0e-6,
            },
        },
    }


def apply_scenario_modifiers(cfg, scenario_code: str) -> None:
    lib = scenario_library()
    if scenario_code not in lib:
        raise KeyError(f"Unknown scenario code: {scenario_code}")
    for k, v in lib[scenario_code]["modifiers"].items():
        setattr(cfg, k, v)


def load_stage_a_config(stage_a_manifest: Path, stage_a_module):
    with open(stage_a_manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_cfg = payload["config"]
    allowed = {f.name for f in fields(stage_a_module.StageAConfig) if f.init}
    filtered = {k: v for k, v in raw_cfg.items() if k in allowed}
    cfg = stage_a_module.StageAConfig(**filtered)
    return cfg, payload


def clone_stage_a_config(cfg, stage_a_module):
    allowed = {f.name for f in fields(stage_a_module.StageAConfig) if f.init}
    return stage_a_module.StageAConfig(**{k: v for k, v in cfg.__dict__.items() if k in allowed})


def parse_scenarios(raw: Optional[str]) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return list(PRIMARY_DEFAULT)
    return [s.strip().upper() for s in str(raw).split(",") if s.strip()]


def interp_series(x: np.ndarray, y: np.ndarray, xi: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot interpolate an empty series")
    if xi <= x.min():
        return float(y[np.argmin(x)])
    if xi >= x.max():
        return float(y[np.argmax(x)])
    return float(np.interp(xi, x, y))


def make_group_definitions() -> pd.DataFrame:
    rows = [
        {"group_name": "PP_UPPER", "target_layers_csv": "0", "spacing_m": 2400.0, "margin_m": 1200.0, "sigma": 0.35, "corr_len_m": 2200.0, "parlbnd": -1.0, "parubnd": 1.0},
        {"group_name": "PP_MAIN", "target_layers_csv": "1;2;3;4", "spacing_m": 1600.0, "margin_m": 800.0, "sigma": 0.45, "corr_len_m": 1800.0, "parlbnd": -1.2, "parubnd": 1.2},
        {"group_name": "PP_LOWER", "target_layers_csv": "5;6;7", "spacing_m": 2200.0, "margin_m": 1100.0, "sigma": 0.35, "corr_len_m": 2600.0, "parlbnd": -1.0, "parubnd": 1.0},
    ]
    return pd.DataFrame(rows)


def regular_points(length_x: float, length_y: float, spacing: float, margin: float) -> List[Tuple[float, float]]:
    xs = np.arange(margin, length_x - margin + 1.0, spacing, dtype=float)
    ys = np.arange(margin, length_y - margin + 1.0, spacing, dtype=float)
    return [(float(x), float(y)) for y in ys for x in xs]


def build_pilot_points(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = make_group_definitions()
    pp_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []
    for _, g in groups.iterrows():
        pts = regular_points(cfg.length_x_m, cfg.length_y_m, float(g.spacing_m), float(g.margin_m))
        for i, (x, y) in enumerate(pts, start=1):
            parnme = f"{g.group_name.lower()}_{i:03d}"
            pp_rows.append(
                {
                    "parnme": parnme,
                    "group_name": g.group_name,
                    "x_m": x,
                    "y_m": y,
                    "parval1": 0.0,
                    "parlbnd": float(g.parlbnd),
                    "parubnd": float(g.parubnd),
                    "sigma": float(g.sigma),
                    "corr_len_m": float(g.corr_len_m),
                }
            )
            meta_rows.append(
                {
                    "parnme": parnme,
                    "pargp": g.group_name,
                    "par_type": "pilot_point_log10mult",
                    "parval1": 0.0,
                    "parlbnd": float(g.parlbnd),
                    "parubnd": float(g.parubnd),
                    "sigma": float(g.sigma),
                    "corr_len_m": float(g.corr_len_m),
                    "x_m": x,
                    "y_m": y,
                    "target_layers_csv": g.target_layers_csv,
                }
            )
    global_rows = [
        {"parnme": "rch_mult", "pargp": "GLOBAL", "par_type": "multiplier", "parval1": 1.0, "parlbnd": 0.5, "parubnd": 1.5, "sigma": 0.15, "corr_len_m": np.nan, "x_m": np.nan, "y_m": np.nan, "target_layers_csv": ""},
        {"parnme": "ghbcond_mult", "pargp": "GLOBAL", "par_type": "multiplier", "parval1": 1.0, "parlbnd": 0.5, "parubnd": 1.5, "sigma": 0.15, "corr_len_m": np.nan, "x_m": np.nan, "y_m": np.nan, "target_layers_csv": ""},
        {"parnme": "riverbedk_mult", "pargp": "GLOBAL", "par_type": "multiplier", "parval1": 1.0, "parlbnd": 0.5, "parubnd": 2.0, "sigma": 0.20, "corr_len_m": np.nan, "x_m": np.nan, "y_m": np.nan, "target_layers_csv": ""},
        {"parnme": "channel_mult", "pargp": "GLOBAL", "par_type": "multiplier", "parval1": 1.0, "parlbnd": 0.5, "parubnd": 2.0, "sigma": 0.20, "corr_len_m": np.nan, "x_m": np.nan, "y_m": np.nan, "target_layers_csv": ""},
        {"parnme": "backfill_mult", "pargp": "GLOBAL", "par_type": "multiplier", "parval1": 1.0, "parlbnd": 0.3, "parubnd": 3.0, "sigma": 0.25, "corr_len_m": np.nan, "x_m": np.nan, "y_m": np.nan, "target_layers_csv": ""},
    ]
    meta_rows.extend(global_rows)
    return pd.DataFrame(pp_rows), pd.DataFrame(meta_rows)


def build_obs_targets_for_scenario(
    scenario_code: str,
    stage_b_heads: pd.DataFrame,
    stage_b_budget: pd.DataFrame,
    stage_b_qoi: pd.DataFrame,
) -> pd.DataFrame:
    heads = stage_b_heads[stage_b_heads["scenario_code"] == scenario_code].copy()
    budget = stage_b_budget[stage_b_budget["scenario_code"] == scenario_code].copy()
    qoi_row = stage_b_qoi.loc[stage_b_qoi["scenario_code"] == scenario_code].iloc[0]

    rows: List[Dict[str, object]] = []
    head_schedule = {
        "compliance": [5, 10, 15, 20, 25, 50, 120],
        "receptor": [5, 10, 20, 50, 120],
        "landholder": [20, 120],
    }
    for obs_name, years in head_schedule.items():
        sub = heads[heads["obs_name"] == obs_name].sort_values("time_years_total")
        x = sub["time_years_total"].to_numpy(dtype=float)
        y = sub["head_m"].to_numpy(dtype=float)
        for yr in years:
            rows.append(
                {
                    "obsnme": f"h_{obs_name}_{int(yr):03d}y",
                    "kind": "head",
                    "series_name": obs_name,
                    "time_years_total": float(yr),
                    "obsval": interp_series(x, y, float(yr)),
                    "weight": 4.0,
                    "obgnme": f"head_{obs_name}",
                }
            )

    inflow_years = [5, 10, 15, 20]
    sub = budget.sort_values("time_years_total")
    x = sub["time_years_total"].to_numpy(dtype=float)
    y = sub["mine_inflow_total_m3_per_d"].to_numpy(dtype=float)
    for yr in inflow_years:
        obsval = interp_series(x, y, float(yr))
        sigma = max(2000.0, 0.10 * abs(obsval))
        rows.append(
            {
                "obsnme": f"q_mineinflow_{int(yr):03d}y",
                "kind": "budget",
                "series_name": "mine_inflow_total_m3_per_d",
                "time_years_total": float(yr),
                "obsval": obsval,
                "weight": 1.0 / sigma,
                "obgnme": "budget_mineinflow",
            }
        )

    forecast_rows = [
        ("fcst_max_receptor_dd", float(qoi_row["max_receptor_drawdown_m"]), "forecast_receptor"),
        ("fcst_max_compliance_dd", float(qoi_row["max_compliance_drawdown_m"]), "forecast_compliance"),
        ("fcst_recovery_years", float(qoi_row["compliance_recovery_years_after_closure"]), "forecast_recovery"),
        ("fcst_stage4_mean_inflow", float(qoi_row["mine_inflow_total_m3_per_d_mean"]), "forecast_inflow"),
    ]
    for name, obsval, grp in forecast_rows:
        rows.append(
            {
                "obsnme": name,
                "kind": "forecast",
                "series_name": name,
                "time_years_total": np.nan,
                "obsval": obsval,
                "weight": 0.0,
                "obgnme": grp,
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)


def make_parvals_df(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[["parnme", "parval1"]].rename(columns={"parval1": "parval"}).copy()


def write_template_and_instruction(meta: pd.DataFrame, obs_targets: pd.DataFrame, outdir: Path) -> None:
    parvals = make_parvals_df(meta)
    parvals.to_csv(outdir / "parvals.dat", index=False)
    with open(outdir / "parvals.dat.tpl", "w", encoding="utf-8") as f:
        f.write("ptf ~\n")
        f.write("parnme,parval\n")
        for _, row in parvals.iterrows():
            token = row["parnme"]
            f.write(f"{token},~   {token:<20s}   ~\n")

    obs_targets[["obsnme", "obsval"]].to_csv(outdir / "obs.dat", index=False, header=False, sep=" ")
    with open(outdir / "obs.dat.ins", "w", encoding="utf-8") as f:
        f.write("pif ~\n")
        for _, row in obs_targets.iterrows():
            f.write(f"l1 w !{row['obsnme']}!\n")
