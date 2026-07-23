#!/usr/bin/env python3
"""Forward model for one Stage C continuous-K / pilot-point scenario.

Patched for robust S3_BUFF support.

This script is meant to live *inside one scenario folder*.  It reads:
- ``base_config.json``
- ``scenario_info.json``
- ``parvals.dat``
- ``pilot_point_groups.csv``
- ``pilot_points.csv``
- ``obs_targets.csv``

and writes:
- ``obs.dat``
- ``stage_c_manifest.json``
- run folders under ``model_run``
"""
from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import stage_a_basecase_flopy as base
import stage_a_postprocess as sap

ORIGINAL_BUILD_BASE_PROPERTY_ARRAYS = base.build_base_property_arrays


def runtime_scenario_modifiers(scenario_info: Dict[str, object]) -> Dict[str, object]:
    """Return runtime-only scenario modifiers for the active scenario.

    This first honours an explicit "scenario_modifiers" payload written into
    scenario_info.json. If that key is absent, it falls back to the local
    stage_c_common.scenario_library() mapping copied into each scenario folder.
    That fallback is what allows S3_BUFF to run even when barrier_* keys are not
    part of StageAConfig and therefore were not preserved in base_config.json.
    """
    explicit = scenario_info.get("scenario_modifiers")
    if isinstance(explicit, dict):
        return dict(explicit)
    try:
        import stage_c_common as scc
        code = str(scenario_info.get("scenario_code", "")).strip().upper()
        lib = scc.scenario_library()
        if code in lib:
            mods = lib[code].get("modifiers", {})
            if isinstance(mods, dict):
                return dict(mods)
    except Exception:
        pass
    return {}


def apply_runtime_scenario_modifiers(cfg, scenario_info: Dict[str, object]) -> None:
    for key, value in runtime_scenario_modifiers(scenario_info).items():
        setattr(cfg, key, value)


def read_parameter_table(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["parnme"].astype(str), df["parval"].astype(float)))


def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    allowed = {f.name for f in fields(base.StageAConfig) if f.init}
    return base.StageAConfig(**{k: v for k, v in raw.items() if k in allowed})


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def barrier_mask(cfg) -> np.ndarray:
    if not bool(getattr(cfg, "barrier_enabled", False)):
        return np.zeros((cfg.nrow, cfg.ncol), dtype=bool)
    xx, yy = base.meshgrid_centres(cfg)
    xmin = float(getattr(cfg, "barrier_xmin_m", 7100.0))
    xmax = float(getattr(cfg, "barrier_xmax_m", 8300.0))
    centre_y = float(getattr(cfg, "barrier_center_y_m", 4150.0))
    halfwidth = float(getattr(cfg, "barrier_halfwidth_m", 260.0))
    return (xx >= xmin) & (xx <= xmax) & (np.abs(yy - centre_y) <= halfwidth)


def idw_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, xx: np.ndarray, yy: np.ndarray, power: float = 2.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    dx = xx[None, :, :] - x[:, None, None]
    dy = yy[None, :, :] - y[:, None, None]
    dist = np.hypot(dx, dy)
    dist = np.maximum(dist, 1.0)
    w = 1.0 / np.power(dist, power)
    return np.sum(w * z[:, None, None], axis=0) / np.sum(w, axis=0)


def build_surfaces(cfg, params: Dict[str, float], pilot_points: pd.DataFrame, groups: pd.DataFrame) -> Dict[str, np.ndarray]:
    xx, yy = base.meshgrid_centres(cfg)
    surfaces: Dict[str, np.ndarray] = {}
    for _, g in groups.iterrows():
        gname = str(g["group_name"])
        pp = pilot_points[pilot_points["group_name"] == gname].copy()
        if pp.empty:
            continue
        vals = np.array([params[p] for p in pp["parnme"]], dtype=float)
        surfaces[gname] = idw_surface(pp["x_m"].to_numpy(dtype=float), pp["y_m"].to_numpy(dtype=float), vals, xx, yy)
    return surfaces


def build_base_property_arrays_stage_c(cfg, params: Dict[str, float], pilot_points: pd.DataFrame, groups: pd.DataFrame):
    kh, k33, sy, ss = ORIGINAL_BUILD_BASE_PROPERTY_ARRAYS(cfg)

    if bool(getattr(cfg, "barrier_enabled", False)):
        mask = barrier_mask(cfg)
        barrier_layers = tuple(getattr(cfg, "barrier_layers_zero_based", (1, 2, 3, 4, 5)))
        barrier_kh = float(getattr(cfg, "barrier_kh_m_per_d", 0.02))
        barrier_kvkh = float(getattr(cfg, "barrier_kvkh", 0.05))
        for k in barrier_layers:
            kh[k, mask] = barrier_kh
            k33[k, mask] = barrier_kh * barrier_kvkh

    surfaces = build_surfaces(cfg, params, pilot_points, groups)
    for _, g in groups.iterrows():
        gname = str(g["group_name"])
        if gname not in surfaces:
            continue
        surf = surfaces[gname]
        mult = np.power(10.0, surf)
        layers = [int(tok) for tok in str(g["target_layers_csv"]).split(";") if str(tok).strip() != ""]
        for k in layers:
            kh[k, :, :] *= mult
            k33[k, :, :] *= mult
    return kh, k33, sy, ss


def compute_qois(heads: pd.DataFrame, budget: pd.DataFrame) -> Dict[str, float]:
    qoi: Dict[str, float] = {}
    base_heads = heads.groupby("obs_name").first()["head_m"].to_dict()

    def max_drawdown(name: str) -> float:
        sub = heads[heads["obs_name"] == name].copy()
        dd = float(base_heads[name]) - sub["head_m"].astype(float)
        return float(dd.max())

    qoi["fcst_max_receptor_dd"] = max_drawdown("receptor")
    qoi["fcst_max_compliance_dd"] = max_drawdown("compliance")

    comp = heads[heads["obs_name"] == "compliance"].sort_values("time_years_total")
    baseline = float(base_heads["compliance"])
    thresh = baseline - 1.0
    rec = comp[comp["time_years_total"] >= 20.0]
    hit = rec[rec["head_m"] >= thresh]
    if hit.empty:
        qoi["fcst_recovery_years"] = np.nan
    else:
        qoi["fcst_recovery_years"] = float(hit.iloc[0]["time_years_total"] - 20.0)

    qoi["fcst_stage4_mean_inflow"] = float(
        budget.loc[budget["stage"] == "ops_stage_4", "mine_inflow_total_m3_per_d"].astype(float).mean()
    )
    return qoi


def extract_obs_values(obs_targets: pd.DataFrame, heads: pd.DataFrame, budget: pd.DataFrame, qois: Dict[str, float]) -> pd.DataFrame:
    out_rows: List[Dict[str, float]] = []
    for _, row in obs_targets.iterrows():
        kind = str(row["kind"])
        if kind == "head":
            sub = heads[heads["obs_name"] == row["series_name"]].sort_values("time_years_total")
            x = sub["time_years_total"].to_numpy(dtype=float)
            y = sub["head_m"].to_numpy(dtype=float)
            val = float(np.interp(float(row["time_years_total"]), x, y))
        elif kind == "budget":
            sub = budget.sort_values("time_years_total")
            x = sub["time_years_total"].to_numpy(dtype=float)
            y = sub[str(row["series_name"])].to_numpy(dtype=float)
            val = float(np.interp(float(row["time_years_total"]), x, y))
        elif kind == "forecast":
            val = float(qois[str(row["obsnme"])])
        else:
            raise KeyError(f"Unknown observation kind: {kind}")
        out_rows.append({"obsnme": row["obsnme"], "simval": val})
    return pd.DataFrame(out_rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = load_cfg(here / "base_config.json")
    with open(here / "scenario_info.json", "r", encoding="utf-8") as f:
        scenario_info = json.load(f)
    cfg.mf6_exe = scenario_info.get("mf6_exe", cfg.mf6_exe)
    apply_runtime_scenario_modifiers(cfg, scenario_info)

    params = read_parameter_table(here / "parvals.dat")
    pilot_points = pd.read_csv(here / "pilot_points.csv")
    groups = pd.read_csv(here / "pilot_point_groups.csv")
    obs_targets = pd.read_csv(here / "obs_targets.csv")

    cfg.base_recharge_mm_per_year *= float(params.get("rch_mult", 1.0))
    cfg.ghb_conductance_factor *= float(params.get("ghbcond_mult", 1.0))
    cfg.riverbed_k_m_per_d *= float(params.get("riverbedk_mult", 1.0))
    cfg.channel_kh_m_per_d *= float(params.get("channel_mult", 1.0))
    cfg.backfill_kh_m_per_d *= float(params.get("backfill_mult", 1.0))

    def patched_build_base_property_arrays(local_cfg):
        return build_base_property_arrays_stage_c(local_cfg, params, pilot_points, groups)

    base.build_base_property_arrays = patched_build_base_property_arrays

    workspace_root = here / "model_run"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_root = workspace_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)

    stage_sequence = base.build_stage_list(cfg)
    manifests = []
    previous_head: Optional[np.ndarray] = None
    start_year = 0.0
    for idx, stage in enumerate(stage_sequence):
        stage_ws = run_root / stage.name
        manifest = base.build_and_run_stage(
            cfg=cfg,
            stage=stage,
            stage_workspace=stage_ws,
            previous_head=previous_head,
            stage_index_in_sequence=idx,
            run_model=True,
        )
        manifest["start_year"] = start_year
        manifest["end_year"] = start_year + (stage.duration_years if stage.kind != "steady" else 0.0)
        manifests.append(manifest)
        previous_head = base.read_last_head(Path(manifest["head_file"]))
        if stage.kind != "steady":
            start_year += stage.duration_years

    top_manifest = {
        "run_id": "stage_c_forward",
        "scenario_code": scenario_info["scenario_code"],
        "scenario_title": scenario_info["scenario_title"],
        "scenario_modifiers": runtime_scenario_modifiers(scenario_info),
        "workspace_root": str(workspace_root.resolve()),
        "run_root": str(run_root.resolve()),
        "config": {k: v for k, v in cfg.__dict__.items()},
        "observation_cells": base.observation_cells(cfg),
        "stages": manifests,
    }
    write_json(here / "stage_c_manifest.json", top_manifest)

    heads = sap.extract_head_timeseries(top_manifest)
    budget = sap.extract_budget_timeseries(top_manifest)
    qois = compute_qois(heads, budget)
    sims = extract_obs_values(obs_targets, heads, budget, qois)
    sims[["obsnme", "simval"]].to_csv(here / "obs.dat", index=False, header=False, sep=" ")
    budget.to_csv(here / "budget_timeseries_model.csv", index=False)
    heads.to_csv(here / "heads_timeseries_model.csv", index=False)
    write_json(here / "qoi_model.json", {k: (float(v) if v == v else None) for k, v in qois.items()})


if __name__ == "__main__":
    main()
