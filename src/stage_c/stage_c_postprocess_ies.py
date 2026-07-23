#!/usr/bin/env python3
"""Best-effort post-processing for Stage C PEST++-IES outputs.

This script is intentionally permissive because PEST++ output filenames can vary
slightly by version and options. It looks for common CSV outputs and extracts the
forecast/QoI observations if they are available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def newest_matching(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process Stage C PEST++-IES outputs.")
    parser.add_argument("--scenario-dir", type=Path, required=True, help="Scenario directory")
    args = parser.parse_args()

    sdir = args.scenario_dir.resolve()
    pst = sdir / "continuous_k.pst"
    if not pst.exists():
        raise FileNotFoundError(f"Missing {pst}")

    phi_file = newest_matching(sdir, "continuous_k.phi*.csv")
    obs_files = sorted(sdir.glob("continuous_k.*.obs.csv"))

    summary = {}
    if phi_file is not None:
        try:
            phi = pd.read_csv(phi_file)
            summary["phi_file"] = str(phi_file)
            summary["phi_rows"] = int(phi.shape[0])
        except Exception:
            summary["phi_file"] = str(phi_file)

    if obs_files:
        latest = obs_files[-1]
        try:
            obs = pd.read_csv(latest)
            forecast_cols = [c for c in obs.columns if c.lower().startswith("fcst_")]
            if forecast_cols:
                obs[forecast_cols].to_csv(sdir / "stage_c_forecasts_latest.csv", index=False)
                summary["forecast_file"] = str((sdir / "stage_c_forecasts_latest.csv").resolve())
            else:
                summary["latest_obs_file"] = str(latest)
        except Exception:
            summary["latest_obs_file"] = str(latest)

    pd.DataFrame([summary]).to_csv(sdir / "stage_c_postprocess_summary.csv", index=False)
    print(f"Stage C post-process summary written to: {sdir / 'stage_c_postprocess_summary.csv'}")


if __name__ == "__main__":
    main()
