#!/usr/bin/env python3
"""Run PEST++-IES for one or more Stage C scenario folders."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEST++-IES for Stage C continuous-K scenarios.")
    parser.add_argument("--scenario-dir", type=Path, default=None, help="Single scenario directory")
    parser.add_argument("--root", type=Path, default=None, help="Root directory containing multiple scenario folders")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario codes when using --root")
    parser.add_argument("--pestpp", type=Path, required=True, help="Path to pestpp-ies executable")
    args = parser.parse_args()

    if args.scenario_dir is None and args.root is None:
        raise ValueError("Provide either --scenario-dir or --root")

    scenario_dirs = []
    if args.scenario_dir is not None:
        scenario_dirs = [args.scenario_dir.resolve()]
    else:
        root = args.root.resolve()
        if args.scenarios:
            scenario_dirs = [(root / s.strip()) for s in args.scenarios.split(",") if s.strip()]
        else:
            scenario_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "continuous_k.pst").exists()]

    pestpp_exe = str(args.pestpp.resolve())
    for sdir in scenario_dirs:
        pst = sdir / "continuous_k.pst"
        if not pst.exists():
            raise FileNotFoundError(f"Missing PEST control file: {pst}")
        print(f"Running PEST++-IES in {sdir} ...")
        subprocess.run([pestpp_exe, pst.name], cwd=sdir, check=True)


if __name__ == "__main__":
    main()
