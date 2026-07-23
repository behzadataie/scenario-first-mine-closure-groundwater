#!/usr/bin/env python3
"""Build a PEST++-IES control file and prior ensemble for one Stage C scenario."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def run_forward_model(scenario_dir: Path, python_exe: str) -> None:
    subprocess.run([python_exe, "forward_model.py"], cwd=scenario_dir, check=True)


def build_prior_ensemble(meta: pd.DataFrame, nreals: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    parnames = meta["parnme"].tolist()
    base = meta.set_index("parnme")["parval1"].astype(float)
    lb = meta.set_index("parnme")["parlbnd"].astype(float)
    ub = meta.set_index("parnme")["parubnd"].astype(float)

    draws = pd.DataFrame(index=[f"real_{i:04d}" for i in range(1, nreals + 1)], columns=parnames, dtype=float)

    # Independent global multipliers.
    gmeta = meta[meta["pargp"] == "GLOBAL"].copy()
    for _, row in gmeta.iterrows():
        vals = rng.normal(float(row["parval1"]), float(row["sigma"]), size=nreals)
        vals = np.clip(vals, float(row["parlbnd"]), float(row["parubnd"]))
        draws.loc[:, row["parnme"]] = vals

    # Spatially correlated pilot-point log10 multipliers, per group.
    for pargp, grp in meta[meta["pargp"] != "GLOBAL"].groupby("pargp"):
        grp = grp.copy().reset_index(drop=True)
        coords = grp[["x_m", "y_m"]].to_numpy(dtype=float)
        sigma = float(grp["sigma"].iloc[0])
        a = float(grp["corr_len_m"].iloc[0])
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        cov = (sigma ** 2) * np.exp(-d / max(a, 1.0))
        cov += np.eye(cov.shape[0]) * 1.0e-8
        vals = rng.multivariate_normal(mean=grp["parval1"].to_numpy(dtype=float), cov=cov, size=nreals)
        vals = np.clip(vals, grp["parlbnd"].to_numpy(dtype=float), grp["parubnd"].to_numpy(dtype=float))
        draws.loc[:, grp["parnme"].tolist()] = vals

    base_row = base.reindex(parnames).to_frame().T
    base_row.index = ["base"]
    return pd.concat([base_row, draws], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage C PEST++-IES setup for one scenario.")
    parser.add_argument("--scenario-dir", type=Path, required=True, help="Scenario folder created by stage_c_setup_continuous_k.py")
    parser.add_argument("--num-prior", type=int, default=300, help="Number of prior realizations to draw")
    parser.add_argument("--num-posterior", type=int, default=100, help="Target number of posterior realizations for IES")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to run forward_model.py")
    parser.add_argument("--skip-forward", action="store_true", help="Do not run the base forward model before building the PST")
    args = parser.parse_args()

    try:
        import pyemu  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyEMU must be installed before running stage_c_build_pst.py") from exc

    sdir = args.scenario_dir.resolve()
    meta = pd.read_csv(sdir / "parameters_metadata.csv")
    obs_targets = pd.read_csv(sdir / "obs_targets.csv")

    if not args.skip_forward:
        run_forward_model(sdir, args.python_exe)

    tpl_files = ["parvals.dat.tpl"]
    in_files = ["parvals.dat"]
    ins_files = ["obs.dat.ins"]
    out_files = ["obs.dat"]
    pst = pyemu.Pst.from_io_files(tpl_files, in_files, ins_files, out_files, pst_path='.')
    pst.model_command = [f'"{args.python_exe}" forward_model.py']

    pmeta = meta.set_index("parnme")
    pdata = pst.parameter_data
    for parnme in pdata.index:
        pdata.loc[parnme, "parval1"] = float(pmeta.loc[parnme, "parval1"])
        pdata.loc[parnme, "parlbnd"] = float(pmeta.loc[parnme, "parlbnd"])
        pdata.loc[parnme, "parubnd"] = float(pmeta.loc[parnme, "parubnd"])
        pdata.loc[parnme, "pargp"] = str(pmeta.loc[parnme, "pargp"])
        pdata.loc[parnme, "partrans"] = "none"

    ometa = obs_targets.set_index("obsnme")
    odata = pst.observation_data
    for oname in odata.index:
        odata.loc[oname, "obsval"] = float(ometa.loc[oname, "obsval"])
        odata.loc[oname, "weight"] = float(ometa.loc[oname, "weight"])
        odata.loc[oname, "obgnme"] = str(ometa.loc[oname, "obgnme"])

    pst.control_data.noptmax = 3
    pst.pestpp_options["ies_num_reals"] = str(args.num_posterior)
    pst.pestpp_options["ies_par_en"] = "prior_parensemble.csv"
    pst.pestpp_options["ies_save_binary"] = "true"
    pst.pestpp_options["ies_include_base"] = "true"
    pst.write(sdir / "continuous_k.pst")

    pe = build_prior_ensemble(meta, nreals=args.num_prior, seed=args.seed)
    pe.to_csv(sdir / "prior_parensemble.csv")

    with open(sdir / "ies_setup_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario_dir": str(sdir),
                "n_parameters": int(meta.shape[0]),
                "n_observations": int(obs_targets.shape[0]),
                "num_prior": int(args.num_prior),
                "num_posterior": int(args.num_posterior),
                "python_exe": args.python_exe,
            },
            f,
            indent=2,
        )

    print(f"Stage C PST written to: {sdir / 'continuous_k.pst'}")
    print(f"Prior ensemble written to: {sdir / 'prior_parensemble.csv'}")


if __name__ == "__main__":
    main()
