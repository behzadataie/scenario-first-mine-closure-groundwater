#!/usr/bin/env python
"""Forecast-directed sensitivity from PESTPP-IES ensemble CSV files.

This script requires no additional MODFLOW runs when final parameter and
observation ensemble CSV files already exist. It computes:
  1. Spearman rank correlation between every adjustable parameter and forecast;
  2. parameter-group summaries;
  3. rank correlation between conditioning observations and forecasts;
  4. editable SVG figures and CSV tables.

PESTPP-IES commonly writes files such as:
  case.0.par.csv, case.1.par.csv, ...
  case.0.obs.csv, case.1.obs.csv, ...
Use the final retained iteration files. Rows are realization names and columns
are parameter/observation names.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "svg.fonttype": "none",
})


def read_ensemble(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip().str.lower()
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def infer_group(name: str) -> str:
    n = name.lower()
    if re.search(r"(^|_)(upp|upper)", n):
        return "upper pilot points"
    if re.search(r"(^|_)(main|mid)", n):
        return "main pilot points"
    if re.search(r"(^|_)(low|lower)", n) and "back" not in n:
        return "lower pilot points"
    if any(k in n for k in ["rch", "recharge"]):
        return "recharge"
    if "ghb" in n or "general_head" in n:
        return "regional support"
    if "riv" in n or "river" in n:
        return "receptor conductance"
    if any(k in n for k in ["paleo", "channel", "conn"]):
        return "palaeochannel multiplier"
    if any(k in n for k in ["backfill", "kbf", "bf_k"]):
        return "backfill conductivity"
    return "other/global"


def load_group_map(path: Path | None, parameters: list[str]) -> pd.Series:
    groups = pd.Series({p: infer_group(p) for p in parameters}, name="parameter_group")
    if path is None:
        return groups
    m = pd.read_csv(path)
    required = {"parameter", "parameter_group"}
    if not required.issubset(m.columns):
        raise ValueError(f"Group map must contain {sorted(required)}")
    for _, row in m.iterrows():
        if row["parameter"] in groups.index:
            groups.loc[row["parameter"]] = row["parameter_group"]
    return groups


def correlation_table(xdf: pd.DataFrame, ydf: pd.DataFrame, xkind: str) -> pd.DataFrame:
    rows = []
    for xname in xdf.columns:
        x = xdf[xname]
        for yname in ydf.columns:
            y = ydf[yname]
            keep = x.notna() & y.notna()
            if keep.sum() < 8 or x[keep].nunique() < 3 or y[keep].nunique() < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(x[keep], y[keep])
            rows.append({
                "source_type": xkind,
                "source_name": xname,
                "forecast": yname,
                "n": int(keep.sum()),
                "spearman_rho": rho,
                "abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                "p_value": pval,
            })
    return pd.DataFrame(rows)


def save_heatmap(data: pd.DataFrame, row_col: str, out: Path, title_note: str, top_n: int = 24) -> None:
    piv = data.pivot(index=row_col, columns="forecast", values="spearman_rho")
    score = piv.abs().max(axis=1).sort_values(ascending=False)
    piv = piv.loc[score.head(top_n).index]
    fig_h = max(3.6, 0.23 * len(piv) + 1.2)
    fig, ax = plt.subplots(figsize=(7.15, fig_h))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(piv.columns)), [c.replace("fcst_", "") for c in piv.columns], rotation=28, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Spearman rank correlation")
    fig.text(0.01, 0.01, title_note, fontsize=7.5, color="#4B555B")
    fig.subplots_adjust(left=0.28, right=0.95, top=0.98, bottom=0.18)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", required=True, type=Path, help="Final PESTPP-IES parameter ensemble CSV")
    ap.add_argument("--obs", required=True, type=Path, help="Matching observation/forecast ensemble CSV")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--forecast-cols", nargs="*", default=None,
                    help="Explicit zero-weight forecast columns. Default: columns beginning fcst_ or forecast")
    ap.add_argument("--conditioning-cols", nargs="*", default=None,
                    help="Explicit weighted observation columns. Default: all non-forecast columns")
    ap.add_argument("--group-map", type=Path, default=None,
                    help="Optional CSV with parameter,parameter_group")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    par = read_ensemble(args.par)
    obs = read_ensemble(args.obs)
    common = par.index.intersection(obs.index)
    if len(common) < 8:
        raise RuntimeError(f"Only {len(common)} common realizations; check row names")
    par = par.loc[common]
    obs = obs.loc[common]

    if args.forecast_cols:
        fcols = [c for c in args.forecast_cols if c in obs.columns]
    else:
        fcols = [c for c in obs.columns if c.lower().startswith(("fcst_", "forecast"))]
    if not fcols:
        raise RuntimeError("No forecast columns found. Supply --forecast-cols.")
    if args.conditioning_cols:
        ocols = [c for c in args.conditioning_cols if c in obs.columns]
    else:
        ocols = [c for c in obs.columns if c not in fcols]

    groups = load_group_map(args.group_map, list(par.columns))
    pcorr = correlation_table(par, obs[fcols], "parameter")
    pcorr["parameter_group"] = pcorr["source_name"].map(groups)
    pcorr.insert(0, "scenario", args.scenario)
    pcorr.to_csv(args.output_dir / f"{args.scenario}_parameter_forecast_sensitivity.csv", index=False)

    group = (pcorr.groupby(["scenario", "parameter_group", "forecast"], dropna=False)
             .agg(max_abs_rho=("abs_rho", "max"), median_abs_rho=("abs_rho", "median"),
                  n_parameters=("source_name", "nunique"))
             .reset_index())
    group.to_csv(args.output_dir / f"{args.scenario}_parameter_group_forecast_sensitivity.csv", index=False)

    if ocols:
        ocorr = correlation_table(obs[ocols], obs[fcols], "conditioning observation")
        ocorr.insert(0, "scenario", args.scenario)
        ocorr.to_csv(args.output_dir / f"{args.scenario}_observation_forecast_association.csv", index=False)
    else:
        ocorr = pd.DataFrame()

    save_heatmap(pcorr, "source_name", args.output_dir / f"{args.scenario}_parameter_forecast_sensitivity",
                 "Rank correlations diagnose ensemble association, not causal derivatives. Confirm important controls with targeted perturbation runs.")
    if not ocorr.empty:
        save_heatmap(ocorr, "source_name", args.output_dir / f"{args.scenario}_observation_forecast_association",
                     "Observation–forecast association indicates which simulated equivalents carry information about each forecast.", top_n=20)

    print(f"Aligned realizations: {len(common)}")
    print(f"Parameters: {len(par.columns)}; forecasts: {len(fcols)}; conditioning outputs: {len(ocols)}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
