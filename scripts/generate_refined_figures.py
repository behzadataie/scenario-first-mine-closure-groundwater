#!/usr/bin/env python3
"""Regenerate the data-driven publication figures used in the revised paper.

The conceptual workflow, pit-operation/closure, retained-scenario cross-section,
and graphical-abstract SVGs are maintained as editable vector masters. This
script regenerates the figures whose content is calculated directly from the
repository's processed tables and compact saved ensembles:

- main Figure 2: stacked benchmark geometry;
- main Figure 4: deterministic screening values;
- main Figure 5: parameter-group/forecast rank associations;
- SI Figure S4: common prior parameter distributions;
- SI Figure S5: successful iteration-0 prior forecast distributions;
- SI Figure S8: strongest observation/forecast linkages.

Run from any directory with:
    python scripts/generate_refined_figures.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SCENARIO_OUTPUTS = ROOT / "data" / "scenario_outputs"
MAIN = ROOT / "figures" / "main"
SI = ROOT / "figures" / "supporting_information"
PREVIEW = ROOT / "outputs" / "figure_previews"
for folder in (MAIN, SI, PREVIEW):
    folder.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

COLORS = {
    "S0_BASE": "#4C78A8",
    "S2_CONN": "#1591B3",
    "S3_BUFF": "#CC79A7",
    "S6_UPRISK": "#D55E00",
    "screened": "#B8B8B8",
    "threshold": "#C63D4F",
}
SCENARIOS = ["S0_BASE", "S1_DRY", "S2_CONN", "S3_BUFF", "S4_LOWKBF", "S5_HIKBF", "S6_UPRISK"]
RETAINED = {"S0_BASE", "S2_CONN", "S3_BUFF", "S6_UPRISK"}


def save(fig: plt.Figure, svg_path: Path) -> None:
    """Save an editable SVG master and a 300-dpi PNG preview."""
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PREVIEW / f"{svg_path.stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_stacked_benchmark_geometry() -> None:
    fig = plt.figure(figsize=(7.25, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.7, 1.45],
        height_ratios=[1.0, 1.25],
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.98,
        wspace=0.16,
        hspace=0.24,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_l = fig.add_subplot(gs[:, 1])
    ax_l.axis("off")

    ax_a.set_xlim(0, 12)
    ax_a.set_ylim(0, 8)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("x distance (km)")
    ax_a.set_ylabel("y distance (km)")
    ax_a.set_xticks(np.arange(0, 13, 2))
    ax_a.set_yticks(np.arange(0, 9, 1))
    ax_a.grid(True, color="#D9DEE3", linewidth=0.6, zorder=0)
    ax_a.add_patch(Rectangle((0, 0), 12, 8, facecolor="#FAFBFC", edgecolor="#607D8B", linewidth=1.1, zorder=-3))
    ax_a.plot([0, 0], [0, 8], color="#6BAED6", linewidth=2.2, zorder=1)
    ax_a.plot([12, 12], [0, 8], color="#6BAED6", linewidth=2.2, zorder=1)
    ax_a.plot([0, 12], [0, 0], color="#9E9E9E", linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
    ax_a.plot([0, 12], [8, 8], color="#9E9E9E", linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
    ax_a.text(0.15, 7.72, "north no-flow", fontsize=7.8, color="#606970", va="top")
    ax_a.text(0.15, 0.18, "west GHB", fontsize=7.8, color="#4F91C7", va="bottom")
    ax_a.text(11.85, 0.18, "east GHB", fontsize=7.8, color="#4F91C7", va="bottom", ha="right")
    ax_a.add_patch(Rectangle((3, 2), 6, 4, facecolor="#F5D76E", edgecolor="#C59A22", alpha=0.18, linewidth=1.0, linestyle="--", zorder=1))
    ax_a.text(3.15, 5.78, "6 km x 4 km stochastic window", fontsize=7.7, color="#8A6A00", va="top")

    x = np.linspace(2.8, 10.1, 300)
    y = 4.05 + 0.16 * np.sin((x - 3.0) * np.pi / 2.8)
    ax_a.plot(x, y, color="#45A9C5", linewidth=14, alpha=0.23, solid_capstyle="round", zorder=2)
    ax_a.plot(x, y, color="#1184A7", linewidth=2.8, solid_capstyle="round", zorder=3)
    xr = np.linspace(7.8, 10.4, 120)
    yr = 4.20 + 0.12 * np.sin((xr - 7.8) * np.pi / 1.25)
    ax_a.plot(xr, yr, color="#1B9E77", linewidth=2.6, zorder=4)
    ax_a.add_patch(Rectangle((7.05, 2.55), 0.55, 3.0, facecolor="#8B6DAA", edgecolor="#5F4A78", alpha=0.75, hatch="////", linewidth=0.9, zorder=4))

    bands = [
        (3.7, 3.8, 4.1, 4.9),
        (3.8, 3.9, 3.9, 5.1),
        (3.9, 4.0, 3.8, 5.2),
        (4.0, 4.1, 3.8, 5.2),
        (4.1, 4.2, 3.9, 5.1),
        (4.2, 4.3, 4.1, 4.9),
    ]
    for y0, y1, x0, x1 in bands:
        ax_a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#5B4A3E", edgecolor="#5B4A3E", zorder=5))
    ax_a.add_patch(Rectangle((3.78, 3.65), 1.44, 0.70, facecolor="none", edgecolor="#4A3B31", linewidth=1.0, zorder=6))

    points = {"PM": (5.3, 4.0), "C": (6.5, 4.0), "R": (8.8, 4.2), "L": (9.3, 4.1)}
    offsets = {"PM": (-0.20, 0.30), "C": (0.0, -0.35), "R": (-0.12, 0.36), "L": (0.32, -0.30)}
    for key, (px, py) in points.items():
        ax_a.scatter(px, py, s=26, facecolor="#252525", edgecolor="white", linewidth=0.6, zorder=8)
        dx, dy = offsets[key]
        ax_a.annotate(
            key,
            xy=(px, py),
            xytext=(px + dx, py + dy),
            ha="center",
            va="center",
            fontsize=8.0,
            arrowprops=dict(arrowstyle="-", color="#424242", lw=0.7, shrinkA=2, shrinkB=3),
            zorder=9,
        )
    ax_a.plot([0.7, 2.7], [0.58, 0.58], color="#222", linewidth=2.0)
    ax_a.plot([0.7, 0.7], [0.48, 0.68], color="#222", linewidth=1.4)
    ax_a.plot([2.7, 2.7], [0.48, 0.68], color="#222", linewidth=1.4)
    ax_a.text(1.7, 0.76, "2 km", ha="center", va="bottom", fontsize=8.2)
    ax_a.text(-0.11, 1.03, "A", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="bottom")

    ax_b.set_xlim(0, 12)
    ax_b.set_ylim(-100, 60)
    ax_b.set_xlabel("distance along pit-receptor corridor (km)")
    ax_b.set_ylabel("elevation (m AHD; conceptual)")
    ax_b.set_xticks(np.arange(0, 13, 2))
    ax_b.set_yticks(np.arange(-100, 61, 20))
    ax_b.grid(True, color="white", linewidth=0.7, zorder=1)
    layers = [
        (60, 40, "#DCC8A5", "L1"),
        (40, 25, "#C9D9D7", "L2"),
        (25, 10, "#EAD48B", "L3"),
        (10, -5, "#EAD48B", "L4"),
        (-5, -25, "#EAD48B", "L5"),
        (-25, -45, "#B8C1C9", "L6"),
        (-45, -55, "#B8C1C9", "L7"),
        (-55, -70, "#B8C1C9", "L8"),
        (-70, -85, "#8F979F", "L9"),
        (-85, -100, "#8F979F", "L10"),
    ]
    for top, bottom, color, label in layers:
        ax_b.axhspan(bottom, top, color=color, zorder=0)
        ax_b.text(0.18, (top + bottom) / 2, label, ha="left", va="center", fontsize=7.6, color="#4C5358")
    px = np.array([3.1, 4.2, 6.1, 8.9, 9.6, 9.6, 8.7, 6.0, 4.0, 3.1])
    py = np.array([22, 21, 20, 22, 20, -20, -22, -24, -20, -18])
    ax_b.add_patch(Polygon(np.c_[px, py], closed=True, facecolor="#52AFC2", edgecolor="#15809B", alpha=0.45, linewidth=1.0, zorder=2))
    ax_b.add_patch(Rectangle((7.05, -58), 0.60, 100, facecolor="#8B6DAA", edgecolor="#5F4A78", alpha=0.74, hatch="////", linewidth=0.9, zorder=3))
    pit = np.array(
        [
            [3.8, 60], [5.2, 60], [5.2, 35], [5.1, 35], [5.1, 10], [5.0, 10], [5.0, -20], [4.9, -20], [4.9, -50],
            [4.1, -50], [4.1, -20], [4.0, -20], [4.0, 10], [3.9, 10], [3.9, 35], [3.8, 35],
        ]
    )
    ax_b.add_patch(Polygon(pit, closed=True, facecolor="#B98E55", edgecolor="#4B3B2F", linewidth=1.1, zorder=5))
    ax_b.text(4.5, -1, "pit / backfill", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white", zorder=6)
    xw = np.linspace(0, 12, 400)
    pre = 56 - 0.83 * xw
    ops = pre - 30 * np.exp(-((xw - 4.55) / 1.15) ** 2)
    rec = pre - 12 * np.exp(-((xw - 5.0) / 1.8) ** 2)
    ax_b.plot(xw, pre, color="#006D7C", lw=1.5, zorder=7)
    ax_b.plot(xw, ops, color="#D6452F", lw=1.5, ls="--", zorder=7)
    ax_b.plot(xw, rec, color="#6A2C91", lw=1.5, ls=":", zorder=7)
    bores = {"PM": (5.3, 10, 25), "C": (6.5, -25, -5), "R": (8.8, 25, 40), "L": (9.3, -25, -5)}
    for key, (bx, bottom, top) in bores.items():
        surface = float(np.interp(bx, xw, pre))
        ax_b.plot([bx, bx], [-55, surface + 2], color="#353535", lw=0.9, zorder=8)
        ax_b.plot([bx, bx], [bottom, top], color="#F0C949", lw=4.0, solid_capstyle="butt", zorder=9)
        ax_b.scatter([bx], [surface + 2.4], marker="v", s=22, color="#2B2B2B", zorder=10)
        ypos = {"PM": 29, "C": 5, "R": 47, "L": 5}[key]
        ax_b.text(bx, ypos, key, ha="center", va="bottom", fontsize=8.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.4), zorder=10)
    ax_b.text(-0.11, 1.02, "B", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="bottom")

    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.text(0.02, 0.985, "Legend", fontsize=11, fontweight="bold", va="top")
    handles = [
        Patch(facecolor="#F5D76E", edgecolor="#C59A22", alpha=0.30, label="stochastic facies window"),
        Patch(facecolor="#5B4A3E", edgecolor="#5B4A3E", label="grid-based pit footprint / backfill"),
        Line2D([0], [0], color="#1184A7", lw=3, label="palaeochannel pathway"),
        Patch(facecolor="#8B6DAA", edgecolor="#5F4A78", hatch="////", alpha=0.75, label="low-K barrier corridor"),
        Line2D([0], [0], color="#1B9E77", lw=3, label="receptor reach"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#252525", markeredgecolor="white", markersize=6, label="observation location"),
        Line2D([0], [0], color="#F0C949", lw=4, label="screen interval"),
        Line2D([0], [0], color="#006D7C", lw=1.6, label="predevelopment head"),
        Line2D([0], [0], color="#D6452F", lw=1.6, ls="--", label="end-of-operations head"),
        Line2D([0], [0], color="#6A2C91", lw=1.6, ls=":", label="partial-recovery head"),
    ]
    ax_l.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, handlelength=2.1, labelspacing=0.80, borderaxespad=0)
    ax_l.text(0.02, 0.26, "Observation codes", fontsize=9.2, fontweight="bold", va="top")
    ax_l.text(0.02, 0.225, "PM  pit margin\nC    compliance\nR    receptor\nL    landholder", fontsize=8.5, va="top", linespacing=1.35)
    ax_l.text(0.02, 0.08, "Plan and section use the\nreported 100 m grid, pit footprint,\ncoordinates, layers, and screens.\nThe section is vertically exaggerated.", fontsize=7.7, color="#626B70", va="top", linespacing=1.25)
    save(fig, MAIN / "Fig02_benchmark_geometry.svg")


def draw_deterministic_screening() -> None:
    data = pd.read_csv(PROCESSED / "deterministic_screening_values.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.7), constrained_layout=False)
    plt.subplots_adjust(left=0.11, right=0.985, bottom=0.19, top=0.97, wspace=0.27, hspace=0.38)
    specs = [
        ("receptor", "maximum receptor drawdown (m)", 2.0, (0, 2.45), "A", "{:.2f}"),
        ("compliance", "maximum compliance drawdown (m)", 30.0, (0, 32.5), "B", "{:.2f}"),
        ("recovery", "recovery to within 1 m (years)", None, (0, 38), "C", "{:.0f}"),
        ("inflow", "mean Stage 4 inflow (10³ m³/d)", 50.0, (0, 57), "D", "{:.1f}"),
    ]
    x = np.arange(len(data))
    colors = [COLORS[s] if s in RETAINED else COLORS["screened"] for s in data.scenario]
    for ax, (key, ylabel, threshold, ylim, letter, fmt) in zip(axes.ravel(), specs):
        bars = ax.bar(x, data[key], color=colors, edgecolor="white", linewidth=0.7, width=0.72, zorder=2)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(data.scenario, rotation=38, ha="right", fontsize=7.5)
        ax.grid(axis="y", color="#E0E3E6", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
        if threshold is not None:
            ax.axhline(threshold, color=COLORS["threshold"], ls="--", lw=1.0, zorder=1)
            ax.text(0.02, 0.98, f"criterion = {threshold:g}", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=COLORS["threshold"], bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0))
        for rect, val in zip(bars, data[key]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02 * (ylim[1] - ylim[0]), fmt.format(val), ha="center", va="bottom", fontsize=7.0)
    handles = [
        Patch(facecolor=COLORS["screened"], edgecolor="none", label="screened only"),
        Patch(facecolor=COLORS["S0_BASE"], edgecolor="none", label="S0_BASE retained"),
        Patch(facecolor=COLORS["S2_CONN"], edgecolor="none", label="S2_CONN retained"),
        Patch(facecolor=COLORS["S3_BUFF"], edgecolor="none", label="S3_BUFF retained"),
        Patch(facecolor=COLORS["S6_UPRISK"], edgecolor="none", label="S6_UPRISK retained"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025), columnspacing=1.4, handlelength=1.4)
    save(fig, MAIN / "Fig04_deterministic_screening.svg")


def draw_parameter_forecast_heatmap() -> None:
    df = pd.read_csv(PROCESSED / "ensemble_parameter_group_forecast_associations.csv")
    groups = ["upper pilot points", "main pilot points", "lower pilot points", "recharge", "regional support", "receptor conductance", "palaeochannel multiplier", "backfill conductivity"]
    forecasts = ["fcst_max_receptor_dd", "fcst_max_compliance_dd", "fcst_stage4_mean_inflow", "fcst_recovery_years"]
    labels = ["receptor DD", "compliance DD", "Stage 4 inflow", "recovery"]
    scenarios = ["S0_BASE", "S2_CONN", "S3_BUFF", "S6_UPRISK"]
    nmap = {"S0_BASE": 58, "S2_CONN": 38, "S3_BUFF": 89, "S6_UPRISK": 42}
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), constrained_layout=False)
    plt.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.965, wspace=0.17, hspace=0.24)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, 0.50)
    for k, (ax, scenario) in enumerate(zip(axes.ravel(), scenarios)):
        pivot = df[df.scenario == scenario].pivot(index="parameter_group", columns="forecast", values="max_abs_rho").reindex(index=groups, columns=forecasts)
        values = pivot.to_numpy(float)
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([] if k < 2 else labels, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(groups if k % 2 == 0 else [])
        ax.set_title(f"{'ABCD'[k]}  {scenario} (n={nmap[scenario]})", loc="left", fontweight="bold", pad=7)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.85)
        ax.tick_params(which="minor", bottom=False, left=False)
        colmax = np.nanmax(values, axis=0)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "black" if luminance > 0.56 else "white"
                weight = "bold" if np.isclose(value, colmax[j], atol=0.005) else "normal"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.8, color=color, fontweight=weight)
    cax = fig.add_axes([0.20, 0.055, 0.785, 0.025])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_label("maximum within-group absolute Spearman rank association")
    save(fig, MAIN / "Fig05_ensemble_parameter_forecast_associations.svg")


def draw_observation_forecast_linkage() -> None:
    df = pd.read_csv(PROCESSED / "dominant_observation_forecast_linkages.csv")
    scenarios = ["S0_BASE", "S2_CONN", "S3_BUFF", "S6_UPRISK"]
    forecasts = ["fcst_max_receptor_dd", "fcst_max_compliance_dd", "fcst_stage4_mean_inflow", "fcst_recovery_years"]
    labels = ["receptor DD", "compliance DD", "Stage 4 inflow", "recovery"]
    short = {
        "h_receptor_020y": "receptor head, 20 yr",
        "h_compliance_020y": "compliance head, 20 yr",
        "h_compliance_050y": "compliance head, 50 yr",
        "h_compliance_120y": "compliance head, 120 yr",
        "q_mineinflow_020y": "mine inflow, 20 yr",
    }
    array = np.zeros((4, 4))
    signed = np.zeros((4, 4))
    names = np.empty((4, 4), dtype=object)
    for i, scenario in enumerate(scenarios):
        for j, forecast in enumerate(forecasts):
            row = df[(df.scenario == scenario) & (df.forecast == forecast)].iloc[0]
            array[i, j] = row.abs_rho
            signed[i, j] = row.spearman_rho
            names[i, j] = short.get(row.source_name, row.source_name)
    fig, ax = plt.subplots(figsize=(7.25, 4.8), constrained_layout=False)
    plt.subplots_adjust(left=0.16, right=0.91, bottom=0.19, top=0.96)
    cmap = plt.get_cmap("magma")
    norm = Normalize(0, 1)
    image = ax.imshow(array, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, rotation=27, ha="right")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(scenarios)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(4):
        for j in range(4):
            rgba = cmap(norm(array[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.60 else "white"
            ax.text(j, i, f"{names[i, j]}\nρ = {signed[i, j]:+.2f}", ha="center", va="center", fontsize=7.8, color=text_color, linespacing=1.22)
    cax = fig.add_axes([0.93, 0.19, 0.022, 0.77])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("absolute Spearman association, |ρ|")
    save(fig, SI / "FigS08_observation_forecast_linkage.svg")


def draw_prior_parameter_distributions() -> None:
    ensemble = pd.read_csv(SCENARIO_OUTPUTS / "S0_BASE" / "prior_parensemble.csv", index_col=0).drop(index="base", errors="ignore")
    metadata = pd.read_csv(SCENARIO_OUTPUTS / "S0_BASE" / "parameters_metadata.csv").set_index("parnme")
    groups = {
        "Global multipliers": metadata.index[metadata.pargp == "GLOBAL"].tolist(),
        "Upper pilot points": metadata.index[metadata.pargp.str.contains("UPPER", case=False, na=False)].tolist(),
        "Main pilot points": metadata.index[metadata.pargp.str.contains("MAIN", case=False, na=False)].tolist(),
        "Lower pilot points": metadata.index[metadata.pargp.str.contains("LOWER", case=False, na=False)].tolist(),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.96, wspace=0.24, hspace=0.34)
    for ax, (title, columns), letter in zip(axes.ravel(), groups.items(), "ABCD"):
        values = ensemble[columns].to_numpy().ravel().astype(float)
        ax.hist(values, bins=24, color="#4C78A8", alpha=0.90, edgecolor="white", linewidth=0.35)
        ax.axvline(1.0 if title == "Global multipliers" else 0.0, color="#222", ls="--", lw=1.0)
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        ax.set_ylabel("count")
        ax.set_xlabel("multiplier" if title == "Global multipliers" else "log10 pilot-point multiplier")
        ax.grid(axis="y", color="#E3E6E9", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.035, "The common prior contains 300 draws (base realization excluded); pilot-point panels pool all locations in each group.", ha="center", fontsize=8.2, color="#4E5961")
    save(fig, SI / "FigS04_prior_parameter_distributions.svg")


def draw_prior_forecast_distributions() -> None:
    data = pd.read_csv(PROCESSED / "successful_prior_forecast_realizations.csv")
    scenarios = ["S0_BASE", "S2_CONN", "S3_BUFF", "S6_UPRISK"]
    metrics = [
        ("fcst_max_compliance_dd", "maximum compliance drawdown (m)", 30.0, "A"),
        ("fcst_max_receptor_dd", "maximum receptor drawdown (m)", 2.0, "B"),
        ("fcst_recovery_years", "recovery to within 1 m (years)", None, "C"),
        ("fcst_stage4_mean_inflow", "mean Stage 4 inflow (10³ m³/d)", 50.0, "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.4), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.94, wspace=0.25, hspace=0.38)
    bins = {
        "fcst_max_compliance_dd": np.linspace(0, 50, 22),
        "fcst_max_receptor_dd": np.linspace(0, 28, 22),
        "fcst_recovery_years": np.arange(10, 75, 3),
        "fcst_stage4_mean_inflow": np.linspace(15, 120, 24),
    }
    for ax, (metric, label, threshold, letter) in zip(axes.ravel(), metrics):
        for scenario in scenarios:
            values = data.loc[data.scenario == scenario, metric].dropna().to_numpy(float)
            if metric == "fcst_stage4_mean_inflow":
                values = values / 1000.0
            ax.hist(values, bins=bins[metric], density=True, histtype="stepfilled", alpha=0.26, color=COLORS[scenario], edgecolor=COLORS[scenario], linewidth=0.8, label=f"{scenario} (n={len(values)})")
        if threshold is not None:
            ax.axvline(threshold, color=COLORS["threshold"], ls="--", lw=1.0)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.grid(color="#E7E9EB", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=COLORS[s], edgecolor=COLORS[s], alpha=0.35, label=s) for s in scenarios]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.005, "Successful iteration-0 simulated-equivalent rows only; counts differ by scenario because some prior model runs did not yield complete outputs.", ha="center", fontsize=7.8, color="#4E5961")
    save(fig, SI / "FigS05_successful_prior_forecast_distributions.svg")


def main() -> None:
    draw_stacked_benchmark_geometry()
    draw_deterministic_screening()
    draw_parameter_forecast_heatmap()
    draw_prior_parameter_distributions()
    draw_prior_forecast_distributions()
    draw_observation_forecast_linkage()
    print(f"Regenerated data-driven SVG figures in {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
