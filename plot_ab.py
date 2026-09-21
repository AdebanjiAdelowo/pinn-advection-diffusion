"""
Figures for the A/B comparison, from the checkpoints and JSON written by run_ab.py.

    python plot_ab.py --results results/ab_results.json --work-dir runs --out-dir results

Writes periodicity_mismatch.png, ab_seeds.png and pinn_result_value_derivative.png
(the same three-panel figure as pinn_result.png, for formulation B, first seed).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import diagnostics
import pinn

COLOR = {"A": "#2a78d6", "B": "#eb6834"}          # blue / orange, checked with validate_palette
STYLE = {"A": dict(ls="-", marker="o"), "B": dict(ls="--", marker="s")}
LABEL = {"A": "A: value only (original)", "B": "B: value + derivative"}
INK, GRID = "#0b0b0b", "#dcdcd8"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(work_dir: Path, name: str, seed: int) -> pinn.PINN:
    model = pinn.PINN()
    model.load_state_dict(torch.load(work_dir / f"{name}_seed{seed}.pt"))
    return model


def periodicity_figure(work_dir: Path, seed: int, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    floor = diagnostics.periodicity_diagnostics(diagnostics.ExactSolution())
    floors = (floor["value"]["rms"], floor["derivative"]["rms"])
    for ax, key, title, fl in zip(
            axes, ("E_u", "E_ux"),
            (r"Value mismatch  $|u(0,t)-u(1,t)|$", r"Derivative mismatch  $|u_x(0,t)-u_x(1,t)|$"), floors):
        for name in ("A", "B"):
            per = diagnostics.periodicity_diagnostics(load(work_dir, name, seed))
            ax.semilogy(per["t"], np.abs(per[key]) + 1e-12, color=COLOR[name], lw=1.2,
                        ls=STYLE[name]["ls"], label=LABEL[name])
        ax.axhline(fl, color="#7a7975", lw=1, ls=":", label="float32 round-off floor (exact solution, RMS)")
        ax.set(title=title, xlabel="t", ylabel="absolute mismatch")
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Periodic-boundary mismatch of the trained networks (seed {seed})", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def seeds_figure(results: dict, out: Path):
    runs = results["runs"]
    seeds = sorted({r["seed"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [("Relative $L^2$ error of $u$", lambda m: m["accuracy"]["relative_l2_u"], False),
              (r"RMS derivative mismatch $u_x(0,t)-u_x(1,t)$", lambda m: m["periodicity"]["derivative"]["rms"], True)]
    for ax, (title, f, log) in zip(axes, panels):
        val = {n: [f(next(r for r in runs if r["formulation"] == n and r["seed"] == s)["metrics"])
                   for s in seeds] for n in ("A", "B")}
        x = np.arange(len(seeds))
        for xi, a, b in zip(x, val["A"], val["B"]):
            ax.plot([xi, xi], [a, b], color="#b9b8b2", lw=1, zorder=1)
        for name in ("A", "B"):
            ax.scatter(x, val[name], s=44, color=COLOR[name], marker=STYLE[name]["marker"],
                       edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=LABEL[name])
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max(max(v) for v in val.values()) * 1.15)
        ax.set_xticks(x, [str(s) for s in seeds])
        ax.set(title=title, xlabel="seed")
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A vs B, same seed, initialisation and batches (paired)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/ab_results.json")
    ap.add_argument("--work-dir", default="runs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    work_dir, out_dir = Path(args.work_dir), Path(args.out_dir)
    results = json.loads(Path(args.results).read_text())
    seed = results["config"]["seeds"][0]
    periodicity_figure(work_dir, seed, out_dir / "periodicity_mismatch.png")
    seeds_figure(results, out_dir / "ab_seeds.png")
    pinn.plot(load(work_dir, "B", seed), str(out_dir / "pinn_result_value_derivative.png"))


if __name__ == "__main__":
    main()
