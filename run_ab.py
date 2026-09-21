"""
Controlled comparison of the two periodic boundary formulations.

  A  value periodicity only:           u(0,t) = u(1,t)            (original)
  B  value + derivative periodicity:   additionally u_x(0,t) = u_x(1,t)

For each seed both formulations use the same initialisation, the same random
batches (the RNG stream is identical), the same optimiser, schedule, epochs and
unit loss weights. The only difference is the extra derivative-mismatch term in B.

Every metric is computed after training by `diagnostics`, independently of the
training loss. Per-run results and checkpoints go to --work-dir; the aggregate
goes to --out (JSON) and a flat CSV next to it.

    python run_ab.py --seeds 42 0 1 2 3 --work-dir runs --out results/ab_results.json
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import diagnostics
import pinn

FORMULATIONS = {"A": False, "B": True}  # name -> derivative_bc
HERE = Path(__file__).resolve().parent

FLAT = {  # CSV column -> path into the metrics dict
    "rel_l2_u": ("accuracy", "relative_l2_u"),
    "rel_l2_ux": ("accuracy", "relative_l2_ux"),
    "max_abs_err_u": ("accuracy", "max_abs_error_u"),
    "pde_res_rms": ("pde_residual", "rms"),
    "pde_res_max": ("pde_residual", "max_abs"),
    "ic_rms": ("initial_condition", "rms"),
    "ic_max": ("initial_condition", "max_abs"),
    "value_per_rms": ("periodicity", "value", "rms"),
    "value_per_max": ("periodicity", "value", "max_abs"),
    "deriv_per_rms": ("periodicity", "derivative", "rms"),
    "deriv_per_max": ("periodicity", "derivative", "max_abs"),
    "deriv_per_rms_rel": ("periodicity", "derivative_rms_relative_to_exact_ux"),
    "diff_flux_rms": ("periodicity", "diffusive_flux", "rms"),
    "total_flux_rms": ("periodicity", "total_flux", "rms"),
    "total_flux_max": ("periodicity", "total_flux", "max_abs"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()


def environment() -> dict:
    return dict(
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), machine=platform.machine(), device="cpu",
        torch_threads=torch.get_num_threads(),
        git_head=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
        source_sha256={f: sha256(HERE / f) for f in ("pinn.py", "diagnostics.py", "run_ab.py")},
    )


def get(d: dict, path: tuple):
    for key in path:
        d = d[key]
    return d


def run_one(name: str, seed: int, epochs: int, work_dir: Path) -> dict:
    record = work_dir / f"{name}_seed{seed}.json"
    if record.exists():
        return json.loads(record.read_text())
    t0 = time.perf_counter()
    model, history = pinn.train(n_epochs=epochs, derivative_bc=FORMULATIONS[name], seed=seed,
                                verbose=False, return_history=True)
    runtime = time.perf_counter() - t0
    torch.save(model.state_dict(), work_dir / f"{name}_seed{seed}.pt")
    result = dict(formulation=name, derivative_bc=FORMULATIONS[name], seed=seed, epochs=epochs,
                  runtime_s=runtime, final_batch=history[-1], history=history,
                  metrics=diagnostics.all_metrics(model))
    record.write_text(json.dumps(result, indent=1))
    print(f"{name} seed={seed:<3d} rel_L2={result['metrics']['accuracy']['relative_l2_u']:.3e} "
          f"value_rms={result['metrics']['periodicity']['value']['rms']:.2e} "
          f"deriv_rms={result['metrics']['periodicity']['derivative']['rms']:.2e} "
          f"runtime={runtime:.0f}s", flush=True)
    return result


def stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else None,
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--formulations", nargs="+", default=list(FORMULATIONS), choices=list(FORMULATIONS))
    ap.add_argument("--note", default=None, help="free-text caveat stored in the result config")
    ap.add_argument("--work-dir", default="runs")
    ap.add_argument("--out", default="results/ab_results.json")
    args = ap.parse_args()

    work_dir, out = Path(args.work_dir), Path(args.out)
    work_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = [run_one(name, seed, args.epochs, work_dir) for seed in args.seeds for name in args.formulations]

    summary = {name: {"n_seeds": len(args.seeds),
                      **{col: stats([get(r["metrics"], p) for r in runs if r["formulation"] == name])
                         for col, p in FLAT.items()},
                      "runtime_s": stats([r["runtime_s"] for r in runs if r["formulation"] == name])}
               for name in args.formulations}
    config = dict(
        pde="u_t + c u_x = nu u_xx on x in [0,1], t in [0,T]", c=pinn.C, nu=pinn.NU, T=pinn.T,
        network=dict(width=64, depth=4, activation="tanh"),
        optimizer="Adam, lr 1e-3, StepLR(step 5000, gamma 0.5)", epochs=args.epochs,
        points_per_epoch=dict(pde=2000, ic=200, bc=100), resampled_every_epoch=True,
        loss_weights=dict(pde=1.0, ic=1.0, bc_value=1.0, bc_derivative_B_only=1.0),
        seeds=args.seeds, note=args.note, evaluation=dict(
            accuracy_grid="300 x 150 uniform (x in [0,1], t in [0,T]), as in pinn.plot",
            pde_residual_grid="201 x 101 uniform including boundaries",
            initial_condition_grid="1001 uniform x at t=0",
            periodicity_grid="2001 uniform t in [0,T]"))
    out.write_text(json.dumps(dict(config=config, environment=environment(), summary=summary,
                                   runs=runs), indent=1))

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formulation", "seed", "runtime_s", *FLAT,
                    "final_batch_loss", "final_batch_pde", "final_batch_ic", "final_batch_bc_u",
                    "final_batch_bc_ux"])
        for r in runs:
            fb = r["final_batch"]
            w.writerow([r["formulation"], r["seed"], f"{r['runtime_s']:.1f}",
                        *[f"{get(r['metrics'], p):.6e}" for p in FLAT.values()],
                        *[("" if fb[k] is None else f"{fb[k]:.6e}") for k in
                          ("loss", "pde", "ic", "bc_u", "bc_ux")]])
    print(f"wrote {out} and {csv_path}")


if __name__ == "__main__":
    main()
