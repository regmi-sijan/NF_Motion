"""
Run many simulations of any NF_Motion model in parallel
=======================================================

One simulation cannot be split across CPU cores (every time step depends on the
previous one), but independent runs can: different random seeds or different
parameter values. This script spreads them over all cores and summarises them.

    --model constant   nf_constant_force.py
    --model motors     motor_variable_force.py
    --model coupled    nf_stochastic_motors.py   (default)

Any option of the chosen model can be passed through (e.g. -T, -N, -F,
--mean-kinesin, --koff-nf ...). Run  python <script> --help  to see them.
Default run settings are taken from the top of the model's script.

Usage examples
--------------
    python run_parallel.py --runs 8                                # coupled model, 8 seeds
    python run_parallel.py --model constant --runs 8 --sweep F=10,20,30
    python run_parallel.py --model motors --runs 4 --sweep mean-dynein=1,2,3
    python run_parallel.py --runs 6 --sweep mean-kinesin=1,2,3 -T 2000 --workers 4
    python run_parallel.py --runs 4 --save-runs                    # keep every run's data + plots

Outputs (in --outdir):
    parallel <model> summary.csv   one row per run (parameters, seed, velocity, forces ...)
    parallel <model> summary.png   distance vs time (constant / coupled) or mean
                                   force per run (motors)
"""

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import nf_common as nf
import motor_variable_force
import nf_constant_force
import nf_stochastic_motors

# =============================================================================
# RUN SETTINGS (defaults - override on the command line)
# =============================================================================
MODEL = "coupled"      # "constant", "motors" or "coupled"
RUNS = 4               # runs (seeds) per parameter set
BASE_SEED = 1          # seeds are BASE_SEED, BASE_SEED+1, ...
WORKERS = os.cpu_count() or 1
TRACE_DT = 1.0         # time resolution of the stored distance traces (s)
# =============================================================================

SCRIPT_DEFAULTS = {"constant": nf_constant_force.DEFAULTS,
                   "motors": motor_variable_force.DEFAULTS,
                   "coupled": nf_stochastic_motors.DEFAULTS}


def one_run(model, sim_args, seed, save_runs, trace_dt, outdir):
    """Worker: run one simulation, return its summary and a thinned distance trace."""
    p = nf.parse_args(model, SCRIPT_DEFAULTS[model], sim_args + ["--seed", str(seed)])
    t0 = time.time()
    res = nf.simulate(p)
    summary = {"seed": seed, **nf.summary_stats(res, p), "runtime_s": time.time() - t0}
    if save_runs:
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, nf.run_name(p))
        nf.save_results(res, stem + ".txt")
        if model != "constant":
            nf.plot_motor_force(res, p, stem + " motors.png")
        if model != "motors":
            nf.plot_displacement(res, p, stem + " displacement.png")
    every = max(1, int(round(trace_dt / p.dt)))
    return summary, res["time"][every - 1::every], res["position"][every - 1::every]


def parse_sweeps(sweeps):
    """['F=10,20', 'N=50,100'] -> list of argument lists, one per combination."""
    names, values = [], []
    for sw in sweeps or []:
        if "=" not in sw:
            raise SystemExit(f"--sweep needs name=v1,v2,...  (got '{sw}')")
        name, vals = sw.split("=", 1)
        names.append(name.strip().lstrip("-"))
        values.append([v.strip() for v in vals.split(",") if v.strip()])
    combos = []
    for combo in (itertools.product(*values) if names else [()]):
        args = []
        for name, v in zip(names, combo):
            args += [f"--{name}", v]
        combos.append(args)
    return combos


def plot_summary(model, groups, rows, traces, path):
    plt = nf._plt(False)
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(10, 6))
    if model == "motors":
        # the NF does not move: show mean net force per run for each parameter set
        for gi, g in enumerate(groups):
            F = [r["mean_F_net_pN"] for r in rows if r["params"] == g]
            ax.scatter(np.full(len(F), gi) + np.linspace(-0.12, 0.12, len(F)), F,
                       color=COLORS[gi % 8], s=40, zorder=3)
            ax.hlines(np.mean(F), gi - 0.25, gi + 0.25, color=nf.INK, lw=2, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=9)
        ax.axhline(0, color=nf.INK2, lw=0.8)
        ax.set_ylabel("Mean net motor force (pN)", fontsize=12, color=nf.INK)
        ax.set_title("Mean motor force per run (dots), mean per parameter set (bar)",
                     loc="left", fontsize=13, color=nf.INK)
    else:
        for gi, g in enumerate(groups):
            color = COLORS[gi % 8]
            runs = traces[g]
            L = min(len(t) for t, _ in runs)
            for t, x in runs:
                ax.plot(t[:L], x[:L], color=color, lw=0.7, alpha=0.35)
            mean = np.mean([x[:L] for _, x in runs], axis=0)
            v = mean[-1] / runs[0][0][L - 1]
            ax.plot(runs[0][0][:L], mean, color=color, lw=2.2,
                    label=f"{g}  (mean v = {v:+.3g} nm/s, {len(runs)} runs)")
        ax.axhline(0, color=nf.INK2, lw=0.8)
        ax.set_xlabel("Time (s)", fontsize=12, color=nf.INK)
        ax.set_ylabel("Distance (nm)", fontsize=12, color=nf.INK)
        ax.set_title("NF distance vs time: thin = single runs, thick = mean", loc="left",
                     fontsize=13, color=nf.INK)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    nf._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Run NF_Motion simulations in parallel. "
                    "Unknown options are passed to the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", choices=nf.MODELS, default=MODEL, help="which model to run")
    ap.add_argument("--runs", type=int, default=RUNS, help="runs (seeds) per parameter set")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED, help="first seed")
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel processes (CPU cores)")
    ap.add_argument("--sweep", action="append",
                    help="parameter to vary, e.g. F=10,20,30 or mean-kinesin=1,2,3 (can repeat)")
    ap.add_argument("--trace-dt", type=float, default=TRACE_DT,
                    help="time resolution of the stored distance traces (s)")
    ap.add_argument("--save-runs", action="store_true",
                    help="also save every run's data file and plots")
    ap.add_argument("--outdir", default=nf.OUTPUT_DIR, help="folder for results")
    args, sim_args = ap.parse_known_args()
    if "--seed" in sim_args:
        raise SystemExit("use --base-seed here (seeds are assigned per run)")
    model = args.model
    sim_args += ["--no-save", "--no-plot"]
    nf.parse_args(model, SCRIPT_DEFAULTS[model], sim_args)  # fail early on a bad option

    combos = parse_sweeps(args.sweep)
    for c in combos:  # check every sweep value is a valid option
        nf.parse_args(model, SCRIPT_DEFAULTS[model], sim_args + c)
    jobs = [(sim_args + c, args.base_seed + r, c) for c in combos for r in range(args.runs)]
    workers = max(1, min(args.workers, len(jobs)))
    print(f"model '{model}': {len(jobs)} runs ({len(combos)} parameter set(s) x "
          f"{args.runs} seeds) on {workers} worker(s)")

    # compile once here so every worker loads the cached machine code
    nf.simulate(nf.parse_args(model, SCRIPT_DEFAULTS[model],
                              ["-T", "0.02", "--seed", "0", "--no-save", "--no-plot"]))

    t0 = time.time()
    rows, traces = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one_run, model, sa, seed, args.save_runs, args.trace_dt,
                            args.outdir): (combo, seed)
                for sa, seed, combo in jobs}
        for n_done, fut in enumerate(as_completed(futs), 1):
            combo, seed = futs[fut]
            summary, t, x = fut.result()
            label = " ".join(combo) if combo else "defaults"
            rows.append({"params": label, **summary})
            traces.setdefault(label, []).append((t, x))
            val = (f"F = {summary['mean_F_net_pN']:+.3f} pN" if model == "motors"
                   else f"v = {summary['velocity_nm_s']:+.3f} nm/s")
            print(f"  [{n_done}/{len(jobs)}] {label:28s} seed {seed:4d}: {val}  "
                  f"({summary['runtime_s']:.1f} s)")
    wall = time.time() - t0
    cpu = sum(r["runtime_s"] for r in rows)
    print(f"done: wall time {wall:.1f} s, total run time {cpu:.1f} s  "
          f"-> speed-up x{cpu / wall:.1f} from running in parallel")

    groups = [(" ".join(c) if c else "defaults") for c in combos]
    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    print("\nmean over seeds (± standard error):")
    for g in groups:
        rr = [r for r in rows if r["params"] == g]
        F = np.array([r["mean_F_net_pN"] for r in rr])
        if model == "motors":
            print(f"  {g:28s} mean net force {F.mean():+.3f} ± {se(F):.3f} pN")
        else:
            v = np.array([r["velocity_nm_s"] for r in rr])
            print(f"  {g:28s} velocity {v.mean():+.3f} ± {se(v):.3f} nm/s "
                  f"({v.mean() * 0.0864:+.4f} mm/day), mean force {F.mean():+.2f} pN")

    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, f"parallel {model} summary.csv")
    keys = list(rows[0].keys())
    rows.sort(key=lambda r: (groups.index(r["params"]), r["seed"]))
    with open(csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[k]}"' if isinstance(r[k], str) else f"{r[k]:.6g}"
                             for k in keys) + "\n")
    png = os.path.join(args.outdir, f"parallel {model} summary.png")
    plot_summary(model, groups, rows, traces, png)
    print(f"\n  table : {csv}\n  plot  : {png}")


if __name__ == "__main__":
    main()
