#!/usr/bin/env python3
"""
Probe-budget (N) sweep on hopper_hop
=====================================

Tests whether hopper_hop's low recall (0.747) is a sample-size problem.
Two other candidate remedies are examined elsewhere: longer horizons
(did not help, see ``run_horizon.py``) and a better probe policy
(Appendix C shows walker F1 *drops* 0.940 -> 0.895 as the scout goes
0 -> 80K steps).  N is the remaining one.

Critical detail: the effective sample size is the number of TRAJECTORIES,
not the number of environment steps.  ``cim._extract`` reduces each
trajectory to a single scalar (the mean absolute h-step displacement over
non-overlapping windows), so the two-sample test sees N values per branch,
not N*T.  Lengthening trajectories therefore does NOT increase test power
the way adding trajectories does -- this sweep varies N, not T.

Design
------
Collect once at N_max per branch, then evaluate every smaller N on a
prefix of the same trajectories.  The N values are perfectly nested and
paired, and one scout + one collection serves the whole sweep.

Note: the N=80 row is a *statistical* reproduction of the main table
(hopper_hop medium, R 0.747), not a bit-exact one -- collecting 640
baseline trajectories advances the RNG further before the intervention
branch starts than collecting 80 would.

Usage:
    python -m experiments.run_nsweep --seeds 42
    python -m experiments.run_nsweep --aggregate
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import logging
import time
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from experiments.run_horizon import score_at_horizons   # noqa: E402

N_GRID = [80, 160, 320, 640]
DEFAULT_SEEDS = [42, 142, 242]
HORIZONS = [1, 5, 10]


def run_seed(domain: str, task: str, seed: int,
             distractors: str = "medium",
             scout_steps: int = 80_000,
             n_max: int = 640, traj_length: int = 200,
             device: str = "cpu", torch_threads: int = 2,
             results_dir: str = "results/nsweep",
             overwrite: bool = False) -> Dict:
    out_dir = Path(results_dir) / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{domain}_{task}_{distractors}_s{seed}.json"
    if path.exists() and not overwrite:
        logger.info(f"SKIP (exists): {path.name}")
        with open(path) as f:
            return json.load(f)

    import torch
    from stable_baselines3 import SAC
    from ibd.probe import IBDProbe
    from experiments.dmcontrol_distractors import make_env

    torch.set_num_threads(torch_threads)
    t0 = time.time()

    scout_env = make_env(domain, task, distractors, seed=seed + 3000)
    probe_env = make_env(domain, task, distractors, seed=seed + 2000)
    true_dims = set(probe_env.true_dims)
    logger.info(f"{domain}_{task} | seed={seed} | "
                f"obs={probe_env.observation_space.shape[0]} "
                f"true={len(true_dims)} | N_max={n_max}")

    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4, batch_size=256,
                buffer_size=scout_steps, seed=seed, verbose=0, device=device)
    scout.learn(total_timesteps=scout_steps)

    def scout_policy(obs):
        with torch.no_grad():
            a, _ = scout.predict(obs, deterministic=False)
        return a

    probe = IBDProbe(probe_env, n_baseline=n_max, n_intervention=n_max,
                     traj_length=traj_length, horizons=HORIZONS,
                     n_permutations=5000, seed=seed)

    t_c = time.time()
    baseline = probe._collect(scout_policy, n_max, intervention=None)
    interv = probe._collect(scout_policy, n_max,
                            intervention={"dim": "all", "mode": "randomize"})
    logger.info(f"  collected {baseline.shape} + {interv.shape} "
                f"in {time.time() - t_c:.0f}s")

    variants = {}
    for N in N_GRID:
        if N > n_max:
            continue
        r = score_at_horizons(probe, baseline[:N], interv[:N], HORIZONS)
        soi = r["soi"]
        tp = len(soi & true_dims); fp = len(soi - true_dims)
        fn = len(true_dims - soi)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        variants[str(N)] = {
            "N": N, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_selected": len(soi),
            "h_star": r["h_star"], "n_tests": r["n_tests"],
            "n_significant": r["n_significant"],
            "missed_true_dims": sorted(true_dims - soi),
            "selected": sorted(soi),
        }
        logger.info(f"  N={N:<4} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                    f"H*={r['h_star']:.2f} | missed={sorted(true_dims - soi)}")

    result = {"domain": domain, "task": task, "distractors": distractors,
              "seed": seed, "n_true": len(true_dims), "n_max": n_max,
              "scout_steps": scout_steps, "variants": variants,
              "wall_s": time.time() - t0}
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    for e in (scout_env, probe_env):
        e.close()
    del scout
    return result


def aggregate(results_dir: str = "results/nsweep") -> str:
    out_dir = Path(results_dir)
    runs = [json.load(open(p))
            for p in sorted((out_dir / "per_run").glob("*.json"))]
    if not runs:
        return "No runs found.\n"
    import collections
    dom = f"{runs[0]['domain']}_{runs[0]['task']}"
    lines = [f"# Probe-budget sweep — {dom} ({len(runs)} seeds, "
             f"n_true={runs[0]['n_true']})", "",
             "N = number of **trajectories** per branch (the effective sample size), not environment steps.",
             "Each N is evaluated on a prefix of the same trajectories, so rows are fully nested and paired.", "",
             "| N | precision | recall | F1 | H* | most frequently missed dims |",
             "|---|---|---|---|---|---|"]
    for N in N_GRID:
        vs = [r["variants"][str(N)] for r in runs if str(N) in r["variants"]]
        if not vs:
            continue
        miss = collections.Counter()
        for v in vs:
            miss.update(v["missed_true_dims"])
        top = ", ".join(f"{d}({c})" for d, c in miss.most_common(5)) or "none"
        f = lambda k: (f"{np.mean([v[k] for v in vs]):.3f}"
                       f"±{np.std([v[k] for v in vs]):.3f}")
        lines.append(f"| {N} | {f('precision')} | {f('recall')} | {f('f1')} "
                     f"| {np.mean([v['h_star'] for v in vs]):.2f} | {top} |")
    report = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report)
    logger.info(f"Saved: {out_dir / 'summary.md'}")
    return report


def main():
    p = argparse.ArgumentParser(description="IBD probe-budget (N) sweep")
    p.add_argument("--domain", type=str, default="hopper")
    p.add_argument("--task", type=str, default="hop")
    p.add_argument("--distractors", type=str, default="medium")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--n_max", type=int, default=640)
    p.add_argument("--scout_steps", type=int, default=80_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--results_dir", type=str, default="results/nsweep")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--no_aggregate", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        print(aggregate(args.results_dir))
        return

    seeds = ([int(x) for x in args.seeds.split(",")]
             if args.seeds else list(DEFAULT_SEEDS))
    kw = dict(distractors=args.distractors, scout_steps=args.scout_steps,
              n_max=args.n_max, device=args.device,
              torch_threads=args.torch_threads,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if args.quick:
        seeds = [42]
        kw.update(scout_steps=1_000, n_max=160,
                  results_dir=args.results_dir + "_smoke")

    logger.info("=" * 60)
    logger.info(f"PROBE-BUDGET SWEEP — {args.domain}_{args.task}")
    logger.info(f"  N grid: {N_GRID}   seeds: {seeds}")
    logger.info("=" * 60)
    for s in seeds:
        run_seed(args.domain, args.task, s, **kw)
    if not args.no_aggregate:
        print(aggregate(kw["results_dir"]))


if __name__ == "__main__":
    main()
