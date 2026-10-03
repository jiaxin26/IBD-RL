#!/usr/bin/env python3
"""
Horizon Diagnostic: is hopper_hop's low recall a horizon problem?
==================================================================

hopper_hop medium is the weakest setting in the paper (recall 0.747,
precision 0.965 -- it *misses* dims, it does not add wrong ones).  The
per-seed masks show the misses are not random:

    dim 3 (position) 4/5 seeds,  dim 7 (touch) 4/5,  dim 2 (position) 3/5,
    dim 5 (position) 2/5,        dim 6 (touch) 2/5,  velocity dims 1/5 each

Velocity dims (8-14) are recovered almost always; the misses concentrate in
position (0-5) and touch (6-7).  That is mechanistically suggestive:
position is the integral of velocity, so its response to an action shows up
later than one control step, and touch is contact-gated.  The main results
use horizons = [1, 5, 10].

Hypothesis: hopper's missed dims are delayed responders, and longer
horizons should recover them.

Why this can fail
-----------------
Horizons are NOT free.  BH correction runs over all (dim, horizon) tests,
so adding horizons inflates the test count m and *tightens* the per-test
threshold.  Longer horizons also have fewer usable samples per trajectory
and lower SNR as the h-step difference saturates.  Whether the extra
detection chances outweigh the heavier multiple-testing burden is exactly
what this script measures -- it is not a foregone conclusion.

Design
------
Horizon choice affects only the *testing* stage, not data collection.  So
each seed collects the two branches ONCE and every horizon set is scored
on the SAME trajectories.  The comparison is therefore perfectly paired,
and costs one probe per seed instead of one per (seed, horizon set).

The horizons=[1,5] row is a reproduction check against the paper's main
table (hopper_hop medium: P 0.965 / R 0.747 / F1 0.836).

Usage:
    python -m experiments.run_horizon --domain hopper --task hop
    python -m experiments.run_horizon --aggregate
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
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Horizon sets to score on the same collected data.
HORIZON_SETS = [
    [1, 5, 10],                 # paper protocol -- reproduction check
    [1, 5, 10],
    [1, 5, 10, 20],
    [1, 5, 10, 20, 40],
    [10, 20],               # long-only: isolates the delayed-response claim
                            # from the extra-tests confound
]
DEFAULT_SEEDS = [42, 142, 242, 342, 442]


def _bh_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up (identical to probe.discover_joint)."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    idx = np.argsort(p)
    sp = p[idx]
    sig = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m):
        if sp[k] <= (k + 1) / m * alpha:
            max_k = k
    if max_k >= 0:
        sig[idx[:max_k + 1]] = True
    return sig


def score_at_horizons(probe, baseline: np.ndarray, intervention: np.ndarray,
                      horizons: List[int], alpha: float = 0.05) -> Dict:
    """Welch + BH over (dim, horizon), exactly as ``discover_joint`` does."""
    from scipy import stats as sp_stats

    keys, pvals, effs = [], [], []
    for sd in range(probe.obs_dim):
        for h in horizons:
            y_b = probe.cim._extract(baseline, sd, h, absolute=True)
            y_i = probe.cim._extract(intervention, sd, h, absolute=True)
            if y_b is None or y_i is None:
                continue
            if len(y_b) < 10 or len(y_i) < 10:
                continue
            _, p = sp_stats.ttest_ind(y_b, y_i, equal_var=False)
            n0, n1 = len(y_b), len(y_i)
            v0, v1 = float(np.var(y_b, ddof=1)), float(np.var(y_i, ddof=1))
            pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
            g = ((np.mean(y_b) - np.mean(y_i)) / pooled
                 if pooled > 1e-12 else 0.0)
            g *= 1 - 3 / (4 * (n0 + n1) - 9)
            keys.append((sd, h))
            pvals.append(float(p))
            effs.append(abs(float(g)))

    sig = _bh_reject(np.asarray(pvals), alpha)
    soi = {keys[i][0] for i in range(len(keys)) if sig[i]}
    n_sel = max(len(soi), 1)
    return {
        "soi": soi,
        "n_tests": len(keys),
        "n_significant": int(sig.sum()),
        # H* = mean number of horizons at which a selected dim is significant
        "h_star": float(sig.sum()) / n_sel,
        "max_g_true": {},
    }


def run_seed(domain: str, task: str, seed: int,
             distractors: str = "medium",
             scout_steps: int = 80_000,
             n_baseline: int = 80, n_intervention: int = 80,
             traj_length: int = 200,
             device: str = "cpu", torch_threads: int = 2,
             results_dir: str = "results/horizon",
             overwrite: bool = False) -> Dict:
    """Collect once, score every horizon set on the same trajectories."""
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

    # Same seed offsets as run_dmcontrol.run_single
    scout_env = make_env(domain, task, distractors, seed=seed + 3000)
    probe_env = make_env(domain, task, distractors, seed=seed + 2000)
    true_dims = set(probe_env.true_dims)
    logger.info(f"{domain}_{task} | {distractors} | seed={seed} | "
                f"obs={probe_env.observation_space.shape[0]} "
                f"true={len(true_dims)}")

    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4, batch_size=256,
                buffer_size=scout_steps, seed=seed, verbose=0, device=device)
    scout.learn(total_timesteps=scout_steps)

    def scout_policy(obs):
        with torch.no_grad():
            a, _ = scout.predict(obs, deterministic=False)
        return a

    probe = IBDProbe(probe_env, n_baseline=n_baseline,
                     n_intervention=n_intervention,
                     traj_length=traj_length, horizons=[1, 5, 10],
                     n_permutations=5000, seed=seed)

    # ── collect ONCE ──────────────────────────────────────────────────
    t_c = time.time()
    baseline = probe._collect(scout_policy, probe.n_baseline, intervention=None)
    interv = probe._collect(scout_policy, probe.n_intervention,
                            intervention={"dim": "all", "mode": "randomize"})
    logger.info(f"  collected in {time.time() - t_c:.0f}s "
                f"(baseline={baseline.shape}, interv={interv.shape})")

    variants = {}
    for hs in HORIZON_SETS:
        r = score_at_horizons(probe, baseline, interv, hs)
        soi = r["soi"]
        tp = len(soi & true_dims); fp = len(soi - true_dims)
        fn = len(true_dims - soi)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        key = ",".join(map(str, hs))
        variants[key] = {
            "horizons": hs, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_selected": len(soi),
            "n_tests": r["n_tests"], "n_significant": r["n_significant"],
            "h_star": r["h_star"],
            "missed_true_dims": sorted(true_dims - soi),
            "selected": sorted(soi),
        }
        logger.info(f"  H={key:<14} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                    f"| m={r['n_tests']:<4} H*={r['h_star']:.2f} "
                    f"| missed={sorted(true_dims - soi)}")

    result = {"domain": domain, "task": task, "distractors": distractors,
              "seed": seed, "n_true": len(true_dims),
              "scout_steps": scout_steps, "variants": variants,
              "wall_s": time.time() - t0}
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    for e in (scout_env, probe_env):
        e.close()
    del scout
    return result


def aggregate(results_dir: str = "results/horizon") -> str:
    out_dir = Path(results_dir)
    runs = [json.load(open(p))
            for p in sorted((out_dir / "per_run").glob("*.json"))]
    if not runs:
        return "No runs found.\n"
    keys = [",".join(map(str, hs)) for hs in HORIZON_SETS]
    dom = f"{runs[0]['domain']}_{runs[0]['task']}"
    n_true = runs[0]["n_true"]

    lines = [f"# Horizon Diagnostic — {dom} ({len(runs)} seeds, "
             f"n_true={n_true})", "",
             "All rows are scored on the same trajectories, so they are exactly paired. `m` is the number of tests entering the BH correction.", "",
             "| horizons | m | precision | recall | F1 | H* | most frequently missed dims |",
             "|---|---|---|---|---|---|---|"]
    import collections
    for k in keys:
        vs = [r["variants"][k] for r in runs if k in r["variants"]]
        if not vs:
            continue
        miss = collections.Counter()
        for v in vs:
            miss.update(v["missed_true_dims"])
        top = ", ".join(f"{d}({c})" for d, c in miss.most_common(5)) or "none"
        f = lambda x: f"{np.mean([v[x] for v in vs]):.3f}±{np.std([v[x] for v in vs]):.3f}"
        lines.append(f"| [{k}] | {np.mean([v['n_tests'] for v in vs]):.0f} "
                     f"| {f('precision')} | {f('recall')} | {f('f1')} "
                     f"| {np.mean([v['h_star'] for v in vs]):.2f} | {top} |")
    report = "\n".join(lines) + "\n"
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report)
    logger.info(f"Saved: {out_dir / 'summary.md'}")
    return report


def main():
    p = argparse.ArgumentParser(description="IBD horizon diagnostic")
    p.add_argument("--domain", type=str, default="hopper")
    p.add_argument("--task", type=str, default="hop")
    p.add_argument("--distractors", type=str, default="medium")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--scout_steps", type=int, default=80_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--results_dir", type=str, default="results/horizon")
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
              device=args.device, torch_threads=args.torch_threads,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if args.quick:
        seeds = [42]
        kw.update(scout_steps=1_000, results_dir=args.results_dir + "_smoke")

    logger.info("=" * 60)
    logger.info(f"HORIZON DIAGNOSTIC — {args.domain}_{args.task}")
    logger.info(f"  Horizon sets: {HORIZON_SETS}")
    logger.info(f"  Seeds: {seeds}")
    logger.info("=" * 60)
    for s in seeds:
        run_seed(args.domain, args.task, s, **kw)
    if not args.no_aggregate:
        print(aggregate(kw["results_dir"]))


if __name__ == "__main__":
    main()
