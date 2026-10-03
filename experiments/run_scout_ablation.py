#!/usr/bin/env python3
"""
Scout Policy Ablation
======================
Tests how IBD mask quality varies with scout training budget.

This is a PROBE-ONLY experiment: no full RL training needed.
Each (task, budget, seed) takes ~3-7 minutes.

Usage:
    python run_scout_ablation.py
    python run_scout_ablation.py --quick   # 1 task, 1 seed
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
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SCOUT_BUDGETS = [0, 10_000, 20_000, 40_000, 80_000, 160_000]
TASKS = [("walker", "walk"), ("cheetah", "run"), ("reacher", "hard")]
DISTRACTOR = "medium"
SEEDS = [42, 142, 242]


def run_single_probe(domain, task, scout_budget, seed, distractor="medium"):
    """Run one IBD probe with a given scout budget. Returns metrics dict."""
    from experiments.dmcontrol_distractors import make_env
    from ibd.probe import IBDProbe

    env = make_env(domain, task, distractor, seed=seed)
    probe_env = make_env(domain, task, distractor, seed=seed + 2000)
    true_soi = env.true_dims
    obs_dim = env.observation_space.shape[0]

    t0 = time.time()

    # Build scout policy
    scout_policy = None
    if scout_budget > 0:
        import torch
        from stable_baselines3 import SAC

        scout_env = make_env(domain, task, distractor, seed=seed + 3000)
        scout = SAC("MlpPolicy", scout_env,
                     learning_rate=3e-4, batch_size=256,
                     buffer_size=max(scout_budget, 10_000),
                     seed=seed, verbose=0)
        scout.learn(total_timesteps=scout_budget)

        def _scout_fn(obs, _model=scout):
            with torch.no_grad():
                action, _ = _model.predict(obs, deterministic=False)
            return action
        scout_policy = _scout_fn

        del scout_env

    scout_time = time.time() - t0

    # Run IBD probe
    probe = IBDProbe(
        probe_env,
        n_baseline=80,
        n_intervention=80,
        traj_length=200,
        horizons=[1, 5, 10],
        n_permutations=5000,
        seed=seed,
    )
    soi, info = probe.discover_joint(policy=scout_policy)

    total_time = time.time() - t0

    # Evaluate
    tp = len(soi & true_soi)
    fp = len(soi - true_soi)
    fn = len(true_soi - soi)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    result = {
        "domain": domain,
        "task": task,
        "distractor": distractor,
        "scout_budget": scout_budget,
        "seed": seed,
        "obs_dim": obs_dim,
        "n_true": len(true_soi),
        "n_discovered": len(soi),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "scout_time_s": round(scout_time, 1),
        "total_time_s": round(total_time, 1),
    }

    logger.info(
        f"{domain}_{task} | scout={scout_budget:>7d} | seed={seed} | "
        f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | "
        f"discovered={len(soi)}/{obs_dim} | {total_time:.0f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 task, 1 seed, 3 budgets")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--distractor", type=str, default="medium")
    args = parser.parse_args()

    if args.quick:
        tasks = [("cheetah", "run")]
        budgets = [0, 40_000, 80_000]
        seeds = [42]
    elif args.domain and args.task:
        tasks = [(args.domain, args.task)]
        budgets = SCOUT_BUDGETS
        seeds = SEEDS
    else:
        tasks = TASKS
        budgets = SCOUT_BUDGETS
        seeds = SEEDS

    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_total = len(tasks) * len(budgets) * len(seeds)
    n_done = 0

    for domain, task in tasks:
        for budget in budgets:
            for seed in seeds:
                # Resume support
                fname = f"scout_{domain}_{task}_{args.distractor}_b{budget}_s{seed}.json"
                fpath = out_dir / fname
                if fpath.exists():
                    logger.info(f"SKIP (exists): {fname}")
                    with open(fpath) as f:
                        all_results.append(json.load(f))
                    n_done += 1
                    continue

                result = run_single_probe(
                    domain, task, budget, seed, args.distractor)
                with open(fpath, "w") as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)
                n_done += 1
                logger.info(f"Progress: {n_done}/{n_total}")

    # Save aggregate
    with open(out_dir / "scout_ablation_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SCOUT ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Task':<20s} {'Budget':>8s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>4s}")
    print("-" * 50)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        key = (f"{r['domain']}_{r['task']}", r["scout_budget"])
        groups[key].append(r)

    for (task, budget), runs in sorted(groups.items()):
        p = np.mean([r["precision"] for r in runs])
        r_ = np.mean([r["recall"] for r in runs])
        f1 = np.mean([r["f1"] for r in runs])
        print(f"{task:<20s} {budget:>8d} {p:>6.3f} {r_:>6.3f} {f1:>6.3f} {len(runs):>4d}")


if __name__ == "__main__":
    main()
