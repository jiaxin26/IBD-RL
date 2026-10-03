#!/usr/bin/env python3
"""
Summary-statistic ablation: level statistic alongside the h-step displacement
=============================================================================

A ∆-mean test is blind to effects that shift the marginal without changing
step-to-step variability.  This script runs a level statistic alongside ∆
and aggregates over both.

Statistics compared, all scored on the SAME collected trajectories so the
comparison is exactly paired:

  delta   per-trajectory mean absolute h-step displacement over
          non-overlapping windows (the paper's summary), one test per (i, h)
  level   per-trajectory mean of o^(i) (the level statistic), one test per i
  both    union of the two test families, with BH applied jointly

Why this can fail -- two mechanisms, both real
----------------------------------------------
1. **Mid-trajectory resets.**  ``_collect`` resets and continues when an
   episode terminates.  Under do(A = Unif) an agent like walker falls almost
   immediately, so the intervention branch resets far more often than the
   baseline branch.  A reset re-initialises the ENTIRE observation vector,
   distractor dimensions included.  The level statistic is highly sensitive
   to this: it will register a between-branch difference on distractor
   dimensions purely because they were re-initialised at different rates.
   The displacement statistic is nearly immune (a reset contributes one
   outlying difference).  This is the most likely way the level statistic
   buys recall at the cost of precision -- and plausibly why a displacement
   summary was chosen in the first place.

2. **Multiple-testing burden.**  Adding a level test per dimension inflates
   the BH test count by 50% at |H| = 2, tightening every threshold.  The
   horizon ablation failed for exactly this reason.

Both are why precision is reported next to recall throughout, and why a
setting that already achieves recall 1.00 is included as a control: a
statistic that lifts recall on the weak settings while damaging precision
on the strong ones is not an improvement.

Trajectories are saved to disk so that any further summary-statistic
ablation is a seconds-long rescoring rather than a fresh probe.

Usage:
    python -m experiments.run_summary_stat --domain walker --task walk
    python -m experiments.run_summary_stat --aggregate
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from experiments.run_horizon import _bh_reject          # noqa: E402

HORIZONS = [1, 5, 10]
DEFAULT_SEEDS = [42, 142, 242, 342, 442]
# (domain, task, distractors, why it is in the grid)
DEFAULT_GRID = [
    ("walker", "walk", "medium", "recall 0.86"),
    ("walker", "walk", "hard", "recall 0.85"),
    ("hopper", "hop", "medium", "recall 0.75 - weakest in the paper"),
    ("cheetah", "run", "medium", "recall 1.00 - PRECISION CONTROL"),
]


def _welch(y_b: np.ndarray, y_i: np.ndarray):
    from scipy import stats as sp
    if len(y_b) < 10 or len(y_i) < 10:
        return None
    _, p = sp.ttest_ind(y_b, y_i, equal_var=False)
    return float(p)


def _delta_summary(trajs: np.ndarray, sd: int, h: int) -> Optional[np.ndarray]:
    """Mean absolute h-step displacement, non-overlapping windows."""
    N, T, D = trajs.shape
    if sd >= D or h >= T:
        return None
    t0 = np.arange(0, T - h, h)
    d = np.abs(trajs[:, t0 + h, sd] - trajs[:, t0, sd])
    return d.mean(axis=1)


def _level_summary(trajs: np.ndarray, sd: int) -> Optional[np.ndarray]:
    """Per-trajectory mean of the raw dimension value."""
    if sd >= trajs.shape[2]:
        return None
    return trajs[:, :, sd].mean(axis=1)


def score(baseline, interv, obs_dim, variant: str, alpha=0.05) -> Dict:
    """variant in {'delta', 'level', 'both'}; BH applied jointly."""
    keys, pv = [], []
    for sd in range(obs_dim):
        if variant in ("delta", "both"):
            for h in HORIZONS:
                a, b = _delta_summary(baseline, sd, h), _delta_summary(interv, sd, h)
                if a is None or b is None:
                    continue
                p = _welch(a, b)
                if p is not None:
                    keys.append((sd, f"d{h}")); pv.append(p)
        if variant in ("level", "both"):
            a, b = _level_summary(baseline, sd), _level_summary(interv, sd)
            if a is None or b is None:
                continue
            p = _welch(a, b)
            if p is not None:
                keys.append((sd, "lvl")); pv.append(p)
    sig = _bh_reject(np.asarray(pv), alpha)
    soi = {keys[i][0] for i in range(len(keys)) if sig[i]}
    return {"soi": soi, "n_tests": len(keys), "n_significant": int(sig.sum())}


def run_seed(domain, task, distractors, seed,
             scout_steps=80_000, n_traj=80, traj_length=200,
             device="cpu", torch_threads=2,
             results_dir="results/summary_stat",
             save_traj=True, overwrite=False) -> Dict:
    out_dir = Path(results_dir) / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{domain}_{task}_{distractors}_s{seed}"
    path = out_dir / f"{tag}.json"
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
    obs_dim = probe_env.observation_space.shape[0]
    logger.info(f"{tag} | obs={obs_dim} true={len(true_dims)}")

    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4, batch_size=256,
                buffer_size=scout_steps, seed=seed, verbose=0, device=device)
    scout.learn(total_timesteps=scout_steps)

    def scout_policy(obs):
        with torch.no_grad():
            a, _ = scout.predict(obs, deterministic=False)
        return a

    probe = IBDProbe(probe_env, n_baseline=n_traj, n_intervention=n_traj,
                     traj_length=traj_length, horizons=HORIZONS, seed=seed)
    baseline = probe._collect(scout_policy, n_traj, intervention=None)
    interv = probe._collect(scout_policy, n_traj,
                            intervention={"dim": "all", "mode": "randomize"})

    if save_traj:
        tdir = Path(results_dir) / "trajectories"
        tdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tdir / f"{tag}.npz",
                            baseline=baseline.astype(np.float32),
                            intervention=interv.astype(np.float32),
                            true_dims=np.array(sorted(true_dims)))

    variants = {}
    for v in ("delta", "level", "both"):
        r = score(baseline, interv, obs_dim, v)
        soi = r["soi"]
        tp = len(soi & true_dims); fp = len(soi - true_dims)
        fn = len(true_dims - soi)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        variants[v] = {"precision": prec, "recall": rec, "f1": f1,
                       "tp": tp, "fp": fp, "fn": fn,
                       "n_selected": len(soi), "n_tests": r["n_tests"],
                       "n_significant": r["n_significant"],
                       "selected": sorted(soi)}
        logger.info(f"  {v:<6} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                    f"(m={r['n_tests']}, |mask|={len(soi)})")

    result = {"domain": domain, "task": task, "distractors": distractors,
              "seed": seed, "obs_dim": obs_dim, "n_true": len(true_dims),
              "scout_steps": scout_steps, "variants": variants,
              "wall_s": time.time() - t0}
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    for e in (scout_env, probe_env):
        e.close()
    del scout
    return result


def aggregate(results_dir="results/summary_stat") -> str:
    out = Path(results_dir)
    runs = [json.load(open(p)) for p in sorted((out / "per_run").glob("*.json"))]
    if not runs:
        return "No runs found.\n"
    lines = ["# Summary-statistic ablation", "",
             "All three statistics are scored on the same trajectories, so the comparison is exactly paired.",
             "`delta` = the h-step absolute displacement used in the paper; `level` = per-trajectory mean of the raw values;",
             "`both` = union of the two test families under a joint BH correction.", ""]
    import collections
    by = collections.defaultdict(list)
    for r in runs:
        by[(r["domain"], r["task"], r["distractors"])].append(r)
    for k in sorted(by):
        rs = by[k]
        lines.append(f"## {k[0]}_{k[1]} — {k[2]} ({len(rs)} seeds)")
        lines.append("")
        lines.append("| statistic | m | precision | recall | F1 | Δrecall vs delta |")
        lines.append("|---|---|---|---|---|---|")
        base = np.mean([r["variants"]["delta"]["recall"] for r in rs])
        for v in ("delta", "level", "both"):
            vs = [r["variants"][v] for r in rs]
            f = lambda x: (f"{np.mean([u[x] for u in vs]):.3f}"
                           f"±{np.std([u[x] for u in vs]):.3f}")
            dr = np.mean([u["recall"] for u in vs]) - base
            lines.append(f"| {v} | {np.mean([u['n_tests'] for u in vs]):.0f} "
                         f"| {f('precision')} | {f('recall')} | {f('f1')} "
                         f"| {dr:+.3f} |")
        lines.append("")
    report = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(report)
    logger.info(f"Saved: {out / 'summary.md'}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--distractors", type=str, default="medium")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--scout_steps", type=int, default=80_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--results_dir", type=str, default="results/summary_stat")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--no_aggregate", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        print(aggregate(args.results_dir)); return

    seeds = ([int(x) for x in args.seeds.split(",")]
             if args.seeds else list(DEFAULT_SEEDS))
    grid = ([(args.domain, args.task, args.distractors, "")]
            if args.domain and args.task else DEFAULT_GRID)
    kw = dict(scout_steps=args.scout_steps, device=args.device,
              torch_threads=args.torch_threads,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if args.quick:
        seeds = [42]
        grid = [("walker", "walk", "medium", "")]
        kw.update(scout_steps=1_000, results_dir=args.results_dir + "_smoke")

    for d, t, dist, _why in grid:
        for s in seeds:
            run_seed(d, t, dist, s, **kw)
    if not args.no_aggregate:
        print(aggregate(kw["results_dir"]))


if __name__ == "__main__":
    main()
