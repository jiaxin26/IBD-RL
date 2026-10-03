#!/usr/bin/env python3
"""
Partial-Override Study: what happens when the do-operator is imperfect
======================================================================

This study asks whether IBD survives when the "override the agent's
actions during probing" assumption only holds approximately -- e.g. on
hardware, where a safety layer, an unmodelled actuator lag, or a human
supervisor keeps part of the incumbent policy in the loop.

Every result in the paper (including the (kappa, lambda) confounding grid)
assumes a *perfect* override: ``IBDProbe._apply_intervention`` overwrites
the commanded action outright, so the intervention branch is exactly
``a ~ Uniform(A)``.  Note that ``kappa`` contaminates the **baseline**
branch; imperfect hardware override contaminates the **intervention**
branch.  They are different manipulations and the kappa grid does not
speak to this one.

Here the intervention branch executes

    a_t = beta * u_t + (1 - beta) * pi(o_t),     u_t ~ Uniform(A)

with beta = 1 recovering the published protocol exactly (the uniform draw
is taken either way, so the RNG stream is untouched).

What the structure of the generative model predicts
---------------------------------------------------
The distractor recursion in ``confounded_distractors.py`` is

    d_t = rho * d_{t-1} + lam * L C_t + eps_t

which contains **no action term**.  The distractor marginal is therefore
identical in both branches whatever the executed action distribution is,
so residual policy leakage into the intervention branch cannot manufacture
distractor false positives.  What it can do is shrink the branch contrast
on the *truly controllable* dims, costing statistical power.

Prediction: precision (and fp_confounded in particular) stays flat as beta
falls; recall degrades.  I.e. imperfect override costs **power**, not
**validity**.  ``mean_hedges_g_true`` is reported as the mechanism.

Protocol is byte-for-byte the 2x2 sweep's (same env factory, same
behaviour policy, same horizons [1, 5, 10], same n_baseline/n_intervention,
same welch_bh primary variant), so the beta=1.0 row is a direct
reproduction check against ``results/confounding_2x2/per_run/``.

Usage:
    # One cell, all betas, all seeds (~100 s per probe)
    python -m experiments.run_override --kappa 0.5 --lam 1.0

    # Aggregate everything already on disk into a markdown table
    python -m experiments.run_override --aggregate

Output:
    results/override/per_run/{domain}_{task}_k{k}_l{l}_b{beta}_s{seed}.json
    results/override/summary.md
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reuse the 2x2 sweep's scoring machinery verbatim so the two studies are
# directly comparable (and so beta=1.0 reproduces its numbers).
from experiments.run_confounding_sweep import (      # noqa: E402
    ibd_all_variants, _metrics, PRIMARY_VARIANT,
)

DEFAULT_BETAS = [1.0, 0.75, 0.5, 0.25, 0.1]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_point(domain: str, task: str, kappa: float, lam: float,
              beta: float, seed: int,
              base_config: str = "medium",
              n_confounded: int = 12,
              n_baseline: int = 80, n_intervention: int = 80,
              traj_length: int = 200,
              horizons: List[int] | None = None,
              results_dir: str = "results/override",
              overwrite: bool = False) -> Dict:
    """One (cell, beta, seed) probe."""
    out_dir = Path(results_dir) / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"{domain}_{task}_k{kappa:g}_l{lam:g}"
             f"_b{beta:g}_s{seed}.json")
    path = out_dir / fname
    if path.exists() and not overwrite:
        logger.info(f"SKIP (exists): {fname}")
        with open(path) as f:
            return json.load(f)

    from experiments.confounded_distractors import (
        make_confounded_env, StructuredProbePolicy, ConfoundedPolicy,
        measure_probe_covariates)
    from ibd.probe import IBDProbe

    horizons = horizons or [1, 5, 10]
    t0 = time.time()

    # Identical construction to run_confounding_sweep.run_point
    env = make_confounded_env(domain, task, base_config=base_config,
                              n_confounded=n_confounded, lam=lam,
                              seed=seed + 2000)
    behaviour = ConfoundedPolicy(env, StructuredProbePolicy(env, seed=seed),
                                 kappa=kappa, seed=seed)

    obs_dim = env.observation_space.shape[0]
    logger.info(f"{domain}_{task} | kappa={kappa:g} lam={lam:g} "
                f"beta={beta:g} | seed={seed} | obs={obs_dim} "
                f"true={env.true_obs_dim} exo={len(env.exogenous_dims)} "
                f"conf={len(env.confounded_dims)}")

    # The 2x2 sweep measures the baseline branch's action distribution
    # here, before probing.  The rollout advances the env and the
    # behaviour policy, so it must be reproduced verbatim (not merely for
    # the covariates it returns) or the beta = 1.0 row would start from a
    # different RNG state than the published cell and fail to reproduce it.
    covariates = measure_probe_covariates(env, behaviour, n_steps=3000,
                                          seed=seed + 55)

    probe = IBDProbe(env, n_baseline=n_baseline,
                     n_intervention=n_intervention,
                     traj_length=traj_length, horizons=horizons,
                     seed=seed, override_beta=beta)

    variants, max_g, info = ibd_all_variants(probe, behaviour)
    soi = variants[PRIMARY_VARIANT]

    # Branch contrast — the covariate that predicts recall.
    g_true = [max_g.get(d, 0.0) for d in sorted(env.true_dims)]
    info["mean_hedges_g_true"] = float(np.mean(g_true)) if g_true else 0.0
    info["mean_hedges_g_distractor"] = float(np.mean(
        [max_g.get(d, 0.0) for d in sorted(env.distractor_dims)]))

    # Full per-dimension |g|, in dimension order.  Needed for the
    # ground-truth-free forms of the diagnostic (upper quantiles of |g|
    # over ALL dimensions), which stay defined even when the mask is
    # empty -- exactly the regime where a diagnostic is most needed.
    g_per_dim = [float(max_g.get(d, 0.0)) for d in range(obs_dim)]
    info["g_per_dim"] = g_per_dim
    ga = np.asarray(g_per_dim)
    for q in (50, 75, 90, 95):
        info[f"g_q{q}"] = float(np.percentile(ga, q))
    info["g_max"] = float(ga.max())

    result = {
        "domain": domain, "task": task,
        "kappa": kappa, "lam": lam, "beta": beta, "seed": seed,
        "base_config": base_config, "n_confounded": n_confounded,
        "obs_dim": obs_dim, "n_true": env.true_obs_dim,
        "n_exogenous": len(env.exogenous_dims),
        "covariates": covariates,
        **_metrics(soi, env),
        "selected": sorted(soi),
        "variant_metrics": {v: _metrics(s, env) for v, s in variants.items()},
        "wall_s": time.time() - t0,
        "info": {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else v)
                 for k, v in info.items()},
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"  -> P={result['precision']:.3f} R={result['recall']:.3f} "
                f"F1={result['f1']:.3f} | FPconf={result['fp_confounded']} "
                f"FPexo={result['fp_exogenous']} | "
                f"g_true={info['mean_hedges_g_true']:.2f} "
                f"({result['wall_s']:.0f}s)")

    env.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _mean_std(runs: List[Dict], key: str) -> str:
    v = [r[key] for r in runs]
    return f"{np.mean(v):.3f}±{np.std(v):.3f}"


def aggregate(results_dir: str = "results/override") -> str:
    """Scan per_run/ and emit a markdown report."""
    out_dir = Path(results_dir)
    runs = []
    for p in sorted((out_dir / "per_run").glob("*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        return "No runs found.\n"

    cells = sorted({(r["domain"], r["task"], r["kappa"], r["lam"])
                    for r in runs})
    betas = sorted({r["beta"] for r in runs}, reverse=True)

    lines = ["# Partial-Override Study", ""]
    lines.append("Intervention branch executes "
                 "`a = beta*u + (1-beta)*pi(o)`; `beta=1` is the "
                 "published perfect-override protocol.")
    lines.append("")

    for (domain, task, kappa, lam) in cells:
        sub = [r for r in runs
               if (r["domain"], r["task"], r["kappa"], r["lam"])
               == (domain, task, kappa, lam)]
        n_seeds = len({r["seed"] for r in sub})
        lines.append(f"## {domain}_{task} — kappa={kappa:g}, lam={lam:g} "
                     f"({n_seeds} seeds)")
        lines.append("")
        # |mask| is shown because precision is defined as 0 when nothing
        # is selected: once the probe loses enough power to abstain, the
        # precision column reads as a validity failure when what actually
        # happened is an empty mask with zero false positives.
        lines.append("| beta | |mask| | precision | recall | F1 | FPconf "
                     "| FPexo | mean Hedges g (true dims) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for b in betas:
            rs = [r for r in sub if abs(r["beta"] - b) < 1e-9]
            if not rs:
                continue
            g = [r["info"].get("mean_hedges_g_true", float("nan"))
                 for r in rs]
            lines.append(
                f"| {b:g} | {np.mean([r['n_selected'] for r in rs]):.1f} "
                f"| {_mean_std(rs, 'precision')} "
                f"| {_mean_std(rs, 'recall')} | {_mean_std(rs, 'f1')} "
                f"| {np.mean([r['fp_confounded'] for r in rs]):.1f} "
                f"| {np.mean([r['fp_exogenous'] for r in rs]):.1f} "
                f"| {np.mean(g):.2f} |")
        lines.append("")

    report = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report)
    logger.info(f"Saved: {out_dir / 'summary.md'}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="IBD partial-override (imperfect do-operator) study")
    p.add_argument("--domain", type=str, default="walker")
    p.add_argument("--task", type=str, default="walk")
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--betas", type=str, default=None,
                   help="comma-separated override fractions")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--base_config", type=str, default="medium")
    p.add_argument("--n_confounded", type=int, default=12)
    p.add_argument("--n_baseline", type=int, default=80)
    p.add_argument("--n_intervention", type=int, default=80)
    p.add_argument("--traj_length", type=int, default=200)
    p.add_argument("--results_dir", type=str, default="results/override")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate", action="store_true",
                   help="only aggregate what is already on disk")
    p.add_argument("--no_aggregate", action="store_true",
                   help="skip the trailing aggregate — use when several "
                        "invocations run concurrently and would race on "
                        "summary.md")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 1 seed, 2 betas, tiny budget")
    args = p.parse_args()

    if args.aggregate:
        print(aggregate(args.results_dir))
        return

    betas = ([float(x) for x in args.betas.split(",")]
             if args.betas else list(DEFAULT_BETAS))
    seeds = ([int(x) for x in args.seeds.split(",")]
             if args.seeds else list(DEFAULT_SEEDS))

    kw = dict(base_config=args.base_config,
              n_confounded=args.n_confounded,
              n_baseline=args.n_baseline,
              n_intervention=args.n_intervention,
              traj_length=args.traj_length,
              results_dir=args.results_dir,
              overwrite=args.overwrite)

    if args.quick:
        # >= 10 trajectories per branch: ``ibd_all_variants`` yields one
        # sample per trajectory and skips any test with fewer than 10.
        betas, seeds = [1.0, 0.25], [0]
        kw.update(n_baseline=16, n_intervention=16, traj_length=60,
                  results_dir=args.results_dir + "_smoke")

    logger.info("=" * 60)
    logger.info("IBD PARTIAL-OVERRIDE STUDY")
    logger.info(f"  Task:  {args.domain}_{args.task}")
    logger.info(f"  Cell:  kappa={args.kappa:g} lam={args.lam:g}")
    logger.info(f"  Betas: {betas}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Probes: {len(betas) * len(seeds)}")
    logger.info("=" * 60)

    for beta in betas:
        for seed in seeds:
            run_point(args.domain, args.task, args.kappa, args.lam,
                      beta, seed, **kw)

    if not args.no_aggregate:
        print(aggregate(kw["results_dir"]))


if __name__ == "__main__":
    main()
