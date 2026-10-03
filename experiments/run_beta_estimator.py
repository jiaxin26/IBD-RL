#!/usr/bin/env python3
"""
Estimating override quality (beta) from probe data alone
=========================================================

``run_override.py`` measures what happens to boundary recovery when the
do-operator is only partially effective.  It does not tell a practitioner
*whether their own setup is in that regime*, i.e. how the override
assumption can be tested in practice.

This script supplies the missing half: an estimator for beta computable
from quantities a practitioner can log.

Estimator
---------
During the intervention branch the probe *commands* u_t ~ Uniform(A).
If the actuator only partially overrides the incumbent policy, the
*executed* action is

    a_t = beta * u_t + (1 - beta) * pi(o_t)

Because u_t is drawn independently of o_t at the same step, pi(o_t) is
uncorrelated with u_t, so the OLS slope of executed-on-commanded is an
unbiased estimate of beta:

    beta_hat = Cov(a_t, u_t) / Var(u_t)          (per action dim, averaged)

Requirement: the practitioner must be able to observe executed actions
(actuator feedback / joint-torque sensing).  Where only commanded actions
are visible, beta is NOT identifiable from probe data and this diagnostic
does not apply -- that is a scope limit, stated rather than papered over.

A second, weaker diagnostic needs no actuator feedback but only detects
gross violations: KL(executed-action distribution || Uniform).  It is
reported alongside for comparison.

Honest note on what this validates
----------------------------------
In simulation the mixing is linear by construction, so recovering it with
a linear regression is close to tautological.  What this run establishes
is (a) the estimator is unbiased at the sample sizes the probe actually
uses, and (b) the KL surrogate is a much weaker instrument -- not that
override failure is linear on real hardware.

Usage:
    python -m experiments.run_beta_estimator
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

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BETAS = [1.0, 0.75, 0.5, 0.25, 0.1]
SEEDS = [0, 1, 2, 3, 4]


def collect_action_pairs(env, policy, beta: float, n_steps: int, seed: int):
    """Roll out the intervention branch, logging (commanded, executed)."""
    rng = np.random.RandomState(seed)
    lo = env.action_space.low.flatten()
    hi = env.action_space.high.flatten()
    obs, _ = env.reset(seed=seed)
    U, A = [], []
    for _ in range(n_steps):
        pi_a = np.asarray(policy(obs), dtype=np.float64).flatten()
        u = rng.uniform(lo, hi)
        a = u if beta >= 1.0 else beta * u + (1.0 - beta) * pi_a
        a = np.clip(a, lo, hi)
        U.append(u)
        A.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    return np.asarray(U), np.asarray(A), lo, hi


def estimate_beta(U: np.ndarray, A: np.ndarray) -> float:
    """OLS slope of executed on commanded, averaged over action dims."""
    slopes = []
    for j in range(U.shape[1]):
        u, a = U[:, j], A[:, j]
        # ddof must match the numerator's: np.cov defaults to ddof=1 while
        # np.var defaults to ddof=0, and mixing them inflates every slope
        # by n/(n-1) -- a constant +0.00033 at n=3000.
        v = np.var(u, ddof=1)
        if v > 1e-12:
            slopes.append(float(np.cov(a, u, ddof=1)[0, 1] / v))
    return float(np.mean(slopes)) if slopes else float("nan")


def kl_to_uniform(A: np.ndarray, lo, hi, n_bins: int = 50) -> float:
    """Histogram estimate of KL(executed || Uniform), averaged over dims."""
    kls = []
    for j in range(A.shape[1]):
        h, edges = np.histogram(A[:, j], bins=n_bins,
                                range=(float(lo[j]), float(hi[j])),
                                density=True)
        w = edges[1] - edges[0]
        p = h * w
        p = p[p > 0]
        q = w / (hi[j] - lo[j])
        kls.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(kls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="walker")
    ap.add_argument("--task", type=str, default="walk")
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n_steps", type=int, default=3000)
    ap.add_argument("--results_dir", type=str, default="results/beta_est")
    args = ap.parse_args()

    from experiments.confounded_distractors import (
        make_confounded_env, StructuredProbePolicy, ConfoundedPolicy)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for beta in BETAS:
        for seed in SEEDS:
            env = make_confounded_env(args.domain, args.task,
                                      base_config="medium", n_confounded=12,
                                      lam=args.lam, seed=seed + 2000)
            pol = ConfoundedPolicy(env, StructuredProbePolicy(env, seed=seed),
                                   kappa=args.kappa, seed=seed)
            U, A, lo, hi = collect_action_pairs(env, pol, beta,
                                                args.n_steps, seed)
            rows.append({"beta": beta, "seed": seed,
                         "beta_hat": estimate_beta(U, A),
                         "kl_to_uniform": kl_to_uniform(A, lo, hi)})
            env.close()
        bs = [r["beta_hat"] for r in rows if r["beta"] == beta]
        ks = [r["kl_to_uniform"] for r in rows if r["beta"] == beta]
        logger.info(f"beta={beta:<5g} -> beta_hat={np.mean(bs):.4f}"
                    f"±{np.std(bs):.4f}  (error {np.mean(bs)-beta:+.4f})"
                    f"  KL={np.mean(ks):.3f}")

    with open(out / "beta_estimates.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ── report ────────────────────────────────────────────────────────
    lines = ["# Beta estimator validation", "",
             "`beta_hat` = OLS slope of executed on commanded actions,"
             f" {args.n_steps} steps, {len(SEEDS)} seeds.", "",
             "| true beta | beta_hat | error | KL(executed‖Unif) |",
             "|---|---|---|---|"]
    for beta in BETAS:
        bs = [r["beta_hat"] for r in rows if r["beta"] == beta]
        ks = [r["kl_to_uniform"] for r in rows if r["beta"] == beta]
        lines.append(f"| {beta:g} | {np.mean(bs):.4f}±{np.std(bs):.4f} "
                     f"| {np.mean(bs)-beta:+.4f} | {np.mean(ks):.3f} |")
    report = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
