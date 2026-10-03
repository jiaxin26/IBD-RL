#!/usr/bin/env python
"""
IBD Benchmark — DMControl Experiments
=======================================

Evaluates IBD against baselines on DMControl tasks with distractors.

Usage::

    # Single task, single method
    python run_dmcontrol.py --domain walker --task walk --method ibd --distractors medium

    # Full benchmark grid
    python run_dmcontrol.py --tier 1 --parallel 3

    # Quick smoke test
    python run_dmcontrol.py --domain cartpole --task swingup --method ibd --distractors easy \\
        --total_steps 50000 --seeds 1

Methods:
    full_state   — SAC on full observation (no masking)
    oracle       — SAC on true dims only (upper bound)
    ibd          — Two-phase: scout SAC → joint-intervention probe → SAC on discovered dims
    random_mask  — SAC on random subset of dims
    mutual_info  — SAC on MI-selected dims
    variance     — SAC on variance-selected dims
    cond_mi      — SAC on dims selected by conditional MI (learned forward model)
    grad_attr    — SAC on dims selected by gradient attribution (learned dynamics + ∂f/∂a)

Budget accounting:
    All methods train the **main** SAC for exactly ``total_steps``.
    IBD has additional one-time overhead (scout training + probing),
    which is reported separately as ``ibd_overhead_s`` in results.
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
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    domain: str = "walker"
    task: str = "walk"
    distractors: str = "medium"
    method: str = "ibd"
    total_steps: int = 300_000
    eval_interval: int = 10_000      # finer-grained for learning curves
    eval_episodes: int = 5
    # IBD-specific
    ibd_scout_steps: int = 80_000    # scout policy training budget
    ibd_n_baseline: int = 80         # more trajs for statistical power
    ibd_n_intervention: int = 80
    ibd_traj_length: int = 200
    ibd_horizons: str = "1,5,10"
    ibd_n_permutations: int = 5000
    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 142, 242, 342, 442])
    results_dir: str = "results/dmcontrol"
    # RL algorithm
    rl_algo: str = "sac"           # "sac" or "td3"
    # SAC/TD3 hyperparams
    sac_lr: float = 3e-4
    sac_batch_size: int = 256
    sac_buffer_size: int = 300_000

    @property
    def horizons(self) -> List[int]:
        return [int(x) for x in self.ibd_horizons.split(",")]


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-SEED RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(cfg: ExperimentConfig, seed: int) -> Dict:
    """Run one (method, task, distractor, seed) experiment."""

    # ── per-seed resume ───────────────────────────────────────────────
    out_dir = Path(cfg.results_dir) / "per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_fname = (f"{cfg.domain}_{cfg.task}_{cfg.distractors}"
                  f"_{cfg.method}"
                  f"{'_' + cfg.rl_algo if cfg.rl_algo != 'sac' else ''}"
                  f"_s{seed}.json")
    seed_path = out_dir / seed_fname
    if seed_path.exists():
        logger.info(f"  SKIP (exists): {seed_fname}")
        with open(seed_path) as f:
            return json.load(f)

    import torch
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from experiments.dmcontrol_distractors import make_env
    from experiments.baselines import (
        NoMask, OracleMask, RandomMask, MutualInfoSelector, VarianceSelector,
        ConditionalMISelector, GradientAttributionSelector,
        MultistepInverseDynamicsSelector)
    from ibd.probe import IBDProbe
    from ibd.wrappers import DimSelectWrapper

    logger.info("=" * 70)
    logger.info(f"{cfg.domain}_{cfg.task} | {cfg.distractors} | "
                f"{cfg.method} | seed={seed}")
    logger.info("=" * 70)

    # ── create base envs ──────────────────────────────────────────────
    base_train = make_env(cfg.domain, cfg.task, cfg.distractors, seed=seed)
    base_eval = make_env(cfg.domain, cfg.task, cfg.distractors,
                         seed=seed + 1000)
    probe_env = make_env(cfg.domain, cfg.task, cfg.distractors,
                         seed=seed + 2000)

    obs_dim = base_train.observation_space.shape[0]
    true_soi = base_train.true_dims
    n_true = base_train.true_obs_dim

    logger.info(f"Obs dim: {obs_dim}  (true: {n_true}, "
                f"distractors: {obs_dim - n_true})")
    logger.info(f"Action dim: {base_train.action_space.shape[0]}")
    logger.info(f"True SoI: {sorted(true_soi)}")

    # ── determine selected dims per method ────────────────────────────
    selected_dims = None     # None = use full obs (full_state)
    boundary_metrics = {}
    ibd_info = {}
    ibd_overhead_s = 0.0     # wall-time overhead (scout + probing)

    if cfg.method == "full_state":
        pass

    elif cfg.method == "oracle":
        selected_dims = np.array(sorted(true_soi))
        boundary_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                            "tp": n_true, "fp": 0, "fn": 0}
        logger.info(f"Oracle: selecting {len(selected_dims)} dims")

    elif cfg.method == "ibd":
        t_ibd_start = time.time()

        # Phase 1: train scout policy on full obs
        logger.info(f"IBD Phase 1: training scout for "
                    f"{cfg.ibd_scout_steps} steps...")
        scout_env = make_env(cfg.domain, cfg.task, cfg.distractors,
                             seed=seed + 3000)
        scout = SAC("MlpPolicy", scout_env,
                    learning_rate=cfg.sac_lr,
                    batch_size=cfg.sac_batch_size,
                    buffer_size=cfg.ibd_scout_steps,
                    seed=seed, verbose=0)
        scout.learn(total_timesteps=cfg.ibd_scout_steps)

        def scout_policy(obs):
            with torch.no_grad():
                action, _ = scout.predict(obs, deterministic=False)
            return action

        # Phase 2: joint-intervention IBD
        logger.info("IBD Phase 2: joint-intervention probing...")
        probe = IBDProbe(
            probe_env,
            n_baseline=cfg.ibd_n_baseline,
            n_intervention=cfg.ibd_n_intervention,
            traj_length=cfg.ibd_traj_length,
            horizons=cfg.horizons,
            n_permutations=cfg.ibd_n_permutations,
            seed=seed)
        soi, ibd_info = probe.discover_joint(policy=scout_policy)

        ibd_overhead_s = time.time() - t_ibd_start
        logger.info(f"IBD overhead: {ibd_overhead_s:.0f}s "
                    f"(scout={cfg.ibd_scout_steps} steps + probing)")
        logger.info(f"IBD discovered {len(soi)}/{obs_dim} causal dims: "
                    f"{sorted(soi)}")

        if len(soi) == 0:
            logger.warning("IBD found no causal dims! Falling back to "
                           "structured random probe...")
            # Retry with the built-in structured random policy
            soi, ibd_info = probe.discover_joint(policy=None)
            logger.info(f"Retry: {len(soi)}/{obs_dim} causal dims: "
                        f"{sorted(soi)}")

        if len(soi) == 0:
            logger.warning("IBD still empty — using all dims (no masking)")
        else:
            selected_dims = np.array(sorted(soi))
            tp = len(soi & true_soi)
            fp = len(soi - true_soi)
            fn = len(true_soi - soi)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            boundary_metrics = {"precision": prec, "recall": rec,
                                "f1": f1, "tp": tp, "fp": fp, "fn": fn}
            logger.info(f"  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        # Free scout memory
        del scout, scout_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif cfg.method == "random_mask":
        selector = RandomMask(obs_dim, n_select=n_true, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)

    elif cfg.method == "mutual_info":
        selector = MutualInfoSelector(
            probe_env, n_episodes=50, episode_length=200,
            n_select=n_true, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)
        logger.info(f"MI mask: {sorted(mask.soi_dims)}")

    elif cfg.method == "variance":
        selector = VarianceSelector(
            probe_env, n_episodes=50, episode_length=200,
            n_select=n_true, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)
        logger.info(f"Variance mask: {sorted(mask.soi_dims)}")

    elif cfg.method == "cond_mi":
        selector = ConditionalMISelector(
            probe_env, n_episodes=200, episode_length=200,
            n_select=n_true, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)
        logger.info(f"Cond MI mask: {sorted(mask.soi_dims)}")

    elif cfg.method == "grad_attr":
        selector = GradientAttributionSelector(
            probe_env, n_episodes=200, episode_length=200,
            n_select=n_true, hidden=128, epochs=80, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)
        logger.info(f"GradAttr mask: {sorted(mask.soi_dims)}")

    elif cfg.method == "inverse_dyn":
        selector = MultistepInverseDynamicsSelector(
            probe_env, n_episodes=200, episode_length=200,
            horizon_k=3, n_select=n_true, hidden=128, epochs=80, seed=seed)
        mask = selector.discover()
        selected_dims = np.array(sorted(mask.soi_dims))
        boundary_metrics = mask.evaluate(true_soi)
        logger.info(f"InverseDyn mask: {sorted(mask.soi_dims)}")

    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # ── wrap envs with dim selection ──────────────────────────────────
    if selected_dims is not None:
        train_env = DimSelectWrapper(base_train, selected_dims)
        eval_env = DimSelectWrapper(base_eval, selected_dims)
        effective_dim = len(selected_dims)
        logger.info(f"DimSelect: {obs_dim} → {effective_dim} dims")
    else:
        train_env = base_train
        eval_env = base_eval
        effective_dim = obs_dim

    # ── create RL agent (input dim matches selected dims) ──────────────
    if cfg.rl_algo == "td3":
        n_actions = train_env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            "MlpPolicy", train_env,
            learning_rate=cfg.sac_lr,
            batch_size=cfg.sac_batch_size,
            buffer_size=cfg.sac_buffer_size,
            action_noise=action_noise,
            seed=seed, verbose=0)
    else:
        model = SAC(
            "MlpPolicy", train_env,
            learning_rate=cfg.sac_lr,
            batch_size=cfg.sac_batch_size,
            buffer_size=cfg.sac_buffer_size,
            seed=seed, verbose=0)

    # ── train with periodic eval ──────────────────────────────────────
    eval_returns = []
    logger.info(f"Training for {cfg.total_steps} steps "
                f"(input dim = {effective_dim})...")
    t0 = time.time()

    steps_done = 0
    while steps_done < cfg.total_steps:
        chunk = min(cfg.eval_interval, cfg.total_steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk

        # Evaluate
        ep_returns = []
        for _ in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            total_r = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = eval_env.step(action)
                total_r += r
                done = terminated or truncated
            ep_returns.append(total_r)

        mean_r = float(np.mean(ep_returns))
        std_r = float(np.std(ep_returns))
        eval_returns.append({"step": steps_done, "mean": mean_r,
                             "std": std_r})
        # Only log every eval_interval (not too noisy)
        if steps_done % 50_000 == 0 or steps_done == cfg.total_steps:
            logger.info(f"  Step {steps_done:>7d} | Return: "
                        f"{mean_r:.1f} +/- {std_r:.1f}")

    train_time = time.time() - t0
    logger.info(f"Training complete in {train_time:.0f}s")

    if boundary_metrics:
        logger.info(
            f"Boundary: P={boundary_metrics.get('precision', 0):.3f} "
            f"R={boundary_metrics.get('recall', 0):.3f} "
            f"F1={boundary_metrics.get('f1', 0):.3f}")

    # ── results ───────────────────────────────────────────────────────
    result = {
        "domain": cfg.domain,
        "task": cfg.task,
        "distractors": cfg.distractors,
        "method": cfg.method,
        "rl_algo": cfg.rl_algo,
        "seed": seed,
        "obs_dim": obs_dim,
        "true_obs_dim": n_true,
        "effective_dim": effective_dim,
        "eval_returns": eval_returns,
        "final_return": eval_returns[-1]["mean"] if eval_returns else 0,
        "boundary_metrics": boundary_metrics,
        "ibd_overhead_s": ibd_overhead_s,
        "train_time_s": train_time,
    }

    with open(seed_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  Saved: {seed_fname}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SEED AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: ExperimentConfig) -> Dict:
    """Run all seeds and aggregate."""
    all_results = []
    for seed in cfg.seeds:
        result = run_single(cfg, seed)
        all_results.append(result)

    # Aggregate
    final_returns = [r["final_return"] for r in all_results]
    agg = {
        "domain": cfg.domain,
        "task": cfg.task,
        "distractors": cfg.distractors,
        "method": cfg.method,
        "n_seeds": len(cfg.seeds),
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "per_seed": all_results,
    }

    bm_keys = ["precision", "recall", "f1"]
    for k in bm_keys:
        vals = [r["boundary_metrics"].get(k, 0)
                for r in all_results if r.get("boundary_metrics")]
        if vals:
            agg[f"boundary_{k}_mean"] = float(np.mean(vals))
            agg[f"boundary_{k}_std"] = float(np.std(vals))

    logger.info("=" * 70)
    logger.info(f"SUMMARY: {cfg.domain}_{cfg.task} | "
                f"{cfg.distractors} | {cfg.method}")
    logger.info(f"  Return: {agg['final_return_mean']:.1f} "
                f"+/- {agg['final_return_std']:.1f}")
    for k in bm_keys:
        mk = f"boundary_{k}_mean"
        sk = f"boundary_{k}_std"
        if mk in agg:
            logger.info(f"  {k}: {agg[mk]:.3f} +/- {agg[sk]:.3f}")
    logger.info("=" * 70)

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"{cfg.domain}_{cfg.task}_{cfg.distractors}"
             f"_{cfg.method}.json")
    with open(out_dir / fname, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    logger.info(f"Saved to {out_dir / fname}")

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# TIERED BENCHMARK PLANS
# ═══════════════════════════════════════════════════════════════════════════════

TIER_1 = {
    "name": "Core (IBD vs baselines on key tasks, medium distractors)",
    "tasks": [("walker", "walk"), ("cheetah", "run")],
    "distractors": ["medium"],
    "methods": ["full_state", "ibd", "oracle"],
    "n_seeds": 5,
    "total_steps": 300_000,
}

TIER_2 = {
    "name": "Hard distractors (where IBD advantage is largest)",
    "tasks": [("walker", "walk"), ("cheetah", "run")],
    "distractors": ["hard"],
    "methods": ["full_state", "ibd", "oracle"],
    "n_seeds": 5,
    "total_steps": 300_000,
}

TIER_3 = {
    "name": "More tasks + observational baselines",
    "tasks": [("cartpole", "swingup"), ("finger", "spin"),
              ("hopper", "hop"), ("reacher", "hard")],
    "distractors": ["medium"],
    "methods": ["full_state", "ibd", "oracle", "mutual_info", "variance", "grad_attr"],
    "n_seeds": 5,
    "total_steps": 300_000,
}

TIER_4 = {
    "name": "TD3 backend validation (algorithm-agnostic claim)",
    "tasks": [("walker", "walk"), ("cheetah", "run")],
    "distractors": ["hard"],
    "methods": ["full_state", "ibd", "oracle"],
    "n_seeds": 5,
    "total_steps": 300_000,
    "rl_algo": "td3",
}


def _build_run_list(tier: dict, cfg_base: ExperimentConfig) -> list:
    """Build list of configs from a tier spec."""
    runs = []
    seeds = [42 + i * 100 for i in range(tier["n_seeds"])]
    rl_algo = tier.get("rl_algo", "sac")
    for domain, task in tier["tasks"]:
        for dist in tier["distractors"]:
            for method in tier["methods"]:
                cfg = ExperimentConfig(
                    domain=domain, task=task, distractors=dist,
                    method=method,
                    total_steps=tier["total_steps"],
                    seeds=seeds,
                    rl_algo=rl_algo,
                    results_dir=cfg_base.results_dir,
                    eval_interval=cfg_base.eval_interval,
                    ibd_scout_steps=cfg_base.ibd_scout_steps,
                    ibd_n_baseline=cfg_base.ibd_n_baseline,
                    ibd_n_intervention=cfg_base.ibd_n_intervention,
                    ibd_traj_length=cfg_base.ibd_traj_length,
                    ibd_horizons=cfg_base.ibd_horizons,
                    ibd_n_permutations=cfg_base.ibd_n_permutations)
                runs.append(cfg)
    return runs


def run_full_benchmark(cfg_base: ExperimentConfig, tier_name: str = "1"):
    """Run a tiered benchmark plan with resume support."""
    tiers = {"1": TIER_1, "2": TIER_2, "3": TIER_3, "4": TIER_4}

    if tier_name == "all":
        selected = [TIER_1, TIER_2, TIER_3, TIER_4]
    elif tier_name in tiers:
        selected = [tiers[tier_name]]
    else:
        raise ValueError(f"Unknown tier: {tier_name}")

    all_agg = []
    out_dir = Path(cfg_base.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tier in selected:
        logger.info("=" * 70)
        logger.info(f"TIER: {tier['name']}")
        n_runs = (len(tier['tasks']) * len(tier['distractors'])
                  * len(tier['methods']))
        n_total = n_runs * tier['n_seeds']
        logger.info(f"  {n_runs} configs × {tier['n_seeds']} seeds "
                    f"= {n_total} runs")
        logger.info("=" * 70)

        run_list = _build_run_list(tier, cfg_base)
        for i, cfg in enumerate(run_list):
            logger.info(f"\n  [{i+1}/{len(run_list)}] "
                        f"{cfg.domain}_{cfg.task} | {cfg.distractors} | "
                        f"{cfg.method}")
            try:
                agg = run_experiment(cfg)
                all_agg.append(agg)
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

    _print_comparison_table(all_agg)
    _aggregate_and_print(cfg_base, selected)
    return all_agg


def _print_comparison_table(results: List[Dict]):
    """Print a task × method comparison table."""
    logger.info("\n" + "=" * 90)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 90)

    by_env = defaultdict(dict)
    for r in results:
        key = f"{r['domain']}_{r['task']}_{r['distractors']}"
        by_env[key][r["method"]] = r

    header = (f"{'Environment':<35} {'Method':<15} "
              f"{'Return':>15} {'P/R/F1':>18}")
    logger.info(header)
    logger.info("-" * len(header))

    for env_key in sorted(by_env.keys()):
        methods = by_env[env_key]
        for method_name in ["full_state", "oracle", "ibd",
                            "random_mask", "mutual_info", "variance",
                            "cond_mi", "grad_attr"]:
            r = methods.get(method_name)
            if r is None:
                continue
            ret_str = (f"{r['final_return_mean']:>7.1f}±"
                       f"{r['final_return_std']:<5.1f}")
            bm_str = ""
            if "boundary_f1_mean" in r:
                bm_str = (f"{r.get('boundary_precision_mean', 0):.2f}/"
                          f"{r.get('boundary_recall_mean', 0):.2f}/"
                          f"{r.get('boundary_f1_mean', 0):.2f}")
            logger.info(f"{env_key:<35} {method_name:<15} "
                        f"{ret_str:>15} {bm_str:>18}")
        logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════

def run_parallel_tier(cfg_base: ExperimentConfig, tier_name: str,
                      n_workers: int):
    """Launch tier runs in parallel using subprocess."""
    tiers = {"1": TIER_1, "2": TIER_2, "3": TIER_3, "4": TIER_4}
    if tier_name == "all":
        selected = [TIER_1, TIER_2, TIER_3, TIER_4]
    else:
        selected = [tiers[tier_name]]

    # Build flat job list
    jobs = []
    for tier in selected:
        seeds = [42 + i * 100 for i in range(tier["n_seeds"])]
        rl_algo = tier.get("rl_algo", "sac")
        for domain, task in tier["tasks"]:
            for dist in tier["distractors"]:
                for method in tier["methods"]:
                    for seed in seeds:
                        algo_suffix = f"_{rl_algo}" if rl_algo != "sac" else ""
                        fname = (f"{domain}_{task}_{dist}"
                                 f"_{method}{algo_suffix}_s{seed}.json")
                        fpath = (Path(cfg_base.results_dir)
                                 / "per_seed" / fname)
                        if fpath.exists():
                            logger.info(f"  SKIP (done): {fname}")
                            continue
                        jobs.append((domain, task, dist, method,
                                     seed, tier["total_steps"], rl_algo))

    if not jobs:
        logger.info("All jobs already complete!")
        _aggregate_and_print(cfg_base, selected)
        return

    logger.info(f"\n{len(jobs)} jobs to run, {n_workers} workers\n")

    script = str(Path(__file__).resolve())
    active = []

    def _drain(wait_all=False):
        remaining = []
        for proc, label in active:
            if wait_all or proc.poll() is not None:
                if wait_all:
                    proc.wait()
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                logger.info(f"  {status}: {label}")
            else:
                remaining.append((proc, label))
        return remaining

    for domain, task, dist, method, seed, steps, rl_algo in jobs:
        while len(active) >= n_workers:
            time.sleep(5)
            active = _drain()

        label = f"{domain}_{task}_{dist}_{method}{'_' + rl_algo if rl_algo != 'sac' else ''}_s{seed}"
        log_dir = Path(cfg_base.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{label}.log"

        cmd = [
            sys.executable, script,
            "--domain", domain, "--task", task,
            "--distractors", dist, "--method", method,
            "--algo", rl_algo,
            "--seeds", "1",
            "--total_steps", str(steps),
            "--eval_interval", str(cfg_base.eval_interval),
            "--results_dir", cfg_base.results_dir,
            "--ibd_scout_steps", str(cfg_base.ibd_scout_steps),
            "--ibd_n_baseline", str(cfg_base.ibd_n_baseline),
            "--ibd_n_intervention", str(cfg_base.ibd_n_intervention),
            "--ibd_traj_length", str(cfg_base.ibd_traj_length),
            "--ibd_horizons", cfg_base.ibd_horizons,
            "--ibd_n_permutations", str(cfg_base.ibd_n_permutations),
            "--_seed", str(seed),
        ]

        logger.info(f"  LAUNCH: {label}")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        active.append((proc, label))

    active = _drain(wait_all=True)
    _aggregate_and_print(cfg_base, selected)


def _aggregate_and_print(cfg_base, tiers):
    """Aggregate per-seed results and print comparison table."""
    out_dir = Path(cfg_base.results_dir)
    seed_dir = out_dir / "per_seed"
    if not seed_dir.exists():
        return

    groups = defaultdict(list)
    for f in seed_dir.glob("*.json"):
        with open(f) as fh:
            r = json.load(fh)
        algo = r.get("rl_algo", "sac")
        algo_suffix = f"_{algo}" if algo != "sac" else ""
        key = f"{r['domain']}_{r['task']}_{r['distractors']}_{r['method']}{algo_suffix}"
        groups[key].append(r)

    all_agg = []
    for key, results in sorted(groups.items()):
        final_returns = [r["final_return"] for r in results]
        agg = {
            "domain": results[0]["domain"],
            "task": results[0]["task"],
            "distractors": results[0]["distractors"],
            "method": results[0]["method"],
            "rl_algo": results[0].get("rl_algo", "sac"),
            "n_seeds": len(results),
            "final_return_mean": float(np.mean(final_returns)),
            "final_return_std": float(np.std(final_returns)),
        }
        for k in ["precision", "recall", "f1"]:
            vals = [r["boundary_metrics"].get(k, 0)
                    for r in results if r.get("boundary_metrics")]
            if vals:
                agg[f"boundary_{k}_mean"] = float(np.mean(vals))
                agg[f"boundary_{k}_std"] = float(np.std(vals))
        all_agg.append(agg)

        algo_suffix = f"_{agg['rl_algo']}" if agg.get('rl_algo', 'sac') != 'sac' else ''
        fname = (f"{agg['domain']}_{agg['task']}"
                 f"_{agg['distractors']}_{agg['method']}{algo_suffix}.json")
        with open(out_dir / fname, "w") as f:
            json.dump(agg, f, indent=2, default=str)

    _print_comparison_table(all_agg)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IBD Benchmark — DMControl Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dmcontrol.py --tier 1 --parallel 4
  python run_dmcontrol.py --domain walker --task walk --method ibd
  python run_dmcontrol.py --aggregate

Tiers:
  1  Core:  walker/cheetah × medium × {full_state, ibd, oracle}  (5 seeds)
  2  Hard:  walker/cheetah × hard   × {full_state, ibd, oracle}  (5 seeds)
  3  Broad: 4 tasks × medium × {full_state, ibd, oracle, MI, var} (5 seeds)
  4  TD3:   walker/cheetah × hard   × {full_state, ibd, oracle}  (5 seeds, TD3 backend)
""")

    parser.add_argument("--tier", type=str, default=None,
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--parallel", type=int, default=1)

    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--distractors", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--method", type=str, default="ibd",
                        choices=["full_state", "oracle", "ibd",
                                 "random_mask", "mutual_info", "variance",
                                 "cond_mi", "grad_attr", "inverse_dyn"])
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "td3"],
                        help="RL backend algorithm")

    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)

    parser.add_argument("--ibd_scout_steps", type=int, default=80_000)
    parser.add_argument("--ibd_n_baseline", type=int, default=80)
    parser.add_argument("--ibd_n_intervention", type=int, default=80)
    parser.add_argument("--ibd_traj_length", type=int, default=200)
    parser.add_argument("--ibd_horizons", type=str, default="1,5,10")
    parser.add_argument("--ibd_n_permutations", type=int, default=5000)

    parser.add_argument("--results_dir", type=str, default="results/dmcontrol")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--_seed", type=int, default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument("--full_benchmark", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    seed_list = [42 + i * 100 for i in range(args.seeds)]
    cfg = ExperimentConfig(
        domain=args.domain, task=args.task, distractors=args.distractors,
        method=args.method, total_steps=args.total_steps,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        seeds=seed_list,
        rl_algo=args.algo,
        ibd_scout_steps=args.ibd_scout_steps,
        ibd_n_baseline=args.ibd_n_baseline,
        ibd_n_intervention=args.ibd_n_intervention,
        ibd_traj_length=args.ibd_traj_length,
        ibd_horizons=args.ibd_horizons,
        ibd_n_permutations=args.ibd_n_permutations,
        results_dir=args.results_dir)

    if args.aggregate:
        _aggregate_and_print(cfg, [TIER_1, TIER_2, TIER_3, TIER_4])

    elif args._seed is not None:
        cfg.seeds = [args._seed]
        run_single(cfg, args._seed)

    elif args.tier:
        if args.parallel > 1:
            run_parallel_tier(cfg, args.tier, args.parallel)
        else:
            run_full_benchmark(cfg, tier_name=args.tier)

    elif args.full_benchmark:
        run_full_benchmark(cfg, tier_name="all")

    else:
        run_experiment(cfg)


if __name__ == "__main__":
    main()