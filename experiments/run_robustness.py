#!/usr/bin/env python3
"""
Robustness Study: Partial Controllability & Weak Causal Effects
================================================================

Tests IBD's detection limits on two axes:
  1. Partial controllability: dims that mix action-dependent + exogenous dynamics
  2. Weak causal effects: action influence scaled by coefficient β

This script ONLY runs IBD probes (scout training + trajectory collection +
statistical testing). It does NOT run full 300K-step RL training.

Estimated wall time: ~5 min per (α, seed) pair.
  - 10 α values × 3 seeds × 2 tasks = 60 probes ≈ 5–6 hours total.

Usage:
    # Full sweep (two tasks, 3 seeds)
    python -m experiments.run_robustness

    # Quick test (one task, 1 seed, 3 alpha values)
    python -m experiments.run_robustness --quick

    # Custom
    python -m experiments.run_robustness --domain cheetah --task run \\
        --alphas 0.0,0.05,0.1,0.2,0.5,1.0 --seeds 42,142,242

Output:
    results/robustness/partial_controllability_{domain}_{task}.json
    results/robustness/detection_threshold_{domain}_{task}.json
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
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIAL-CONTROLLABILITY ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class PartialControllableSource:
    """
    Observation dimension with mixed causal + exogenous dynamics.

        x_{t+1} = α · action_effect(a_t) + (1-α) · OU_exogenous(z_t)

    When α=1: purely causal (action fully determines dynamics)
    When α=0: purely exogenous (no causal path from action)
    When 0<α<1: partial controllability

    The action_effect maps a randomly chosen action dimension through
    a nonlinear function, then integrates it into a state variable.
    This ensures the causal signal is realistic (not trivially detectable).
    """

    def __init__(self, dim: int, alpha: float = 0.5,
                 action_dim: int = 6, seed: int = 42):
        self.dim = dim
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

        # Which action dim drives each partial dim (random assignment)
        self.action_mapping = self.rng.randint(0, action_dim, dim)

        # Nonlinear transform coefficients per dim
        self._gain = self.rng.uniform(0.3, 1.5, dim)
        self._bias = self.rng.uniform(-0.5, 0.5, dim)

        # Exogenous OU process parameters (matched to typical DMC scale)
        self._ou_tau = self.rng.uniform(1.0, 4.0, dim)
        self._ou_sigma = self.rng.uniform(0.2, 0.6, dim)

        # Causal dynamics rate — must be large enough that the causal
        # component produces state changes comparable to the OU noise.
        # DMControl dt ≈ 0.01, so raw integration gives ~0.01/step.
        # We use an effective rate of 50× to match OU noise scale (~0.04/step).
        self._causal_rate = 50.0

        # State
        self._causal_state = np.zeros(dim)
        self._exo_state = np.zeros(dim)
        self._state = np.zeros(dim)

    def reset(self):
        self._causal_state = self.rng.normal(0, 0.1, self.dim)
        self._exo_state = self.rng.normal(0, 0.1, self.dim)
        self._state = (self.alpha * self._causal_state
                       + (1 - self.alpha) * self._exo_state)
        return self._state.copy()

    def step(self, dt: float, action: np.ndarray) -> np.ndarray:
        """Step with action input (causal pathway)."""
        action = np.asarray(action).flatten()

        # Causal component: nonlinear function of mapped action dims
        # Use effective_dt = dt * causal_rate so the causal signal is
        # on the same scale as the OU exogenous component
        effective_dt = dt * self._causal_rate
        for i in range(self.dim):
            a_idx = self.action_mapping[i]
            if a_idx < len(action):
                a_val = action[a_idx]
            else:
                a_val = 0.0
            # Leaky integration with nonlinear action drive
            drive = self._gain[i] * np.tanh(a_val + self._bias[i])
            self._causal_state[i] += (-self._causal_state[i] * 0.3
                                       + drive) * effective_dt
        # Clip causal state to prevent blow-up
        self._causal_state = np.clip(self._causal_state, -5.0, 5.0)

        # Exogenous OU component (independent of action)
        noise = self.rng.normal(0, 1.0, self.dim) * self._ou_sigma * np.sqrt(dt)
        self._exo_state += (-self._exo_state / self._ou_tau * dt + noise)

        # Mix
        self._state = (self.alpha * self._causal_state
                       + (1 - self.alpha) * self._exo_state)
        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()


class RobustnessEnv(gym.Env):
    """
    DMControl task augmented with:
      - N_partial partially-controllable dims (causal strength = α)
      - N_exo purely exogenous distractors (for discrimination challenge)

    Ground truth:
      - true_dims: original DMC dims (fully causal)
      - partial_dims: partially controllable (should be in SoI when α > 0)
      - exo_dims: purely exogenous (should NOT be in SoI)
    """

    metadata = {"render_modes": []}

    def __init__(self, domain_name: str, task_name: str,
                 alpha: float = 0.5,
                 n_partial: int = 6,
                 n_exo: int = 20,
                 seed: int = 42):
        super().__init__()
        from dm_control import suite

        self._dm_env = suite.load(
            domain_name, task_name, task_kwargs={"random": seed})
        self._domain = domain_name
        self._task = task_name

        # True observation dims
        obs_spec = self._dm_env.observation_spec()
        self._obs_keys = sorted(obs_spec.keys())
        self._true_obs_dim = sum(
            int(np.prod(obs_spec[k].shape)) for k in self._obs_keys)

        # Action spec
        action_spec = self._dm_env.action_spec()
        self._action_dim = int(np.prod(action_spec.shape))

        # Partial controllable dims
        self.alpha = alpha
        self._partial = PartialControllableSource(
            dim=n_partial, alpha=alpha,
            action_dim=self._action_dim, seed=seed + 5000)
        self._n_partial = n_partial

        # Pure exogenous OU distractors
        from experiments.dmcontrol_distractors import AutonomousOU, MimickingDistractor
        base_scale = 0.5 + 0.05 * self._true_obs_dim
        self._exo_sources = [
            AutonomousOU(dim=n_exo // 2, tau=2.0, sigma=0.3, seed=seed + 6000),
            MimickingDistractor(dim=n_exo - n_exo // 2,
                                ref_scale=base_scale, seed=seed + 7000),
        ]
        self._n_exo = sum(s.dim for s in self._exo_sources)

        # Total obs
        total_dim = self._true_obs_dim + self._n_partial + self._n_exo
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32)

        # Ground truth labels
        d0 = self._true_obs_dim
        d1 = d0 + self._n_partial
        d2 = d1 + self._n_exo

        self.true_dims = set(range(d0))               # fully causal
        self.partial_dims = set(range(d0, d1))         # partially causal
        self.exo_dims = set(range(d1, d2))             # not causal

        # For IBD evaluation: SoI = true_dims ∪ partial_dims (when α > 0)
        if alpha > 0:
            self.soi_dims = self.true_dims | self.partial_dims
        else:
            self.soi_dims = set(self.true_dims)  # α=0 → partial dims aren't causal
        self.distractor_dims = self.exo_dims if alpha > 0 else (self.exo_dims | self.partial_dims)

        self.true_obs_dim = self._true_obs_dim
        self._dt = self._dm_env.control_timestep()
        self._rng = np.random.RandomState(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        timestep = self._dm_env.reset()
        self._partial.reset()
        for s in self._exo_sources:
            s.reset()
        return self._build_obs(timestep), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action,
                         self.action_space.low, self.action_space.high)

        timestep = self._dm_env.step(action)

        # Partial controllable: receives action
        self._partial.step(self._dt, action=action)

        # Exogenous: no action
        for s in self._exo_sources:
            s.step(self._dt)

        obs = self._build_obs(timestep)
        reward = float(timestep.reward or 0.0)
        terminated = timestep.last()
        return obs, reward, terminated, False, {}

    def _build_obs(self, timestep) -> np.ndarray:
        parts = []
        for k in self._obs_keys:
            val = timestep.observation[k]
            parts.append(np.asarray(val, dtype=np.float32).flatten())
        true_obs = np.concatenate(parts)

        partial_obs = self._partial.state.astype(np.float32)
        exo_parts = [s.state.astype(np.float32) for s in self._exo_sources]

        return np.concatenate([true_obs, partial_obs] + exo_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN IBD PROBE ON ROBUSTNESS ENV
# ═══════════════════════════════════════════════════════════════════════════════

def run_ibd_probe(domain: str, task: str, alpha: float, seed: int,
                  scout_steps: int = 80_000,
                  n_trajs: int = 80, traj_len: int = 200,
                  ) -> Dict:
    """Run IBD probe and return detection metrics for partial dims."""
    from ibd.probe import IBDProbe

    env = RobustnessEnv(domain, task, alpha=alpha, seed=seed)
    scout_env = RobustnessEnv(domain, task, alpha=alpha, seed=seed + 100)
    probe_env = RobustnessEnv(domain, task, alpha=alpha, seed=seed + 200)

    obs_dim = env.observation_space.shape[0]
    true_soi = env.soi_dims
    partial_dims = env.partial_dims

    logger.info(f"  α={alpha:.2f} seed={seed} | obs={obs_dim} "
                f"(true={len(env.true_dims)}, partial={len(partial_dims)}, "
                f"exo={len(env.exo_dims)})")

    # Train scout
    import torch
    from stable_baselines3 import SAC

    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4,
                batch_size=256, buffer_size=scout_steps,
                seed=seed, verbose=0)
    scout.learn(total_timesteps=scout_steps)

    def scout_policy(obs):
        with torch.no_grad():
            action, _ = scout.predict(obs, deterministic=False)
        return action

    # Run IBD joint probe
    probe = IBDProbe(
        probe_env, n_baseline=n_trajs, n_intervention=n_trajs,
        traj_length=traj_len, horizons=[1, 5, 10], seed=seed)

    soi, info = probe.discover_joint(policy=scout_policy)

    # Evaluate detection of partial dims specifically
    partial_detected = soi & partial_dims
    partial_recall = (len(partial_detected) / len(partial_dims)
                      if len(partial_dims) > 0 else 1.0)

    # Overall boundary metrics
    tp = len(soi & true_soi)
    fp = len(soi - true_soi)
    fn = len(true_soi - soi)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    result = {
        "domain": domain,
        "task": task,
        "alpha": alpha,
        "seed": seed,
        "obs_dim": obs_dim,
        "n_true": len(env.true_dims),
        "n_partial": len(partial_dims),
        "n_exo": len(env.exo_dims),
        # Overall
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "soi_size": len(soi),
        # Partial-dim specific
        "partial_detected": len(partial_detected),
        "partial_total": len(partial_dims),
        "partial_recall": partial_recall,
        # Details
        "soi": sorted(soi),
        "true_soi": sorted(true_soi),
        "partial_dims_list": sorted(partial_dims),
    }

    # Cleanup
    for e in [env, scout_env, probe_env]:
        e.close()
    del scout

    logger.info(f"    → P={prec:.2f} R={rec:.2f} F1={f1:.2f} | "
                f"partial_recall={partial_recall:.2f} "
                f"({len(partial_detected)}/{len(partial_dims)})")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(domain: str, task: str,
              alphas: List[float],
              seeds: List[int],
              results_dir: str = "results/robustness") -> List[Dict]:
    """Run full α sweep for one task."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for alpha in alphas:
        for seed in seeds:
            # Check cache
            cache_name = f"probe_{domain}_{task}_a{alpha:.3f}_s{seed}.json"
            cache_path = out_dir / cache_name
            if cache_path.exists():
                logger.info(f"  SKIP (cached): {cache_name}")
                with open(cache_path) as f:
                    result = json.load(f)
                all_results.append(result)
                continue

            t0 = time.time()
            result = run_ibd_probe(domain, task, alpha, seed)
            result["time_s"] = time.time() - t0

            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            all_results.append(result)

    # Aggregate by alpha (string keys for JSON compatibility)
    aggregated = {}
    for alpha in alphas:
        runs = [r for r in all_results if abs(r["alpha"] - alpha) < 1e-6]
        if runs:
            aggregated[f"{alpha}"] = {
                "alpha": alpha,
                "precision_mean": float(np.mean([r["precision"] for r in runs])),
                "precision_std": float(np.std([r["precision"] for r in runs])),
                "recall_mean": float(np.mean([r["recall"] for r in runs])),
                "recall_std": float(np.std([r["recall"] for r in runs])),
                "f1_mean": float(np.mean([r["f1"] for r in runs])),
                "f1_std": float(np.std([r["f1"] for r in runs])),
                "partial_recall_mean": float(np.mean([r["partial_recall"] for r in runs])),
                "partial_recall_std": float(np.std([r["partial_recall"] for r in runs])),
                "n_seeds": len(runs),
            }

    # Save aggregate
    agg_path = out_dir / f"sweep_{domain}_{task}.json"
    with open(agg_path, "w") as f:
        json.dump({"task": f"{domain}_{task}",
                   "alphas": alphas,
                   "seeds": seeds,
                   "results": aggregated,
                   "per_run": all_results}, f, indent=2, default=str)
    logger.info(f"Saved: {agg_path}")

    # Print summary
    logger.info(f"\n{'α':>6} {'P':>8} {'R':>8} {'F1':>8} {'Part.R':>8}")
    logger.info("-" * 42)
    for alpha in alphas:
        a = aggregated.get(alpha, {})
        if a:
            logger.info(f"{alpha:>6.3f} "
                        f"{a['precision_mean']:>7.3f} "
                        f"{a['recall_mean']:>7.3f} "
                        f"{a['f1_mean']:>7.3f} "
                        f"{a['partial_recall_mean']:>7.3f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IBD Robustness Study: Partial Controllability")
    parser.add_argument("--domain", type=str, default=None,
                        help="Single domain (default: run both cheetah & walker)")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--alphas", type=str,
                        default="0.0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0",
                        help="Comma-separated mixing coefficients")
    parser.add_argument("--seeds", type=str, default="42,142,242",
                        help="Comma-separated seeds")
    parser.add_argument("--results_dir", type=str,
                        default="results/robustness")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 task, 1 seed, 3 alphas")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    if args.quick:
        alphas = [0.0, 0.1, 0.5]
        seeds = [42]
        tasks = [("cheetah", "run")]
    elif args.domain and args.task:
        tasks = [(args.domain, args.task)]
    else:
        tasks = [("cheetah", "run"), ("walker", "walk")]

    logger.info("=" * 60)
    logger.info("IBD ROBUSTNESS STUDY")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Alphas: {alphas}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Total probes: {len(tasks) * len(alphas) * len(seeds)}")
    logger.info("=" * 60)

    for domain, task in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK: {domain}_{task}")
        logger.info(f"{'='*60}")
        run_sweep(domain, task, alphas, seeds, args.results_dir)


if __name__ == "__main__":
    main()