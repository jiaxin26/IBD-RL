#!/usr/bin/env python
"""
Distractor Scaling Experiment
==============================

Sweeps the number of distractor dimensions at finer granularity than
easy/medium/hard to precisely locate the performance cliff for full-state
RL and confirm IBD's stability.

For each (task, n_distractor) pair, runs three methods:
  - full_state  (no masking — shows degradation)
  - ibd         (IBD mask — should be flat)
  - oracle      (ground truth — upper bound)

Usage:

    # Quick smoke test (1 task, 1 seed, 50K steps)
    python run_scaling.py --tasks walker_walk --seeds 1 --total_steps 50000

    # Full run for walker + cheetah (5 seeds, 300K steps)
    python run_scaling.py --tasks walker_walk cheetah_run --seeds 5

    # Add reacher for the extreme low-dim case
    python run_scaling.py --tasks walker_walk cheetah_run reacher_hard --seeds 5

    # Custom distractor counts
    python run_scaling.py --distractor_counts 6 18 30 50 80 100

    # Use GPU 0
    CUDA_VISIBLE_DEVICES=0 python run_scaling.py --tasks walker_walk --seeds 5

Outputs:
    results/scaling/         — per-seed JSON files
    results/scaling/summary/ — aggregated tables & plot data

Estimated wall time (single GPU):
    1 task × 8 counts × 3 methods × 5 seeds × ~20min = ~40 hours
    Can be parallelized across GPUs with --gpu_split
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

# Default sweep: original 3 points + 5 intermediate
DEFAULT_DISTRACTOR_COUNTS = [6, 12, 24, 36, 50, 75, 100, 150]

TASK_SPECS = {
    "walker_walk":   {"domain": "walker",   "task": "walk",    "true_dim": 24},
    "cheetah_run":   {"domain": "cheetah",  "task": "run",     "true_dim": 17},
    "reacher_hard":  {"domain": "reacher",  "task": "hard",    "true_dim": 6},
}

METHODS = ["full_state", "ibd", "oracle"]


@dataclass
class ScalingConfig:
    domain: str = "walker"
    task: str = "walk"
    n_distractors: int = 50
    method: str = "full_state"
    total_steps: int = 300_000
    eval_interval: int = 10_000
    eval_episodes: int = 5
    # IBD params
    ibd_scout_steps: int = 80_000
    ibd_n_trajs: int = 80
    ibd_traj_length: int = 200
    ibd_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    # RL
    sac_lr: float = 3e-4
    sac_batch_size: int = 256
    sac_buffer_size: int = 300_000
    # Seeds
    seed: int = 42
    results_dir: str = "results/scaling"


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(cfg: ScalingConfig) -> Dict:
    """Run one (method, task, n_distractors, seed) experiment."""

    out_dir = Path(cfg.results_dir) / "per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"{cfg.domain}_{cfg.task}_d{cfg.n_distractors}"
             f"_{cfg.method}_s{cfg.seed}.json")
    fpath = out_dir / fname

    if fpath.exists():
        logger.info(f"  SKIP (exists): {fname}")
        with open(fpath) as f:
            return json.load(f)

    import torch
    from stable_baselines3 import SAC
    from experiments.dmcontrol_distractors import make_env
    from ibd.probe import IBDProbe
    from ibd.wrappers import DimSelectWrapper

    logger.info("=" * 70)
    logger.info(f"{cfg.domain}_{cfg.task} | d={cfg.n_distractors} | "
                f"{cfg.method} | seed={cfg.seed}")
    logger.info("=" * 70)

    # ── create envs with integer distractor count ─────────────────────
    base_train = make_env(cfg.domain, cfg.task, cfg.n_distractors,
                          seed=cfg.seed)
    base_eval = make_env(cfg.domain, cfg.task, cfg.n_distractors,
                         seed=cfg.seed + 1000)
    probe_env = make_env(cfg.domain, cfg.task, cfg.n_distractors,
                         seed=cfg.seed + 2000)

    obs_dim = base_train.observation_space.shape[0]
    true_soi = base_train.true_dims
    n_true = base_train.true_obs_dim
    ratio = cfg.n_distractors / max(n_true, 1)

    logger.info(f"Obs: {obs_dim} (true={n_true}, dist={cfg.n_distractors}, "
                f"ratio={ratio:.1f}:1)")

    # ── method-specific mask discovery ────────────────────────────────
    selected_dims = None
    boundary_metrics = {}
    ibd_overhead_s = 0.0

    if cfg.method == "full_state":
        pass  # no masking

    elif cfg.method == "oracle":
        selected_dims = np.array(sorted(true_soi))
        boundary_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    elif cfg.method == "ibd":
        t0_ibd = time.time()

        # Scout
        logger.info(f"  IBD scout: {cfg.ibd_scout_steps} steps...")
        scout_env = make_env(cfg.domain, cfg.task, cfg.n_distractors,
                             seed=cfg.seed + 3000)
        scout = SAC("MlpPolicy", scout_env,
                    learning_rate=cfg.sac_lr,
                    batch_size=cfg.sac_batch_size,
                    buffer_size=cfg.ibd_scout_steps,
                    seed=cfg.seed, verbose=0)
        scout.learn(total_timesteps=cfg.ibd_scout_steps)

        def scout_policy(obs):
            with torch.no_grad():
                action, _ = scout.predict(obs, deterministic=False)
            return action

        # Joint intervention
        logger.info("  IBD probing...")
        probe = IBDProbe(
            probe_env,
            n_baseline=cfg.ibd_n_trajs,
            n_intervention=cfg.ibd_n_trajs,
            traj_length=cfg.ibd_traj_length,
            horizons=cfg.ibd_horizons,
            seed=cfg.seed)
        soi, _ = probe.discover_joint(policy=scout_policy)

        ibd_overhead_s = time.time() - t0_ibd
        logger.info(f"  IBD: {len(soi)}/{obs_dim} causal dims "
                    f"({ibd_overhead_s:.0f}s)")

        if len(soi) == 0:
            soi, _ = probe.discover_joint(policy=None)
            logger.info(f"  Retry: {len(soi)}/{obs_dim}")

        if len(soi) > 0:
            selected_dims = np.array(sorted(soi))
            tp = len(soi & true_soi)
            fp = len(soi - true_soi)
            fn = len(true_soi - soi)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            boundary_metrics = {"precision": p, "recall": r, "f1": f1}
            logger.info(f"  P={p:.3f} R={r:.3f} F1={f1:.3f}")

        del scout, scout_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── wrap env ──────────────────────────────────────────────────────
    if selected_dims is not None:
        train_env = DimSelectWrapper(base_train, selected_dims)
        eval_env = DimSelectWrapper(base_eval, selected_dims)
        eff_dim = len(selected_dims)
    else:
        train_env = base_train
        eval_env = base_eval
        eff_dim = obs_dim

    # ── train SAC ─────────────────────────────────────────────────────
    model = SAC("MlpPolicy", train_env,
                learning_rate=cfg.sac_lr,
                batch_size=cfg.sac_batch_size,
                buffer_size=cfg.sac_buffer_size,
                seed=cfg.seed, verbose=0)

    eval_returns = []
    logger.info(f"  Training {cfg.total_steps} steps (input={eff_dim}d)...")
    t0 = time.time()

    steps = 0
    while steps < cfg.total_steps:
        chunk = min(cfg.eval_interval, cfg.total_steps - steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps += chunk

        ep_returns = []
        for _ in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            total_r, done = 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = eval_env.step(action)
                total_r += r
                done = terminated or truncated
            ep_returns.append(total_r)

        mean_r = float(np.mean(ep_returns))
        eval_returns.append({"step": steps, "mean": mean_r,
                             "std": float(np.std(ep_returns))})

        if steps % 50_000 == 0 or steps >= cfg.total_steps:
            logger.info(f"    Step {steps:>7d} | {mean_r:.1f}")

    train_s = time.time() - t0

    # ── save ──────────────────────────────────────────────────────────
    result = {
        "domain": cfg.domain,
        "task": cfg.task,
        "n_distractors": cfg.n_distractors,
        "distractor_ratio": round(ratio, 2),
        "method": cfg.method,
        "seed": cfg.seed,
        "obs_dim": obs_dim,
        "true_obs_dim": n_true,
        "effective_dim": eff_dim,
        "final_return": eval_returns[-1]["mean"] if eval_returns else 0,
        "eval_returns": eval_returns,
        "boundary_metrics": boundary_metrics,
        "ibd_overhead_s": ibd_overhead_s,
        "train_time_s": train_s,
    }

    with open(fpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  Saved: {fname}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(results_dir: str, tasks: List[str],
                      distractor_counts: List[int]) -> Dict:
    """Load all per-seed results and aggregate into tables.

    Special handling for oracle: if only one distractor count was run,
    replicate its result across all counts (oracle is invariant to
    distractor count since it masks them all away).
    """
    per_seed_dir = Path(results_dir) / "per_seed"
    summary_dir = Path(results_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    tables = {}  # task -> {method -> {n_dist -> {"mean": ..., "std": ...}}}

    for task_key in tasks:
        spec = TASK_SPECS[task_key]
        domain, task = spec["domain"], spec["task"]
        tables[task_key] = {}

        for method in METHODS:
            tables[task_key][method] = {}
            for n_dist in distractor_counts:
                pattern = f"{domain}_{task}_d{n_dist}_{method}_s*.json"
                files = sorted(per_seed_dir.glob(pattern))
                if not files:
                    continue
                returns = []
                for f in files:
                    with open(f) as fh:
                        r = json.load(fh)
                    returns.append(r["final_return"])
                tables[task_key][method][n_dist] = {
                    "mean": float(np.mean(returns)),
                    "std": float(np.std(returns)),
                    "n_seeds": len(returns),
                    "ratio": round(n_dist / spec["true_dim"], 2),
                }

            # Oracle shortcut: if only 1 count was run, fill all counts
            if method == "oracle" and 0 < len(tables[task_key][method]) < len(distractor_counts):
                one_entry = list(tables[task_key][method].values())[0]
                for n_dist in distractor_counts:
                    if n_dist not in tables[task_key][method]:
                        tables[task_key][method][n_dist] = {
                            **one_entry,
                            "ratio": round(n_dist / spec["true_dim"], 2),
                            "extrapolated": True,
                        }

    # Save aggregated table
    with open(summary_dir / "scaling_table.json", "w") as f:
        json.dump(tables, f, indent=2)
    logger.info(f"Saved aggregated table to {summary_dir / 'scaling_table.json'}")

    # Print LaTeX-friendly table
    print("\n" + "=" * 80)
    for task_key, task_data in tables.items():
        print(f"\n{'─'*40}")
        print(f" {task_key}  (true_dim={TASK_SPECS[task_key]['true_dim']})")
        print(f"{'─'*40}")
        header = f"{'n_dist':>7} {'ratio':>6}"
        for m in METHODS:
            header += f" | {m:>15}"
        print(header)

        all_counts = sorted(set().union(
            *(d.keys() for d in task_data.values())))
        for n in all_counts:
            true_d = TASK_SPECS[task_key]["true_dim"]
            row = f"{n:>7d} {n/true_d:>5.1f}:1"
            for m in METHODS:
                entry = task_data.get(m, {}).get(n)
                if entry:
                    row += f" | {entry['mean']:>7.1f}±{entry['std']:>5.1f}"
                else:
                    row += f" | {'—':>13}"
            print(row)
    print("=" * 80)

    return tables


def generate_plot_script(results_dir: str, tasks: List[str]):
    """Write a self-contained matplotlib plotting script."""
    script = '''#!/usr/bin/env python
"""Plot distractor scaling curves from run_scaling.py results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = "{results_dir}"
TASKS = {tasks}

TASK_LABELS = {{
    "walker_walk": "Walker Walk (24 true dims)",
    "cheetah_run": "Cheetah Run (17 true dims)",
    "reacher_hard": "Reacher Hard (6 true dims)",
}}
TRUE_DIMS = {{
    "walker_walk": 24, "cheetah_run": 17, "reacher_hard": 6,
}}

METHODS = ["full_state", "ibd", "oracle"]
COLORS = {{"full_state": "#d62728", "ibd": "#1f77b4", "oracle": "#2ca02c"}}
LABELS = {{"full_state": "Full State", "ibd": "IBD (ours)", "oracle": "Oracle"}}
STYLES = {{"full_state": "o-", "ibd": "s-", "oracle": "--"}}

with open(Path(RESULTS_DIR) / "summary" / "scaling_table.json") as f:
    tables = json.load(f)

fig, axes = plt.subplots(1, len(TASKS), figsize=(5.5 * len(TASKS), 4.2),
                         sharey=False)
if len(TASKS) == 1:
    axes = [axes]

for ax, task in zip(axes, TASKS):
    td = TRUE_DIMS[task]
    for method in METHODS:
        data = tables.get(task, {{}}).get(method, {{}})
        if not data:
            continue
        counts = sorted(int(k) for k in data.keys())
        means = [data[str(c)]["mean"] for c in counts]
        stds = [data[str(c)]["std"] for c in counts]

        if method == "oracle":
            # Draw as horizontal band (oracle is distractor-invariant)
            m, s = np.mean(means), np.mean(stds)
            ax.axhline(m, color=COLORS[method], linestyle="--",
                       linewidth=1.5, label=LABELS[method], alpha=0.8)
            ax.axhspan(m - s, m + s, color=COLORS[method], alpha=0.08)
        else:
            ax.plot(counts, means, STYLES[method], color=COLORS[method],
                    label=LABELS[method], linewidth=2, markersize=5)
            ax.fill_between(counts,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.15, color=COLORS[method])

    # Mark approximate 3:1 ratio
    threshold_n = td * 3
    ylo, yhi = ax.get_ylim()
    ax.axvline(threshold_n, color="gray", linestyle=":", alpha=0.5,
               linewidth=1)
    ax.text(threshold_n + 2, ylo + (yhi - ylo) * 0.05,
            f"3:1", fontsize=8, color="gray")

    # Secondary x-axis: ratio
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    fs_data = tables.get(task, {{}}).get("full_state", {{}})
    tick_counts = sorted(int(k) for k in fs_data.keys()) if fs_data else counts
    if tick_counts:
        ax2.set_xticks(tick_counts)
        ax2.set_xticklabels([f"{{c/td:.1f}}" for c in tick_counts],
                            fontsize=7)
        ax2.set_xlabel("Distractor / signal ratio", fontsize=9)

    ax.set_xlabel("Distractor dimensions", fontsize=10)
    ax.set_ylabel("Episode return", fontsize=10)
    ax.set_title(TASK_LABELS.get(task, task), fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(RESULTS_DIR) / "summary" / "scaling_curve.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=150)
plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved: {{out_path}} and .png")
plt.show()
'''.format(results_dir=results_dir, tasks=tasks)

    out = Path(results_dir) / "summary" / "plot_scaling.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(script)
    logger.info(f"Plot script: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Distractor scaling experiment for IBD paper")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["walker_walk", "cheetah_run"],
        choices=list(TASK_SPECS.keys()),
        help="Tasks to run (default: walker_walk cheetah_run)")
    parser.add_argument(
        "--distractor_counts", nargs="+", type=int,
        default=DEFAULT_DISTRACTOR_COUNTS,
        help=f"Distractor counts to sweep (default: {DEFAULT_DISTRACTOR_COUNTS})")
    parser.add_argument(
        "--methods", nargs="+",
        default=METHODS,
        help="Methods to compare")
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds (default: 5)")
    parser.add_argument(
        "--total_steps", type=int, default=300_000,
        help="SAC training steps per run")
    parser.add_argument(
        "--results_dir", default="results/scaling")
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip training, just aggregate existing results")
    args = parser.parse_args()

    seeds = [42 + i * 100 for i in range(args.seeds)]

    if not args.aggregate_only:
        # ── run all experiments ──────────────────────────────────────
        total_runs = (len(args.tasks) * len(args.distractor_counts)
                      * len(args.methods) * len(seeds))
        logger.info(f"Scaling experiment: {total_runs} total runs")
        logger.info(f"  Tasks: {args.tasks}")
        logger.info(f"  Distractor counts: {args.distractor_counts}")
        logger.info(f"  Methods: {args.methods}")
        logger.info(f"  Seeds: {seeds}")
        logger.info(f"  Steps: {args.total_steps}")

        done = 0
        for task_key in args.tasks:
            spec = TASK_SPECS[task_key]
            for n_dist in args.distractor_counts:
                for method in args.methods:
                    for seed in seeds:
                        cfg = ScalingConfig(
                            domain=spec["domain"],
                            task=spec["task"],
                            n_distractors=n_dist,
                            method=method,
                            total_steps=args.total_steps,
                            seed=seed,
                            results_dir=args.results_dir,
                        )
                        try:
                            run_single(cfg)
                        except Exception as e:
                            logger.error(f"FAILED: {task_key} d={n_dist} "
                                         f"{method} s={seed}: {e}")
                        done += 1
                        if done % 10 == 0:
                            logger.info(f"Progress: {done}/{total_runs}")

    # ── aggregate ────────────────────────────────────────────────────
    tables = aggregate_results(args.results_dir, args.tasks,
                               args.distractor_counts)
    generate_plot_script(args.results_dir, args.tasks)

    logger.info("Done! To generate the figure, run:")
    logger.info(f"  python {args.results_dir}/summary/plot_scaling.py")


if __name__ == "__main__":
    main()