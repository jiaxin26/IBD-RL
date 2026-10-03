#!/usr/bin/env python3
"""
Learning Curve Plotter
=======================
Generates training curves from per-seed JSON results.

Expects results in: results/dmcontrol/per_seed/*.json
Each JSON must contain an "eval_returns" list of {step, mean, std}.

Usage:
    python plot_learning_curves.py
    python plot_learning_curves.py --results_dir results/dmcontrol/per_seed
    python plot_learning_curves.py --tasks walker_walk cheetah_run --distractor hard
"""

from __future__ import annotations

import argparse
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


METHOD_STYLE = {
    "full_state": {"color": "#d62728", "label": "Full State", "ls": "-"},
    "ibd":        {"color": "#1f77b4", "label": "IBD (ours)", "ls": "-"},
    "oracle":     {"color": "#2ca02c", "label": "Oracle",     "ls": "--"},
    "mutual_info":{"color": "#ff7f0e", "label": "MI Select",  "ls": "-."},
    "variance":   {"color": "#9467bd", "label": "Var Select", "ls": "-."},
    "cond_mi":    {"color": "#8c564b", "label": "Cond. MI",   "ls": "-."},
    "random_mask":{"color": "#7f7f7f", "label": "Random",     "ls": ":"},
}

TASK_TITLES = {
    "walker_walk": "Walker Walk",
    "cheetah_run": "Cheetah Run",
    "reacher_hard": "Reacher Hard",
    "cartpole_swingup": "Cartpole Swingup",
    "finger_spin": "Finger Spin",
    "hopper_hop": "Hopper Hop",
}


def load_results(results_dir, distractor_filter=None):
    """Load per-seed JSONs and group by (task, distractor, method)."""
    groups = defaultdict(list)
    for fpath in sorted(glob.glob(f"{results_dir}/*.json")):
        with open(fpath) as f:
            r = json.load(f)
        if distractor_filter and r.get("distractors") != distractor_filter:
            continue
        if "eval_returns" not in r or not r["eval_returns"]:
            continue
        task = f"{r['domain']}_{r['task']}"
        dist = r["distractors"]
        method = r["method"]
        algo = r.get("rl_algo", "sac")
        key = (task, dist, method, algo)
        groups[key].append(r)
    return groups


def aggregate_curves(runs):
    """Aggregate learning curves across seeds."""
    all_steps = []
    all_means = []
    for r in runs:
        steps = [e["step"] for e in r["eval_returns"]]
        means = [e["mean"] for e in r["eval_returns"]]
        all_steps.append(steps)
        all_means.append(means)

    # Align to shortest curve
    min_len = min(len(s) for s in all_steps)
    steps = all_steps[0][:min_len]
    arr = np.array([m[:min_len] for m in all_means])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return np.array(steps), mean, std, len(runs)


def plot_task(task_key, method_runs, ax, show_legend=True):
    """Plot learning curves for one task on given axes."""
    # Sort methods for consistent legend order
    method_order = ["oracle", "ibd", "full_state", "cond_mi",
                    "mutual_info", "variance", "random_mask"]

    for method in method_order:
        matching = {k: v for k, v in method_runs.items() if k[2] == method}
        if not matching:
            continue
        key = list(matching.keys())[0]
        runs = matching[key]
        steps, mean, std, n = aggregate_curves(runs)
        style = METHOD_STYLE.get(method, {"color": "gray", "label": method,
                                           "ls": "-"})
        ax.plot(steps, mean, color=style["color"], ls=style["ls"],
                label=f"{style['label']} (n={n})", linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.12, color=style["color"])

    title = TASK_TITLES.get(task_key, task_key)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_ylabel("Episode Return", fontsize=9)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="results/dmcontrol/per_seed")
    parser.add_argument("--distractor", type=str, default=None,
                        help="Filter by distractor level")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="e.g. walker_walk cheetah_run")
    parser.add_argument("--output", type=str, default="results/figures")
    args = parser.parse_args()

    groups = load_results(args.results_dir, args.distractor)
    if not groups:
        print(f"No results found in {args.results_dir}")
        return

    # Group by (task, distractor)
    task_dist_groups = defaultdict(dict)
    for key, runs in groups.items():
        task, dist, method, algo = key
        if args.tasks and task not in args.tasks:
            continue
        task_dist_groups[(task, dist)][key] = runs

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Individual plots ──
    for (task, dist), method_runs in sorted(task_dist_groups.items()):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))
        plot_task(task, method_runs, ax)
        fig.tight_layout()
        fname = f"curve_{task}_{dist}.pdf"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir / fname}")

    # ── Combined panel figure ──
    task_dists = sorted(task_dist_groups.keys())
    if len(task_dists) >= 2:
        n = len(task_dists)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 3.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (task, dist) in enumerate(task_dists):
            plot_task(task, task_dist_groups[(task, dist)],
                      axes[i], show_legend=(i == 0))
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        fname = f"curves_panel{'_' + args.distractor if args.distractor else ''}.pdf"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir / fname}")


if __name__ == "__main__":
    main()
