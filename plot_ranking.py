#!/usr/bin/env python3
"""
Figure 2 — Causal vs Observational Feature Ranking

Runs IBD and MI on a single task to extract per-dim scores, then
plots side-by-side bar charts showing that MI cannot separate
causal dims from confounded distractors, while IBD can.

Usage:
    python plot_ranking.py
    python plot_ranking.py --domain reacher --task hard --distractors medium
    python plot_ranking.py --from_cache ranking_data.json   # reuse saved data

The first run takes ~5-8 min (trains a scout + runs probing).
Subsequent runs with --from_cache are instant.
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def collect_ranking_data(domain, task, distractors, seed=42):
    """
    Run IBD probe and MI selector on one task and return per-dim scores.
    Requires the full project to be importable.
    """
    import sys
    # Add project root to path if needed
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.dmcontrol_distractors import make_env
    from experiments.baselines import MutualInfoSelector
    from ibd.probe import IBDProbe
    from stable_baselines3 import SAC

    print(f"Building env: {domain}_{task} / {distractors} / seed={seed}")
    env = make_env(domain, task, distractor_config=distractors, seed=seed)
    obs_dim = env.observation_space.shape[0]
    true_dims = sorted(env.true_dims)
    distractor_dims = sorted(env.distractor_dims)

    # ── IBD ──
    print("Running IBD probe (scout + joint intervention)...")
    probe_env = make_env(domain, task, distractor_config=distractors, seed=seed + 1000)
    scout_env = make_env(domain, task, distractor_config=distractors, seed=seed + 2000)

    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4,
                batch_size=256, seed=seed, verbose=0)
    scout.learn(total_timesteps=80_000)

    probe = IBDProbe(
        probe_env,
        n_baseline=80, n_intervention=80,
        traj_length=200, horizons=[1, 5, 10],
        n_permutations=1000, seed=seed)

    # IBDProbe expects policy as callable (obs) -> action
    import torch
    def scout_policy(obs):
        with torch.no_grad():
            action, _ = scout.predict(obs, deterministic=False)
        return action

    soi, ibd_info = probe.discover_joint(policy=scout_policy)

    # Extract per-dim p-values (already min across horizons)
    ibd_pvals = np.ones(obs_dim)
    if "p_values" in ibd_info:
        ibd_pvals = np.array(ibd_info["p_values"])

    # Extract effect sizes (already max across horizons)
    ibd_effect = np.zeros(obs_dim)
    if "effect_sizes" in ibd_info:
        ibd_effect = np.array(ibd_info["effect_sizes"])

    # ── MI ──
    print("Running MI selector...")
    mi_env = make_env(domain, task, distractor_config=distractors, seed=seed + 3000)
    mi_selector = MutualInfoSelector(
        mi_env, n_episodes=50, n_select=len(true_dims), seed=seed)
    mi_mask = mi_selector.discover()

    # Extract per-dim MI scores from mask.effect_sizes
    mi_scores = np.zeros(obs_dim)
    if mi_mask.effect_sizes is not None:
        mi_scores = np.array(mi_mask.effect_sizes)

    # ── package results ──
    result = {
        "domain": domain,
        "task": task,
        "distractors": distractors,
        "obs_dim": obs_dim,
        "true_dims": true_dims,
        "distractor_dims": distractor_dims,
        "ibd_soi": sorted(soi),
        "ibd_pvals": ibd_pvals.tolist(),
        "ibd_effect_sizes": ibd_effect.tolist(),
        "mi_scores": mi_scores.tolist(),
        "mi_selected": sorted(mi_mask.soi_dims),
    }

    # Cleanup
    for e in [env, probe_env, scout_env, mi_env]:
        e.close()
    del scout

    return result


def plot_ranking(data, out_path="fig2_ranking.pdf"):
    """Plot side-by-side ranking comparison."""
    obs_dim = data["obs_dim"]
    true_dims = set(data["true_dims"])
    distractor_dims = set(data["distractor_dims"])

    mi_scores = np.array(data["mi_scores"])
    ibd_pvals = np.array(data["ibd_pvals"])

    # Convert p-values to -log10 for visibility
    ibd_score = -np.log10(np.clip(ibd_pvals, 1e-50, 1.0))

    # Color arrays
    colors_causal = "#2980B9"
    colors_distractor = "#E74C3C"

    dim_colors = [colors_causal if i in true_dims else colors_distractor
                  for i in range(obs_dim)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: MI ranking ──
    mi_order = np.argsort(mi_scores)[::-1]  # descending
    mi_sorted = mi_scores[mi_order]
    mi_colors_sorted = [dim_colors[i] for i in mi_order]

    ax1.bar(range(obs_dim), mi_sorted, color=mi_colors_sorted, alpha=0.8,
            edgecolor="none", width=1.0)
    ax1.set_title("Observational Ranking (MI)", fontsize=13,
                  fontweight="bold")
    ax1.set_xlabel("Observation Dimension (sorted by MI)", fontsize=10)
    ax1.set_ylabel("MI Score", fontsize=10)
    ax1.set_xlim(-0.5, obs_dim - 0.5)

    # mark how many of top-k are distractors
    n_true = len(true_dims)
    top_k_mi = set(mi_order[:n_true])
    fp_mi = len(top_k_mi & distractor_dims)
    ax1.axvline(n_true - 0.5, color="black", ls=":", lw=1, alpha=0.5)
    ax1.text(n_true + 1, mi_sorted.max() * 0.9,
             f"Top-{n_true}: {fp_mi} distractors\nmisranked as causal",
             fontsize=8, va="top", color="#E74C3C")

    # ── Panel B: IBD ranking ──
    ibd_order = np.argsort(ibd_score)[::-1]  # descending
    ibd_sorted = ibd_score[ibd_order]
    ibd_colors_sorted = [dim_colors[i] for i in ibd_order]

    ax2.bar(range(obs_dim), ibd_sorted, color=ibd_colors_sorted, alpha=0.8,
            edgecolor="none", width=1.0)
    ax2.set_title("Interventional Ranking (IBD)", fontsize=13,
                  fontweight="bold")
    ax2.set_xlabel("Observation Dimension (sorted by −log₁₀ p)", fontsize=10)
    ax2.set_ylabel("−log₁₀(p-value)", fontsize=10)
    ax2.set_xlim(-0.5, obs_dim - 0.5)

    # significance threshold
    alpha = 0.05
    thresh = -np.log10(alpha)
    ax2.axhline(thresh, color="black", ls="--", lw=1, alpha=0.6)
    ax2.text(obs_dim - 2, thresh + 0.3, f"α = {alpha}",
             fontsize=8, ha="right", va="bottom")

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_causal, alpha=0.8, label="Causal dim"),
        Patch(facecolor=colors_distractor, alpha=0.8, label="Distractor"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))

    task_label = f"{data['domain']}_{data['task']} / {data['distractors']}"
    fig.suptitle(task_label, fontsize=9, color="gray", y=0.98,
                 style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="reacher", type=str)
    parser.add_argument("--task", default="hard", type=str)
    parser.add_argument("--distractors", default="medium", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--from_cache", default=None, type=str,
                        help="Load pre-computed data instead of re-running")
    parser.add_argument("--save_cache", default="ranking_data.json", type=str)
    parser.add_argument("--out", default="fig2_ranking.pdf", type=str)
    args = parser.parse_args()

    if args.from_cache and Path(args.from_cache).exists():
        print(f"Loading cached data from {args.from_cache}")
        with open(args.from_cache) as f:
            data = json.load(f)
    else:
        data = collect_ranking_data(
            args.domain, args.task, args.distractors, args.seed)
        with open(args.save_cache, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Cached ranking data to {args.save_cache}")

    plot_ranking(data, args.out)