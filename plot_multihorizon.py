#!/usr/bin/env python3
"""
Multi-horizon distribution figure for IBD.
==========================================

Produces one figure from a single probe run:

  fig_multihorizon_dist.pdf  — KDE of Δ^h_i(τ) for baseline vs
                               do(a = noise), for one representative
                               causal dim and one representative
                               distractor, across horizons h ∈ {1, 5, 10}.

No changes to the ibd library. Uses:
  IBDProbe._collect           — baseline + joint-intervention rollouts
  CIMEstimator._extract       — per-trajectory Δ^h_i summary
and re-runs the Welch t-test + BH correction here so the per-horizon
breakdown is retained (discover_joint collapses it via min/max).

Usage
-----
    # Default: reacher_hard / medium, structured random probe
    python plot_multihorizon.py

    # Re-plot from the cached JSON (instant, no env needed).
    # Also happens automatically if a cache for the same task exists.
    python plot_multihorizon.py --from_cache figures/mh_cache_reacher_hard_medium.json

    # Different task / distractor tier, outputs go into figures/
    python plot_multihorizon.py --domain cheetah --task run --distractors medium \
        --out_dir figures/

    # Exact parity with Figure 1 (80K-step SAC scout)
    python plot_multihorizon.py --scout_steps 80000 --out_dir figures/

Batch over appendix tasks
-------------------------
    mkdir -p figures
    for task in "reacher hard" "cheetah run" "walker walk" \
                "cartpole swingup" "finger spin" "hopper hop"; do
      read d t <<< "$task"
      python plot_multihorizon.py --domain $d --task $t --distractors medium \
          --scout_steps 80000 --out_dir figures/
    done

Auto-naming: outputs are written to
    <out_dir>/fig_multihorizon_<domain>_<task>_<distractors>.pdf  (+ .png)
    <out_dir>/mh_cache_<domain>_<task>_<distractors>.json
so batch runs never clobber each other.

Cost (default): ~32K env steps total, <3 min, no GPU required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Make the repo importable when running from its root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HORIZONS = [1, 5, 10]
ALPHA = 0.05

BASELINE_COLOR = "#35B779"   # teal-green
INTERV_COLOR   = "#7570B3"   # blue-purple
SIG_COLOR      = "#333333"   # near-black for significant p
NS_COLOR       = "#999999"   # gray for non-significant p
# Kept for back-compat with any external callers
CAUSAL_COLOR = BASELINE_COLOR
DISTR_COLOR  = INTERV_COLOR


# ─────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────

def collect_per_horizon_data(domain, task, distractors, seed,
                             n_trajs, traj_length, scout_steps):
    """Run one IBD probe and return per-horizon raw samples + stats.

    Mathematical procedure exactly matches IBDProbe.discover_joint:
      - Welch t-test on |h-step diffs|
      - Hedges' g effect size (bias-corrected Cohen's d)
      - Benjamini-Hochberg correction across all (dim × horizon) tests
    The only difference is we retain per-(dim, horizon) values instead
    of collapsing with min/max.
    """
    from experiments.dmcontrol_distractors import make_env
    from ibd.probe import IBDProbe

    env = make_env(domain, task, distractor_config=distractors, seed=seed)
    obs_dim = env.observation_space.shape[0]
    true_dims = sorted(env.true_dims)
    distractor_dims = sorted(env.distractor_dims)
    env.close()

    probe_env = make_env(domain, task, distractor_config=distractors,
                         seed=seed + 1000)
    probe = IBDProbe(
        probe_env,
        n_baseline=n_trajs,
        n_intervention=n_trajs,
        traj_length=traj_length,
        horizons=HORIZONS,
        alpha=ALPHA,
        seed=seed,
    )

    # ── Policy: SAC scout if requested, else structured random ──
    scout = None
    scout_env = None
    if scout_steps > 0:
        from stable_baselines3 import SAC
        import torch

        scout_env = make_env(domain, task, distractor_config=distractors,
                             seed=seed + 2000)
        scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4,
                    batch_size=256, seed=seed, verbose=0)
        print(f"Training SAC scout for {scout_steps} steps...")
        scout.learn(total_timesteps=scout_steps)

        def policy(obs):
            with torch.no_grad():
                a, _ = scout.predict(obs, deterministic=False)
            return a
    else:
        policy = probe._random_policy

    # ── Collect rollouts (one baseline + one joint-intervention set) ──
    print(f"Collecting {n_trajs} baseline + {n_trajs} joint-intervention "
          f"trajectories × {traj_length} steps on "
          f"{domain}_{task}/{distractors}...")
    baseline_trajs = probe._collect(policy, n_trajs, intervention=None)
    intervention_trajs = probe._collect(
        policy, n_trajs,
        intervention={"dim": "all", "mode": "randomize"})

    # ── Per-(dim, horizon) Δ samples, raw p, and effect size ──
    delta_base = {}   # (d, h) -> list of N Δ values
    delta_int = {}
    raw_p = np.ones((obs_dim, len(HORIZONS)))
    eff_sz = np.zeros((obs_dim, len(HORIZONS)))

    for d in range(obs_dim):
        for hi, h in enumerate(HORIZONS):
            yb = probe.cim._extract(baseline_trajs, d, h, absolute=True)
            yi = probe.cim._extract(intervention_trajs, d, h, absolute=True)
            if yb is None or yi is None:
                continue
            delta_base[(d, h)] = yb.tolist()
            delta_int[(d, h)] = yi.tolist()

            # Welch t-test
            t_stat, p = sp_stats.ttest_ind(yb, yi, equal_var=False)
            raw_p[d, hi] = float(p) if np.isfinite(p) else 1.0

            # Hedges' g
            n0, n1 = len(yb), len(yi)
            v0 = float(np.var(yb, ddof=1))
            v1 = float(np.var(yi, ddof=1))
            pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) /
                             (n0 + n1 - 2))
            if pooled > 1e-12:
                g = abs(np.mean(yb) - np.mean(yi)) / pooled
                g *= 1 - 3 / (4 * (n0 + n1) - 9)
                eff_sz[d, hi] = float(g)

    # ── Benjamini-Hochberg across all d × |H| tests ──
    flat = raw_p.flatten()
    m = len(flat)
    order = np.argsort(flat)
    sorted_p = flat[order]
    adj = np.ones(m)
    for k in range(m - 1, -1, -1):
        i = order[k]
        adj[i] = min(sorted_p[k] * m / (k + 1), 1.0)
        if k < m - 1:
            adj[i] = min(adj[i], adj[order[k + 1]])
    adj_p = adj.reshape(raw_p.shape)

    probe_env.close()
    if scout is not None:
        del scout
    if scout_env is not None:
        scout_env.close()

    return {
        "domain": domain,
        "task": task,
        "distractors": distractors,
        "obs_dim": int(obs_dim),
        "horizons": HORIZONS,
        "true_dims": list(true_dims),
        "distractor_dims": list(distractor_dims),
        "raw_p": raw_p.tolist(),
        "adj_p": adj_p.tolist(),
        "effect_size": eff_sz.tolist(),
        # JSON can't key on tuples — use "d,h" strings
        "delta_baseline": {f"{d},{h}": v for (d, h), v in delta_base.items()},
        "delta_intervention": {f"{d},{h}": v
                               for (d, h), v in delta_int.items()},
    }


# ─────────────────────────────────────────────────────────────────────
# FIGURE — DISTRIBUTION PANEL (1 causal + 1 distractor × 3 horizons)
# ─────────────────────────────────────────────────────────────────────

def pick_representative_dims(data):
    """Pick (causal_dim, distractor_dim) that best illustrate the story.

    Causal dim  : largest spread in -log10 adj_p across horizons
                  (i.e. a dim where some horizons catch it better than
                   others — this is the whole point of multi-horizon).
    Distractor  : smallest min adj_p across horizons
                  (i.e. the distractor that comes closest to a false
                   positive — shows FDR control holds on the hardest
                   fake).
    """
    adj_p = np.array(data["adj_p"])
    scores = -np.log10(np.clip(adj_p, 1e-50, 1.0))
    true_dims = sorted(data["true_dims"])
    distractor_dims = sorted(data["distractor_dims"])

    # Causal: max horizon spread; tie-break by max score
    spreads = {d: float(scores[d].max() - scores[d].min())
               for d in true_dims}
    max_spread = max(spreads.values()) if spreads else 0.0
    if max_spread < 0.5:
        # All horizons detect everything equally — just pick the dim
        # with the highest score at any horizon.
        causal_dim = max(true_dims, key=lambda d: scores[d].max())
    else:
        causal_dim = max(spreads, key=spreads.get)

    # Distractor: smallest min adj_p = closest to threshold
    min_p = {d: float(adj_p[d].min()) for d in distractor_dims}
    distractor_dim = min(min_p, key=min_p.get) if min_p else None

    return causal_dim, distractor_dim


def _kde_or_hist(ax, yb, yi, baseline_label, interv_label, show_rug=False):
    """Plot KDE of baseline vs intervention. Falls back to hist if KDE
    is singular (e.g. constant data). Rug plot is optional (off by
    default — it gets cluttered at N ≥ 80 per group)."""
    x_lo = float(min(yb.min(), yi.min()))
    x_hi = float(max(yb.max(), yi.max()))
    pad = 0.1 * max(x_hi - x_lo, 1e-9)
    xs = np.linspace(x_lo - pad, x_hi + pad, 300)
    try:
        kb_curve = sp_stats.gaussian_kde(yb)(xs)
        ki_curve = sp_stats.gaussian_kde(yi)(xs)
        ax.fill_between(xs, kb_curve, alpha=0.30, color=BASELINE_COLOR,
                        label=baseline_label)
        ax.fill_between(xs, ki_curve, alpha=0.30, color=INTERV_COLOR,
                        label=interv_label)
        ax.plot(xs, kb_curve, color=BASELINE_COLOR, lw=1.3)
        ax.plot(xs, ki_curve, color=INTERV_COLOR, lw=1.3)
        y_top = max(kb_curve.max(), ki_curve.max()) * 1.15
    except (np.linalg.LinAlgError, ValueError):
        ax.hist(yb, bins=20, alpha=0.45, color=BASELINE_COLOR,
                label=baseline_label, density=True)
        ax.hist(yi, bins=20, alpha=0.45, color=INTERV_COLOR,
                label=interv_label, density=True)
        y_top = ax.get_ylim()[1]
    if show_rug:
        ax.scatter(yb, np.full_like(yb, -0.02 * y_top),
                   marker="|", color=BASELINE_COLOR, s=18, alpha=0.7)
        ax.scatter(yi, np.full_like(yi, -0.05 * y_top),
                   marker="|", color=INTERV_COLOR, s=18, alpha=0.7)
        ax.set_ylim(-0.08 * y_top, y_top)
    else:
        ax.set_ylim(0, y_top)


def plot_distribution_panel(data, out_path, show_rug=False):
    """Schematic-style figure: curves only, no text, no ticks, no labels.

    Each panel is a small bordered box showing baseline vs intervention
    KDE curves (filled + outlined). Horizon headers at the top, row
    labels on the left — that's it. Everything else goes in the caption.
    """
    horizons = data["horizons"]
    causal_dim, distractor_dim = pick_representative_dims(data)

    with plt.rc_context({
        "axes.linewidth":  0.9,
        "font.size":       11,
        "legend.frameon":  False,
    }):
        fig, axes = plt.subplots(
            2, len(horizons),
            figsize=(3.5 * len(horizons), 4.3),
            sharey=False,
        )
        if len(horizons) == 1:
            axes = axes.reshape(2, 1)

        rows = [(causal_dim, "Causal", 0),
                (distractor_dim, "Distractor", 1)]

        for dim, label, row in rows:
            if dim is None:
                continue
            for hi, h in enumerate(horizons):
                ax = axes[row, hi]
                yb = np.array(data["delta_baseline"].get(f"{dim},{h}", []))
                yi = np.array(data["delta_intervention"].get(f"{dim},{h}", []))
                if len(yb) == 0 or len(yi) == 0:
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                _kde_or_hist(ax, yb, yi,
                             baseline_label="Baseline",
                             interv_label=r"$\mathrm{do}(a = \mathrm{noise})$",
                             show_rug=show_rug)

                # Schematic look: drop ticks, tick labels, all 4 spines
                # stay so each panel reads as a boxed icon.
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xticklabels([]); ax.set_yticklabels([])
                for side in ("top", "right", "bottom", "left"):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(0.9)
                    ax.spines[side].set_color("#888")

                # Horizon header — top row only
                if row == 0:
                    ax.set_title(rf"$h = {h}$", pad=4, fontsize=13)
                # Row label — leftmost column only
                if hi == 0:
                    ax.set_ylabel(label, fontsize=12, labelpad=8)

        fig.subplots_adjust(wspace=0.18, hspace=0.22,
                            left=0.08, right=0.98,
                            top=0.90, bottom=0.04)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(str(out_path).replace(".pdf", ".png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved: {out_path}  "
          f"(causal dim {causal_dim}, distractor dim {distractor_dim})")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="reacher", type=str)
    parser.add_argument("--task", default="hard", type=str)
    parser.add_argument("--distractors", default="medium", type=str,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_trajs", default=80, type=int)
    parser.add_argument("--traj_length", default=200, type=int)
    parser.add_argument("--scout_steps", default=0, type=int,
                        help="SAC scout training budget. 0 = use the "
                             "structured random probe (default).")
    parser.add_argument("--from_cache", default=None, type=str,
                        help="Load cached JSON instead of re-running probe.")
    parser.add_argument("--out_dir", default=".", type=str,
                        help="Directory for output PDFs/PNGs and cache. "
                             "Created if missing.")
    parser.add_argument("--cache_path", default=None, type=str,
                        help="Override auto-named cache JSON path.")
    parser.add_argument("--out_dist", default=None, type=str,
                        help="Override auto-named figure PDF path.")
    parser.add_argument("--show_rug", action="store_true",
                        help="Overlay a rug plot of raw samples under each "
                             "KDE curve. Off by default (gets cluttered at "
                             "N ≥ 80 per group).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-name per-task so batch runs for the appendix don't clobber.
    task_tag = f"{args.domain}_{args.task}_{args.distractors}"
    cache_path = Path(args.cache_path) if args.cache_path \
        else out_dir / f"mh_cache_{task_tag}.json"
    out_dist = Path(args.out_dist) if args.out_dist \
        else out_dir / f"fig_multihorizon_{task_tag}.pdf"

    if args.from_cache and Path(args.from_cache).exists():
        print(f"Loading cached data from {args.from_cache}")
        with open(args.from_cache) as f:
            data = json.load(f)
    elif cache_path.exists() and not args.from_cache:
        print(f"Loading cached data from {cache_path} "
              f"(pass --from_cache <other> to override)")
        with open(cache_path) as f:
            data = json.load(f)
    else:
        data = collect_per_horizon_data(
            args.domain, args.task, args.distractors, args.seed,
            args.n_trajs, args.traj_length, args.scout_steps)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"Cached data to {cache_path}")

    plot_distribution_panel(data, out_dist, show_rug=args.show_rug)