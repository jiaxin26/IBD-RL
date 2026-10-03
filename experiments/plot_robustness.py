#!/usr/bin/env python3
"""
Plot Robustness Study Results
==============================

Generates a dual-panel figure:
  Left:  Partial controllability — IBD recall vs mixing coefficient α
  Right: Detection threshold — overall P/R/F1 vs α

Usage:
    python -m experiments.plot_robustness
    python -m experiments.plot_robustness --results_dir results/robustness --out fig_robustness.pdf
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style (matches paper figures) ─────────────────────────────────────────────

C = {
    "ibd":       "#1B6CA8",
    "recall":    "#C0392B",
    "precision": "#16A085",
    "f1":        "#8E44AD",
    "partial":   "#E67E22",
    "grid":      "#E0E0E0",
    "thresh":    "#C0392B",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})


def load_sweep(results_dir: Path, domain: str, task: str) -> dict:
    """Load sweep results for one task."""
    fpath = results_dir / f"sweep_{domain}_{task}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")
    with open(fpath) as f:
        data = json.load(f)
    # Re-aggregate from per_run for robustness (avoids JSON key issues)
    per_run = data.get("per_run", [])
    alphas = sorted(set(r["alpha"] for r in per_run))
    agg = {}
    for a in alphas:
        runs = [r for r in per_run if abs(r["alpha"] - a) < 1e-6]
        agg[a] = {
            "precision_mean": float(np.mean([r["precision"] for r in runs])),
            "precision_std":  float(np.std([r["precision"] for r in runs])),
            "recall_mean":    float(np.mean([r["recall"] for r in runs])),
            "recall_std":     float(np.std([r["recall"] for r in runs])),
            "f1_mean":        float(np.mean([r["f1"] for r in runs])),
            "f1_std":         float(np.std([r["f1"] for r in runs])),
            "partial_recall_mean": float(np.mean([r["partial_recall"] for r in runs])),
            "partial_recall_std":  float(np.std([r["partial_recall"] for r in runs])),
        }
    data["_agg"] = agg
    data["_alphas"] = alphas
    return data


def plot_robustness(results_dir: Path, out_path: str = "fig_robustness.pdf"):
    """Generate the dual-panel robustness figure."""

    tasks = []
    for domain, task, label in [("cheetah", "run", "Cheetah Run"),
                                 ("walker", "walk", "Walker Walk")]:
        fpath = results_dir / f"sweep_{domain}_{task}.json"
        if fpath.exists():
            tasks.append((domain, task, label))

    if not tasks:
        print("No sweep results found. Run `run_robustness.py` first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # ── Left panel: Partial-dim recall vs α ──
    ax = axes[0]
    for domain, task, label in tasks:
        data = load_sweep(results_dir, domain, task)
        agg = data["_agg"]
        alphas = data["_alphas"]

        partial_r = np.array([agg[a]["partial_recall_mean"] for a in alphas])
        partial_r_std = np.array([agg[a]["partial_recall_std"] for a in alphas])

        ax.plot(alphas, partial_r, marker="o", label=label,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.fill_between(alphas, partial_r - partial_r_std,
                        np.minimum(partial_r + partial_r_std, 1.0),
                        alpha=0.12)

    # Detection threshold line
    ax.axhline(0.5, color=C["thresh"], ls=":", lw=0.8, alpha=0.5)
    ax.text(0.02, 0.53, "50% detection", fontsize=7, color=C["thresh"],
            alpha=0.7, va="bottom")

    ax.set_xlabel(r"Causal mixing coefficient $\alpha$")
    ax.set_ylabel("Recall on partial dims")
    ax.set_title("Partial Controllability", fontweight="bold", pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, ls="-", color=C["grid"])
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#DDD")

    # ── Right panel: Overall P/R/F1 vs α ──
    ax = axes[1]
    domain, task, label = tasks[0]  # Use first task for detail
    data = load_sweep(results_dir, domain, task)
    agg = data["_agg"]
    alphas = data["_alphas"]

    for metric, color, marker, lbl in [
        ("precision", C["precision"], "^", "Precision"),
        ("recall",    C["recall"],    "v", "Recall"),
        ("f1",        C["f1"],        "s", "F1"),
    ]:
        vals = np.array([agg[a][f"{metric}_mean"] for a in alphas])
        stds = np.array([agg[a][f"{metric}_std"] for a in alphas])
        ax.plot(alphas, vals, marker=marker, color=color, label=lbl,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.fill_between(alphas, vals - stds, np.minimum(vals + stds, 1.0),
                        alpha=0.10, color=color)

    ax.set_xlabel(r"Causal mixing coefficient $\alpha$")
    ax.set_ylabel("Score")
    ax.set_title(f"Detection Metrics ({label})", fontweight="bold", pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, ls="-", color=C["grid"])
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#DDD",
              loc="center right")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="results/robustness")
    parser.add_argument("--out", type=str, default="fig_robustness.pdf")
    args = parser.parse_args()

    plot_robustness(Path(args.results_dir), args.out)
