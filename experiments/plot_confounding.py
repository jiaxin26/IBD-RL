"""
Aggregate + plot the confounding-strength sweep.

Produces:
  * fig_confounding_kappa.pdf  — F1 vs kappa (action-channel confounding)
  * fig_confounding_lam.pdf    — F1 vs lambda (distractor loading)
  * fig_confounding_heatmap.pdf — per-method F1 over the (kappa, lam) grid
  * table_confounding.tex      — mean +/- std at the strongest setting

Usage::

    python experiments/plot_confounding.py --results_dir results/confounding
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABEL = {
    "ibd": "IBD (ours)",
    "mutual_info": "Mutual information",
    "variance": "Variance",
    "cond_mi": "Cond. MI (fwd model)",
    "grad_attr": "Gradient attribution",
    "inverse_dyn": "Multistep inverse dyn.",
    "random_mask": "Random mask",
}
METHOD_COLOR = {
    "ibd": "#d62728", "mutual_info": "#1f77b4", "variance": "#8c564b",
    "cond_mi": "#2ca02c", "grad_attr": "#ff7f0e",
    "inverse_dyn": "#9467bd", "random_mask": "#7f7f7f",
}
ORDER = ["ibd", "cond_mi", "grad_attr", "inverse_dyn", "mutual_info",
         "variance", "random_mask"]


def load(results_dir: str):
    recs = []
    for p in sorted(Path(results_dir).glob("per_run/*.json")):
        with open(p) as f:
            r = json.load(f)
        if "error" not in r:
            recs.append(r)
    return recs


def agg(recs, xkey, fixed_key=None, fixed_val=None, metric="f1"):
    """-> {method: (xs, means, stds)}"""
    buckets = defaultdict(list)
    for r in recs:
        if fixed_key is not None and r[fixed_key] != fixed_val:
            continue
        buckets[(r["method"], r[xkey])].append(r[metric])
    out = {}
    methods = sorted({m for m, _ in buckets},
                     key=lambda m: ORDER.index(m) if m in ORDER else 99)
    for m in methods:
        xs = sorted({x for mm, x in buckets if mm == m})
        mu = [float(np.mean(buckets[(m, x)])) for x in xs]
        sd = [float(np.std(buckets[(m, x)])) for x in xs]
        out[m] = (xs, mu, sd)
    return out


def line_plot(data, xlabel, title, path, ylabel="Boundary F1"):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for m, (xs, mu, sd) in data.items():
        xs = np.asarray(xs); mu = np.asarray(mu); sd = np.asarray(sd)
        ax.plot(xs, mu, "-o", ms=4, lw=2 if m == "ibd" else 1.4,
                color=METHOD_COLOR.get(m, "k"),
                label=METHOD_LABEL.get(m, m),
                zorder=3 if m == "ibd" else 2)
        ax.fill_between(xs, mu - sd, mu + sd, alpha=0.15,
                        color=METHOD_COLOR.get(m, "k"), lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def heatmap(recs, path, metric="f1"):
    methods = sorted({r["method"] for r in recs},
                     key=lambda m: ORDER.index(m) if m in ORDER else 99)
    kappas = sorted({r["kappa"] for r in recs})
    lams = sorted({r["lam"] for r in recs})
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.7), squeeze=False)
    for ax, m in zip(axes[0], methods):
        M = np.full((len(lams), len(kappas)), np.nan)
        for i, l in enumerate(lams):
            for j, k in enumerate(kappas):
                vals = [r[metric] for r in recs
                        if r["method"] == m and r["kappa"] == k
                        and r["lam"] == l]
                if vals:
                    M[i, j] = float(np.mean(vals))
        im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis",
                       aspect="auto")
        ax.set_xticks(range(len(kappas)))
        ax.set_xticklabels([f"{k:g}" for k in kappas], fontsize=7)
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"{l:g}" for l in lams], fontsize=7)
        ax.set_xlabel(r"$\kappa$", fontsize=8)
        ax.set_ylabel(r"$\lambda$", fontsize=8)
        ax.set_title(METHOD_LABEL.get(m, m), fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02,
                 label="F1")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def table(recs, path):
    kmax = max(r["kappa"] for r in recs)
    lmax = max(r["lam"] for r in recs)
    sel = [r for r in recs if r["kappa"] == kmax and r["lam"] == lmax]
    methods = sorted({r["method"] for r in sel},
                     key=lambda m: ORDER.index(m) if m in ORDER else 99)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & Precision & Recall & F1 & FP (conf.) & FP (exo.) \\",
             r"\midrule"]
    for m in methods:
        rs = [r for r in sel if r["method"] == m]
        def ms(k):
            v = [r[k] for r in rs]
            return f"{np.mean(v):.3f}$\\pm${np.std(v):.3f}"
        def mi(k):
            v = [r[k] for r in rs]
            return f"{np.mean(v):.1f}"
        name = METHOD_LABEL.get(m, m)
        if m == "ibd":
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {ms('precision')} & {ms('recall')} & "
                     f"{ms('f1')} & {mi('fp_confounded')} & "
                     f"{mi('fp_exogenous')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines))
    print(f"wrote {path}  (kappa={kmax:g}, lambda={lmax:g}, "
          f"n_seeds={len(sel) // max(len(methods), 1)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/confounding")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    recs = load(args.results_dir)
    if not recs:
        print(f"No results in {args.results_dir}/per_run/")
        return
    print(f"loaded {len(recs)} runs")

    out = Path(args.out_dir or (Path(args.results_dir) / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    lmax = max(r["lam"] for r in recs)
    kmax = max(r["kappa"] for r in recs)

    line_plot(agg(recs, "kappa", "lam", lmax),
              r"action-channel confounding $\kappa$",
              rf"Confounder $\to$ action strength ($\lambda={lmax:g}$)",
              out / "fig_confounding_kappa.pdf")
    line_plot(agg(recs, "lam", "kappa", kmax),
              r"distractor loading $\lambda$",
              rf"Confounder $\to$ distractor strength ($\kappa={kmax:g}$)",
              out / "fig_confounding_lam.pdf")
    if len({r["kappa"] for r in recs}) > 1 and len({r["lam"] for r in recs}) > 1:
        heatmap(recs, out / "fig_confounding_heatmap.pdf")
    table(recs, out / "table_confounding.tex")


if __name__ == "__main__":
    main()
