#!/usr/bin/env python3
"""
plot_scaling_v2.py
==================

Generates two figures from the dense distractor scaling sweep
(experiments/run_scaling.py output, default results/scaling/per_seed/):

    Figure A — fig_scaling_curves.{pdf,png}
        Linear-scale return vs distractor count, one panel per task.
        Drop-in upgrade for the current Figure 2 but with 8 distractor
        counts instead of 3.

    Figure B — fig_scaling_law.{pdf,png}
        Power-law scaling figure on log-log axes, with x = distractor-
        to-signal ratio (d / d_c). Tasks collapse onto a common curve;
        we fit R / R_oracle = a * (d/d_c)^(-alpha) on the Full State
        points and print fitted exponents with bootstrap 95 % CIs.

Compatible JSON schema (from experiments/run_scaling.py, run_single):
    {
        "domain": "walker", "task": "walk",
        "n_distractors": 50, "method": "full_state",
        "seed": 42, "true_obs_dim": 24,
        "final_return": 412.3, ...
    }
File naming: {domain}_{task}_d{n_distractors}_{method}_s{seed}.json

Usage:
    python plot_scaling_v2.py
    python plot_scaling_v2.py --results_dir results/scaling
    python plot_scaling_v2.py --tasks walker_walk cheetah_run reacher_hard
    python plot_scaling_v2.py --out_dir figures_v2 --no_fit_ibd
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ─── configuration ──────────────────────────────────────────────────────────

TASK_SPECS = {
    # task_key:    (domain, task, true_dim, display_label)
    "walker_walk":  ("walker",  "walk", 24, "Walker Walk"),
    "cheetah_run":  ("cheetah", "run",  17, "Cheetah Run"),
    "reacher_hard": ("reacher", "hard",  6, "Reacher Hard"),
    "hopper_hop":   ("hopper",  "hop",  15, "Hopper Hop"),
    "finger_spin":  ("finger",  "spin",  9, "Finger Spin"),
    "cartpole_swingup": ("cartpole", "swingup", 5, "Cartpole Swingup"),
}

METHODS = ["full_state", "ibd", "oracle"]
METHOD_STYLE = {
    "full_state": dict(color="#BB4E51", marker="s", ls="-",  label="Full State", zorder=2),
    "ibd":        dict(color="#4D73CA", marker="o", ls="-",  label="IBD (ours)", zorder=3),
    "oracle":     dict(color="#60A765", marker="^", ls="--", label="Oracle",     zorder=1),
}

# distinct task colors for the collapse plot
# chosen to avoid red (#BB4E51, full_state) and blue (#4D73CA, ibd) used for fits
TASK_COLORS = {
    "walker_walk":      "#7B3F99",  # purple
    "cheetah_run":      "#E67E22",  # orange
    "reacher_hard":     "#16A085",  # teal-green
    "hopper_hop":       "#8E44AD",  # violet
    "finger_spin":      "#A0522D",  # sienna
    "cartpole_swingup": "#C0392B",  # dark red (only if no full_state fit shown)
}


# ─── data loading ───────────────────────────────────────────────────────────

def load_per_seed(per_seed_dir: Path, domain: str, task: str,
                  n_dist: int, method: str) -> List[float]:
    """Return list of final_return values across all seeds for one cell."""
    pattern = f"{domain}_{task}_d{n_dist}_{method}_s*.json"
    files = sorted(per_seed_dir.glob(pattern))
    out = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            if "final_return" in d:
                out.append(float(d["final_return"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out


def collect(results_dir: Path, tasks: List[str]) -> Dict:
    """
    Returns nested dict:
        data[task_key][method][n_dist] = {
            "mean": float, "std": float, "n": int, "returns": [...]
        }
    Plus data[task_key]["true_dim"] copied from TASK_SPECS.
    Distractor counts are auto-discovered from the filenames present.
    """
    per_seed_dir = results_dir / "per_seed"
    if not per_seed_dir.exists():
        # accept flat layout too
        per_seed_dir = results_dir
    if not per_seed_dir.exists():
        raise FileNotFoundError(f"No data dir: {per_seed_dir}")

    data: Dict = {}
    for task_key in tasks:
        if task_key not in TASK_SPECS:
            print(f"[warn] unknown task {task_key}, skipping")
            continue
        domain, task, true_dim, _ = TASK_SPECS[task_key]
        data[task_key] = {"true_dim": true_dim}

        # discover n_dist values present for this task across any method
        present_counts: set = set()
        for f in per_seed_dir.glob(f"{domain}_{task}_d*_*.json"):
            stem = f.stem  # walker_walk_d50_full_state_s42
            try:
                d_tok = next(t for t in stem.split("_") if t.startswith("d"))
                n = int(d_tok[1:])
                present_counts.add(n)
            except (StopIteration, ValueError):
                continue

        for method in METHODS:
            data[task_key][method] = {}
            for n_dist in sorted(present_counts):
                returns = load_per_seed(per_seed_dir, domain, task, n_dist, method)
                if not returns:
                    continue
                data[task_key][method][n_dist] = {
                    "mean":    float(np.mean(returns)),
                    "std":     float(np.std(returns)),
                    "n":       len(returns),
                    "returns": returns,
                }

        # broadcast oracle: if oracle was only run at one distractor count,
        # it is invariant (oracle masks all distractors), so reuse it.
        oracle = data[task_key]["oracle"]
        if oracle and len(oracle) == 1 and len(present_counts) > 1:
            (only_n, only_v), = list(oracle.items())
            for n_dist in sorted(present_counts):
                if n_dist != only_n:
                    data[task_key]["oracle"][n_dist] = dict(only_v)

    return data


# ─── power-law fitting ──────────────────────────────────────────────────────

def power_law(x, a, alpha):
    """R/R_oracle = a * x^(-alpha), x = d / d_c."""
    return a * np.power(x, -alpha)


def fit_power_law(x: np.ndarray, y: np.ndarray,
                  yerr: Optional[np.ndarray] = None,
                  bootstrap: int = 1000,
                  rng: Optional[np.random.Generator] = None,
                  x_min: float = 0.0,
                  ) -> Optional[Dict]:
    """
    Fit y = a * x^(-alpha) on log-log axes via least squares.

    x_min: only points with x >= x_min are used (default 0 = no filter).
    Use this to restrict the fit to the degradation regime, e.g.
    x_min=1.0 fits only points where d >= d_c (distractors >= signal).

    Returns dict with (a, alpha, alpha_lo, alpha_hi, r2, n, x_min)
    or None if fewer than 3 finite, positive points remain.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y) & (x >= x_min)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None

    log_x = np.log(x)
    log_y = np.log(y)

    # weighted least squares in log-space (log-error ~ yerr/y)
    sigma = None
    if yerr is not None:
        ye = np.asarray(yerr, dtype=float)[mask]
        ye = np.where(ye > 0, ye, np.median(ye[ye > 0]) if np.any(ye > 0) else 1.0)
        sigma = ye / y  # delta(log y) ~ dy / y

    try:
        # log y = log a - alpha * log x  → linear fit
        if sigma is None:
            coeffs, cov = np.polyfit(log_x, log_y, 1, cov=True)
        else:
            coeffs, cov = np.polyfit(log_x, log_y, 1, w=1.0 / sigma, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None

    slope, intercept = coeffs           # log y = slope * log x + intercept
    alpha = -slope
    a = float(np.exp(intercept))

    # R^2 in log-space
    y_pred = slope * log_x + intercept
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Bootstrap CI on alpha (resample with replacement, refit)
    if rng is None:
        rng = np.random.default_rng(0)
    alphas: List[float] = []
    n = len(x)
    for _ in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c = np.polyfit(log_x[idx], log_y[idx], 1)
            alphas.append(-c[0])
        except (np.linalg.LinAlgError, ValueError):
            continue
    if alphas:
        alpha_lo, alpha_hi = float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))
    else:
        alpha_lo = alpha_hi = float("nan")

    return dict(a=a, alpha=float(alpha),
                alpha_lo=alpha_lo, alpha_hi=alpha_hi,
                r2=float(r2), n=int(n), x_min=float(x_min))


# ─── Figure A: linear curves per task ───────────────────────────────────────

def plot_curves(data: Dict, out_path: Path) -> None:
    tasks = [t for t in data.keys()
             if any(data[t][m] for m in METHODS if m in data[t])]
    if not tasks:
        print("[warn] no data to plot for Figure A")
        return

    n_panels = len(tasks)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.2 * n_panels, 3.8),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, task_key in zip(axes, tasks):
        td = data[task_key]
        true_dim = td["true_dim"]

        for method in METHODS:
            md = td.get(method, {})
            if not md:
                continue
            xs = sorted(md.keys())
            ys = np.array([md[x]["mean"] for x in xs])
            es = np.array([md[x]["std"] for x in xs])
            style = METHOD_STYLE[method]
            ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                    ls=style["ls"], lw=2.0, ms=6,
                    label=style["label"], zorder=style["zorder"])
            ax.fill_between(xs, ys - es, ys + es,
                            alpha=0.15, color=style["color"],
                            zorder=style["zorder"])

        _, _, _, label = TASK_SPECS[task_key]
        ax.set_title(f"{label}  (d_c = {true_dim})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Distractor dimensions  d", fontsize=11)
        ax.grid(alpha=0.3, ls="--")

    axes[0].set_ylabel("Episode return", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(handles), fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.with_suffix('.pdf')}")
    print(f"  wrote {out_path.with_suffix('.png')}")


# ─── Figure B: log-log ratio collapse + power-law fit ───────────────────────

def plot_scaling_law(data: Dict, out_path: Path,
                     fit_ibd: bool = True,
                     fit_x_min: float = 1.0) -> Dict:
    """
    Pool Full State (and IBD) across tasks on a common axis:
        x = d / d_c   (distractor-to-signal ratio)
        y = R / R_oracle   (normalized return)

    fit_x_min: power-law fit is restricted to points with d/d_c >= fit_x_min.
        Default 1.0 = degradation regime (distractors >= signal). Set to 0.0
        to fit all points. Points below the threshold are still plotted but
        drawn with reduced opacity.

    Returns the fit results dict (used for the printed summary).
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    pooled: Dict[str, Dict[str, list]] = {
        "full_state": {"x": [], "y": [], "yerr": []},
        "ibd":        {"x": [], "y": [], "yerr": []},
    }

    plotted_tasks: List[str] = []

    for task_key, td in data.items():
        true_dim = td["true_dim"]

        # determine R_oracle as the mean across all available oracle runs
        oracle_means = [v["mean"] for v in td.get("oracle", {}).values()]
        if not oracle_means:
            print(f"[warn] {task_key}: no oracle runs; skipping in Fig B")
            continue
        R_oracle = float(np.mean(oracle_means))
        if R_oracle <= 0:
            continue
        plotted_tasks.append(task_key)

        color = TASK_COLORS.get(task_key, "#444444")
        _, _, _, label = TASK_SPECS[task_key]

        # full state
        fs = td.get("full_state", {})
        if fs:
            xs = np.array(sorted(fs.keys()), dtype=float)
            ys = np.array([fs[int(x)]["mean"] for x in xs])
            es = np.array([fs[int(x)]["std"]  for x in xs])
            x_ratio = xs / true_dim
            y_norm  = ys / R_oracle
            e_norm  = es / R_oracle
            ax.errorbar(x_ratio, y_norm, yerr=e_norm,
                        fmt="o", color=color, mfc=color, mec="white",
                        ms=7, lw=1.0, capsize=2.5,
                        label=f"{label}  Full State", zorder=3)
            pooled["full_state"]["x"].extend(x_ratio.tolist())
            pooled["full_state"]["y"].extend(y_norm.tolist())
            pooled["full_state"]["yerr"].extend(e_norm.tolist())

        # ibd
        ibd = td.get("ibd", {})
        if ibd:
            xs = np.array(sorted(ibd.keys()), dtype=float)
            ys = np.array([ibd[int(x)]["mean"] for x in xs])
            es = np.array([ibd[int(x)]["std"]  for x in xs])
            x_ratio = xs / true_dim
            y_norm  = ys / R_oracle
            e_norm  = es / R_oracle
            ax.errorbar(x_ratio, y_norm, yerr=e_norm,
                        fmt="s", color=color, mfc="white", mec=color,
                        ms=7, lw=1.0, capsize=2.5,
                        label=f"{label}  IBD", zorder=4)
            pooled["ibd"]["x"].extend(x_ratio.tolist())
            pooled["ibd"]["y"].extend(y_norm.tolist())
            pooled["ibd"]["yerr"].extend(e_norm.tolist())

    # Fit pooled Full State power law (only on x >= fit_x_min)
    fits: Dict = {}
    if pooled["full_state"]["x"]:
        fit_fs = fit_power_law(np.array(pooled["full_state"]["x"]),
                               np.array(pooled["full_state"]["y"]),
                               np.array(pooled["full_state"]["yerr"]),
                               x_min=fit_x_min)
        fits["full_state_pooled"] = fit_fs
        if fit_fs is not None:
            x_lo = max(fit_x_min, 1e-3)
            x_hi = max(pooled["full_state"]["x"]) * 1.2
            xx = np.geomspace(x_lo, x_hi, 100)
            yy = power_law(xx, fit_fs["a"], fit_fs["alpha"])
            fit_label = (rf"Full State fit ($d/d_c \geq {fit_x_min:g}$): "
                         rf"$\alpha = {fit_fs['alpha']:.3f}$ "
                         rf"[{fit_fs['alpha_lo']:.3f}, {fit_fs['alpha_hi']:.3f}]   "
                         rf"$R^2={fit_fs['r2']:.2f}$")
            ax.plot(xx, yy, "-", color="#BB4E51", lw=2.0, alpha=0.9,
                    label=fit_label, zorder=2)

    # Optional fit on IBD (expected slope ≈ 0). Same x_min for consistency.
    if fit_ibd and pooled["ibd"]["x"]:
        fit_ibd_res = fit_power_law(np.array(pooled["ibd"]["x"]),
                                    np.array(pooled["ibd"]["y"]),
                                    np.array(pooled["ibd"]["yerr"]),
                                    x_min=fit_x_min)
        fits["ibd_pooled"] = fit_ibd_res
        if fit_ibd_res is not None:
            x_lo = max(fit_x_min, 1e-3)
            x_hi = max(pooled["ibd"]["x"]) * 1.2
            xx = np.geomspace(x_lo, x_hi, 100)
            yy = power_law(xx, fit_ibd_res["a"], fit_ibd_res["alpha"])
            # IBD is expected to be flat (alpha~0); R^2 isn't informative
            # there, so we just report the exponent + CI and label it as
            # "no degradation" when CI sits at zero.
            ibd_label = (rf"IBD fit ($d/d_c \geq {fit_x_min:g}$): "
                         rf"$\alpha = {fit_ibd_res['alpha']:+.3f}$ "
                         rf"[{fit_ibd_res['alpha_lo']:+.3f}, "
                         rf"{fit_ibd_res['alpha_hi']:+.3f}]")
            ax.plot(xx, yy, "-", color="#4D73CA", lw=2.0, alpha=0.9,
                    label=ibd_label, zorder=2)

    # Optional vertical guide line at the fit threshold
    if fit_x_min > 0:
        ax.axvline(fit_x_min, color="#999999", ls=":", lw=0.8, alpha=0.7,
                   zorder=0)

    # Per-task Full State fits (for the printed summary, not plotted)
    for task_key in plotted_tasks:
        td = data[task_key]
        oracle_means = [v["mean"] for v in td.get("oracle", {}).values()]
        if not oracle_means:
            continue
        R_oracle = float(np.mean(oracle_means))
        fs = td.get("full_state", {})
        if not fs:
            continue
        xs = np.array(sorted(fs.keys()), dtype=float)
        ys = np.array([fs[int(x)]["mean"] for x in xs]) / R_oracle
        es = np.array([fs[int(x)]["std"]  for x in xs]) / R_oracle
        x_ratio = xs / td["true_dim"]
        fits[f"full_state_{task_key}"] = fit_power_law(
            x_ratio, ys, es, x_min=fit_x_min)

    # Reference line at oracle level
    oracle_handle = ax.axhline(1.0, color="#60A765", ls="--", lw=1.0, alpha=0.7,
                                label="Oracle level", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Distractor-to-signal ratio  $d / d_c$", fontsize=11)
    ax.set_ylabel(r"Normalized return  $R / R_{\mathrm{oracle}}$", fontsize=11)
    ax.set_title("Distractor scaling: tasks collapse onto a common ratio axis",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, ls="--")

    # Reorder legend: fit lines first (the headline result), then task markers,
    # then oracle reference line. Two-column layout for readability when fit
    # labels include alpha + CI + R^2.
    handles, labels = ax.get_legend_handles_labels()
    fit_keywords = ("Full State fit", "IBD fit")
    fit_pairs   = [(h, l) for h, l in zip(handles, labels)
                   if any(k in l for k in fit_keywords)]
    other_pairs = [(h, l) for h, l in zip(handles, labels)
                   if not any(k in l for k in fit_keywords)]
    ordered = fit_pairs + other_pairs
    if ordered:
        ax.legend([h for h, _ in ordered], [l for _, l in ordered],
                  fontsize=8, loc="lower left",
                  framealpha=0.92, ncol=1)

    # sensible y-range: don't let degenerate near-zero points squash the plot
    all_y = np.array(pooled["full_state"]["y"] + pooled["ibd"]["y"])
    if len(all_y):
        y_lo = max(min(all_y[all_y > 0]) * 0.6, 1e-3)
        y_hi = max(max(all_y) * 1.4, 1.5)
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.with_suffix('.pdf')}")
    print(f"  wrote {out_path.with_suffix('.png')}")
    return fits


# ─── stdout summary ─────────────────────────────────────────────────────────

def print_summary(data: Dict, fits: Dict) -> None:
    print()
    print("=" * 78)
    print(" SCALING SWEEP SUMMARY")
    print("=" * 78)
    for task_key, td in data.items():
        true_dim = td["true_dim"]
        _, _, _, label = TASK_SPECS[task_key]
        print(f"\n  {label}  (d_c = {true_dim})")
        print(f"    {'d':>5}  {'d/d_c':>6}  "
              f"{'full_state':>16}  {'ibd':>16}  {'oracle':>16}")
        all_d = sorted(set().union(*(td[m].keys()
                       for m in METHODS if m in td and td[m])))
        for d in all_d:
            row = f"    {d:>5}  {d/true_dim:>6.2f}"
            for m in METHODS:
                cell = td.get(m, {}).get(d)
                if cell is None:
                    row += f"  {'—':>16}"
                else:
                    row += f"  {cell['mean']:>7.1f}±{cell['std']:>5.1f}(n={cell['n']})"
            print(row)

    print()
    print("-" * 78)
    print(" POWER-LAW FITS  (R / R_oracle = a * (d/d_c)^(-alpha))")
    print("-" * 78)
    for name, f in fits.items():
        if f is None:
            print(f"  {name:30s}  insufficient data")
            continue
        xmin_tag = (f"  fit on d/d_c>={f['x_min']:g}"
                    if f.get("x_min", 0) > 0 else "")
        # flag low-quality fits — typically single-task with too few
        # degradation-regime points or non-monotonic data
        flag = ""
        if f["r2"] < 0.5 and not name.endswith("_pooled"):
            flag = "   [LOW R^2 — DO NOT REPORT]"
        elif f["r2"] < 0.5:
            flag = "   [low R^2]"
        print(f"  {name:30s}  alpha = {f['alpha']:+.3f}  "
              f"[{f['alpha_lo']:+.3f}, {f['alpha_hi']:+.3f}]   "
              f"a = {f['a']:.3f}   R^2 = {f['r2']:+.3f}   n = {f['n']}"
              f"{xmin_tag}{flag}")
    print()
    print(" Interpretation:")
    print("   alpha ~ 0   → method is robust to distractor scaling")
    print("   alpha > 0   → return decays as a power of (d/d_c)")
    print("   The Full State pooled fit is the headline scaling exponent.")
    print("=" * 78)


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--results_dir", type=str, default="results/scaling",
                   help="Directory containing per_seed/ subdir")
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["walker_walk", "cheetah_run", "reacher_hard"])
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--no_fit_ibd", action="store_true",
                   help="Don't draw the IBD power-law fit line")
    p.add_argument("--fit_x_min", type=float, default=1.0,
                   help="Restrict power-law fit to points with d/d_c >= "
                        "FIT_X_MIN. Default 1.0 (degradation regime). "
                        "Set 0 to fit all points.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(results_dir, args.tasks)
    # Drop tasks that ended up with nothing
    data = {t: v for t, v in data.items()
            if any(v.get(m) for m in METHODS)}
    if not data:
        raise SystemExit(f"No data found under {results_dir}")

    print(f"Loaded data for {len(data)} task(s) from {results_dir}")

    print("\n[Figure A] linear-scale distractor curves...")
    plot_curves(data, out_dir / "fig_scaling_curves")

    print("\n[Figure B] log-log scaling law with power-law fit...")
    fits = plot_scaling_law(data, out_dir / "fig_scaling_law",
                            fit_ibd=not args.no_fit_ibd,
                            fit_x_min=args.fit_x_min)

    print_summary(data, fits)


if __name__ == "__main__":
    main()