#!/usr/bin/env python
"""
Aggregate per-seed JSONs into a baseline comparison table.

Usage:
    python aggregate_invdyn.py
    python aggregate_invdyn.py --results_dir results/dmcontrol/per_seed

Outputs:
    1. Prints a plain-text table of (setting, method) -> return mean ± std
       and (for dim-selection methods) boundary P/R/F1.
    2. Writes a LaTeX snippet to aggregate_invdyn_table.tex that you can
       paste into the paper.

Designed to be drop-in: no changes needed to the existing codebase.
The script reads all JSONs matching the 4 representative settings and
aggregates across seeds.  If you have IBD / Oracle / Full State results
for the same 4 settings in per_seed/, it will include them for context.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SETTINGS = [
    ("reacher", "hard",  "medium"),
    ("walker",  "walk",  "hard"),
    ("cheetah", "run",   "medium"),
    ("finger",  "spin",  "medium"),
]

# methods we care about in the comparison table, in display order
METHODS_ORDER = [
    "full_state", "inverse_dyn", "ibd", "oracle",
]
# Optional: include these too if present
METHODS_OPTIONAL = ["mutual_info", "variance", "cond_mi", "grad_attr"]

METHOD_DISPLAY = {
    "full_state":  "Full State",
    "oracle":      "Oracle",
    "ibd":         "IBD (ours)",
    "mutual_info": "MI",
    "variance":    "Variance",
    "cond_mi":     "Cond. MI",
    "grad_attr":   "Grad. Attr.",
    "inverse_dyn": "Inverse Dyn.",
}


def load_per_seed(results_dir: Path):
    """Return nested dict: {(dom, task, distr)}{method} -> list of results."""
    bucket = defaultdict(lambda: defaultdict(list))
    for p in sorted(results_dir.glob("*.json")):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception as e:
            print(f"  skip unreadable: {p.name}  ({e})")
            continue
        key = (r["domain"], r["task"], r["distractors"])
        bucket[key][r["method"]].append(r)
    return bucket


def summarize(runs):
    """Mean / std across seeds for final_return, plus boundary metrics."""
    returns = [r["final_return"] for r in runs]
    row = {
        "n_seeds":    len(runs),
        "return_mean": float(np.mean(returns)),
        "return_std":  float(np.std(returns)),
    }
    # boundary metrics (only for dim-selection methods)
    bm_runs = [r for r in runs if r.get("boundary_metrics")]
    if bm_runs:
        for key in ("precision", "recall", "f1"):
            vals = [r["boundary_metrics"].get(key, 0.0) for r in bm_runs]
            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"]  = float(np.std(vals))
    # effective dim (selector budget actually used)
    eff = [r.get("effective_dim", r.get("obs_dim")) for r in runs]
    row["eff_dim_mean"] = float(np.mean(eff))
    # wall-time
    tt = [r.get("train_time_s", 0.0) for r in runs]
    row["train_time_s_mean"] = float(np.mean(tt))
    return row


def print_table(bucket):
    # Work out which optional methods actually have any data
    optional_present = set()
    for _, per_method in bucket.items():
        for m in METHODS_OPTIONAL:
            if per_method.get(m):
                optional_present.add(m)
    methods = METHODS_ORDER + [m for m in METHODS_OPTIONAL
                               if m in optional_present]

    # header
    print()
    print("=" * 100)
    print("AGGREGATED RESULTS (return: mean ± std over seeds)")
    print("=" * 100)
    hdr = f"{'Setting':<28}" + "".join(
        f"{METHOD_DISPLAY.get(m, m):>16}" for m in methods)
    print(hdr)
    print("-" * len(hdr))
    for key in SETTINGS:
        if key not in bucket:
            continue
        dom, task, distr = key
        label = f"{dom}_{task} ({distr})"
        row = f"{label:<28}"
        for m in methods:
            runs = bucket[key].get(m, [])
            if not runs:
                row += f"{'—':>16}"
            else:
                s = summarize(runs)
                cell = f"{s['return_mean']:.0f}±{s['return_std']:.0f}"
                row += f"{cell:>16}"
        print(row)
    print()

    # Boundary accuracy (only for dim-selection methods that have it)
    dim_methods = [m for m in methods
                   if m not in ("full_state", "oracle")]
    print("=" * 100)
    print("BOUNDARY ACCURACY (P / R / F1, mean over seeds)")
    print("=" * 100)
    hdr2 = f"{'Setting':<28}" + "".join(
        f"{METHOD_DISPLAY.get(m, m):>22}" for m in dim_methods)
    print(hdr2)
    print("-" * len(hdr2))
    for key in SETTINGS:
        if key not in bucket:
            continue
        dom, task, distr = key
        label = f"{dom}_{task} ({distr})"
        row = f"{label:<28}"
        for m in dim_methods:
            runs = bucket[key].get(m, [])
            s = summarize(runs) if runs else None
            if s and "f1_mean" in s:
                cell = (f"{s['precision_mean']:.2f}/"
                        f"{s['recall_mean']:.2f}/"
                        f"{s['f1_mean']:.2f}")
                row += f"{cell:>22}"
            else:
                row += f"{'—':>22}"
        print(row)
    print()

    # Also report seed counts & wall-time for reproducibility
    print("=" * 100)
    print("SEED COUNTS & MEAN TRAIN TIME (seconds)")
    print("=" * 100)
    for key in SETTINGS:
        if key not in bucket:
            continue
        dom, task, distr = key
        print(f"  {dom}_{task}_{distr}:")
        for m in methods:
            runs = bucket[key].get(m, [])
            if runs:
                s = summarize(runs)
                print(f"    {METHOD_DISPLAY.get(m, m):<18} "
                      f"n={s['n_seeds']}  eff_dim={s['eff_dim_mean']:.0f}  "
                      f"train_time={s['train_time_s_mean']:.0f}s")
    print()


def write_latex(bucket, out_path):
    """Emit a minimal LaTeX tabular for the inverse_dyn comparison table."""
    present_optional = [m for m in METHODS_OPTIONAL
                        if any(bucket[k].get(m) for k in SETTINGS
                               if k in bucket)]
    methods = METHODS_ORDER + present_optional

    lines = []
    lines.append("% Auto-generated by aggregate_invdyn.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison against multistep inverse dynamics"
                 r" baseline on 4 representative settings"
                 r" (episode return, mean $\pm$ std over 3 seeds).}")
    lines.append(r"\label{tab:invdyn}")
    col_spec = "l" + "c" * len(methods)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = "Setting & " + " & ".join(
        METHOD_DISPLAY.get(m, m) for m in methods) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for key in SETTINGS:
        if key not in bucket:
            continue
        dom, task, distr = key
        label = f"{dom}\\_{task} ({distr})"
        cells = [label]
        for m in methods:
            runs = bucket[key].get(m, [])
            if not runs:
                cells.append("---")
            else:
                s = summarize(runs)
                cells.append(f"{s['return_mean']:.0f}$\\pm${s['return_std']:.0f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX snippet written to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default="results/dmcontrol/per_seed",
                    help="directory of per-seed JSONs")
    ap.add_argument("--latex_out",
                    default="aggregate_invdyn_table.tex")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    if not rd.exists():
        raise SystemExit(f"no such dir: {rd}")

    bucket = load_per_seed(rd)
    if not bucket:
        raise SystemExit("no JSONs found")

    # Report what was found
    total = sum(len(v) for per in bucket.values() for v in per.values())
    print(f"Loaded {total} runs across {len(bucket)} settings "
          f"from {rd}")

    print_table(bucket)
    write_latex(bucket, Path(args.latex_out))


if __name__ == "__main__":
    main()
