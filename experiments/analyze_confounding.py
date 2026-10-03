"""
Analysis for the confounded-distractor factorial experiment.

Reports, in this order:

  1. Per-cell precision / recall / F1 with the three error classes split
     (missed true dims, leaked confounded dims, leaked exogenous dims).
     F1 is reported but never used alone — it hides which error moved.

  2. Two-way ANOVA (kappa x lambda) per method with the **interaction**
     term highlighted.  The causal-graph prediction is specifically that
     degradation requires BOTH edges (C->a and C->d): neither main effect
     alone should produce confounded-dim leakage.  That is an interaction
     hypothesis, so it is tested as one.  Partial eta^2 with a
     seed-bootstrap CI accompanies each term.

  3. Empirical FDR against the nominal level, split into the
     theory-relevant part (confounded dims) and the pre-existing
     exogenous leakage.

  4. IBD test-statistic / correction ablation: {Welch, KS} x {BH, BY},
     all computed on the same collected sample.

  5. Probe-policy covariates per cell (saturation, entropy, KL to uniform,
     branch contrast) so that any recall loss can be explained mechanically.

Usage::

    python experiments/analyze_confounding.py --results_dir results/confounding
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METHOD_LABEL = {
    "ibd": "IBD (ours)",
    "inverse_dyn": "Multistep inverse dyn.",
    "mutual_info": "Mutual information",
    "grad_attr": "Gradient attribution",
    "cond_mi": "Cond. MI (fwd model)",
    "variance": "Variance",
    "random_mask": "Random mask",
}
ORDER = ["ibd", "inverse_dyn", "grad_attr", "mutual_info", "cond_mi",
         "variance", "random_mask"]
ALPHA = 0.05


def load(results_dir: str):
    recs = []
    for p in sorted(Path(results_dir).glob("per_run/*.json")):
        with open(p) as f:
            r = json.load(f)
        if "error" not in r:
            recs.append(r)
    return recs


def _sorted_methods(recs):
    return sorted({r["method"] for r in recs},
                  key=lambda m: ORDER.index(m) if m in ORDER else 99)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PER-CELL TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def per_cell_table(recs, domain=None):
    rs = [r for r in recs if domain is None or r["domain"] == domain]
    if not rs:
        return
    cells = sorted({(r["kappa"], r["lam"]) for r in rs})
    n_conf = rs[0]["n_confounded"]
    n_true = rs[0]["n_true"]

    print(f"\n{'='*104}")
    print(f"PER-CELL RESULTS   {domain or 'all'}   "
          f"(n_true={n_true}, n_confounded={n_conf}, "
          f"n_exogenous={rs[0]['n_exogenous']})")
    print(f"{'='*104}")
    print(f"{'method':<22}{'kappa':>6}{'lam':>6} | {'precision':>15}"
          f"{'recall':>15}{'F1':>15} | {'miss':>6}{'FPconf':>8}{'FPexo':>7}")
    print("-" * 104)

    for m in _sorted_methods(rs):
        for (k, l) in cells:
            sel = [r for r in rs if r["method"] == m
                   and r["kappa"] == k and r["lam"] == l]
            if not sel:
                continue

            def ms(key, prec=3):
                v = [r[key] for r in sel]
                return f"{np.mean(v):.{prec}f}±{np.std(v):.{prec}f}"

            def mi(key):
                v = [r[key] for r in sel]
                return f"{np.mean(v):.1f}"

            both = (k > 0 and l > 0)
            tag = " *" if both else "  "
            print(f"{METHOD_LABEL.get(m, m):<22}{k:>6g}{l:>6g} |"
                  f"{ms('precision'):>15}{ms('recall'):>15}{ms('f1'):>15} |"
                  f"{mi('fn'):>6}{mi('fp_confounded'):>8}"
                  f"{mi('fp_exogenous'):>7}{tag}")
        print("-" * 104)
    print("  * = both edges active (C->a and C->d); the only cell where the "
          "backdoor path exists.")
    print(f"  n_seeds = {len(sel)};  values are mean±std across seeds.")


def raw_counts_table(recs, domain=None):
    """Per-cell raw confusion counts, summed over seeds.

    The precision/recall/F1 table reports the *mean of per-seed ratios*,
    which by Jensen does not equal the ratio computed from mean counts --
    so those columns cannot be reconciled arithmetically.  These are the
    underlying integers; every rate in this analysis is derivable from
    them.  Reported as totals over seeds plus the per-seed mean.
    """
    rs = [r for r in recs if domain is None or r["domain"] == domain]
    if not rs:
        return
    cells = sorted({(r["kappa"], r["lam"]) for r in rs})
    n_true = rs[0]["n_true"]
    n_conf = rs[0]["n_confounded"]
    n_exo = rs[0]["n_exogenous"]

    print(f"\n{'='*104}")
    print(f"RAW COUNTS   {domain or 'all'}   "
          f"(per cell: totals over seeds, mean per seed in parentheses)")
    print(f"  n_true={n_true}  n_confounded={n_conf}  n_exogenous={n_exo}")
    print(f"{'='*104}")
    print(f"{'method':<22}{'kappa':>6}{'lam':>5} |{'tp':>12}{'fn':>12}"
          f"{'FPconf':>12}{'FPexo':>12} |{'selected':>12}")
    print("-" * 104)
    for m in _sorted_methods(rs):
        for (k, l) in cells:
            sel = [r for r in rs if r["method"] == m
                   and r["kappa"] == k and r["lam"] == l]
            if not sel:
                continue

            def c(key):
                v = [r[key] for r in sel]
                return f"{int(np.sum(v)):>5} ({np.mean(v):4.1f})"
            tag = " *" if (k > 0 and l > 0) else ""
            print(f"{METHOD_LABEL.get(m, m):<22}{k:>6g}{l:>5g} |"
                  f"{c('tp'):>12}{c('fn'):>12}{c('fp_confounded'):>12}"
                  f"{c('fp_exogenous'):>12} |{c('n_selected'):>12}{tag}")
        print("-" * 104)
    print(f"  n_seeds = {len(sel)}.  tp + fn = n_true; "
          f"tp + FPconf + FPexo = selected.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TWO-WAY ANOVA WITH INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════

def two_way_anova(cells: dict):
    """Balanced two-way ANOVA.  cells maps (a_level, b_level) -> [values].

    Returns dict of term -> (F, p, partial_eta_sq).
    """
    a_levels = sorted({a for a, _ in cells})
    b_levels = sorted({b for _, b in cells})
    a, b = len(a_levels), len(b_levels)
    ns = {len(v) for v in cells.values()}
    if len(ns) != 1:
        return None                     # unbalanced -> skip
    n = ns.pop()
    if n < 2 or a < 2 or b < 2:
        return None

    Y = np.array([[cells[(ai, bj)] for bj in b_levels] for ai in a_levels])
    grand = Y.mean()
    cell_m = Y.mean(axis=2)                    # (a, b)
    row_m = cell_m.mean(axis=1)                # (a,)
    col_m = cell_m.mean(axis=0)                # (b,)

    ss_a = b * n * np.sum((row_m - grand) ** 2)
    ss_b = a * n * np.sum((col_m - grand) ** 2)
    ss_ab = n * np.sum((cell_m - row_m[:, None] - col_m[None, :]
                        + grand) ** 2)
    ss_e = np.sum((Y - cell_m[:, :, None]) ** 2)

    df_a, df_b = a - 1, b - 1
    df_ab, df_e = (a - 1) * (b - 1), a * b * (n - 1)
    if df_e == 0:
        return None

    # Zero residual variance is not a missing result — it is the strongest
    # form of one.  Two very different cases hide behind it:
    #   * every cell identical  -> no effect at all
    #   * cells differ, but every seed within a cell agrees exactly
    #     -> a perfectly deterministic effect (F is 0/0, not undefined
    #        in substance).  Dropping these as "unbalanced" silently
    #        deleted the cleanest interaction in the reacher results.
    if ss_e <= 1e-12:
        out = {}
        for name, ss, df in (("kappa", ss_a, df_a), ("lambda", ss_b, df_b),
                             ("kappa x lambda", ss_ab, df_ab)):
            if ss <= 1e-12:
                out[name] = (0.0, 1.0, 0.0)          # constant everywhere
            else:
                out[name] = (float("inf"), 0.0, 1.0)  # deterministic effect
        return out

    ms_e = ss_e / df_e

    out = {}
    for name, ss, df in (("kappa", ss_a, df_a), ("lambda", ss_b, df_b),
                         ("kappa x lambda", ss_ab, df_ab)):
        F = (ss / df) / ms_e
        p = float(stats.f.sf(F, df, df_e))
        eta = ss / (ss + ss_e)              # partial eta^2
        out[name] = (float(F), p, float(eta))
    return out


def _bootstrap_eta(cells: dict, term: str, B: int = 4000, seed: int = 0):
    """Percentile CI for partial eta^2, resampling seeds within cells."""
    rng = np.random.RandomState(seed)
    keys = list(cells)
    n = len(cells[keys[0]])
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        boot = {k: [cells[k][i] for i in idx] for k in keys}
        res = two_way_anova(boot)
        if res is not None and np.isfinite(res[term][0]):
            vals.append(res[term][2])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def anova_report(recs, dv="confounded_fp_rate", domain=None, boot=True):
    rs = [r for r in recs if domain is None or r["domain"] == domain]
    print(f"\n{'='*104}")
    print(f"TWO-WAY ANOVA   DV = {dv}   {domain or 'all'}")
    print(f"{'='*104}")
    print(f"{'method':<22}{'term':<18}{'F':>10}{'p':>12}"
          f"{'partial eta^2':>15}{'95% CI (bootstrap)':>26}")
    print("-" * 104)
    for m in _sorted_methods(rs):
        cells = defaultdict(list)
        for r in rs:
            if r["method"] == m:
                cells[(r["kappa"], r["lam"])].append(r[dv])
        res = two_way_anova(dict(cells))
        if res is None:
            print(f"{METHOD_LABEL.get(m, m):<22}"
                  f"(design not balanced / too few levels — skipped)")
            continue
        for term, (F, p, eta) in res.items():
            ci = ""
            if boot and term == "kappa x lambda" and np.isfinite(F):
                lo, hi = _bootstrap_eta(dict(cells), term)
                ci = f"[{lo:.3f}, {hi:.3f}]"
            elif not np.isfinite(F):
                ci = "deterministic (zero within-cell var)"
            star = "  <-- interaction" if term == "kappa x lambda" else ""
            f_str = "     inf" if not np.isfinite(F) else f"{F:>10.2f}"
            print(f"{METHOD_LABEL.get(m, m) if term == 'kappa' else '':<22}"
                  f"{term:<18}{f_str:>10}{p:>12.3g}{eta:>15.3f}"
                  f"{ci:>38}{star}")
        print("-" * 104)
    print("  The causal-graph prediction is an INTERACTION: confounded-dim")
    print("  leakage requires both C->a and C->d, so neither main effect")
    print("  alone should suffice.  Read the interaction row first.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMPIRICAL FDR
# ═══════════════════════════════════════════════════════════════════════════════

def fdr_report(recs, domain=None):
    rs = [r for r in recs if (domain is None or r["domain"] == domain)
          and r["method"] == "ibd"]
    if not rs:
        return
    print(f"\n{'='*104}")
    print(f"EMPIRICAL FDR vs NOMINAL alpha={ALPHA}   (IBD, {domain or 'all'})")
    print(f"{'='*104}")
    print(f"{'kappa':>6}{'lam':>6} | {'FDR(confounded)':>20}"
          f"{'FDR(exogenous)':>18}{'FDR(all)':>18} | {'verdict':<30}")
    print("-" * 104)
    for (k, l) in sorted({(r["kappa"], r["lam"]) for r in rs}):
        sel = [r for r in rs if r["kappa"] == k and r["lam"] == l]
        fc = [r["fdr_confounded"] for r in sel]
        fe = [r["fdr_exogenous"] for r in sel]
        fa = [r["fdr_all"] for r in sel]
        ok = np.mean(fc) <= ALPHA
        verdict = ("confounded FDR within alpha" if ok
                   else "confounded FDR EXCEEDS alpha -> use BY")
        print(f"{k:>6g}{l:>6g} | {np.mean(fc):.3f}±{np.std(fc):.3f}".ljust(48)
              + f"{np.mean(fe):.3f}±{np.std(fe):.3f}".rjust(14)
              + f"{np.mean(fa):.3f}±{np.std(fa):.3f}".rjust(18)
              + f" | {verdict:<30}")
    print("-" * 104)
    print("  FDR(confounded) is what Prop. 3.3 speaks to.  FDR(exogenous) is")
    print("  a pre-existing baseline error rate: if it is roughly constant")
    print("  across cells it is not caused by confounding.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IBD VARIANT ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def variant_report(recs, domain=None):
    rs = [r for r in recs if (domain is None or r["domain"] == domain)
          and r["method"] == "ibd" and r.get("variant_metrics")]
    if not rs:
        return
    variants = sorted(rs[0]["variant_metrics"])
    print(f"\n{'='*104}")
    print(f"IBD TEST-STATISTIC x CORRECTION ABLATION   {domain or 'all'}")
    print("(all four computed on the SAME collected sample)")
    print(f"{'='*104}")
    print(f"{'kappa':>6}{'lam':>6}{'variant':>12} | {'precision':>14}"
          f"{'recall':>14}{'F1':>14} | {'FPconf':>8}{'FDR(conf)':>11}")
    print("-" * 104)
    for (k, l) in sorted({(r["kappa"], r["lam"]) for r in rs}):
        sel = [r for r in rs if r["kappa"] == k and r["lam"] == l]
        for v in variants:
            def ms(key):
                vals = [r["variant_metrics"][v][key] for r in sel]
                return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

            def mi(key):
                vals = [r["variant_metrics"][v][key] for r in sel]
                return f"{np.mean(vals):.1f}"
            mark = " *" if v == "welch_bh" else ""
            print(f"{k:>6g}{l:>6g}{v:>12} | {ms('precision'):>14}"
                  f"{ms('recall'):>14}{ms('f1'):>14} | "
                  f"{mi('fp_confounded'):>8}  {ms('fdr_confounded'):>13}{mark}")
        print("-" * 104)
    print("  * = configuration reported in the main paper.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COVARIATES
# ═══════════════════════════════════════════════════════════════════════════════

def covariate_report(recs, domain=None):
    rs = [r for r in recs if (domain is None or r["domain"] == domain)
          and r.get("covariates")]
    if not rs:
        return
    print(f"\n{'='*104}")
    print(f"PROBE-POLICY COVARIATES   {domain or 'all'}")
    print(f"{'='*104}")
    print(f"{'kappa':>6}{'lam':>6} | {'sat_frac':>10}{'action_std':>12}"
          f"{'diff_entropy':>14}{'KL(pi||Unif)':>14} | "
          f"{'mean g (true)':>14}{'mean g (dist)':>14}")
    print("-" * 104)
    for (k, l) in sorted({(r["kappa"], r["lam"]) for r in rs}):
        sel = [r for r in rs if r["kappa"] == k and r["lam"] == l]
        cov = {key: np.mean([r["covariates"][key] for r in sel])
               for key in sel[0]["covariates"]}
        ibd = [r for r in sel if r["method"] == "ibd"
               and "mean_hedges_g_true" in r.get("info", {})]
        gt = np.mean([r["info"]["mean_hedges_g_true"] for r in ibd]) if ibd else float("nan")
        gd = np.mean([r["info"]["mean_hedges_g_distractor"] for r in ibd]) if ibd else float("nan")
        print(f"{k:>6g}{l:>6g} | {cov['sat_frac']:>10.3f}"
              f"{cov['action_std']:>12.3f}{cov['diff_entropy']:>14.3f}"
              f"{cov['kl_to_uniform']:>14.3f} | {gt:>14.3f}{gd:>14.3f}")
    print("-" * 104)
    print("  Uniform[-1,1] reference: diff_entropy=0.693, KL=0.")
    print("  Clipping means large kappa drives the probe towards saturated")
    print("  bang-bang, NOT towards uniform — so KL is non-monotone in kappa")
    print("  and the branch contrast (mean g on true dims) must be read")
    print("  directly rather than inferred from kappa.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results/confounding")
    ap.add_argument("--domain", type=str, default=None)
    ap.add_argument("--no_bootstrap", action="store_true")
    args = ap.parse_args()

    recs = load(args.results_dir)
    if not recs:
        print(f"No results in {args.results_dir}/per_run/")
        return
    domains = sorted({r["domain"] for r in recs})
    print(f"loaded {len(recs)} runs over domains: {domains}")

    for dom in ([args.domain] if args.domain else domains):
        per_cell_table(recs, dom)
        raw_counts_table(recs, dom)
        anova_report(recs, "confounded_fp_rate", dom, boot=not args.no_bootstrap)
        anova_report(recs, "f1", dom, boot=not args.no_bootstrap)
        fdr_report(recs, dom)
        variant_report(recs, dom)
        covariate_report(recs, dom)


if __name__ == "__main__":
    main()
