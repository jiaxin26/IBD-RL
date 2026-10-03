"""
Confounding-strength sweep: boundary F1 vs (kappa, lambda)
============================================================

Runs every selector on the confounded-distractor regime over a grid of
confounding strengths and records precision / recall / F1 against ground
truth.  This is the *boundary-discovery* experiment only — no downstream
RL training — which is what makes the full sweep cheap.

Crucially, **every method collects its observational data under the same
confounded behaviour policy** ``a_t = pi_probe(o_t) + kappa*K C_t``.  If
the baselines were allowed to roll out uniform-random actions (their
default) they would be performing the intervention themselves and the
comparison would be vacuous.

Usage::

    # one point
    python experiments/run_confounding_sweep.py --single \\
        --domain walker --task walk --method ibd --kappa 0.5 --lam 1.0 --seed 0

    # full grid, 12 workers
    python experiments/run_confounding_sweep.py --grid --parallel 12

    # DGP sanity check (no selectors)
    python experiments/run_confounding_sweep.py --sanity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METHODS = ["ibd", "mutual_info", "variance", "cond_mi", "grad_attr",
           "inverse_dyn", "random_mask"]

DEFAULT_KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0]
DEFAULT_LAMS = [0.0, 0.25, 0.5, 1.0]

# The primary IBD configuration reported in the paper
PRIMARY_VARIANT = "welch_bh"

# ── epochs semantics: paper text vs. baselines.py ────────────────────────
# ConditionalMISelector and GradientAttributionSelector both implement
# "epochs" as ONE minibatch per iteration:
#     for _ in range(epochs): idx = randperm(n)[:batch_size]; step()
# so `epochs=50` is 50 *gradient steps*, not 50 passes over the data.  With
# 200x200 = 40000 transitions and batch_size=2048 one true epoch is ~20
# steps, so the documented 50 epochs corresponds to ~1000 steps -- the code
# trains these two baselines ~20x less than the paper describes.
# MultistepInverseDynamicsSelector is NOT affected: it runs a full inner
# pass per epoch.
#
# Passing `epochs = documented_epochs * steps_per_epoch` restores the
# documented budget without editing baselines.py (whose semantics the main
# results depend on).
CORRECT_EPOCHS = True
DOC_EPOCHS = {"cond_mi": 50, "grad_attr": 80}


def _steps_for(doc_epochs: int, n_samples: int, batch_size: int) -> int:
    """Gradient steps equivalent to `doc_epochs` true passes over the data."""
    steps_per_epoch = max(1, -(-n_samples // batch_size))   # ceil
    return int(doc_epochs * steps_per_epoch)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPLE-TESTING CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _bh_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up.  Controls FDR under independence / PRDS."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    idx = np.argsort(p)
    sp = p[idx]
    sig = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m):
        if sp[k] <= (k + 1) / m * alpha:
            max_k = k
    if max_k >= 0:
        sig[idx[:max_k + 1]] = True
    return sig


def _by_reject(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Yekutieli: BH at alpha / sum_{i=1..m} 1/i.

    Valid under *arbitrary* dependence across tests — which is the relevant
    regime here, since confounded distractors are coupled to one another
    (shared C) and, through the action channel, to the true dims.  Reported alongside BH as the conservative fallback.
    """
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return _bh_reject(p, alpha / c_m)


def ibd_all_variants(probe, policy, alpha: float = 0.05):
    """Run IBD once and score {Welch, KS} x {BH, BY} on the SAME sample.

    The two-sample tests and the multiple-testing corrections are both
    post-hoc functions of the collected trajectories, so all four variants
    come for the price of one data collection.  This covers the
    test-statistic comparison (Welch vs a distributional test) and provides
    the BY fallback without a second sweep.

    Returns:
        (variants, mean_g_true, diagnostics) where ``variants`` maps
        ``"<test>_<correction>"`` -> set of selected dims.
    """
    from scipy import stats as sp_stats

    t0 = time.time()
    baseline = probe._collect(policy, probe.n_baseline, intervention=None)
    interv = probe._collect(policy, probe.n_intervention,
                            intervention={"dim": "all", "mode": "randomize"})
    t_collect = time.time() - t0

    keys, p_welch, p_ks, eff = [], [], [], []
    for sd in range(probe.obs_dim):
        for h in probe.horizons:
            y_b = probe.cim._extract(baseline, sd, h, absolute=True)
            y_i = probe.cim._extract(interv, sd, h, absolute=True)
            if y_b is None or y_i is None:
                continue
            if len(y_b) < 10 or len(y_i) < 10:
                continue

            _, pw = sp_stats.ttest_ind(y_b, y_i, equal_var=False)
            _, pk = sp_stats.ks_2samp(y_b, y_i)

            n0, n1 = len(y_b), len(y_i)
            v0 = float(np.var(y_b, ddof=1))
            v1 = float(np.var(y_i, ddof=1))
            pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
            g = ((np.mean(y_b) - np.mean(y_i)) / pooled
                 if pooled > 1e-12 else 0.0)
            g *= 1 - 3 / (4 * (n0 + n1) - 9)      # Hedges correction

            keys.append((sd, h))
            p_welch.append(float(pw))
            p_ks.append(float(pk))
            eff.append(abs(float(g)))

    variants = {}
    for tname, pv in (("welch", np.asarray(p_welch)),
                      ("ks", np.asarray(p_ks))):
        for cname, fn in (("bh", _bh_reject), ("by", _by_reject)):
            sig = fn(pv, alpha)
            variants[f"{tname}_{cname}"] = {
                keys[i][0] for i in range(len(keys)) if sig[i]}

    # Per-dim max |Hedges' g| — the contrast between the two branches
    max_g = defaultdict(float)
    for i, (sd, _) in enumerate(keys):
        max_g[sd] = max(max_g[sd], eff[i])

    diagnostics = {
        "n_tests": len(keys),
        "time_collect_s": t_collect,
        "min_raw_p_welch": float(np.min(p_welch)) if p_welch else 1.0,
        "min_raw_p_ks": float(np.min(p_ks)) if p_ks else 1.0,
    }
    return variants, dict(max_g), diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _metrics(soi, env) -> Dict:
    """Precision / recall reported separately, errors split three ways.

    F1 alone hides which error moved.  The three error classes have
    different theoretical status:

      * ``fn``            — true dims missed.  A *power* property; degrades
                            with anything that shrinks the branch contrast.
      * ``fp_confounded`` — confounded distractors leaked in.  The ONLY
                            error class the identifiability claim speaks to;
                            Prop. 3.3 predicts this stays at the nominal
                            level regardless of confounding strength.
      * ``fp_exogenous``  — plain exogenous distractors leaked in.  A
                            pre-existing, confounding-independent baseline
                            error rate.

    Two FDR estimates are reported for the same reason: ``fdr_confounded``
    is the quantity the theory bounds, ``fdr_all`` is the honest overall
    number and includes the pre-existing exogenous leakage.
    """
    true_soi = env.true_dims
    tp = len(soi & true_soi)
    fp = len(soi - true_soi)
    fn = len(true_soi - soi)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    fp_conf = len(soi & env.confounded_dims)
    fp_exo = len(soi & env.exogenous_dims)
    n_sel = len(soi)

    return {
        "precision": prec, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "fp_confounded": fp_conf, "fp_exogenous": fp_exo,
        "n_selected": n_sel,
        # leaked confounded dims as a fraction of those available
        "confounded_fp_rate": fp_conf / max(len(env.confounded_dims), 1),
        # empirical FDR: false discoveries / total discoveries
        "fdr_all": fp / n_sel if n_sel > 0 else 0.0,
        "fdr_confounded": fp_conf / n_sel if n_sel > 0 else 0.0,
        "fdr_exogenous": fp_exo / n_sel if n_sel > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_point(domain: str, task: str, method: str,
              kappa: float, lam: float, seed: int,
              base_config: str = "medium",
              n_confounded: int = 12,
              n_baseline: int = 80, n_intervention: int = 80,
              traj_length: int = 200,
              horizons: Optional[List[int]] = None,
              baseline_episodes: int = 200,
              cheap_episodes: int = 50,
              episode_length: int = 200,
              results_dir: str = "results/confounding",
              overwrite: bool = False) -> Dict:
    """Discover the boundary with one method at one (kappa, lam) point."""

    out_dir = Path(results_dir) / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"{domain}_{task}_{method}"
             f"_k{kappa:g}_l{lam:g}_s{seed}.json")
    path = out_dir / fname
    if path.exists() and not overwrite:
        logger.info(f"SKIP (exists): {fname}")
        with open(path) as f:
            return json.load(f)

    from experiments.confounded_distractors import (
        make_confounded_env, StructuredProbePolicy, ConfoundedPolicy,
        measure_probe_covariates)
    from experiments.baselines import (
        RandomMask, MutualInfoSelector, VarianceSelector,
        ConditionalMISelector, GradientAttributionSelector,
        MultistepInverseDynamicsSelector)
    from ibd.probe import IBDProbe

    horizons = horizons or [1, 5, 10]
    t0 = time.time()

    env = make_confounded_env(domain, task, base_config=base_config,
                              n_confounded=n_confounded, lam=lam,
                              seed=seed + 2000)
    obs_dim = env.observation_space.shape[0]
    true_soi = env.true_dims
    n_true = env.true_obs_dim

    # Shared confounded behaviour policy — identical for every method
    behaviour = ConfoundedPolicy(env, StructuredProbePolicy(env, seed=seed),
                                 kappa=kappa, seed=seed)

    logger.info(f"{domain}_{task} | {method} | kappa={kappa:g} lam={lam:g} "
                f"| seed={seed} | obs={obs_dim} true={n_true} "
                f"exo={len(env.exogenous_dims)} conf={len(env.confounded_dims)}")

    # ── probe-policy covariates (explain any power loss, don't assert it) ──
    # These depend only on (env, kappa, lambda, seed), not on the selector,
    # so measure them once per cell -- on the IBD run -- instead of paying
    # for an extra rollout in every method.
    covariates = (measure_probe_covariates(env, behaviour, n_steps=3000,
                                           seed=seed + 55)
                  if method == "ibd" else {})

    info: Dict = {}
    variant_metrics: Dict = {}

    if method == "ibd":
        probe = IBDProbe(env, n_baseline=n_baseline,
                         n_intervention=n_intervention,
                         traj_length=traj_length, horizons=horizons,
                         seed=seed)
        variants, max_g, info = ibd_all_variants(probe, behaviour)
        soi = variants[PRIMARY_VARIANT]

        # Contrast between the baseline and intervention branch, restricted
        # to the dims the theory says should respond.  This is the covariate
        # that actually predicts recall.
        g_true = [max_g.get(d, 0.0) for d in sorted(true_soi)]
        info["mean_hedges_g_true"] = float(np.mean(g_true)) if g_true else 0.0
        info["mean_hedges_g_distractor"] = float(np.mean(
            [max_g.get(d, 0.0) for d in sorted(env.distractor_dims)]))

        for vname, vsoi in variants.items():
            variant_metrics[vname] = _metrics(vsoi, env)
    else:
        if method == "mutual_info":
            sel = MutualInfoSelector(env, n_episodes=cheap_episodes,
                                     episode_length=episode_length,
                                     n_select=n_true, seed=seed)
        elif method == "variance":
            sel = VarianceSelector(env, n_episodes=cheap_episodes,
                                   episode_length=episode_length,
                                   n_select=n_true, seed=seed)
        elif method == "cond_mi":
            ep = (_steps_for(DOC_EPOCHS["cond_mi"],
                             baseline_episodes * episode_length, 2048)
                  if CORRECT_EPOCHS else 50)
            sel = ConditionalMISelector(env, n_episodes=baseline_episodes,
                                        episode_length=episode_length,
                                        n_select=n_true, epochs=ep, seed=seed)
            info["grad_steps"] = ep
        elif method == "grad_attr":
            ep = (_steps_for(DOC_EPOCHS["grad_attr"],
                             baseline_episodes * episode_length, 2048)
                  if CORRECT_EPOCHS else 80)
            sel = GradientAttributionSelector(env, n_episodes=baseline_episodes,
                                              episode_length=episode_length,
                                              n_select=n_true, hidden=128,
                                              epochs=ep, seed=seed)
            info["grad_steps"] = ep
        elif method == "inverse_dyn":
            sel = MultistepInverseDynamicsSelector(
                env, n_episodes=baseline_episodes,
                episode_length=episode_length, horizon_k=3,
                n_select=n_true, hidden=128, epochs=80, seed=seed)
        elif method == "random_mask":
            sel = RandomMask(obs_dim, n_select=n_true, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        mask = (sel.discover() if method == "random_mask"
                else sel.discover(policy=behaviour))
        soi = mask.soi_dims

    m = _metrics(soi, env)

    rec_out = {
        "domain": domain, "task": task, "method": method,
        "kappa": kappa, "lam": lam, "seed": seed,
        "base_config": base_config, "n_confounded": n_confounded,
        "obs_dim": obs_dim, "n_true": n_true,
        "n_exogenous": len(env.exogenous_dims),
        **m,
        "selected": sorted(int(d) for d in soi),
        "covariates": covariates,
        "variant_metrics": variant_metrics,
        "wall_s": time.time() - t0,
        "info": {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else v) for k, v in info.items()},
    }

    with open(path, "w") as f:
        json.dump(rec_out, f, indent=2)

    logger.info(f"  -> P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} | miss={m['fn']} "
                f"FPconf={m['fp_confounded']}/{len(env.confounded_dims)} "
                f"FPexo={m['fp_exogenous']} | "
                f"FDR_all={m['fdr_all']:.3f} FDR_conf={m['fdr_confounded']:.3f}"
                f" | {rec_out['wall_s']:.0f}s")
    return rec_out


# ═══════════════════════════════════════════════════════════════════════════════
# DGP SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def sanity_check(domain="walker", task="walk", kappa=0.5, lam=1.0,
                 n_confounded=12, base_config="medium",
                 n_steps=20000, seed=0):
    """Verify the DGP does what the design claims, before spending compute.

    Checks:
      1. |corr(a_t, d_t)| is materially > 0 for confounded dims and ~0 for
         exogenous dims  -> the backdoor path exists.
      2. Under do(a ~ Unif), the marginal law of d is unchanged (KS test
         should NOT reject) while the true state's law does change
         -> the intervention severs C->a without touching C->d.
    """
    from scipy import stats
    from experiments.confounded_distractors import (
        make_confounded_env, StructuredProbePolicy, ConfoundedPolicy)

    env = make_confounded_env(domain, task, base_config=base_config,
                              n_confounded=n_confounded, lam=lam, seed=seed)
    pi = ConfoundedPolicy(env, StructuredProbePolicy(env, seed=seed),
                          kappa=kappa, seed=seed)
    rng = np.random.RandomState(seed)
    lo, hi = env.action_space.low, env.action_space.high

    def rollout(interventional: bool):
        obs, _ = env.reset(seed=seed + (1 if interventional else 0))
        O, A = [], []
        for t in range(n_steps):
            a = (rng.uniform(lo, hi) if interventional else pi(obs))
            O.append(np.asarray(obs).flatten())
            A.append(np.asarray(a).flatten())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()
        return np.asarray(O), np.asarray(A)

    O_b, A_b = rollout(False)
    O_i, A_i = rollout(True)

    true_d = sorted(env.true_dims)
    exo_d = sorted(env.exogenous_dims)
    conf_d = sorted(env.confounded_dims)

    def mean_abs_corr(O, A, dims):
        out = []
        for d in dims:
            c = [abs(np.corrcoef(O[:, d], A[:, j])[0, 1])
                 for j in range(A.shape[1])]
            out.append(np.nanmax(c))
        return float(np.mean(out))

    print("\n=== 1. observational action-obs correlation (baseline branch) ===")
    print(f"  true dims       : {mean_abs_corr(O_b, A_b, true_d):.4f}")
    print(f"  exogenous dims  : {mean_abs_corr(O_b, A_b, exo_d):.4f}")
    print(f"  CONFOUNDED dims : {mean_abs_corr(O_b, A_b, conf_d):.4f}"
          "   <- must be well above exogenous")

    print("\n=== 2. do(a~Unif): is the marginal law of each block moved? ===")
    for name, dims in [("true", true_d), ("exogenous", exo_d),
                       ("CONFOUNDED", conf_d)]:
        ps = [stats.ks_2samp(O_b[:, d], O_i[:, d]).pvalue for d in dims]
        frac_moved = float(np.mean([p < 0.01 for p in ps]))
        print(f"  {name:<12}: median KS p={np.median(ps):.3e}  "
              f"frac dims shifted={frac_moved:.2f}")
    print("  (expect: true -> shifted; exogenous & CONFOUNDED -> not shifted)")


# ═══════════════════════════════════════════════════════════════════════════════
# GRID
# ═══════════════════════════════════════════════════════════════════════════════

def build_grid(domains, methods, kappas, lams, seeds, mode="cross"):
    """mode='cross' -> full kappa x lam grid.
       mode='cross-diag' -> two 1-D sweeps through (kappa*, lam*)."""
    pairs = []
    if mode == "cross":
        pairs = [(k, l) for k in kappas for l in lams]
    else:
        k_star, l_star = kappas[-1], lams[-1]
        pairs = ([(k, l_star) for k in kappas]
                 + [(k_star, l) for l in lams if l != l_star])
    runs = []
    for (dom, tsk) in domains:
        for m in methods:
            for (k, l) in pairs:
                for s in seeds:
                    runs.append(dict(domain=dom, task=tsk, method=m,
                                     kappa=k, lam=l, seed=s))
    return runs


def _worker(payload):
    kwargs, common = payload
    try:
        return run_point(**kwargs, **common)
    except Exception as e:  # keep the sweep alive
        logger.error(f"FAILED {kwargs}: {type(e).__name__}: {e}")
        return {**kwargs, "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--single", action="store_true")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--sanity", action="store_true")

    p.add_argument("--domain", type=str, default="walker")
    p.add_argument("--task", type=str, default="walk")
    p.add_argument("--domains", type=str, default="walker:walk",
                   help="comma-separated domain:task pairs")
    p.add_argument("--method", type=str, default="ibd", choices=METHODS)
    p.add_argument("--methods", type=str,
                   default="ibd,inverse_dyn,mutual_info,variance,grad_attr")
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--kappas", type=str, default=None)
    p.add_argument("--lams", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--grid_mode", type=str, default="cross",
                   choices=["cross", "cross-diag"])

    p.add_argument("--base_config", type=str, default="medium")
    p.add_argument("--n_confounded", type=int, default=12)
    p.add_argument("--baseline_episodes", type=int, default=200)
    p.add_argument("--cheap_episodes", type=int, default=50)
    p.add_argument("--n_baseline", type=int, default=80)
    p.add_argument("--n_intervention", type=int, default=80)
    p.add_argument("--traj_length", type=int, default=200)

    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--results_dir", type=str, default="results/confounding")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    common = dict(base_config=args.base_config,
                  n_confounded=args.n_confounded,
                  n_baseline=args.n_baseline,
                  n_intervention=args.n_intervention,
                  traj_length=args.traj_length,
                  baseline_episodes=args.baseline_episodes,
                  cheap_episodes=args.cheap_episodes,
                  results_dir=args.results_dir,
                  overwrite=args.overwrite)

    if args.sanity:
        sanity_check(args.domain, args.task, kappa=args.kappa, lam=args.lam,
                     n_confounded=args.n_confounded,
                     base_config=args.base_config, seed=args.seed)
        return

    if args.single:
        run_point(args.domain, args.task, args.method,
                  args.kappa, args.lam, args.seed, **common)
        return

    if args.grid:
        domains = [tuple(d.split(":")) for d in args.domains.split(",")]
        methods = args.methods.split(",")
        kappas = ([float(x) for x in args.kappas.split(",")]
                  if args.kappas else DEFAULT_KAPPAS)
        lams = ([float(x) for x in args.lams.split(",")]
                if args.lams else DEFAULT_LAMS)
        seeds = list(range(args.seeds))
        runs = build_grid(domains, methods, kappas, lams, seeds,
                          mode=args.grid_mode)
        logger.info(f"Grid: {len(runs)} runs "
                    f"({len(methods)} methods x {len(kappas)}x{len(lams)} "
                    f"strengths x {len(seeds)} seeds)")

        if args.parallel > 1:
            import multiprocessing as mp
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            with mp.get_context("spawn").Pool(args.parallel) as pool:
                pool.map(_worker, [(r, common) for r in runs])
        else:
            for r in runs:
                _worker((r, common))
        logger.info("Grid complete.")
        return

    p.print_help()


if __name__ == "__main__":
    main()
