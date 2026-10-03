#!/usr/bin/env python3
"""
Dynamics-Misspecification Study: probing in a lossy simulator
==============================================================

Question: to what extent is the method useful when the simulation is a
lossy approximation of the causal dynamics of the real-world environment?

The (kappa, lambda) confounding grid cannot answer this: both knobs move
the *confounding strength* inside a fixed generative model, never the
model itself.  Here the model itself is wrong.

Protocol
--------
The probe never sees the real plant.  For a misspecification level
``delta`` we scale the MuJoCo model parameters

    body_mass, body_inertia, dof_damping, geom_friction[:, 0]  *=  (1 + delta)

and then do *everything* in that wrong simulator: train the scout there,
collect both probe branches there, run the tests there.  The resulting
mask is scored against the ground-truth boundary, which is *unchanged* by
the perturbation -- scaling masses and damping does not alter which
observation dimensions the action can reach, so `true_dims` at
delta = +-0.5 is exactly `true_dims` at delta = 0 and the F1 column is
directly comparable across rows.  Two things are reported:

  * F1 / precision / recall of the wrong-sim mask against ground truth
  * Jaccard overlap and symmetric difference against the *nominal*
    (delta = 0) mask at the same seed -- i.e. "would deploying the
    wrong-sim mask on the real plant change anything?"  When the masks
    coincide exactly the downstream return is unchanged by construction
    and no RL training is needed to say so.

``dynamics_shift`` is recorded as a sanity covariate: the same action
sequence is replayed from the same initial state in the nominal and the
perturbed model, and the divergence of the true observation dims is
reported in units of the nominal per-dim std.  A misspecification that
does not move this number is not a misspecification and should be
disregarded.

Everything else matches the main-results protocol in ``run_dmcontrol.py``
(80k-step SAC scout, 80/80 trajectories of length 200, horizons [1, 5, 10],
Welch + BH), so the delta = 0 row is comparable to the paper's Table 1.

Usage:
    # One task, full delta sweep, 3 seeds (~7 min per probe)
    python -m experiments.run_misspec --domain walker --task walk

    # Smoke test
    python -m experiments.run_misspec --quick

    # Aggregate what is on disk
    python -m experiments.run_misspec --aggregate

Output:
    results/misspec/per_run/{domain}_{task}_{dist}_d{delta}_s{seed}.json
    results/misspec/summary.md
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
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from experiments.dmcontrol_distractors import (      # noqa: E402
    DistractingDMControlEnv)

DEFAULT_DELTAS = [-0.5, -0.2, 0.0, 0.2, 0.5]
DEFAULT_SEEDS = [42, 142, 242]
DEFAULT_TASKS = [("walker", "walk"), ("cheetah", "run"), ("reacher", "hard")]

# Model fields scaled by (1 + delta).  Mass and inertia move together so
# the perturbed model stays a physically coherent rigid body rather than
# an object with mass and inertia tensor that disagree.
PARAM_GROUPS = {
    "mass": ("body_mass", "body_inertia"),
    "damping": ("dof_damping",),
    "friction": ("geom_friction",),      # sliding component only
}


# ═══════════════════════════════════════════════════════════════════════════════
# MISSPECIFIED ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class MisspecifiedDMControlEnv(DistractingDMControlEnv):
    """``DistractingDMControlEnv`` with MuJoCo model parameters scaled.

    The perturbation is applied to ``mjModel`` once at construction.
    ``mjModel`` is untouched by ``physics.reset()``, so it persists for
    the lifetime of the env.  Observation layout, distractor processes
    and ground-truth labels are inherited unchanged.
    """

    def __init__(self, *args, delta: float = 0.0,
                 groups: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = float(delta)
        self.groups = list(groups) if groups else list(PARAM_GROUPS)
        self.perturbation_info = self._perturb()

    def _perturb(self) -> Dict:
        """Scale the selected model fields by (1 + delta)."""
        model = self._dm_env.physics.model
        k = 1.0 + self.delta
        info: Dict[str, Dict] = {"delta": self.delta, "scale": k,
                                 "fields": {}}

        for g in self.groups:
            for field in PARAM_GROUPS[g]:
                arr = getattr(model, field)
                if field == "geom_friction":
                    view = arr[:, 0]        # sliding friction only
                else:
                    view = arr
                before = np.array(view, dtype=np.float64, copy=True)
                if self.delta != 0.0:
                    view *= k
                after = np.array(view, dtype=np.float64, copy=True)
                info["fields"][field] = {
                    "group": g,
                    "n": int(before.size),
                    # A field that is all zeros cannot be perturbed by
                    # scaling; record it so a silently inert knob is
                    # visible in the results rather than assumed to bite.
                    "n_nonzero": int(np.count_nonzero(before)),
                    "sum_before": float(before.sum()),
                    "sum_after": float(after.sum()),
                }

        # Recompute derived constants (subtree masses, etc.).
        if self.delta != 0.0:
            try:
                import mujoco
                mujoco.mj_setConst(self._dm_env.physics.model.ptr,
                                   self._dm_env.physics.data.ptr)
                info["mj_setConst"] = True
            except Exception as e:                      # pragma: no cover
                info["mj_setConst"] = f"failed: {e}"
                logger.warning(f"mj_setConst failed: {e}")

        return info


def make_misspec_env(domain: str, task: str, distractors="medium",
                     delta: float = 0.0, seed: int = 42,
                     groups: Optional[List[str]] = None,
                     ) -> MisspecifiedDMControlEnv:
    return MisspecifiedDMControlEnv(
        domain_name=domain, task_name=task,
        distractor_config=distractors, seed=seed,
        delta=delta, groups=groups)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: does the perturbation actually move the dynamics?
# ═══════════════════════════════════════════════════════════════════════════════

def measure_dynamics_shift(domain: str, task: str, distractors,
                           delta: float, seed: int,
                           n_steps: int = 400,
                           groups: Optional[List[str]] = None) -> Dict:
    """Replay one action sequence in the nominal and perturbed models.

    Both envs are constructed with the same seed and reset once, so they
    start from the same state and receive identical actions; any
    divergence of the true observation dims is attributable to the model
    perturbation alone.  Reported in units of the nominal per-dim std so
    it is comparable across tasks.
    """
    env_nom = make_misspec_env(domain, task, distractors, 0.0, seed, groups)
    env_pert = make_misspec_env(domain, task, distractors, delta, seed, groups)

    n_true = env_nom.true_obs_dim
    rng = np.random.RandomState(seed + 99)
    lo, hi = env_nom.action_space.low, env_nom.action_space.high

    o_nom, _ = env_nom.reset(seed=seed)
    o_pert, _ = env_pert.reset(seed=seed)

    traj_nom, traj_pert = [o_nom[:n_true]], [o_pert[:n_true]]
    for _ in range(n_steps):
        a = rng.uniform(lo, hi)
        o_nom, _, d_n, _, _ = env_nom.step(a)
        o_pert, _, d_p, _, _ = env_pert.step(a)
        traj_nom.append(o_nom[:n_true])
        traj_pert.append(o_pert[:n_true])
        if d_n or d_p:      # termination times may differ; stop at first
            break

    A = np.asarray(traj_nom, dtype=np.float64)
    B = np.asarray(traj_pert, dtype=np.float64)
    std = A.std(axis=0)
    std[std < 1e-8] = 1.0
    normalised = np.abs(A - B) / std

    env_nom.close()
    env_pert.close()
    return {
        "n_steps_compared": int(A.shape[0]),
        "mean_abs_diff": float(np.abs(A - B).mean()),
        "mean_normalised_diff": float(normalised.mean()),
        "max_normalised_diff": float(normalised.max()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════════

def _score(soi: set, env) -> Dict:
    true_soi = env.true_dims
    tp = len(soi & true_soi)
    fp = len(soi - true_soi)
    fn = len(true_soi - soi)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_selected": len(soi)}


def run_point(domain: str, task: str, delta: float, seed: int,
              distractors: str = "medium",
              scout_steps: int = 80_000,
              n_baseline: int = 80, n_intervention: int = 80,
              traj_length: int = 200,
              horizons: Optional[List[int]] = None,
              n_permutations: int = 5000,
              device: str = "cpu",
              torch_threads: int = 2,
              shift_steps: int = 400,
              groups: Optional[List[str]] = None,
              results_dir: str = "results/misspec",
              overwrite: bool = False) -> Dict:
    """Train scout + probe entirely inside the (mis)specified model."""
    out_dir = Path(results_dir) / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{domain}_{task}_{distractors}_d{delta:g}_s{seed}.json"
    path = out_dir / fname
    if path.exists() and not overwrite:
        logger.info(f"SKIP (exists): {fname}")
        with open(path) as f:
            return json.load(f)

    import torch
    from stable_baselines3 import SAC
    from ibd.probe import IBDProbe

    torch.set_num_threads(torch_threads)
    horizons = horizons or [1, 5, 10]
    t0 = time.time()

    # Same seed offsets as run_dmcontrol.run_single
    scout_env = make_misspec_env(domain, task, distractors,
                                 delta, seed + 3000, groups)
    probe_env = make_misspec_env(domain, task, distractors,
                                 delta, seed + 2000, groups)

    obs_dim = probe_env.observation_space.shape[0]
    logger.info(f"{domain}_{task} | {distractors} | delta={delta:+.2f} "
                f"| seed={seed} | obs={obs_dim} "
                f"true={probe_env.true_obs_dim}")

    # Does the perturbation bite?
    shift = (measure_dynamics_shift(domain, task, distractors, delta,
                                    seed, shift_steps, groups)
             if delta != 0.0 else
             {"n_steps_compared": 0, "mean_abs_diff": 0.0,
              "mean_normalised_diff": 0.0, "max_normalised_diff": 0.0})
    logger.info(f"  dynamics shift (norm.): "
                f"mean={shift['mean_normalised_diff']:.3f} "
                f"max={shift['max_normalised_diff']:.2f}")

    # Phase 1: scout, trained in the wrong model
    scout = SAC("MlpPolicy", scout_env, learning_rate=3e-4,
                batch_size=256, buffer_size=scout_steps,
                seed=seed, verbose=0, device=device)
    scout.learn(total_timesteps=scout_steps)

    def scout_policy(obs):
        with torch.no_grad():
            action, _ = scout.predict(obs, deterministic=False)
        return action

    # Phase 2: probe, in the wrong model
    probe = IBDProbe(probe_env, n_baseline=n_baseline,
                     n_intervention=n_intervention,
                     traj_length=traj_length, horizons=horizons,
                     n_permutations=n_permutations, seed=seed)
    soi, info = probe.discover_joint(policy=scout_policy)

    result = {
        "domain": domain, "task": task, "distractors": distractors,
        "delta": delta, "seed": seed,
        "obs_dim": obs_dim, "n_true": probe_env.true_obs_dim,
        **_score(soi, probe_env),
        "selected": sorted(soi),
        "dynamics_shift": shift,
        "perturbation": probe_env.perturbation_info,
        "scout_steps": scout_steps,
        "wall_s": time.time() - t0,
        "info": {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else str(v))
                 for k, v in info.items()
                 if k not in ("soi", "p_values", "effect_sizes")},
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"  -> P={result['precision']:.3f} R={result['recall']:.3f} "
                f"F1={result['f1']:.3f} | selected={result['n_selected']} "
                f"({result['wall_s']:.0f}s)")

    for e in (scout_env, probe_env):
        e.close()
    del scout
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _ms(vals) -> str:
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def aggregate(results_dir: str = "results/misspec") -> str:
    out_dir = Path(results_dir)
    runs = []
    for p in sorted((out_dir / "per_run").glob("*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        return "No runs found.\n"

    tasks = sorted({(r["domain"], r["task"], r["distractors"])
                    for r in runs})
    deltas = sorted({r["delta"] for r in runs})

    lines = ["# Dynamics-Misspecification Study", ""]
    lines.append("The scout is trained and the probe is run entirely "
                 "inside a model whose `body_mass`, `body_inertia`, "
                 "`dof_damping` and sliding `geom_friction` are scaled by "
                 "`1 + delta`.  Ground truth is unchanged by the "
                 "perturbation, so F1 is comparable down each column.")
    lines.append("")
    lines.append("`Jaccard` and `symm.diff` compare the wrong-sim mask "
                 "against the nominal (`delta=0`) mask at the same seed: "
                 "an exact match means deploying the wrong-sim mask on the "
                 "real plant changes nothing downstream.")
    lines.append("")

    for (domain, task, dist) in tasks:
        sub = [r for r in runs
               if (r["domain"], r["task"], r["distractors"])
               == (domain, task, dist)]
        nominal = {r["seed"]: set(r["selected"])
                   for r in sub if r["delta"] == 0.0}
        n_seeds = len({r["seed"] for r in sub})

        lines.append(f"## {domain}_{task} — {dist} distractors "
                     f"({n_seeds} seeds)")
        lines.append("")
        # |mask| is shown so that a degenerate all-empty run cannot hide
        # behind Jaccard = 1.0 (two empty sets are trivially identical).
        lines.append("| delta | dyn. shift | |mask| | precision | recall "
                     "| F1 | Jaccard vs nominal | symm.diff | exact match |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for d in deltas:
            rs = [r for r in sub if abs(r["delta"] - d) < 1e-9]
            if not rs:
                continue
            jac, sym, exact = [], [], []
            for r in rs:
                nom = nominal.get(r["seed"])
                if nom is None:
                    continue
                sel = set(r["selected"])
                union = nom | sel
                jac.append(len(nom & sel) / len(union) if union else 1.0)
                sym.append(len(nom ^ sel))
                exact.append(1.0 if nom == sel else 0.0)
            shift = np.mean([r["dynamics_shift"]["mean_normalised_diff"]
                             for r in rs])
            lines.append(
                f"| {d:+g} | {shift:.3f} "
                f"| {np.mean([r['n_selected'] for r in rs]):.1f} "
                f"| {_ms([r['precision'] for r in rs])} "
                f"| {_ms([r['recall'] for r in rs])} "
                f"| {_ms([r['f1'] for r in rs])} "
                f"| {(_ms(jac) if jac else 'n/a')} "
                f"| {(f'{np.mean(sym):.1f}' if sym else 'n/a')} "
                f"| {(f'{np.mean(exact) * 100:.0f}%' if exact else 'n/a')} |")
        lines.append("")

    report = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report)
    logger.info(f"Saved: {out_dir / 'summary.md'}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="IBD dynamics-misspecification (lossy simulator) study")
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--distractors", type=str, default="medium")
    p.add_argument("--deltas", type=str, default=None,
                   help="comma-separated model scale offsets, e.g. -0.5,0,0.5")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--groups", type=str, default=None,
                   help=f"comma-separated subset of {list(PARAM_GROUPS)}")
    p.add_argument("--scout_steps", type=int, default=80_000)
    p.add_argument("--n_baseline", type=int, default=80)
    p.add_argument("--n_intervention", type=int, default=80)
    p.add_argument("--traj_length", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--results_dir", type=str, default="results/misspec")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--no_aggregate", action="store_true",
                   help="skip the trailing aggregate — use when several "
                        "invocations run concurrently and would race on "
                        "summary.md")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny scout + tiny probe budget")
    args = p.parse_args()

    if args.aggregate:
        print(aggregate(args.results_dir))
        return

    deltas = ([float(x) for x in args.deltas.split(",")]
              if args.deltas else list(DEFAULT_DELTAS))
    seeds = ([int(x) for x in args.seeds.split(",")]
             if args.seeds else list(DEFAULT_SEEDS))
    groups = args.groups.split(",") if args.groups else None
    tasks = ([(args.domain, args.task)] if args.domain and args.task
             else list(DEFAULT_TASKS))

    kw = dict(distractors=args.distractors,
              scout_steps=args.scout_steps,
              n_baseline=args.n_baseline,
              n_intervention=args.n_intervention,
              traj_length=args.traj_length,
              device=args.device,
              torch_threads=args.torch_threads,
              groups=groups,
              results_dir=args.results_dir,
              overwrite=args.overwrite)

    if args.quick:
        # >= 10 trajectories per branch: ``discover_joint`` yields one
        # sample per trajectory and skips any test with fewer than 10.
        deltas, seeds, tasks = [0.0, 0.5], [42], [("walker", "walk")]
        kw.update(scout_steps=1_000, n_baseline=16, n_intervention=16,
                  traj_length=60, shift_steps=50,
                  results_dir=args.results_dir + "_smoke")

    logger.info("=" * 60)
    logger.info("IBD DYNAMICS-MISSPECIFICATION STUDY")
    logger.info(f"  Tasks:  {tasks}")
    logger.info(f"  Deltas: {deltas}")
    logger.info(f"  Seeds:  {seeds}")
    logger.info(f"  Groups: {groups or list(PARAM_GROUPS)}")
    logger.info(f"  Probes: {len(tasks) * len(deltas) * len(seeds)}")
    logger.info("=" * 60)

    for domain, task in tasks:
        for delta in deltas:
            for seed in seeds:
                run_point(domain, task, delta, seed, **kw)

    if not args.no_aggregate:
        print(aggregate(kw["results_dir"]))


if __name__ == "__main__":
    main()
