#!/usr/bin/env python3
"""
Constructed level-shift dimensions: a positive control for the level statistic.

A level-shift dimension is action-reachable ONLY through its mean level:

    m_t = (1-a) m_{t-1} + a * (action_j,t)^2        slow EMA, a = 0.02
    x_t = g * m_t + eps_t,   eps_t ~ N(0, sigma^2)  iid

E[a^2] = 1/3 under Uniform[-1,1] but is smaller under the probe policy, so
the trajectory MEAN of x differs between branches.  The EMA is slow and
sigma dominates, so the step-to-step displacement |x_{t+h} - x_t| is
essentially unchanged between branches.  This is exactly the blind spot of
the displacement statistic: an effect that shifts the marginal without
changing step-to-step variability.

Ground truth SoI = true dm_control dims + level-shift dims.
Plain exogenous distractors remain outside the SoI.
"""
from __future__ import annotations
import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse, json, logging
import numpy as np
from gymnasium import spaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from experiments.dmcontrol_distractors import DistractingDMControlEnv  # noqa
from experiments.run_summary_stat import score                        # noqa

HORIZONS = [1, 5, 10]


class LevelShiftSource:
    def __init__(self, dim, action_dim, gain=10.0, sigma=0.5,
                 ema=0.02, seed=0):
        self.dim, self.gain, self.sigma, self.ema = dim, gain, sigma, ema
        self.rng = np.random.RandomState(seed)
        self.amap = self.rng.randint(0, action_dim, dim)
        self._m = np.zeros(dim)

    def reset(self):
        self._m = np.zeros(self.dim)
        return self.state

    def step(self, action):
        a = np.asarray(action).flatten()
        for i in range(self.dim):
            j = self.amap[i] if self.amap[i] < len(a) else 0
            self._m[i] = (1 - self.ema) * self._m[i] + self.ema * a[j] ** 2

    @property
    def state(self):
        return self.gain * self._m + self.rng.normal(0, self.sigma, self.dim)


class LevelShiftEnv(DistractingDMControlEnv):
    def __init__(self, *a, n_level=6, **kw):
        super().__init__(*a, **kw)
        self._ls = LevelShiftSource(n_level, self._action_dim,
                                    seed=kw.get("seed", 0) + 9000)
        d0 = self.observation_space.shape[0]
        self.level_dims = set(range(d0, d0 + n_level))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(d0 + n_level,), dtype=np.float32)
        # level-shift dims ARE action-reachable
        self.soi_dims = set(self.true_dims) | self.level_dims

    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        self._ls.reset()
        return np.concatenate([o, self._ls.state.astype(np.float32)]), i

    def step(self, action):
        o, r, te, tr, i = super().step(action)
        self._ls.step(action)
        return np.concatenate([o, self._ls.state.astype(np.float32)]), r, te, tr, i


def run(domain, task, distractors, seed, n_level=6, n_traj=80, traj_len=200,
        results_dir="results/levelshift", overwrite=False):
    out = Path(results_dir) / "per_run"; out.mkdir(parents=True, exist_ok=True)
    p = out / f"{domain}_{task}_{distractors}_s{seed}.json"
    if p.exists() and not overwrite:
        return json.load(open(p))

    from ibd.probe import IBDProbe
    from experiments.confounded_distractors import StructuredProbePolicy

    env = LevelShiftEnv(domain_name=domain, task_name=task,
                        distractor_config=distractors, seed=seed + 2000,
                        n_level=n_level)
    soi, lvl = env.soi_dims, env.level_dims
    obs_dim = env.observation_space.shape[0]
    pol = StructuredProbePolicy(env, seed=seed)
    probe = IBDProbe(env, n_baseline=n_traj, n_intervention=n_traj,
                     traj_length=traj_len, horizons=HORIZONS, seed=seed)
    base = probe._collect(pol, n_traj, intervention=None)
    itv = probe._collect(pol, n_traj, intervention={"dim": "all", "mode": "randomize"})

    variants = {}
    for v in ("delta", "level", "both"):
        r = score(base, itv, obs_dim, v)
        s = r["soi"]
        tp, fp, fn = len(s & soi), len(s - soi), len(soi - s)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        variants[v] = {
            "precision": pr, "recall": rc,
            "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "level_recall": len(s & lvl) / len(lvl),
            "level_detected": sorted(s & lvl), "n_tests": r["n_tests"],
        }
        logger.info(f"  {domain}_{task} s{seed} {v:<6} "
                    f"level_recall={variants[v]['level_recall']:.2f} "
                    f"({len(s&lvl)}/{len(lvl)})  P={pr:.3f} R={rc:.3f}")
    res = {"domain": domain, "task": task, "distractors": distractors,
           "seed": seed, "n_level": n_level, "obs_dim": obs_dim,
           "variants": variants}
    json.dump(res, open(p, "w"), indent=2, default=str)
    env.close()
    return res


def aggregate(results_dir="results/levelshift"):
    out = Path(results_dir)
    runs = [json.load(open(x)) for x in sorted((out / "per_run").glob("*.json"))]
    if not runs:
        return "no runs\n"
    import collections
    by = collections.defaultdict(list)
    for r in runs:
        by[(r["domain"], r["task"], r["distractors"])].append(r)
    L = ["# Constructed level-shift dimensions: statistic comparison", "",
         "Level-shift dimensions are action-reachable only through their mean level; their step-to-step displacement distribution is unchanged.",
         "`level_recall` = detection rate on these constructed dimensions.", ""]
    for k in sorted(by):
        rs = by[k]
        L += [f"## {k[0]}_{k[1]} — {k[2]} ({len(rs)} seeds, "
              f"{rs[0]['n_level']} level-shift dims)", "",
              "| statistic | level-dim recall | overall precision | overall recall | F1 |",
              "|---|---|---|---|---|"]
        for v in ("delta", "level", "both"):
            g = lambda x: (f"{np.mean([r['variants'][v][x] for r in rs]):.3f}"
                           f"±{np.std([r['variants'][v][x] for r in rs]):.3f}")
            L.append(f"| {v} | **{g('level_recall')}** | {g('precision')} "
                     f"| {g('recall')} | {g('f1')} |")
        L.append("")
    rep = "\n".join(L) + "\n"
    (out / "summary.md").write_text(rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="walker"); ap.add_argument("--task", default="walk")
    ap.add_argument("--distractors", default="medium")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_level", type=int, default=6)
    ap.add_argument("--results_dir", default="results/levelshift")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        print(aggregate(a.results_dir)); sys.exit()
    for s in [int(x) for x in a.seeds.split(",")]:
        run(a.domain, a.task, a.distractors, s, a.n_level,
            results_dir=a.results_dir, overwrite=a.overwrite)
    print(aggregate(a.results_dir))
