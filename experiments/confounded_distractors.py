"""
Confounded Distractor Regime
=============================

The main benchmark's distractors are appended *exogenous* processes with
no edges from action or state, so ``do(a)`` leaves them invariant almost
by construction.

This module adds a regime where a genuine latent confounder enters the
**action channel**, so that observational selectors are fooled by a real
backdoor path rather than by finite-sample noise.

Data-generating process
~~~~~~~~~~~~~~~~~~~~~~~~
Latent confounder (never observed by any method)::

    C_t = rho_c * C_{t-1} + sqrt(1 - rho_c^2) * xi_t,    xi ~ N(0, I_{n_c})

so ``C`` is a discrete-time OU / AR(1) process with unit stationary
variance.

Behaviour (probe) policy — the operator is also affected by the ambient
condition::

    a_t = clip( pi_probe(o_t) + kappa * K C_t )

Confounded distractors — driven by the same latent condition::

    d_t = rho * d_{t-1} + lam * L C_t + eps_t          (ground truth: NOT causal)

Causal graph::

           C
          / \\
      kappa   lam
        /       \\
       a --------> s        d
       ^                    (no edge a -> d, no edge s -> d)
       |
    pi_probe(o)

Why this is the hard case
~~~~~~~~~~~~~~~~~~~~~~~~~
* ``I(d_{t+1}; a_t) > 0``            -> MI selector fooled
* ``R^2(s,a -> d') > R^2(s -> d')``  -> conditional-MI / forward-model fooled
  (``a_t`` reveals ``C_t``, which ``s_t`` does not fully determine)
* ``d(d')/d(a) != 0`` in a learned model -> gradient attribution fooled
* ``a_t`` is predictable from ``(s_t, s_{t+k})`` including ``d`` dims
  -> multistep inverse dynamics fooled
* Under ``do(a ~ Unif)`` the edge ``C -> a`` is severed while ``d``'s law is
  unchanged -> IBD's two-sample test sees no shift -> correctly excluded.

Oracle validity is preserved: ``C`` has **no direct edge into the physics
state**, it only acts through the action channel, so the distractor dims
still carry no control-relevant information beyond ``s`` and
``oracle >= full_state`` still holds.

Usage::

    from experiments.confounded_distractors import (
        make_confounded_env, StructuredProbePolicy, ConfoundedPolicy)

    env = make_confounded_env("walker", "walk", base_config="medium",
                              n_confounded=12, lam=1.0, seed=0)
    pi = ConfoundedPolicy(env, StructuredProbePolicy(env, seed=0),
                          kappa=0.5, seed=0)
    obs, _ = env.reset(seed=0)
    obs, r, term, trunc, _ = env.step(pi(obs))
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import List, Optional

from experiments.dmcontrol_distractors import (
    DistractingDMControlEnv, DistractorSource, _build_distractors,
    _build_distractors_custom)


# ═══════════════════════════════════════════════════════════════════════════════
# LATENT CONFOUNDER
# ═══════════════════════════════════════════════════════════════════════════════

class LatentOUConfounder:
    """Unobserved AR(1) / OU process with unit stationary variance.

    Never appears in the observation.  Read by the behaviour policy
    (simulating an operator affected by ambient conditions) and by the
    confounded distractor sources.
    """

    def __init__(self, n_c: int = 4, rho_c: float = 0.95, seed: int = 42):
        self.n_c = n_c
        self.rho_c = rho_c
        self.rng = np.random.RandomState(seed)
        self._c = np.zeros(n_c)

    def reset(self) -> np.ndarray:
        # Draw from the stationary distribution (unit variance)
        self._c = self.rng.normal(0.0, 1.0, self.n_c)
        return self._c.copy()

    def step(self, dt: float = 0.01) -> np.ndarray:
        innov = self.rng.normal(0.0, 1.0, self.n_c)
        self._c = (self.rho_c * self._c
                   + np.sqrt(max(1.0 - self.rho_c ** 2, 1e-12)) * innov)
        return self._c.copy()

    @property
    def state(self) -> np.ndarray:
        return self._c.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFOUNDED DISTRACTOR SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

class ConfoundedDistractor(DistractorSource):
    """AR(1) distractor driven by the shared latent confounder.

        d_t = rho * d_{t-1} + lam * L C_t + eps_t

    The output is rescaled so that the *stationary standard deviation*
    equals ``ref_scale`` regardless of ``lam``.  This is essential: without
    it, sweeping ``lam`` would also sweep the marginal variance of ``d``,
    and a variance-based selector would separate the regimes for the wrong
    reason.  The scale factor is computed once at construction by
    simulating the recursion, so it is exact up to Monte-Carlo error.

    Args:
        dim:        number of distractor dimensions from this source
        confounder: the shared ``LatentOUConfounder`` instance
        lam:        confounder loading (0 = plain exogenous AR(1))
        rho:        AR(1) persistence of the distractor itself
        ref_scale:  target stationary std (calibrated to true obs dims)
    """

    def __init__(self, dim: int, confounder: LatentOUConfounder,
                 lam: float = 1.0, rho: float = 0.9,
                 ref_scale: float = 1.0, seed: int = 42):
        super().__init__(dim, seed)
        self.confounder = confounder
        self.lam = float(lam)
        self.rho = float(rho)
        self.ref_scale = float(ref_scale)

        # Loading matrix C -> d  (rows unit-norm so `lam` is the only knob)
        L = self.rng.normal(0.0, 1.0, (dim, confounder.n_c))
        L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-12
        self.L = L

        # Innovation std held FIXED across lam; the post-hoc rescale below
        # restores a constant marginal variance.
        self.eps_std = 1.0

        self._scale = self._calibrate_scale()

    # ── calibration ───────────────────────────────────────────────────

    def _calibrate_scale(self, n_burn: int = 2000, n_sim: int = 20000) -> float:
        """Empirical stationary std of the raw recursion -> scale factor."""
        rng = np.random.RandomState(12345)
        c = rng.normal(0.0, 1.0, self.confounder.n_c)
        rc = self.confounder.rho_c
        s_c = np.sqrt(max(1.0 - rc ** 2, 1e-12))
        d = np.zeros(self.dim)
        acc = []
        for t in range(n_burn + n_sim):
            c = rc * c + s_c * rng.normal(0.0, 1.0, self.confounder.n_c)
            d = (self.rho * d + self.lam * (self.L @ c)
                 + self.eps_std * rng.normal(0.0, 1.0, self.dim))
            if t >= n_burn:
                acc.append(d.copy())
        sd = np.std(np.asarray(acc), axis=0).mean()
        return float(self.ref_scale / max(sd, 1e-8))

    # ── DistractorSource interface ────────────────────────────────────

    def reset(self):
        self._raw = self.rng.normal(0.0, 1.0, self.dim)
        self._state = self._raw * self._scale
        return self._state.copy()

    def step(self, dt: float = 0.01) -> np.ndarray:
        c = self.confounder.state
        eps = self.rng.normal(0.0, self.eps_std, self.dim)
        self._raw = self.rho * self._raw + self.lam * (self.L @ c) + eps
        self._state = self._raw * self._scale
        return self._state.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ConfoundedDMControlEnv(DistractingDMControlEnv):
    """DMControl + exogenous distractors + **confounded** distractors.

    Adds ``n_confounded`` extra observation dims driven by a latent OU
    process that also biases the behaviour policy (via
    :class:`ConfoundedPolicy`, applied outside the env so that the
    do-operator can override it).

    The confounder is exposed as ``env.confounder_state`` — the behaviour
    policy reads it, no method under evaluation ever does.

    Timing convention (contemporaneous confounding)::

        step(a_t):   C_{t}   -> C_{t+1}
                     physics(a_t)
                     d_{t+1} = rho d_t + lam L C_{t+1} + eps
        obs_{t+1} contains d_{t+1};  the policy then reads C_{t+1}
        to build a_{t+1}.  So a_{t+1} and d_{t+1} share C_{t+1}.
    """

    def __init__(self, domain_name: str = "walker",
                 task_name: str = "walk",
                 base_config: str = "medium",
                 n_confounded: int = 12,
                 lam: float = 1.0,
                 rho_d: float = 0.9,
                 rho_c: float = 0.95,
                 n_c: int = 4,
                 seed: int = 42,
                 time_limit: Optional[float] = None):
        # Build the standard exogenous distractor bank first
        super().__init__(domain_name=domain_name, task_name=task_name,
                         distractor_config=base_config, seed=seed,
                         time_limit=time_limit)

        self.lam = float(lam)
        self.n_confounded = int(n_confounded)
        self._base_config = base_config

        # ── keep the TOTAL distractor budget identical to `base_config` ──
        # The confounded dims are *carved out of* the exogenous bank, not
        # appended to it, so the observation dimensionality (and hence the
        # (kappa=0, lambda=0) control cell) stays directly comparable to the
        # main benchmark table for the same `base_config`.
        total_distractors = self._distractor_dim          # e.g. medium -> 50
        if self.n_confounded >= total_distractors:
            raise ValueError(
                f"n_confounded={self.n_confounded} must be < the "
                f"'{base_config}' budget of {total_distractors} distractors")
        n_exogenous = total_distractors - self.n_confounded

        # Latent confounder
        self._confounder = LatentOUConfounder(n_c=n_c, rho_c=rho_c,
                                              seed=seed + 777)

        # Exogenous bank, rebuilt at the reduced count with the same
        # composition ratios used by the distractor-scaling experiment.
        exo_sources = _build_distractors_custom(
            n_exogenous, self._true_obs_dim, seed)

        # Confounded distractor sources (same scale calibration as the
        # mimicking bank, so variance selection cannot separate them)
        base_scale = 0.5 + 0.05 * self._true_obs_dim
        scales = [0.5, 0.8, 1.0, 1.2]
        conf_sources: List[DistractorSource] = []
        remaining, si, s = self.n_confounded, 0, seed + 900
        while remaining > 0:
            chunk = min(6, remaining)
            conf_sources.append(ConfoundedDistractor(
                dim=chunk, confounder=self._confounder, lam=self.lam,
                rho=rho_d, ref_scale=base_scale * scales[si % len(scales)],
                seed=s))
            s += 1
            si += 1
            remaining -= chunk

        # Confounded dims occupy the tail of the observation vector
        exo_dim = n_exogenous
        self._distractors = exo_sources + conf_sources
        self._distractor_dim = total_distractors
        assert sum(d.dim for d in self._distractors) == total_distractors

        total_dim = self._true_obs_dim + self._distractor_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        # Ground truth: true dims causal, ALL distractors (exogenous and
        # confounded alike) non-causal
        self.true_dims = set(range(self._true_obs_dim))
        self.distractor_dims = set(range(self._true_obs_dim, total_dim))
        self.confounded_dims = set(
            range(self._true_obs_dim + exo_dim, total_dim))
        self.exogenous_dims = set(
            range(self._true_obs_dim, self._true_obs_dim + exo_dim))

    # ── confounder access (for the behaviour policy only) ─────────────

    @property
    def confounder_state(self) -> np.ndarray:
        return self._confounder.state

    @property
    def n_c(self) -> int:
        return self._confounder.n_c

    # ── gymnasium interface ───────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._confounder.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        # Advance C first so that d_{t+1} and the next action share C_{t+1}
        self._confounder.step(self._dt)
        return super().step(action)

    @property
    def name(self):
        return (f"{self._domain}_{self._task}_conf"
                f"{self.n_confounded}_lam{self.lam:g}")

    def __repr__(self):
        return (f"ConfoundedDMControlEnv(domain={self._domain}, "
                f"task={self._task}, base={self._base_config}, "
                f"true={self._true_obs_dim}, exo={len(self.exogenous_dims)}, "
                f"confounded={self.n_confounded}, lam={self.lam:g}, "
                f"total={self.observation_space.shape[0]})")


def make_confounded_env(domain: str, task: str, **kwargs
                        ) -> ConfoundedDMControlEnv:
    return ConfoundedDMControlEnv(domain_name=domain, task_name=task, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOUR POLICIES
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredProbePolicy:
    """``pi_probe``: sinusoid + weak state feedback + exploration noise.

    Identical in form to ``IBDProbe._random_policy`` so that every method
    in the sweep collects data under the *same* behaviour policy.  Kept as
    a standalone object (rather than reused from the probe) so baselines
    and IBD share one instance-independent definition.
    """

    def __init__(self, env: gym.Env, seed: int = 42):
        self.action_dim = int(np.prod(env.action_space.shape))
        self.low = env.action_space.low.flatten()
        self.high = env.action_space.high.flatten()
        self.rng = np.random.RandomState(seed)
        self._t = 0

    def reset_time(self):
        self._t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        t = self._t
        self._t += 1
        a = np.zeros(self.action_dim)
        for i in range(self.action_dim):
            a[i] = 0.4 * np.sin(t * 0.05 * (i + 1))
        obs_flat = np.asarray(obs).flatten()
        for i in range(min(self.action_dim, len(obs_flat))):
            a[i] += np.clip(-0.2 * obs_flat[i], -0.3, 0.3)
        a += self.rng.randn(self.action_dim) * 0.2
        return np.clip(a, self.low, self.high).astype(np.float32)


class ConfoundedPolicy:
    """``a_t = clip( pi_probe(o_t) + kappa * K C_t )``.

    Reads the latent confounder straight off the env.  This is the *only*
    place the confounder enters the action channel — which is exactly what
    makes ``do(a ~ Unif)`` a severing intervention: the probe overwrites
    the returned action, so the ``C -> a`` edge disappears under
    intervention while ``C -> d`` stays intact.
    """

    def __init__(self, env, base_policy, kappa: float = 0.5, seed: int = 42):
        self.env = env
        self.base = base_policy
        self.kappa = float(kappa)
        self.low = env.action_space.low.flatten()
        self.high = env.action_space.high.flatten()
        rng = np.random.RandomState(seed + 31)
        K = rng.normal(0.0, 1.0, (int(np.prod(env.action_space.shape)),
                                  env.n_c))
        K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-12
        self.K = K

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = np.asarray(self.base(obs), dtype=np.float64).flatten()
        if self.kappa != 0.0:
            a = a + self.kappa * (self.K @ self.env.confounder_state)
        return np.clip(a, self.low, self.high).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE-POLICY COVARIATES
# ═══════════════════════════════════════════════════════════════════════════════

def measure_probe_covariates(env, policy, n_steps: int = 3000,
                             n_bins: int = 50, seed: int = 0) -> dict:
    """Characterise the baseline branch's action distribution.

    Recorded per (env, kappa, lambda, seed) cell so that any loss of
    detection power can be *explained* rather than asserted.  Note that
    ``kappa`` does **not** drive the policy towards uniform: because the
    action is clipped to the box, large ``kappa`` drives it towards
    saturated bang-bang, which is further from uniform in KL, not closer.
    The contrast between the baseline and intervention branch is therefore
    non-monotone in ``kappa`` — hence these covariates are measured, not
    predicted.

    Returns:
        sat_frac         fraction of action components at the box boundary
        action_std       marginal std of the action components
        diff_entropy     histogram estimate of differential entropy
                         (Uniform[-1,1] = 0.693)
        kl_to_uniform    histogram estimate of KL(pi || Unif) (Uniform = 0);
                         a lower bound only, since boundary atoms make the
                         true KL divergent
    """
    lo = env.action_space.low.flatten()
    hi = env.action_space.high.flatten()
    obs, _ = env.reset(seed=seed)
    A = []
    for _ in range(n_steps):
        a = np.asarray(policy(obs), dtype=np.float64).flatten()
        A.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    A = np.asarray(A)

    tol = 1e-3 * (hi - lo)
    sat = float(np.mean((A >= hi - tol) | (A <= lo + tol)))

    ents, kls = [], []
    for j in range(A.shape[1]):
        h, edges = np.histogram(A[:, j], bins=n_bins,
                                range=(float(lo[j]), float(hi[j])),
                                density=True)
        w = edges[1] - edges[0]
        p = h * w
        p = p[p > 0]
        ents.append(float(-np.sum(p * np.log(p / w))))
        q = w / (hi[j] - lo[j])
        kls.append(float(np.sum(p * np.log(p / q))))

    return {"sat_frac": sat,
            "action_std": float(A.std()),
            "diff_entropy": float(np.mean(ents)),
            "kl_to_uniform": float(np.mean(kls))}
