"""
Distracting DMControl Environments
====================================

Wraps DeepMind Control Suite tasks with state-space distractors
for benchmarking causal dimension discovery.

Distractor types
~~~~~~~~~~~~~~~~~
* **Autonomous** — OU processes and coupled oscillators,
  independent of everything.  Should be trivially filtered by
  any method if their variance / scale differs from true dims.
* **Mimicking** — exogenous processes deliberately tuned to have
  similar variance, autocorrelation, and frequency content as
  typical DMControl proprioceptive dimensions.  Observational
  methods (variance, MI) cannot reliably distinguish them from
  true state dims.  IBD's do-operator reveals they are not
  causally downstream of actions.
* **Correlated-reward** — driven by exogenous process that is
  correlated with episode progress / reward magnitude (but not
  causally downstream of actions).  Tricks reward-predictive
  feature selectors but not IBD.

All distractors are **truly exogenous** — no causal path from
actions exists.  Crucially, distractors do NOT perturb observations
or bias actions.  This guarantees that the oracle mask (keeping
only true state dims) is a valid upper bound: oracle >= full_state,
because distractors carry zero information useful for control.

Why observational methods fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Variance-based selection** fails because mimicking distractors
  are calibrated to the same scale as true dims.
* **Mutual-information selection** fails because (a) MI estimators
  are noisy with finite data, especially on high-dimensional
  autocorrelated time series, and (b) mimicking distractors have
  temporal structure that creates spurious finite-sample MI with
  actions/rewards.
* **IBD** succeeds because intervening on action dimensions has
  zero effect on exogenous distractor distributions, regardless
  of their statistical properties.

Usage::

    from experiments.dmcontrol_distractors import make_env, BENCHMARKS

    env = make_env("walker", "walk", distractor_config="medium")
    print(env.true_dims)         # {0, 1, ..., 23}
    print(env.distractor_dims)   # {24, 25, ..., 38}
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRACTOR SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class DistractorSource:
    """Base class for distractor signal generators."""
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self._state = np.zeros(dim)

    def reset(self):
        self._state = self.rng.normal(0, 0.1, self.dim)
        return self._state.copy()

    def step(self, dt: float = 0.01) -> np.ndarray:
        raise NotImplementedError

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()


class AutonomousOU(DistractorSource):
    """Ornstein–Uhlenbeck processes — independent of everything."""

    def __init__(self, dim: int, tau: float = 2.0, sigma: float = 0.3,
                 seed: int = 42):
        super().__init__(dim, seed)
        self.tau = tau
        self.sigma = sigma

    def step(self, dt=0.01):
        noise = self.rng.normal(0, self.sigma * np.sqrt(dt), self.dim)
        self._state += -self._state / self.tau * dt + noise
        return self._state.copy()


class AutonomousOscillator(DistractorSource):
    """Coupled damped oscillators — rich autonomous dynamics."""

    def __init__(self, dim: int, seed: int = 42):
        super().__init__(dim, seed)
        self.freqs = self.rng.uniform(0.5, 3.0, dim)
        self.coupling = self.rng.normal(0, 0.05, (dim, dim))
        np.fill_diagonal(self.coupling, 0)
        self._vel = np.zeros(dim)

    def reset(self):
        self._state = self.rng.normal(0, 0.2, self.dim)
        self._vel = self.rng.normal(0, 0.1, self.dim)
        return self._state.copy()

    def step(self, dt=0.01):
        acc = (-self.freqs**2 * self._state
               + self.coupling @ self._state
               - 0.05 * self._vel
               + self.rng.normal(0, 0.02, self.dim))
        self._vel += acc * dt
        self._state += self._vel * dt
        return self._state.copy()


class MimickingDistractor(DistractorSource):
    """
    Exogenous process that mimics the statistical fingerprint of
    real DMControl proprioceptive dimensions.

    Combines multi-frequency sinusoidal signals, OU drift, and
    episode-phase-dependent variance to produce trajectories with:
      - Variance calibrated to match true observation dimensions
        (using ``ref_scale`` derived from ``true_obs_dim``)
      - Autocorrelation decay similar to physics-simulated state
      - Non-trivial frequency content in the 0.5–5 Hz range
        (matching typical locomotion cycle frequencies)
      - **Episode-phase-dependent variance**: amplitude ramps up
        during the episode, mimicking how joint velocities and
        angular displacements grow as a controller activates.
        This makes the per-step Δs variance look action-conditioned
        even though it is entirely exogenous.

    These properties make mimicking distractors hard to distinguish
    from true dims using variance-based or MI-based selection, but
    IBD trivially rejects them because they are not downstream of
    any action.

    Design note: NO observation perturbation, NO action bias.
    The confounding challenge is purely statistical, not causal.
    """

    def __init__(self, dim: int,
                 ref_scale: float = 1.0,
                 seed: int = 42):
        super().__init__(dim, seed)
        self.ref_scale = ref_scale
        # Multi-frequency sinusoidal components (mimic locomotion cycles)
        self._n_harmonics = 3
        self._freqs = self.rng.uniform(0.3, 5.0,
                                       (dim, self._n_harmonics))
        self._phases = self.rng.uniform(0, 2 * np.pi,
                                        (dim, self._n_harmonics))
        self._amps = self.rng.uniform(0.1, 0.6,
                                      (dim, self._n_harmonics))
        # OU drift (matches slow state drift)
        self._ou = np.zeros(dim)
        self._ou_tau = self.rng.uniform(1.0, 5.0, dim)
        self._ou_sigma = self.rng.uniform(0.1, 0.4, dim)
        # Episode-phase ramp: variance grows during episode
        self._ramp_rate = self.rng.uniform(0.3, 1.5, dim)
        self._ramp_max = self.rng.uniform(1.5, 3.0, dim)
        # Time counter
        self._t = 0.0
        self._ep_t = 0.0  # within-episode time

    def reset(self):
        self._t = self.rng.uniform(0, 10.0)  # random global phase
        self._ep_t = 0.0
        self._ou = self.rng.normal(0, 0.2, self.dim)
        self._phases = self.rng.uniform(0, 2 * np.pi,
                                        (self.dim, self._n_harmonics))
        self._update_state()
        return self._state.copy()

    def _update_state(self):
        """Combine sinusoidal + OU + episode-phase ramp."""
        # Episode-phase amplitude ramp: starts low, ramps up
        ramp = np.minimum(1.0 + self._ramp_rate * self._ep_t,
                          self._ramp_max)
        # Sinusoidal part: sum of harmonics per dim
        sin_part = np.zeros(self.dim)
        for h in range(self._n_harmonics):
            sin_part += (self._amps[:, h]
                         * np.sin(2 * np.pi * self._freqs[:, h] * self._t
                                  + self._phases[:, h]))
        self._state = (sin_part + self._ou) * ramp * self.ref_scale

    def step(self, dt=0.01):
        self._t += dt
        self._ep_t += dt
        # OU drift
        noise = self.rng.normal(0, 1.0, self.dim) * self._ou_sigma * np.sqrt(dt)
        self._ou += -self._ou / self._ou_tau * dt + noise
        self._update_state()
        return self._state.copy()


class RewardCorrelatedDistractor(DistractorSource):
    """
    Signal that mimics within-episode reward structure.

    Instead of a simple linear trend (trivially detectable as
    non-stationary), this uses a periodic "effort curve" that
    resembles the reward profile of locomotion tasks: low at
    episode start, rising to a plateau, with periodic oscillations
    matching gait cycles.

    Tricks reward-predictive feature selectors into thinking this
    dim is useful, but IBD rejects it because the intervention
    breaks the spurious link.
    """

    def __init__(self, dim: int, growth_rate: float = 0.01,
                 seed: int = 42):
        super().__init__(dim, seed)
        self.growth_rate = growth_rate
        self._t = 0.0
        # Per-dim oscillation frequencies (mimic gait cycles)
        self._gait_freq = self.rng.uniform(0.5, 2.5, dim)
        self._gait_amp = self.rng.uniform(0.05, 0.15, dim)
        self._ou = np.zeros(dim)
        self._ou_sigma = self.rng.uniform(0.02, 0.06, dim)

    def reset(self):
        self._t = 0.0
        self._ou = self.rng.normal(0, 0.02, self.dim)
        self._state = self.rng.normal(0, 0.05, self.dim)
        return self._state.copy()

    def step(self, dt=0.01):
        self._t += dt
        # Saturating ramp (like reward rising to plateau)
        plateau = 1.0 - np.exp(-self.growth_rate * self._t * 50)
        # Gait-like oscillations on top
        gait = self._gait_amp * np.sin(
            2 * np.pi * self._gait_freq * self._t)
        # OU noise for realism
        self._ou += (-self._ou * 0.5 * dt
                     + self.rng.normal(0, 1.0, self.dim)
                     * self._ou_sigma * np.sqrt(dt))
        self._state = plateau * np.ones(self.dim) + gait + self._ou
        return self._state.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRACTOR CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_distractors(config_name: str, true_obs_dim: int,
                       seed: int) -> List[DistractorSource]:
    """Build a list of distractors from a named config.

    Args:
        config_name: 'easy', 'medium', or 'hard'
        true_obs_dim: dimensionality of true observation (used for
            calibrating mimicking distractor scales)
        seed: random seed
    """
    # Calibrate mimicking distractor scale to match typical DMControl
    # state magnitude.  Environments with more dims (walker=24) tend
    # to have larger-magnitude velocity components than small envs
    # (cartpole=5).  This heuristic keeps distractors in the right
    # ballpark so variance-based selectors can't trivially reject them.
    base_scale = 0.5 + 0.05 * true_obs_dim  # walker→1.7, cartpole→0.75

    configs = {
        # Minimal: 6 autonomous only (easy sanity check)
        # Low variance, obviously different from true state dims.
        "easy": lambda: [
            AutonomousOU(dim=4, tau=2.0, sigma=0.3, seed=seed),
            AutonomousOscillator(dim=2, seed=seed + 1),
        ],

        # Medium: 50 distractors total
        # With 24 true dims → 74 total obs.  The 256-unit MLP must
        # allocate capacity to ~2x as many input dims, slowing
        # learning and capping asymptotic performance for full_state.
        "medium": lambda: [
            AutonomousOU(dim=6, tau=2.0, sigma=0.3, seed=seed),
            AutonomousOU(dim=6, tau=3.5, sigma=0.2, seed=seed + 10),
            AutonomousOscillator(dim=4, seed=seed + 1),
            MimickingDistractor(dim=8, ref_scale=base_scale * 0.6,
                                seed=seed + 2),
            MimickingDistractor(dim=8, ref_scale=base_scale * 1.0,
                                seed=seed + 3),
            MimickingDistractor(dim=6, ref_scale=base_scale * 0.4,
                                seed=seed + 4),
            MimickingDistractor(dim=6, ref_scale=base_scale * 1.3,
                                seed=seed + 5),
            RewardCorrelatedDistractor(dim=3, growth_rate=0.01,
                                       seed=seed + 6),
            RewardCorrelatedDistractor(dim=3, growth_rate=0.02,
                                       seed=seed + 7),
        ],

        # Hard: 100 distractors total
        # With 24 true dims → 124 total obs.  The signal-to-noise
        # ratio in the input is ~1:4, severely degrading full_state.
        "hard": lambda: [
            AutonomousOU(dim=10, tau=1.5, sigma=0.35, seed=seed),
            AutonomousOU(dim=8, tau=4.0, sigma=0.25, seed=seed + 10),
            AutonomousOscillator(dim=8, seed=seed + 1),
            MimickingDistractor(dim=10, ref_scale=base_scale * 0.5,
                                seed=seed + 2),
            MimickingDistractor(dim=10, ref_scale=base_scale * 0.8,
                                seed=seed + 3),
            MimickingDistractor(dim=10, ref_scale=base_scale * 1.2,
                                seed=seed + 4),
            MimickingDistractor(dim=8, ref_scale=base_scale * 0.3,
                                seed=seed + 5),
            MimickingDistractor(dim=8, ref_scale=base_scale * 0.7,
                                seed=seed + 6),
            MimickingDistractor(dim=8, ref_scale=base_scale * 1.0,
                                seed=seed + 7),
            MimickingDistractor(dim=6, ref_scale=base_scale * 1.5,
                                seed=seed + 8),
            RewardCorrelatedDistractor(dim=4, growth_rate=0.015,
                                       seed=seed + 9),
            RewardCorrelatedDistractor(dim=4, growth_rate=0.008,
                                       seed=seed + 11),
            RewardCorrelatedDistractor(dim=6, growth_rate=0.025,
                                       seed=seed + 12),
        ],
    }
    builder = configs.get(config_name)
    if builder is None:
        raise ValueError(
            f"Unknown distractor config: '{config_name}'. "
            f"Choose from: {list(configs.keys())}")
    return builder()


def _build_distractors_custom(n_total: int, true_obs_dim: int,
                              seed: int) -> List[DistractorSource]:
    """Build exactly ``n_total`` distractor dims with a realistic mix.

    Composition (matching medium/hard ratios):
      - 25% autonomous (OU + oscillators)
      - 55% mimicking
      - 20% reward-correlated

    Used by the distractor-scaling experiment (run_scaling.py) to sweep
    intermediate counts like 12, 24, 36, 75.
    """
    base_scale = 0.5 + 0.05 * true_obs_dim

    n_auto = max(2, round(n_total * 0.25))
    n_reward = max(1, round(n_total * 0.20))
    n_mimic = n_total - n_auto - n_reward
    if n_mimic < 0:
        n_mimic = 0
        n_auto = n_total - n_reward

    sources: List[DistractorSource] = []
    s = seed  # seed counter

    # --- autonomous ---
    n_ou = n_auto // 2
    n_osc = n_auto - n_ou
    if n_ou > 0:
        sources.append(AutonomousOU(dim=n_ou, tau=2.0, sigma=0.3, seed=s))
        s += 1
    if n_osc > 0:
        sources.append(AutonomousOscillator(dim=n_osc, seed=s))
        s += 1

    # --- mimicking (split into chunks of ~8 with varied scales) ---
    scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    remaining = n_mimic
    si = 0
    while remaining > 0:
        chunk = min(8, remaining)
        sc = scales[si % len(scales)]
        sources.append(MimickingDistractor(
            dim=chunk, ref_scale=base_scale * sc, seed=s))
        s += 1
        si += 1
        remaining -= chunk

    # --- reward-correlated ---
    rates = [0.01, 0.015, 0.025]
    remaining = n_reward
    ri = 0
    while remaining > 0:
        chunk = min(4, remaining)
        sources.append(RewardCorrelatedDistractor(
            dim=chunk, growth_rate=rates[ri % len(rates)], seed=s))
        s += 1
        ri += 1
        remaining -= chunk

    actual = sum(d.dim for d in sources)
    assert actual == n_total, f"Built {actual} dims, expected {n_total}"
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# GYMNASIUM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class DistractingDMControlEnv(gym.Env):
    """
    DMControl task with state-space distractors appended to observations.

    The wrapper:
      * Loads a dm_control Suite task
      * Flattens the proprioceptive observation to a 1-D vector
      * Appends distractor dimensions to the observation
      * Tracks ground-truth SoI labels for evaluation

    All distractors are purely exogenous.  No observation perturbation
    or action bias is applied — the agent's dynamics are identical to
    the unwrapped task.  This guarantees that oracle masking (retaining
    only true dims) is a valid performance upper bound.

    Args:
        domain_name:   dm_control domain (e.g. 'walker', 'cheetah')
        task_name:     dm_control task (e.g. 'walk', 'run')
        distractor_config: 'easy', 'medium', or 'hard'
        seed:          random seed
        time_limit:    episode length in seconds (default: env default)
    """

    metadata = {"render_modes": []}

    def __init__(self, domain_name: str = "walker",
                 task_name: str = "walk",
                 distractor_config: str = "medium",
                 seed: int = 42,
                 time_limit: Optional[float] = None):
        super().__init__()
        from dm_control import suite

        # Create dm_control env
        task_kwargs = {"random": seed}
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        self._dm_env = suite.load(domain_name, task_name,
                                  task_kwargs=task_kwargs)
        self._domain = domain_name
        self._task = task_name

        # True observation dimensions
        obs_spec = self._dm_env.observation_spec()
        self._obs_keys = sorted(obs_spec.keys())
        self._true_obs_dim = sum(
            int(np.prod(obs_spec[k].shape)) for k in self._obs_keys)

        # Action spec
        action_spec = self._dm_env.action_spec()
        self._action_dim = int(np.prod(action_spec.shape))

        # Build distractors: accept "easy"/"medium"/"hard" OR an integer
        if isinstance(distractor_config, int):
            self._distractors = _build_distractors_custom(
                distractor_config, self._true_obs_dim, seed)
            self._distractor_config = f"custom_{distractor_config}"
        else:
            self._distractors = _build_distractors(
                distractor_config, self._true_obs_dim, seed)
            self._distractor_config = distractor_config
        self._distractor_dim = sum(d.dim for d in self._distractors)
        self._distractor_config = distractor_config

        # Total obs = true + distractors
        total_dim = self._true_obs_dim + self._distractor_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32)

        # Ground truth labels
        self.true_dims = set(range(self._true_obs_dim))
        self.distractor_dims = set(range(self._true_obs_dim, total_dim))
        self.true_obs_dim = self._true_obs_dim

        self._rng = np.random.RandomState(seed)
        self._dt = self._dm_env.control_timestep()

    # ── gymnasium interface ───────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        timestep = self._dm_env.reset()
        for d in self._distractors:
            d.reset()
        obs = self._build_obs(timestep)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action,
                         self.action_space.low, self.action_space.high)

        # Clean execution — no action bias.
        # Distractors are purely exogenous.
        timestep = self._dm_env.step(action)

        # Step distractors (independent of action/state)
        for d in self._distractors:
            d.step(self._dt)

        obs = self._build_obs(timestep)
        reward = float(timestep.reward or 0.0)
        terminated = timestep.last()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _build_obs(self, timestep) -> np.ndarray:
        """Flatten dm_control obs + append distractor dims.

        No perturbation is applied to true observation dimensions.
        """
        # 1. Flatten true observation (unmodified)
        parts = []
        for k in self._obs_keys:
            val = timestep.observation[k]
            parts.append(np.asarray(val, dtype=np.float32).flatten())
        true_obs = np.concatenate(parts)

        # 2. Append distractor dims
        distractor_parts = [d.state.astype(np.float32)
                            for d in self._distractors]
        return np.concatenate([true_obs] + distractor_parts)

    # ── info ──────────────────────────────────────────────────────────

    @property
    def name(self):
        return f"{self._domain}_{self._task}_{self._distractor_config}"

    def __repr__(self):
        return (f"DistractingDMControlEnv("
                f"domain={self._domain}, task={self._task}, "
                f"distractors={self._distractor_config}, "
                f"true_dim={self._true_obs_dim}, "
                f"distractor_dim={self._distractor_dim}, "
                f"total_dim={self.observation_space.shape[0]})")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(domain: str, task: str,
             distractor_config = "medium",
             seed: int = 42, **kwargs) -> DistractingDMControlEnv:
    """Convenience factory.  distractor_config can be 'easy'/'medium'/'hard' or an int."""
    return DistractingDMControlEnv(
        domain_name=domain, task_name=task,
        distractor_config=distractor_config,
        seed=seed, **kwargs)


# Standard benchmark tasks
BENCHMARK_TASKS = [
    ("walker", "walk"),
    ("cheetah", "run"),
    ("cartpole", "swingup"),
    ("finger", "spin"),
    ("hopper", "hop"),
    ("reacher", "hard"),
]

DISTRACTOR_CONFIGS = ["easy", "medium", "hard"]