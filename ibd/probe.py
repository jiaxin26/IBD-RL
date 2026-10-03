"""
IBD Probe — General-purpose interventional boundary discovery
===============================================================

Works with **any** gymnasium-compatible environment.  No special
env interface required — interventions are applied externally by
overriding action dimensions before ``env.step()``.

Usage::

    import gymnasium as gym
    from ibd import IBDProbe

    env = gym.make("HalfCheetah-v4")
    probe = IBDProbe(env)
    mask = probe.discover()                   # random-policy probe
    mask = probe.discover(my_trained_policy)   # policy-guided probe

    print(mask.soi_dims)        # {0, 1, 2, 5, 8, ...}
    print(mask.illusory_dims)   # {3, 4, 6, 7, ...}
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import logging
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from ibd.cim import CIMEstimator, CIMResult
from ibd.mask import CausalMask

logger = logging.getLogger(__name__)

Policy = Callable[[np.ndarray], np.ndarray]   # obs -> action


class IBDProbe:
    """
    Discovers which observation dimensions are causally influenced
    by actions via interventional testing.

    The probe:
      1. Collects baseline trajectories under a given policy
      2. For each action dim, collects intervention trajectories
         where that dim is randomised (do-operator)
      3. Runs a permutation test (CIM) to find significantly
         different observation responses
      4. Returns a ``CausalMask`` over observation dimensions

    Args:
        env: gymnasium environment (or anything with .reset / .step /
             .observation_space / .action_space)
        n_baseline: number of baseline trajectories
        n_intervention: number of trajectories per action-dim intervention
        traj_length: steps per trajectory
        horizons: h-step differences to test (e.g. [1, 5, 10])
        alpha: significance level after multiple-testing correction
        mask_mode: 'soft' or 'hard'
        n_permutations: permutations for the CIM test
        seed: random seed for reproducibility
        override_beta: fraction of the commanded action that actually
             reaches the plant during an intervention.  ``1.0`` (default)
             is a perfect do-operator; ``beta < 1`` models an actuator
             that can only partially override the incumbent policy,
             executing ``beta * u + (1 - beta) * pi(o)``.  See
             ``experiments/run_override.py``.
    """

    def __init__(
        self,
        env: gym.Env,
        n_baseline: int = 80,
        n_intervention: int = 50,
        traj_length: int = 200,
        horizons: Optional[List[int]] = None,
        alpha: float = 0.05,
        mask_mode: str = "soft",
        n_permutations: int = 1000,
        seed: int = 42,
        override_beta: float = 1.0,
    ):
        self.env = env
        self.n_baseline = n_baseline
        self.n_intervention = n_intervention
        self.traj_length = traj_length
        self.horizons = horizons or [1, 5, 10]
        self.alpha = alpha
        self.mask_mode = mask_mode
        self.seed = seed
        if not 0.0 <= override_beta <= 1.0:
            raise ValueError(
                f"override_beta must lie in [0, 1], got {override_beta}")
        self.override_beta = float(override_beta)

        # Infer dimensions
        obs_space = env.observation_space
        act_space = env.action_space
        assert isinstance(obs_space, gym.spaces.Box), \
            f"IBDProbe requires Box obs space, got {type(obs_space)}"
        assert isinstance(act_space, gym.spaces.Box), \
            f"IBDProbe requires Box action space, got {type(act_space)}"
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(act_space.shape))
        self.action_low = act_space.low.flatten()
        self.action_high = act_space.high.flatten()

        # CIM estimator
        self.cim = CIMEstimator(
            n_permutations=n_permutations, alpha=alpha,
            seed=seed, adaptive_permutations=True)

        self._rng = np.random.RandomState(seed)
        self._round = 0

    # ── public API ────────────────────────────────────────────────────

    def discover(self, policy: Optional[Policy] = None,
                 skip_dims: Optional[Set[int]] = None) -> CausalMask:
        """
        Run one round of interventional boundary discovery.

        Args:
            policy: obs -> action callable.  If None, uses uniform random.
            skip_dims: state dims already confirmed as causal — skip
                       expensive permutation testing for these.

        Returns:
            CausalMask with discovered SoI dimensions.
        """
        if policy is None:
            policy = self._random_policy

        self._round += 1
        t0 = time.time()

        # Collect data
        baseline = self._collect(policy, self.n_baseline, intervention=None)
        interventions = {}
        for a in range(self.action_dim):
            interventions[a] = self._collect(
                policy, self.n_intervention,
                intervention={"dim": a, "mode": "randomize"})

        logger.info(f"IBD round {self._round}: data collected in "
                    f"{time.time() - t0:.1f}s  "
                    f"(baseline={baseline.shape}, "
                    f"per-interv={list(interventions.values())[0].shape})")

        # Run CIM
        result = self.cim.estimate_matrix(
            baseline, interventions,
            state_dims=list(range(self.obs_dim)),
            horizons=self.horizons,
            skip_dims=skip_dims)

        # Build mask
        soi = result.get_estimated_soi()
        es = result.get_effect_sizes(self.obs_dim)
        pv = result.get_min_pvalues(self.obs_dim)

        mask = CausalMask.from_soi(
            soi, self.obs_dim, mode=self.mask_mode,
            p_values=pv, effect_sizes=es, round_idx=self._round)

        dt = time.time() - t0
        logger.info(f"IBD round {self._round} complete ({dt:.1f}s): "
                    f"SoI={sorted(soi)}, "
                    f"illusory={sorted(mask.illusory_dims)}")

        return mask

    def discover_joint(self, policy: Optional[Policy] = None,
                       test_type: str = "welch",
                       ) -> Tuple[Set[int], dict]:
        """
        Simplified IBD using **joint** intervention: randomise ALL
        action dimensions simultaneously.

        This is the recommended approach for dimension selection:
        - 1 intervention group instead of action_dim groups → 6× faster
        - Only obs_dim tests per horizon → light MTC burden
        - Stronger signal (all action coupling broken at once)
        - Uses Welch t-test (not permutation) for continuous p-values,
          avoiding the discrete p-value floor of permutation tests
          that makes BH correction infeasible with large obs_dim.

        Args:
            policy: obs -> action callable.  If None, uses structured
                random (sinusoidal + feedback).
            test_type: statistical test to use.
                ``"welch"`` — Welch t-test on mean absolute h-step
                    diffs (default).  Most powerful when the causal
                    effect shifts the *mean* of the diff distribution.
                ``"ks"`` — two-sample Kolmogorov-Smirnov test.
                    Detects *any* distributional difference (mean,
                    variance, shape).  Useful when causal effects
                    change variance without shifting the mean.

        Returns:
            (soi_set, info_dict) where info_dict contains diagnostics.
        """
        if test_type not in ("welch", "ks"):
            raise ValueError(
                f"test_type must be 'welch' or 'ks', got '{test_type}'")
        from scipy import stats as sp_stats

        if policy is None:
            policy = self._random_policy

        self._round += 1
        t0 = time.time()

        # Collect baseline (normal policy)
        baseline = self._collect(policy, self.n_baseline, intervention=None)

        # Collect intervention: ALL actions randomised
        intervention = self._collect(
            policy, self.n_intervention,
            intervention={"dim": "all", "mode": "randomize"})

        dt_collect = time.time() - t0
        logger.info(f"IBD joint: data collected in {dt_collect:.1f}s  "
                    f"(baseline={baseline.shape}, interv={intervention.shape})")

        # ── Per-dim statistical tests on absolute h-step diffs ─────────
        # Use ONLY absolute diffs (more powerful for detecting "any change
        # in dynamics" without assuming direction).
        raw_pvals = []
        effect_sizes = []
        test_keys = []    # (state_dim, horizon)

        for sd in range(self.obs_dim):
            for h in self.horizons:
                y_b = self.cim._extract(baseline, sd, h, absolute=True)
                y_i = self.cim._extract(intervention, sd, h, absolute=True)

                if y_b is None or y_i is None:
                    continue
                if len(y_b) < 10 or len(y_i) < 10:
                    continue

                if test_type == "ks":
                    # ── Kolmogorov-Smirnov test ──────────────────────
                    # Detects any distributional difference (mean, var,
                    # shape).  Effect size = KS statistic (max CDF gap).
                    ks_stat, p_val = sp_stats.ks_2samp(y_b, y_i)
                    d = float(ks_stat)

                else:
                    # ── Welch t-test (default) ───────────────────────
                    # Most powerful for mean-shift alternatives.
                    # Effect size = Hedges' g (bias-corrected Cohen's d).
                    t_stat, p_val = sp_stats.ttest_ind(
                        y_b, y_i, equal_var=False)

                    n0, n1 = len(y_b), len(y_i)
                    v0 = float(np.var(y_b, ddof=1))
                    v1 = float(np.var(y_i, ddof=1))
                    pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1)
                                     / (n0 + n1 - 2))
                    d = ((np.mean(y_b) - np.mean(y_i)) / pooled
                         if pooled > 1e-12 else 0.0)
                    d *= 1 - 3 / (4 * (n0 + n1) - 9)   # Hedges correction

                raw_pvals.append(float(p_val))
                effect_sizes.append(float(abs(d)))
                test_keys.append((sd, h))

        # ── Benjamini-Hochberg correction ─────────────────────────────
        m = len(raw_pvals)
        raw_arr = np.array(raw_pvals)
        idx = np.argsort(raw_arr)
        sorted_p = raw_arr[idx]

        # Find BH threshold
        significant = np.zeros(m, dtype=bool)
        max_k = -1
        for k in range(m):
            if sorted_p[k] <= (k + 1) / m * self.alpha:
                max_k = k
        if max_k >= 0:
            significant[idx[:max_k + 1]] = True

        # Adjusted p-values
        adj_p = np.ones(m)
        for k in range(m - 1, -1, -1):
            i = idx[k]
            adj_p[i] = min(sorted_p[k] * m / (k + 1), 1.0)
            if k < m - 1:
                adj_p[i] = min(adj_p[i], adj_p[idx[k + 1]])

        # ── Build SoI ────────────────────────────────────────────────
        soi = set()
        min_pvals = np.ones(self.obs_dim)
        max_effect = np.zeros(self.obs_dim)

        for i, (sd, h) in enumerate(test_keys):
            if significant[i]:
                soi.add(sd)
            min_pvals[sd] = min(min_pvals[sd], adj_p[i])
            max_effect[sd] = max(max_effect[sd], effect_sizes[i])

        dt_total = time.time() - t0

        # Log diagnostics
        n_sig = int(significant.sum())
        if m > 0:
            min_raw = float(raw_arr.min())
            med_raw = float(np.median(raw_arr))
        else:
            min_raw = med_raw = 1.0
        logger.info(f"IBD joint ({dt_total:.1f}s): {len(soi)}/{self.obs_dim} "
                    f"causal dims | {n_sig}/{m} tests significant | "
                    f"min_raw_p={min_raw:.2e} med_raw_p={med_raw:.2e}")
        if soi:
            logger.info(f"  SoI={sorted(soi)}")

        info = {
            "soi": soi,
            "test_type": test_type,
            "n_tests": m,
            "n_significant": n_sig,
            "time_collect_s": dt_collect,
            "time_total_s": dt_total,
            "p_values": min_pvals,
            "effect_sizes": max_effect,
            "min_raw_p": min_raw,
        }
        return soi, info

    def get_cim_result(self, policy: Optional[Policy] = None) -> CIMResult:
        """Run IBD and return the full CIM result (for analysis)."""
        if policy is None:
            policy = self._random_policy
        baseline = self._collect(policy, self.n_baseline, intervention=None)
        interventions = {}
        for a in range(self.action_dim):
            interventions[a] = self._collect(
                policy, self.n_intervention,
                intervention={"dim": a, "mode": "randomize"})
        return self.cim.estimate_matrix(
            baseline, interventions,
            state_dims=list(range(self.obs_dim)),
            horizons=self.horizons)

    # ── data collection ───────────────────────────────────────────────

    def _collect(self, policy: Policy, n_trajs: int,
                 intervention: Optional[dict]) -> np.ndarray:
        """
        Collect n_trajs trajectories of length self.traj_length.

        If ``intervention`` is given, the specified action dim is
        randomised (do-operator) at each step.
        """
        trajs = []
        for i in range(n_trajs):
            obs, _ = self.env.reset(
                seed=int(self._rng.randint(0, 2**31)))
            traj = [obs.flatten()]
            for t in range(self.traj_length - 1):
                action = np.asarray(policy(obs), dtype=np.float64).flatten()
                if intervention is not None:
                    action = self._apply_intervention(action, intervention)
                action = np.clip(action, self.action_low, self.action_high)
                step_result = self.env.step(action)
                # Handle gymnasium (5-tuple) and gym (4-tuple)
                if len(step_result) == 5:
                    obs, rew, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, rew, done, info = step_result
                traj.append(obs.flatten())
                if done:
                    obs, _ = self.env.reset()
                    traj.append(obs.flatten())
                    # Pad to fixed length if episode ended early
                    if len(traj) >= self.traj_length:
                        break
            # Pad or truncate to exactly traj_length
            traj_arr = np.array(traj[:self.traj_length])
            if len(traj_arr) < self.traj_length:
                pad = np.tile(traj_arr[-1:],
                              (self.traj_length - len(traj_arr), 1))
                traj_arr = np.concatenate([traj_arr, pad], axis=0)
            trajs.append(traj_arr)
        return np.array(trajs)    # (n_trajs, traj_length, obs_dim)

    def _apply_intervention(self, action: np.ndarray,
                            intervention: dict) -> np.ndarray:
        """Override action dimension(s) with random values (do-operator).

        If dim='all', randomise every action dimension (joint intervention).
        If dim=int, randomise only that dimension.

        With ``override_beta < 1`` the override is *partial*: the executed
        action is ``beta * u + (1 - beta) * a_policy``, i.e. the actuator
        cannot fully sever the incumbent policy's influence.  The uniform
        draw is taken in both cases so that the RNG stream — and hence
        every result produced before this option existed — is unchanged
        at ``beta = 1``.
        """
        action = action.copy()
        d = intervention["dim"]
        beta = self.override_beta

        def _blend(u: float, a: float) -> float:
            return u if beta >= 1.0 else beta * u + (1.0 - beta) * a

        if d == "all":
            for i in range(len(action)):
                u = self._rng.uniform(self.action_low[i],
                                      self.action_high[i])
                action[i] = _blend(u, action[i])
        elif isinstance(d, int) and d < len(action):
            u = self._rng.uniform(self.action_low[d],
                                  self.action_high[d])
            action[d] = _blend(u, action[d])
        return action

    def _random_policy(self, obs: np.ndarray) -> np.ndarray:
        """Default probe policy: sinusoidal + feedback + noise.

        A purely random policy makes baseline and intervention
        identically distributed (no coupling to break).  This
        structured default creates enough action-state correlation
        for the CIM to detect causal dims even before RL training.
        """
        t = getattr(self, '_policy_t', 0)
        self._policy_t = t + 1
        a = np.zeros(self.action_dim)
        # Sinusoidal base (different freq per dim → temporal structure)
        for i in range(self.action_dim):
            a[i] = 0.4 * np.sin(t * 0.05 * (i + 1))
        # Weak state feedback (creates action-state coupling)
        obs_flat = obs.flatten()
        for i in range(min(self.action_dim, len(obs_flat))):
            a[i] += np.clip(-0.2 * obs_flat[i], -0.3, 0.3)
        # Exploration noise
        a += self._rng.randn(self.action_dim) * 0.2
        return np.clip(a, self.action_low, self.action_high).astype(np.float32)