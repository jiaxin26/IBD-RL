"""
Stable-Baselines3 Integration
===============================

Two components for plugging IBD into any SB3 agent:

``CausalMaskExtractor``
    Custom features extractor that soft-masks observations.
    Observation space shape is preserved, so the SB3 policy
    architecture doesn't change when the mask updates.

``IBDCallback``
    Training callback that periodically runs an IBD probe
    and updates the feature extractor's mask.

Usage::

    from ibd.sb3 import make_ibd_sac

    model, probe = make_ibd_sac(env, probe_env=probe_env)
    model.learn(total_timesteps=500_000)
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional

import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ibd.probe import IBDProbe
from ibd.mask import CausalMask

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CausalMaskExtractor(BaseFeaturesExtractor):
    """
    Multiplies observations by the current causal mask weights.

    The mask can be updated in-place during training (by the callback).
    Output dimension = input dimension (soft masking preserves shape).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 initial_mask: Optional[CausalMask] = None):
        features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim)

        if initial_mask is not None:
            w = initial_mask.weights
        else:
            w = np.ones(features_dim, dtype=np.float32)
        self.register_buffer(
            "_weights", torch.as_tensor(w, dtype=torch.float32))
        self._mask: Optional[CausalMask] = initial_mask

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self._weights

    def update_mask(self, mask: CausalMask):
        """Called by IBDCallback to update the mask mid-training."""
        self._mask = mask
        self._weights.copy_(
            torch.as_tensor(mask.weights, dtype=torch.float32))
        logger.info(f"Mask updated: SoI={sorted(mask.soi_dims)} "
                    f"({mask.n_soi}/{mask.obs_dim} dims)")


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class IBDCallback(BaseCallback):
    """
    Periodically runs IBD probing and updates the agent's causal mask.

    Args:
        probe: IBDProbe instance (should use a *separate* env instance)
        update_interval: run IBD every this many timesteps
        start_after: don't run IBD before this many timesteps
            (let the policy learn a bit first)
        use_learned_policy: if True, use the current SB3 policy as
            the probe policy; otherwise use random
        ground_truth_soi: if provided, log boundary accuracy
        mask_ema: EMA coefficient for blending new mask weights with
            old (0.0 = ignore new, 1.0 = hard replace).  Default 0.5
            converges in ~3 rounds while damping oscillations.
        mask_floor: minimum weight for non-SoI dims (default 0.05)
        vote_threshold: a dim must be significant in more than this
            fraction of non-empty rounds to enter the stable SoI
            (default 0.5).  Eliminates random false-positive
            distractors that pass the test in only 1-2 rounds.
        min_vote_rounds: number of non-empty IBD rounds before voting
            kicks in (default 3).  In early rounds the raw mask is
            used directly to avoid dropping true dims that haven't
            accumulated enough votes yet.
        freeze_after: freeze mask (stop probing) after the voted SoI
            has been identical for this many consecutive rounds.
            Set to 0 to disable freezing.  Default 3.
        confirm_after: a dim is "confirmed" (skipped in future CIM
            tests) after being significant in this many consecutive
            rounds.  Confirmed dims get synthetic p=0 entries,
            saving ~60-80% of permutation test time once most dims
            are confirmed.  Set to 0 to disable.  Default 3.
    """

    def __init__(self, probe: IBDProbe,
                 update_interval: int = 20_000,
                 start_after: int = 5_000,
                 use_learned_policy: bool = True,
                 ground_truth_soi: Optional[set] = None,
                 mask_ema: float = 0.5,
                 mask_floor: float = 0.05,
                 vote_threshold: float = 0.5,
                 min_vote_rounds: int = 3,
                 freeze_after: int = 3,
                 confirm_after: int = 3,
                 verbose: int = 1):
        super().__init__(verbose)
        self.probe = probe
        self.update_interval = update_interval
        self.start_after = start_after
        self.use_learned_policy = use_learned_policy
        self.ground_truth_soi = ground_truth_soi
        self.mask_ema = mask_ema
        self.mask_floor = mask_floor
        self.vote_threshold = vote_threshold
        self.min_vote_rounds = min_vote_rounds
        self.freeze_after = freeze_after
        self.confirm_after = confirm_after
        self.history: list = []
        self._prev_weights: Optional[np.ndarray] = None
        # Voting accumulators: count how many rounds each dim was significant
        self._sig_counts: Optional[np.ndarray] = None
        self._total_rounds: int = 0
        # Freeze tracking
        self._frozen: bool = False
        self._prev_voted_soi: Optional[set] = None
        self._stable_count: int = 0
        # Confirmed-dim tracking: consecutive significant rounds per dim
        self._consec_sig: Optional[np.ndarray] = None
        self._confirmed_dims: set = set()

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_after:
            return True
        if self.num_timesteps % self.update_interval != 0:
            return True

        # ── If frozen, skip probing entirely ──────────────────────────
        if self._frozen:
            return True

        # Build probe policy
        policy = None
        if self.use_learned_policy:
            def policy(obs):
                with torch.no_grad():
                    a, _ = self.model.predict(obs, deterministic=False)
                return a

        # Run IBD (pass confirmed dims to skip expensive testing)
        skip = self._confirmed_dims if self.confirm_after > 0 else None
        mask = self.probe.discover(policy=policy, skip_dims=skip)

        # Lazy-init voting and confirmation arrays
        if self._sig_counts is None:
            self._sig_counts = np.zeros(mask.obs_dim, dtype=np.float64)
        if self._consec_sig is None:
            self._consec_sig = np.zeros(mask.obs_dim, dtype=np.int32)

        # Safety valve: skip empty SoI rounds entirely (don't count them)
        if mask.n_soi == 0:
            logger.warning(f"IBD @ step {self.num_timesteps}: SoI is empty "
                           f"— skipping (not counted in vote)")
            record = {"timestep": self.num_timesteps,
                      "soi": [], "n_soi": 0, "skipped": True}
            if self.ground_truth_soi is not None:
                metrics = mask.evaluate(self.ground_truth_soi)
                record.update(metrics)
            self.history.append(record)
            return True

        # ── Accumulate votes ──────────────────────────────────────────
        self._total_rounds += 1
        raw_soi = set(mask.soi_dims)  # save before any mutation
        for d in raw_soi:
            self._sig_counts[d] += 1

        # ── Update consecutive-significance counters ──────────────────
        if self.confirm_after > 0:
            for d in range(mask.obs_dim):
                if d in raw_soi or d in self._confirmed_dims:
                    self._consec_sig[d] += 1
                else:
                    self._consec_sig[d] = 0
                # Promote to confirmed
                if (d not in self._confirmed_dims
                        and self._consec_sig[d] >= self.confirm_after):
                    self._confirmed_dims.add(d)

            n_conf = len(self._confirmed_dims)
            if n_conf > 0:
                logger.info(f"  Confirmed dims: {n_conf}/{mask.obs_dim} "
                            f"(skipped in future CIM tests)")

        # Compute stable SoI via voting (only after enough rounds)
        if self._total_rounds < self.min_vote_rounds:
            # Too few rounds for reliable voting — trust raw mask
            stable_soi = raw_soi
        else:
            vote_rate = self._sig_counts / self._total_rounds
            stable_soi = {d for d in range(mask.obs_dim)
                          if vote_rate[d] > self.vote_threshold}
            # Fallback if voting produces empty set
            if not stable_soi:
                stable_soi = raw_soi

        # ── Check freeze condition ────────────────────────────────────
        if self.freeze_after > 0 and self._total_rounds >= self.min_vote_rounds:
            if (self._prev_voted_soi is not None
                    and stable_soi == self._prev_voted_soi):
                self._stable_count += 1
            else:
                self._stable_count = 1  # reset (current round counts as 1)
            self._prev_voted_soi = stable_soi.copy()

            if self._stable_count >= self.freeze_after:
                self._frozen = True
                logger.info(
                    f"IBD FROZEN @ step {self.num_timesteps}: "
                    f"SoI stable for {self._stable_count} consecutive rounds. "
                    f"No further probing.")
        else:
            self._prev_voted_soi = stable_soi.copy()

        # Build voted mask weights: 1.0 for stable SoI, floor otherwise
        new_w = np.full(mask.obs_dim, self.mask_floor, dtype=np.float32)
        for d in stable_soi:
            new_w[d] = 1.0

        # EMA blend for smooth transitions
        if self._prev_weights is not None:
            blended = (self.mask_ema * new_w
                       + (1 - self.mask_ema) * self._prev_weights)
        else:
            blended = new_w
        self._prev_weights = blended.copy()

        # Update mask object with voted SoI + blended weights
        mask.soi_dims = stable_soi
        mask.weights = blended.astype(np.float32)

        # ── Update the feature extractor(s) ───────────────────────────
        updated = False
        for net_name in ("actor", "critic", "critic_target"):
            net = getattr(self.model.policy, net_name, None)
            if net is None:
                continue
            fe = getattr(net, "features_extractor", None)
            if hasattr(fe, "update_mask"):
                fe.update_mask(mask)
                updated = True
        if not updated:
            fe = getattr(self.model.policy, "features_extractor", None)
            if hasattr(fe, "update_mask"):
                fe.update_mask(mask)
                updated = True
        if not updated:
            logger.warning("No CausalMaskExtractor found — mask not applied")

        # ── Log ───────────────────────────────────────────────────────
        record = {"timestep": self.num_timesteps,
                  "raw_soi": sorted(raw_soi),
                  "voted_soi": sorted(stable_soi),
                  "n_soi": len(stable_soi),
                  "total_rounds": self._total_rounds,
                  "n_confirmed": len(self._confirmed_dims),
                  "frozen": self._frozen}

        if self.ground_truth_soi is not None:
            metrics = mask.evaluate(self.ground_truth_soi)
            record.update(metrics)
            logger.info(
                f"IBD @ step {self.num_timesteps}: "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} | "
                f"voted SoI={sorted(stable_soi)} "
                f"(round {self._total_rounds}"
                f", confirmed={len(self._confirmed_dims)}"
                f"{', FROZEN' if self._frozen else ''})")

        self.history.append(record)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def make_ibd_sac(
    env: gym.Env,
    probe_env: Optional[gym.Env] = None,
    ibd_kwargs: Optional[dict] = None,
    sac_kwargs: Optional[dict] = None,
    update_interval: int = 20_000,
    ground_truth_soi: Optional[set] = None,
):
    """
    Create a SAC agent with IBD integration.

    Args:
        env: training environment
        probe_env: separate env instance for IBD probing
            (if None, uses ``env`` — works but may reset training state)
        ibd_kwargs: kwargs for IBDProbe
        sac_kwargs: kwargs for SAC
        update_interval: IBD probe interval in timesteps
        ground_truth_soi: for logging boundary accuracy

    Returns:
        (model, callback) — call model.learn(callbacks=[callback])
    """
    if probe_env is None:
        probe_env = env

    probe = IBDProbe(probe_env, **(ibd_kwargs or {}))
    callback = IBDCallback(
        probe, update_interval=update_interval,
        ground_truth_soi=ground_truth_soi)

    policy_kwargs = {
        "features_extractor_class": CausalMaskExtractor,
        "features_extractor_kwargs": {"initial_mask": None},
    }
    sac_kw = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    sac_kw.update(sac_kwargs or {})
    model = SAC(**sac_kw)

    return model, callback