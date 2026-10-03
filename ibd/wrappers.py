"""
Gymnasium Wrappers for IBD
===========================

``MaskedObsWrapper``  — applies a CausalMask to observations,
zeroing-out or down-weighting illusory dimensions.

``DimSelectWrapper``  — physically removes non-SoI dimensions,
reducing the observation space dimensionality.  This gives the
downstream policy a smaller input, which is more sample-efficient
than soft masking for high-distractor-ratio settings.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ibd.mask import CausalMask


class MaskedObsWrapper(gym.ObservationWrapper):
    """
    Multiply observations element-wise by the causal mask weights.

    Observation space shape is preserved (same dim), so downstream
    networks don't need to be rebuilt when the mask updates.
    """

    def __init__(self, env: gym.Env, mask: CausalMask):
        super().__init__(env)
        self._mask = mask

    @property
    def mask(self) -> CausalMask:
        return self._mask

    @mask.setter
    def mask(self, new_mask: CausalMask):
        self._mask = new_mask

    def observation(self, obs: np.ndarray) -> np.ndarray:
        flat = obs.flatten()
        if len(flat) == len(self._mask.weights):
            return (flat * self._mask.weights).reshape(obs.shape)
        return obs   # dimension mismatch → pass through unchanged


class DimSelectWrapper(gym.ObservationWrapper):
    """
    Select a subset of observation dimensions, physically reducing
    the observation space.

    Unlike MaskedObsWrapper (which preserves shape and soft-masks),
    this wrapper outputs only the selected dims.  A 74-dim obs with
    24 selected dims becomes a 24-dim obs.  This means the downstream
    MLP has fewer input neurons, reducing sample complexity.

    Args:
        env: gymnasium environment
        selected_dims: 1-D integer array of dimension indices to keep,
            e.g. np.array([0, 1, 2, ..., 23])
    """

    def __init__(self, env: gym.Env, selected_dims: np.ndarray):
        super().__init__(env)
        self._dims = np.asarray(selected_dims, dtype=np.intp)
        assert self._dims.ndim == 1

        # Build new observation space
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), \
            f"DimSelectWrapper requires Box obs space, got {type(old_space)}"
        low = old_space.low.flatten()[self._dims]
        high = old_space.high.flatten()[self._dims]
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=old_space.dtype)

    @property
    def selected_dims(self) -> np.ndarray:
        return self._dims.copy()

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()[self._dims]
