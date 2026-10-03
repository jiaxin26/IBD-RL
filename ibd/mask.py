"""
Causal Mask
============

Stores the result of IBD: which observation dims are causally
influenced by actions (Sphere of Influence, SoI) and which are not.

Supports soft and hard masking, gymnasium wrapping, and serialisation.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    import torch
except ImportError:
    torch = None


@dataclass
class CausalMask:
    """Binary / soft mask over observation dimensions."""

    obs_dim: int
    soi_dims: Set[int]                         # dims classified as SoI
    weights: np.ndarray                        # (obs_dim,) soft weights in [0, 1]
    p_values: Optional[np.ndarray] = None      # (obs_dim,) adjusted p-values
    effect_sizes: Optional[np.ndarray] = None  # (obs_dim,) |Cohen's d|
    round_idx: int = 0

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def ones(cls, obs_dim: int) -> "CausalMask":
        """No masking — all dims pass through."""
        return cls(obs_dim=obs_dim,
                   soi_dims=set(range(obs_dim)),
                   weights=np.ones(obs_dim, dtype=np.float32))

    @classmethod
    def from_soi(cls, soi_dims: Set[int], obs_dim: int,
                 mode: str = "soft",
                 p_values: Optional[np.ndarray] = None,
                 effect_sizes: Optional[np.ndarray] = None,
                 round_idx: int = 0,
                 floor: float = 0.05) -> "CausalMask":
        """Build a mask from a set of SoI dimensions.

        Args:
            floor: minimum weight for non-SoI dims in soft mode.
                Set to 0.0 for hard-zero masking, or ~0.05 to retain
                a faint signal from non-causal dims (useful when those
                dims carry information correlated with action disturbances
                or reward structure, even if not causally influenced).
        """
        weights = np.zeros(obs_dim, dtype=np.float32)
        if mode == "hard":
            for d in soi_dims:
                if d < obs_dim:
                    weights[d] = 1.0
        elif mode == "soft":
            for d in range(obs_dim):
                if d in soi_dims:
                    weights[d] = 1.0
                else:
                    weights[d] = floor
        else:
            raise ValueError(f"Unknown mask mode: {mode}")

        return cls(obs_dim=obs_dim, soi_dims=set(soi_dims),
                   weights=weights, p_values=p_values,
                   effect_sizes=effect_sizes, round_idx=round_idx)

    # ── properties ────────────────────────────────────────────────────

    @property
    def illusory_dims(self) -> Set[int]:
        return set(range(self.obs_dim)) - self.soi_dims

    @property
    def n_soi(self) -> int:
        return len(self.soi_dims)

    # ── torch interop ─────────────────────────────────────────────────

    def weights_tensor(self, device: str = "cpu"):
        if torch is None:
            raise ImportError("PyTorch required")
        return torch.as_tensor(self.weights, dtype=torch.float32,
                               device=device)

    # ── gymnasium wrapper ─────────────────────────────────────────────

    def wrap(self, env):
        """Return a gymnasium env whose observations are soft-masked."""
        from ibd.wrappers import MaskedObsWrapper
        return MaskedObsWrapper(env, self)

    # ── evaluation against ground truth ───────────────────────────────

    def evaluate(self, true_soi: Set[int]) -> Dict[str, float]:
        """Compute P / R / F1 against ground-truth SoI."""
        tp = len(self.soi_dims & true_soi)
        fp = len(self.soi_dims - true_soi)
        fn = len(true_soi - self.soi_dims)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"obs_dim": self.obs_dim,
                "soi_dims": sorted(self.soi_dims),
                "weights": self.weights.tolist(),
                "round_idx": self.round_idx}

    @classmethod
    def from_dict(cls, d: dict) -> "CausalMask":
        return cls(obs_dim=d["obs_dim"],
                   soi_dims=set(d["soi_dims"]),
                   weights=np.array(d["weights"], dtype=np.float32),
                   round_idx=d.get("round_idx", 0))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CausalMask":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        return (f"CausalMask(obs_dim={self.obs_dim}, "
                f"soi={sorted(self.soi_dims)}, "
                f"illusory={sorted(self.illusory_dims)})")