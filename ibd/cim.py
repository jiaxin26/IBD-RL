"""
Causal Influence Matrix (CIM) Estimator
=========================================

Vectorised permutation-test estimator for detecting which observation
dimensions are causally affected by which action dimensions.

Given *baseline* trajectories (normal policy) and *intervention*
trajectories (one action dim randomised), the CIM computes per-dim
test statistics and adjusted p-values.

Key design: **one summary per trajectory** (trajectory-level mean of
non-overlapping h-step differences) so that samples fed to the
permutation test are genuinely independent.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CIMEntry:
    """Single (action, state, horizon) test result."""
    action_dim: int
    state_dim: int
    horizon: int
    ate: float = 0.0
    ate_std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0
    p_value_raw: float = 1.0
    significant: bool = False
    cohens_d: float = 0.0
    n_baseline: int = 0
    n_intervened: int = 0

    @property
    def effect_magnitude(self) -> str:
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"


@dataclass
class CIMResult:
    """Full CIM estimation result across all (action, state, horizon) triples."""
    entries: Dict[Tuple[int, int, int], CIMEntry]
    horizons: List[int]
    n_action_dims: int
    n_state_dims: int
    alpha: float

    def get_estimated_soi(self) -> Set[int]:
        """State dims significant at ANY (action, horizon) combination."""
        soi = set()
        for (a, s, h), e in self.entries.items():
            if e.significant:
                soi.add(s)
        return soi

    def get_effect_sizes(self, n_state_dims: int) -> np.ndarray:
        """Max |d| across actions and horizons for each state dim."""
        es = np.zeros(n_state_dims)
        for (a, s, h), e in self.entries.items():
            if s < n_state_dims:
                es[s] = max(es[s], abs(e.cohens_d))
        return es

    def get_min_pvalues(self, n_state_dims: int) -> np.ndarray:
        """Min adjusted p-value across actions and horizons for each dim."""
        pv = np.ones(n_state_dims)
        for (a, s, h), e in self.entries.items():
            if s < n_state_dims:
                pv[s] = min(pv[s], e.p_value)
        return pv


# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CIMEstimator:
    """
    Vectorised CIM estimator with trajectory-level summaries.

    Key correctness property: each trajectory contributes exactly ONE
    sample (mean of non-overlapping h-step diffs), so permutation-test
    exchangeability is respected.
    """

    def __init__(self, n_bootstrap: int = 500, n_permutations: int = 1000,
                 alpha: float = 0.05,
                 correction: str = "benjamini-hochberg",
                 seed: int = 42, min_samples: int = 10,
                 adaptive_permutations: bool = True,
                 adaptive_burnin: int = 200):
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.correction = correction
        self.rng = np.random.RandomState(seed)
        self.min_samples = min_samples
        self.adaptive = adaptive_permutations
        self.adaptive_burnin = min(adaptive_burnin, n_permutations)

    # ── trajectory-level feature extraction ────────────────────────────

    @staticmethod
    def _extract(trajs: np.ndarray, sd: int, h: int,
                 absolute: bool = False) -> Optional[np.ndarray]:
        """One summary value per trajectory: mean of non-overlapping h-step diffs."""
        if trajs.ndim == 2:
            trajs = trajs[np.newaxis]
        N, T, nd = trajs.shape
        if sd >= nd or h >= T:
            return None
        t_start = np.arange(0, T - h, h)          # non-overlapping windows
        t_end = t_start + h
        diffs = trajs[:, t_end, sd] - trajs[:, t_start, sd]   # (N, n_windows)
        if absolute:
            diffs = np.abs(diffs)
        out = diffs.mean(axis=1)                               # (N,)
        return out if len(out) >= 10 else None

    # ── vectorised permutation test ───────────────────────────────────

    def _permutation_test(self, y0: np.ndarray, y1: np.ndarray) -> float:
        n0 = len(y0)
        combined = np.concatenate([y0, y1])
        n = len(combined)
        obs = abs(np.mean(y0) - np.mean(y1))
        if obs < 1e-15:
            return 1.0

        def _batch(n_perms):
            idx = np.argsort(self.rng.random((n_perms, n)), axis=1)
            shuffled = combined[idx]
            m0 = shuffled[:, :n0].mean(axis=1)
            m1 = shuffled[:, n0:].mean(axis=1)
            return int(np.sum(np.abs(m0 - m1) >= obs))

        # Adaptive: quick burn-in check
        if self.adaptive and self.adaptive_burnin < self.n_permutations:
            cnt = _batch(self.adaptive_burnin)
            if cnt == 0:
                return 1.0 / (self.adaptive_burnin + 1)
            if cnt >= self.adaptive_burnin * 0.8:
                return (cnt + 1) / (self.adaptive_burnin + 1)
            remaining = self.n_permutations - self.adaptive_burnin
            cnt += _batch(remaining)
            return (cnt + 1) / (self.n_permutations + 1)

        cnt = _batch(self.n_permutations)
        return (cnt + 1) / (self.n_permutations + 1)

    # ── bootstrap CI ──────────────────────────────────────────────────

    def _bootstrap_ci(self, y0: np.ndarray, y1: np.ndarray):
        n0, n1 = len(y0), len(y1)
        idx0 = self.rng.randint(0, n0, size=(self.n_bootstrap, n0))
        idx1 = self.rng.randint(0, n1, size=(self.n_bootstrap, n1))
        boot_ate = y0[idx0].mean(axis=1) - y1[idx1].mean(axis=1)
        lo = self.alpha / 2 * 100
        hi = (1 - self.alpha / 2) * 100
        return float(np.percentile(boot_ate, lo)), float(np.percentile(boot_ate, hi))

    # ── single entry ──────────────────────────────────────────────────

    def _estimate_single(self, y0, y1, ad, sd, h) -> Optional[CIMEntry]:
        if y0 is None or y1 is None:
            return None
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        y1 = np.asarray(y1, dtype=np.float64).ravel()
        n0, n1 = len(y0), len(y1)
        if n0 < self.min_samples or n1 < self.min_samples:
            return None

        m0, m1 = float(np.mean(y0)), float(np.mean(y1))
        ate = m0 - m1
        v0 = float(np.var(y0, ddof=1))
        v1 = float(np.var(y1, ddof=1))
        se = np.sqrt(v0 / n0 + v1 / n1)
        ci_lo, ci_hi = self._bootstrap_ci(y0, y1)
        pval = self._permutation_test(y0, y1)
        pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
        d = ate / pooled if pooled > 1e-12 else 0.0
        d *= 1 - 3 / (4 * (n0 + n1) - 9)          # Hedges correction

        return CIMEntry(action_dim=ad, state_dim=sd, horizon=h,
                        ate=ate, ate_std=se, ci_lower=ci_lo, ci_upper=ci_hi,
                        p_value=pval, p_value_raw=pval, cohens_d=d,
                        n_baseline=n0, n_intervened=n1)

    # ── full matrix ───────────────────────────────────────────────────

    def estimate_matrix(
        self,
        baseline_trajs: np.ndarray,
        intervened_trajs: Dict[int, np.ndarray],
        state_dims: List[int],
        horizons: List[int],
        action_dims: Optional[List[int]] = None,
        skip_dims: Optional[Set[int]] = None,
    ) -> CIMResult:
        """
        Estimate the full Causal Influence Matrix.

        Args:
            baseline_trajs:  (N_b, T, D) array of baseline trajectories
            intervened_trajs: {action_dim: (N_i, T, D)} intervention trajs
            state_dims:       which observation dims to test
            horizons:         list of h values (e.g. [1, 5, 10])
            action_dims:      which action dims (default: all keys)
            skip_dims:        state dims to skip testing (already confirmed
                              as causal); they are marked significant with
                              synthetic p_value=0 and large effect size.

        Returns:
            CIMResult with per-entry test statistics and adjusted p-values.
        """
        if action_dims is None:
            action_dims = sorted(intervened_trajs.keys())
        if skip_dims is None:
            skip_dims = set()

        # Dims that actually need testing
        test_dims = [sd for sd in state_dims if sd not in skip_dims]
        n_skipped = len(state_dims) - len(test_dims)
        if n_skipped > 0:
            logger.info(f"CIM: skipping {n_skipped} confirmed dims, "
                        f"testing {len(test_dims)}")

        # Collect tested and synthetic entries separately
        tested_entries: List[CIMEntry] = []
        synthetic_entries: List[CIMEntry] = []
        entries_dict: Dict[Tuple[int, int, int], CIMEntry] = {}

        for ad in action_dims:
            it = intervened_trajs.get(ad)
            if it is None:
                continue

            # Inject synthetic entries for confirmed (skipped) dims
            for sd in skip_dims:
                if sd in state_dims:
                    for h in horizons:
                        entry = CIMEntry(
                            action_dim=ad, state_dim=sd, horizon=h,
                            ate=1.0, ate_std=0.0,
                            ci_lower=0.5, ci_upper=1.5,
                            p_value=0.0, p_value_raw=0.0,
                            significant=True, cohens_d=5.0,
                            n_baseline=0, n_intervened=0)
                        synthetic_entries.append(entry)
                        entries_dict[(ad, sd, h)] = entry

            # Test remaining dims
            for sd in test_dims:
                for h in horizons:
                    # Signed and absolute variants
                    ob = self._extract(baseline_trajs, sd, h, absolute=False)
                    oi = self._extract(it, sd, h, absolute=False)
                    oba = self._extract(baseline_trajs, sd, h, absolute=True)
                    oia = self._extract(it, sd, h, absolute=True)

                    es = self._estimate_single(ob, oi, ad, sd, h)
                    ea = self._estimate_single(oba, oia, ad, sd, h)

                    # Pick the more significant of the two
                    best = None
                    if es and ea:
                        best = ea if ea.p_value_raw < es.p_value_raw else es
                        best.p_value_raw = min(best.p_value_raw * 2.0, 1.0)
                        best.p_value = best.p_value_raw
                    elif es:
                        best = es
                    elif ea:
                        best = ea

                    if best is not None:
                        tested_entries.append(best)
                        entries_dict[(ad, sd, h)] = best

        # Apply multiple-testing correction PER ACTION DIM.
        #
        # Each action dim is an independent interventional experiment.
        # Global correction across all (action × state × horizon)
        # entries is overly conservative: with 74 obs × 6 actions ×
        # 3 horizons = 1332 tests, BH rank-1 threshold is 3.75e-5,
        # but permutation tests with N=5000 can only reach p ≈ 2e-4.
        #
        # Per-action correction: m = n_state_dims × n_horizons.
        # For walker with 1 horizon: m=74, rank-1 threshold = 6.8e-4,
        # which is achievable with 5000 permutations.
        per_action: Dict[int, List[CIMEntry]] = defaultdict(list)
        for e in tested_entries:
            per_action[e.action_dim].append(e)
        for ad, entries in per_action.items():
            self._apply_correction(entries)

        return CIMResult(entries=entries_dict, horizons=horizons,
                         n_action_dims=len(action_dims),
                         n_state_dims=len(state_dims),
                         alpha=self.alpha)

    # ── multiple-testing correction ───────────────────────────────────

    def _apply_correction(self, entries: List[CIMEntry]):
        if not entries:
            return
        raw = np.array([e.p_value_raw for e in entries])
        m = len(raw)

        if self.correction == "none":
            for e in entries:
                e.significant = e.p_value_raw < self.alpha

        elif self.correction == "bonferroni":
            for e in entries:
                e.p_value = min(e.p_value_raw * m, 1.0)
                e.significant = e.p_value_raw < self.alpha / m

        elif self.correction == "benjamini-hochberg":
            idx = np.argsort(raw)
            sp = raw[idx]
            sig = np.zeros(m, dtype=bool)
            max_k = -1
            for k in range(m):
                if sp[k] <= (k + 1) / m * self.alpha:
                    max_k = k
            if max_k >= 0:
                sig[idx[:max_k + 1]] = True
            # Adjusted p-values
            adj = np.ones(m)
            for k in range(m - 1, -1, -1):
                i = idx[k]
                adj[i] = min(sp[k] * m / (k + 1), 1.0)
                if k < m - 1:
                    adj[i] = min(adj[i], adj[idx[k + 1]])
            for i, e in enumerate(entries):
                e.p_value = adj[i]
                e.significant = sig[i]
        else:
            raise ValueError(f"Unknown correction: {self.correction}")