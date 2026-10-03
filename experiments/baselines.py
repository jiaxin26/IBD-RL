"""
Baseline Feature-Selection Methods
====================================

Methods that IBD must be compared against for a convincing paper.

Each baseline implements ``discover(env, ...) -> CausalMask`` so
the experiment runner can treat them uniformly.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Set
import gymnasium as gym

from ibd.mask import CausalMask


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE
# ═══════════════════════════════════════════════════════════════════════════════

class OracleMask:
    """Uses ground-truth labels.  Upper bound on any method."""

    def __init__(self, true_soi: Set[int], obs_dim: int,
                 mode: str = "soft"):
        self.true_soi = true_soi
        self.obs_dim = obs_dim
        self.mode = mode

    def discover(self, **kwargs) -> CausalMask:
        return CausalMask.from_soi(self.true_soi, self.obs_dim,
                                   mode=self.mode)


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM MASK
# ═══════════════════════════════════════════════════════════════════════════════

class RandomMask:
    """Randomly selects k dims as SoI.  Calibration baseline."""

    def __init__(self, obs_dim: int, n_select: int, seed: int = 42,
                 mode: str = "soft"):
        self.obs_dim = obs_dim
        self.n_select = n_select
        self.rng = np.random.RandomState(seed)
        self.mode = mode

    def discover(self, **kwargs) -> CausalMask:
        dims = set(self.rng.choice(self.obs_dim, self.n_select,
                                   replace=False).tolist())
        return CausalMask.from_soi(dims, self.obs_dim, mode=self.mode)


# ═══════════════════════════════════════════════════════════════════════════════
# MUTUAL INFORMATION (observational)
# ═══════════════════════════════════════════════════════════════════════════════

class MutualInfoSelector:
    """
    Ranks obs dims by mutual information with actions (observational).

    Vulnerable to confounders: confounded dims show high MI with
    actions even though no causal link exists.
    """

    def __init__(self, env: gym.Env, n_episodes: int = 100,
                 episode_length: int = 200,
                 n_select: Optional[int] = None,
                 mode: str = "soft", seed: int = 42):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.n_select = n_select
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

    def discover(self, policy=None, **kwargs) -> CausalMask:
        if policy is None:
            low = self.env.action_space.low
            high = self.env.action_space.high
            policy = lambda obs: self.rng.uniform(low, high)

        # Collect observational data
        all_obs, all_acts = [], []
        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            for t in range(self.episode_length):
                a = policy(obs)
                all_obs.append(obs.flatten())
                all_acts.append(np.asarray(a).flatten())
                step = self.env.step(a)
                obs = step[0]
                done = step[2] if len(step) == 4 else (step[2] or step[3])
                if done:
                    obs, _ = self.env.reset()

        O = np.array(all_obs)   # (N, obs_dim)
        A = np.array(all_acts)  # (N, act_dim)

        # Estimate MI via binned histogram (simple but effective)
        scores = np.zeros(self.obs_dim)
        n_bins = 20
        for d in range(self.obs_dim):
            mi = 0.0
            for a in range(self.action_dim):
                mi += self._binned_mi(O[:, d], A[:, a], n_bins)
            scores[d] = mi / self.action_dim

        # Select top-k
        n_sel = self.n_select or max(1, self.obs_dim // 3)
        threshold = np.sort(scores)[::-1][min(n_sel - 1, len(scores) - 1)]
        soi = {d for d in range(self.obs_dim) if scores[d] >= threshold}

        es = scores / max(scores.max(), 1e-8)   # normalised as effect_sizes
        return CausalMask.from_soi(soi, self.obs_dim, mode=self.mode,
                                   effect_sizes=es)

    @staticmethod
    def _binned_mi(x, y, n_bins=20):
        """Binned MI estimator (fast, biased but consistent for ranking)."""
        eps = 1e-12
        x_bins = np.digitize(x, np.linspace(x.min() - eps, x.max() + eps, n_bins + 1))
        y_bins = np.digitize(y, np.linspace(y.min() - eps, y.max() + eps, n_bins + 1))
        pxy, _, _ = np.histogram2d(x_bins, y_bins,
                                   bins=[n_bins, n_bins],
                                   density=True)
        pxy = pxy / (pxy.sum() + eps)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        # MI = sum pxy * log(pxy / px*py)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > eps and px[i] > eps and py[j] > eps:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
        return max(mi, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANCE-BASED SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

class VarianceSelector:
    """
    Select dims with highest variance in action-conditioned next-state
    prediction residuals.  A simple model-based baseline.
    """

    def __init__(self, env: gym.Env, n_episodes: int = 100,
                 episode_length: int = 200,
                 n_select: Optional[int] = None,
                 mode: str = "soft", seed: int = 42):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.n_select = n_select
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.obs_dim = int(np.prod(env.observation_space.shape))

    def discover(self, policy=None, **kwargs) -> CausalMask:
        if policy is None:
            low = self.env.action_space.low
            high = self.env.action_space.high
            policy = lambda obs: self.rng.uniform(low, high)

        # Collect (s, a, s') tuples
        deltas = []
        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            for t in range(self.episode_length):
                a = policy(obs)
                step = self.env.step(a)
                next_obs = step[0]
                deltas.append(next_obs.flatten() - obs.flatten())
                obs = next_obs
                done = step[2] if len(step) == 4 else (step[2] or step[3])
                if done:
                    obs, _ = self.env.reset()

        D = np.array(deltas)     # (N, obs_dim)
        var = D.var(axis=0)      # high variance → action-dependent?

        n_sel = self.n_select or max(1, self.obs_dim // 3)
        threshold = np.sort(var)[::-1][min(n_sel - 1, len(var) - 1)]
        soi = {d for d in range(self.obs_dim) if var[d] >= threshold}

        es = var / max(var.max(), 1e-8)
        return CausalMask.from_soi(soi, self.obs_dim, mode=self.mode,
                                   effect_sizes=es)


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL MI (learned forward model — stronger observational baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalMISelector:
    """
    Ranks dims by how much a learned forward model benefits from
    action information when predicting the next value of each dim.

    For each observation dimension i, trains two small MLPs:
        f_sa:  (s, a) → s'_i       (action-conditioned predictor)
        f_s :  (s)    → s'_i       (state-only baseline)
    and scores dim i by  R²(f_sa) - R²(f_s).

    This is substantially stronger than raw MI or variance because it
    conditions on the current state, filtering out much of the marginal
    correlation.  However, it is still **observational**: a confounded
    distractor that can be predicted from (s, a) better than from s
    alone will score highly even without a causal link from a to the
    distractor.  Mimicking distractors whose dynamics correlate with
    the agent's proprioceptive state create exactly this pattern.

    IBD is immune because the do-operator severs confounding paths;
    this baseline is not.
    """

    def __init__(self, env: gym.Env,
                 n_episodes: int = 200,
                 episode_length: int = 200,
                 n_select: Optional[int] = None,
                 hidden: int = 64,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 2048,
                 mode: str = "soft",
                 seed: int = 42):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.n_select = n_select
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

    def discover(self, policy=None, **kwargs) -> CausalMask:
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.seed)
        if policy is None:
            low, high = self.env.action_space.low, self.env.action_space.high
            policy = lambda obs: rng.uniform(low, high)

        # ── Collect (s, a, s') transitions ────────────────────────────
        states, actions, next_states = [], [], []
        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            for t in range(self.episode_length):
                a = policy(obs)
                step = self.env.step(a)
                ns = step[0]
                states.append(obs.flatten())
                actions.append(np.asarray(a).flatten())
                next_states.append(ns.flatten())
                obs = ns
                done = step[2] if len(step) == 4 else (step[2] or step[3])
                if done:
                    obs, _ = self.env.reset()

        S = np.array(states, dtype=np.float32)      # (N, obs_dim)
        A = np.array(actions, dtype=np.float32)      # (N, act_dim)
        NS = np.array(next_states, dtype=np.float32) # (N, obs_dim)

        device = "cpu"
        S_t = torch.from_numpy(S).to(device)
        A_t = torch.from_numpy(A).to(device)
        SA_t = torch.cat([S_t, A_t], dim=1)

        # ── Per-dim: compare R²(s,a→s'_i) vs R²(s→s'_i) ─────────────
        scores = np.zeros(self.obs_dim)

        for dim in range(self.obs_dim):
            y_t = torch.from_numpy(NS[:, dim]).to(device)

            r2_sa = self._fit_and_eval(SA_t, y_t, SA_t.shape[1])
            r2_s = self._fit_and_eval(S_t, y_t, S_t.shape[1])

            # Score = how much action info helps prediction
            scores[dim] = max(float(r2_sa - r2_s), 0.0)

        # ── Select top-k ─────────────────────────────────────────────
        n_sel = self.n_select or max(1, self.obs_dim // 3)
        threshold = np.sort(scores)[::-1][min(n_sel - 1, len(scores) - 1)]
        soi = {d for d in range(self.obs_dim) if scores[d] >= threshold}

        es = scores / max(scores.max(), 1e-8)
        return CausalMask.from_soi(soi, self.obs_dim, mode=self.mode,
                                   effect_sizes=es)

    def _fit_and_eval(self, X: 'torch.Tensor', y: 'torch.Tensor',
                      in_dim: int) -> float:
        """Train a small MLP and return its R² on the full dataset."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        n = len(X)

        model.train()
        for _ in range(self.epochs):
            idx = torch.randperm(n)[:min(self.batch_size, n)]
            pred = model(X[idx]).squeeze(-1)
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_all = model(X).squeeze(-1)
            ss_res = ((pred_all - y) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        return r2


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT ATTRIBUTION (learned dynamics model — strongest observational baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class GradientAttributionSelector:
    """
    Ranks dims by gradient sensitivity of a learned dynamics model
    to the action input.

    Trains a joint forward model  f(s, a) → s'  and scores each
    observation dimension i by:

        score_i = E_data[ || ∂f_i(s, a) / ∂a ||_1 ]

    This directly measures "how sensitive is the predicted next-state
    dimension to the action input" — arguably the most natural
    model-based approach to identifying action-influenced dimensions.

    However, it remains **observational**: if a mimicking distractor
    co-varies with true state (which co-varies with actions), the
    learned model develops spurious gradient paths  a → s → distractor,
    and the distractor receives a non-zero gradient score even though
    do(a) has no causal effect on it.  The model captures correlational
    structure, not causal structure.
    """

    def __init__(self, env: gym.Env,
                 n_episodes: int = 200,
                 episode_length: int = 200,
                 n_select: Optional[int] = None,
                 hidden: int = 128,
                 epochs: int = 80,
                 lr: float = 1e-3,
                 batch_size: int = 2048,
                 mode: str = "soft",
                 seed: int = 42):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.n_select = n_select
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

    def discover(self, policy=None, **kwargs) -> CausalMask:
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.seed)
        if policy is None:
            low, high = self.env.action_space.low, self.env.action_space.high
            policy = lambda obs: rng.uniform(low, high)

        # ── Collect (s, a, s') transitions ────────────────────────────
        states, actions, next_states = [], [], []
        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            for t in range(self.episode_length):
                a = policy(obs)
                step = self.env.step(a)
                ns = step[0]
                states.append(obs.flatten())
                actions.append(np.asarray(a).flatten())
                next_states.append(ns.flatten())
                obs = ns
                done = step[2] if len(step) == 4 else (step[2] or step[3])
                if done:
                    obs, _ = self.env.reset()

        S = np.array(states, dtype=np.float32)
        A = np.array(actions, dtype=np.float32)
        NS = np.array(next_states, dtype=np.float32)

        device = "cpu"
        S_t = torch.from_numpy(S).to(device)
        A_t = torch.from_numpy(A).to(device)

        # ── Train a joint dynamics model f(s, a) → s' ────────────────
        in_dim = self.obs_dim + self.action_dim
        torch.manual_seed(self.seed)
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.obs_dim),
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        SA_t = torch.cat([S_t, A_t], dim=1)
        NS_t = torch.from_numpy(NS).to(device)

        n = len(SA_t)
        model.train()
        for epoch in range(self.epochs):
            idx = torch.randperm(n)[:min(self.batch_size, n)]
            pred = model(SA_t[idx])
            loss = ((pred - NS_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # ── Compute per-dim gradient attribution ──────────────────────
        # For each dim i, score_i = E[ |∂f_i/∂a| ] averaged over data.
        # We do a separate forward+backward per dim to avoid retain_graph
        # (which causes OOM when obs_dim is large).
        model.eval()
        scores = np.zeros(self.obs_dim)
        grad_batch = min(4096, n)
        n_batches = (n + grad_batch - 1) // grad_batch
        total_samples = 0

        for b in range(n_batches):
            start = b * grad_batch
            end = min(start + grad_batch, n)
            bs = end - start
            s_b = S_t[start:end].detach()
            a_base = A_t[start:end].detach()

            for dim in range(self.obs_dim):
                a_b = a_base.clone().requires_grad_(True)
                sa_b = torch.cat([s_b, a_b], dim=1)
                pred_dim = model(sa_b)[:, dim]      # (batch,)
                pred_dim.sum().backward()            # no retain_graph
                grad = a_b.grad.detach().abs().mean(dim=1)  # (batch,)
                scores[dim] += grad.sum().item()

            total_samples += bs

        scores /= total_samples

        # ── Select top-k ──────────────────────────────────────────────
        n_sel = self.n_select or max(1, self.obs_dim // 3)
        threshold = np.sort(scores)[::-1][min(n_sel - 1, len(scores) - 1)]
        soi = {d for d in range(self.obs_dim) if scores[d] >= threshold}

        es = scores / max(scores.max(), 1e-8)
        return CausalMask.from_soi(soi, self.obs_dim, mode=self.mode,
                                   effect_sizes=es)


# ═══════════════════════════════════════════════════════════════════════════════
# NO MASK (full state baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class NoMask:
    """Identity mask — all dims pass through.  Standard RL baseline."""

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim

    def discover(self, **kwargs) -> CausalMask:
        return CausalMask.ones(self.obs_dim)




# ═══════════════════════════════════════════════════════════════════════════════
# MULTISTEP INVERSE DYNAMICS (observational; Efroni et al. 2022 style)
# ═══════════════════════════════════════════════════════════════════════════════

class MultistepInverseDynamicsSelector:
    """
    Continuous-control adaptation of the multistep inverse dynamics
    principle from EX-BMDP (Efroni et al. 2022).

    Trains a network  g(o_t, o_{t+k}) -> a_t  on observational rollouts.
    Each observation dimension is scored by how much its *ablation*
    (zeroing it out at test time) degrades action prediction.  Dims
    whose values carry action information receive high scores; purely
    exogenous dims carry none and receive low scores.

    Relation to EX-BMDP.  The full EX-BMDP machinery (discrete latent
    state recovery, PAC guarantees, fast-mixing assumption) does not
    port directly to continuous state-action spaces; this selector
    isolates the multistep inverse dynamics signal that is the core
    empirical ingredient of that line and applies it at the
    observation-dimension level, matching our budget-controlled
    feature selection protocol (same n_select as MI / Var / Cond. MI).

    This baseline remains **observational**: it uses natural action
    variation in the data collection policy, not interventions.  It is
    therefore still vulnerable to confounded distractors whose dynamics
    co-vary with action-influenced state dimensions — in contrast to
    IBD, which severs such confounding via do(a = noise).
    """

    def __init__(
        self,
        env: gym.Env,
        n_episodes: int = 200,
        episode_length: int = 200,
        horizon_k: int = 3,
        n_select: Optional[int] = None,
        hidden: int = 128,
        epochs: int = 80,
        lr: float = 1e-3,
        batch_size: int = 2048,
        val_frac: float = 0.2,
        mode: str = "soft",
        seed: int = 42,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.horizon_k = horizon_k
        self.n_select = n_select
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.mode = mode
        self.seed = seed
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

    def discover(self, policy=None, **kwargs) -> CausalMask:
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.seed)
        if policy is None:
            low, high = self.env.action_space.low, self.env.action_space.high
            policy = lambda obs: rng.uniform(low, high)

        # ── Collect episodes; keep (o_t, o_{t+k}, a_t) tuples ────────
        # Do not span episode boundaries.
        obs_t_list, obs_tk_list, a_t_list = [], [], []
        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            ep_obs = [obs.flatten().astype(np.float32)]
            ep_act = []
            for t in range(self.episode_length):
                a = policy(obs)
                step = self.env.step(a)
                ns = step[0]
                ep_act.append(np.asarray(a, dtype=np.float32).flatten())
                ep_obs.append(ns.flatten().astype(np.float32))
                obs = ns
                done = step[2] if len(step) == 4 else (step[2] or step[3])
                if done:
                    break

            ep_obs = np.asarray(ep_obs, dtype=np.float32)
            ep_act = np.asarray(ep_act, dtype=np.float32)
            T = len(ep_act)
            k = self.horizon_k
            if T <= k:
                continue
            for t in range(T - k + 1):
                obs_t_list.append(ep_obs[t])
                obs_tk_list.append(ep_obs[t + k])
                a_t_list.append(ep_act[t])

        if not obs_t_list:
            raise RuntimeError(
                "MultistepInverseDynamicsSelector: no (o_t, o_{t+k}, a_t) "
                "tuples collected; episodes too short relative to horizon_k?")

        Ot = np.asarray(obs_t_list, dtype=np.float32)
        Otk = np.asarray(obs_tk_list, dtype=np.float32)
        At = np.asarray(a_t_list, dtype=np.float32)

        # train / val split
        N = len(Ot)
        idx = rng.permutation(N)
        n_val = max(1, int(N * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Ot_t = torch.from_numpy(Ot).to(device)
        Otk_t = torch.from_numpy(Otk).to(device)
        At_t = torch.from_numpy(At).to(device)
        X_full = torch.cat([Ot_t, Otk_t], dim=1)

        X_tr, y_tr = X_full[tr_idx], At_t[tr_idx]
        X_val, y_val = X_full[val_idx], At_t[val_idx]

        # ── Train g(o_t, o_{t+k}) -> a_t ─────────────────────────────
        torch.manual_seed(self.seed)
        in_dim = 2 * self.obs_dim
        model = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.action_dim),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        n_tr = len(X_tr)
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_tr, device=device)
            for start in range(0, n_tr, self.batch_size):
                b = perm[start:start + self.batch_size]
                pred = model(X_tr[b])
                loss = ((pred - y_tr[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ── Baseline val loss on full input ──────────────────────────
        model.eval()
        with torch.no_grad():
            pred_full = model(X_val)
            mse_full = float(((pred_full - y_val) ** 2).mean().item())

        # ── Per-dim ablation: zero out dim i in both o_t and o_{t+k} ─
        scores = np.zeros(self.obs_dim)
        with torch.no_grad():
            for i in range(self.obs_dim):
                X_abl = X_val.clone()
                X_abl[:, i] = 0.0                   # zero in o_t
                X_abl[:, self.obs_dim + i] = 0.0    # zero in o_{t+k}
                pred_abl = model(X_abl)
                mse_abl = float(((pred_abl - y_val) ** 2).mean().item())
                scores[i] = max(mse_abl - mse_full, 0.0)

        # ── Select top-k ─────────────────────────────────────────────
        n_sel = self.n_select or max(1, self.obs_dim // 3)
        threshold = np.sort(scores)[::-1][min(n_sel - 1, len(scores) - 1)]
        soi = {d for d in range(self.obs_dim) if scores[d] >= threshold}

        es = scores / max(scores.max(), 1e-8)
        return CausalMask.from_soi(
            soi, self.obs_dim, mode=self.mode, effect_sizes=es)