# Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2603.18257-b31b1b.svg)](https://arxiv.org/abs/2603.18257)

Code for the paper:

> **Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning**
> 
> Jiaxin Liu, Anzhe Cheng, Paul Bogdan

When an RL agent's observations contain distractors driven by the same confounders as its true state, observational data alone cannot identify which dimensions the agent controls. **Interventional Boundary Discovery (IBD)** treats the agent's own action channel as a source of randomized interventions: it compares trajectories under a probe policy against trajectories with randomized actions, runs per-dimension two-sample tests with FDR control, and returns a binary mask over observation dimensions — the agent's **Causal Sphere of Influence (SoI)**. The mask is computed once and plugged in front of any downstream RL algorithm (SAC, TD3, …).

---

## Highlights

- **Model-free and cheap** — no world model, no latent recovery, no neural network in the probe. A single probe is ~32K environment steps and runs in 2–3 minutes on one CPU core.
- **Near-oracle control** — across 12 DeepMind Control settings with up to 100 distractors, IBD matches oracle return in 11 of 12, while mutual information, variance, state-conditioned forward models, gradient attribution, and multistep inverse dynamics often underperform plain Full-State SAC.
- **Accurate boundaries** — mean precision, recall, and F1 of 0.95 across all settings.
- **Algorithm-agnostic** — the same mask transfers unchanged from SAC to TD3.
- **Diagnostic** — comparing Full State, IBD, and Oracle tells you whether poor RL performance is caused by distractors or by something else (e.g., exploration).

---

## Repository structure

```
ibd_benchmark/
├── ibd/                               # The IBD library (pip-installable)
│   ├── probe.py                       # IBDProbe: trajectory collection + interventional testing
│   ├── cim.py                         # Per-(action, obs, horizon) permutation-test estimator
│   ├── mask.py                        # CausalMask: soft/hard masks, evaluation, save/load
│   ├── wrappers.py                    # Gymnasium wrappers (MaskedObsWrapper, DimSelectWrapper)
│   └── sb3.py                         # Stable-Baselines3 integration (feature extractor, callback)
├── experiments/
│   ├── dmcontrol_distractors.py       # Distractor benchmark (autonomous / mimicking / reward-correlated)
│   ├── baselines.py                   # MI, Variance, Cond. MI, Grad. Attr., Inverse Dyn., Random, Oracle
│   ├── run_dmcontrol.py               # Main downstream-RL experiments (Tables 1–3, 6; Section 4.7)
│   ├── run_scaling.py                 # Dense distractor-count sweep (Appendix H)
│   ├── run_robustness.py              # Partial controllability (Section 4.8, Appendix B)
│   ├── run_scout_ablation.py          # Probe/scout budget ablation (Appendix C)
│   ├── plot_robustness.py             # Figure 4
│   ├── plot_learning_curves.py        # Learning curves from per-seed results
│   ├── confounded_distractors.py      # Explicitly confounded environment (latent C → a, C → d)
│   └── run_*.py / analyze_* / plot_*  # Additional analyses (see below)
├── plot_scaling_v2.py                 # Figures 5 and 6 (dense scaling sweep, power-law fit)
├── aggregate_invdyn.py                # Table 3 (multistep inverse dynamics comparison)
├── plot_ranking.py                    # Per-dimension IBD vs. MI score ranking
├── plot_multihorizon.py               # Per-horizon test statistics
├── requirements.txt
└── pyproject.toml
```

---

## Setup

```bash
cd ibd_benchmark/

conda create -n ibd python=3.10 -y
conda activate ibd

pip install -r requirements.txt
pip install -e .            # installs the `ibd` package
```

`requirements.txt` pulls in Gymnasium, Stable-Baselines3 (and PyTorch), `dm_control`, MuJoCo, and plotting libraries. SciPy, which the probe uses for the Welch t-test, is installed as a dependency of `dm_control`; if you install only the `ibd` package on its own, add `pip install scipy`.

**Hardware.** Everything runs on CPU; no GPU is required. One 300K-step SAC/TD3 run takes about 27 minutes on a single core, and the full set of experiments in the paper takes roughly 280 CPU-hours. Observations are proprioceptive state vectors, so no rendering is needed; on a headless machine you can set `MUJOCO_GL=egl` if MuJoCo complains about a display.

---

## Quick start

### Discover the Sphere of Influence

```python
import numpy as np
from experiments.dmcontrol_distractors import make_env
from ibd import IBDProbe
from ibd.wrappers import DimSelectWrapper

# cartpole_swingup: 5 true dims + 50 distractors ("medium")
env = make_env("cartpole", "swingup", distractor_config="medium", seed=0)

probe = IBDProbe(env, n_baseline=80, n_intervention=80, traj_length=200,
                 horizons=[1, 5, 10], alpha=0.05)
soi, info = probe.discover_joint()   # no policy → structured random probe

print(sorted(soi))                   # e.g. [0, 1, 2, 3, 4]
print(info["p_values"])              # per-dimension adjusted p-values

# Keep only the discovered dimensions for downstream RL
masked_env = DimSelectWrapper(env, np.array(sorted(soi)))
```

`discover_joint()` implements the procedure in the paper: all action dimensions are randomized jointly in the intervention branch, each dimension is summarized per trajectory by its mean absolute h-step difference, and Welch t-tests with Benjamini–Hochberg correction decide membership. You can pass your own `policy` (any `obs -> action` callable) as the probe policy, or `test_type="ks"` to use a Kolmogorov–Smirnov test instead.

`IBDProbe` works with any Gymnasium-compatible environment with a `Box` action space, because interventions are applied externally by overriding actions before `env.step()`.

### Train a downstream agent

```python
from stable_baselines3 import SAC

model = SAC("MlpPolicy", masked_env, learning_rate=3e-4, batch_size=256,
            buffer_size=300_000, policy_kwargs=dict(net_arch=[256, 256]))
model.learn(total_timesteps=300_000)
```

---

## Reproducing the paper

All scripts are run from `ibd_benchmark/`. Seeds follow `42 + 100·i`, so `--seeds 5` gives `{42, 142, 242, 342, 442}` as in the paper.

### Main results (Tables 1, 2, 3, 6)

`run_dmcontrol.py` trains one method on one task/distractor setting across seeds and stores per-seed JSONs (final return, learning curve, and boundary precision/recall/F1) under `results/dmcontrol/per_seed/`.

```bash
# One setting, one method
python experiments/run_dmcontrol.py --domain reacher --task hard \
    --distractors medium --method ibd --seeds 5 \
    --eval_interval 50000 --eval_episodes 10
```

The flags `--eval_interval 50000 --eval_episodes 10` match the evaluation protocol in the paper (the script defaults are 10K and 5). Available methods:

| `--method` | Description | Paper |
|---|---|---|
| `full_state` | SAC on the full observation | Tables 1–3 |
| `oracle` | SAC on the ground-truth state dimensions | Tables 1–3 |
| `ibd` | SAC on the IBD mask | Tables 1–3, 6 |
| `mutual_info` | MI-based selection (given true budget d_c) | Table 1 |
| `variance` | Variance-based selection (given d_c) | Table 1 |
| `cond_mi` | State-conditioned forward-model MI (given d_c) | Tables 1, 2 |
| `grad_attr` | Gradient attribution on a learned forward model | Table 2 |
| `inverse_dyn` | Multistep inverse dynamics, k = 3 | Table 3 |
| `random_mask` | Random subset of dimensions | — |

Tasks are `walker/walk`, `cheetah/run`, `reacher/hard`, `cartpole/swingup`, `finger/spin`, `hopper/hop`; distractor levels are `easy` (6), `medium` (50), and `hard` (100).

For `--method ibd`, the runner uses an 80K-step SAC scout as the probe policy by default (`--ibd_scout_steps`). The scout ablation below shows that the untrained structured random probe gives identical or better masks.

Preset grids are also available:

```bash
python experiments/run_dmcontrol.py --tier 1 --parallel 3   # walker/cheetah, medium
python experiments/run_dmcontrol.py --tier 2 --parallel 3   # walker/cheetah, hard
python experiments/run_dmcontrol.py --tier 3 --parallel 3   # 4 more tasks + baselines, medium
python experiments/run_dmcontrol.py --tier 4 --parallel 3   # TD3 backend
```

Aggregate results into mean ± std tables (returns and boundary P/R/F1):

```bash
python experiments/run_dmcontrol.py --aggregate
python aggregate_invdyn.py --results_dir results/dmcontrol/per_seed   # Table 3
```

### TD3 transfer (Section 4.7)

```bash
python experiments/run_dmcontrol.py --domain walker --task walk \
    --distractors hard --method ibd --algo td3 --seeds 5
```

or run `--tier 4`.

### Dense distractor scaling (Appendix H, Figures 5–6)

```bash
python experiments/run_scaling.py \
    --tasks walker_walk cheetah_run reacher_hard \
    --distractor_counts 6 24 50 100 150 --seeds 3

python plot_scaling_v2.py --results_dir results/scaling
```

`plot_scaling_v2.py` produces the linear-scale curves (`fig_scaling_curves`) and the log–log ratio plot with bootstrap power-law fits (`fig_scaling_law`).

### Partial controllability (Section 4.8, Appendix B, Figure 4)

```bash
python -m experiments.run_robustness          # cheetah + walker, 10 α values, 3 seeds
python -m experiments.plot_robustness --results_dir results/robustness --out fig_robustness.pdf
```

This is probe-only (no downstream RL training). Use `--quick` for a short smoke test.

### Scout budget ablation (Appendix C, Table 5)

```bash
python experiments/run_scout_ablation.py      # budgets {0, 10K, 20K, 40K, 80K, 160K}
```

Budget 0 is the structured random probe. Results are written to `results/ablation/`.

### Confounded distractors (Section 4.6, Appendices I and K; Tables 4, 8, 10)

```bash
# 2x2 design over (kappa, lambda), 3 tasks, 6 selectors, 5 seeds
python experiments/run_confounding_sweep.py --grid \
    --domains walker:walk,cheetah:run,reacher:hard \
    --methods ibd,inverse_dyn,mutual_info,variance,grad_attr,cond_mi \
    --kappas 0,0.5 --lams 0,1 --seeds 5 --parallel 12

python experiments/analyze_confounding.py --results_dir results/confounding
```

This is probe-only (no downstream RL training). The environment is defined in `confounded_distractors.py`. Each IBD run scores the {Welch, KS} × {BH, BY} variants on the same trajectories, and `analyze_confounding.py` reports them together with the per-cell false-positive breakdown; `plot_confounding.py` plots the sweep.

### Summary statistic (Appendix J, Table 9)

```bash
python -m experiments.run_summary_stat                       # delta / level / both on four standard settings
python experiments/run_levelshift.py --domain cheetah --task run     # constructed level-shift dimensions
python experiments/run_levelshift.py --domain reacher --task hard
python experiments/run_levelshift.py --domain walker --task walk
```

### Additional analyses

The `experiments/` folder also contains scripts for supplementary studies: partial action override (`run_override.py`), misspecification (`run_misspec.py`), horizon and trajectory-count sweeps (`run_horizon.py`, `run_nsweep.py`), and effect-size estimation (`run_beta_estimator.py`). Each script documents its usage in its header.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026discoveringcontrolinterventionalboundary,
      title={Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning}, 
      author={Jiaxin Liu and Anzhe Cheng and Paul Bogdan},
      year={2026},
      eprint={2603.18257},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.18257}, 
}
```
