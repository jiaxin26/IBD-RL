"""
IBD — Interventional Boundary Discovery
=========================================

A general-purpose module for discovering which observation dimensions
an RL agent can causally influence, and masking out the rest.

Quick start::

    from ibd import IBDProbe, CausalMask

    probe = IBDProbe(env)
    mask  = probe.discover()          # random-policy probe
    mask  = probe.discover(policy)    # policy-guided probe

    # Use with any RL library
    masked_env = mask.wrap(env)       # gymnasium wrapper

    # Or with Stable-Baselines3
    from ibd.sb3 import make_ibd_sac
    model = make_ibd_sac(env, probe)
"""

from ibd.mask import CausalMask
from ibd.probe import IBDProbe

__all__ = ["IBDProbe", "CausalMask"]
__version__ = "0.1.0"
