"""Standalone, minimal-dependency module for the safe-RL portfolio adjustment layer.

Produced by Step6B_PackagePolicy.ipynb (Step 6 of SAFE_RL_GUIDE.pdf). Depends only on numpy and
stable_baselines3 -- no gymnasium, no pandas, no dataset files -- because inference with a saved
Stable-Baselines3 model needs neither the training environment nor the training data.

The wrapped policy (ppo_lagrangian_trained.zip) was trained entirely on TRAIN (2018-01-03 to
2024-12-31), against the no-views bl_baseline_train_weights.csv anchor -- not on TEST, and not on
bl_llm_memory_weights.csv. It is deliberately anchor-agnostic: safe_rl_adjust() takes whatever
today's candidate weights are (bl_weights_today) and a matching 12-dim state (5-day trailing mean
return + vol per asset, plus that same candidate weight vector), and returns a safety-adjusted
version of them. That is what lets the exact same trained policy be applied, unmodified, to
bl_llm_memory_weights.csv's TEST-range values in Step6B_WalkForwardEvaluation.ipynb -- a candidate
weight series this policy has never seen -- without retraining anything.

Usage:
    from safe_rl_policy import safe_rl_adjust
    final_weights = safe_rl_adjust(bl_weights_today, state)
"""
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_MODEL_PATH = Path(__file__).parent / "ppo_lagrangian_trained"
_model = PPO.load(str(_MODEL_PATH))


def safe_rl_adjust(bl_weights_today: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Takes today's BL-recommended weights (from ANY upstream source -- BL_baseline,
    BL_worldmodel, BL_llm_memory, or a freshly recomputed candidate series) + a matching 12-dim
    state, returns the final, safety-adjusted weights. This is the only function the rest of the
    team needs."""
    action, _ = _model.predict(state, deterministic=True)
    adjusted = np.clip(np.asarray(bl_weights_today) + action, 0, 1)
    return adjusted / adjusted.sum()
