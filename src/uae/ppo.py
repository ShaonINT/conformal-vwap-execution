"""PPO agent (Stable-Baselines3) over the execution environment.

Trained across multiple seeds and evaluated against the classical schedules. The
paper's finding -- reproduced here -- is that an off-the-shelf PPO agent is
high-variance across seeds and does not reliably beat the simple baselines, which
is exactly why the distribution-free conformal gate is the more dependable route
to uncertainty-aware execution.

Imports of torch / stable-baselines3 are deferred so the rest of the package runs
without them installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .simulator import MarketParams, execute_schedule, simulate_path
from .rl_env import ExecutionEnv


@dataclass
class PPOEvalResult:
    seed: int
    slippage_mean_bps: float
    slippage_std_bps: float


def train_ppo(
    params: MarketParams,
    parent_qty: float,
    direction: str,
    train_seed_pool: range,
    total_timesteps: int = 40_000,
    seed: int = 0,
):
    """Train a PPO policy; returns the SB3 model. Requires stable-baselines3."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def _factory():
        return ExecutionEnv(params, parent_qty, direction, seed_pool=train_seed_pool)

    env = make_vec_env(_factory, n_envs=1, seed=seed)
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        n_steps=1024, batch_size=256, gae_lambda=0.95, gamma=0.999,
        ent_coef=0.0, learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_ppo(model, params, parent_qty, direction, eval_seeds: range) -> PPOEvalResult:
    """Roll the trained policy over held-out sessions; report slippage vs VWAP."""
    slips = []
    for s in eval_seeds:
        path = simulate_path(params, np.random.default_rng(s))
        T = params.n_intervals
        # deterministic rollout: reconstruct the child schedule the policy takes
        remaining = parent_qty
        child = np.zeros(T)
        perm = 0.0
        expected = None
        env = ExecutionEnv(params, parent_qty, direction)
        # drive the env's obs construction manually against this fixed path
        env.path = path
        env.t = 0
        env.remaining = parent_qty
        env.perm_shift = 0.0
        obs = env._obs()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            t = env.t
            a = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
            q = env.remaining if t == T - 1 else a * env.remaining
            child[t] = min(q, env.remaining)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        if child.sum() <= 0:
            child[:] = parent_qty / T
        res = execute_schedule(path, child, direction=direction)
        slips.append(res["slippage_vs_vwap_bps"])
    slips = np.asarray(slips)
    return PPOEvalResult(seed=-1,
                         slippage_mean_bps=float(slips.mean()),
                         slippage_std_bps=float(slips.std()))
