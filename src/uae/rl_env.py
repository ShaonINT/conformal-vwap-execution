"""Gymnasium execution environment wrapping the controlled simulator.

State / action / reward follow the plan's MDP (section 4):

* **state**  : inventory remaining (frac), time remaining (frac), last realised
               return, current volatility proxy, and volume advancement
               (executed-vs-expected schedule progress).
* **action** : continuous scalar in [0, 1] -- the fraction of *remaining*
               inventory to execute this interval (PPO-friendly).
* **reward** : negative incremental execution cost in bps vs the arrival price,
               with a terminal forced-liquidation penalty on any unexecuted
               inventory. Transaction costs are always included.

Uses the Gymnasium API (Stable-Baselines3 >= 2.0).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
except Exception:  # pragma: no cover
    _HAS_GYMNASIUM = False
    gym = object  # type: ignore

from .simulator import MarketParams, simulate_path, _u_shaped_curve


class ExecutionEnv(gym.Env if _HAS_GYMNASIUM else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: Optional[MarketParams] = None,
        parent_qty: float = 50_000.0,
        direction: str = "BUY",
        seed_pool: Optional[range] = None,
    ):
        if not _HAS_GYMNASIUM:  # pragma: no cover
            raise ImportError("gymnasium is required for ExecutionEnv")
        super().__init__()
        self.params = params or MarketParams()
        self.parent_qty = parent_qty
        self.direction = direction
        self.sign = 1.0 if direction.upper() == "BUY" else -1.0
        self._seed_pool = list(seed_pool) if seed_pool is not None else None
        self._expected = _u_shaped_curve(self.params.n_intervals, self.params.u_shape)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # obs: [inv_frac, time_frac, last_ret, vol_proxy, advancement]
        high = np.array([1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    # ---- helpers ----
    def _obs(self) -> np.ndarray:
        t = self.t
        T = self.params.n_intervals
        inv_frac = self.remaining / self.parent_qty
        time_frac = (T - t) / T
        last_ret = self.path.returns[t - 1] if t > 0 else 0.0
        vol_proxy = float(np.std(self.path.returns[max(0, t - 10):t])) if t >= 2 else \
            self.params.base_vol_bps / 1e4
        expected_done = self._expected[:t].sum()
        actual_done = 1.0 - inv_frac
        advancement = actual_done - expected_done
        return np.array([inv_frac, time_frac, last_ret, vol_proxy, advancement], dtype=np.float32)

    # ---- gym API ----
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._seed_pool is not None:
            s = int(self._rng.choice(self._seed_pool))
        else:
            s = int(self._rng.integers(0, 2**31 - 1))
        self.path = simulate_path(self.params, np.random.default_rng(s))
        self.t = 0
        self.remaining = self.parent_qty
        self.cash = 0.0
        self.perm_shift = 0.0
        return self._obs(), {}

    def step(self, action):
        p = self.params
        T = p.n_intervals
        a = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        # final interval: force-execute all remaining
        if self.t == T - 1:
            q = self.remaining
        else:
            q = a * self.remaining
        q = min(q, self.remaining)

        mid = self.path.mid
        half_spread = p.half_spread_bps / 1e4
        fee = p.fee_bps / 1e4
        participation = q / max(self.path.volume[self.t], 1.0)
        base_mid = mid[self.t] + self.sign * self.perm_shift * p.start_price
        temp = p.eta / 1e4 * participation * base_mid
        exec_price = base_mid + self.sign * (temp + half_spread * base_mid + fee * base_mid)
        self.cash += self.sign * q * exec_price
        self.perm_shift += p.gamma / 1e4 * participation
        self.remaining -= q

        arrival = self.path.arrival_price
        # incremental cost of this slice vs arrival, in bps, normalised by parent
        slice_cost_bps = self.sign * (exec_price - arrival) / arrival * 1e4 * (q / self.parent_qty)
        reward = -slice_cost_bps

        self.t += 1
        terminated = self.t >= T
        truncated = False
        if terminated and self.remaining > 1e-6:  # safety; shouldn't trigger
            reward -= 1000.0 * (self.remaining / self.parent_qty)
        return self._obs(), float(reward), terminated, truncated, {}
