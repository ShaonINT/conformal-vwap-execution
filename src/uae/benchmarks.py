"""Classical execution schedules (the non-negotiable comparators, plan section 6).

Each function returns a per-interval child-quantity vector summing to Q. They are
*static* schedules (fixed before the session) except where they read the known
expected volume curve, matching how these baselines are used in practice.
"""

from __future__ import annotations

import numpy as np

from .simulator import MarketParams, _u_shaped_curve


def twap_schedule(Q: float, n_intervals: int) -> np.ndarray:
    """Equal slices over time -- the floor benchmark."""
    return np.full(n_intervals, Q / n_intervals, dtype=float)


def vwap_tracking_schedule(Q: float, params: MarketParams) -> np.ndarray:
    """Slices following the historical average (expected) volume curve.

    This is the benchmark the paper is named after: match the U-shaped volume
    profile so the execution's own VWAP tracks the market VWAP.
    """
    weights = _u_shaped_curve(params.n_intervals, params.u_shape)
    return Q * weights


def almgren_chriss_schedule(
    Q: float,
    n_intervals: int,
    risk_aversion: float = 2.0,
) -> np.ndarray:
    """Almgren-Chriss optimal schedule under linear impact + risk penalty.

    Uses the closed-form holdings trajectory
        x_j = Q * sinh(kappa (T - j)) / sinh(kappa T),
    where `kappa` grows with risk aversion. Child trade in interval j is the
    decrement x_{j-1} - x_j, which front-loads execution as risk aversion rises
    and reduces to TWAP as `kappa -> 0`.
    """
    T = n_intervals
    kappa = np.sqrt(max(risk_aversion, 1e-8)) / T * np.pi  # monotone in risk aversion
    if kappa < 1e-6:
        return twap_schedule(Q, T)
    j = np.arange(0, T + 1)
    x = Q * np.sinh(kappa * (T - j)) / np.sinh(kappa * T)  # remaining holdings, x[0]=Q, x[T]=0
    child = -np.diff(x)  # length T, sums to Q
    child = np.clip(child, 0.0, None)
    child *= Q / child.sum()
    return child
