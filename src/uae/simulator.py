"""Controlled intraday market simulator.

A deliberately *simple and fully reproducible* data-generating process so that
every reported number can be regenerated from a seed (the paper's central
reproducibility argument). Two ingredients matter for execution:

  1. Returns with **AR(1) momentum** and **stochastic volatility** (log-AR(1)).
     Momentum is what a forecaster can (weakly) exploit; stochastic vol is what
     makes the forecast's *reliability* vary over time -- the thing the conformal
     interval width measures.
  2. An **intraday U-shaped volume curve** (heavier at the open and close), used
     both to define the VWAP benchmark and to drive participation-based market
     impact.

Execution cost model (Almgren-Chriss-style, linear):
  - temporary impact:  eta  * participation * S_t   (paid on the slice only)
  - permanent impact:  gamma* participation * S_t   (shifts the future mid)
  - per-trade cost:    half-spread + fee, in bps    (always > 0; plan section 13)

All prices are in absolute units; costs are reported in basis points relative to
the arrival mid price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MarketParams:
    # --- session discretisation ---
    n_intervals: int = 78          # 6.5h of 5-minute bars
    # --- returns: LATENT AR(1) momentum + stochastic volatility ---
    # Momentum is a *latent* state theta (as in the paper): the return is
    # theta_t plus SV noise, so a predictor sees only a noisy proxy of the
    # momentum and has modest, realistic skill (not a near-oracle).
    phi: float = 0.94              # persistence of the latent momentum state theta
    momentum_vol_bps: float = 5.0  # innovation std of the latent momentum (thin edge)
    mu: float = 0.0                # intraday drift per interval (≈0)
    sv_persistence: float = 0.95   # beta in log-variance AR(1)
    sv_vol_of_vol: float = 0.55    # xi, volatility of log-variance (spreads gate firing)
    base_vol_bps: float = 22.0     # long-run per-interval return-noise std
    # --- volume ---
    adv: float = 1_000_000.0       # average daily volume (shares)
    u_shape: float = 0.6           # strength of the U-shaped intraday curve
    vol_noise: float = 0.0         # lognormal vol noise; 0 => VWAP-tracking has ~0 variance
    # --- microstructure / impact ---
    start_price: float = 100.0
    half_spread_bps: float = 17.0  # fixed spread/impact floor crossed on every fill
    fee_bps: float = 1.0           # exchange/broker fee per trade (bps)
    eta: float = 8.0               # temporary impact coeff (bps at 100% participation)
    gamma: float = 1.0             # permanent impact coeff (bps at 100% participation)


@dataclass
class MarketPath:
    """One realised trading session (frozen; execution reads from it)."""
    params: MarketParams
    mid: np.ndarray                # (T+1,) mid prices incl. arrival at index 0
    returns: np.ndarray            # (T,) realised interval log-returns
    vol: np.ndarray                # (T,) realised per-interval return std (SV state)
    volume: np.ndarray             # (T,) market volume per interval
    arrival_price: float
    market_vwap: float             # volume-weighted average mid over the session

    @property
    def n_intervals(self) -> int:
        return self.params.n_intervals


def _u_shaped_curve(n: int, strength: float) -> np.ndarray:
    """Normalised U-shaped intraday volume weights that sum to 1."""
    x = np.linspace(0.0, 1.0, n)
    bowl = 1.0 + strength * (np.cos(2.0 * np.pi * x) * 0.5 + 0.5) * 2.0
    return bowl / bowl.sum()


def simulate_path(params: MarketParams, rng: np.random.Generator) -> MarketPath:
    """Simulate one intraday session under `params` using generator `rng`."""
    T = params.n_intervals
    base_var = (params.base_vol_bps / 1e4) ** 2

    # Stochastic volatility: log-variance follows an AR(1) around log(base_var).
    h = np.empty(T)
    h0 = np.log(base_var)
    prev = h0
    for t in range(T):
        shock = params.sv_vol_of_vol * rng.standard_normal()
        prev = h0 + params.sv_persistence * (prev - h0) + shock
        h[t] = prev
    sigma = np.exp(0.5 * h)  # per-interval return std

    # Latent AR(1) momentum theta; the observed return adds SV noise on top, so
    # the momentum is only partially recoverable from past returns.
    sigma_theta = params.momentum_vol_bps / 1e4
    returns = np.empty(T)
    theta = 0.0
    for t in range(T):
        theta = params.phi * theta + sigma_theta * rng.standard_normal()
        eps = rng.standard_normal()
        returns[t] = params.mu + theta + sigma[t] * eps

    mid = np.empty(T + 1)
    mid[0] = params.start_price
    mid[1:] = params.start_price * np.exp(np.cumsum(returns))

    # Volume: U-shaped base curve * lognormal noise.
    weights = _u_shaped_curve(T, params.u_shape)
    noise = np.exp(params.vol_noise * rng.standard_normal(T) - 0.5 * params.vol_noise ** 2)
    volume = params.adv * weights * noise

    # Market VWAP uses the *start-of-interval* mid, consistent with how the
    # execution model fills each slice (see execute_schedule). Using a single
    # consistent price stamp is what makes slippage-vs-VWAP well defined.
    interval_price = mid[:-1]
    market_vwap = float(np.sum(volume * interval_price) / np.sum(volume))

    return MarketPath(
        params=params,
        mid=mid,
        returns=returns,
        vol=sigma,
        volume=volume,
        arrival_price=float(mid[0]),
        market_vwap=market_vwap,
    )


def execute_schedule(
    path: MarketPath,
    child_qty: np.ndarray,
    direction: str = "BUY",
) -> dict:
    """Execute a per-interval quantity schedule against a frozen `path`.

    Applies linear temporary + permanent impact and a per-trade cost, then
    returns cost metrics in basis points relative to the arrival price.

    `child_qty` is shares to trade each interval (non-negative); it is scaled so
    it sums exactly to the parent order size implied by the schedule.
    """
    p = path.params
    T = path.n_intervals
    child_qty = np.asarray(child_qty, dtype=float)
    assert child_qty.shape == (T,), f"schedule must have shape ({T},)"
    child_qty = np.clip(child_qty, 0.0, None)
    parent = child_qty.sum()
    if parent <= 0:
        raise ValueError("schedule executes zero quantity")

    sign = 1.0 if direction.upper() == "BUY" else -1.0
    half_spread = p.half_spread_bps / 1e4
    fee = p.fee_bps / 1e4

    mid = path.mid.copy()
    perm_shift = 0.0  # accumulated permanent impact on the mid
    cash = 0.0        # signed cash paid (BUY: positive outflow)
    for t in range(T):
        q = child_qty[t]
        participation = q / max(path.volume[t], 1.0)
        # Fill at the START-of-interval mid (mid[t]); a return forecast for
        # interval t is therefore actionable *before* the move is realised.
        base_mid = mid[t] + sign * perm_shift * p.start_price
        temp = p.eta / 1e4 * participation * base_mid
        exec_price = base_mid + sign * (temp + half_spread * base_mid) + sign * fee * base_mid
        cash += sign * q * exec_price
        perm_shift += p.gamma / 1e4 * participation  # permanent, in return units

    avg_exec = cash / (sign * parent)
    arrival = path.arrival_price
    vwap = path.market_vwap

    # Buy: positive shortfall = paid above benchmark = worse.
    is_bps = sign * (avg_exec - arrival) / arrival * 1e4
    slip_vwap_bps = sign * (avg_exec - vwap) / vwap * 1e4
    return {
        "avg_exec_price": avg_exec,
        "arrival_price": arrival,
        "market_vwap": vwap,
        "parent_qty": parent,
        "implementation_shortfall_bps": float(is_bps),
        "slippage_vs_vwap_bps": float(slip_vwap_bps),
    }
