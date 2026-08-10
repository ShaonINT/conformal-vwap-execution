"""Forecast-driven execution policies built on the conformal predictor.

Two policies, both tilting the VWAP-tracking base schedule by the return
forecast:

* **point-forecast policy** -- always acts on the forecast. A buyer front-loads
  into intervals with a positive predicted return (buy before the rise) and
  defers when a negative return is predicted. Lowers mean slippage *but adds
  cost variance*, because it commits on a noisy signal.

* **conformal-gated policy** -- acts on the forecast only when the conformal
  interval half-width is below a threshold ``tau`` (i.e. only when the forecast
  is reliable); otherwise it falls back to the base schedule. Sweeping ``tau``
  traces a monotone cost-variance-vs-mean frontier: tight ``tau`` -> rarely act
  -> low variance (near VWAP-tracking); loose ``tau`` -> the point policy.
"""

from __future__ import annotations

import numpy as np

from .conformal import ConformalReturnPredictor
from .features import build_xy
from .simulator import MarketParams, MarketPath, _u_shaped_curve


def _base_weights(params: MarketParams) -> np.ndarray:
    return _u_shaped_curve(params.n_intervals, params.u_shape)


def per_interval_forecasts(
    path: MarketPath, predictor: ConformalReturnPredictor
) -> tuple[np.ndarray, np.ndarray]:
    X, _, scale = build_xy(path)
    return predictor.predict(X, scale)


def forecast_tilt_schedule(
    Q: float,
    path: MarketPath,
    predictor: ConformalReturnPredictor,
    beta: float,
    direction: str = "SELL",
    kappa: float | None = 0.0,
) -> np.ndarray:
    """VWAP-tracking base schedule tilted by the (optionally gated) forecast.

    The action is a log-multiplier on the VWAP-tracking quantity (a value of 0
    reproduces volume-curve tracking), exactly as in the paper.

    Gate (paper's formulation): act on the forecast only when it escapes its own
    conformal band, ``|forecast| > kappa * half_width``. The multiple ``kappa``
    is the single interpretable dial:
        kappa = 0        -> always act  (Forecast-greedy)
        kappa -> large   -> never act   (VWAP-tracking)
    """
    base = _base_weights(path.params)
    yhat, half = per_interval_forecasts(path, predictor)
    sign = 1.0 if direction.upper() == "BUY" else -1.0
    # Trade more into predicted relative weakness (buy dips / sell into strength):
    # transacting *with* momentum would systematically fill away from VWAP. The
    # predictable component is tiny, so beta is large; clip to keep the schedule
    # sane rather than collapsing onto a single interval.
    raw = np.clip(-beta * sign * yhat, -1.5, 1.5)
    if kappa is not None and kappa > 0.0:
        fire = np.abs(yhat) > kappa * half
        raw = np.where(fire, raw, 0.0)
    w = base * np.exp(raw)
    w = w / w.sum()
    return Q * w


def kappa_grid(n: int = 14, k_max: float = 4.0) -> np.ndarray:
    """Gating dial sweep: 0 (Forecast-greedy) -> large (VWAP-tracking)."""
    return np.concatenate([[0.0], np.linspace(0.05, k_max, n)])
