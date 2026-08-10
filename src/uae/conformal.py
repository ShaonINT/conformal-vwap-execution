"""Normalised split-conformal predictor of next-interval returns.

Split conformal gives a *distribution-free, finite-sample marginal coverage*
guarantee: with a proper train/calibration/test split and exchangeable scores,
the prediction interval covers the truth with probability >= 1 - alpha. We use
the **normalised** variant, dividing the residual by a scale estimate
sigma_hat(x) (a rolling realised-vol proxy) so the interval widens in volatile
regimes -- which is exactly the signal the execution policy gates on.

We report *empirical* coverage on held-out test data rather than assuming it,
per the plan's exchangeability caveat (financial series are not i.i.d.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .features import stack_sessions
from .simulator import MarketPath


@dataclass
class ConformalResult:
    alpha: float
    q: float                 # conformal quantile of normalised scores
    coverage: float          # empirical coverage on the test split
    mean_halfwidth_bps: float
    n_calib: int
    n_test: int


class ConformalReturnPredictor:
    """Gradient-boosted point predictor + normalised split-conformal interval.

    A gradient-boosted regressor (as in the paper) predicts the next-interval
    return from causal features. It recovers the AR(1) momentum, which is the
    only genuinely predictable part of the DGP -- deliberately so: the paper's
    point is that the *forecast edge is thin*, and the value is in the calibrated
    uncertainty, not the point forecast.
    """

    def __init__(self, alpha: float = 0.10, n_estimators: int = 200,
                 max_depth: int = 2, learning_rate: float = 0.05):
        self.alpha = alpha
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, random_state=0,
        )
        self.q_: Optional[float] = None
        self._fitted = False

    def fit(self, train_paths: list[MarketPath]) -> "ConformalReturnPredictor":
        X, y, _ = stack_sessions(train_paths)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def calibrate(self, calib_paths: list[MarketPath]) -> None:
        """Compute the conformal quantile of normalised nonconformity scores."""
        assert self._fitted, "call fit() before calibrate()"
        X, y, scale = stack_sessions(calib_paths)
        yhat = self.model.predict(X)
        scores = np.abs(y - yhat) / np.maximum(scale, 1e-8)
        n = len(scores)
        # finite-sample conformal quantile level
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_ = float(np.quantile(scores, level, method="higher"))

    def predict(self, X: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (point forecast, interval half-width) for feature rows X."""
        assert self.q_ is not None, "call calibrate() before predict()"
        yhat = self.model.predict(X)
        half = self.q_ * np.maximum(scale, 1e-8)
        return yhat, half

    def evaluate_coverage(self, test_paths: list[MarketPath]) -> ConformalResult:
        assert self.q_ is not None, "call calibrate() before evaluate_coverage()"
        X, y, scale = stack_sessions(test_paths)
        yhat, half = self.predict(X, scale)
        covered = np.abs(y - yhat) <= half
        return ConformalResult(
            alpha=self.alpha,
            q=self.q_,
            coverage=float(np.mean(covered)),
            mean_halfwidth_bps=float(np.mean(half) * 1e4),
            n_calib=0,
            n_test=len(y),
        )
