"""Feature construction for the next-interval return predictor.

At the decision point for interval ``t`` we may only use information realised
strictly before ``t`` (no look-ahead). Features are intentionally light: lagged
returns (which carry the AR(1) momentum signal), a rolling realised-volatility
proxy (which also serves as the conformal normaliser), and time-of-day.
"""

from __future__ import annotations

import numpy as np

from .simulator import MarketPath

N_LAGS = 3
VOL_WINDOW = 10
_VOL_FLOOR = 1e-5


def build_xy(path: MarketPath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, scale) for one session.

    X      : (T, d) features known before each interval's return.
    y      : (T,)   realised interval return (the prediction target).
    scale  : (T,)   rolling realised-vol proxy sigma_hat(x), the conformal
                    normaliser (strictly a function of past returns).
    """
    r = path.returns
    T = len(r)
    X = np.zeros((T, N_LAGS + 2), dtype=float)
    scale = np.full(T, path.params.base_vol_bps / 1e4, dtype=float)
    for t in range(T):
        for k in range(N_LAGS):
            idx = t - 1 - k
            if idx >= 0:
                X[t, k] = r[idx]
        window = r[max(0, t - VOL_WINDOW):t]
        if window.size >= 2:
            s = float(np.std(window))
            scale[t] = max(s, _VOL_FLOOR)
        X[t, N_LAGS] = scale[t]           # realised-vol proxy as a feature too
        X[t, N_LAGS + 1] = t / T          # time-of-day
    y = r.copy()
    return X, y, scale


def stack_sessions(paths: list[MarketPath]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (X, y, scale) across multiple sessions."""
    Xs, ys, ss = [], [], []
    for p in paths:
        X, y, s = build_xy(p)
        Xs.append(X)
        ys.append(y)
        ss.append(s)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
