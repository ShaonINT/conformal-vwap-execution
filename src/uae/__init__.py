"""uae - Uncertainty-Aware Execution.

Reference implementation accompanying:

  Irshad, A. & Biswas, S. (2026). "Uncertainty-Aware AI: Conformal Prediction
  versus Reinforcement Learning for Optimal Trade Execution."
  Statistics, Optimization & Information Computing, 16(2), 1334-1349.
  DOI: 10.19139/soic-2310-5070-4159

The package implements a fully reproducible VWAP-execution study inside a
controlled simulator (stochastic volatility + AR(1) return momentum), comparing
classical schedules (TWAP, Almgren-Chriss, VWAP-tracking), a PPO agent, and
forecast-driven policies built on a normalised split-conformal predictor of
next-interval returns -- including the conformal *gate* that trades off mean cost
against cost variance.

Simulator-based research only. No live-trading claims.
"""

__version__ = "1.0.0"
