"""End-to-end experiment orchestration: baselines, conformal, gating frontier.

Uses strict walk-forward splits (train / calibrate / test) and never tunes on the
test window. All cost metrics are reported in basis points as mean +/- std over
held-out episodes (multiple seeds), per plan sections 9 and 13.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .benchmarks import almgren_chriss_schedule, twap_schedule, vwap_tracking_schedule
from .conformal import ConformalReturnPredictor
from .policies import forecast_tilt_schedule, kappa_grid
from .simulator import MarketParams, execute_schedule, simulate_path


@dataclass
class ExperimentConfig:
    params: MarketParams = field(default_factory=MarketParams)
    parent_qty: float = 50_000.0
    direction: str = "SELL"      # the paper sells a parent order of Q shares
    n_train: int = 300
    n_calib: int = 300
    n_test: int = 250            # 250 held-out seeds, as in the paper
    alpha: float = 0.10
    beta: float = 260.0          # forecast tilt strength (predictable signal is tiny)
    ac_risk_aversion: float = 0.1
    kappa_report: tuple = (0.5, 1.0)  # highlighted gated operating points
    seed: int = 0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "params"}
        d["params"] = self.params.__dict__
        return d


def _make_paths(params: MarketParams, seeds: range) -> list:
    return [simulate_path(params, np.random.default_rng(s)) for s in seeds]


def _eval_schedule(policy_fn: Callable, eval_paths: list, Q: float, direction: str) -> dict:
    """Apply a path->schedule policy across eval paths; return bps metric arrays."""
    slip, isf = [], []
    for p in eval_paths:
        sched = policy_fn(p)
        res = execute_schedule(p, sched, direction=direction)
        slip.append(res["slippage_vs_vwap_bps"])
        isf.append(res["implementation_shortfall_bps"])
    return {"slippage_bps": np.asarray(slip), "is_bps": np.asarray(isf)}


def _summary(name: str, m: dict) -> dict:
    return {
        "method": name,
        "slippage_mean_bps": float(np.mean(m["slippage_bps"])),
        "slippage_std_bps": float(np.std(m["slippage_bps"])),
        "is_mean_bps": float(np.mean(m["is_bps"])),
        "is_std_bps": float(np.std(m["is_bps"])),
    }


def run_all(cfg: ExperimentConfig, out_dir: str | Path = "results") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.params
    Q = cfg.parent_qty

    # --- walk-forward splits (disjoint seed ranges; test never used for tuning) ---
    s0 = cfg.seed
    train = _make_paths(p, range(s0, s0 + cfg.n_train))
    s1 = s0 + cfg.n_train
    calib = _make_paths(p, range(s1, s1 + cfg.n_calib))
    s2 = s1 + cfg.n_calib
    test = _make_paths(p, range(s2, s2 + cfg.n_test))

    # --- conformal predictor ---
    predictor = ConformalReturnPredictor(alpha=cfg.alpha)
    predictor.fit(train)
    predictor.calibrate(calib)
    cov = predictor.evaluate_coverage(test)

    rows = []

    # --- classical baselines ---
    def _immediate(path):
        s = np.zeros(p.n_intervals); s[0] = Q; return s
    rows.append(_summary("Immediate", _eval_schedule(_immediate, test, Q, cfg.direction)))
    rows.append(_summary("TWAP", _eval_schedule(
        lambda path: twap_schedule(Q, p.n_intervals), test, Q, cfg.direction)))
    rows.append(_summary("Almgren-Chriss", _eval_schedule(
        lambda path: almgren_chriss_schedule(Q, p.n_intervals, cfg.ac_risk_aversion),
        test, Q, cfg.direction)))
    rows.append(_summary("VWAP-tracking", _eval_schedule(
        lambda path: vwap_tracking_schedule(Q, p), test, Q, cfg.direction)))

    # --- forecast-greedy (kappa=0, always act) ---
    rows.append(_summary("Forecast-greedy", _eval_schedule(
        lambda path: forecast_tilt_schedule(Q, path, predictor, cfg.beta, cfg.direction, kappa=0.0),
        test, Q, cfg.direction)))

    # --- highlighted gated operating points ---
    for k in cfg.kappa_report:
        rows.append(_summary(f"Conformal-gated (kappa={k})", _eval_schedule(
            lambda path, _k=k: forecast_tilt_schedule(Q, path, predictor, cfg.beta, cfg.direction, kappa=_k),
            test, Q, cfg.direction)))

    # --- conformal-gated frontier (sweep the kappa dial) ---
    frontier = []
    for k in kappa_grid(n=14, k_max=4.0):
        m = _eval_schedule(
            lambda path, _k=k: forecast_tilt_schedule(Q, path, predictor, cfg.beta, cfg.direction, kappa=_k),
            test, Q, cfg.direction)
        frontier.append({
            "kappa": float(k),
            "slippage_mean_bps": float(np.mean(m["slippage_bps"])),
            "slippage_std_bps": float(np.std(m["slippage_bps"])),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "results_table.csv", index=False)
    pd.DataFrame(frontier).to_csv(out_dir / "cost_risk_frontier.csv", index=False)

    coverage = {
        "nominal_coverage": 1 - cfg.alpha,
        "empirical_coverage": cov.coverage,
        "mean_halfwidth_bps": cov.mean_halfwidth_bps,
        "conformal_q": cov.q,
        "n_test_points": cov.n_test,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    return {"table": table, "frontier": frontier, "coverage": coverage}
