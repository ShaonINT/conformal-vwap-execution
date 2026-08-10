"""Generate the paper-style figures from results CSVs.

Produces:
  results/cost_risk_frontier.png  -- mean cost vs cost-std, baselines + gate sweep
  results/coverage.png            -- empirical vs nominal conformal coverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def cost_risk_frontier() -> None:
    table = pd.read_csv(RESULTS / "results_table.csv")
    frontier = pd.read_csv(RESULTS / "cost_risk_frontier.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # gate sweep as a connected frontier
    fr = frontier.sort_values("slippage_std_bps")
    ax.plot(fr["slippage_std_bps"], fr["slippage_mean_bps"],
            "-o", color="#1f77b4", lw=2, ms=4, label="Conformal gate (kappa sweep)", zorder=3)

    markers = {"TWAP": "s", "VWAP-tracking": "D", "Almgren-Chriss": "^",
               "Forecast-greedy": "*"}
    colors = {"TWAP": "#555555", "VWAP-tracking": "#2ca02c",
              "Almgren-Chriss": "#d62728", "Forecast-greedy": "#9467bd"}
    for _, row in table.iterrows():
        name = row["method"]
        if name in markers:
            ax.scatter(row["slippage_std_bps"], row["slippage_mean_bps"],
                       marker=markers[name], s=140, color=colors[name],
                       edgecolor="black", linewidth=0.6, zorder=5, label=name)

    ax.set_xlabel("Cost variability  —  std of slippage vs VWAP (bps)")
    ax.set_ylabel("Mean cost  —  slippage vs VWAP (bps)")
    ax.set_title("Cost–risk frontier: the conformal gate trades variance for mean")
    lo = min(table["slippage_mean_bps"].min(), fr["slippage_mean_bps"].min())
    hi = max(table[table.method != "Immediate"]["slippage_mean_bps"].max(),
             fr["slippage_mean_bps"].max())
    ax.set_ylim(lo - 1.0, hi + 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / "cost_risk_frontier.png", dpi=150)
    plt.close(fig)


def coverage_plot() -> None:
    cov = json.loads((RESULTS / "coverage.json").read_text())
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    ax.bar(["nominal", "empirical"],
           [cov["nominal_coverage"], cov["empirical_coverage"]],
           color=["#bbbbbb", "#1f77b4"], edgecolor="black")
    ax.axhline(cov["nominal_coverage"], color="#d62728", ls="--", lw=1.2)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("coverage")
    ax.set_title(f"Split-conformal coverage\nempirical {cov['empirical_coverage']:.3f} "
                 f"vs nominal {cov['nominal_coverage']:.2f}")
    for i, v in enumerate([cov["nominal_coverage"], cov["empirical_coverage"]]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png", dpi=150)
    plt.close(fig)


def main() -> int:
    if not (RESULTS / "results_table.csv").exists():
        print("Run scripts/run_experiments.py first.", file=sys.stderr)
        return 1
    cost_risk_frontier()
    coverage_plot()
    print("Wrote results/cost_risk_frontier.png and results/coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
