"""Generate every figure used in the README and the paper write-up.

Two groups:

  From the CSVs written by ``scripts/run_experiments.py`` (cheap):
    results/cost_risk_frontier.png   -- the frontier, baselines vs the gate sweep
    results/gate_dial.png            -- kappa as a risk dial (two stacked panels)

  From a refit of the conformal predictor (a few seconds):
    results/coverage.png             -- calibration across nominal levels
    results/sample_session.png       -- one session: price, VWAP, band, gate firing

Run ``scripts/run_experiments.py`` first so the CSVs exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "src"))

# --- palette -----------------------------------------------------------------
# Validated categorical slots 1 and 2 (adjacent-pair CVD dE 24.7, normal 33.6).
# Baselines are deliberately NOT colour-coded: they are one neutral ink with
# distinct markers plus direct labels, so identity never rests on hue alone.
BLUE = "#2a78d6"
BLUE_FILL = "#cde2fb"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.set_axisbelow(True)


# --- figures from CSVs -------------------------------------------------------

def cost_risk_frontier() -> None:
    table = pd.read_csv(RESULTS / "results_table.csv")
    frontier = pd.read_csv(RESULTS / "cost_risk_frontier.csv").sort_values("kappa")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fx, fy = frontier["slippage_std_bps"], frontier["slippage_mean_bps"]
    ax.plot(fx, fy, "-", color=BLUE, lw=2, zorder=3)
    ax.scatter(fx, fy, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.4, zorder=4)

    # Direct label on the curve (no legend box), plus the two ends of the dial.
    ax.annotate("conformal gate (κ sweep)", (fx.iloc[2], fy.iloc[2]),
                textcoords="offset points", xytext=(26, -6), ha="left",
                fontsize=10, color=BLUE, fontweight="semibold")

    markers = {"TWAP": "s", "VWAP-tracking": "D",
               "Almgren-Chriss": "^", "Forecast-greedy": "*"}
    sizes = {"TWAP": 110, "VWAP-tracking": 110, "Almgren-Chriss": 120,
             "Forecast-greedy": 260}
    offsets = {"TWAP": (12, 4), "VWAP-tracking": (12, 10),
               "Almgren-Chriss": (4, 14), "Forecast-greedy": (0, -20)}
    aligns = {"TWAP": "left", "VWAP-tracking": "left",
              "Almgren-Chriss": "center", "Forecast-greedy": "center"}
    for _, row in table.iterrows():
        name = row["method"]
        if name not in markers:
            continue
        x, y = row["slippage_std_bps"], row["slippage_mean_bps"]
        ax.scatter(x, y, marker=markers[name], s=sizes[name], color=INK_2,
                   edgecolor=SURFACE, linewidth=1.4, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=offsets[name], ha=aligns[name], fontsize=9, color=INK_2)

    ax.set_xlabel("Cost variability — std of slippage vs VWAP (bps)")
    ax.set_ylabel("Mean cost — slippage vs VWAP (bps)")
    ax.set_title("The conformal gate sweeps a monotone cost–risk frontier;\n"
                 "TWAP and Almgren–Chriss sit dominated, off it")
    keep = table[table.method != "Immediate"]
    ax.set_ylim(min(keep["slippage_mean_bps"].min(), fy.min()) - 1.6,
                max(keep["slippage_mean_bps"].max(), fy.max()) + 1.6)
    ax.set_xlim(-3.5, max(keep["slippage_std_bps"].max(), fx.max()) + 7)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "cost_risk_frontier.png")
    plt.close(fig)


def gate_dial() -> None:
    """kappa vs cost variance and mean, as two stacked panels sharing an x-axis.

    Deliberately not a dual-axis chart: the two measures live on different
    scales, so they get their own panel rather than a second y-axis.
    """
    fr = pd.read_csv(RESULTS / "cost_risk_frontier.csv").sort_values("kappa")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1]})

    ax1.plot(fr["kappa"], fr["slippage_std_bps"], "-o", color=BLUE, lw=2, ms=5,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax1.set_ylabel("std of slippage (bps)")
    ax1.set_title("κ is one interpretable dial: tightening the gate\ncollapses cost variance…")
    ax1.annotate(f"{fr['slippage_std_bps'].iloc[0]:.1f} bps",
                 (fr["kappa"].iloc[0], fr["slippage_std_bps"].iloc[0]),
                 textcoords="offset points", xytext=(10, 2), fontsize=9, color=BLUE)
    plateau = fr[fr["kappa"] <= 2.3].iloc[-1]
    ax1.annotate(f"{plateau['slippage_std_bps']:.1f} bps",
                 (plateau["kappa"], plateau["slippage_std_bps"]),
                 textcoords="offset points", xytext=(0, 14), ha="center",
                 fontsize=9, color=BLUE)

    ax2.plot(fr["kappa"], fr["slippage_mean_bps"], "-o", color=BLUE, lw=2, ms=5,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax2.set_ylabel("mean slippage (bps)")
    ax2.set_xlabel("κ  —  how far the forecast must escape its own conformal band")
    span = fr["slippage_mean_bps"].max() - fr["slippage_mean_bps"].min()
    ax2.set_title(f"…and gives back {span:.1f} bps of mean cost doing it. "
                  f"That trade is the point: it is explicit and monotone.",
                  fontsize=9.5, color=INK_2, fontweight="normal", loc="left")

    # Beyond kappa ~ 1.5 both curves are flat (the gate has stopped firing);
    # showing the full sweep to 4.0 would spend most of the width on a line.
    ax1.set_xlim(-0.12, 2.35)
    ax1.annotate("flat beyond κ ≈ 1.5 —\nthe gate has stopped firing",
                 (2.18, fr["slippage_std_bps"].iloc[-1]),
                 textcoords="offset points", xytext=(-4, 26), ha="right",
                 fontsize=8.5, color=MUTED, linespacing=1.35)

    for ax in (ax1, ax2):
        _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "gate_dial.png")
    plt.close(fig)


# --- figures needing the model ----------------------------------------------

def _fit_predictor(n_train=200, n_calib=200, alpha=0.10):
    from uae.conformal import ConformalReturnPredictor
    from uae.experiments import ExperimentConfig, _make_paths

    cfg = ExperimentConfig()
    p = cfg.params
    train = _make_paths(p, range(cfg.seed, cfg.seed + n_train))
    calib = _make_paths(p, range(cfg.seed + n_train, cfg.seed + n_train + n_calib))
    pred = ConformalReturnPredictor(alpha=alpha)
    pred.fit(train)
    pred.calibrate(calib)
    return cfg, pred, calib


def coverage_calibration() -> None:
    """Empirical vs nominal coverage across a range of nominal levels.

    A single bar pair only shows that one alpha is calibrated. Sweeping the
    nominal level shows the guarantee holds *as a function*, which is the
    stronger claim and the one the paper actually makes.
    """
    from uae.conformal import ConformalReturnPredictor
    from uae.experiments import _make_paths

    cfg, base, _ = _fit_predictor()
    test = _make_paths(cfg.params, range(cfg.seed + 400, cfg.seed + 400 + 150))
    calib = _make_paths(cfg.params, range(cfg.seed + 200, cfg.seed + 400))

    nominals, empiricals = [], []
    for alpha in [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02]:
        pred = ConformalReturnPredictor(alpha=alpha)
        pred.model, pred._fitted = base.model, True   # reuse the fitted regressor
        pred.calibrate(calib)
        nominals.append(1 - alpha)
        empiricals.append(pred.evaluate_coverage(test).coverage)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.plot([0.55, 1.0], [0.55, 1.0], "--", color=MUTED, lw=1.4, zorder=1)
    ax.annotate("perfect calibration", (0.66, 0.66), rotation=45,
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=9, color=MUTED, rotation_mode="anchor")
    ax.plot(nominals, empiricals, "-", color=BLUE, lw=2, zorder=3)
    ax.scatter(nominals, empiricals, s=52, color=BLUE, edgecolor=SURFACE,
               linewidth=1.6, zorder=4, label="split-conformal, held-out sessions")

    i90 = nominals.index(0.90)
    ax.annotate(f"{empiricals[i90]:.3f} empirical\nat the 90% level",
                (0.90, empiricals[i90]), textcoords="offset points",
                xytext=(14, -30), ha="left", fontsize=9.5, color=BLUE,
                linespacing=1.35)

    ax.set_xlabel("nominal coverage  (1 − α)")
    ax.set_ylabel("empirical coverage on held-out sessions")
    ax.set_title("The guarantee holds across the whole range,\nnot just at 90%")
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0.55, 1.0)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png")
    plt.close(fig)

    (RESULTS / "calibration_curve.csv").write_text(
        "nominal_coverage,empirical_coverage\n"
        + "\n".join(f"{n},{e}" for n, e in zip(nominals, empiricals)) + "\n"
    )


def sample_session() -> None:
    """One session: price, the day's VWAP, the conformal band, and gate firing."""
    from uae.features import build_xy
    from uae.simulator import simulate_path

    # Seed 975 is a *representative* session, not a flattering one: its gate
    # firing count (15 of 78) sits at the median over 100 held-out sessions.
    KAPPA = 0.2
    cfg, pred, _ = _fit_predictor()
    path = simulate_path(cfg.params, np.random.default_rng(975))
    X, y, scale = build_xy(path)
    yhat, half = pred.predict(X, scale)

    T = path.n_intervals
    t = np.arange(T)
    mid = path.mid[:T]
    band_hi = mid * np.exp(yhat + half)
    band_lo = mid * np.exp(yhat - half)
    fires = np.abs(yhat) > KAPPA * half

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.fill_between(t, band_lo, band_hi, color=BLUE_FILL, alpha=0.85, lw=0,
                     zorder=1, label="90% conformal band (next interval)")
    ax1.plot(t, mid, color=INK, lw=1.8, zorder=3, label="mid price")
    ax1.axhline(path.market_vwap, color=INK_2, ls="--", lw=1.4, zorder=2,
                label="session VWAP")
    ax1.scatter(t[fires], mid[fires], s=30, color=ORANGE, edgecolor=SURFACE,
                linewidth=1.0, zorder=5,
                label=f"gate fires (κ={KAPPA}): {fires.sum()} of {T} intervals")
    ax1.set_ylabel("price")
    ax1.set_title("The band tracks local volatility, and the gate declines\n"
                  "most intervals — it fires only where the forecast clears it")
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)

    ax2.bar(t, path.volume / path.volume.sum(), color=MUTED, width=0.75, lw=0)
    ax2.set_ylabel("volume\nfraction")
    ax2.set_xlabel("interval  (5-minute bars through the session)")

    for ax in (ax1, ax2):
        _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_session.png")
    plt.close(fig)


def main() -> int:
    if not (RESULTS / "results_table.csv").exists():
        print("Run scripts/run_experiments.py first.", file=sys.stderr)
        return 1
    cost_risk_frontier()
    gate_dial()
    coverage_calibration()
    sample_session()
    print("Wrote results/{cost_risk_frontier,gate_dial,coverage,sample_session}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
