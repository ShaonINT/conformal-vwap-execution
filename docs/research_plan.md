# Uncertainty-Aware Reinforcement Learning for Optimal Trade Execution
### A Conformal Approach to VWAP-Benchmarked Order Scheduling

**Status:** Working research plan (v1)
**Authorship:** First author — [PhD applicant], leading. Second author — Shaon Biswas (mentor / methods / writing).
**Goal:** Indexed (Scopus/WoS) publication. Citation-oriented; simulator-based, no live-trading claims.

---

## 1. The one-line pitch

Most RL-for-trading papers try to *predict the market* and quietly fail to replicate. This paper does not. It takes a decision that is already made — "execute this large order today" — and learns *how* to slice it over the day to beat the VWAP benchmark at lower cost and lower cost-variance, while using a conformal uncertainty layer to trade cautiously exactly when the market is hardest to read.

The thesis is consistent with the view that returns are unpredictable: the edge is in execution and risk control, not forecasting direction.

---

## 2. Why this topic survives review

- **Bounded, agreed objective.** Beating VWAP / minimizing implementation shortfall is a standard, well-defined problem with established baselines, so reviewers can actually evaluate the result. This is unlike alpha-seeking agents, where success is unfalsifiable.
- **Reproducible.** Built on an open agent-based limit-order-book simulator where market impact *emerges* from order matching rather than being hand-assumed.
- **A genuine novelty hook.** Conformal prediction applied to RL execution is essentially open space; most uncertainty work in trading stops at point forecasts or ad-hoc confidence.
- **Honest about its limits.** Frames everything as simulator-based and methodological.

---

## 3. The contribution (what is actually new)

1. A reinforcement-learning trade-execution agent whose **state includes a distribution-free uncertainty signal** derived from conformal prediction, letting the policy condition aggressiveness on how reliable the near-term cost/volume estimate is.
2. An **empirical comparison** of this uncertainty-aware agent against (a) classical execution baselines and (b) an identical RL agent *without* the uncertainty signal — isolating the value of the conformal layer specifically.
3. A **reproducible evaluation protocol** with walk-forward splits, realistic transaction costs, and reported conformal coverage — addressing the field's known reproducibility gap.

> **Minimal-viable fallback:** if the conformal layer proves fiddly, the plain RL-vs-baselines execution paper still stands as a complete, publishable contribution. The uncertainty layer is the upgrade, not the dependency.

---

## 4. Problem formulation (the MDP)

A parent order of size `Q` (buy or sell) must be executed over a fixed window (e.g. one trading day, discretized into `T` intervals of e.g. 5 minutes).

**State `s_t`:**
- Time remaining in window (fraction) and inventory remaining (fraction of `Q`)
- Recent price return and realized volatility over a short lookback
- Bid–ask spread and order-book imbalance (top-of-book depth ratio)
- Elapsed-vs-expected volume (are we ahead of or behind the volume curve?)
- **Conformal uncertainty feature:** width of the prediction interval for the near-term quantity being forecast (see §5)

**Action `a_t`:** fraction of *remaining* inventory to execute this interval (continuous in [0,1] for PPO; discretized buckets for DQN). Optionally extend to a limit-vs-market order choice in a later iteration.

**Reward `r_t`:** negative execution cost relative to benchmark, i.e. interval implementation shortfall vs interval VWAP, **minus** a market-impact / turnover penalty term. Terminal penalty if inventory is not fully executed by window close (forced liquidation at a worse price).

> *Shaon's input most valuable here:* the reward shaping (shortfall vs differential-Sharpe-style vs drawdown-penalized) and which state features actually carry signal versus noise. An active trader's intuition on spread/imbalance behavior is exactly the edge a pure-ML co-author lacks.

---

## 5. The conformal layer (the novel hook)

VWAP execution fundamentally depends on forecasting **how volume will be distributed across the rest of the day** (to know whether to speed up or slow down). That forecast is uncertain, and the uncertainty is itself informative.

**Approach (split conformal):**
1. Train a lightweight predictor (gradient-boosted trees or a small MLP) for a short-horizon target — start with **remaining-window volume share**, alternative target: near-term adverse price move.
2. On a held-out calibration set, compute nonconformity scores and derive a prediction interval with guaranteed marginal coverage at level `1 − α` (e.g. 90%).
3. Feed the **interval width** into the RL state as the uncertainty feature. Wide band → the agent learns to avoid over-committing; narrow band → it can front-load.
4. Report empirical coverage to confirm the conformal guarantee holds out-of-sample.

**Why this is defensible:** the conformal step is model-agnostic and gives a distribution-free coverage guarantee — a clean statistical contribution that fits a statistics/optimization venue, not just an ML one.

*(Optional advanced variant: adaptive/Mondrian conformal to maintain coverage under intraday regime shifts. Hold for a v2 / extension; do not block v1 on it.)*

---

## 6. Baselines (non-negotiable comparators)

| Baseline | What it is | Why included |
|---|---|---|
| **TWAP** | Equal slices over time | Simplest schedule; floor benchmark |
| **VWAP-tracking (static)** | Slices following a historical avg volume curve | The benchmark the paper is named after |
| **Almgren–Chriss** | Closed-form optimal execution under a linear impact + risk model | The classical academic standard; its absence is a common rejection reason |

The RL agent (with and without the conformal feature) is measured against all three.

---

## 7. Environment & data (tiered — start at Tier 1)

**Tier 1 — primary, reproducible, free:**
**ABIDES-Gym** (Agent-Based Interactive Discrete Event Simulation; open-source on GitHub, with an OpenAI-Gym-style execution environment). Market impact emerges from the matching engine rather than being assumed — this is the single most important property for credibility. Fully runnable on consumer hardware.

**Tier 2 — validation / calibration with real data (optional but strengthens the paper):**
- **LOBSTER** reconstructed NASDAQ limit-order-book data (free sample days for a few tickers such as AAPL/AMZN; paid for full history) — use to calibrate the simulator's parameters or validate the learned volume curve.
- Free intraday minute-bar OHLCV (e.g. via a free market-data API) is enough to build a realistic intraday volume profile for the VWAP benchmark even without full book depth.

**Tier 3 — do NOT chase for a first paper:**
True tick-level multi-venue feeds (Refinitiv, NYSE TAQ, Databento). Expensive, unnecessary, and a scope trap.

> **Verify before committing:** confirm ABIDES-Gym installs and runs cleanly, and re-check LOBSTER's current free-sample availability — both have shifted over time. This is the first task for the first author (see §10).

---

## 8. Agent & tooling

- **Algorithms:** PPO (continuous action) as primary; DQN (discretized) as a secondary comparison. Use **Stable-Baselines3**.
- **Predictor for conformal layer:** scikit-learn / XGBoost + a small split-conformal wrapper (MAPIE library handles split conformal cleanly, or ~30 lines by hand).
- **Stack:** Python, Gymnasium, Stable-Baselines3, NumPy/Pandas, the ABIDES repo. All open-source, all local.

---

## 9. Evaluation metrics

- **Implementation shortfall** vs arrival price (primary cost metric)
- **Slippage vs VWAP** (the named benchmark)
- **Variance / standard deviation of execution cost** across episodes (the risk axis — this is where uncertainty-awareness should pay off)
- **Conformal coverage** (does the 90% interval actually cover ~90%?)
- **Cost–risk frontier:** plot mean cost vs cost-variance for every method; the story is "uncertainty-aware RL dominates or matches on the frontier"

Report mean ± std over many randomized episodes / seeds, not a single run.

---

## 10. Scoped task list for the first author (so the credit is earned and defensible)

These are real, sequenced tasks. He should be able to *explain and defend each one* in a PhD interview — that is the point.

1. **Environment setup (week 1–2):** install ABIDES-Gym, get the execution environment running, reproduce its example episode. Confirm LOBSTER sample access. *Deliverable: a working notebook that runs one random-agent episode.*
2. **Baselines (week 2–3):** implement TWAP and static VWAP-tracking; integrate a reference Almgren–Chriss schedule. *Deliverable: baseline cost numbers on the simulator.*
3. **RL agent (week 4–6):** wire Stable-Baselines3 PPO to the environment; train; tune basic hyperparameters. *Deliverable: trained agent beating TWAP.*
4. **Conformal predictor (week 6–8, with Shaon):** build the volume-share predictor + split-conformal interval; verify coverage. *Deliverable: calibrated intervals with a coverage plot.*
5. **Integration + experiments (week 8–10):** add the uncertainty feature to the state; run the with/without ablation; collect metrics over seeds. *Deliverable: results tables + cost–risk frontier plot.*
6. **Writing (week 10–12):** first author drafts Introduction, Methods, Experiments; Shaon edits, writes the conformal/statistics framing, and handles submission.

If he genuinely does 1–3 and 5, first authorship is legitimate.

---

## 11. Division of labour

- **First author:** environment, baselines, RL training runs, experiment execution, first draft.
- **Shaon (2nd author / mentor):** problem framing, conformal methodology, reward/state design (trader intuition), results interpretation, statistical framing, final writing and submission.

---

## 12. Target venues

- **Statistics, Optimization & Information Computing (SOIC)** — Scopus-indexed; scope explicitly covers optimization methods and statistical analysis in markets/finance. Strong fit for the conformal-statistics framing.
- **Expert Systems with Applications** — higher tier (Q1), if the first author wants a stronger line for admissions and the results are solid.
- **IEEE Access / Neural Computing and Applications** — credible mid-tier fallbacks.

Whatever the choice: **must be indexed**, or the citations don't accrue and the exercise is pointless.

---

## 13. Risks & limitations to disclose up front (these prevent rejection)

- **Impact-model realism:** results depend on the simulator's impact dynamics. State the assumptions explicitly; prefer emergent impact (ABIDES) over hand-coded impact.
- **Overfitting / data snooping:** use walk-forward train/validation/test splits; **never** tune on the test window.
- **Transaction costs:** include realistic per-trade costs from the start; results without them are meaningless.
- **No live-trading claim:** frame contributions as methodological and simulator-based. Do not imply real-money efficacy.
- **Conformal exchangeability caveat:** financial time series violate i.i.d.; acknowledge this and prefer block/adaptive calibration, and report coverage empirically rather than assuming it.

---

## 14. Immediate next steps

1. First author completes Task 1 (environment + data verification) and reports back what actually installs and runs.
2. Shaon finalizes the reward function and state feature set (the trader-judgement call).
3. Lock the predictor target (volume-share vs adverse-move) once Task 1 shows what the simulator exposes cleanly.

---

*Plan is intentionally modular: the plain RL-vs-baselines execution paper is the safe core; the conformal layer is the differentiator. Build the core first, add the hook once it works.*
