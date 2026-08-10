"""Train PPO across seeds and append its results to the table.

Reproduces the paper's RL comparator: an off-the-shelf PPO agent trained over a
few seeds, evaluated on held-out sessions. Expected outcome -- high across-seed
variance, no reliable win over the simple schedules. Requires stable-baselines3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uae.experiments import ExperimentConfig  # noqa: E402
from uae.ppo import train_ppo, evaluate_ppo    # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=40_000)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    p = cfg.params
    # train on the same 'train' seed range used elsewhere; eval on the 'test' range
    train_pool = range(cfg.seed, cfg.seed + cfg.n_train)
    test_pool = range(cfg.seed + cfg.n_train + cfg.n_calib,
                      cfg.seed + cfg.n_train + cfg.n_calib + cfg.n_test)

    per_seed = []
    for s in args.seeds:
        model = train_ppo(p, cfg.parent_qty, cfg.direction, train_pool,
                          total_timesteps=args.timesteps, seed=s)
        r = evaluate_ppo(model, p, cfg.parent_qty, cfg.direction, test_pool)
        print(f"PPO seed {s}: mean slippage {r.slippage_mean_bps:+.2f} bps "
              f"(within-seed std {r.slippage_std_bps:.2f})")
        per_seed.append(r.slippage_mean_bps)

    per_seed = np.asarray(per_seed)
    print(f"\nPPO across seeds: mean {per_seed.mean():+.2f} bps, "
          f"ACROSS-SEED std {per_seed.std():.2f} bps (n={len(per_seed)})")

    # append/update a PPO row in the results table
    table_path = REPO / "results" / "results_table.csv"
    table = pd.read_csv(table_path)
    table = table[table["method"] != "PPO (across-seed)"]
    ppo_row = {
        "method": "PPO (across-seed)",
        "slippage_mean_bps": float(per_seed.mean()),
        "slippage_std_bps": float(per_seed.std()),
        "is_mean_bps": float("nan"),
        "is_std_bps": float("nan"),
    }
    table = pd.concat([table, pd.DataFrame([ppo_row])], ignore_index=True)
    table.to_csv(table_path, index=False)
    pd.DataFrame({"ppo_seed_mean_bps": per_seed}).to_csv(
        REPO / "results" / "ppo_per_seed.csv", index=False)
    print(f"Updated {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
