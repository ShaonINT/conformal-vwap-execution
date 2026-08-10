"""Run the baseline + conformal experiments and print the headline results."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uae.experiments import ExperimentConfig, run_all  # noqa: E402


def main() -> int:
    cfg = ExperimentConfig()
    res = run_all(cfg, out_dir=REPO / "results")
    print("\n=== RESULTS (bps) ===")
    print(res["table"].to_string(index=False))
    print("\n=== COVERAGE ===")
    for k, v in res["coverage"].items():
        print(f"  {k}: {v}")
    print("\n=== FRONTIER (kappa -> mean, std slippage bps) ===")
    for f in res["frontier"]:
        print(f"  kappa={f['kappa']:.3f}  mean={f['slippage_mean_bps']:+.2f}  std={f['slippage_std_bps']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
