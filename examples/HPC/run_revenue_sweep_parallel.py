#!/usr/bin/env python3
"""
HPC entry point for running a WINPACT revenue parameter sweep
with optional parallel execution.

Location:
  <repo-root>/examples/HPC/run_revenue_sweep_parallel.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)



def main() -> int:
    # ---------------------------------------------------------------------
    # Resolve repository root
    # examples/HPC -> examples -> repo root
    # ---------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Make repo importable
    sys.path.insert(0, str(PROJECT_ROOT))

    from core.Simulation import build_experiment  # noqa: E402

    # ---------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------
    LIBRARY_PATH = PROJECT_ROOT / "examples" / "Inputs" / "HKN"
    RESULT_DIR = PROJECT_ROOT / "results"

    if not LIBRARY_PATH.exists():
        raise FileNotFoundError(f"LIBRARY_PATH not found: {LIBRARY_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Simulation configuration
    # ---------------------------------------------------------------------
    sim_cfg = {
        "run_marketenv": True,
        "run_metenv": False,
        "run_capex": True,
        "capex_dashboard": False,
        "run_windfarm": True,
        "run_opex": True,
        "opex_dashboard": False,
        "run_lifetime_extension": False,
        "run_revenue": True,
        "run_valuation": True,
        "valuation_dashboard": False,
        "collect_results": True,
    }

    # ---------------------------------------------------------------------
    # Design of Experiments
    # ---------------------------------------------------------------------
    parameter_space = {
        "Revenue_overrides.strike_price": [80, 100],
        "Revenue_overrides.scheme_type": ["FiT", "CfD"],
        "Scenario.name": ["FiT", "CfD"],
    }

    zip_groups = {
        "macro_scenarios": [
            "Revenue_overrides.strike_price",
            "Revenue_overrides.scheme_type",
            "Scenario.name",
        ]
    }

    # ---------------------------------------------------------------------
    # Parallel execution settings
    # ---------------------------------------------------------------------
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if n_jobs <= 0:
        n_jobs = 1

    # ---------------------------------------------------------------------
    # Build + run experiment
    # ---------------------------------------------------------------------
    exp = build_experiment(
        library_path=str(LIBRARY_PATH),
        base_config_path="Config.yaml",
        parameter_space=parameter_space,
        simulation_config=sim_cfg,
        base_seed=42,
        replicates=1,
        name="Revenue_Sweep_Parallel",
        result_directory=str(RESULT_DIR),
        zip_groups=zip_groups,

        # Parallel execution (safe default is sequential)
        execution={"backend": "process", "n_jobs": n_jobs},

        # Required for parallel execution
        debug=False,
    )

    df = exp.run()

    # ---------------------------------------------------------------------
    # Minimal logging for SLURM output
    # ---------------------------------------------------------------------
    print(f"Completed {len(df)} runs")
    print(f"Results written to: {RESULT_DIR}")


    print("=== DF columns ===")
    print(df.columns.tolist())
    print("=== DF head ===")
    print(df.head())

    print("=== RESULT_DIR ===")
    print(RESULT_DIR)

    print("=== RESULT_DIR contents (maxdepth=3) ===")
    for p in RESULT_DIR.rglob("*"):
        if p.is_file():
            print(p)

        return 0

    print(df[["scenario_id", "status", "error_message"]])

if __name__ == "__main__":
    raise SystemExit(main())
