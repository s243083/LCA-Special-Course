#!/usr/bin/env python3
"""
HPC sweep script for the LTE case study:
- Varies LTE_overrides.lte_input.lambda_factor over [1.00, 0.90, 0.70]
- Runs Lifetime Extension + downstream modules (revenue/valuation) as configured
- Uses process-based parallelism (n_jobs from SLURM_CPUS_PER_TASK if available)
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_lte_lambda_factor_sweep")


def main() -> int:
    # ---------------------------------------------------------------------
    # Resolve repository root robustly for HPC execution
    # Assumes this script lives somewhere under the repo; adjust parents[] if needed.
    # Example: <repo>/examples/HPC/<this_script>.py  -> parents[2] == <repo>
    # ---------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    # Simulation config (mirrors your local script)
    # ---------------------------------------------------------------------
    sim_cfg = {
        "run_marketenv": True,
        "run_metenv": False,
        "run_capex": True,
        "capex_dashboard": False,
        "run_opex": True,
        "opex_dashboard": False,
        "run_lifetime_extension": True,
        "run_revenue": True,
        "run_valuation": True,
        "valuation_dashboard": True,
        "collect_results": True,
    }

    # ---------------------------------------------------------------------
    # Parameter sweep (LTE lambda_factor scenarios)
    # ---------------------------------------------------------------------
    parameter_space = {
        "LTE_overrides.lte_input.lambda_factor": [
            1.00,  # S0 – Reference
            0.90,  # S1 – reduced failure rates
            0.70,  # S2 – more reduced failure rates
        ],
        "Scenario.name": [
            "S0 – Reference",
            "S1 – reduced failure rates",
            "S2 – more reduced failure rates",
        ],
    }

    zip_groups = {
        "lte_lambda_factor_scenarios": [
            "LTE_overrides.lte_input.lambda_factor",
            "Scenario.name",
        ]
    }

    # ---------------------------------------------------------------------
    # Parallel execution settings (SLURM-friendly)
    # ---------------------------------------------------------------------
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if n_jobs <= 0:
        n_jobs = 1

    log.info("PROJECT_ROOT=%s", PROJECT_ROOT)
    log.info("LIBRARY_PATH=%s", LIBRARY_PATH)
    log.info("RESULT_DIR=%s", RESULT_DIR)
    log.info("n_jobs=%s", n_jobs)

    # ---------------------------------------------------------------------
    # Build + run experiment
    # ---------------------------------------------------------------------
    exp = build_experiment(
        library_path=str(LIBRARY_PATH),
        base_config_path="Config.yaml",
        simulation_config=sim_cfg,
        parameter_space=parameter_space,
        base_seed=42,
        replicates=10,
        name="LTE_LambdaFactor_Sweep",
        result_directory=str(RESULT_DIR),
        zip_groups=zip_groups,
        execution={"backend": "process", "n_jobs": n_jobs},
        debug=False,
    )

    df = exp.run()

    print(f"Completed {len(df)} runs")
    print(f"Results written to: {RESULT_DIR}")

    # Optional quick failure summary (if these columns exist)
    if hasattr(df, "columns") and all(c in df.columns for c in ("status", "scenario_id")):
        failures = df[df["status"].astype(str).str.lower().ne("success")]
        if len(failures) > 0:
            cols = [c for c in ("scenario_id", "status", "error_message") if c in df.columns]
            print("Some runs did not succeed:")
            print(failures[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
