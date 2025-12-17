#!/usr/bin/env python3
"""
HPC entry point for running the OPEX uncertainty/mean-shift scenario sweep.

Suggested location:
  <repo-root>/examples/HPC/run_opex_uncertainty_sweep_parallel.py
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
log = logging.getLogger("run_opex_uncertainty_sweep")


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
    # Simulation config (from notebook)
    # ---------------------------------------------------------------------
    sim_cfg = {
        "run_marketenv": True,
        "run_metenv": False,
        "run_capex": True,
        "capex_dashboard": False,
        "run_opex": True,
        "opex_dashboard": False,
        "run_lifetime_extension": False,
        "run_revenue": True,
        "run_valuation": True,
        "valuation_dashboard": False,
        "collect_results": True,
    }

    # ---------------------------------------------------------------------
    # Design of Experiments (from notebook)
    # ---------------------------------------------------------------------
    parameter_space = {
        # Mean-shift factors
        "OPEX_overrides.parameters.analytic_ctmc.mean_shift.lambda_factor": [
            1.00, 1.00, 1.00, 1.00, 0.775
        ],
        "OPEX_overrides.parameters.analytic_ctmc.mean_shift.mttr_factor": [
            1.00, 1.00, 1.00, 1.00, 0.75
        ],
        "OPEX_overrides.parameters.analytic_ctmc.mean_shift.mttwL_factor": [
            1.00, 1.00, 1.00, 1.00, 0.70
        ],
        "OPEX_overrides.parameters.analytic_ctmc.mean_shift.tau_factor": [
            1.00, 1.00, 1.00, 1.00, 1.00
        ],

        # Uncertainty
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.lamda_sigma": [
            0.00, 0.55, 0.20, 0.175, 0.10
        ],
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttr_sigma": [
            0.00, 0.20, 0.55, 0.20, 0.175
        ],
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttwL_sigma": [
            0.00, 0.30, 0.80, 0.25, 0.20
        ],

        # Scenario labels
        "Scenario.name": [
            "S0 – Reference",
            "S1 – High reliability uncertainty",
            "S2 – High process uncertainty",
            "S3 – Mature fleet",
            "S4 – Best practice / optimised O&M",
        ],
    }

    zip_groups = {
        "macro_scenarios": [
            # mean shifts
            "OPEX_overrides.parameters.analytic_ctmc.mean_shift.lambda_factor",
            "OPEX_overrides.parameters.analytic_ctmc.mean_shift.mttr_factor",
            "OPEX_overrides.parameters.analytic_ctmc.mean_shift.mttwL_factor",
            "OPEX_overrides.parameters.analytic_ctmc.mean_shift.tau_factor",
            # uncertainties
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.lamda_sigma",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttr_sigma",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttwL_sigma",
            # label
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
        replicates=1000,
        name="OPEX_Uncertainty",
        result_directory=str(RESULT_DIR),
        zip_groups=zip_groups,

        # Parallel execution (safe default is sequential if n_jobs==1)
        execution={"backend": "process", "n_jobs": n_jobs},

        # Required for parallel execution in the example pattern
        debug=False,
    )

    df = exp.run()

    # ---------------------------------------------------------------------
    # Minimal logging for SLURM output
    # ---------------------------------------------------------------------
    print(f"Completed {len(df)} runs")
    print(f"Results written to: {RESULT_DIR}")

    # Optional: show failures summary if those columns exist
    for col in ("scenario_id", "status", "error_message"):
        if col not in df.columns:
            break
    else:
        failures = df[df["status"].astype(str).str.lower().ne("success")]
        if len(failures) > 0:
            print("Some runs did not succeed:")
            print(failures[["scenario_id", "status", "error_message"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
