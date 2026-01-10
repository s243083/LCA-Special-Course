#!/usr/bin/env python3
"""
HPC entry point for running failure-rate epistemic uncertainty scenarios
using Gamma-distributed lambda with prescribed coefficient of variation (CV).

Scenarios match the LaTeX table:
  Reference: CV=0.00 (deterministic)
  R1:        CV=0.15
  R2:        CV=0.30
  R3:        CV=0.60
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
log = logging.getLogger("run_opex_failure_rate_epistemic_gammacv")


def main() -> int:
    # ---------------------------------------------------------------------
    # Resolve repository root
    # examples/HPC -> examples -> repo root
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
    # Simulation config (same pattern as your existing HPC runner)
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
    # Design of Experiments: Failure-rate epistemic Gamma(CV) scenarios
    # ---------------------------------------------------------------------
    parameter_space = {
        # Keep deterministic mean-shifts neutral (alpha_lambda = 1.0 for all)
        "OPEX_overrides.parameters.analytic_ctmc.mean_shift.lambda_factor": [
            1.00,  # Reference
            1.00,  # R1
            1.00,  # R2
            1.00,  # R3
        ],

        # Explicitly toggle epistemic uncertainty on/off
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.flag_apply_epistemic_lambda": [
            False,  # Reference (no epistemic uncertainty)
            True,   # R1
            True,   # R2
            True,   # R3
        ],

        # Gamma CV for lambda (matches your YAML key used by OPEX.py)
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.lambda_gamma_cv": [
            0.00,  # Reference
            0.20,  # R1 -- Low epistemic uncertainty
            0.50,  # R2 -- Mid epistemic uncertainty
            1.40,  # R3 -- High epistemic uncertainty
        ],

        # Ensure process uncertainty is OFF for this experiment (optional but keeps it “pure”)
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.flag_apply_process": [
            False, False, False, False
        ],
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttr_sigma": [
            0.00, 0.00, 0.00, 0.00
        ],
        "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttwL_sigma": [
            0.00, 0.00, 0.00, 0.00
        ],

        # Labels
        "Scenario.name": [
            "Reference",
            "R1",
            "R2",
            "R3",
        ],
    }

    zip_groups = {
        "failure_rate_epistemic_gammacv": [
            "OPEX_overrides.parameters.analytic_ctmc.mean_shift.lambda_factor",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.flag_apply_epistemic_lambda",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.lambda_gamma_cv",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.flag_apply_process",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttr_sigma",
            "OPEX_overrides.parameters.analytic_ctmc.uncertainty.mttwL_sigma",
            "Scenario.name",
        ]
    }

    # ---------------------------------------------------------------------
    # Parallel execution settings (SLURM-friendly, same pattern)
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

        # “replicates” here is what drives repeated draws of the uncertain lambda
        # (i.e., repeated epistemic realisations).
        replicates=5000,

        name="OPEX_Reliability",
        result_directory=str(RESULT_DIR),
        zip_groups=zip_groups,

        execution={"backend": "process", "n_jobs": n_jobs},
        debug=False,
    )

    df = exp.run()

    print(f"Completed {len(df)} runs")
    print(f"Results written to: {RESULT_DIR}")

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
