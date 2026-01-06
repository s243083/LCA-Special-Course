#!/usr/bin/env python3
"""
HPC sweep script for LTE scenario study (Baseline / EOL1 / EOL2):

- Baseline: no lifetime extension (LTE runs as no-op via apply_lte=false)
- EOL1: moderate reliability degradation + moderate AEP haircut + moderate refurb uplift
- EOL2: substantial reliability degradation + larger AEP haircut + larger refurb uplift

Runs Lifetime Extension + downstream modules (revenue/valuation) as configured.
Uses process-based parallelism (n_jobs from SLURM_CPUS_PER_TASK if available).
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
log = logging.getLogger("run_lte_scenario_sweep")


def main() -> int:
    # ---------------------------------------------------------------------
    # Resolve repository root robustly for HPC execution
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
    # Simulation config
    # ---------------------------------------------------------------------
    sim_cfg = {
        "run_marketenv": True,
        "run_metenv": False,
        "run_capex": True,
        "capex_dashboard": False,
        "run_opex": True,
        "opex_dashboard": False,
        "run_lifetime_extension": True,  # keep TRUE; baseline becomes no-op via apply_lte=false
        "run_revenue": True,
        "run_valuation": True,
        "valuation_dashboard": False,
        "collect_results": True,
    }

    # ---------------------------------------------------------------------
    # Scenario definitions (zipped together)
    #
    # NOTE on units:
    # - LTE.py expects AEP haircut in *fractions* (e.g., -0.036 = -3.6%).
    # - Refurb uplift is €/turbine and then multiplied by n_turbines in LTE.py.
    # ---------------------------------------------------------------------
    parameter_space = {
        # Human-readable label
        "Scenario.name": [
            "Baseline – No LTE",
            "EOL1 – Moderate degradation",
            "EOL2 – Substantial degradation",
        ],

        # Enable / disable LTE
        "LTE_overrides.lte_input.LTE.apply_lte": [
            False,
            True,
            True,
        ],

        # Extension horizon (hours)
        # Baseline: keep extension_h=0 to avoid any accidental extension logic
        "LTE_overrides.lte_input.LTE.extension_h": [
            0,
            43000,
            43000,
        ],

        # ----------------------------
        # AEP haircut distribution (fractional)
        # ----------------------------
        "LTE_overrides.lte_input.LTE.aep_haircut.mu": [
            0.0,
            -0.020,   # moderate AEP loss (=-2.0%)
            -0.040,   # substantial AEP loss (=-4.0%)
        ],
        "LTE_overrides.lte_input.LTE.aep_haircut.sigma": [
            0.0,
            0.010,
            0.015,
        ],
        # Keep bounds conservative and code-consistent
        "LTE_overrides.lte_input.LTE.aep_haircut.min": [
            0.0,     # irrelevant when sigma=0, but keep consistent
            -0.30,
            -0.30,
        ],
        "LTE_overrides.lte_input.LTE.aep_haircut.max": [
            0.0,
            0.0,
            0.0,
        ],

        # ----------------------------
        # Reliability / OPEX mean-shift factors (extension regime only)
        # ----------------------------
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.flag_apply": [
            False,
            True,
            True,
        ],
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.lambda_factor": [
            1.00,
            1.20,
            1.35,
        ],
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.mttr_factor": [
            1.00,
            1.10,
            1.20,
        ],
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.mttwL_factor": [
            1.00,
            1.10,
            1.20,
        ],
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.tau_factor": [
            1.00,
            1.00,
            1.00,
        ],
        "LTE_overrides.lte_input.LTE.opex_extension.analytic_ctmc.mean_shift.p_access_factor": [
            1.00,
            0.90,
            0.80,
        ],

        # ----------------------------
        # Refurbishment uplift distribution (€/turbine)
        # NOTE: LTE.py samples refurb_uplift and multiplies by n_turbines
        # ----------------------------
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.dist": [
            "fixed",        # baseline
            "normal_trunc",  # EOL1
            "normal_trunc",  # EOL2
        ],
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.value": [
            0.0,    # only used when dist=fixed
            None,
            None,
        ],
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.mu": [
            0.0,        # baseline (unused if dist=fixed, but safe)
            1_500_000,  # moderate uplift
            3_000_000,  # substantial uplift
        ],
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.sigma": [
            0.0,
            500_000,
            1_000_000,
        ],
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.min": [
            0.0,
            0.0,
            0.0,
        ],
        "LTE_overrides.lte_input.LTE.costs.refurb_uplift.max": [
            0.0,     # baseline (unused if fixed)
            None,    # let it float or set a cap if you prefer
            None,
        ],
    }

    # Zip everything so we get exactly 3 coherent scenarios (not a Cartesian product)
    zip_groups = {
        "lte_scenarios": list(parameter_space.keys())
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
        replicates=500,
        name="LTE_Scenario_Experiment",
        result_directory=str(RESULT_DIR),
        zip_groups=zip_groups,
        execution={"backend": "process", "n_jobs": n_jobs},
        debug=False,
    )

    df = exp.run()

    print(f"Completed {len(df)} runs")
    print(f"Results written to: {RESULT_DIR}")

    # Optional quick failure summary
    if hasattr(df, "columns") and all(c in df.columns for c in ("status", "scenario_id")):
        failures = df[df["status"].astype(str).str.lower().ne("success")]
        if len(failures) > 0:
            cols = [c for c in ("scenario_id", "status", "error_message") if c in df.columns]
            print("Some runs did not succeed:")
            print(failures[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
