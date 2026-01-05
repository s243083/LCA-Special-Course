#!/usr/bin/env python3
"""
HPC sweep script for Curtailment uncertainty case study.

Mirrors your local build_experiment(...) call, but is SLURM/HPC-friendly:
- Robust PROJECT_ROOT resolution via __file__
- Uses process-based parallelism with n_jobs taken from SLURM_CPUS_PER_TASK
- Sweeps 3 curtailment scenarios (C0/C1/C2) with Gamma shape/scale ranges
- Keeps the tuple-based parameter values (hashable), as in your local script
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
log = logging.getLogger("run_curtailment_uncertainty_sweep")


def main() -> int:
    # ---------------------------------------------------------------------
    # Resolve repository root robustly for HPC execution
    # Assumes this script lives under the repo, e.g. <repo>/examples/HPC/
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
    # Simulation config (as provided)
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
    # Parameter sweep (Curtailment uncertainty scenarios)
    # NOTE: keep tuples (hashable) for gamma_shape/gamma_scale as in your script
    # ---------------------------------------------------------------------
    parameter_space = {
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_epistemic_uncertainty": (
            True, True, True
        ),
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_aleatory_uncertainty": (
            True, True, True
        ),
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_shape": (
            (0.3, 1.0),   # C0 – Low
            (1.5, 3.5),   # C1 – Mid
            (4.5, 7.0),   # C2 – High
        ),
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_scale": (
            (0.02, 0.08),   # C0 – Low
            (0.015, 0.05),  # C1 – Mid
            (0.008, 0.025), # C2 – High
        ),
        "Scenario.name": (
            "C0 – Low curtailment (integrated grid)",
            "C1 – Mid curtailment (reference)",
            "C2 – High curtailment (bottlenecked grid)",
        ),
    }

    zip_groups = {
        "curtailment_scenarios": [
            "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_epistemic_uncertainty",
            "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_aleatory_uncertainty",
            "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_shape",
            "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_scale",
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
        name="Curtailment_Uncertainty",
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
