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

    # Toggle which modules to run, dashboards, result collection, etc.
    # (Adjust flags / keys to match your setup if they differ.)
    sim_cfg = {
        "run_marketenv": True,
        "run_metenv": False,
        "run_capex": True,
        "capex_dashboard": False,
        "run_curtailment": True,
        "run_opex": True,
        "opex_dashboard": False,
        "run_lifetime_extension": False,
        "run_revenue": True,
        "run_valuation": True,
        "valuation_dashboard": False,
        "collect_results": True,
    }

    parameter_space = {
        # Explicit uncertainty activation (no implicit defaults)
        # If you'd prefer C0 (no curtailment) to be fully deterministic, set the first entries to False.
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_epistemic_uncertainty": (
            False,  # C0
            True,  # C1
            True,  # C2
            True,  # C3
            True,  # C4
            True,  # C5
        ),
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.apply_aleatory_uncertainty": (
            False,  # C0
            True,  # C1
            True,  # C2
            True,  # C3
            True,  # C4
            True,  # C5
        ),

        # IMPORTANT: use tuples (hashable), not lists (unhashable)
        # Table mapping:
        # - gamma_shape  -> alpha range [min, max]
        # - gamma_scale  -> theta range [min, max]
        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_shape": (
            (1.0, 1.0),        # C0 — Reference
            (0.414, 0.506),    # C1 — Low transmission constraints
            (0.747, 0.913),    # C2 — Medium transmission constraints
            (1.395, 1.705),    # C3 — High transmission constraints
            (0.567, 0.693),    # C4 — Very high market curtailment
            (0.351, 0.429),    # C5 — Storage solutions
        ),

        "Curtailment_overrides.curt_input.Curtailment.reduceProduction.gamma_scale": (
            (0.0, 0.0),                    # C0 — Reference
            (0.01503, 0.01837),             # C1 — 1.503–1.837 %
            (0.04014, 0.04906),             # C2 — 4.014–4.906 %
            (0.05994, 0.07326),             # C3 — 5.994–7.326 %
            (0.04653, 0.05687),             # C4 — 4.653–5.687 %
            (0.01620, 0.01980),             # C5 — 1.62–1.98 %
        ),
        "Scenario.name": (
            "C0 — Reference (no curtailment)",
            "C1 — Low transmission constraints",
            "C2 — Medium transmission constraints",
            "C3 — High transmission constraints",
            "C4 — Very high market curtailment occurrence",
            "C5 — Storage solutions development",
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
        replicates=1000,
        name="Curtailment",
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
