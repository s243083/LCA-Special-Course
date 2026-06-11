#!/usr/bin/env python3
"""
run_opex_diagnostics.py  --  Standalone OPEX diagnostic runner.

Boots the simulation with only the OPEX module active and saves the following
tables to results/opex/:

    windows_overview.xlsx            -- per-window farm availability + total costs
    cost_breakdown.xlsx              -- CM / PM / transport / labour / spares split
    mode_cost_breakdown.xlsx         -- row-per-intervention detail (component x mode)
    mode_interventions_summary.xlsx  -- N_interventions grouped by component & mode
    component_cost_breakdown.xlsx    -- costs rolled up per component
    downtime_breakdown.xlsx          -- downtime split by logistics / weather / repair state

Falls back to .csv if openpyxl is not installed.

NOTE: vessel_records is NOT produced by the current OPEX implementation.
The aggregated outputs are the opex_*_df DataFrames built by build_extras_tables().

HOW TO RUN
----------
    cd <repo-root>
    python examples/HPC/run_opex_diagnostics.py
"""

from pathlib import Path
import sys
import traceback

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "opex"

try:
    from core.Simulation import Simulation
    from core.File_Handling import load_yaml

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def save_table(df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to Excel; fall back to CSV if openpyxl is missing."""
        path = OUTPUT_DIR / filename
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_excel(path, index=False)
            print(f"  Saved {path.relative_to(PROJECT_ROOT)}")
        except ModuleNotFoundError as exc:
            if exc.name != "openpyxl":
                raise
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path.relative_to(PROJECT_ROOT)}  (openpyxl not installed)")

    def build_vessel_tables(opex) -> tuple:
        """
        Reconstruct vessel-level summaries equivalent to the original
        vessel_records["summary_by_vessel"] and vessel_records["by_mode"].

        vessel_records is not produced by the current OPEX implementation, so
        we derive the same information by combining:
          - opex._load_vessels()            -> vessel specs (day rate, distance, ...)
          - opex._load_maintenance_specs()  -> which vessel capability each mode needs
          - mode_cost_breakdown             -> costs per mode
        """
        try:
            vessels = opex._load_vessels()
            specs   = opex._load_maintenance_specs()
        except Exception as exc:
            print(f"  WARN: could not load vessel/maintenance specs: {exc}")
            return pd.DataFrame(), pd.DataFrame()

        mcb = get_mode_cost_breakdown_df(opex)
        if mcb.empty:
            return pd.DataFrame(), pd.DataFrame()

        # map (component, mode_id) -> vessel capability requested
        mode_to_cap = {
            (s.component, str(s.mode_id)): (s.preferred_vessels[0] if s.preferred_vessels else "CTV")
            for s in specs
        }

        # map capability -> first vessel name that has it
        cap_to_name = {}
        for v_name, v_spec in vessels.items():
            cap_to_name.setdefault(v_spec.capability, v_name)

        # annotate the mode breakdown with vessel info
        mcb2 = mcb.copy()
        mcb2["vessel_capability"] = mcb2.apply(
            lambda r: mode_to_cap.get((r.get("component"), str(r.get("mode_id", ""))), "unknown"),
            axis=1,
        )
        mcb2["vessel_name"] = mcb2["vessel_capability"].map(
            lambda c: cap_to_name.get(c, c)
        )

        cost_cols = [c for c in ["N_interventions", "transport_eur", "labour_eur",
                                  "spares_eur", "total_eur"] if c in mcb2.columns]

        # --- summary_by_vessel ---
        summary = (
            mcb2.groupby(["vessel_name", "vessel_capability"], as_index=False)[cost_cols]
            .sum()
            .sort_values("vessel_name", ignore_index=True)
        )
        # attach vessel specs (day rate, distance)
        summary["day_rate_eur_per_day"] = summary["vessel_name"].map(
            {v.name: v.day_rate_eur for v in vessels.values()}
        )
        summary["port_distance_km"] = summary["vessel_name"].map(
            {v.name: v.base_distance_km for v in vessels.values()}
        )

        # --- by_mode ---
        keep = ["vessel_name", "vessel_capability", "component", "mode_id", "task_type"] + cost_cols
        by_mode = mcb2[[c for c in keep if c in mcb2.columns]].copy()

        return summary, by_mode

    def get_mode_cost_breakdown_df(opex) -> pd.DataFrame:
        """
        Pull mode-level cost / intervention rows from the OPEX object.
        Tries three attribute paths to stay robust across code versions.
        """
        # Path 1: dedicated DataFrame attribute
        mode_cost_df = getattr(opex, "opex_mode_cost_breakdown_df", pd.DataFrame())
        if isinstance(mode_cost_df, pd.DataFrame) and not mode_cost_df.empty:
            return mode_cost_df.copy()

        # Path 2: flat list inside OPEX_records_extras
        extras = getattr(opex, "OPEX_records_extras", None)
        if isinstance(extras, dict):
            rows = extras.get("mode_cost_breakdown")
            if isinstance(rows, list) and rows:
                return pd.DataFrame(rows)

            # Path 3: per-window breakdown nested inside extras["windows"]
            nested_rows = []
            for window_extras in extras.get("windows", []) or []:
                for row in window_extras.get("mode_cost_breakdown", []) or []:
                    if isinstance(row, dict):
                        r = dict(row)
                        r["window_label"] = window_extras.get("window_label")
                        r["mode"] = window_extras.get("mode")
                        nested_rows.append(r)
            if nested_rows:
                return pd.DataFrame(nested_rows)

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Simulation setup -- OPEX only
    # -------------------------------------------------------------------------

    sim_cfg = {
        "run_marketenv":  False,
        "run_metenv":     False,
        "run_capex":      False,
        "run_windfarm":   False,
        "run_opex":       True,
        "opex_dashboard": False,
        "run_revenue":    False,
        "run_valuation":  False,
        "collect_results": False,
    }

    library_path = PROJECT_ROOT / "examples" / "Inputs" / "HKN"

    print("Loading Config.yaml ...")
    config_dict = load_yaml(str(library_path), "Config.yaml")

    # Fields required by Simulation.from_config() that Config.yaml does not store
    config_dict.setdefault("experiment_name",   "opex_diagnostics")
    config_dict.setdefault("result_directory",  str(PROJECT_ROOT / "results"))
    config_dict.setdefault("scenario_id",       "diag_001")
    config_dict.setdefault("scenario_label",    "opex_diagnostic_run")
    config_dict.setdefault("seed",              42)

    # Duration fields (20-year operational window: 20 * 8760 = 175 200 h)
    config_dict.setdefault("Project_Duration_h",    175_200)
    config_dict.setdefault("WF_OperationsStart_h",  0)
    config_dict.setdefault("WF_OperationsEnd_h",    175_200)

    print("Creating Simulation ...")
    sim = Simulation.from_config(
        library_path=str(library_path),
        config=config_dict,
        simulation_config=sim_cfg,
    )

    print("Running simulation ...")
    sim.run()
    print("Simulation complete.\n")

    opex = getattr(sim.env, "opex", None)

    if opex is None:
        print("ERROR: OPEX object not found on sim.env -- nothing to report.")
    else:
        # ---------------------------------------------------------------------
        # Table 1 & 2: vessel summaries
        # (vessel_records is not produced by the current OPEX implementation;
        #  we reconstruct equivalent tables from the internal loaders + mode costs)
        # ---------------------------------------------------------------------
        summary_by_vessel, by_mode = build_vessel_tables(opex)
        if not summary_by_vessel.empty:
            print("=== SUMMARY BY VESSEL ===")
            print(summary_by_vessel.to_string(index=False))
            print()
            save_table(summary_by_vessel, "vessel_summary.xlsx")
        else:
            print("INFO: vessel summary could not be built.")

        if not by_mode.empty:
            print("=== BY MODE (WITH VESSEL) ===")
            print(by_mode.to_string(index=False))
            print()
            save_table(by_mode, "vessel_by_mode.xlsx")

        # ---------------------------------------------------------------------
        # Table 4: windows overview (farm availability + total costs per window)
        # ---------------------------------------------------------------------
        windows_df = getattr(opex, "opex_windows_df", pd.DataFrame())
        if not windows_df.empty:
            print("=== WINDOWS OVERVIEW ===")
            print(windows_df.to_string(index=False))
            print()
            save_table(windows_df, "windows_overview.xlsx")
        else:
            print("INFO: opex_windows_df is empty.")

        # ---------------------------------------------------------------------
        # Table 5: cost breakdown (CM / PM / transport / labour / spares)
        # ---------------------------------------------------------------------
        breakdown_df = getattr(opex, "opex_breakdown_df", pd.DataFrame())
        if not breakdown_df.empty:
            print("=== COST BREAKDOWN ===")
            print(breakdown_df.to_string(index=False))
            print()
            save_table(breakdown_df, "cost_breakdown.xlsx")
        else:
            print("INFO: opex_breakdown_df is empty.")

        # ---------------------------------------------------------------------
        # Table 6: mode cost breakdown (row per component x failure-mode / PM)
        # ---------------------------------------------------------------------
        mcb = get_mode_cost_breakdown_df(opex)
        if not mcb.empty:
            print("=== MODE COST BREAKDOWN ===")
            print(mcb.to_string(index=False))
            print()
            save_table(mcb, "mode_cost_breakdown.xlsx")

            # Table 4: interventions count grouped by component & mode
            id_cols  = [c for c in ["component", "mode_id", "task_type"] if c in mcb.columns]
            val_cols = [c for c in ["N_interventions"]                   if c in mcb.columns]
            if id_cols and val_cols:
                summary = (
                    mcb.groupby(id_cols, as_index=False)[val_cols]
                    .sum()
                    .sort_values(id_cols, ascending=True, ignore_index=True)
                )
                print("=== N INTERVENTIONS PER COMPONENT AND MODE ===")
                print(summary.to_string(index=False))
                print()
                save_table(summary, "mode_interventions_summary.xlsx")
        else:
            print("INFO: mode_cost_breakdown is empty.")

        # ---------------------------------------------------------------------
        # Table 7: component cost breakdown
        # ---------------------------------------------------------------------
        comp_df = getattr(opex, "opex_component_cost_breakdown_df", pd.DataFrame())
        if not comp_df.empty:
            print("=== COMPONENT COST BREAKDOWN ===")
            print(comp_df.to_string(index=False))
            print()
            save_table(comp_df, "component_cost_breakdown.xlsx")
        else:
            print("INFO: opex_component_cost_breakdown_df is empty.")

        # ---------------------------------------------------------------------
        # Table 8: downtime breakdown (logistics / weather / repair fractions)
        # ---------------------------------------------------------------------
        dt_df = getattr(opex, "opex_downtime_breakdown_df", pd.DataFrame())
        if not dt_df.empty:
            print("=== DOWNTIME BREAKDOWN ===")
            print(dt_df.to_string(index=False))
            print()
            save_table(dt_df, "downtime_breakdown.xlsx")
        else:
            print("INFO: opex_downtime_breakdown_df is empty.")

        # ---------------------------------------------------------------------
        # Surface any additional repair-state / availability attributes
        # ---------------------------------------------------------------------
        extra_attrs = [
            a for a in dir(opex)
            if any(kw in a.lower() for kw in ("state", "avail", "repair", "downtime"))
            and not a.startswith("_")
        ]
        if extra_attrs:
            print("\nOther repair/availability attributes on the OPEX object:")
            for a in extra_attrs:
                val = getattr(opex, a, None)
                shape = getattr(val, "shape", None) or (len(val) if hasattr(val, "__len__") else "")
                print(f"  opex.{a}  ->  {type(val).__name__}  {shape}")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
