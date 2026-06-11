from pathlib import Path
import sys
import traceback

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.Simulation import Simulation
    from core.File_Handling import load_yaml

    def save_table(df, filename):
        path = PROJECT_ROOT / filename
        try:
            df.to_excel(path, index=False)
            print(f"Saved {path.name}")
        except ModuleNotFoundError as exc:
            if exc.name != "openpyxl":
                raise
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path.name} (openpyxl is not installed)")

    # Test single run with OPEX enabled
    sim_cfg = {
        "run_marketenv": False,
        "run_metenv": False,
        "run_capex": False,
        "run_windfarm": False,
        "run_opex": True,
        "opex_dashboard": False,
        "run_revenue": False,
        "run_valuation": False,
        "collect_results": False,
    }

    print("Loading Config.yaml...")
    library_path = PROJECT_ROOT / "examples" / "Inputs" / "HKN"
    config_dict = load_yaml(str(library_path), "Config.yaml")

    # Add required fields if missing
    config_dict["experiment_name"] = "test_opex"
    config_dict["result_directory"] = str(PROJECT_ROOT / "results")
    config_dict["scenario_id"] = "test_001"
    config_dict["scenario_label"] = "test_scenario"
    config_dict["seed"] = 42

    # Add duration fields if missing
    if "Project_Duration_h" not in config_dict:
        config_dict["Project_Duration_h"] = 175200  # 20 years, as given in the example config (Config.yaml)
    if "WF_OperationsStart_h" not in config_dict:
        config_dict["WF_OperationsStart_h"] = 0
    if "WF_OperationsEnd_h" not in config_dict:
        config_dict["WF_OperationsEnd_h"] = 175200

    print("Creating Simulation...")
    sim = Simulation.from_config(
        library_path=str(library_path),
        config=config_dict,
        simulation_config=sim_cfg,
    )

    print("Running simulation...")
    sim.run()
    print("OK: Simulation completed!")

    def get_mode_cost_breakdown_df(opex):
        """Return mode-level cost/intervention rows from the OPEX extras payload."""
        mode_cost_df = getattr(opex, "opex_mode_cost_breakdown_df", pd.DataFrame())
        if isinstance(mode_cost_df, pd.DataFrame) and not mode_cost_df.empty:
            return mode_cost_df.copy()

        extras = getattr(opex, "OPEX_records_extras", None)
        if isinstance(extras, dict):
            mode_cost_rows = extras.get("mode_cost_breakdown")
            if isinstance(mode_cost_rows, list) and mode_cost_rows:
                return pd.DataFrame(mode_cost_rows)

            rows = []
            for window_extras in extras.get("windows", []) or []:
                for row in window_extras.get("mode_cost_breakdown", []) or []:
                    if isinstance(row, dict):
                        rr = dict(row)
                        rr["window_label"] = window_extras.get("window_label")
                        rr["mode"] = window_extras.get("mode")
                        rows.append(rr)
            if rows:
                return pd.DataFrame(rows)

        return pd.DataFrame()

    # Access OPEX diagnostics directly from the environment
    print("\nChecking for OPEX diagnostics...")
    opex = sim.env.opex if hasattr(sim.env, 'opex') else None

    if opex is not None and hasattr(opex, 'vessel_records'):
        vessel_records = opex.vessel_records
        if vessel_records:
            print("OK: Vessel records found!")
            print("\n=== SUMMARY BY VESSEL ===")
            print(vessel_records["summary_by_vessel"])
            print("\n=== BY MODE ===")
            print(vessel_records["by_mode"])

            # Save vessel records to Excel files
            save_table(vessel_records["summary_by_vessel"], "vessel_summary.xlsx")
            save_table(vessel_records["by_mode"], "vessel_by_mode.xlsx")
        else:
            print("WARN: vessel_records is empty or None")
    else:
        print("WARN: No vessel_records attribute found")
    if opex is not None:
        mode_cost_breakdown = get_mode_cost_breakdown_df(opex)
        if not mode_cost_breakdown.empty:
            print("\n=== MODE COST BREAKDOWN ===")
            print(mode_cost_breakdown)

            id_cols = [
                col
                for col in ["component", "mode_id", "task_type"]
                if col in mode_cost_breakdown.columns
            ]
            value_cols = [
                col
                for col in ["N_interventions"]
                if col in mode_cost_breakdown.columns
            ]

            if id_cols and value_cols:
                mode_interventions_summary = (
                    mode_cost_breakdown
                    .groupby(id_cols, as_index=False)[value_cols]
                    .sum()
                    .sort_values(
                        id_cols,
                        ascending=True,
                        ignore_index=True,
                    )
                )

                print("\n=== N INTERVENTIONS PER COMPONENT AND MODE ===")
                print(mode_interventions_summary)

                save_table(
                    mode_interventions_summary,
                    "mode_interventions_summary.xlsx",
                )

            save_table(mode_cost_breakdown, "mode_cost_breakdown.xlsx")
            if id_cols and value_cols:
                print("OK: Mode interventions summary saved")
        else:
            print("mode_cost_breakdown is empty or None")
    else:
        print("No OPEX object found")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
