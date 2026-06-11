import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl", "matplotlib", "pyyaml", "--break-system-packages"])

# =============================================================================
# main.py — Wind Farm LCA Module Entry Point
# =============================================================================

import yaml
import os

_yaml_path = os.path.join(os.path.dirname(__file__), "inventory_codes.yaml")
with open(_yaml_path, "r", encoding="utf-8") as _f:
    INVENTORY_CODES = yaml.safe_load(_f)

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    IMPACT_COLUMNS = yaml.safe_load(_f)["IMPACT_COLUMNS"]

from inventory_masses import INVENTORY_MASSES
from lca_engine import (
    load_ecoinvent_database,
    resolve_emission_factors,
    merge_inventories,
    calculate_impacts,
    aggregate_by_stage,
    validate_inventory_coverage,
)
from results import (
    print_summary_table,
    print_co2_comparison_table,
    plot_materials_gwp,
    plot_stage_gwp_pie,
    print_full_emissions_table,
)


def confirm_impact_settings():
    """
    Show the current impact method settings from config.py and ask the user
    to confirm before proceeding.

    Returns:
        bool: True if user confirms, False if user wants to exit.
    """
    print("\n" + "=" * 60)
    print("  IMPACT METHOD CONFIRMATION")
    print("=" * 60)
    print("  The following impact indicator(s) are configured in config.py:\n")

    for label, definition in IMPACT_COLUMNS.items():
        print(f"  [{label}]")
        print(f"    Method    : {definition['method']}")
        print(f"    Category  : {definition['category']}")
        print(f"    Indicator : {definition['indicator']}")
        print(f"    Unit      : {definition['unit']}")
        print()

    print("  To change these, edit IMPACT_COLUMNS in config.py before running.\n")

    while True:
        answer = input("  Are these settings correct? (yes / no): ").strip().lower()
        if answer in ("yes", "y"):
            print()
            return True
        elif answer in ("no", "n"):
            print("\n  Please update IMPACT_COLUMNS in config.py and run again.\n")
            return False
        else:
            print("  Please type 'yes' or 'no'.")


def select_stages():
    """
    Show a numbered menu of available life stages and ask the user which
    ones to include in the analysis.

    Returns:
        list of str: names of selected life stages
    """
    all_stages = list(INVENTORY_CODES.keys())

    print("  Select life stages to analyse:")
    for i, stage in enumerate(all_stages, start=1):
        print(f"    {i}. {stage}")
    print()

    while True:
        raw = input("  Enter numbers separated by commas, or press Enter to select all: ").strip()

        if raw == "":
            print(f"  → All {len(all_stages)} stages selected.\n")
            return all_stages

        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            selected = []
            valid = True
            for idx in indices:
                if 1 <= idx <= len(all_stages):
                    selected.append(all_stages[idx - 1])
                else:
                    print(f"  [ERROR] '{idx}' is not a valid option. "
                          f"Choose between 1 and {len(all_stages)}.")
                    valid = False
                    break
            if valid and selected:
                print(f"  → Selected: {', '.join(selected)}\n")
                return selected
        except ValueError:
            print("  [ERROR] Please enter numbers separated by commas (e.g. 1,3,5).\n")


def select_scope():
    """
    Ask the user which scope(s) to analyse.

    Returns:
        list of str: selected scopes from ["per_turbine", "full_farm", "per_FU"]
    """
    print("  Select analysis scope:")
    print("    1. Per turbine")
    print("    2. Full farm")
    print("    3. Per FU (functional unit)")
    print("    4. Per turbine + Full farm")
    print("    5. Per turbine + Per FU")
    print("    6. Full farm + Per FU")
    print("    7. All (Per turbine + Full farm + Per FU)")
    print()

    options = {
        "1": ["per_turbine"],
        "2": ["full_farm"],
        "3": ["per_FU"],
        "4": ["per_turbine", "full_farm"],
        "5": ["per_turbine", "per_FU"],
        "6": ["full_farm", "per_FU"],
        "7": ["per_turbine", "full_farm", "per_FU"],
    }

    while True:
        raw = input("  Enter a number (1-7): ").strip()
        if raw in options:
            selected = options[raw]
            labels = [s.replace("_", " ") for s in selected]
            print(f"  → Scope: {', '.join(labels)}\n")
            return selected
        else:
            print("  [ERROR] Please enter a number between 1 and 7.\n")


def run_analysis(selected_stages, scope, resolved_efs):
    """
    Run the full LCA pipeline for one scope.

    Args:
        selected_stages (list of str): life stages chosen by the user
        scope (str): "per_turbine", "full_farm", or "per_FU"
        resolved_efs (dict): output of resolve_emission_factors()
    """
    scope_label = scope.replace("_", " ").title()
    print(f"\n{'─' * 60}")
    print(f"  Running analysis: {scope_label}")
    print(f"{'─' * 60}")

    merged = merge_inventories(INVENTORY_CODES, INVENTORY_MASSES, selected_stages, scope)
    print(f"  Components matched: {len(merged)}")

    if not merged:
        print("  [WARNING] No components were matched.\n")
        return

    impact_results = calculate_impacts(merged, resolved_efs)
    print(f"  Components with valid impacts: {len(impact_results)}")

    if not impact_results:
        print("  [WARNING] No impacts could be calculated.\n")
        return

    aggregated = aggregate_by_stage(impact_results)

    print_summary_table(aggregated, scope)
    print_full_emissions_table(aggregated, scope)

    print("  Generating charts...")
    plot_materials_gwp(aggregated, scope)
    plot_stage_gwp_pie(aggregated, scope)

    return aggregated


def main():
    """Main entry point — orchestrates the full LCA analysis."""

    # --- Step 1: Confirm impact method settings ---
    if not confirm_impact_settings():
        raise SystemExit(0)

    # --- Step 2: Load ecoinvent database ---
    print("  Loading ecoinvent database...")
    ecoinvent_db, n_processes = load_ecoinvent_database()
    print(f"  Loaded {n_processes} processes from ecoinvent database.\n")

    # --- Step 3: Validate inventory coverage ---
    missing = validate_inventory_coverage(INVENTORY_CODES, ecoinvent_db)
    if missing:
        print(f"  [NOTE] {len(missing)} UID(s) in inventory_codes.py "
              f"were not found in the database:")
        for comp_name, life_stage, code in missing:
            print(f"    - {comp_name} ({life_stage}): {code}")
        print()

    # --- Step 4: Stage selection ---
    selected_stages = select_stages()

    # --- Step 5: Scope selection ---
    scopes = select_scope()

    # --- Step 6: Resolve emission factors once (scope-independent) ---
    print("  Resolving emission factors...")
    resolved_efs = resolve_emission_factors(INVENTORY_CODES, selected_stages, ecoinvent_db)
    print(f"  Resolved {len(resolved_efs)} unique UIDs.\n")

    # --- Step 7: Run analysis for each scope ---
    results = {}
    for scope in scopes:
        aggregated = run_analysis(selected_stages, scope, resolved_efs)
        if aggregated is not None:
            results[scope] = aggregated

    # --- Step 8: CO2 comparison table (when per_turbine + full_farm both selected) ---
    if "per_turbine" in results and "full_farm" in results:
        print_co2_comparison_table(results["per_turbine"], results["full_farm"])

    print("\n  Analysis complete.\n")


if __name__ == "__main__":
    main()
