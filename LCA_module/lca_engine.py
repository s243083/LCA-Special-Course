# =============================================================================
# lca_engine.py — LCA Calculation Engine
# =============================================================================
# Contains all calculation logic: loading the database, merging inventories,
# performing impact calculations, and aggregating results by life stage.
# =============================================================================

import pandas as pd
import yaml
import os
import io

# --- Configuration loading ---------------------------------------------------
# Read the YAML config once at import time and expose every setting as a
# module-level constant so all functions below share the same values without
# having to re-read the file each time they are called.
_cfg_path = os.path.join(os.path.dirname(__file__), "inputs", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

ECOINVENT_FILE       = os.path.join(os.path.dirname(__file__), _cfg["ECOINVENT_FILE"])
ECOINVENT_SHEET      = _cfg["ECOINVENT_SHEET"]
ECOINVENT_HEADER_ROW = _cfg["ECOINVENT_HEADER_ROW"]
IMPACT_COLUMNS       = _cfg["IMPACT_COLUMNS"]
UID_COLUMN           = _cfg["UID_COLUMN"]
REF_UNIT_COLUMN      = _cfg["REF_UNIT_COLUMN"]
REF_AMOUNT_COLUMN    = _cfg["REF_AMOUNT_COLUMN"]


def _find_impact_column_index(raw_df, impact_def):
    """
    Find the column index in the raw (multi-row header) DataFrame that matches
    a given impact definition by searching Method, Category, Indicator and unit.

    The ecoinvent LCIA sheet has 3 metadata rows before the actual header:
        row 0 → Method
        row 1 → Category
        row 2 → Indicator
        row 3 → column names (unit, UID, etc.)

    Args:
        raw_df (DataFrame): raw DataFrame read with header=None (all rows included)
        impact_def (dict): keys "method", "category", "indicator", "unit"

    Returns:
        int: column index, or None if not found
    """
    # Normalise all search terms to lowercase so the match is case-insensitive
    method    = str(impact_def["method"]).strip().lower()
    category  = str(impact_def["category"]).strip().lower()
    indicator = str(impact_def["indicator"]).strip().lower()
    unit      = str(impact_def["unit"]).strip().lower()

    # --- Column search -------------------------------------------------------
    # Walk every column in the raw sheet and read its four metadata rows.
    # A match requires all four fields to agree, ensuring we pick exactly the
    # right impact method even when the same indicator appears in multiple methods.
    all_matches = []
    for col_idx in range(raw_df.shape[1]):
        row0 = str(raw_df.iloc[0, col_idx]).strip().lower()
        row1 = str(raw_df.iloc[1, col_idx]).strip().lower()
        row2 = str(raw_df.iloc[2, col_idx]).strip().lower()
        row3 = str(raw_df.iloc[3, col_idx]).strip().lower()

        if method in row0 and category in row1 and indicator in row2 and unit in row3:
            all_matches.append((col_idx, row0, row1, row2, row3))

    if not all_matches:
        return None

    # --- Ambiguity check -----------------------------------------------------
    # If more than one column matched (e.g. the same indicator appears under two
    # methods), list all candidates so the user can tighten the config, then
    # fall back to the first hit to keep the run going.
    if len(all_matches) > 1:
        print(f"\n  [DEBUG] Multiple columns matched for '{impact_def.get('indicator')}' — listing all:")
        for idx, r0, r1, r2, r3 in all_matches:
            print(f"    col {idx:>4}: method='{r0}' | category='{r1}' | indicator='{r2}' | unit='{r3}'")
        print(f"  [DEBUG] Using first match: col {all_matches[0][0]}\n")

    chosen = all_matches[0]
    print(f"  [DEBUG] Column selected for '{impact_def.get('indicator')}': col {chosen[0]} "
          f"| method='{chosen[1]}' | category='{chosen[2]}' | indicator='{chosen[3]}' | unit='{chosen[4]}'")
    return chosen[0]


def load_ecoinvent_database():
    """
    Load the ecoinvent LCIA database from the Excel file defined in config.py.

    Reads the raw sheet first (no header) to locate impact columns by matching
    Method / Category / Indicator / Unit across the 3 metadata rows.
    Then reads the data from row 4 onwards, keyed by UID string.

    Returns:
        dict: {uid_string: {impact_label: value, ref_unit: str, ref_amount: float}}
        int: total number of processes loaded
    """
    # --- Read the raw Excel sheet --------------------------------------------
    # We read the sheet with header=None so that the three metadata rows at the
    # top (Method, Category, Indicator) are included as ordinary rows and can be
    # inspected by _find_impact_column_index.
    try:
        raw = pd.read_excel(
            ECOINVENT_FILE,
            sheet_name=ECOINVENT_SHEET,
            header=None,
            engine="openpyxl"
        )
    except FileNotFoundError:
        print(f"\n[ERROR] Ecoinvent database file not found: '{ECOINVENT_FILE}'")
        print("  Place the Excel file in the same folder as this script and try again.\n")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n[ERROR] Could not read ecoinvent database: {e}\n")
        raise SystemExit(1)

    # --- In-memory CSV conversion --------------------------------------------
    # Parsing a large Excel file with openpyxl is slow. To avoid doing it twice,
    # we dump the raw DataFrame into an in-memory CSV string and reuse that for
    # the second (header-aware) parse — no file is written to disk.
    _csv_buffer = io.StringIO()
    raw.to_csv(_csv_buffer, index=False, header=False)
    _csv_buffer.seek(0)

    # --- Locate impact columns -----------------------------------------------
    # The impact values are scattered across many columns in the sheet. For each
    # impact category defined in the config we search the metadata rows to find
    # exactly which column index holds its values.
    impact_col_indices = {}
    for label, impact_def in IMPACT_COLUMNS.items():
        col_idx = _find_impact_column_index(raw, impact_def)
        if col_idx is None:
            print(f"  [WARNING] Could not find column for impact '{label}' "
                  f"— check IMPACT_COLUMNS in config.py.")
        else:
            impact_col_indices[label] = col_idx

    # --- Parse structured data -----------------------------------------------
    # Now read the CSV buffer again, this time treating row ECOINVENT_HEADER_ROW
    # as the actual column header so we get named columns (UID, ref unit, etc.).
    df = pd.read_csv(_csv_buffer, header=ECOINVENT_HEADER_ROW, low_memory=False)
    _csv_buffer.close()

    if UID_COLUMN not in df.columns:
        print(f"\n[ERROR] UID column '{UID_COLUMN}' not found in sheet '{ECOINVENT_SHEET}'.")
        print(f"  Available columns: {list(df.columns[:10])}\n")
        raise SystemExit(1)

    # Drop blank/summary rows that have no UID and clean up whitespace
    df = df.dropna(subset=[UID_COLUMN])
    df[UID_COLUMN] = df[UID_COLUMN].astype(str).str.strip()

    # --- Build the UID lookup dictionary -------------------------------------
    # For every process row we build a dict entry that stores the reference unit
    # and the numerical impact value for each category. Impact values are read
    # directly from the raw DataFrame using the column indices resolved above,
    # because the CSV header-parse reorders columns and loses the raw positions.
    data_start_row = ECOINVENT_HEADER_ROW + 1

    db = {}
    for df_idx, row in df.iterrows():
        uid = str(row[UID_COLUMN]).strip()
        if not uid or uid == "nan":
            continue

        entry = {
            REF_UNIT_COLUMN:   str(row.get(REF_UNIT_COLUMN, "")).strip(),
            REF_AMOUNT_COLUMN: row.get(REF_AMOUNT_COLUMN, 1),
        }

        raw_row_idx = data_start_row + (df_idx - df.index[0])
        for label, col_idx in impact_col_indices.items():
            try:
                val = raw.iloc[raw_row_idx, col_idx]
                entry[label] = float(val) if pd.notna(val) else 0.0
            except (IndexError, ValueError, TypeError):
                entry[label] = 0.0

        db[uid] = entry

    return db, len(db)


def flatten_inventory(inventory_dict, scope):
    """
    Walk a nested inventory dictionary and return a flat list of components.

    Args:
        inventory_dict (dict): nested dict from inventory_codes.py or inventory_masses.py
        scope (str): "per_turbine", "full_farm", or "per_FU"

    Returns:
        list of dict: each dict has keys:
            "life_stage", "path", "component_name",
            "process_code" (for codes) or "quantity" + "unit" (for masses)
    """
    results = []

    # --- Recursive tree walk -------------------------------------------------
    # The inventory dicts are arbitrarily deep nested structures. This inner
    # function recurses until it hits a leaf node, which is either:
    #   • a dict that contains scope keys ("per_turbine" / "full_farm" / "per_FU")
    #     → a mass leaf: extract the (quantity, unit) tuple for the requested scope
    #   • a plain string (or None) value
    #     → a code leaf: the string is the ecoinvent UID for that component
    def _walk(d, life_stage, path):
        for key, value in d.items():
            current_path = path + [key]
            if isinstance(value, dict):
                if "per_turbine" in value or "full_farm" in value or "per_FU" in value:
                    qty_unit = value.get(scope)
                    if qty_unit is not None:
                        quantity, unit = qty_unit
                        results.append({
                            "life_stage":     life_stage,
                            "path":           " > ".join(current_path[1:]),
                            "component_name": key,
                            "quantity":       quantity,
                            "unit":           unit,
                        })
                else:
                    _walk(value, life_stage, current_path)
            else:
                if value is not None:
                    results.append({
                        "life_stage":    life_stage,
                        "path":          " > ".join(current_path[1:]),
                        "component_name": key,
                        "process_code":  value,
                    })

    for life_stage, stage_data in inventory_dict.items():
        _walk(stage_data, life_stage, [life_stage])

    return results


def merge_inventories(codes_dict, masses_dict, selected_stages, scope):
    """
    Combine inventory codes and masses into one unified component list,
    filtered to the selected life stages and scope.

    Args:
        codes_dict (dict): INVENTORY_CODES from inventory_codes.py
        masses_dict (dict): INVENTORY_MASSES from inventory_masses.py
        selected_stages (list of str): life stage names chosen by the user
        scope (str): "per_turbine", "full_farm", or "per_FU"

    Returns:
        list of dict: each component has keys:
            life_stage, component_name, path, process_code, quantity, unit
    """
    # --- Filter and flatten --------------------------------------------------
    # Only keep the life stages the user selected, then flatten both nested dicts
    # into plain lists so they are easy to join component by component.
    filtered_codes  = {k: v for k, v in codes_dict.items() if k in selected_stages}
    filtered_masses = {k: v for k, v in masses_dict.items() if k in selected_stages}

    flat_codes  = flatten_inventory(filtered_codes, scope)
    flat_masses = flatten_inventory(filtered_masses, scope)

    # --- Join codes with masses ----------------------------------------------
    # Build a lookup from (life_stage, component_name) → mass entry so we can
    # pair each code entry with its corresponding quantity and unit in O(1).
    # Components that have a UID but no matching mass entry are skipped with a warning.
    mass_lookup = {}
    for entry in flat_masses:
        key = (entry["life_stage"], entry["component_name"])
        mass_lookup[key] = entry

    merged = []
    for code_entry in flat_codes:
        key = (code_entry["life_stage"], code_entry["component_name"])
        mass_entry = mass_lookup.get(key)

        if mass_entry is None:
            print(f"  [WARNING] No mass found for: {code_entry['component_name']} "
                  f"({code_entry['life_stage']}) — skipping.")
            continue

        quantity = mass_entry.get("quantity")
        unit     = mass_entry.get("unit")

        if quantity is None:
            print(f"  [WARNING] Quantity is None for: {code_entry['component_name']} "
                  f"({code_entry['life_stage']}) — skipping.")
            continue

        merged.append({
            "life_stage":     code_entry["life_stage"],
            "component_name": code_entry["component_name"],
            "path":           code_entry["path"],
            "process_code":   code_entry["process_code"],
            "quantity":       quantity,
            "unit":           unit,
        })

    return merged


def resolve_emission_factors(codes_dict, selected_stages, ecoinvent_db):
    """
    Walk inventory codes and resolve every unique UID to its emission factors
    and reference unit/amount from the ecoinvent database.

    This is a scope-independent step — results can be reused across all scopes.
    Warns once per missing UID rather than once per (UID, scope) combination.

    Args:
        codes_dict (dict): INVENTORY_CODES from inventory_codes.py
        selected_stages (list of str): life stage names chosen by the user
        ecoinvent_db (dict): output of load_ecoinvent_database()

    Returns:
        dict: {uid: {impact_label: float, REF_UNIT_COLUMN: str, REF_AMOUNT_COLUMN: float}}
              Missing UIDs are stored as None so callers can skip silently.
    """
    impact_keys = list(IMPACT_COLUMNS.keys())
    resolved    = {}

    # --- UID resolution walk -------------------------------------------------
    # Recurse through the code tree for every selected life stage and collect
    # the emission factors for each unique UID encountered. Storing results in
    # `resolved` also acts as a visited set, so each UID is looked up only once
    # even if it appears in multiple components or stages.
    def _walk(d):
        for key, value in d.items():
            if isinstance(value, dict):
                _walk(value)
            elif value is not None and value not in resolved:
                if value not in ecoinvent_db:
                    print(f"  [WARNING] UID '{value}' not found in ecoinvent database — skipping.")
                    resolved[value] = None
                else:
                    process = ecoinvent_db[value]
                    entry = {
                        REF_UNIT_COLUMN: str(process.get(REF_UNIT_COLUMN, "")).strip(),
                    }
                    for label in impact_keys:
                        raw_val = process.get(label, 0.0)
                        try:
                            entry[label] = float(raw_val)
                        except (ValueError, TypeError):
                            entry[label] = 0.0
                    resolved[value] = entry

    for stage, stage_data in codes_dict.items():
        if stage in selected_stages:
            _walk(stage_data)

    return resolved


def calculate_impacts(merged_components, resolved_efs):
    """
    For each component, multiply its quantity by the pre-resolved emission factor.

    Impact = (quantity / reference_product_amount) * impact_value

    Skips silently if quantity is zero or UID was not found during resolution.
    Warns and skips on unit mismatch between inventory and ecoinvent reference unit.

    Args:
        merged_components (list of dict): output of merge_inventories()
        resolved_efs (dict): output of resolve_emission_factors()

    Returns:
        list of dict: each component dict extended with impact values per category
    """
    results     = []
    impact_keys = list(IMPACT_COLUMNS.keys())

    for comp in merged_components:
        process_code = comp["process_code"]
        quantity     = comp["quantity"]
        unit         = comp["unit"]

        # --- Pre-calculation checks ------------------------------------------
        # Skip the component before doing any maths if it is unusable:
        # zero/missing quantity means there is nothing to multiply, a missing UID
        # means the process is not in the database, and a unit mismatch would
        # silently produce a wrong number (e.g. multiplying kg factors by tonnes).
        if quantity == 0 or quantity is None:
            print(f"  [SKIPPED] '{comp['component_name']}' ({comp['life_stage']}): quantity is zero/None.")
            continue

        process = resolved_efs.get(process_code)
        if process is None:
            print(f"  [SKIPPED] '{comp['component_name']}' ({comp['life_stage']}): UID '{process_code}' not in database.")
            continue

        ref_unit = process[REF_UNIT_COLUMN]
        if unit.lower() != ref_unit.lower():
            print(f"  [SKIPPED] '{comp['component_name']}' ({comp['life_stage']}): "
                  f"unit mismatch — inventory='{unit}' vs ecoinvent='{ref_unit}'.")
            continue

        # --- Impact multiplication -------------------------------------------
        # For each impact category multiply the ecoinvent characterisation factor
        # (per reference unit) by the component's quantity to get its total
        # contribution. The results are merged back into the component record so
        # the output carries both the metadata and the computed values.
        component_impacts = {}
        for label in impact_keys:
            component_impacts[label] = process.get(label, 0.0) * quantity

        results.append({**comp, **component_impacts})

    return results


def aggregate_by_stage(impact_results):
    """
    Sum component-level impacts up to life stage totals and grand total.

    Args:
        impact_results (list of dict): output of calculate_impacts()

    Returns:
        dict: {
            "by_stage":     {stage_name: {impact_label: total_value}},
            "grand_total":  {impact_label: total_value},
            "by_component": impact_results
        }
    """
    impact_labels = list(IMPACT_COLUMNS.keys())
    by_stage      = {}
    grand_total   = {label: 0.0 for label in impact_labels}

    # --- Summation loop ------------------------------------------------------
    # Iterate once over all components and accumulate each impact into two
    # running totals: a per-stage subtotal (used for stage-level charts) and a
    # grand total across all stages (used for the overall LCA score).
    for comp in impact_results:
        stage = comp["life_stage"]
        if stage not in by_stage:
            by_stage[stage] = {label: 0.0 for label in impact_labels}
        for label in impact_labels:
            value = comp.get(label, 0.0)
            by_stage[stage][label] += value
            grand_total[label]     += value

    return {
        "by_stage":     by_stage,
        "grand_total":  grand_total,
        "by_component": impact_results,
    }


def validate_inventory_coverage(codes_dict, ecoinvent_db):
    """
    Check all UIDs in inventory_codes.py against the loaded ecoinvent database
    and report any that are missing.

    Args:
        codes_dict (dict): INVENTORY_CODES from inventory_codes.py
        ecoinvent_db (dict): output of load_ecoinvent_database()

    Returns:
        list of tuple: [(component_name, life_stage, uid), ...] for missing ones
    """
    missing = []

    # --- Full-tree coverage check --------------------------------------------
    # Walk the entire codes dict (all stages, not just selected ones) and check
    # every UID leaf against the loaded database. This gives a complete picture
    # of which processes are missing before any calculation is attempted.
    def _walk(d, life_stage):
        for key, value in d.items():
            if isinstance(value, dict):
                _walk(value, life_stage)
            elif value is not None:
                if value not in ecoinvent_db:
                    missing.append((key, life_stage, value))

    for life_stage, stage_data in codes_dict.items():
        _walk(stage_data, life_stage)

    return missing
