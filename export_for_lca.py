#!/usr/bin/env python3
"""
export_for_lca.py  --  Standalone data extractor for the LCA interface.

HOW TO RUN
----------
    cd <project_root>          # winpact/
    python export_for_lca.py

OUTPUT
------
    lca_interface/technical_output.py   (created automatically)

REQUIREMENTS
------------
    Python 3.8+
    PyYAML        (pip install pyyaml)

CONFIGURATION
-------------
Edit the path constants in the USER-ADJUSTABLE PATHS section below to target
a different farm or input folder.

SOURCE MAPPING (HKN example farm)
----------------------------------
  WindFarm.yaml        -> N_TURBINES, MW_PER_TURBINE, TOTAL_FARM_SIZE_MW, CAPACITY_FACTOR
  Config.yaml          -> LIFETIME_YEARS  (WF_OperationsEnd - WF_OperationsStart)
  CAPEX.yaml           -> DISTANCE_TO_SHORE_KM (inferred), all component masses
  iea_22s.py           -> rotor diameter (needed for DTU pitch-bearing formula)
  layout CSV           -> TURBINE_COORDINATES, array_cable_length_m (MST)

MASS EXTRACTION NOTES
---------------------
All masses are read dynamically from CAPEX.yaml each time this script runs so
that changes in CAPEX.yaml are automatically reflected in technical_output.py.

  Fixed material masses  : every `mass:` field under `material:` lists is summed
                           per component, including nested subsubcategories.

  Blade structural mass  : CAPEX.yaml material entries for "3_blades" carry only
                           cost-scaling factors (CF), not weights. The total mass
                           is documented in a YAML comment:
                               "# Mass basis: 3 x 82.301 t = 246.903 t (report Table 1)."
                           This script reads that value via regex from the raw file.

  Blade surface coating  : "blade_surface_coatings" (subsubcategory of "3_blades")
                           has Gelcoat_resin: 2.5 t -- added to blade total.

  Pitch bearing mass     : CAPEX.yaml marks "pitch_bearings" with
                               flag_DTU_scaling_model: true
                           so no fixed mass is stored there.  The DTU cost model
                           formula is replicated here:
                               mass_kg = 500 + 0.07 * rotor_diameter_m ** 2.5
                           The rotor diameter is read from iea_22s.py (diameter=284 m).
                           This mass is added to the hub_system total.

KNOWN GAPS
----------
  DISTANCE_TO_SHORE_KM : No explicit YAML parameter exists.  110 km is inferred
                         from the CAPEX.yaml inline comment on the offshore export
                         cable entry: "110 km x 2.5 M EUR/km".
                         Override via DISTANCE_TO_SHORE_OVERRIDE_KM below.
"""

import os
import re
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required -- install it with: pip install pyyaml")

# ---------------------------------------------------------------------------
# USER-ADJUSTABLE PATHS
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ROOT_DIR, "examples", "Inputs", "HKN")

CONFIG_YAML        = os.path.join(INPUTS_DIR, "Config.yaml")
WINDFARM_YAML      = os.path.join(INPUTS_DIR, "WindFarm.yaml")
CAPEX_YAML         = os.path.join(INPUTS_DIR, "CAPEX.yaml")
LAYOUT_CSV         = os.path.join(
    INPUTS_DIR, "Response_Framework", "HKN_layout_subset_with_scaled.csv"
)
IEA22_TURBINE_FILE = os.path.join(
    ROOT_DIR, "core", "ResponseFramework", "data", "turbine", "iea_22s.py"
)

OUTPUT_DIR  = os.path.join(ROOT_DIR, "lca_interface")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "technical_output.py")
DESKTOP_OUTPUT_FILE = r"C:\Users\andre\OneDrive\Desktop\technical_output.py"

# Set to a float (km) to override the inferred value; leave as None to use 110.0 km.
DISTANCE_TO_SHORE_OVERRIDE_KM = None



# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()



# ---------------------------------------------------------------------------
# MASS EXTRACTION
# ---------------------------------------------------------------------------

def _material_mass_sum(node):
    """
    Recursively sum every `mass` field in a CAPEX YAML node's `material` list
    and its `subsubcategories`.  Returns total in tonnes.
    Skips materials that have only a CF field and no mass value.
    """
    total = 0.0
    for mat in node.get("material") or []:
        m = mat.get("mass")
        if m is not None:
            total += float(m)
    for child in node.get("subsubcategories") or []:
        total += _material_mass_sum(child)
    return total


def _read_rotor_diameter_m(turbine_file):
    """
    Parse turbine rotor diameter from iea_22s.py.
    Looks for the keyword argument:  diameter=<number>
    inside WindTurbine.__init__().
    """
    text = _load_text(turbine_file)
    m = re.search(r"\bdiameter\s*=\s*(\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(
            f"Could not find 'diameter = <number>' in {turbine_file}"
        )
    return float(m.group(1))


def _blade_mass_from_capex_comment_t(capex_raw_text):
    """
    Extract total 3-blade structural mass (tonnes) from the YAML comment:
        '# Mass basis: 3 x 82.301 t = 246.903 t (report Table 1).'

    Uses regex on the raw file text so the value is always read from CAPEX.yaml,
    never hardcoded.  The multiplication sign may be ASCII 'x' or the Unicode
    times-sign (U+00D7).
    """
    pattern = r"Mass\s+basis\s*:.*?=\s*([\d.]+)\s*t"
    m = re.search(pattern, capex_raw_text, re.IGNORECASE)
    if not m:
        raise ValueError(
            "Could not find '# Mass basis: ... = X.X t' comment in CAPEX.yaml.\n"
            "Check that the 3_blades section still contains this comment."
        )
    return float(m.group(1))


def _dtu_pitch_bearing_mass_kg(rotor_diameter_m):
    """
    Replicate the DTU cost model formula for pitch bearing mass (kg).

    Source: core/DTU_Cost_Model/dtu_wind_cm_main.py, method medium_speed_drivetrain:
        pitch_bearings: 5.0e2 + 0.07 * rotor_diameter ** 2.5    [kg]

    Applied here because CAPEX.yaml pitch_bearings entry carries:
        scaling_models:
            flag_DTU_scaling_model: true
    and therefore stores no fixed mass value.
    """
    return 5.0e2 + 0.07 * (rotor_diameter_m ** 2.5)


def _find_node(capex, target_name):
    """BFS through CAPEX Phase/categories/subcategories/subsubcategories to find a node by name."""
    queue = []
    for phase in capex.get("Phase") or []:
        for cat in phase.get("categories") or []:
            queue.append(cat)
    while queue:
        node = queue.pop(0)
        if node.get("name") == target_name:
            return node
        for child in (node.get("subcategories") or []) + (node.get("subsubcategories") or []):
            queue.append(child)
    return {}


def _mat(node, material_name):
    """Return mass (tonnes) of a named material from a node's material list. 0 if not found."""
    for m in (node or {}).get("material") or []:
        if m.get("name") == material_name and m.get("mass") is not None:
            return float(m["mass"])
    return 0.0


def _extract_material_inventory(capex, capex_raw_text):
    """
    Extract per-material masses (tonnes) matching the LCA inventory format.

    Hub steel = hub shell (40 t) + pitch_system (78 t); pitch bearings excluded.
    Blade mass split into glass fibre / epoxy using BLADE_*_FRACTION constants.
    Cabling split uses CABLING_COPPER_T / CABLING_PLASTIC_T constants.
    """
    blade_t = _blade_mass_from_capex_comment_t(capex_raw_text)

    hub    = _find_node(capex, "Hub_system")
    ps     = _find_node(capex, "pitch_system")
    carb   = _find_node(capex, "CARB_upwind_bearing")
    srb    = _find_node(capex, "SRB_downwind_bearing")
    shaft  = _find_node(capex, "main_shaft")
    gen    = _find_node(capex, "Generator")
    brake  = _find_node(capex, "Brake")
    turret = _find_node(capex, "Turret")
    bed    = _find_node(capex, "Bedplate")
    yaw    = _find_node(capex, "Yaw_system")
    nac    = _find_node(capex, "Nacelle_cover_and_platforms")
    hvac_n = _find_node(capex, "HVAC_and_auxiliaries")
    conv   = _find_node(capex, "Converter")
    trans  = _find_node(capex, "Transformer")
    tower  = _find_node(capex, "Tower")
    mono   = _find_node(capex, "Monopile")
    tp     = _find_node(capex, "Transition_piece")

    return {
        "TURBINE_HUB_STEEL_T":             _mat(hub, "Steel") + _mat(ps, "Steel"),
        "TURBINE_HUB_GLASS_FIBER_T":       _mat(hub, "Uniax_GlassFiber"),
        "TURBINE_BLADES_T":                 blade_t,
        "TURBINE_MAIN_SHAFT_STEEL_T":      _mat(carb, "Steel") + _mat(srb, "Steel") + _mat(shaft, "Steel"),
        "TURBINE_GENERATOR_NDFEB_T":       _mat(gen, "NdFeB"),
        "TURBINE_GENERATOR_COPPER_T":      _mat(gen, "Copper"),
        "TURBINE_GENERATOR_ELEC_STEEL_T":  _mat(gen, "Electrical_steel"),
        "TURBINE_GENERATOR_STEEL_T":       _mat(gen, "Steel"),
        "TURBINE_BRAKE_STEEL_T":           _mat(brake, "Steel"),
        "TURBINE_TURRET_STEEL_T":          _mat(turret, "Steel"),
        "TURBINE_BEDPLATE_STEEL_T":        _mat(bed, "Steel"),
        "TURBINE_YAW_STEEL_T":             _mat(yaw, "Steel"),
        "TURBINE_NACELLE_STEEL_T":         _mat(nac, "Steel"),
        "TURBINE_NACELLE_COMPOSITE_T":     _mat(nac, "Composite_panel"),
        "TURBINE_HVAC_T":                  _mat(hvac_n, "HVAC_pack"),
        "TURBINE_CABLING_T":               _mat(hvac_n, "Cabling_internal"),
        "TURBINE_LUBE_HYDRAULICS_T":       _mat(hvac_n, "Lube_and_hydraulics"),
        "TURBINE_CONVERTER_T":             _mat(conv, "Power_electronics_modules"),
        "TURBINE_TRANSFORMER_T":           _mat(trans, "Transformer_unit"),
        "TURBINE_TOWER_STEEL_T":           _mat(tower, "Steel"),
        "SUBSTRUCTURE_MONOPILE_STEEL_T":   _mat(mono, "Steel"),
        "SUBSTRUCTURE_TRANSITION_STEEL_T": _mat(tp, "Steel"),
    }


# Maps CAPEX YAML component names to output dict keys.
# "3_blades" and "pitch_bearings" are handled separately inside _extract_masses().
_CAPEX_NAME_MAP = {
    "Hub_system":                  "hub_system",
    "Main_shaft_and_bearings":     "main_shaft_and_bearings",
    "Generator":                   "generator",
    "Brake":                       "brake",
    "Turret":                      "turret",
    "Bedplate":                    "bedplate",
    "Yaw_system":                  "yaw_system",
    "Nacelle_cover_and_platforms": "nacelle_cover_platforms",
    "HVAC_and_auxiliaries":        "hvac_auxiliaries",
    "Converter":                   "converter",
    "Transformer":                 "transformer",
    "Tower":                       "tower",
    "Monopile":                    "monopile",
    "Transition_piece":            "transition_piece",
}


def _extract_masses(capex, capex_raw_text, rotor_diameter_m):
    """
    Walk the CAPEX.yaml Phase/categories/subcategories/subsubcategories tree
    and return per-component masses in tonnes.

    Returns
    -------
    masses_t : dict {component_key: mass_tonnes}
    meta     : dict with intermediate values used to build source comments
    """
    masses_t = {key: 0.0 for key in _CAPEX_NAME_MAP.values()}
    masses_t["blades"] = 0.0

    # Blade structural mass: read from YAML comment (no mass fields in material list)
    blade_comment_t = _blade_mass_from_capex_comment_t(capex_raw_text)

    # Pitch bearing mass: DTU formula (returns kg; convert to tonnes for internal use)
    pitch_bearing_kg = _dtu_pitch_bearing_mass_kg(rotor_diameter_m)
    pitch_bearing_t  = pitch_bearing_kg / 1000.0

    # Diagnostic values used to build inline source comments in the output file
    _info = {
        "blade_comment_t":  blade_comment_t,
        "blade_coating_t":  0.0,    # populated during traversal
        "hub_fixed_t":      0.0,    # populated during traversal
        "pitch_bearing_kg": pitch_bearing_kg,
        "pitch_bearing_t":  pitch_bearing_t,
        "rotor_diameter_m": rotor_diameter_m,
    }

    def visit(node):
        name = node.get("name", "")

        if name == "3_blades":
            # Structural mass from YAML comment + surface-coating from subsubcategory.
            # _material_mass_sum on this node returns 0 for the main materials
            # (they have CF only, no mass field) and 2.5 t for blade_surface_coatings.
            coating_t = _material_mass_sum(node)
            _info["blade_coating_t"] = coating_t
            masses_t["blades"] += blade_comment_t + coating_t

        elif name == "pitch_bearings":
            # CAPEX.yaml has flag_DTU_scaling_model: true and no fixed mass.
            # Add the replicated DTU formula result to hub_system.
            masses_t["hub_system"] += pitch_bearing_t

        elif name in _CAPEX_NAME_MAP:
            fixed_t = _material_mass_sum(node)
            masses_t[_CAPEX_NAME_MAP[name]] += fixed_t
            if name == "Hub_system":
                _info["hub_fixed_t"] = fixed_t   # for source comment

        # Always recurse: mapped components may be nested anywhere in the tree.
        for child in node.get("subcategories") or []:
            visit(child)
        for child in node.get("subsubcategories") or []:
            visit(child)

    for phase in capex.get("Phase") or []:
        for cat in phase.get("categories") or []:
            visit(cat)

    return masses_t, _info


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _run_opex_simulation():
    """
    Run the OPEX simulation (analytic_ctmc, same config as the HKN example)
    and return mode-3 interventions and vessel hours in a single run.

    Returns
    -------
    mode3_interventions  : dict {component_name: float}
    vessel_hours_by_type : dict {vessel_type: expected_vessel_hours}  -- summary
    vessel_hours_by_mode : list of dicts  -- full breakdown by component/mode
    """
    sys.path.insert(0, ROOT_DIR)

    try:
        from core.Simulation import Simulation
        from core.File_Handling import load_yaml as _sim_load_yaml
    except ImportError as exc:
        raise ImportError(
            "Could not import winpact core modules. "
            "Make sure you are running from the project root."
        ) from exc

    config_dict = _sim_load_yaml(str(INPUTS_DIR), "Config.yaml")
    config_dict.setdefault("experiment_name",   "lca_export")
    config_dict.setdefault("result_directory",  os.path.join(ROOT_DIR, "results"))
    config_dict.setdefault("scenario_id",       "lca_001")
    config_dict.setdefault("scenario_label",    "lca_export")
    config_dict.setdefault("seed",              42)
    config_dict.setdefault("Project_Duration_h",    175_200)
    config_dict.setdefault("WF_OperationsStart_h",  0)
    config_dict.setdefault("WF_OperationsEnd_h",    175_200)

    sim_cfg = {
        "run_marketenv":   False,
        "run_metenv":      False,
        "run_capex":       False,
        "run_windfarm":    False,
        "run_opex":        True,
        "opex_dashboard":  False,
        "run_revenue":     False,
        "run_valuation":   False,
        "collect_results": False,
    }

    sim = Simulation.from_config(
        library_path=str(INPUTS_DIR),
        config=config_dict,
        simulation_config=sim_cfg,
    )
    sim.run()

    opex = getattr(sim.env, "opex", None)
    if opex is None:
        return {}, {}, []

    # Mode-3 interventions
    mcb = getattr(opex, "opex_mode_cost_breakdown_df", None)
    if mcb is not None and not mcb.empty:
        mode3 = mcb[mcb["mode_id"].astype(str) == "3"]
        mode3_interventions = dict(zip(mode3["component"], mode3["N_interventions"].astype(float)))
    else:
        mode3_interventions = {}

    # Vessel hours
    vessel_hours_by_type = {}
    vr_summary = getattr(opex, "vessel_records_summary", None)
    if vr_summary is not None and not vr_summary.empty:
        vessel_hours_by_type = dict(
            zip(vr_summary["vessel_type"], vr_summary["expected_vessel_hours"].astype(float))
        )

    return mode3_interventions, vessel_hours_by_type


def main():
    # ---- Load inputs -------------------------------------------------------
    config     = _load_yaml(CONFIG_YAML)
    wf         = _load_yaml(WINDFARM_YAML)
    capex      = _load_yaml(CAPEX_YAML)
    capex_raw  = _load_text(CAPEX_YAML)

    # ---- General parameters ------------------------------------------------
    n_turbines      = int(wf["WindFarm"]["n_turbines"])
    mw_per_turbine  = float(wf["WindFarm"]["turbine_rated_power"])
    total_farm_mw   = float(wf["WindFarm"]["fixed"]["rated_power"])
    capacity_factor = float(wf["WindFarm"]["fixed"]["capacity_factor"])

    # Operational lifetime = WF_OperationsEnd - WF_OperationsStart (years)
    ops_start      = int(config["WF_OperationsStart"]["value"])   # 3
    ops_end        = int(config["WF_OperationsEnd"]["value"])     # 23
    lifetime_years = ops_end - ops_start                          # 20

    # Distance to shore -- no explicit YAML field found anywhere.
    if DISTANCE_TO_SHORE_OVERRIDE_KM is not None:
        dist_km     = float(DISTANCE_TO_SHORE_OVERRIDE_KM)
        dist_source = "user override via DISTANCE_TO_SHORE_OVERRIDE_KM"
    else:
        dist_km     = 110.0
        dist_source = (
            "inferred from CAPEX.yaml -- Export_cable_supply_offshore_220kV"
            " comment: '110 km x 2.5 M EUR/km'"
        )

    # Net energy (kWh) = total_farm_MW * CF * hours * 1000 kWh/MWh
    lifetime_h     = lifetime_years * 365 * 24
    energy_net_kwh = total_farm_mw * capacity_factor * float(lifetime_h) * 1_000.0

    # ---- Component masses --------------------------------------------------
    rotor_d = _read_rotor_diameter_m(IEA22_TURBINE_FILE)
    masses_t, meta = _extract_masses(capex, capex_raw, rotor_d)
    masses_kg = {k: round(v * 1_000.0, 3) for k, v in masses_t.items()}
    inventory = _extract_material_inventory(capex, capex_raw)

    # ---- OPEX: N_interventions for mode_id == 3 + vessel hours ----------------
    print("Running OPEX simulation to extract mode-3 interventions and vessel hours ...")
    mode3_interventions, vessel_hours_by_type = _run_opex_simulation()
    print(f"  Mode-3 components found: {list(mode3_interventions.keys())}")
    print(f"  Vessel types found: {list(vessel_hours_by_type.keys())}")

    # ---- Build output file content -----------------------------------------
    def kg(key):
        """Format mass value with 1 decimal place."""
        return f"{masses_kg[key]:.1f}"

    # Per-component source comments
    hub_src = (
        f"# CAPEX.yaml: Hub_system fixed materials ({meta['hub_fixed_t']:.1f} t) "
        f"+ DTU pitch_bearings ({meta['pitch_bearing_t']:.3f} t, "
        f"formula: 500 + 0.07*D^2.5 kg, D={meta['rotor_diameter_m']:.0f} m from iea_22s.py)"
    )
    blade_src = (
        f"# CAPEX.yaml: 3_blades comment ({meta['blade_comment_t']} t) "
        f"+ blade_surface_coatings.Gelcoat_resin ({meta['blade_coating_t']} t)"
    )

    output_lines = [
        "# =============================================================================",
        "# technical_output.py -- Auto-generated by export_for_lca.py",
        "# Do not edit manually. Re-run export_for_lca.py to update.",
        "# =============================================================================",
        "",
        f"N_TURBINES              = {n_turbines}      # int   -- Number of turbines",
        f"MW_PER_TURBINE          = {mw_per_turbine}     # float -- Rated capacity per turbine (MW)",
        f"TOTAL_FARM_SIZE_MW      = {total_farm_mw}   # float -- Total installed capacity (MW)",
        f"CAPACITY_FACTOR         = {capacity_factor}   # float -- Capacity factor (e.g. 0.53)",
        (
            f"DISTANCE_TO_SHORE_KM    = {dist_km}   "
            f"# float -- Distance to shore (km); {dist_source}"
        ),
        (
            f"LIFETIME_YEARS          = {lifetime_years}      "
            f"# int   -- Project lifetime (years); "
            f"WF_OperationsEnd - WF_OperationsStart ({ops_end} - {ops_start})"
        ),
        f"ENERGY_NET_KWH          = {energy_net_kwh:.6e}  # float -- Net energy over lifetime (kWh)",
        "",
        "# All masses converted from tonnes (source) to kg (multiply by 1000)",
        "MASSES_KG = {",
        f'    "hub_system":               {kg("hub_system")},    {hub_src}',
        f'    "blades":                   {kg("blades")},  {blade_src}',
        (
            f'    "main_shaft_and_bearings":  {kg("main_shaft_and_bearings")},    '
            "# CAPEX.yaml: CARB upwind bearing (42.6 t) + SRB downwind bearing (12.4 t)"
            " + main shaft (4.1 t)"
        ),
        (
            f'    "generator":                {kg("generator")},  '
            "# CAPEX.yaml: NdFeB (25.2 t) + Cu (17.1 t) + electrical steel (105.6 t)"
            " + structural steel (357.09 t)"
        ),
        f'    "brake":                    {kg("brake")},    # CAPEX.yaml: Brake.Steel',
        f'    "turret":                   {kg("turret")},    # CAPEX.yaml: Turret.Steel',
        f'    "bedplate":                 {kg("bedplate")},    # CAPEX.yaml: Bedplate.Steel',
        (
            f'    "yaw_system":               {kg("yaw_system")},    '
            "# CAPEX.yaml: Yaw_system.Steel (includes friction plate + motors)"
        ),
        (
            f'    "nacelle_cover_platforms":  {kg("nacelle_cover_platforms")},    '
            "# CAPEX.yaml: Composite_panel (17.8 t) + Steel (41.4 t)"
        ),
        (
            f'    "hvac_auxiliaries":         {kg("hvac_auxiliaries")},    '
            "# CAPEX.yaml: HVAC_pack (12.8 t) + Cabling_internal (4.0 t)"
            " + Lube_and_hydraulics (2.5 t)"
        ),
        f'    "converter":                {kg("converter")},    # CAPEX.yaml: Converter.Power_electronics_modules',
        f'    "transformer":              {kg("transformer")},    # CAPEX.yaml: Transformer.Transformer_unit',
        (
            f'    "tower":                    {kg("tower")},  '
            "# CAPEX.yaml: Tower.Steel (subsubcategory of Turbine.Substructure)"
        ),
        f'    "monopile":                 {kg("monopile")},  # CAPEX.yaml: Monopile.Steel (BoP Substructure)',
        f'    "transition_piece":         {kg("transition_piece")},   # CAPEX.yaml: Transition_piece.Steel (BoP Substructure)',
        "}",
        "",
        "# N_interventions for failure mode_id == 3 per component",
        "# Source: OPEX analytic_ctmc simulation (OM_22MW.yaml / OPEX.yaml)",
        "N_INTERVENTIONS_MODE3 = {",
    ] + [
        f'    "{comp}":    {val:.6f},'
        for comp, val in mode3_interventions.items()
    ] + [
        "}",
        "",
        "# Expected vessel hours over the operational lifetime, summed by vessel type",
        "# Source: OPEX analytic_ctmc simulation -- vessel_records[summary_by_vessel]",
        "VESSEL_HOURS_BY_TYPE = {",
    ] + [
        f'    "{vtype}":    {hours:.2f},'
        for vtype, hours in vessel_hours_by_type.items()
    ] + [
        "}",
    ]

    # ---- Write output file -------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines) + "\n")

    # ---- Console summary ---------------------------------------------------
    print(f"Written: {OUTPUT_FILE}\n")
    print(f"  N_TURBINES           = {n_turbines}")
    print(f"  MW_PER_TURBINE       = {mw_per_turbine} MW")
    print(f"  TOTAL_FARM_SIZE_MW   = {total_farm_mw} MW")
    print(f"  CAPACITY_FACTOR      = {capacity_factor}")
    print(f"  DISTANCE_TO_SHORE_KM = {dist_km} km")
    print(f"  LIFETIME_YEARS       = {lifetime_years} yrs")
    print(f"  ENERGY_NET_KWH       = {energy_net_kwh:.4e} kWh")
    print()
    print(f"  Rotor diameter (iea_22s.py)  = {meta['rotor_diameter_m']:.0f} m")
    print(
        f"  Pitch bearing mass (DTU)     = {meta['pitch_bearing_kg']:.1f} kg"
        f"  ({meta['pitch_bearing_t']:.3f} t)"
    )
    print(
        f"  Blade mass breakdown:"
        f"  comment={meta['blade_comment_t']} t"
        f" + coating={meta['blade_coating_t']} t"
        f" = {masses_t['blades']:.3f} t total"
    )
    print()
    print("  Component masses (tonnes -> kg):")
    for k, v_t in masses_t.items():
        print(f"    {k:<32}  {v_t:>10.3f} t  ->  {masses_kg[k]:>12,.1f} kg")
    print()
    print("  N_interventions mode_id == 3:")
    for comp, val in mode3_interventions.items():
        print(f"    {comp:<32}  {val:.6f}")
    print()
    print("  Vessel hours by type (expected over lifetime):")
    for vtype, hours in vessel_hours_by_type.items():
        print(f"    {vtype:<32}  {hours:.2f} h")

    # ---- Write desktop output file -----------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # N_INTERVENTIONS keys use underscores (spaces replaced) to be valid identifiers
    interventions_clean = {
        k.replace(" ", "_"): v for k, v in mode3_interventions.items()
    }

    desktop_lines = [
        "# =============================================================================",
        "# technical_output.py -- Auto-generated by export_for_lca.py",
        "# Do not edit manually. Re-run export_for_lca.py to update.",
        f"# Generated on: {timestamp}",
        "# =============================================================================",
        "",
        "# --- General Information ---",
        f"N_TURBINES              = {n_turbines}",
        f"MW_PER_TURBINE          = {mw_per_turbine}",
        f"TOTAL_FARM_SIZE_MW      = {total_farm_mw}",
        f"CAPACITY_FACTOR         = {capacity_factor}",
        f"DISTANCE_TO_SHORE_KM    = {dist_km}",
        f"LIFETIME_YEARS          = {lifetime_years}",
        f"ENERGY_NET_KWH          = {energy_net_kwh:.4e}",
        "",
        "# --- Component Masses (tonnes) ---",
        f"PITCH_BEARING_MASS_KG            = {meta['pitch_bearing_kg']:.1f}",
        "",
        "# 1.1 Hub System (Rotor)",
        f"TURBINE_HUB_STEEL_T              = {inventory['TURBINE_HUB_STEEL_T']}",
        f"TURBINE_HUB_GLASS_FIBER_T        = {inventory['TURBINE_HUB_GLASS_FIBER_T']}",
        "",
        "# 1.2 Blades (Rotor)",
        f"TURBINE_BLADES_T                 = {inventory['TURBINE_BLADES_T']}  # split: Glass Fibre = TURBINE_BLADES_T * 0.40 | Epoxy = TURBINE_BLADES_T * 0.60",
        "",
        "# 1.3 Main Shaft and Bearings",
        f"TURBINE_MAIN_SHAFT_STEEL_T       = {inventory['TURBINE_MAIN_SHAFT_STEEL_T']}",
        "",
        "# 1.4 Generator",
        f"TURBINE_GENERATOR_NDFEB_T        = {inventory['TURBINE_GENERATOR_NDFEB_T']}",
        f"TURBINE_GENERATOR_COPPER_T       = {inventory['TURBINE_GENERATOR_COPPER_T']}",
        f"TURBINE_GENERATOR_ELEC_STEEL_T   = {inventory['TURBINE_GENERATOR_ELEC_STEEL_T']}",
        f"TURBINE_GENERATOR_STEEL_T        = {inventory['TURBINE_GENERATOR_STEEL_T']}",
        "",
        "# 1.5 Brake",
        f"TURBINE_BRAKE_STEEL_T            = {inventory['TURBINE_BRAKE_STEEL_T']}",
        "",
        "# 1.6 Turret (Nacelle)",
        f"TURBINE_TURRET_STEEL_T           = {inventory['TURBINE_TURRET_STEEL_T']}",
        "",
        "# 1.7 Bedplate (Nacelle)",
        f"TURBINE_BEDPLATE_STEEL_T         = {inventory['TURBINE_BEDPLATE_STEEL_T']}",
        "",
        "# 1.8 Yaw System (Nacelle)",
        f"TURBINE_YAW_STEEL_T              = {inventory['TURBINE_YAW_STEEL_T']}",
        "",
        "# 1.9 Nacelle Cover and Platforms",
        f"TURBINE_NACELLE_STEEL_T          = {inventory['TURBINE_NACELLE_STEEL_T']}",
        f"TURBINE_NACELLE_COMPOSITE_T      = {inventory['TURBINE_NACELLE_COMPOSITE_T']}",
        "",
        "# 1.10 HVAC and Auxiliaries",
        f"TURBINE_HVAC_T                   = {inventory['TURBINE_HVAC_T']}",
        f"TURBINE_CABLING_T                = {inventory['TURBINE_CABLING_T']}  # split: Copper = 2.6 t | Plastic = 1.4 t",
        f"TURBINE_LUBE_HYDRAULICS_T        = {inventory['TURBINE_LUBE_HYDRAULICS_T']}",
        "",
        "# 1.11 Converter",
        f"TURBINE_CONVERTER_T              = {inventory['TURBINE_CONVERTER_T']}",
        "",
        "# 1.12 Transformer",
        f"TURBINE_TRANSFORMER_T            = {inventory['TURBINE_TRANSFORMER_T']}",
        "",
        "# 1.13 Tower",
        f"TURBINE_TOWER_STEEL_T            = {inventory['TURBINE_TOWER_STEEL_T']}",
        "",
        "# 2.1 Monopile",
        f"SUBSTRUCTURE_MONOPILE_STEEL_T    = {inventory['SUBSTRUCTURE_MONOPILE_STEEL_T']}",
        "",
        "# 2.2 Transition Piece",
        f"SUBSTRUCTURE_TRANSITION_STEEL_T  = {inventory['SUBSTRUCTURE_TRANSITION_STEEL_T']}",
        "",
        "# --- O&M Interventions (average per year, mode_id == 3) ---",
        "N_INTERVENTIONS = {",
    ] + [
        f'    "{k}":{"" if len(k) >= 24 else " " * (24 - len(k))}  {v:.6f},'
        for k, v in interventions_clean.items()
    ] + [
        "}",
        "",
        "# --- Vessel Hours (expected over operational lifetime) ---",
        "VESSEL_HOURS_BY_TYPE = {",
    ] + [
        f'    "{vtype}":{"" if len(vtype) >= 24 else " " * (24 - len(vtype))}  {hours:.2f},'
        for vtype, hours in vessel_hours_by_type.items()
    ] + [
        "}",
    ]

    with open(DESKTOP_OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(desktop_lines) + "\n")

    print(f"\nAlso written to: {DESKTOP_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
