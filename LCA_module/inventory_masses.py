# =============================================================================
# inventory_masses.py — Wind Farm LCA Component Masses and Quantities
# =============================================================================
# Each leaf component maps to a dict with three keys:
#     "per_turbine" : (quantity, unit)
#     "full_farm"   : (quantity, unit)
#     "per_FU"      : (quantity, unit)  ← functional unit, used for calculations
#
# per_FU is used for Materials, Manufacturing and Transport stages.
# All other stages have (0, "kg") for per_FU for now.
# None means the value is not yet available.
#
# _MAT is defined first so that Manufacturing can reference Materials values
# directly instead of duplicating or recomputing them.
# =============================================================================

from technical_output import (
    N_INTERVENTIONS, N_TURBINES, ENERGY_NET_KWH,
    TURBINE_HUB_STEEL_T, TURBINE_HUB_GLASS_FIBER_T,
    TURBINE_BLADES_T,
    TURBINE_MAIN_SHAFT_STEEL_T,
    TURBINE_GENERATOR_NDFEB_T, TURBINE_GENERATOR_COPPER_T,
    TURBINE_GENERATOR_ELEC_STEEL_T, TURBINE_GENERATOR_STEEL_T,
    TURBINE_BRAKE_STEEL_T, TURBINE_TURRET_STEEL_T,
    TURBINE_BEDPLATE_STEEL_T, TURBINE_YAW_STEEL_T,
    TURBINE_NACELLE_STEEL_T, TURBINE_NACELLE_COMPOSITE_T,
    TURBINE_HVAC_T, TURBINE_CABLING_T, TURBINE_LUBE_HYDRAULICS_T,
    TURBINE_CONVERTER_T, TURBINE_TRANSFORMER_T, TURBINE_TOWER_STEEL_T,
    SUBSTRUCTURE_MONOPILE_STEEL_T, SUBSTRUCTURE_TRANSITION_STEEL_T,
)


import yaml
import os

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

EOL_FACTORS                    = _cfg["EOL_FACTORS"]
DISTANCE_TO_SHORE_KM           = _cfg["DISTANCE_TO_SHORE_KM"]
INSTALLATION_ACTIVITIES        = _cfg["INSTALLATION_ACTIVITIES"]
ENERGY_CONTENT_FUEL_MJ_PER_KG  = _cfg["ENERGY_CONTENT_FUEL_MJ_PER_KG"]
ARRAY_CABLE_240MM2_LENGTH_KM   = _cfg["ARRAY_CABLE_240MM2_LENGTH_KM"]
ARRAY_CABLE_630MM2_LENGTH_KM   = _cfg["ARRAY_CABLE_630MM2_LENGTH_KM"]
ARRAY_CABLE_800MM2_LENGTH_KM   = _cfg["ARRAY_CABLE_800MM2_LENGTH_KM"]
CABLE_FRACTION_COPPER          = _cfg["CABLE_FRACTION_COPPER"]
CABLE_FRACTION_POLYETHYLENE    = _cfg["CABLE_FRACTION_POLYETHYLENE"]
CABLE_FRACTION_LEAD            = _cfg["CABLE_FRACTION_LEAD"]
CABLE_FRACTION_STEEL           = _cfg["CABLE_FRACTION_STEEL"]
CABLE_FRACTION_GLASS           = _cfg["CABLE_FRACTION_GLASS"]
CABLE_KGM_240MM2               = _cfg["CABLE_KGM_240MM2"]
CABLE_KGM_630MM2               = _cfg["CABLE_KGM_630MM2"]
CABLE_KGM_800MM2               = _cfg["CABLE_KGM_800MM2"]
OSS_STEEL_UNALLOYED_T          = _cfg["OSS_STEEL_UNALLOYED_T"]
OSS_STEEL_HIGHLY_ALLOYED_T     = _cfg["OSS_STEEL_HIGHLY_ALLOYED_T"]
OSS_CAST_IRON_T                = _cfg["OSS_CAST_IRON_T"]
OSS_ALUMINIUM_T                = _cfg["OSS_ALUMINIUM_T"]
OSS_COPPER_T                   = _cfg["OSS_COPPER_T"]
OSS_ZINC_ALLOYS_T              = _cfg["OSS_ZINC_ALLOYS_T"]
OSS_POLYMER_T                  = _cfg["OSS_POLYMER_T"]
OSS_ORGANIC_MATERIALS_T        = _cfg["OSS_ORGANIC_MATERIALS_T"]
OSS_CERAMIC_GLASS_T            = _cfg["OSS_CERAMIC_GLASS_T"]
OSS_CONCRETE_T                 = _cfg["OSS_CONCRETE_T"]
OSS_SF6_GAS_T                  = _cfg["OSS_SF6_GAS_T"]
OSS_LUBRICANTS_T               = _cfg["OSS_LUBRICANTS_T"]
ONS_STEEL_UNALLOYED_T          = _cfg["ONS_STEEL_UNALLOYED_T"]
ONS_STEEL_HIGHLY_ALLOYED_T     = _cfg["ONS_STEEL_HIGHLY_ALLOYED_T"]
ONS_CAST_IRON_T                = _cfg["ONS_CAST_IRON_T"]
ONS_ALUMINIUM_T                = _cfg["ONS_ALUMINIUM_T"]
ONS_COPPER_T                   = _cfg["ONS_COPPER_T"]
ONS_ZINC_ALLOYS_T              = _cfg["ONS_ZINC_ALLOYS_T"]
ONS_POLYMER_T                  = _cfg["ONS_POLYMER_T"]
ONS_ORGANIC_MATERIALS_T        = _cfg["ONS_ORGANIC_MATERIALS_T"]
ONS_CERAMIC_GLASS_T            = _cfg["ONS_CERAMIC_GLASS_T"]
ONS_CONCRETE_T                 = _cfg["ONS_CONCRETE_T"]
ONS_SF6_GAS_T                  = _cfg["ONS_SF6_GAS_T"]
ONS_LUBRICANTS_T               = _cfg["ONS_LUBRICANTS_T"]
EXPORT_CABLE_MASSES_T_PER_KM   = _cfg["EXPORT_CABLE_MASSES_T_PER_KM"]
# Split factors for components reported as totals in technical_output.py
BLADES_GLASS_FIBRE_FRACTION    = 0.40
BLADES_EPOXY_FRACTION          = 0.60
CABLING_COPPER_FRACTION        = 2.6 / 4.0
CABLING_PLASTIC_FRACTION       = 1.4 / 4.0

TURBINE_BLADES_GLASS_FIBRE_T   = TURBINE_BLADES_T * BLADES_GLASS_FIBRE_FRACTION
TURBINE_BLADES_EPOXY_T         = TURBINE_BLADES_T * BLADES_EPOXY_FRACTION
TURBINE_CABLING_COPPER_T       = TURBINE_CABLING_T * CABLING_COPPER_FRACTION
TURBINE_CABLING_PLASTIC_T      = TURBINE_CABLING_T * CABLING_PLASTIC_FRACTION

# FU factor
FU_FACTOR = 1 / ENERGY_NET_KWH

# Unit conversion
TON_TO_KG = 1000

# Total installation/decommissioning energy (MJ): sum of t/day × days × kg/t × MJ/kg across all activities
INSTALLATION_ENERGY_TOTAL_MJ = sum(
    t_per_day * days * TON_TO_KG * ENERGY_CONTENT_FUEL_MJ_PER_KG
    for t_per_day, days in INSTALLATION_ACTIVITIES.values()
)

ZERO_INTERVENTIONS = 0  # No intervention data available for this component

# =============================================================================
# MATERIALS — defined separately so Manufacturing can reference these values
# =============================================================================
_MAT = {
    "1. Wind Turbine": {
        "1.1 Hub System (Rotor)": {
            "1.1.1 Hub System Steel": {
                "per_turbine": (TURBINE_HUB_STEEL_T * TON_TO_KG,               "kg"),
                "full_farm":   (TURBINE_HUB_STEEL_T * TON_TO_KG * N_TURBINES,          "kg"),
                "per_FU":      (TURBINE_HUB_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,    "kg"),
            },
            "1.1.2 Hub System Uniax_Glass Fiber": {
                "per_turbine": (TURBINE_HUB_GLASS_FIBER_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_HUB_GLASS_FIBER_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_HUB_GLASS_FIBER_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.2 Blades (Rotor)": {
            "1.2.1 Blades Glass Fibre (60%)": {
                "per_turbine": (TURBINE_BLADES_GLASS_FIBRE_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_BLADES_GLASS_FIBRE_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_BLADES_GLASS_FIBRE_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.2.2 Blades Epoxy (40%)": {
                "per_turbine": (TURBINE_BLADES_EPOXY_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_BLADES_EPOXY_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_BLADES_EPOXY_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.3 Main Shaft and Bearings": {
            "1.3.1 Main Shaft and Bearings Steel": {
                "per_turbine": (TURBINE_MAIN_SHAFT_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_MAIN_SHAFT_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_MAIN_SHAFT_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.4 Generator": {
            "1.4.1 Generator NdFeB": {
                "per_turbine": (TURBINE_GENERATOR_NDFEB_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_GENERATOR_NDFEB_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_GENERATOR_NDFEB_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.4.2 Generator Copper": {
                "per_turbine": (TURBINE_GENERATOR_COPPER_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_GENERATOR_COPPER_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_GENERATOR_COPPER_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.4.3 Generator Electrical Steel": {
                "per_turbine": (TURBINE_GENERATOR_ELEC_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_GENERATOR_ELEC_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_GENERATOR_ELEC_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.4.4 Generator Steel": {
                "per_turbine": (TURBINE_GENERATOR_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_GENERATOR_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_GENERATOR_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.5 Brake": {
            "1.5.1 Brake Steel": {
                "per_turbine": (TURBINE_BRAKE_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_BRAKE_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_BRAKE_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.6 Turret (Nacelle)": {
            "1.6.1 Turret Steel": {
                "per_turbine": (TURBINE_TURRET_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_TURRET_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_TURRET_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.7 Bedplate (Nacelle)": {
            "1.7.1 Bedplate Steel": {
                "per_turbine": (TURBINE_BEDPLATE_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_BEDPLATE_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_BEDPLATE_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.8 Yaw System (Nacelle)": {
            "1.8.1 Yaw System Steel": {
                "per_turbine": (TURBINE_YAW_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_YAW_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_YAW_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.9 Nacelle_cover_and_platforms (Nacelle)": {
            "1.9.1 Nacelle Steel": {
                "per_turbine": (TURBINE_NACELLE_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_NACELLE_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_NACELLE_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.9.2 Nacelle Composite": {
                "per_turbine": (TURBINE_NACELLE_COMPOSITE_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_NACELLE_COMPOSITE_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_NACELLE_COMPOSITE_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.10 HVAC_and_auxiliarie (Nacelle)": {
            "1.10.1 HVAC_pack": {
                "per_turbine": (TURBINE_HVAC_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_HVAC_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_HVAC_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.10.2.1 Cabling_internal Copper": {
                "per_turbine": (TURBINE_CABLING_COPPER_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_CABLING_COPPER_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_CABLING_COPPER_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.10.2.2 Cabling_internal Plastic": {
                "per_turbine": (TURBINE_CABLING_PLASTIC_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_CABLING_PLASTIC_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_CABLING_PLASTIC_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
            "1.10.3 Lube_and_hydraulics": {
                "per_turbine": (TURBINE_LUBE_HYDRAULICS_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_LUBE_HYDRAULICS_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_LUBE_HYDRAULICS_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.11 Converter (Electrical components)": {
            "1.11.1 Power Electronics Converter": {
                "per_turbine": (TURBINE_CONVERTER_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_CONVERTER_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_CONVERTER_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.12 Transformer (Electrical components)": {
            "1.12.1 Transformer": {
                "per_turbine": (TURBINE_TRANSFORMER_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_TRANSFORMER_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_TRANSFORMER_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "1.13 Tower": {
            "1.13.1 Tower Steel": {
                "per_turbine": (TURBINE_TOWER_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (TURBINE_TOWER_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (TURBINE_TOWER_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
    },
    "2. Substructure": {
        "2.1 Monopile": {
            "2.1.1 Monopile Steel": {
                "per_turbine": (SUBSTRUCTURE_MONOPILE_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (SUBSTRUCTURE_MONOPILE_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (SUBSTRUCTURE_MONOPILE_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
        "2.2 Transitioning piece": {
            "2.2.1 Transitioning piece Steel": {
                "per_turbine": (SUBSTRUCTURE_TRANSITION_STEEL_T * TON_TO_KG,             "kg"),
                "full_farm":   (SUBSTRUCTURE_TRANSITION_STEEL_T * TON_TO_KG * N_TURBINES,        "kg"),
                "per_FU":      (SUBSTRUCTURE_TRANSITION_STEEL_T * TON_TO_KG * N_TURBINES * FU_FACTOR,  "kg"),
            },
        },
    },
    "3. Electrical Infrastructure": {
        "3.1 Array Cables 240mm2 (66kV)": {
            "3.1.1 Copper": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_COPPER,             "kg"),
                "per_FU":      (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_COPPER * FU_FACTOR,       "kg"),
            },
            "3.1.2 Polyethylene": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_POLYETHYLENE,       "kg"),
                "per_FU":      (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_POLYETHYLENE * FU_FACTOR, "kg"),
            },
            "3.1.3 Lead": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_LEAD,               "kg"),
                "per_FU":      (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_LEAD * FU_FACTOR,         "kg"),
            },
            "3.1.4 Steel": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_STEEL,              "kg"),
                "per_FU":      (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_STEEL * FU_FACTOR,        "kg"),
            },
            "3.1.5 Glass": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_GLASS,              "kg"),
                "per_FU":      (ARRAY_CABLE_240MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_240MM2 * CABLE_FRACTION_GLASS * FU_FACTOR,        "kg"),
            },
        },
        "3.2 Array Cables 630mm2 (66kV)": {
            "3.2.1 Copper": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_COPPER,             "kg"),
                "per_FU":      (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_COPPER * FU_FACTOR,       "kg"),
            },
            "3.2.2 Polyethylene": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_POLYETHYLENE,       "kg"),
                "per_FU":      (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_POLYETHYLENE * FU_FACTOR, "kg"),
            },
            "3.2.3 Lead": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_LEAD,               "kg"),
                "per_FU":      (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_LEAD * FU_FACTOR,         "kg"),
            },
            "3.2.4 Steel": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_STEEL,              "kg"),
                "per_FU":      (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_STEEL * FU_FACTOR,        "kg"),
            },
            "3.2.5 Glass": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_GLASS,              "kg"),
                "per_FU":      (ARRAY_CABLE_630MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_630MM2 * CABLE_FRACTION_GLASS * FU_FACTOR,        "kg"),
            },
        },
        "3.3 Array Cables 800mm2 (66kV)": {
            "3.3.1 Copper": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_COPPER,             "kg"),
                "per_FU":      (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_COPPER * FU_FACTOR,       "kg"),
            },
            "3.3.2 Polyethylene": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_POLYETHYLENE,       "kg"),
                "per_FU":      (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_POLYETHYLENE * FU_FACTOR, "kg"),
            },
            "3.3.3 Lead": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_LEAD,               "kg"),
                "per_FU":      (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_LEAD * FU_FACTOR,         "kg"),
            },
            "3.3.4 Steel": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_STEEL,              "kg"),
                "per_FU":      (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_STEEL * FU_FACTOR,        "kg"),
            },
            "3.3.5 Glass": {
                "per_turbine": (None,                                                                                        "kg"),
                "full_farm":   (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_GLASS,              "kg"),
                "per_FU":      (ARRAY_CABLE_800MM2_LENGTH_KM * TON_TO_KG * CABLE_KGM_800MM2 * CABLE_FRACTION_GLASS * FU_FACTOR,        "kg"),
            },
        },
        "3.4 Export Cables": {
            "3.4.1 Copper": {
                "per_turbine": (None, "kg"),
                "full_farm":   (EXPORT_CABLE_MASSES_T_PER_KM["Copper"]  * DISTANCE_TO_SHORE_KM * TON_TO_KG, "kg"),
                "per_FU":      (EXPORT_CABLE_MASSES_T_PER_KM["Copper"]  * DISTANCE_TO_SHORE_KM * TON_TO_KG * FU_FACTOR, "kg"),
            },
            "3.4.2 Lead": {
                "per_turbine": (None, "kg"),
                "full_farm":   (EXPORT_CABLE_MASSES_T_PER_KM["Lead"]    * DISTANCE_TO_SHORE_KM * TON_TO_KG, "kg"),
                "per_FU":      (EXPORT_CABLE_MASSES_T_PER_KM["Lead"]    * DISTANCE_TO_SHORE_KM * TON_TO_KG * FU_FACTOR, "kg"),
            },
            "3.4.3 Steel": {
                "per_turbine": (None, "kg"),
                "full_farm":   (EXPORT_CABLE_MASSES_T_PER_KM["Steel"]   * DISTANCE_TO_SHORE_KM * TON_TO_KG, "kg"),
                "per_FU":      (EXPORT_CABLE_MASSES_T_PER_KM["Steel"]   * DISTANCE_TO_SHORE_KM * TON_TO_KG * FU_FACTOR, "kg"),
            },
            "3.4.4 Polyethylene": {
                "per_turbine": (None, "kg"),
                "full_farm":   (EXPORT_CABLE_MASSES_T_PER_KM["Polyethylene"] * DISTANCE_TO_SHORE_KM * TON_TO_KG, "kg"),
                "per_FU":      (EXPORT_CABLE_MASSES_T_PER_KM["Polyethylene"] * DISTANCE_TO_SHORE_KM * TON_TO_KG * FU_FACTOR, "kg"),
            },
        },
    },
    "4. Offshore Substation": {
        "4.1 Steel": {
            "per_turbine": (None,                                                                    "kg"),
            "full_farm":   ((OSS_STEEL_UNALLOYED_T + OSS_STEEL_HIGHLY_ALLOYED_T) * TON_TO_KG,            "kg"),
            "per_FU":      ((OSS_STEEL_UNALLOYED_T + OSS_STEEL_HIGHLY_ALLOYED_T) * TON_TO_KG * FU_FACTOR,      "kg"),
        },
        "4.2 Copper": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_COPPER_T * TON_TO_KG,            "kg"),
            "per_FU":      (OSS_COPPER_T * TON_TO_KG * FU_FACTOR,      "kg"),
        },
        "4.3 Aluminium": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_ALUMINIUM_T * TON_TO_KG,         "kg"),
            "per_FU":      (OSS_ALUMINIUM_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "4.4 Polyethylene": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_POLYMER_T * TON_TO_KG,           "kg"),
            "per_FU":      (OSS_POLYMER_T * TON_TO_KG * FU_FACTOR,     "kg"),
        },
        "4.5 Lubricating oil": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_LUBRICANTS_T * TON_TO_KG,        "kg"),
            "per_FU":      (OSS_LUBRICANTS_T * TON_TO_KG * FU_FACTOR,  "kg"),
        },
        "4.6 Cast iron": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_CAST_IRON_T * TON_TO_KG,         "kg"),
            "per_FU":      (OSS_CAST_IRON_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "4.7 Modified organic natural materials": {
            "4.7.1 Kraft paper": {
                "per_turbine": (None,                                                  "kg"),
                "full_farm":   (OSS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.91,               "kg"),
                "per_FU":      (OSS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.91 * FU_FACTOR,         "kg"),
            },
            "4.7.2 Vegetable oil methyl ester": {
                "per_turbine": (None,                                                  "kg"),
                "full_farm":   (OSS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.09,               "kg"),
                "per_FU":      (OSS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.09 * FU_FACTOR,         "kg"),
            },
        },
        "4.8 Ceramic / glass": {
            "per_turbine": (None,                                "kg"),
            "full_farm":   (OSS_CERAMIC_GLASS_T * TON_TO_KG,         "kg"),
            "per_FU":      (OSS_CERAMIC_GLASS_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "4.9 SF6 Gas": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_SF6_GAS_T * TON_TO_KG,           "kg"),
            "per_FU":      (OSS_SF6_GAS_T * TON_TO_KG * FU_FACTOR,     "kg"),
        },
        "4.10 Zinc alloys": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (OSS_ZINC_ALLOYS_T * TON_TO_KG,       "kg"),
            "per_FU":      (OSS_ZINC_ALLOYS_T * TON_TO_KG * FU_FACTOR, "kg"),
        },
    },
    "5. Onshore Substation": {
        "5.1 Steel": {
            "per_turbine": (None,                                                                    "kg"),
            "full_farm":   ((ONS_STEEL_UNALLOYED_T + ONS_STEEL_HIGHLY_ALLOYED_T) * TON_TO_KG,            "kg"),
            "per_FU":      ((ONS_STEEL_UNALLOYED_T + ONS_STEEL_HIGHLY_ALLOYED_T) * TON_TO_KG * FU_FACTOR,      "kg"),
        },
        "5.2 Copper": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_COPPER_T * TON_TO_KG,            "kg"),
            "per_FU":      (ONS_COPPER_T * TON_TO_KG * FU_FACTOR,      "kg"),
        },
        "5.3 Aluminium": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_ALUMINIUM_T * TON_TO_KG,         "kg"),
            "per_FU":      (ONS_ALUMINIUM_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "5.4 Polyethylene": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_POLYMER_T * TON_TO_KG,           "kg"),
            "per_FU":      (ONS_POLYMER_T * TON_TO_KG * FU_FACTOR,     "kg"),
        },
        "5.5 Lubricating oil": {
            "per_turbine": (None,                                        "kg"),
            "full_farm":   (ONS_LUBRICANTS_T * TON_TO_KG,               "kg"),
            "per_FU":      (ONS_LUBRICANTS_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "5.6 Cast iron": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_CAST_IRON_T * TON_TO_KG,         "kg"),
            "per_FU":      (ONS_CAST_IRON_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "5.7 Modified organic natural materials": {
            "5.7.1 Kraft paper": {
                "per_turbine": (None,                                                  "kg"),
                "full_farm":   (ONS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.91,               "kg"),
                "per_FU":      (ONS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.91 * FU_FACTOR,         "kg"),
            },
            "5.7.2 Vegetable oil methyl ester": {
                "per_turbine": (None,                                                  "kg"),
                "full_farm":   (ONS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.09,               "kg"),
                "per_FU":      (ONS_ORGANIC_MATERIALS_T * TON_TO_KG * 0.09 * FU_FACTOR,         "kg"),
            },
        },
        "5.8 Ceramic / glass": {
            "per_turbine": (None,                                "kg"),
            "full_farm":   (ONS_CERAMIC_GLASS_T * TON_TO_KG,         "kg"),
            "per_FU":      (ONS_CERAMIC_GLASS_T * TON_TO_KG * FU_FACTOR,   "kg"),
        },
        "5.9 SF6 Gas": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_SF6_GAS_T * TON_TO_KG,           "kg"),
            "per_FU":      (ONS_SF6_GAS_T * TON_TO_KG * FU_FACTOR,     "kg"),
        },
        "5.10 Concrete": {
            "per_turbine": (None,                            "kg"),
            "full_farm":   (ONS_CONCRETE_T * TON_TO_KG,          "kg"),
            "per_FU":      (ONS_CONCRETE_T * TON_TO_KG * FU_FACTOR,    "kg"),
        },
    },
}

# Shorthand accessors to keep Manufacturing references concise
_T   = _MAT["1. Wind Turbine"]
_SUB = _MAT["2. Substructure"]
_EI  = _MAT["3. Electrical Infrastructure"]
_OSS = _MAT["4. Offshore Substation"]
_ONS = _MAT["5. Onshore Substation"]

def _ff(d):
    """Return the full_farm numeric value from a leaf materials entry."""
    return d["full_farm"][0]

def _pt(d):
    """Return the per_turbine numeric value from a leaf materials entry."""
    return d["per_turbine"][0]

# Flat lookup: leaf key name → leaf mass dict, built from turbine and substructure materials
_MAT_FLAT = {}
def _build_mat_flat(d):
    for k, v in d.items():
        if isinstance(v, dict) and "per_turbine" in v:
            _MAT_FLAT[k] = v
        elif isinstance(v, dict):
            _build_mat_flat(v)
_build_mat_flat(_T)
_build_mat_flat(_SUB)
_build_mat_flat(_OSS)
_build_mat_flat(_ONS)

def _eol(material_key, factor_id, scenario):
    """Return the per-turbine EOL mass: per_turbine_mass × recovered × scenario_fraction."""
    pt_mass = _pt(_MAT_FLAT[material_key])
    return pt_mass * EOL_FACTORS[factor_id]["recovered"] * EOL_FACTORS[factor_id][scenario]

def _eol_ff(material_key, factor_id, scenario):
    """Return the full-farm EOL mass for whole-farm materials: full_farm_mass × recovered × scenario_fraction."""
    ff_mass = _ff(_MAT_FLAT[material_key])
    return ff_mass * EOL_FACTORS[factor_id]["recovered"] * EOL_FACTORS[factor_id][scenario]


# Transport mass sums (kg) — used to compute t*km dynamically from DISTANCE_TO_SHORE_KM
_TRANSPORT_WT_KG = (
    _ff(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"])                           +
    _ff(_T["1.1 Hub System (Rotor)"]["1.1.2 Hub System Uniax_Glass Fiber"])               +
    _ff(_T["1.2 Blades (Rotor)"]["1.2.1 Blades Glass Fibre (60%)"])                       +
    _ff(_T["1.2 Blades (Rotor)"]["1.2.2 Blades Epoxy (40%)"])                             +
    _ff(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"])         +
    _ff(_T["1.4 Generator"]["1.4.1 Generator NdFeB"])                                     +
    _ff(_T["1.4 Generator"]["1.4.2 Generator Copper"])                                    +
    _ff(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"])                          +
    _ff(_T["1.4 Generator"]["1.4.4 Generator Steel"])                                     +
    _ff(_T["1.5 Brake"]["1.5.1 Brake Steel"])                                             +
    _ff(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"])                                 +
    _ff(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"])                             +
    _ff(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"])                        +
    _ff(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"])           +
    _ff(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.2 Nacelle Composite"])       +
    _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.1 HVAC_pack"])                    +
    _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"])    +
    _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"])   +
    _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.3 Lube_and_hydraulics"])          +
    _ff(_T["1.11 Converter (Electrical components)"]["1.11.1 Power Electronics Converter"]) +
    _ff(_T["1.12 Transformer (Electrical components)"]["1.12.1 Transformer"])             +
    _ff(_T["1.13 Tower"]["1.13.1 Tower Steel"])
)

_TRANSPORT_SUB_KG = (
    _ff(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"])                                     +
    _ff(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"])
)

_TRANSPORT_ARRAY_KG = (
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.1 Copper"])                            +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.2 Polyethylene"])                      +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.3 Lead"])                              +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.4 Steel"])                             +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.5 Glass"])                             +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.1 Copper"])                            +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.2 Polyethylene"])                      +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.3 Lead"])                              +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.4 Steel"])                             +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.5 Glass"])                             +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.1 Copper"])                            +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.2 Polyethylene"])                      +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.3 Lead"])                              +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.4 Steel"])                             +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.5 Glass"])
)

_TRANSPORT_EXPORT_KG = (
    _ff(_EI["3.4 Export Cables"]["3.4.1 Copper"])                                         +
    _ff(_EI["3.4 Export Cables"]["3.4.2 Lead"])                                           +
    _ff(_EI["3.4 Export Cables"]["3.4.3 Steel"])                                          +
    _ff(_EI["3.4 Export Cables"]["3.4.4 Polyethylene"])
)

_TRANSPORT_OSS_KG = (
    _ff(_OSS["4.1 Steel"])                                                                +
    _ff(_OSS["4.2 Copper"])                                                               +
    _ff(_OSS["4.3 Aluminium"])                                                            +
    _ff(_OSS["4.4 Polyethylene"])                                                         +
    _ff(_OSS["4.5 Lubricating oil"])                                                      +
    _ff(_OSS["4.6 Cast iron"])                                                            +
    _ff(_OSS["4.7 Modified organic natural materials"]["4.7.1 Kraft paper"])              +
    _ff(_OSS["4.7 Modified organic natural materials"]["4.7.2 Vegetable oil methyl ester"]) +
    _ff(_OSS["4.8 Ceramic / glass"])                                                      +
    _ff(_OSS["4.9 SF6 Gas"])                                                              +
    _ff(_OSS["4.10 Zinc alloys"])
)

_TRANSPORT_WT_PT_KG = (
    _pt(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"])                           +
    _pt(_T["1.1 Hub System (Rotor)"]["1.1.2 Hub System Uniax_Glass Fiber"])               +
    _pt(_T["1.2 Blades (Rotor)"]["1.2.1 Blades Glass Fibre (60%)"])                       +
    _pt(_T["1.2 Blades (Rotor)"]["1.2.2 Blades Epoxy (40%)"])                             +
    _pt(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"])         +
    _pt(_T["1.4 Generator"]["1.4.1 Generator NdFeB"])                                     +
    _pt(_T["1.4 Generator"]["1.4.2 Generator Copper"])                                    +
    _pt(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"])                          +
    _pt(_T["1.4 Generator"]["1.4.4 Generator Steel"])                                     +
    _pt(_T["1.5 Brake"]["1.5.1 Brake Steel"])                                             +
    _pt(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"])                                 +
    _pt(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"])                             +
    _pt(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"])                        +
    _pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"])           +
    _pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.2 Nacelle Composite"])       +
    _pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.1 HVAC_pack"])                    +
    _pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"])    +
    _pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"])   +
    _pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.3 Lube_and_hydraulics"])          +
    _pt(_T["1.11 Converter (Electrical components)"]["1.11.1 Power Electronics Converter"]) +
    _pt(_T["1.12 Transformer (Electrical components)"]["1.12.1 Transformer"])             +
    _pt(_T["1.13 Tower"]["1.13.1 Tower Steel"])
)

_TRANSPORT_SUB_PT_KG = (
    _pt(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"])                                     +
    _pt(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"])
)

# Cable full-farm mass sums (kg) — used for EOL calculations
_EOL_ARRAY_240_KG = (
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.1 Copper"])                            +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.2 Polyethylene"])                      +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.3 Lead"])                              +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.4 Steel"])                             +
    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.5 Glass"])
)

_EOL_ARRAY_630_KG = (
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.1 Copper"])                            +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.2 Polyethylene"])                      +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.3 Lead"])                              +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.4 Steel"])                             +
    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.5 Glass"])
)

_EOL_ARRAY_800_KG = (
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.1 Copper"])                            +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.2 Polyethylene"])                      +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.3 Lead"])                              +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.4 Steel"])                             +
    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.5 Glass"])
)

# Register cable totals in _MAT_FLAT so _eol_ff() can look them up like any other material
_MAT_FLAT["3.1 Array Cables 240mm2 (66kV)"] = {"per_turbine": (None, "kg"), "full_farm": (_EOL_ARRAY_240_KG, "kg"), "per_FU": (_EOL_ARRAY_240_KG * FU_FACTOR, "kg")}
_MAT_FLAT["3.2 Array Cables 630mm2 (66kV)"] = {"per_turbine": (None, "kg"), "full_farm": (_EOL_ARRAY_630_KG, "kg"), "per_FU": (_EOL_ARRAY_630_KG * FU_FACTOR, "kg")}
_MAT_FLAT["3.3 Array Cables 800mm2 (66kV)"] = {"per_turbine": (None, "kg"), "full_farm": (_EOL_ARRAY_800_KG, "kg"), "per_FU": (_EOL_ARRAY_800_KG * FU_FACTOR, "kg")}
_MAT_FLAT["3.4 Export Cables"]               = {"per_turbine": (None, "kg"), "full_farm": (_TRANSPORT_EXPORT_KG, "kg"), "per_FU": (_TRANSPORT_EXPORT_KG * FU_FACTOR, "kg")}


INVENTORY_MASSES = {

    "Materials": _MAT,

    # =========================================================================
    # MANUFACTURING
    # =========================================================================
    "Manufacturing": {
        "1. Steel": {
            "per_turbine": (None, "kg"),
            "full_farm": (
                _ff(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"])          +
                _ff(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"]) +
                _ff(_T["1.4 Generator"]["1.4.4 Generator Steel"])                    +
                _ff(_T["1.5 Brake"]["1.5.1 Brake Steel"])                            +
                _ff(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"])                +
                _ff(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"])             +
                _ff(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"])        +
                _ff(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"]) +
                _ff(_T["1.13 Tower"]["1.13.1 Tower Steel"])                          +
                _ff(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"])                    +
                _ff(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"]) +
                _ff(_OSS["4.1 Steel"])                                               +
                _ff(_ONS["5.1 Steel"]),
                "kg",
            ),
            "per_FU": (
                (
                _ff(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"])          +
                _ff(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"]) +
                _ff(_T["1.4 Generator"]["1.4.4 Generator Steel"])                    +
                _ff(_T["1.5 Brake"]["1.5.1 Brake Steel"])                            +
                _ff(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"])                +
                _ff(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"])             +
                _ff(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"])        +
                _ff(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"]) +
                _ff(_T["1.13 Tower"]["1.13.1 Tower Steel"])                          +
                _ff(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"])                    +
                _ff(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"]) +
                _ff(_OSS["4.1 Steel"])                                               +
                _ff(_ONS["5.1 Steel"])
                ) * FU_FACTOR,
                "kg",
            ),
        },
        "2. Chromium Steel": {
            "per_turbine": (None, "kg"),
            "full_farm":   (_ff(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"]),          "kg"),
            "per_FU":      (_ff(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"]) * FU_FACTOR,    "kg"),
        },
        "3. Copper": {
            "per_turbine": (None, "kg"),
            "full_farm": (
                _ff(_T["1.4 Generator"]["1.4.2 Generator Copper"])                   +
                _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"]) +
                _ff(_OSS["4.2 Copper"])                                              +
                _ff(_ONS["5.2 Copper"]),
                "kg",
            ),
            "per_FU": (
                (
                _ff(_T["1.4 Generator"]["1.4.2 Generator Copper"])                   +
                _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"]) +
                _ff(_OSS["4.2 Copper"])                                              +
                _ff(_ONS["5.2 Copper"])
                ) * FU_FACTOR,
                "kg",
            ),
        },
        "4. Cast iron": {
            "per_turbine": (None, "kg"),
            "full_farm":   (_ff(_OSS["4.6 Cast iron"]) + _ff(_ONS["5.6 Cast iron"]),        "kg"),
            "per_FU":      ((_ff(_OSS["4.6 Cast iron"]) + _ff(_ONS["5.6 Cast iron"])) * FU_FACTOR,"kg"),
        },
        "5. Polyethylene PE": {
            "per_turbine": (None, "kg"),
            "full_farm": (
                _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"]) +
                _ff(_OSS["4.4 Polyethylene"])                                        +
                _ff(_ONS["5.4 Polyethylene"]),
                "kg",
            ),
            "per_FU": (
                (
                _ff(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"]) +
                _ff(_OSS["4.4 Polyethylene"])                                        +
                _ff(_ONS["5.4 Polyethylene"])
                ) * FU_FACTOR,
                "kg",
            ),
        },
        "6. Inter-array cables": {
            "6.1 Copper": {
                "per_turbine": (None, "kg"),
                "full_farm": (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.1 Copper"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.1 Copper"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.1 Copper"]),
                    "kg",
                ),
                "per_FU": (
                    (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.1 Copper"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.1 Copper"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.1 Copper"])
                    ) * FU_FACTOR,
                    "kg",
                ),
            },
            "6.2 Steel": {
                "per_turbine": (None, "kg"),
                "full_farm": (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.4 Steel"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.4 Steel"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.4 Steel"]),
                    "kg",
                ),
                "per_FU": (
                    (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.4 Steel"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.4 Steel"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.4 Steel"])
                    ) * FU_FACTOR,
                    "kg",
                ),
            },
            "6.3 Polyethylene PE": {
                "per_turbine": (None, "kg"),
                "full_farm": (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.2 Polyethylene"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.2 Polyethylene"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.2 Polyethylene"]),
                    "kg",
                ),
                "per_FU": (
                    (
                    _ff(_EI["3.1 Array Cables 240mm2 (66kV)"]["3.1.2 Polyethylene"]) +
                    _ff(_EI["3.2 Array Cables 630mm2 (66kV)"]["3.2.2 Polyethylene"]) +
                    _ff(_EI["3.3 Array Cables 800mm2 (66kV)"]["3.3.2 Polyethylene"])
                    ) * FU_FACTOR,
                    "kg",
                ),
            },
        },
        "7. Export Cables": {
            "7.1 Copper": {
                "per_turbine": (None, "kg"),
                "full_farm":   (_ff(_EI["3.4 Export Cables"]["3.4.1 Copper"]),        "kg"),
                "per_FU":      (_ff(_EI["3.4 Export Cables"]["3.4.1 Copper"]) * FU_FACTOR,  "kg"),
            },
            "7.2 Steel": {
                "per_turbine": (None, "kg"),
                "full_farm":   (_ff(_EI["3.4 Export Cables"]["3.4.3 Steel"]),         "kg"),
                "per_FU":      (_ff(_EI["3.4 Export Cables"]["3.4.3 Steel"]) * FU_FACTOR,   "kg"),
            },
            "7.3 Polyethylene PE": {
                "per_turbine": (None, "kg"),
                "full_farm":   (_ff(_EI["3.4 Export Cables"]["3.4.4 Polyethylene"]),       "kg"),
                "per_FU":      (_ff(_EI["3.4 Export Cables"]["3.4.4 Polyethylene"]) * FU_FACTOR, "kg"),
            },
        },
    },

    # =========================================================================
    # TRANSPORT
    # =========================================================================
    "Transport": {
        "1. Wind Turbine": {
            "per_turbine": (_TRANSPORT_WT_PT_KG  * DISTANCE_TO_SHORE_KM / TON_TO_KG,                    "metric ton*km"),
            "full_farm":   (_TRANSPORT_WT_KG     * DISTANCE_TO_SHORE_KM / TON_TO_KG,                    "metric ton*km"),
            "per_FU":      (_TRANSPORT_WT_KG     * DISTANCE_TO_SHORE_KM / TON_TO_KG * FU_FACTOR,        "metric ton*km"),
        },
        "2. Substructure": {
            "per_turbine": (_TRANSPORT_SUB_PT_KG * DISTANCE_TO_SHORE_KM / TON_TO_KG,                    "metric ton*km"),
            "full_farm":   (_TRANSPORT_SUB_KG    * DISTANCE_TO_SHORE_KM / TON_TO_KG,                    "metric ton*km"),
            "per_FU":      (_TRANSPORT_SUB_KG    * DISTANCE_TO_SHORE_KM / TON_TO_KG * FU_FACTOR,        "metric ton*km"),
        },
        "3.1 Array Cables": {
            "per_turbine": (None,                                                                        "metric ton*km"),
            "full_farm":   (_TRANSPORT_ARRAY_KG * DISTANCE_TO_SHORE_KM / TON_TO_KG,                          "metric ton*km"),
            "per_FU":      (_TRANSPORT_ARRAY_KG * DISTANCE_TO_SHORE_KM / TON_TO_KG * FU_FACTOR,              "metric ton*km"),
        },
        "3.2 Export Cables": {
            "per_turbine": (None,                                                                        "metric ton*km"),
            "full_farm":   (_TRANSPORT_EXPORT_KG * DISTANCE_TO_SHORE_KM / TON_TO_KG,                         "metric ton*km"),
            "per_FU":      (_TRANSPORT_EXPORT_KG * DISTANCE_TO_SHORE_KM / TON_TO_KG * FU_FACTOR,             "metric ton*km"),
        },
        "3.3 Offshore Substation": {
            "per_turbine": (None,                                                                        "metric ton*km"),
            "full_farm":   (_TRANSPORT_OSS_KG   * DISTANCE_TO_SHORE_KM / TON_TO_KG,                          "metric ton*km"),
            "per_FU":      (_TRANSPORT_OSS_KG   * DISTANCE_TO_SHORE_KM / TON_TO_KG * FU_FACTOR,              "metric ton*km"),
        },
    },

    # =========================================================================
    # INSTALLATION
    # =========================================================================
    "Installation": {
        "1. Total Vessel Fuel Consumption": {
            "per_turbine": (None,                                           "MJ"),
            "full_farm":   (INSTALLATION_ENERGY_TOTAL_MJ,                   "MJ"),
            "per_FU":      (INSTALLATION_ENERGY_TOTAL_MJ * FU_FACTOR,       "MJ"),
        },
    },

    # =========================================================================
    # OPERATION
    # =========================================================================
    "Operation": {
            "1. Routine Maintenance and Inspection": {
                "1.1 CTV Vessel Operation and Maintenance": {
                    "per_turbine": (None,              "MJ"),
                    "full_farm":   (527_754_216.88,    "MJ"),
                    "per_FU":      (7.60e-3,           "MJ"),
                },
                "1.2 LCN Vessel Operation and Maintenance": {
                    "per_turbine": (None,              "MJ"),
                    "full_farm":   (648_087_667.92,    "MJ"),
                    "per_FU":      (9.33e-3,           "MJ"),
                },
            },
            "2. Material Replacements": {
                "2.1 Wind Turbine": {
                    "2.1.1 Hub System (Rotor)": {
                        "2.1.1.1 Hub System Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.1 Hub System (Rotor)"]["1.1.1 Hub System Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                        "2.1.1.2 Hub System Uniax_Glass Fiber": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.1 Hub System (Rotor)"]["1.1.2 Hub System Uniax_Glass Fiber"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.1 Hub System (Rotor)"]["1.1.2 Hub System Uniax_Glass Fiber"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.2 Blades (Rotor)": {
                        "2.1.2.1 Blades Glass Fibre (60%)": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.2 Blades (Rotor)"]["1.2.1 Blades Glass Fibre (60%)"]) * N_INTERVENTIONS["rotor_blades"], "kg"),
                            "per_FU":      (_pt(_T["1.2 Blades (Rotor)"]["1.2.1 Blades Glass Fibre (60%)"]) * N_INTERVENTIONS["rotor_blades"] * FU_FACTOR, "kg"),
                        },
                        "2.1.2.2 Blades Epoxy (40%)": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.2 Blades (Rotor)"]["1.2.2 Blades Epoxy (40%)"]) * N_INTERVENTIONS["rotor_blades"], "kg"),
                            "per_FU":      (_pt(_T["1.2 Blades (Rotor)"]["1.2.2 Blades Epoxy (40%)"]) * N_INTERVENTIONS["rotor_blades"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.3 Main Shaft and Bearings": {
                        "2.1.3.1 Main Shaft and Bearings Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"]) * N_INTERVENTIONS["drive_train"], "kg"),
                            "per_FU":      (_pt(_T["1.3 Main Shaft and Bearings"]["1.3.1 Main Shaft and Bearings Steel"]) * N_INTERVENTIONS["drive_train"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.4 Generator": {
                        "2.1.4.1 Generator NdFeB": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.4 Generator"]["1.4.1 Generator NdFeB"]) * N_INTERVENTIONS["generator"], "kg"),
                            "per_FU":      (_pt(_T["1.4 Generator"]["1.4.1 Generator NdFeB"]) * N_INTERVENTIONS["generator"] * FU_FACTOR, "kg"),
                        },
                        "2.1.4.2 Generator Copper": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.4 Generator"]["1.4.2 Generator Copper"]) * N_INTERVENTIONS["generator"], "kg"),
                            "per_FU":      (_pt(_T["1.4 Generator"]["1.4.2 Generator Copper"]) * N_INTERVENTIONS["generator"] * FU_FACTOR, "kg"),
                        },
                        "2.1.4.3 Generator Electrical Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"]) * N_INTERVENTIONS["generator"], "kg"),
                            "per_FU":      (_pt(_T["1.4 Generator"]["1.4.3 Generator Electrical Steel"]) * N_INTERVENTIONS["generator"] * FU_FACTOR, "kg"),
                        },
                        "2.1.4.4 Generator Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.4 Generator"]["1.4.4 Generator Steel"]) * N_INTERVENTIONS["generator"], "kg"),
                            "per_FU":      (_pt(_T["1.4 Generator"]["1.4.4 Generator Steel"]) * N_INTERVENTIONS["generator"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.5 Brake": {
                        "2.1.5.1 Brake Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.5 Brake"]["1.5.1 Brake Steel"]) * N_INTERVENTIONS["hydraulic_pitch_system"], "kg"),
                            "per_FU":      (_pt(_T["1.5 Brake"]["1.5.1 Brake Steel"]) * N_INTERVENTIONS["hydraulic_pitch_system"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.6 Turret (Nacelle)": {
                        "2.1.6.1 Turret Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.6 Turret (Nacelle)"]["1.6.1 Turret Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.7 Bedplate (Nacelle)": {
                        "2.1.7.1 Bedplate Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.7 Bedplate (Nacelle)"]["1.7.1 Bedplate Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.8 Yaw System (Nacelle)": {
                        "2.1.8.1 Yaw System Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"]) * N_INTERVENTIONS["yaw_system"], "kg"),
                            "per_FU":      (_pt(_T["1.8 Yaw System (Nacelle)"]["1.8.1 Yaw System Steel"]) * N_INTERVENTIONS["yaw_system"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.9 Nacelle_cover_and_platforms (Nacelle)": {
                        "2.1.9.1 Nacelle Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.1 Nacelle Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                        "2.1.9.2 Nacelle Composite": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.2 Nacelle Composite"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.9 Nacelle_cover_and_platforms (Nacelle)"]["1.9.2 Nacelle Composite"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.10 HVAC_and_auxiliarie (Nacelle)": {
                        "2.1.10.1 HVAC_pack": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.1 HVAC_pack"]) * N_INTERVENTIONS["electrical_system"], "kg"),
                            "per_FU":      (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.1 HVAC_pack"]) * N_INTERVENTIONS["electrical_system"] * FU_FACTOR, "kg"),
                        },
                        "2.1.10.2.1 Cabling_internal Copper": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"]) * N_INTERVENTIONS["electrical_system"], "kg"),
                            "per_FU":      (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.1 Cabling_internal Copper"]) * N_INTERVENTIONS["electrical_system"] * FU_FACTOR, "kg"),
                        },
                        "2.1.10.2.2 Cabling_internal Plastic": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"]) * N_INTERVENTIONS["electrical_system"], "kg"),
                            "per_FU":      (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.2.2 Cabling_internal Plastic"]) * N_INTERVENTIONS["electrical_system"] * FU_FACTOR, "kg"),
                        },
                        "2.1.10.3 Lube_and_hydraulics": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.3 Lube_and_hydraulics"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.10 HVAC_and_auxiliarie (Nacelle)"]["1.10.3 Lube_and_hydraulics"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.11 Converter (Electrical components)": {
                        "2.1.11.1 Power Electronics Converter": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.11 Converter (Electrical components)"]["1.11.1 Power Electronics Converter"]) * N_INTERVENTIONS["power_converter"], "kg"),
                            "per_FU":      (_pt(_T["1.11 Converter (Electrical components)"]["1.11.1 Power Electronics Converter"]) * N_INTERVENTIONS["power_converter"] * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.12 Transformer (Electrical components)": {
                        "2.1.12.1 Transformer": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.12 Transformer (Electrical components)"]["1.12.1 Transformer"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.12 Transformer (Electrical components)"]["1.12.1 Transformer"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.1.13 Tower": {
                        "2.1.13.1 Tower Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_T["1.13 Tower"]["1.13.1 Tower Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_T["1.13 Tower"]["1.13.1 Tower Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                },
                "2.2 Substructure": {
                    "2.2.1 Monopile": {
                        "2.2.1.1 Monopile Steel": {
                            "per_turbine": (None, "kg"),
                            "full_farm":   (_pt(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_SUB["2.1 Monopile"]["2.1.1 Monopile Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                    "2.2.2 Transitioning piece": {
                        "2.2.2.1 Transitioning piece Steel": {
                            "per_turbine": (0, "kg"),
                            "full_farm":   (_pt(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"]) * ZERO_INTERVENTIONS, "kg"),
                            "per_FU":      (_pt(_SUB["2.2 Transitioning piece"]["2.2.1 Transitioning piece Steel"]) * ZERO_INTERVENTIONS * FU_FACTOR, "kg"),
                        },
                    },
                },
            },
        },
    # =========================================================================
    # DECOMMISSIONING
    # =========================================================================
    "Decommissioning": {
        "1. Total Vessel Fuel Consumption": {
            "per_turbine": (None,                                           "MJ"),
            "full_farm":   (INSTALLATION_ENERGY_TOTAL_MJ,                   "MJ"),
            "per_FU":      (INSTALLATION_ENERGY_TOTAL_MJ * FU_FACTOR,       "MJ"),
        },
    },

    # =========================================================================
    # END OF LIFE
    # =========================================================================
    "End of life": {
            "1. Wind Turbine": {
                "1.1 Hub System (Rotor)": {
                    "1.1.1 Hub System Steel Recycling": {
                        "per_turbine": (_eol("1.1.1 Hub System Steel",                 1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.1.1 Hub System Steel",                 1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.1.1 Hub System Steel",                 1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.1.1 Hub System Steel Landfill": {
                        "per_turbine": (_eol("1.1.1 Hub System Steel",                 1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.1.1 Hub System Steel",                 1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.1.1 Hub System Steel",                 1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.1.2 Hub System Uniax_Glass Fiber Incineration": {
                        "per_turbine": (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.1.2 Hub System Uniax_Glass Fiber Landfill": {
                        "per_turbine": (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.1.2 Hub System Uniax_Glass Fiber",     6, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.2 Blades (Rotor)": {
                    "1.2.1 Blades Glass Fibre (60%) Incineration": {
                        "per_turbine": (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.2.1 Blades Glass Fibre (60%) Landfill": {
                        "per_turbine": (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.2.1 Blades Glass Fibre (60%)",         6, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.2.2 Blades Epoxy (40%) Incineration": {
                        "per_turbine": (_eol("1.2.2 Blades Epoxy (40%)",               6, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.2.2 Blades Epoxy (40%)",               6, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.2.2 Blades Epoxy (40%)",               6, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.2.2 Blades Epoxy (40%) Landfill": {
                        "per_turbine": (_eol("1.2.2 Blades Epoxy (40%)",               6, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.2.2 Blades Epoxy (40%)",               6, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.2.2 Blades Epoxy (40%)",               6, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.3 Main Shaft and Bearings": {
                    "1.3.1 Main Shaft and Bearings Steel Recycling": {
                        "per_turbine": (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.3.1 Main Shaft and Bearings Steel Landfill": {
                        "per_turbine": (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.3.1 Main Shaft and Bearings Steel",    1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.4 Generator": {
                    "1.4.1 Generator NdFeB Landfill": {
                        "per_turbine": (_eol("1.4.1 Generator NdFeB",                  7, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.4.1 Generator NdFeB",                  7, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.1 Generator NdFeB",                  7, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.2 Generator Copper Recycling": {
                        "per_turbine": (_eol("1.4.2 Generator Copper",                 3, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.4.2 Generator Copper",                 3, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.2 Generator Copper",                 3, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.2 Generator Copper Landfill": {
                        "per_turbine": (_eol("1.4.2 Generator Copper",                 3, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.4.2 Generator Copper",                 3, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.2 Generator Copper",                 3, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.3 Generator Electrical Steel Recycling": {
                        "per_turbine": (_eol("1.4.3 Generator Electrical Steel",       3, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.4.3 Generator Electrical Steel",       3, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.3 Generator Electrical Steel",       3, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.3 Generator Electrical Steel Landfill": {
                        "per_turbine": (_eol("1.4.3 Generator Electrical Steel",       3, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.4.3 Generator Electrical Steel",       3, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.3 Generator Electrical Steel",       3, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.4 Generator Steel Recycling": {
                        "per_turbine": (_eol("1.4.4 Generator Steel",                  2, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.4.4 Generator Steel",                  2, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.4 Generator Steel",                  2, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.4.4 Generator Steel Landfill": {
                        "per_turbine": (_eol("1.4.4 Generator Steel",                  2, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.4.4 Generator Steel",                  2, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.4.4 Generator Steel",                  2, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.5 Brake": {
                    "1.5.1 Brake Steel Recycling": {
                        "per_turbine": (_eol("1.5.1 Brake Steel",                      1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.5.1 Brake Steel",                      1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.5.1 Brake Steel",                      1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.5.1 Brake Steel Landfill": {
                        "per_turbine": (_eol("1.5.1 Brake Steel",                      1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.5.1 Brake Steel",                      1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.5.1 Brake Steel",                      1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.6 Turret (Nacelle)": {
                    "1.6.1 Turret Steel Recycling": {
                        "per_turbine": (_eol("1.6.1 Turret Steel",                     1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.6.1 Turret Steel",                     1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.6.1 Turret Steel",                     1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.6.1 Turret Steel Landfill": {
                        "per_turbine": (_eol("1.6.1 Turret Steel",                     1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.6.1 Turret Steel",                     1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.6.1 Turret Steel",                     1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.7 Bedplate (Nacelle)": {
                    "1.7.1 Bedplate Steel Recycling": {
                        "per_turbine": (_eol("1.7.1 Bedplate Steel",                   1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.7.1 Bedplate Steel",                   1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.7.1 Bedplate Steel",                   1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.7.1 Bedplate Steel Landfill": {
                        "per_turbine": (_eol("1.7.1 Bedplate Steel",                   1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.7.1 Bedplate Steel",                   1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.7.1 Bedplate Steel",                   1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.8 Yaw System (Nacelle)": {
                    "1.8.1 Yaw System Steel Recycling": {
                        "per_turbine": (_eol("1.8.1 Yaw System Steel",                 2, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.8.1 Yaw System Steel",                 2, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.8.1 Yaw System Steel",                 2, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.8.1 Yaw System Steel Landfill": {
                        "per_turbine": (_eol("1.8.1 Yaw System Steel",                 2, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.8.1 Yaw System Steel",                 2, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.8.1 Yaw System Steel",                 2, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.9 Nacelle_cover_and_platforms (Nacelle)": {
                    "1.9.1 Nacelle Steel Recycling": {
                        "per_turbine": (_eol("1.9.1 Nacelle Steel",                    1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.9.1 Nacelle Steel",                    1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.9.1 Nacelle Steel",                    1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.9.1 Nacelle Steel Landfill": {
                        "per_turbine": (_eol("1.9.1 Nacelle Steel",                    1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.9.1 Nacelle Steel",                    1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.9.1 Nacelle Steel",                    1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.9.2 Nacelle Composite Glass Fibre Incineration": {
                        "per_turbine": (_eol("1.9.2 Nacelle Composite",                6, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.9.2 Nacelle Composite",                6, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.9.2 Nacelle Composite",                6, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.9.2 Nacelle Composite Glass Fibre Landfill": {
                        "per_turbine": (_eol("1.9.2 Nacelle Composite",                6, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.9.2 Nacelle Composite",                6, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.9.2 Nacelle Composite",                6, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.10 HVAC_and_auxiliarie (Nacelle)": {
                    "1.10.1 HVAC_pack Treatment and Disposal": {
                        "per_turbine": (_eol("1.10.1 HVAC_pack",                       7, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.10.1 HVAC_pack",                       7, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.1 HVAC_pack",                       7, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.10.2.1 Cabling_internal Copper Recycling": {
                        "per_turbine": (_eol("1.10.2.1 Cabling_internal Copper",       3, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.10.2.1 Cabling_internal Copper",       3, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.2.1 Cabling_internal Copper",       3, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.10.2.1 Cabling_internal Copper Landfill": {
                        "per_turbine": (_eol("1.10.2.1 Cabling_internal Copper",       3, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.10.2.1 Cabling_internal Copper",       3, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.2.1 Cabling_internal Copper",       3, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.10.2.2 Cabling_internal Plastic Landfill": {
                        "per_turbine": (_eol("1.10.2.2 Cabling_internal Plastic",      6, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.10.2.2 Cabling_internal Plastic",      6, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.2.2 Cabling_internal Plastic",      6, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.10.2.2 Cabling_internal Plastic Incineration": {
                        "per_turbine": (_eol("1.10.2.2 Cabling_internal Plastic",      6, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.10.2.2 Cabling_internal Plastic",      6, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.2.2 Cabling_internal Plastic",      6, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.10.3 Lube_and_hydraulics Incineration": {
                        "per_turbine": (_eol("1.10.3 Lube_and_hydraulics",             5, "incineration"),                        "kg"),
                        "full_farm":   (_eol("1.10.3 Lube_and_hydraulics",             5, "incineration") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.10.3 Lube_and_hydraulics",             5, "incineration") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.11 Converter (Electrical components)": {
                    "1.11.1 Power Electronics Converter": {
                        "per_turbine": (_eol("1.11.1 Power Electronics Converter",     7, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.11.1 Power Electronics Converter",     7, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.11.1 Power Electronics Converter",     7, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.12 Transformer (Electrical components)": {
                    "1.12.1 Transformer Treatment and Disposal": {
                        "per_turbine": (_eol("1.12.1 Transformer",                     7, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.12.1 Transformer",                     7, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.12.1 Transformer",                     7, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "1.13 Tower": {
                    "1.13.1 Tower Steel Recycling": {
                        "per_turbine": (_eol("1.13.1 Tower Steel",                     1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("1.13.1 Tower Steel",                     1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.13.1 Tower Steel",                     1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "1.13.1 Tower Steel Landfill": {
                        "per_turbine": (_eol("1.13.1 Tower Steel",                     1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("1.13.1 Tower Steel",                     1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("1.13.1 Tower Steel",                     1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
            },
            "2. Substructure": {
                "2.1 Monopile": {
                    "2.1.1 Monopile Steel Recycling": {
                        "per_turbine": (_eol("2.1.1 Monopile Steel",                   11, "recycled"),                        "kg"),
                        "full_farm":   (_eol("2.1.1 Monopile Steel",                   11, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("2.1.1 Monopile Steel",                   11, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "2.1.1 Monopile Steel Landfill": {
                        "per_turbine": (_eol("2.1.1 Monopile Steel",                   11, "landfill"),                        "kg"),
                        "full_farm":   (_eol("2.1.1 Monopile Steel",                   11, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("2.1.1 Monopile Steel",                   11, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
                "2.2 Transitioning piece": {
                    "2.2.1 Transitioning piece Steel Recycling": {
                        "per_turbine": (_eol("2.2.1 Transitioning piece Steel",        1, "recycled"),                        "kg"),
                        "full_farm":   (_eol("2.2.1 Transitioning piece Steel",        1, "recycled") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("2.2.1 Transitioning piece Steel",        1, "recycled") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                    "2.2.1 Transitioning piece Steel Landfill": {
                        "per_turbine": (_eol("2.2.1 Transitioning piece Steel",        1, "landfill"),                        "kg"),
                        "full_farm":   (_eol("2.2.1 Transitioning piece Steel",        1, "landfill") * N_TURBINES,           "kg"),
                        "per_FU":      (_eol("2.2.1 Transitioning piece Steel",        1, "landfill") * N_TURBINES * FU_FACTOR, "kg"),
                    },
                },
            },
            "3. Electrical Infrastructure": {
                "3.1 Array Cables 240mm2 (66kV)": {
                    "per_turbine": (None,                                                                        "kg"),
                    "full_farm":   (_eol_ff("3.1 Array Cables 240mm2 (66kV)", 8, "recycled"),                   "kg"),
                    "per_FU":      (_eol_ff("3.1 Array Cables 240mm2 (66kV)", 8, "recycled") * FU_FACTOR,       "kg"),
                },
                "3.2 Array Cables 630mm2 (66kV)": {
                    "per_turbine": (None,                                                                        "kg"),
                    "full_farm":   (_eol_ff("3.2 Array Cables 630mm2 (66kV)", 8, "recycled"),                   "kg"),
                    "per_FU":      (_eol_ff("3.2 Array Cables 630mm2 (66kV)", 8, "recycled") * FU_FACTOR,       "kg"),
                },
                "3.3 Array Cables 800mm2 (66kV)": {
                    "per_turbine": (None,                                                                        "kg"),
                    "full_farm":   (_eol_ff("3.3 Array Cables 800mm2 (66kV)", 8, "recycled"),                   "kg"),
                    "per_FU":      (_eol_ff("3.3 Array Cables 800mm2 (66kV)", 8, "recycled") * FU_FACTOR,       "kg"),
                },
                "3.4 Export Cables": {
                    "per_turbine": (None,                                                                        "kg"),
                    "full_farm":   (_eol_ff("3.4 Export Cables",               8, "recycled"),                   "kg"),
                    "per_FU":      (_eol_ff("3.4 Export Cables",               8, "recycled") * FU_FACTOR,       "kg"),
                },
                "3.5 Offshore Substation": {
                    "3.5.1 Offshore Substation Steel Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.1 Steel", 3, "recycled"),                               "kg"),
                        "per_FU":      (_eol_ff("4.1 Steel", 3, "recycled") * FU_FACTOR,                   "kg"),
                    },
                    "3.5.1 Offshore Substation Steel Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.1 Steel", 3, "landfill"),                               "kg"),
                        "per_FU":      (_eol_ff("4.1 Steel", 3, "landfill") * FU_FACTOR,                   "kg"),
                    },
                    "3.5.2 Offshore Substation Copper Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.2 Copper", 3, "recycled"),                              "kg"),
                        "per_FU":      (_eol_ff("4.2 Copper", 3, "recycled") * FU_FACTOR,                  "kg"),
                    },
                    "3.5.2 Offshore Substation Copper Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.2 Copper", 3, "landfill"),                              "kg"),
                        "per_FU":      (_eol_ff("4.2 Copper", 3, "landfill") * FU_FACTOR,                  "kg"),
                    },
                    "3.5.3 Offshore Substation Aluminium Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.3 Aluminium", 3, "recycled"),                           "kg"),
                        "per_FU":      (_eol_ff("4.3 Aluminium", 3, "recycled") * FU_FACTOR,               "kg"),
                    },
                    "3.5.3 Offshore Substation Aluminium Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.3 Aluminium", 3, "landfill"),                           "kg"),
                        "per_FU":      (_eol_ff("4.3 Aluminium", 3, "landfill") * FU_FACTOR,               "kg"),
                    },
                    "3.5.4 Offshore Substation Polyethylene Incineration": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.4 Polyethylene", 6, "incineration"),                    "kg"),
                        "per_FU":      (_eol_ff("4.4 Polyethylene", 6, "incineration") * FU_FACTOR,        "kg"),
                    },
                    "3.5.4 Offshore Substation Polyethylene Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.4 Polyethylene", 6, "landfill"),                        "kg"),
                        "per_FU":      (_eol_ff("4.4 Polyethylene", 6, "landfill") * FU_FACTOR,            "kg"),
                    },
                    "3.5.5 Offshore Substation Lubricating Oil Incineration": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.5 Lubricating oil", 5, "incineration"),                 "kg"),
                        "per_FU":      (_eol_ff("4.5 Lubricating oil", 5, "incineration") * FU_FACTOR,     "kg"),
                    },
                    "3.5.6 Offshore Substation Cast Iron Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.6 Cast iron", 3, "recycled"),                           "kg"),
                        "per_FU":      (_eol_ff("4.6 Cast iron", 3, "recycled") * FU_FACTOR,               "kg"),
                    },
                    "3.5.6 Offshore Substation Cast Iron Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.6 Cast iron", 3, "landfill"),                           "kg"),
                        "per_FU":      (_eol_ff("4.6 Cast iron", 3, "landfill") * FU_FACTOR,               "kg"),
                    },
                    "3.5.7 Modified Organic Natural Materials": {
                        "3.5.7.1 Kraft Paper": {
                            "per_turbine": (None,       "kg"),
                            "full_farm":   (None,       "kg"),
                            "per_FU":      (None,       "kg"),
                        },
                        "3.5.7.2 Vegetable Oil Methyl Ester Incineration": {
                            "per_turbine": (None,                                                                               "kg"),
                            "full_farm":   (_eol_ff("4.7.2 Vegetable oil methyl ester", 5, "incineration"),                    "kg"),
                            "per_FU":      (_eol_ff("4.7.2 Vegetable oil methyl ester", 5, "incineration") * FU_FACTOR,        "kg"),
                        },
                    },
                    "3.5.8 Ceramic / Glass Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.8 Ceramic / glass", 13, "landfill"),                    "kg"),
                        "per_FU":      (_eol_ff("4.8 Ceramic / glass", 13, "landfill") * FU_FACTOR,        "kg"),
                    },
                    "3.5.9 SF6 Gas": {
                        "per_turbine": (None,       "kg"),
                        "full_farm":   (None,       "kg"),
                        "per_FU":      (None,       "kg"),
                    },
                    "3.5.10 Offshore Substation Zinc Alloys Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.10 Zinc alloys", 3, "recycled"),                        "kg"),
                        "per_FU":      (_eol_ff("4.10 Zinc alloys", 3, "recycled") * FU_FACTOR,            "kg"),
                    },
                    "3.5.10 Offshore Substation Zinc Alloys Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("4.10 Zinc alloys", 3, "landfill"),                        "kg"),
                        "per_FU":      (_eol_ff("4.10 Zinc alloys", 3, "landfill") * FU_FACTOR,            "kg"),
                    },
                },
                "3.6 Onshore Substation": {
                    "3.6.1 Onshore Substation Steel Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.1 Steel", 3, "recycled"),                               "kg"),
                        "per_FU":      (_eol_ff("5.1 Steel", 3, "recycled") * FU_FACTOR,                   "kg"),
                    },
                    "3.6.1 Onshore Substation Steel Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.1 Steel", 3, "landfill"),                               "kg"),
                        "per_FU":      (_eol_ff("5.1 Steel", 3, "landfill") * FU_FACTOR,                   "kg"),
                    },
                    "3.6.2 Onshore Substation Copper Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.2 Copper", 3, "recycled"),                              "kg"),
                        "per_FU":      (_eol_ff("5.2 Copper", 3, "recycled") * FU_FACTOR,                  "kg"),
                    },
                    "3.6.2 Onshore Substation Copper Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.2 Copper", 3, "landfill"),                              "kg"),
                        "per_FU":      (_eol_ff("5.2 Copper", 3, "landfill") * FU_FACTOR,                  "kg"),
                    },
                    "3.6.3 Onshore Substation Aluminium Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.3 Aluminium", 3, "recycled"),                           "kg"),
                        "per_FU":      (_eol_ff("5.3 Aluminium", 3, "recycled") * FU_FACTOR,               "kg"),
                    },
                    "3.6.3 Onshore Substation Aluminium Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.3 Aluminium", 3, "landfill"),                           "kg"),
                        "per_FU":      (_eol_ff("5.3 Aluminium", 3, "landfill") * FU_FACTOR,               "kg"),
                    },
                    "3.6.4 Onshore Substation Polyethylene Incineration": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.4 Polyethylene", 4, "incineration"),                    "kg"),
                        "per_FU":      (_eol_ff("5.4 Polyethylene", 4, "incineration") * FU_FACTOR,        "kg"),
                    },
                    "3.6.4 Onshore Substation Polyethylene Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.4 Polyethylene", 4, "landfill"),                        "kg"),
                        "per_FU":      (_eol_ff("5.4 Polyethylene", 4, "landfill") * FU_FACTOR,            "kg"),
                    },
                    "3.6.5 Onshore Substation Lubricating Oil Incineration": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.5 Lubricating oil", 5, "incineration"),                 "kg"),
                        "per_FU":      (_eol_ff("5.5 Lubricating oil", 5, "incineration") * FU_FACTOR,     "kg"),
                    },
                    "3.6.6 Onshore Substation Cast Iron Recycling": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.6 Cast iron", 3, "recycled"),                           "kg"),
                        "per_FU":      (_eol_ff("5.6 Cast iron", 3, "recycled") * FU_FACTOR,               "kg"),
                    },
                    "3.6.6 Onshore Substation Cast Iron Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.6 Cast iron", 3, "landfill"),                           "kg"),
                        "per_FU":      (_eol_ff("5.6 Cast iron", 3, "landfill") * FU_FACTOR,               "kg"),
                    },
                    "3.6.7 Modified Organic Natural Materials": {
                        "3.6.7.1 Kraft Paper": {
                            "per_turbine": (None,       "kg"),
                            "full_farm":   (None,       "kg"),
                            "per_FU":      (None,       "kg"),
                        },
                        "3.6.7.2 Vegetable Oil Methyl Ester Incineration": {
                            "per_turbine": (None,                                                                               "kg"),
                            "full_farm":   (_eol_ff("5.7.2 Vegetable oil methyl ester", 5, "incineration"),                    "kg"),
                            "per_FU":      (_eol_ff("5.7.2 Vegetable oil methyl ester", 5, "incineration") * FU_FACTOR,        "kg"),
                        },
                    },
                    "3.6.8 Ceramic / Glass Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.8 Ceramic / glass", 13, "landfill"),                    "kg"),
                        "per_FU":      (_eol_ff("5.8 Ceramic / glass", 13, "landfill") * FU_FACTOR,        "kg"),
                    },
                    "3.6.9 SF6 Gas": {
                        "per_turbine": (None,       "kg"),
                        "full_farm":   (None,       "kg"),
                        "per_FU":      (None,       "kg"),
                    },
                    "3.6.10 Onshore Substation Concrete Landfill": {
                        "per_turbine": (None,                                                               "kg"),
                        "full_farm":   (_eol_ff("5.10 Concrete", 12, "landfill"),                          "kg"),
                        "per_FU":      (_eol_ff("5.10 Concrete", 12, "landfill") * FU_FACTOR,              "kg"),
                    },
                },
            },
        },
}