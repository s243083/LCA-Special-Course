import numpy as np
import pandas as pd
from datetime import datetime
from astral import LocationInfo
from astral.sun import sun
import re
import sys


def constant_control(yaw=0, power=100.0, shutdown=False):
    """
    Returns a controller that sets per-turbine constant setpoints.
    yaw, power, shutdown can each be scalars or lists indexed by tid.
    """
    def control_fn(wsp, TI, wdir, ts=None):
        # Support per-turbine: pick index if list, else use scalar

        result = {}
        for tid in range(len(yaw) if isinstance(yaw, (list, np.ndarray)) else 1):
            y = yaw[tid] if isinstance(yaw, (list, np.ndarray)) else yaw
            p = power[tid] if isinstance(power, (list, np.ndarray)) else power
            s = shutdown[tid] if isinstance(shutdown, (list, np.ndarray)) else shutdown
            result[tid] = {
                "yaw": y,
                "power": p,
                "control_requested_shutdown": s
            }
        return result

    control_fn.describe = lambda: {
        "type": "constant_control",
        "yaw": yaw.tolist() if isinstance(yaw, np.ndarray) else yaw,
        "power": power.tolist() if isinstance(power, np.ndarray) else power,
        "control_requested_shutdown": shutdown.tolist() if isinstance(shutdown, np.ndarray) else shutdown
    }

    return control_fn


def lookup_controller(lookup_df, bin_specs, lookup_file=None):
    """
    Creates a wind farm controller function based on a lookup table with binned setpoints
    in a multi-column format.

    Each row in the lookup_df must contain:
        - Bin center values: ['wsp_bin', 'wdir_bin', 'TI_bin']
        - Turbine-specific setpoints: ['yaw_0', 'power_0', 'yaw_1', 'power_1', ..., 'yaw_N', 'power_N']

    Parameters:
    ----------
    lookup_df : pd.DataFrame
        DataFrame containing the lookup table with bin centers and setpoints.
    bin_specs : dict with keys 'wsp', 'wdir', 'TI'
        Each key maps to a dict with:
        - if the variable is truly binned:
          {'constant': False, 'edges': np.ndarray[edges]}
        - if the variable is fixed (i.e., a single center for all cases):
          {'constant': True, 'center': float}
    lookup_file : str or None
        Optional path to the CSV file used to create this lookup_df.
        Used for reproducibility in settings.json.


    Returns:
    -------
    control_fn : function
        Function with signature (wsp, TI, wdir, tid, ts=None) -> dict
        where dict contains:
            - 'yaw': float, requested yaw angle for this turbine and condition
            - 'power': float, requested power level (normalized)
        Falls back to {'yaw': 0, 'power': 100.0} if input is outside bins or no match is found.

    Assumptions:
    -----------
    - The turbine IDs are inferred from the suffixes in the 'yaw_<i>' / 'power_<i>'
      columns. All turbines present in the row are returned every time.
    - This function does not read or return any rule/extra inputs; it is purely a
      base setpoint mapper.
    - `control_fn.describe()` records the lookup file, the bin edges/centers used,
      and the number of entries in the table, to capture provenance.

    """

    # === Validate required columns ===
    required_columns = {'wsp_bin', 'wdir_bin', 'TI_bin'}
    if not required_columns.issubset(lookup_df.columns):
        missing = required_columns - set(lookup_df.columns)
        raise ValueError(f"Missing required columns in lookup_df: {missing}")
    

    # === Convert lookup_df to a dictionary for fast access ===
    # Key: (wsp_bin_center, wdir_bin_center, TI_bin_center) -> row dict
    lookup_dict = {}
    for _, row in lookup_df.iterrows():
        key = (row['wsp_bin'], row['wdir_bin'], row['TI_bin'])
        lookup_dict[key] = row.to_dict()

    # === Helper: map continuous value to bin center ===
    def find_bin(value, bins):
        """
        Maps a continuous value to its bin center using the provided bin edges.

        Parameters:
        ----------
        value : float
            The continuous input value to bin.
        bins : array-like
            The bin edges.

        Returns:
        -------
        bin_center : float or None
            The center of the bin this value falls into, or None if outside range.
        """
        bin_idx = np.digitize([value], bins)[0] - 1

        # Out-of-range check
        if bin_idx < 0 or bin_idx >= len(bins) - 1:
            return None

        # Compute bin center from edges
        bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
        return np.round(bin_center, 5)

    def make_mapper(spec):
        if spec['constant']:
            # always map to the single center
            return lambda _: spec['center']
        else:
            # map via find_bin on the full range
            return lambda x: find_bin(x, spec['edges'])

    wsp_mapper  = make_mapper(bin_specs['wsp'])
    wdir_mapper = make_mapper(bin_specs['wdir'])
    TI_mapper   = make_mapper(bin_specs['TI'])
    

    # === Controller function ===
    def control_fn(wsp, TI, wdir, tid, ts=None):
        """
        Computes the control setpoints for a given turbine and condition.

        Parameters:
        ----------
        wsp : float
            Wind speed for this timestep.
        TI : float
            Turbulence intensity for this timestep.
        wdir : float
            Wind direction for this timestep.
        tid : int
            Turbine ID.
        ts : datetime or None
            Optional timestamp (not used here).

        Returns:
        -------
        dict
            {'yaw': float, 'power': float}
        """
        wsp_bin  = wsp_mapper(wsp)
        wdir_bin = wdir_mapper(wdir)
        TI_bin   = TI_mapper(TI)
        
        if None in (wsp_bin, wdir_bin, TI_bin):
            return {}  #  out-of-range: fallback to normal operation for all turbines

        key = (wsp_bin, wdir_bin, TI_bin)
        row = lookup_dict.get(key, None)


        
        if row is None:
            return {} #  out-of-range: fallback to normal operation for all turbines

        # Extract yaw and power for all turbine IDs found in the row
        turbine_controls = {}
        for col in row:
            if col.startswith('yaw_'):
                tid = int(col.split('_')[1])
                yaw = row.get(f'yaw_{tid}', 0.0)
                power = row.get(f'power_{tid}', 100.0)
                turbine_controls[tid] = {'yaw': yaw, 'power': power}

        return turbine_controls
    
    # === Attach describe for reproducibility ===
    control_fn.describe = lambda: {
        "type": "lookup_controller_multi_column",
        "lookup_file": lookup_file,
        "wsp_bins": (
        bin_specs["wsp"]["edges"].tolist()
        if not bin_specs["wsp"]["constant"]
        else [bin_specs["wsp"]["center"]]
        ),
        "wdir_bins": (
            bin_specs["wdir"]["edges"].tolist()
            if not bin_specs["wdir"]["constant"]
            else [bin_specs["wdir"]["center"]]
        ),
        "TI_bins": (
            bin_specs["TI"]["edges"].tolist()
            if not bin_specs["TI"]["constant"]
            else [bin_specs["TI"]["center"]]
        ),
            "lookup_entries": len(lookup_df)
        }

    return control_fn


def evaluate_condition_groups(groups, groups_logic, context):
    """
    Evaluate multiple groups of conditions with AND/OR logic.

    Each group is a dict:
      {
        "conditions": [
            {"var": "wsp", "op": ">", "value": 5},
            ...
        ],
        "logic": "AND" or "OR"
      }

    groups_logic:
      - "AND" means all groups must be True
      - "OR" means any group can be True

    context:
      Dict mapping variable names to their current values:
        {"wsp": 7.0, "TI": 0.12, "price": 45, ...}

    Returns:
      True if the full group logic is satisfied, False otherwise.
    """
    group_results = []
    for group in groups:
        results = []
        for cond in group["conditions"]:
            var = cond["var"]
            op = cond["op"]
            val = cond["value"]

            v = context.get(var)
            if v is None:
                raise ValueError(f"Unknown variable: {var}")

            if op == ">":
                results.append(v > val)
            elif op == "<":
                results.append(v < val)
            elif op == "==":
                results.append(v == val)
            elif op == ">=":
                results.append(v >= val)
            elif op == "<=":
                results.append(v <= val)
            else:
                raise ValueError(f"Unsupported operator: {op}")

        group_result = all(results) if group["logic"].upper() == "AND" else any(results)
        group_results.append(group_result)


    if groups_logic.upper() == "AND":
        return all(group_results)
    elif groups_logic.upper() == "OR":
        return any(group_results)
    else:
        raise ValueError(f"Unsupported groups_logic: {groups_logic}")

def shutdown_rule(groups, groups_logic="OR"):
    """
    Returns a shutdown rule function that checks groups of conditions.

    Parameters:
    ----------
    groups: list of groups
      Each group: dict with 'conditions' list and 'logic' ("AND"/"OR")
    groups_logic: str
      "AND" or "OR" to combine groups.

    Returns:
    -------
    rule function: callable(wsp, TI, wdir, tid, ts, extra_inputs, params) -> dict
    """
    def _rule(wsp, TI, wdir, ts, extra_inputs):
        # Always present core ambient inputs
        context = {
            "wsp": wsp,
            "TI": TI,
            "wdir": wdir
        }

     
        # for var in ["price", "temperature", "curtailment", "bat_act1", "grid_signal"]:
        #     if var in extra_inputs:
        #         context[var] = extra_inputs[var]
        # Bring ALL extra inputs into context (scalars for this timestep)
        if extra_inputs:
            context.update(extra_inputs)

    

        triggered = evaluate_condition_groups(groups, groups_logic, context)
        result = {"control_requested_shutdown": triggered, "disable_control": False}
        return result
        

    _rule.describe = lambda: {
        "type": "shutdown_rule",
        "groups": groups,
        "groups_logic": groups_logic
    }
    
    return _rule

def wffc_activation_rule(groups, groups_logic="OR"):
    """
    Returns a WF control activation rule function that checks groups of conditions.

    Parameters:
    ----------
    groups: list of groups
      Each group: dict with 'conditions' list and 'logic' ("AND"/"OR")
    groups_logic: str
      "AND" or "OR" to combine groups.

    Returns:
    -------
    rule function: callable(wsp, TI, wdir, tid, ts, extra_inputs, params) -> dict
    """
    def _rule(wsp, TI, wdir, ts, extra_inputs):
        # Always present core ambient inputs
        context = {
            "wsp": wsp,
            "TI": TI,
            "wdir": wdir
        }

        # Optional: if extra_inputs has these, add them
        # for var in ["price", "temperature", "curtailment", "bat_act1", "grid_signal"]:
        #     if var in extra_inputs:
        #         context[var] = extra_inputs[var]
        if extra_inputs:
            context.update(extra_inputs)


        triggered = evaluate_condition_groups(groups, groups_logic, context)
        result = {"control_requested_shutdown": False, "disable_control": triggered}
        return result

    _rule.describe = lambda: {
        "type": "wf_activation_rule",
        "groups": groups,
        "groups_logic": groups_logic
    }
    return _rule


def bat_shutdown_static(wsp, TI, wdir, tid, ts, extra_inputs, bat_params):
    """
    Static bat shutdown rule based on date, time, wsp, and temperature.

    Parameters:
    - wsp, TI, wdir, tid, ts: standard inputs
    - extra_inputs: dict with 'temperature'
    - bat_params: dict with date range, wsp threshold, temp threshold, and location info

    Returns:
    - dict with control_requested_shutdown and disable_control flags
    """
    temp = extra_inputs.get("temperature", 15)
    location = bat_params.get("location", None)
    ts = pd.Timestamp(ts)

    # Extract date information
    date1 = datetime.strptime(bat_params['date1'], '%d.%m')
    date2 = datetime.strptime(bat_params['date2'], '%d.%m')
    current_month, current_day = ts.month, ts.day
    within_date_range = (date1.month, date1.day) <= (current_month, current_day) <= (date2.month, date2.day)

    # Calculate sunrise/sunset
    is_nighttime = False
    if location:
        loc = LocationInfo(**location)
        s = sun(loc.observer, date=ts.date(), tzinfo=loc.timezone)
        is_nighttime = (ts.time() < s['sunrise'].time()) or (ts.time() > s['sunset'].time())



    # Shutdown logic
    if within_date_range and is_nighttime and (wsp < bat_params['wsp']) and (temp > bat_params['temp']):
        return {"control_requested_shutdown": True, "disable_control": False}
    else:
        return {"control_requested_shutdown": False, "disable_control": False}


def bat_shutdown_dynamic(wsp, TI, wdir, tid, ts, extra_inputs, bat_params):
    """
    Dynamic bat shutdown rule based on bat activity indicators.

    Parameters:
    - extra_inputs: dict with bat activity flags (bat_act1, bat_act2, ...)

    Returns:
    - dict with control_requested_shutdown and disable_control flags
    """

    species_selection = bat_params.get("species", "all")

    if species_selection == "all":
        bat_activity = sum(extra_inputs.get(f"bat_act{i}", 0) for i in range(1, 11))
    else:
        bat_activity = sum(extra_inputs.get(f"bat_act{i}", 0) for i in species_selection) 

    if bat_activity > 0:
        return {"control_requested_shutdown": True, "disable_control": False}
    else:
        return {"control_requested_shutdown": False, "disable_control": False}

def bat_dynamic_mock(activity_keys=("bat_act1", "bat_act2")):
    """
    If ANY of the given activity_keys is truthy at this timestep, request shutdown for all turbines.
    """
    def _ov(wsp, TI, wdir, ts, extra_inputs, turbine_ids):
        active = any(bool(extra_inputs.get(k, 0)) for k in activity_keys)
        result = {}
        if active:
            for tid in turbine_ids:
                result[tid] = {
                    "yaw": None,
                    "power": None,
                    "disable_wffc": False,
                    "control_requested_shutdown": True,
                }
        else:
            for tid in turbine_ids:
                result[tid] = {
                    "yaw": None,
                    "power": None,
                    "disable_wffc": False,
                    "control_requested_shutdown": False,
                }
        return result


    _ov.describe = lambda: {"family": "bat", "type": "bat_dynamic_mock"}
    return _ov

def yaw_band_override(wsp_min, wsp_max, yaw_angle, target_turbines="all"):
    """
    Set yaw=yaw_angle for target_turbines when wsp is in [wsp_min, wsp_max].
    Leaves power unchanged, does not disable WFFC, and does not request shutdown.
    """
    wsp_min = float(wsp_min)
    wsp_max = float(wsp_max)
    yaw_angle = float(yaw_angle)

    def _ov(wsp, TI, wdir, ts, extra_inputs, turbine_ids):
        in_band = (float(wsp) >= wsp_min) and (float(wsp) <= wsp_max)
        if target_turbines == "all":
            targets = set(turbine_ids)
        else:
            # accept list/tuple/np array of ints
            targets = set(int(t) for t in target_turbines)

        result = {}
        for tid in turbine_ids:
            if in_band and tid in targets:
                result[tid] = {
                    "yaw": yaw_angle,
                    "power": None,
                    "disable_wffc": False,
                    "control_requested_shutdown": False,
                }
            else:
                result[tid] = {
                    "yaw": None,
                    "power": None,
                    "disable_wffc": False,
                    "control_requested_shutdown": False,
                }
        return result

    _ov.describe = lambda: {
        "family": "yaw",
        "type": "yaw_band_override",
        "wsp_min": wsp_min,
        "wsp_max": wsp_max,
        "yaw_angle": yaw_angle,
        "target_turbines": target_turbines,
    }
    return _ov

def rule_controller(rule_config, rule_library):
    """
    Builds a list of wrapped rule functions based on the user config and the rule library.

    Each rule in the config is matched to its factory in the rule_library.
    The factory is called with the user-defined parameters to create an instance.
    The instance is then wrapped with a standard signature and a describe method.

    Parameters
    ----------
    rule_config : dict
        A mapping of rule names (str) to their user-defined parameter dicts.
        Example:
            {
                "shutdown_rule_1": {
                    "groups": [...],
                    "groups_logic": "OR"
                },
                ...
            }

    rule_library : dict
        A mapping of rule names (str) to the rule factory functions.
        Example:
            {
                "shutdown_rule_1": shutdown_rule,
                "wf_activation_rule_1": wffc_activation_rule,
                ...
            }

    Returns
    -------
    list
        A list of wrapped rule functions. Each function takes:
          (wsp, TI, wdir, tid, ts,  extra_inputs, params)
        and returns a dict with:
          {
              "control_requested_shutdown": bool,
              "disable_control": bool,
              "rule_name": str
          }
    """
    rules = []

    for name, params in rule_config.items():
        if name not in rule_library:
            raise ValueError(f"Unknown rule: {name}")

        # Get the rule factory from the library
        rule_factory = rule_library[name]

        # === Call the factory with its config ===
        rule_fn = rule_factory(**params)

        if rule_factory is shutdown_rule:
            family = "shutdown"
        elif rule_factory is wffc_activation_rule:
            family = "wffc_deactivate"
        else:
            family = getattr(rule_factory, "__name__", "custom")

        # === Wrap the instantiated rule for consistent interface ===
        def wrapped_rule(
            wsp, TI, wdir, ts, extra_inputs,turbine_ids,
            _rule_fn=rule_fn, _name=name
        ):
            eff = _rule_fn(wsp, TI, wdir, ts, extra_inputs) 
            shut = bool(eff.get("control_requested_shutdown", False))
            dis  = bool(eff.get("disable_control", False))

            result = {}
            for tid in turbine_ids:
                d = {"rule_name": _name}
                if shut:
                    d["control_requested_shutdown"] = True
                if dis:
                    d["disable_control"] = True
                result[tid] = d
            return result

        # Add a describe method for reproducibility in settings.json
        wrapped_rule.describe = lambda _name=name, _params=params, _family=family: {
            "name": _name,
            "parameters": _params,
            "type": _family
        }

        rules.append(wrapped_rule)

    return rules



def override_controller(override_config):
    """
    Build a list of *override* controller instances from user config.

    Usage pattern (in the notebook):
        override_flags = {
            "grid_curtailment": {"family": "grid", "cap": 60.0},
            "bat_dynamic":      {"family": "bat",  "species": "all"},
            # ...
        }
        override_wrappers = override_controller(override_flags)

    Contracts
    ---------
    - Each *factory* in `override_library` must be a callable that accepts **params** via kwargs:
          factory(**params) -> override_fn
      and returns a function with signature:
          override_fn(wsp, TI, wdir, tid, ts, extra_inputs, params_unused) -> dict

    - The returned dict is a *partial effect* for that turbine/timestep.
      Include only keys you intend to change:
          {
              "control_requested_shutdown": bool,   # optional
              "disable_wffc": bool,                 # optional (interpreted as yaw reset to baseline)
              "yaw": float,                         # optional; absolute degrees
              "power": float,                       # optional; absolute % rated
              "tag": str                            # optional; for trace/debug
          }

    - `override_config` is an *ordered* dict: its item order defines application priority.
      Within the "override" stage, later instances overwrite earlier setpoints ("last write wins").
      A requested shutdown is terminal for that turbine at this timestep.

    Returns
    -------
    list[callable]
        Each callable takes (wsp, TI, wdir, tid, ts, extra_inputs) and returns a partial-effect dict.
        Each also has `.describe()` for metadata in settings.json:
            {
              "name": "<instance_name>",
              "type": "override",
              "family": "<bat|grid|...>",   # from params['family'] if present, else inferred from factory name
              "parameters": {...}           # exactly what you passed in override_config
            }

    Raises
    ------
    ValueError
        If an instance name is not present in override_library.
    """
    overrides = []
    this_mod = sys.modules[__name__]
    for name, params in override_config.items():
        
        try:
            factory = getattr(this_mod, name)
        except AttributeError:
            raise ValueError(
                f"Unknown override controller: '{name}'. "
                "Define a factory with this exact name in control_interface.py."
            )

        if not callable(factory):
            raise TypeError(f"'{name}' is not callable (got {type(factory)}).")
        # Instantiate the concrete override with its params (closure over params)
        override_fn = factory(**params)

        # Derive a simple 'family' tag:
        # 1) explicit in params['family'] (preferred, optional)
        # 2) else from factory.__name__ prefix before first underscore
        family = None
        if hasattr(override_fn, "describe"):
            try:
                desc = override_fn.describe()
                if isinstance(desc, dict):
                    family = desc.get("family")
            except Exception:
                family = None

        # Runtime wrapper (bind function & name to avoid late-binding issues)
        def wrapped_override(
            wsp, TI, wdir, ts, extra_inputs, turbine_ids,
            _fn=override_fn
        ):
            return _fn(wsp, TI, wdir, ts, extra_inputs, turbine_ids)
         
        # Rich, stable metadata for settings.json (capture values as defaults!)
        wrapped_override.describe = (
            lambda _name=name, _params=params, _family=family: {
                "name": _name,
                "type": "override",
                "family": _family,
                "parameters": _params,
            }
        )

        overrides.append(wrapped_override)

    return overrides



def combine_controllers(base_control, rule_wrappers=None,override_wrappers=None):
    """
    Combines a base controller with rule functions.

    Rule priority follows order of rule_wrappers.
    """
    def combined_control(wsp, TI, wdir, ts=None, extra_inputs={}):
        
        # Get base WF control once for all turbines
        base_outputs = base_control(wsp, TI, wdir, ts)
        
        # Start with base control output
        # Assumption base WF control only sets setpoints (yaw, power) no shutdown or overrides
        control_dict = {}
        for tid, base in base_outputs.items():
            control_dict[tid] = {
                "yaw": base.get("yaw", 0.0),
                "power": base.get("power", 100.0),
                "control_requested_shutdown": base.get("control_requested_shutdown", False),
                "triggered_shutdown_by": None,
                "disabled_wf_control_by": None
            }

        turbine_ids = list(control_dict.keys())
        # Apply each rule per turbine. Here the hierarchy of the controllers is imposed based on the order of rule_wrappers.
        # If a rule requests shutdown, it sets yaw and power to 0.0 and 0.0 respectively and shutdown flag is issued. Rest of rules are skipped for that turbine.
        # Here also the WFFC deactivation is triggered by the rule controller setting the disable_control flag to True.
        if rule_wrappers:
            for rule_fn in rule_wrappers:
                
                eff = rule_fn(wsp, TI, wdir, ts, extra_inputs, turbine_ids)
                for tid in turbine_ids:
                    rule_out = eff.get(tid)
                    if rule_out.get("control_requested_shutdown", False):
                        control_dict[tid]["yaw"] = 0.0
                        control_dict[tid]["power"] = 0.0
                        control_dict[tid]["control_requested_shutdown"] = True
                        control_dict[tid]["triggered_shutdown_by"] = rule_out.get("rule_name")
                        # No need to check further rules
                        continue

                    if rule_out.get("disable_control", False):
                        control_dict[tid]["yaw"] = 0.0
                        control_dict[tid]["power"] = 100.0
                        control_dict[tid]["control_requested_shutdown"] = False
                        control_dict[tid]["disabled_wf_control_by"] = rule_out.get("rule_name")
            
        if override_wrappers:
            for ov_fn in override_wrappers:
                eff = ov_fn(wsp, TI, wdir, ts, extra_inputs, turbine_ids)
                # try to get an identifier for trace
                ov_name = None
                if hasattr(ov_fn, "describe"):
                    try:
                        d = ov_fn.describe()
                        if isinstance(d, dict):
                            # prefer 'name', else 'type', else None
                            ov_name =  d.get("name", d.get("type"))
                    except Exception:
                        ov_name = None

                for tid in turbine_ids:
                    if control_dict[tid]["control_requested_shutdown"]:
                        # terminal from earlier stage
                        continue

                    ov = eff.get(tid)
                    if ov is None:
                        raise ValueError(f"Override wrapper returned no entry for turbine id {tid}")

                    if ov.get("control_requested_shutdown", False):
                        control_dict[tid]["yaw"] = 0.0
                        control_dict[tid]["power"] = 0.0
                        control_dict[tid]["control_requested_shutdown"] = True
                        # reuse same field for provenance
                        control_dict[tid]["triggered_shutdown_by"] = ov_name or "override"
                        continue

                    if ov.get("disable_wffc", False):
                        control_dict[tid]["yaw"] = 0.0
                        control_dict[tid]["power"] = 100.0
                        control_dict[tid]["disabled_wf_control_by"] = ov_name or "override"

                    # yaw/power: only apply if not None (None means "leave as is")
                    if ov.get("yaw") is not None:
                        control_dict[tid]["yaw"] = float(ov["yaw"])
                    if ov.get("power") is not None:
                        control_dict[tid]["power"] = float(ov["power"])

        return control_dict

    # # Metadata for settings.json
    # if rule_wrappers or override_wrappers:
    #     # We have stacked rules: so we describe the full hierarchy
    #     def _describe():
    #         descr = {
    #                 "type": "combined_control",
    #                 "base": base_control.describe() if hasattr(base_control, "describe") else str(base_control),
    #                 "rules": [r.describe() if hasattr(r, "describe") else str(r) for r in rule_wrappers] if rule_wrappers else []
    #             }
    #         if override_wrappers:
    #                 descr["overrides"] = [o.describe() if hasattr(o, "describe") else str(o) for o in override_wrappers]
    #     combined_control.describe = _describe
    # else:
    #     # No stacked rules: we’re just the base controller!
    #     combined_control.describe = lambda: base_control.describe() if hasattr(base_control, "describe") else str(base_control)

    # --- Metadata for settings.json (REPLACE THIS WHOLE BLOCK) ---

    def _describe_base():
        return base_control.describe() if hasattr(base_control, "describe") else str(base_control)

    # normalize Nones to empty lists so downstream logic is consistent
    rule_wrappers = rule_wrappers or []
    override_wrappers = override_wrappers or []

    if rule_wrappers or override_wrappers:
        def _describe():
            descr = {
                "type": "combined_control",
                "base": _describe_base(),
                "rules": [r.describe() if hasattr(r, "describe") else str(r) for r in rule_wrappers],
            }
            if override_wrappers:
                descr["overrides"] = [o.describe() if hasattr(o, "describe") else str(o) for o in override_wrappers]
            return descr
        combined_control.describe = _describe
    else:
        combined_control.describe = _describe_base

    return combined_control

