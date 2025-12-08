from __future__ import annotations
from pathlib import Path
from attrs import define, field

import pandas as pd
import itertools, time, hashlib, gc, traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, Callable

# Your existing imports
from core.File_Handling import load_yaml, process_duration_fields
from core.utils import save_sceanarios
from core.File_Handling import calculate_duration_in_hours  # if used elsewhere
from core.Data_classes import FromDictMixin
from core.ValueWindEnv import ValueWindEnv
from core.SimulationConfig import SimulationConfig


# ------------------------- Scenario Builder ------------------------- #
# Uses your existing utilities & classes:
# - load_yaml, process_duration_fields
# - Configuration, Simulation


# --- tiny helpers (kept minimal) --------------------------------------------

def _load_base_config_yaml(library_path: Union[str, Path],
                           config_path: Union[str, Path]) -> dict:
    lp, cp = Path(library_path), Path(config_path)
    cfg = load_yaml((lp / cp).parent, cp.name)
    return process_duration_fields(cfg)

def _set_by_dotted_path(d: dict, path: str, value: Any) -> None:
    """
    Minimal dotted-path setter that assumes all intermediate nodes are dicts.
    e.g. _set_by_dotted_path(cfg, "CAPEX_overrides.material_price.copper", 5000)
    """
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _apply_overrides(base_cfg: dict, overrides: Mapping[str, Any]) -> dict:
    cfg = dict(base_cfg)
    for k, v in overrides.items():
        _set_by_dotted_path(cfg, k, v)
    return cfg

def _scenario_id(overrides: Mapping[str, Any], seed: int) -> str:
    payload = repr(sorted(overrides.items())) + f"|{seed}"
    return hashlib.sha1(payload.encode()).hexdigest()[:10]

def _append_index(results_root: Union[str, Path], rows: List[Mapping[str, Any]]) -> None:
    root = Path(results_root); root.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    pq, csv = root / "index.parquet", root / "index.csv"
    try:
        if pq.exists():
            df_all = pd.concat([pd.read_parquet(pq), df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(pq, index=False)
    except Exception:
        if csv.exists():
            df_new.to_csv(csv, mode="a", header=False, index=False)
        else:
            df_new.to_csv(csv, index=False)

# ------------------------- Core classes -------------------------------------#

@define(auto_attribs=True)
class Simulation:
    """The primary API to interact with the simulation methodologies."""
    library_path: Path
    config: 'Configuration'
    simulation_config: SimulationConfig
    env: ValueWindEnv = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._setup_simulation()

    @classmethod
    def from_config(
        cls,
        library_path: Union[str, Path],
        config: Union[str, Path, dict, 'Configuration'],
        simulation_config: Union[SimulationConfig, Mapping[str, Any]],
    ):
        """
        Build a Simulation from a config and a REQUIRED SimulationConfig
        (or a dict of flags).
        """
        library_path = Path(library_path)

        # Load and process YAML / dict into Configuration
        if isinstance(config, (str, Path)):
            config_path = library_path / config
            config = load_yaml(config_path.parent, config_path.name)
            config = process_duration_fields(config)
        if isinstance(config, dict):
            config = Configuration.from_dict(config)
        if not isinstance(config, Configuration):
            raise TypeError("``config`` must be a dictionary or ``Configuration`` object!")

        # Build SimulationConfig from dict or pass through
        if isinstance(simulation_config, Mapping):
            sim_cfg = SimulationConfig.from_dict(simulation_config)
        else:
            sim_cfg = simulation_config

        return cls(library_path=library_path, config=config, simulation_config=sim_cfg)

    def _setup_simulation(self):
        # Pass simulation_config into the environment
        self.env = ValueWindEnv(self.config, simulation_config=self.simulation_config)

    def run(self, until: Union[int, float, None] = None):
        """
        Run the simulation using the SimulationConfig attached to this Simulation.
        """
        self.env.run_simulation(until=until)


@define(auto_attribs=True)
class Configuration(FromDictMixin):
    """Configuration for the Simulation."""
    name: str
    valuewind_inputFolder: str
    Finex_inputFiles: str
    Capex_inputFiles: dict[str, str]
    Opex_inputFiles: dict[str, str]
    MetEnv_inputFiles: dict[str, str]
    Material_inputFiles: dict[str, str]
    WindFarm_inputFiles: dict[str, str]
    WindTurbine_inputFiles: dict[str, str]
    Valuation_inputFiles: dict[str, str]
    Market_inputFiles: dict[str, str]
    LTE_inputFiles: dict[str, str]
    Project_Duration: dict[str, Union[int, str]]
    Project_StartDate: str
    WF_OperationsStart: dict[str, Union[int, str]]
    WF_OperationsEnd_h: int
    WF_OperationsEnd: dict[str, Union[int, str]]
    WF_OperationsStart_h: int
    TimeStep: int
    Project_Duration_h: int

    experiment_name: str
    result_directory: str
    scenario_label: str
    scenario_id: str
    seed: int

    WindFarm_overrides: dict[str, Any] = field(factory=dict)
    Revenue_overrides: dict[str, Any] = field(factory=dict)
    CAPEX_overrides: dict[str, Any] = field(factory=dict)

# ------------------------- Scenario generation ------------------------------#

def _last_segment(path: str) -> str:
    return path.split(".")[-1] if path else path

def _build_label(overrides: Mapping[str, Any]) -> str:
    if not overrides:
        return "baseline"
    return "__".join(f"{_last_segment(k)}={v}" for k, v in sorted(overrides.items()))

def generate_scenarios(
    library_path: Union[str, Path],
    base_config_path: Union[str, Path],
    parameter_space: Mapping[str, Sequence[Any]],
    base_seed: int = 0,
    replicates: int = 1,
    max_runs: Optional[int] = None,
    *,
    zip_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build scenarios as dicts: {'scenario_id','label','overrides','seed'}.

    Behavior:
      • Keys listed together in a zip group are paired positionally
        (with broadcasting of singletons within that group).
      • Keys not in any zip group remain independent and are Cartesian-combined
        with all groups.
      • Multiple zip groups are supported; groups are Cartesian with each other.

    Examples:
      zip_groups = {
        "materials": [
          "CAPEX_overrides.material_data.Commodity.Copper.CostParameters.material_cost",
          "CAPEX_overrides.material_data.Commodity.Steel.CostParameters.material_cost",
        ]
      }
    """
    # Validate base config path
    _ = _load_base_config_yaml(library_path, base_config_path)

    items = sorted(parameter_space.items(), key=lambda kv: kv[0])
    all_keys = [k for k, _ in items]
    param_lists = {k: list(v) for k, v in items}

    # Assign keys to groups (or None)
    key_to_group: Dict[str, Optional[str]] = {k: None for k in all_keys}
    if zip_groups:
        seen = set()
        for gname, gkeys in zip_groups.items():
            for k in gkeys:
                if k in seen:
                    raise ValueError(f"Key '{k}' appears in multiple zip groups.")
                if k not in parameter_space:
                    raise KeyError(f"Key '{k}' in zip group '{gname}' not found in parameter_space.")
                key_to_group[k] = gname
                seen.add(k)

    # Partition: grouped vs standalone
    groups: Dict[str, List[str]] = {}
    for k, g in key_to_group.items():
        if g is not None:
            groups.setdefault(g, []).append(k)
    standalone_keys = [k for k, g in key_to_group.items() if g is None]

    # Build zipped combos for each group (with broadcasting inside the group)
    grouped_combo_lists: List[List[Dict[str, Any]]] = []
    for gname, gkeys in sorted(groups.items(), key=lambda kv: kv[0]):
        vlists = [param_lists[k] for k in gkeys]
        lengths = [len(v) for v in vlists]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            grouped_combo_lists.append([{}])
            continue

        bad = [(k, l) for k, l in zip(gkeys, lengths) if l not in (1, max_len)]
        if bad:
            detail = ", ".join(f"{k} (len={l})" for k, l in bad)
            raise ValueError(
                f"In zip group '{gname}', each list must have length 1 or {max_len}. "
                f"Non-broadcastable lists: {detail}"
            )

        group_combos: List[Dict[str, Any]] = []
        for i in range(max_len):
            d = {}
            for k, vlist in zip(gkeys, vlists):
                d[k] = vlist[i] if len(vlist) > 1 else vlist[0]
            group_combos.append(d)
        grouped_combo_lists.append(group_combos)

    # Build Cartesian combos for standalone keys
    if standalone_keys:
        standalone_vals = [param_lists[k] for k in standalone_keys]
        standalone_combos = [dict(zip(standalone_keys, combo))
                             for combo in itertools.product(*standalone_vals)]
    else:
        standalone_combos = [{}]

    # Cartesian across: [each group combos] × [standalone combos]
    all_blocks = grouped_combo_lists + [standalone_combos]
    if not all_blocks:
        all_overrides_dicts = [{}]
    else:
        all_overrides_dicts = []
        for combo in itertools.product(*all_blocks):
            merged = {}
            for part in combo:
                merged.update(part)
            all_overrides_dicts.append(merged)

    # Build scenarios with deterministic seeds + labels
    scenarios: List[Dict[str, Any]] = []
    count = 0
    for overrides in all_overrides_dicts:
        label = _build_label(overrides)
        for r in range(replicates):
            seed = abs(hash((base_seed, tuple(sorted(overrides.items())), r))) % (2**31)
            sid = _scenario_id(overrides, seed)
            scenarios.append({"scenario_id": sid, "label": label, "overrides": overrides, "seed": seed})
            count += 1
            if max_runs is not None and count >= max_runs:
                return scenarios

    return scenarios

# ------------------------- Runner -------------------------------------------#

def run_scenarios(
    library_path: Union[str, Path],
    base_config_path: Union[str, Path],
    scenarios: Iterable[Mapping[str, Any]],
    *,
    simulation_config: Union[SimulationConfig, Mapping[str, Any]],
    name: Optional[str] = None,
    result_directory: Optional[Union[str, Path]] = None,
    debug: bool = True,
    on_result: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> pd.DataFrame:
    """
    Sequentially run scenarios and return a status DataFrame.

    All scenarios are executed with the same SimulationConfig (or dict of flags).
    """
    base_cfg = _load_base_config_yaml(library_path, base_config_path)

    # Normalize simulation_config once
    if isinstance(simulation_config, Mapping):
        sim_cfg = SimulationConfig.from_dict(simulation_config)
    else:
        sim_cfg = simulation_config

    rows: list[dict[str, Any]] = []

    for sc in scenarios:
        sid = str(sc.get("scenario_id"))
        label = str(sc.get("label", sid))
        overrides = dict(sc.get("overrides", {}))
        seed = int(sc.get("seed", 0))

        # Build config with overrides
        cfg = _apply_overrides(base_cfg, overrides)

        # Inject experiment metadata BEFORE building the Simulation
        cfg = dict(cfg)  # shallow copy
        if name is not None:
            cfg["experiment_name"] = str(name)
        if result_directory is not None:
            cfg["result_directory"] = str(result_directory)
        cfg["scenario_label"] = label
        cfg["scenario_id"] = sid
        cfg["seed"] = seed

        t0 = time.time()
        status, err, tb_txt, duration_s = "success", None, None, None

        try:
            # Build Simulation with shared SimulationConfig
            sim = Simulation.from_config(library_path, cfg, simulation_config=sim_cfg)

            if hasattr(sim.env, "seed"):
                sim.env.seed = seed

            sim.run()

        except Exception as e:
            duration_s = time.time() - t0
            status = "failed"
            err = f"{type(e).__name__}: {e}"
            tb_txt = traceback.format_exc()
            if debug:
                raise

        finally:
            if duration_s is None:
                duration_s = time.time() - t0
            try:
                del sim
            except Exception:
                pass
            gc.collect()

        row = {
            "scenario_id": sid,
            "label": label,
            "seed": seed,
            "status": status,
            "duration_s": duration_s,
            "error_message": err,
            "traceback": tb_txt,
            "experiment_name": name,
            "result_directory": str(result_directory) if result_directory is not None else None,
        }
        rows.append(row)
        if on_result:
            on_result(row)

    return pd.DataFrame(rows)


# ------------------------- One-call sweep -----------------------------------#
def sweep(
    library_path: Union[str, Path],
    base_config_path: Union[str, Path],
    parameter_space: Mapping[str, Sequence[Any]],
    simulation_config: Union[SimulationConfig, Mapping[str, Any]],
    base_seed: int = 0,
    replicates: int = 1,
    max_runs: Optional[int] = None,
    resume: bool = True,  # kept for API shape; not used
    *,
    name: Optional[str] = None,
    result_directory: Optional[Union[str, Path]] = None,
    zip_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    One-call sweep over a parameter space.

    Requires a SimulationConfig (or dict of flags) that will be used
    for all scenarios.
    """
    scenarios = generate_scenarios(
        library_path=library_path,
        base_config_path=base_config_path,
        parameter_space=parameter_space,
        base_seed=base_seed,
        replicates=replicates,
        max_runs=max_runs,
        zip_groups=zip_groups,
    )

    # Save scenarios JSON BEFORE running the simulations
    save_root = Path(result_directory) if result_directory is not None else Path("results")
    save_sceanarios(scenarios=scenarios, result_directory=save_root, name=name)

    return run_scenarios(
        library_path=library_path,
        base_config_path=base_config_path,
        scenarios=scenarios,
        simulation_config=simulation_config,
        name=name,
        result_directory=result_directory,
    )
