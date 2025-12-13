from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod

import gc
import hashlib
import itertools
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
import logging

import pandas as pd
from attrs import define, field

from core.File_Handling import load_yaml, process_duration_fields
from core.File_Handling import calculate_duration_in_hours  # if used elsewhere
from core.utils import save_sceanarios
from core.Data_classes import FromDictMixin
from core.ValueWindEnv import ValueWindEnv
from core.SimulationConfig import SimulationConfig
from core.utils import init_experiment_logging

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _load_base_config_yaml(
    library_path: Union[str, Path],
    config_path: Union[str, Path],
) -> dict:
    """Load a base YAML config and run it through duration-field processing."""
    lp, cp = Path(library_path), Path(config_path)
    cfg = load_yaml((lp / cp).parent, cp.name)
    return process_duration_fields(cfg)


def _set_by_dotted_path(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted path.

    Example:
        _set_by_dotted_path(cfg, "CAPEX_overrides.material_price.copper", 5000)
    """
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_overrides(base_cfg: dict, overrides: Mapping[str, Any]) -> dict:
    """Return a *shallow* copy of base_cfg with dotted-path overrides applied."""
    cfg = dict(base_cfg)
    for k, v in overrides.items():
        _set_by_dotted_path(cfg, k, v)
    return cfg


def _scenario_id(overrides: Mapping[str, Any], seed: int) -> str:
    """Deterministic short scenario id from overrides+seed."""
    payload = repr(sorted(overrides.items())) + f"|{seed}"
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _append_index(results_root: Union[str, Path], rows: List[Mapping[str, Any]]) -> None:
    """Append rows to an index parquet / csv in results_root."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
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


def _last_segment(path: str) -> str:
    return path.split(".")[-1] if path else path


def _build_label(overrides: Mapping[str, Any]) -> str:
    if not overrides:
        return "baseline"
    return "__".join(f"{_last_segment(k)}={v}" for k, v in sorted(overrides.items()))


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@define(auto_attribs=True)
class Configuration(FromDictMixin):
    """Configuration for the Simulation.

    This is the structured representation of the YAML config *plus*
    scenario metadata (experiment/scenario ids, seed, etc.).
    """

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

    # Experiment / scenario metadata
    experiment_name: str
    result_directory: str
    scenario_label: str
    scenario_id: str
    seed: int

    # Optional overrides the code understands:
    WindFarm_overrides: dict[str, Any] = field(factory=dict)
    Revenue_overrides: dict[str, Any] = field(factory=dict)
    CAPEX_overrides: dict[str, Any] = field(factory=dict)


@define(auto_attribs=True)
class Simulation:
    """Low-level wrapper that owns a ValueWindEnv instance.

    Users normally shouldn't instantiate this directly – they should
    go through `build_experiment`. It remains useful internally and
    for advanced users.
    """

    library_path: Path
    config: Configuration
    simulation_config: SimulationConfig
    env: ValueWindEnv = field(init=False)
    logger: logging.Logger = field(
        factory=lambda: logging.getLogger("winpact.sim")
    )

    def __attrs_post_init__(self) -> None:
        self._setup_simulation()

    @classmethod
    def from_config(
        cls,
        library_path: Union[str, Path],
        config: Union[str, Path, dict, Configuration],
        simulation_config: Union[SimulationConfig, Mapping[str, Any]],
        logger: Optional[logging.Logger] = None, 
    ) -> Simulation:
        """Build a Simulation from a config and a SimulationConfig.

        `config` may be:
          * path relative to `library_path`
          * raw dict
          * `Configuration` instance
        """
        library_path = Path(library_path).resolve()

        # Load and process YAML / dict into Configuration
        if isinstance(config, (str, Path)):
            config_path = library_path / config
            raw = load_yaml(config_path.parent, config_path.name)
            raw = process_duration_fields(raw)
            config = Configuration.from_dict(raw)
        elif isinstance(config, dict):
            config = Configuration.from_dict(config)

       # --- normalize valuewind_inputFolder (avoid Inputs/HKN duplication) ---

        cfg_dir = config_path.parent.resolve() if "config_path" in locals() else library_path.resolve()

        raw = getattr(config, "valuewind_inputFolder", None)
        if raw:
            rel = Path(str(raw).replace("\\", "/"))

            if rel.is_absolute():
                base = rel.resolve()
            else:
                # Candidate 1: relative to config directory (old behavior)
                cand1 = (cfg_dir / rel).resolve()

                # If cand1 duplicates the library folder (e.g. .../Inputs/HKN/Inputs/HKN), use cfg_dir instead
                # This happens when Config.yaml already lives inside Inputs/HKN
                if cfg_dir.as_posix().endswith(rel.as_posix()):
                    base = cfg_dir
                # Otherwise, prefer a path relative to the config dir's parent (common repo layout: examples/Inputs/HKN)
                elif (cfg_dir.parent / rel).exists():
                    base = (cfg_dir.parent / rel).resolve()
                else:
                    # Fall back to cand1
                    base = cand1

            config.valuewind_inputFolder = str(base)



        if not isinstance(config, Configuration):
            raise TypeError("`config` must be a dictionary or `Configuration` object!")

        # Build SimulationConfig from dict or pass through
        if isinstance(simulation_config, Mapping):
            sim_cfg = SimulationConfig.from_dict(simulation_config)
        else:
            sim_cfg = simulation_config

        if logger is None:
            logger = logging.getLogger("winpact.sim")

        return cls(
            library_path=library_path,
            config=config,
            simulation_config=sim_cfg,
            logger=logger,
        )

    def _setup_simulation(self) -> None:
        # Pass simulation_config into the environment
        self.env = ValueWindEnv(self.config, simulation_config=self.simulation_config, logger=self.logger)

    def run(self, until: Union[int, float, None] = None) -> None:
        """Run the underlying ValueWindEnv using the stored SimulationConfig."""
        self.env.run_simulation(until=until)


# ---------------------------------------------------------------------------
# Experiment abstractions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionConfig:
    """
    Execution settings for running experiments.
    Default is sequential so local + HPC work the same out of the box.
    """
    backend: str = "sequential"   # "sequential" | "process" | "thread"
    n_jobs: int = 1
    chunksize: int = 1

class Experiment(ABC):
    """High-level façade for single-run or sweep experiments."""

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the experiment and return a status/results DataFrame."""
        raise NotImplementedError


@define(auto_attribs=True)
class _Scenario:
    """Internal helper representing one scenario configuration."""

    scenario_id: str
    label: str
    overrides: Dict[str, Any]
    seed: int


def _normalise_sim_cfg(
    simulation_config: Union[SimulationConfig, Mapping[str, Any]]
) -> SimulationConfig:
    if isinstance(simulation_config, Mapping):
        return SimulationConfig.from_dict(simulation_config)
    return simulation_config


def _normalise_execution_config(execution):
    """
    Accepts:
      - None  -> default sequential
      - ExecutionConfig
      - dict-like: {"backend": "...", "n_jobs": ..., "chunksize": ...}
    """
    if execution is None:
        return ExecutionConfig()

    # Allow dict-like config
    if isinstance(execution, dict):
        return ExecutionConfig(
            backend=str(execution.get("backend", "sequential")),
            n_jobs=int(execution.get("n_jobs", 1)),
            chunksize=int(execution.get("chunksize", 1)),
        )

    # Already an ExecutionConfig
    return execution


def _build_configuration_dict(
    base_cfg: Mapping[str, Any],
    scenario: _Scenario,
    *,
    name: Optional[str],
    result_directory: Optional[Union[str, Path]],
) -> Dict[str, Any]:
    """Inject scenario + experiment metadata into a base configuration dict."""
    cfg = dict(base_cfg)
    if name is not None:
        cfg["experiment_name"] = str(name)
    if result_directory is not None:
        cfg["result_directory"] = str(result_directory)
    cfg["scenario_label"] = scenario.label
    cfg["scenario_id"] = scenario.scenario_id
    cfg["seed"] = int(scenario.seed)
    return cfg


def _run_single_scenario(
    *,
    library_path: Path,
    base_cfg: Mapping[str, Any],
    scenario: _Scenario,
    simulation_config: SimulationConfig,
    name: Optional[str],
    result_directory: Optional[Union[str, Path]],
    debug: bool,
    logger_name: str,
) -> Dict[str, Any]:
    """Core routine to run one scenario and return a status row.

    This is shared between SingleExperiment and SweepExperiment.
    """
    
    logger = logging.getLogger(logger_name)

    t0 = time.time()
    status: str = "success"
    err: Optional[str] = None
    tb_txt: Optional[str] = None
    duration_s: Optional[float] = None

    # Build config with overrides + metadata
    cfg_overridden = _apply_overrides(dict(base_cfg), scenario.overrides)
    cfg_full = _build_configuration_dict(
        cfg_overridden,
        scenario,
        name=name,
        result_directory=result_directory,
    )

    logger.info(
        "[scenario:start] id=%s label=%s seed=%s result_dir=%s experiment=%s",
        scenario.scenario_id, scenario.label, scenario.seed,
        str(result_directory), str(name),
    )

    try:
        sim = Simulation.from_config(
            library_path, cfg_full,
            simulation_config=simulation_config,
            logger=logger,
        )

        # Seed injection, if your environment supports it
        if hasattr(sim.env, "seed"):
            sim.env.seed = scenario.seed

        sim.run()

        logger.info(
            "[scenario:done] id=%s duration_s=%.3f",
            scenario.scenario_id, time.time() - t0,
        )

    except Exception as e:
        logger.exception(
            "[scenario:fail] id=%s label=%s seed=%s",
            scenario.scenario_id, scenario.label, scenario.seed,
        )
        raise


    finally:
        if duration_s is None:
            duration_s = time.time() - t0
        try:
            del sim
        except Exception:
            pass
        gc.collect()

    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "seed": scenario.seed,
        "status": status,
        "duration_s": duration_s,
        "error_message": err,
        "traceback": tb_txt,
        "experiment_name": name,
        "result_directory": str(result_directory) if result_directory is not None else None,
    }

def _run_scenarios(
    *,
    scenarios,
    library_path,
    base_cfg,
    simulation_config,
    name,
    result_directory,
    debug,
    logger_name,
    execution: ExecutionConfig,
    on_result=None,
):
    """
    Runs scenarios sequentially by default.
    Optional thread/process execution when execution.n_jobs > 1.
    """
    # Default: sequential (safe everywhere)
    if execution.backend == "sequential" or execution.n_jobs <= 1:
        rows = []
        for sc in scenarios:
            row = _run_single_scenario(
                library_path=library_path,
                base_cfg=base_cfg,
                scenario=sc,
                simulation_config=simulation_config,
                name=name,
                result_directory=result_directory,
                debug=debug,
                logger_name=logger_name,
            )
            rows.append(row)
            if on_result:
                on_result(row)
        return rows

    # Parallel mode: don’t allow debug=True, because your debug mode likely re-raises exceptions
    if debug:
        raise ValueError("debug=True is not compatible with parallel execution. Set debug=False.")

    if execution.backend not in ("process", "thread"):
        raise ValueError(f"Unknown execution backend: {execution.backend!r}")

    Executor = ProcessPoolExecutor if execution.backend == "process" else ThreadPoolExecutor

    rows = []
    with Executor(max_workers=execution.n_jobs) as ex:
        futures = [
            ex.submit(
                _run_single_scenario,
                library_path=library_path,
                base_cfg=base_cfg,
                scenario=sc,
                simulation_config=simulation_config,
                name=name,
                result_directory=result_directory,
                debug=False,
                logger_name=logger_name,
            )
            for sc in scenarios
        ]

        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            if on_result:
                on_result(row)

    # Optional: restore original scenario order
    order = {s.scenario_id: i for i, s in enumerate(scenarios)}
    rows.sort(key=lambda r: order.get(r.get("scenario_id"), 10**9))
    return rows


@define(auto_attribs=True)
class SingleExperiment(Experiment):
    """Experiment that conceptually represents a *single* configuration.

    If you only want one realisation, you'll get a single run.
    If you request `replicates > 1` via `build_experiment`, you'll
    actually get a SweepExperiment instead.
    """

    library_path: Path
    base_config_path: Path
    simulation_config: SimulationConfig
    scenario: _Scenario
    name: Optional[str] = None
    result_directory: Optional[Union[str, Path]] = None
    debug: bool = True
    logger: logging.Logger = field(factory=lambda: logging.getLogger("winpact.single"))
    execution: ExecutionConfig = field(factory=ExecutionConfig)


    def run(self) -> pd.DataFrame:
        base_cfg = _load_base_config_yaml(self.library_path, self.base_config_path)
        row = _run_single_scenario(
            library_path=self.library_path,
            base_cfg=base_cfg,
            scenario=self.scenario,
            simulation_config=self.simulation_config,
            name=self.name,
            result_directory=self.result_directory,
            debug=self.debug,
            logger_name=self.logger.name,

        )
        return pd.DataFrame([row])


@define(auto_attribs=True)
class SweepExperiment(Experiment):
    """Experiment representing potentially many scenarios (sweep, plus replicates)."""

    library_path: Path
    base_config_path: Path
    simulation_config: SimulationConfig
    scenarios: List[_Scenario]
    name: Optional[str] = None
    result_directory: Optional[Union[str, Path]] = None
    debug: bool = True
    on_result: Optional[Callable[[Mapping[str, Any]], None]] = None
    logger: logging.Logger = field(factory=lambda: logging.getLogger("winpact.sweep"))
    execution: ExecutionConfig = field(factory=ExecutionConfig)


    def run(self) -> pd.DataFrame:
        base_cfg = _load_base_config_yaml(self.library_path, self.base_config_path)

        rows: List[Dict[str, Any]] = []

        rows = _run_scenarios(
            scenarios=self.scenarios,
            library_path=self.library_path,
            base_cfg=base_cfg,
            simulation_config=self.simulation_config,
            name=self.name,
            result_directory=self.result_directory,
            debug=self.debug,
            logger_name=self.logger.name,
            execution=self.execution,
            on_result=self.on_result,
        )
        df = pd.DataFrame(rows)


        # Optionally persist an index of runs for bookkeeping
        if self.result_directory is not None:
            _append_index(self.result_directory, rows)

        return df


# ---------------------------------------------------------------------------
# Scenario construction utilities
# ---------------------------------------------------------------------------


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
    """Build scenarios as dicts: {'scenario_id','label','overrides','seed'}.

    Behavior:
      • Keys listed together in a zip group are paired positionally
        (with broadcasting of singletons within that group).
      • Keys not in any zip group remain independent and are
        Cartesian-combined with all groups.
      • Multiple zip groups are supported; groups are Cartesian with
        each other.
    """
    # Validate base config path early to fail fast on typos
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
            d: Dict[str, Any] = {}
            for k, vlist in zip(gkeys, vlists):
                d[k] = vlist[i] if len(vlist) > 1 else vlist[0]
            group_combos.append(d)
        grouped_combo_lists.append(group_combos)

    # Build Cartesian combos for standalone keys
    if standalone_keys:
        standalone_vals = [param_lists[k] for k in standalone_keys]
        standalone_combos = [
            dict(zip(standalone_keys, combo))
            for combo in itertools.product(*standalone_vals)
        ]
    else:
        standalone_combos = [{}]

    # Cartesian across: [each group combos] × [standalone combos]
    all_blocks = grouped_combo_lists + [standalone_combos]
    if not all_blocks:
        all_overrides_dicts = [{}]
    else:
        all_overrides_dicts = []
        for combo in itertools.product(*all_blocks):
            merged: Dict[str, Any] = {}
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
            scenarios.append(
                {"scenario_id": sid, "label": label, "overrides": overrides, "seed": seed}
            )
            count += 1
            if max_runs is not None and count >= max_runs:
                return scenarios

    return scenarios


def _build_single_config_scenarios(
    *,
    overrides: Optional[Mapping[str, Any]],
    base_seed: int,
    replicates: int,
    explicit_seed: Optional[int] = None,
) -> List[_Scenario]:
    """Create `_Scenario` objects for a single configuration plus replicates.

    If `replicates == 1`, this returns exactly one scenario.
    If `explicit_seed` is provided, it is used as a base to derive
    replicate seeds; otherwise `base_seed` is used.
    """
    overrides_dict: Dict[str, Any] = dict(overrides or {})
    label = _build_label(overrides_dict)
    scenarios: List[_Scenario] = []

    base = explicit_seed if explicit_seed is not None else base_seed

    for r in range(replicates):
        seed = abs(hash((base, tuple(sorted(overrides_dict.items())), r))) % (2**31)
        sid = _scenario_id(overrides_dict, seed)
        scenarios.append(
            _Scenario(
                scenario_id=sid,
                label=label,
                overrides=overrides_dict,
                seed=seed,
            )
        )

    return scenarios


# ---------------------------------------------------------------------------
# Unified public entry point
# ---------------------------------------------------------------------------


def build_experiment(
    library_path: Union[str, Path],
    base_config_path: Union[str, Path],
    simulation_config: Union[SimulationConfig, Mapping[str, Any]],
    *,
    # single-run style:
    overrides: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    # sweep style:
    parameter_space: Optional[Mapping[str, Sequence[Any]]] = None,
    base_seed: int = 0,
    replicates: int = 1,
    max_runs: Optional[int] = None,
    zip_groups: Optional[Mapping[str, Sequence[str]]] = None,
    # shared metadata:
    name: Optional[str] = None,
    result_directory: Optional[Union[str, Path]] = None,
    debug: bool = True,
    execution: Optional[Union[ExecutionConfig, Mapping[str, Any]]] = None,
) -> Experiment:
    """Build an Experiment (single-run or sweep) in a unified way.

    Usage pattern:

        exp = build_experiment(...)
        df  = exp.run()

    Behaviours:

      * If `parameter_space` is None and `replicates == 1`:
          -> SingleExperiment (one configuration, one realisation)

      * If `parameter_space` is None and `replicates > 1`:
          -> SweepExperiment over `replicates` stochastic realisations
             of the same configuration.

      * If `parameter_space` is provided:
          -> SweepExperiment over parameter combinations and replicates.
    """
    library_path = Path(library_path)
    base_config_path = Path(base_config_path)
    sim_cfg = _normalise_sim_cfg(simulation_config)
    exec_cfg = _normalise_execution_config(execution)


    # Initialize experiment-wide logging
    # Initialize logging once for this experiment
    logger = init_experiment_logging(
        result_directory=result_directory or "results",
        name=name or "experiment",
        console=False,   # suppress terminal output
        level=logging.INFO,
    )


    # --- Case 1: genuine single configuration (no parameter_space) ---
    if parameter_space is None:
        scenarios = _build_single_config_scenarios(
            overrides=overrides,
            base_seed=base_seed,
            replicates=replicates,
            explicit_seed=seed,
        )

        if replicates == 1:
            # SingleExperiment: one configuration, one realisation
            return SingleExperiment(
                library_path=library_path,
                base_config_path=base_config_path,
                simulation_config=sim_cfg,
                execution=exec_cfg,
                scenario=scenarios[0],
                name=name,
                result_directory=result_directory,
                debug=debug,
                logger=logger,
            )

        # replicates > 1 -> treat as a small sweep
        return SweepExperiment(
            library_path=library_path,
            base_config_path=base_config_path,
            simulation_config=sim_cfg,
            execution=exec_cfg,
            scenarios=scenarios,
            name=name,
            result_directory=result_directory,
            debug=debug,
            logger=logger,
        )

    # --- Case 2: parameter sweep (possibly with replicates) ---
    scenario_dicts = generate_scenarios(
        library_path=library_path,
        base_config_path=base_config_path,
        parameter_space=parameter_space,
        base_seed=base_seed,
        replicates=replicates,
        max_runs=max_runs,
        zip_groups=zip_groups,
    )

    scenarios = [
        _Scenario(
            scenario_id=str(sd["scenario_id"]),
            label=str(sd.get("label", sd["scenario_id"])),
            overrides=dict(sd.get("overrides", {})),
            seed=int(sd.get("seed", 0)),
        )
        for sd in scenario_dicts
    ]

    # Save scenarios JSON BEFORE running the simulations
    if result_directory is not None:
        save_root = Path(result_directory)
    else:
        save_root = Path("results")

    save_sceanarios(
        scenarios=[
            dict(
                scenario_id=s.scenario_id,
                label=s.label,
                overrides=s.overrides,
                seed=s.seed,
            )
            for s in scenarios
        ],
        result_directory=save_root,
        name=name,
    )

    return SweepExperiment(
        library_path=library_path,
        base_config_path=base_config_path,
        simulation_config=sim_cfg,
        execution=exec_cfg,
        scenarios=scenarios,
        name=name,
        result_directory=result_directory,
        debug=debug,
        logger=logger,
    )
