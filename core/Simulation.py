from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
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
# Solve helpers (1D root finding)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable, Tuple


def _bisect_root(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-3,
    max_iter: int = 80,
) -> Tuple[float, int, float]:
    """
    Bisection root-finder for f(x)=0. Requires bracketing: f(lo)*f(hi) <= 0.
    Returns (x*, iters, f(x*)).
    """
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo, 0, flo
    if fhi == 0.0:
        return hi, 0, fhi
    if flo * fhi > 0:
        raise ValueError(f"Root not bracketed: f({lo})={flo}, f({hi})={fhi}")

    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        if abs(fmid) <= tol or abs(hi - lo) <= tol:
            return mid, i, fmid

        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    mid = 0.5 * (lo + hi)
    fmid = f(mid)
    return mid, max_iter, fmid


def _bisect_min_feasible(
    feasible_at: Callable[[float], bool],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-3,
    max_iter: int = 80,
) -> Tuple[float, int]:
    """
    Returns smallest x in [lo,hi] s.t. feasible_at(x) is True,
    assuming feasible_at(lo)=False, feasible_at(hi)=True.
    """
    if feasible_at(lo):
        return lo, 0
    if not feasible_at(hi):
        raise ValueError("Upper bound is not feasible; cannot bisect minimal feasible.")

    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        if feasible_at(mid):
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) <= tol:
            return hi, i

    return hi, max_iter

def _auto_bracket(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    expand: float = 2.0,
    max_expand: int = 20,
) -> Tuple[float, float]:
    """
    Expand [lo,hi] symmetrically until it brackets a root (or fail).
    """
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0 or fhi == 0.0 or flo * fhi < 0:
        return lo, hi

    width = hi - lo
    for _ in range(max_expand):
        width *= expand
        lo2 = lo - 0.5 * width
        hi2 = hi + 0.5 * width
        flo2 = f(lo2)
        fhi2 = f(hi2)
        if flo2 == 0.0 or fhi2 == 0.0 or flo2 * fhi2 < 0:
            return lo2, hi2

    raise ValueError("Failed to bracket root after expansions.")


def _auto_bracket_feasible(
    feasible_at: Callable[[float], bool],
    lo: float,
    hi: float,
    *,
    expand: float = 2.0,
    max_expand: int = 20,
) -> Tuple[float, float]:
    """
    Expand hi upward until feasible (keeping lo fixed as infeasible ideally).
    If lo is feasible already, returns immediately.
    """
    if feasible_at(lo):
        return lo, lo

    cur_lo, cur_hi = lo, hi
    if feasible_at(cur_hi):
        return cur_lo, cur_hi

    width = cur_hi - cur_lo
    for _ in range(max_expand):
        width *= expand
        cur_hi = cur_lo + width
        if feasible_at(cur_hi):
            return cur_lo, cur_hi

    raise ValueError("Failed to find a feasible upper bound after expansions.")


def _evaluate_at_strike(env, K: float, iteration_cfg: SimulationConfig, solve_config: SolveConfig) -> Dict[str, float]:
    # set decision variable
    env.RevenueModel.strike_price = float(K)

    # run iteration pass (should NOT collect results)
    env.simulation_config = iteration_cfg
    env.run_simulation()

    # read metrics
    valuation = env.valuation
    npv_val = float(getattr(valuation, solve_config.npv_attr))
    dscr_min = getattr(valuation, "dscr_min", None)
    dscr_min = float(dscr_min) if dscr_min is not None else float("nan")

    return {
        "npv": npv_val,
        "dscr_min": dscr_min,
    }

def _is_feasible(metrics: Mapping[str, float], solve_config: SolveConfig) -> bool:
    # NPV constraint (>= target)
    if metrics["npv"] < float(solve_config.target_value):
        return False

    # DSCR constraint if enabled
    if solve_config.dscr_min_floor is not None:
        dscr = metrics.get("dscr_min", float("nan"))
        if not np.isfinite(dscr) or dscr < float(solve_config.dscr_min_floor):
            return False

    return True





@dataclass(frozen=True)
class SolveConfig:
    """Configuration for the 1-D strike-price solver.

    WINPACT's solver finds the strike price :math:`K` for which a target
    project-valuation metric (by default equity NPV) is satisfied, either
    as a root-finding problem or as a minimum-feasible search under
    constraints.

    Parameters
    ----------
    target_value : float, default 0.0
        Target value for the NPV metric. In ``"root"`` mode the solver
        finds :math:`K` such that ``metric(K) == target_value``; in
        ``"min_feasible"`` mode it finds the smallest :math:`K` such that
        ``metric(K) >= target_value``.
    npv_attr : str, default ``"npv_equity"``
        Name of the attribute to read from ``env.valuation`` as the NPV
        metric (e.g. ``"npv_equity"``, ``"npv_project"``).
    bracket_lo, bracket_hi : float
        Initial bracket for the bisection. Units match ``strike_price``
        (currency per MWh).
    auto_bracket : bool, default True
        If True, the bracket is expanded automatically when it does not
        yet contain a root or a feasible point.
    tol : float, default 1e-3
        Convergence tolerance on residual (root mode) or bracket width
        (min-feasible mode).
    max_iter : int, default 80
        Maximum number of bisection iterations.
    dscr_min_floor : float, optional
        Minimum debt-service-coverage ratio constraint. If set, a
        candidate :math:`K` is feasible only when ``dscr_min >= floor``
        in addition to the NPV constraint. ``None`` disables the
        constraint.
    mode : {"min_feasible", "root"}, default ``"min_feasible"``
        Solver mode. ``"root"`` solves ``NPV(K) = target`` and ignores
        ``dscr_min_floor``. ``"min_feasible"`` returns the smallest
        :math:`K` that satisfies all constraints.

    See Also
    --------
    build_experiment : pass an instance via the ``solve_config`` kwarg.
    """

    target_value: float = 0.0
    npv_attr: str = "npv_equity"

    # strike bracket
    bracket_lo: float = -200.0
    bracket_hi: float = 400.0
    auto_bracket: bool = True

    # tolerances
    tol: float = 1e-3
    max_iter: int = 80

    dscr_min_floor: Optional[float] = None

    mode: str = "min_feasible"



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
    LCA_inputFiles: dict[str, str]
    Opex_inputFiles: dict[str, str]
    MetEnv_inputFiles: dict[str, str]
    Material_inputFiles: dict[str, str]
    WindFarm_inputFiles: dict[str, str]
    WindTurbine_inputFiles: dict[str, str]
    Valuation_inputFiles: dict[str, str]
    Market_inputFiles: dict[str, str]
    LTE_inputFiles: dict[str, str]
    Curtailment_inputFiles: dict[str, str]
    Project_Duration: dict[str, Union[int, str]]
    Project_StartDate: str
    WF_OperationsStart: dict[str, Union[int, str]]
    WF_OperationsEnd_h: int
    WF_OperationsEnd: dict[str, Union[int, str]]
    WF_OperationsStart_h: int
    TimeStep: int
    Project_Duration_h: int

    # Experiment / scenario metadata (optional when using Simulation.from_config directly)
    experiment_name: str = ""
    result_directory: str = ""
    scenario_label: str = ""
    scenario_id: str = ""
    seed: int = 0

    # Optional overrides the code understands:
    WindFarm_overrides: dict[str, Any] = field(factory=dict)
    Revenue_overrides: dict[str, Any] = field(factory=dict)
    CAPEX_overrides: dict[str, Any] = field(factory=dict)
    LCA_overrides: dict[str, Any] = field(factory=dict)
    OPEX_overrides: dict[str, Any] = field(factory=dict)
    FINEX_overrides: dict[str, Any] = field(factory=dict)
    LTE_overrides: dict[str, Any] = field(factory=dict)
    Curtailment_overrides: dict[str, Any] = field(factory=dict)


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
    """Execution settings for running experiments locally or on HPC.

    The same ``ExecutionConfig`` is honoured by both sweep experiments
    and solve-sweep experiments. The default (sequential, one job) is
    safe everywhere: a local laptop, a shared workstation, or an HPC
    compute node. Parallelism is opt-in.

    Parameters
    ----------
    backend : {"sequential", "process", "thread"}, default ``"sequential"``
        Execution backend.

        - ``"sequential"`` : run scenarios one after another in the
          current process. Recommended for debugging and for very cheap
          scenarios where process-pool overhead dominates.
        - ``"process"`` : use ``concurrent.futures.ProcessPoolExecutor``.
          Best for CPU-bound WINPACT runs since each worker gets its
          own Python interpreter (bypasses the GIL). This is the
          typical HPC mode — set ``n_jobs`` to the number of cores
          allocated by SLURM/PBS.
        - ``"thread"`` : use ``ThreadPoolExecutor``. Useful only when
          the bottleneck is I/O (e.g. large file reads) or a native
          library that releases the GIL.
    n_jobs : int, default 1
        Number of worker processes/threads. Values ``<= 1`` collapse to
        sequential execution regardless of ``backend``. On HPC, match
        this to the number of cores requested from the scheduler.
    chunksize : int, default 1
        Reserved for future use (batching of scenarios per worker);
        currently accepted but not consumed by the executors.

    Notes
    -----
    ``debug=True`` on an experiment is incompatible with parallel
    backends — parallel execution swallows tracebacks across process
    boundaries, so the experiment layer raises a ``ValueError`` instead
    of silently degrading.

    Examples
    --------
    Local quick iteration::

        execution = ExecutionConfig()  # sequential

    HPC node with 32 cores::

        execution = ExecutionConfig(backend="process", n_jobs=32)

    Passed via :func:`build_experiment`::

        exp = build_experiment(..., execution=execution)
    """
    backend: str = "sequential"
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


def _solve_single_scenario(
    *,
    library_path: Path,
    base_cfg: Mapping[str, Any],
    scenario: _Scenario,
    freeze_simulation_config: SimulationConfig,
    iteration_simulation_config: SimulationConfig,
    final_simulation_config: Optional[SimulationConfig] = None,
    name: Optional[str],
    result_directory: Optional[Union[str, Path]],
    debug: bool,
    logger_name: str,
    solve_config: SolveConfig,
) -> Dict[str, Any]:
    """
    Run one scenario, but instead of a single evaluation run:
      1) build env
      2) freeze-run once (expensive modules)
      3) iterate strike price using iteration_simulation_config (no persistence)
      4) final pass using final_simulation_config (persist results once)
    """
    logger = logging.getLogger(logger_name)

    t0 = time.time()
    status: str = "success"
    err: Optional[str] = None
    tb_txt: Optional[str] = None
    duration_s: Optional[float] = None

    cfg_overridden = _apply_overrides(dict(base_cfg), scenario.overrides)
    cfg_full = _build_configuration_dict(
        cfg_overridden,
        scenario,
        name=name,
        result_directory=result_directory,
    )

    logger.info(
        "[scenario:solve:start] id=%s label=%s seed=%s target=%s npv_attr=%s",
        scenario.scenario_id, scenario.label, scenario.seed,
        solve_config.target_value, solve_config.npv_attr,
    )

    sim = None
    try:
        # Build Simulation with FREEZE config
        sim = Simulation.from_config(
            library_path, cfg_full,
            simulation_config=freeze_simulation_config,
            logger=logger,
        )

        # Seed injection, if supported
        if hasattr(sim.env, "seed"):
            sim.env.seed = scenario.seed

        # --- 1) Freeze pass (heavy modules once) ---
        sim.run()
        env = sim.env

        # Basic sanity checks
        revenue_model = getattr(env, "RevenueModel", None)
        valuation = getattr(env, "valuation", None)
        if revenue_model is None or valuation is None:
            raise RuntimeError("Environment missing RevenueModel or valuation module.")
        # --- 2) Define evaluation + feasibility ---
        def metrics_at(K: float) -> Dict[str, float]:
            return _evaluate_at_strike(env, K, iteration_simulation_config, solve_config)

        def feasible_at(K: float) -> bool:
            m = metrics_at(K)
            return _is_feasible(m, solve_config)

        # --- 3) Solve ---
        if solve_config.mode == "root":
            # old behavior: solve NPV(K) = target (ignores DSCR constraint)
            def f(K: float) -> float:
                m = metrics_at(K)
                return float(m["npv"]) - float(solve_config.target_value)

            lo, hi = solve_config.bracket_lo, solve_config.bracket_hi
            if solve_config.auto_bracket:
                lo, hi = _auto_bracket(f, lo, hi)
            k_star, iters, residual = _bisect_root(f, lo, hi, tol=solve_config.tol, max_iter=solve_config.max_iter)

            final_metrics = metrics_at(k_star)

        elif solve_config.mode == "min_feasible":
            lo, hi = solve_config.bracket_lo, solve_config.bracket_hi

            if solve_config.auto_bracket:
                lo, hi = _auto_bracket_feasible(feasible_at, lo, hi)
            else:
                # require lo infeasible and hi feasible
                if feasible_at(lo) and not feasible_at(hi):
                    raise ValueError("Bad feasibility bracket: lo feasible but hi infeasible.")
                if not feasible_at(hi):
                    raise ValueError("Upper bound must be feasible in min_feasible mode.")

            k_star, iters = _bisect_min_feasible(
                feasible_at, lo, hi, tol=solve_config.tol, max_iter=solve_config.max_iter
            )
            final_metrics = metrics_at(k_star)

            residual = float(final_metrics["npv"]) - float(solve_config.target_value)

        else:
            raise ValueError(f"Unknown solve_config.mode: {solve_config.mode!r}")

        # --- 4) Final pass (persist once) ---
        final_cfg = final_simulation_config or iteration_simulation_config
        env.RevenueModel.strike_price = float(k_star)
        env.simulation_config = final_cfg
        env.run_simulation()


        logger.info(
            "[scenario:solve:done] id=%s k_star=%.6f iters=%d residual=%.6f duration_s=%.3f",
            scenario.scenario_id, k_star, iters, residual, time.time() - t0,
        )

        duration_s = time.time() - t0

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

            "optimal_strike_price": float(k_star),
            "npv_attr": solve_config.npv_attr,
            "target_value": float(solve_config.target_value),
            "final_residual": float(residual),
            "iters": int(iters),
            "bracket_lo_used": float(lo),
            "bracket_hi_used": float(hi),

            # NEW: constraint metric
            "dscr_min": float(final_metrics.get("dscr_min", float("nan"))),
            "dscr_min_floor": None if solve_config.dscr_min_floor is None else float(solve_config.dscr_min_floor),
            "solve_mode": str(solve_config.mode),
        }

    except Exception as e:
        status = "fail"
        err = str(e)
        tb_txt = traceback.format_exc()
        logger.exception(
            "[scenario:solve:fail] id=%s label=%s seed=%s",
            scenario.scenario_id, scenario.label, scenario.seed,
        )
        if debug:
            raise
        duration_s = time.time() - t0
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

    finally:
        try:
            del sim
        except Exception:
            pass
        gc.collect()

def _solve_scenarios(
    *,
    scenarios,
    library_path,
    base_cfg,
    freeze_simulation_config: SimulationConfig,
    iteration_simulation_config: SimulationConfig,
    final_simulation_config: Optional[SimulationConfig] = None,
    name,
    result_directory,
    debug,
    logger_name,
    execution: ExecutionConfig,
    solve_config: SolveConfig,
    on_result=None,
):
    """
    Runs strike-price solving per scenario.

    Execution:
      - Freeze pass: freeze_simulation_config (heavy modules)
      - Iteration pass: iteration_simulation_config (typically revenue+valuation, collect_results=False)
      - Final pass: final_simulation_config (typically revenue+valuation, collect_results=True)
    """
    if execution.backend == "sequential" or execution.n_jobs <= 1:
        rows = []
        for sc in scenarios:
            row = _solve_single_scenario(
                library_path=library_path,
                base_cfg=base_cfg,
                scenario=sc,
                freeze_simulation_config=freeze_simulation_config,
                iteration_simulation_config=iteration_simulation_config,
                final_simulation_config=final_simulation_config,
                name=name,
                result_directory=result_directory,
                debug=debug,
                logger_name=logger_name,
                solve_config=solve_config,
            )
            rows.append(row)
            if on_result:
                on_result(row)
        return rows

    if debug:
        raise ValueError("debug=True is not compatible with parallel execution. Set debug=False.")

    if execution.backend not in ("process", "thread"):
        raise ValueError(f"Unknown execution backend: {execution.backend!r}")

    Executor = ProcessPoolExecutor if execution.backend == "process" else ThreadPoolExecutor

    rows = []
    with Executor(max_workers=execution.n_jobs) as ex:
        futures = [
            ex.submit(
                _solve_single_scenario,
                library_path=library_path,
                base_cfg=base_cfg,
                scenario=sc,
                freeze_simulation_config=freeze_simulation_config,
                iteration_simulation_config=iteration_simulation_config,
                final_simulation_config=final_simulation_config,
                name=name,
                result_directory=result_directory,
                debug=False,
                logger_name=logger_name,
                solve_config=solve_config,
            )
            for sc in scenarios
        ]

        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            if on_result:
                on_result(row)

    # restore original scenario order
    order = {s.scenario_id: i for i, s in enumerate(scenarios)}
    rows.sort(key=lambda r: order.get(r.get("scenario_id"), 10**9))
    return rows



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
    """A single-configuration WINPACT experiment (one scenario, one run).

    This is the simplest ``Experiment`` type: one base YAML config, an
    optional set of dotted-path overrides, and a single seed. It is
    produced by :func:`build_experiment` when neither ``parameter_space``
    nor ``replicates > 1`` are requested.

    Users normally should not instantiate this directly — go through
    :func:`build_experiment`, which handles config normalisation,
    scenario construction, logging setup, and dispatch to the right
    ``Experiment`` subclass.

    Parameters
    ----------
    library_path : pathlib.Path
        Root directory for input files. All relative paths in the
        YAML config are resolved against this.
    base_config_path : pathlib.Path
        Path to the base YAML configuration (relative to
        ``library_path``).
    simulation_config : SimulationConfig
        Which WINPACT modules to activate for this run.
    scenario : _Scenario
        The (singleton) scenario to execute, carrying overrides, seed,
        label, and deterministic ``scenario_id``.
    name : str, optional
        Experiment name recorded in the output row and used for log
        grouping.
    result_directory : str or pathlib.Path, optional
        Destination directory for generated artefacts and the run index.
    debug : bool, default True
        If True, exceptions propagate (useful in notebooks). If False,
        the failure is recorded on the status row and execution
        continues.
    logger : logging.Logger, optional
        Logger to use; defaults to ``"winpact.single"``.
    execution : ExecutionConfig, optional
        Parallelism settings. Ignored for a single scenario, but kept
        in the signature for symmetry with :class:`SweepExperiment`.

    Returns
    -------
    pandas.DataFrame
        ``run()`` returns a one-row DataFrame with columns
        ``scenario_id``, ``label``, ``seed``, ``status``, ``duration_s``,
        ``error_message``, ``traceback``, ``experiment_name``,
        ``result_directory``.

    See Also
    --------
    SweepExperiment : multi-scenario counterpart.
    SolveSingleExperiment : single-scenario strike-price solver.
    build_experiment : the recommended construction path.
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
    """A multi-scenario WINPACT experiment (parameter sweep + replicates).

    ``SweepExperiment`` evaluates a list of ``_Scenario`` objects —
    typically produced by :func:`generate_scenarios` from a
    ``parameter_space`` — and returns one status row per scenario. It
    honours ``execution`` for sequential, process-, or thread-based
    parallelism, which is the standard HPC path.

    Parameters
    ----------
    library_path : pathlib.Path
        Input-file root used to resolve relative paths in the YAML.
    base_config_path : pathlib.Path
        Path to the base YAML (relative to ``library_path``).
    simulation_config : SimulationConfig
        Modules to activate for every scenario.
    scenarios : list of _Scenario
        One entry per evaluation; includes deterministic ``scenario_id``,
        human-readable ``label``, dotted-path ``overrides``, and ``seed``.
    name : str, optional
        Experiment name recorded on every row.
    result_directory : str or pathlib.Path, optional
        Destination directory. If given, an ``index.parquet`` (or
        ``index.csv`` fallback) is appended with one row per scenario
        after the sweep completes.
    debug : bool, default True
        If True, an exception in any scenario aborts the sweep — useful
        when developing. Must be False for parallel backends.
    on_result : callable, optional
        Callback ``on_result(row: dict)`` fired as each scenario
        completes, in whatever order the executor returns them.
    logger : logging.Logger, optional
        Logger (default ``"winpact.sweep"``).
    execution : ExecutionConfig, optional
        Parallelism settings — see :class:`ExecutionConfig`.

    Returns
    -------
    pandas.DataFrame
        One row per scenario, ordered to match the input ``scenarios``
        list even when executed in parallel.

    Notes
    -----
    Results are also persisted incrementally to ``result_directory`` for
    bookkeeping, so a very long sweep can be safely inspected mid-run.

    See Also
    --------
    generate_scenarios : build the ``scenarios`` list.
    SingleExperiment : single-scenario counterpart.
    SolveSweepExperiment : sweep variant that solves a strike price per scenario.
    """

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
    
@define(auto_attribs=True)
class SolveSweepExperiment(Experiment):
    """Multi-scenario experiment where each scenario runs a strike-price solver.

    For every scenario the environment is built once, the expensive
    modules are executed in a single "freeze" pass, and then the
    revenue/valuation modules are iterated to find the strike price
    :math:`K` satisfying :class:`SolveConfig`. A final pass persists
    results at the solved :math:`K`.

    The freeze/iteration split is the key performance optimisation:
    windfarm production, metocean, CAPEX/OPEX etc. are independent of
    the strike price and only need to run once per scenario.

    Parameters
    ----------
    library_path : pathlib.Path
        Input-file root.
    base_config_path : pathlib.Path
        Base YAML configuration path.
    freeze_simulation_config : SimulationConfig
        Modules to execute in the one-time freeze pass (typically
        everything except the solver-dependent pieces).
    iteration_simulation_config : SimulationConfig
        Modules executed on every bisection iteration. Usually just
        ``run_revenue=True`` and ``run_valuation=True`` with
        ``collect_results=False`` to avoid I/O per iteration.
    scenarios : list of _Scenario
        Scenarios to solve.
    final_simulation_config : SimulationConfig, optional
        Modules executed once after convergence, typically the same as
        ``iteration_simulation_config`` but with ``collect_results=True``
        so the solved run is the one persisted to disk.
    solve_config : SolveConfig, optional
        Solver settings (target, bracket, tolerance, constraints, mode).
    name, result_directory, debug, on_result, logger, execution
        Same semantics as :class:`SweepExperiment`.

    Returns
    -------
    pandas.DataFrame
        One row per scenario, augmented with solver diagnostics:
        ``optimal_strike_price``, ``npv_attr``, ``target_value``,
        ``final_residual``, ``iters``, ``bracket_lo_used``,
        ``bracket_hi_used``, ``dscr_min``, ``dscr_min_floor``,
        ``solve_mode``.

    See Also
    --------
    SolveSingleExperiment : single-scenario counterpart.
    SolveConfig : solver parameters.
    SweepExperiment : non-solver sweep.
    """
    library_path: Path
    base_config_path: Path

    freeze_simulation_config: SimulationConfig
    iteration_simulation_config: SimulationConfig
    scenarios: List[_Scenario]                 # <-- move up (required)

    # defaults after required:
    final_simulation_config: Optional[SimulationConfig] = None
    solve_config: SolveConfig = field(factory=SolveConfig)

    name: Optional[str] = None
    result_directory: Optional[Union[str, Path]] = None
    debug: bool = True
    on_result: Optional[Callable[[Mapping[str, Any]], None]] = None
    logger: logging.Logger = field(factory=lambda: logging.getLogger("winpact.solve_sweep"))
    execution: ExecutionConfig = field(factory=ExecutionConfig)


    def run(self) -> pd.DataFrame:
        base_cfg = _load_base_config_yaml(self.library_path, self.base_config_path)

        rows = _solve_scenarios(
            scenarios=self.scenarios,
            library_path=self.library_path,
            base_cfg=base_cfg,
            freeze_simulation_config=self.freeze_simulation_config,
            iteration_simulation_config=self.iteration_simulation_config,
            final_simulation_config=self.final_simulation_config,
            name=self.name,
            result_directory=self.result_directory,
            debug=self.debug,
            logger_name=self.logger.name,
            execution=self.execution,
            solve_config=self.solve_config,
            on_result=self.on_result,
        )
        df = pd.DataFrame(rows)

        # optional: persist an index of runs for bookkeeping (same as SweepExperiment)
        if self.result_directory is not None:
            _append_index(self.result_directory, rows)

        return df


@define(auto_attribs=True)
class SolveSingleExperiment(Experiment):
    """Single-scenario strike-price solver experiment.

    Counterpart of :class:`SolveSweepExperiment` for a single scenario:
    one freeze pass, a bisection on strike price, and a final persist
    pass. See :class:`SolveSweepExperiment` for the full parameter and
    return-value documentation — this class accepts the same arguments
    with ``scenario`` (singular) in place of ``scenarios``.

    Returns
    -------
    pandas.DataFrame
        A one-row DataFrame with the standard status columns plus the
        solver diagnostic columns.

    See Also
    --------
    SolveSweepExperiment, SolveConfig, build_experiment
    """

    library_path: Path
    base_config_path: Path

    freeze_simulation_config: SimulationConfig
    iteration_simulation_config: SimulationConfig
    scenario: _Scenario                     # <-- REQUIRED must be before defaults

    # defaults after required:
    final_simulation_config: Optional[SimulationConfig] = None
    solve_config: SolveConfig = field(factory=SolveConfig)

    name: Optional[str] = None
    result_directory: Optional[Union[str, Path]] = None
    debug: bool = True
    logger: logging.Logger = field(factory=lambda: logging.getLogger("winpact.solve_single"))
    execution: ExecutionConfig = field(factory=ExecutionConfig)

    def run(self) -> pd.DataFrame:
        base_cfg = _load_base_config_yaml(self.library_path, self.base_config_path)
        row = _solve_single_scenario(
            library_path=self.library_path,
            base_cfg=base_cfg,
            scenario=self.scenario,
            freeze_simulation_config=self.freeze_simulation_config,
            iteration_simulation_config=self.iteration_simulation_config,
            final_simulation_config=self.final_simulation_config,
            name=self.name,
            result_directory=self.result_directory,
            debug=self.debug,
            logger_name=self.logger.name,
            solve_config=self.solve_config,
        )
        return pd.DataFrame([row])




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
    """Design a set of scenarios from a parameter space.

    This is WINPACT's **scenario designer**: it turns a mapping of
    dotted-path parameters to value lists into a deterministic list of
    scenario descriptors ready to be consumed by
    :class:`SweepExperiment` or :class:`SolveSweepExperiment`.

    By default, every key in ``parameter_space`` is treated as an
    independent axis and combined with all others via a Cartesian
    product. ``zip_groups`` lets you override this per-group: keys in
    the same group are paired *positionally* rather than crossed, which
    is how you express correlated sweeps (e.g. changing a turbine
    rating *and* its rotor diameter together).

    Parameters
    ----------
    library_path : str or pathlib.Path
        Input-file root. Used only to validate ``base_config_path``
        early so that typos fail fast.
    base_config_path : str or pathlib.Path
        YAML config whose existence is checked up front.
    parameter_space : mapping of str to sequence
        Keys are dotted paths into the configuration
        (e.g. ``"CAPEX_overrides.material_price.copper"``); values are
        the sequences of overrides to sweep over.
    base_seed : int, default 0
        Seed used to derive deterministic per-scenario seeds.
    replicates : int, default 1
        Number of seeded replicates to emit per parameter combination.
    max_runs : int, optional
        Hard cap on the number of scenarios returned. Useful for smoke
        tests of very large sweeps.
    zip_groups : mapping of str to sequence of str, optional
        Named groups of keys to pair positionally. Within a group, each
        value list must have length 1 (broadcast) or the group's
        maximum length. Groups are Cartesian with each other and with
        the ungrouped keys.

    Returns
    -------
    list of dict
        One dict per scenario with keys ``scenario_id`` (deterministic
        10-char sha1 of overrides + seed), ``label`` (human-readable
        summary), ``overrides`` (the dotted-path mapping applied to the
        base config), and ``seed``.

    Raises
    ------
    KeyError
        If a key in ``zip_groups`` is not in ``parameter_space``.
    ValueError
        If a key appears in more than one zip group, or if list lengths
        inside a group are incompatible.

    Examples
    --------
    Independent axes (full Cartesian)::

        scenarios = generate_scenarios(
            library_path="examples",
            base_config_path="Inputs/HKN/Config.yaml",
            parameter_space={
                "CAPEX_overrides.material_price.copper": [7000, 9000],
                "Revenue_overrides.strike_price": [50, 70, 90],
            },
            replicates=3,
        )
        # 2 × 3 × 3 replicates = 18 scenarios

    Correlated turbine sweep via ``zip_groups``::

        scenarios = generate_scenarios(
            ...,
            parameter_space={
                "WindTurbine_inputFiles.rating_MW": [10, 15, 20],
                "WindTurbine_inputFiles.rotor_D":    [180, 220, 260],
                "Revenue_overrides.strike_price":   [50, 70],
            },
            zip_groups={"turbine": ["WindTurbine_inputFiles.rating_MW",
                                    "WindTurbine_inputFiles.rotor_D"]},
        )
        # 3 turbines (zipped) × 2 strike prices = 6 scenarios
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

    # SOLVE MODE:
    solve_config: Optional[SolveConfig] = None,
    freeze_simulation_config: Optional[Union[SimulationConfig, Mapping[str, Any]]] = None,
    iteration_simulation_config: Optional[Union[SimulationConfig, Mapping[str, Any]]] = None,
    final_simulation_config: Optional[Union[SimulationConfig, Mapping[str, Any]]] = None,
) -> Experiment:
    """Unified entry point for constructing a WINPACT experiment.

    ``build_experiment`` is the recommended way to create any of the
    four ``Experiment`` variants. It inspects the arguments and returns
    the right subclass:

    .. list-table::
       :header-rows: 1
       :widths: 30 30 40

       * - ``parameter_space``
         - ``solve_config``
         - returned type
       * - None, ``replicates == 1``
         - None
         - :class:`SingleExperiment`
       * - None, ``replicates > 1``
         - None
         - :class:`SweepExperiment`
       * - given
         - None
         - :class:`SweepExperiment`
       * - None, ``replicates == 1``
         - given
         - :class:`SolveSingleExperiment`
       * - any (multi-scenario)
         - given
         - :class:`SolveSweepExperiment`

    Parameters
    ----------
    library_path : str or pathlib.Path
        Root directory for input files.
    base_config_path : str or pathlib.Path
        Base YAML configuration, relative to ``library_path``.
    simulation_config : SimulationConfig or mapping
        Which modules to run. Mappings are converted via
        ``SimulationConfig.from_dict``. In solve mode this acts as the
        default for ``freeze_simulation_config``.

    Other Parameters
    ----------------
    overrides : mapping, optional
        Dotted-path overrides for a single scenario (ignored when
        ``parameter_space`` is given).
    seed : int, optional
        Explicit seed for the single-scenario case; otherwise
        ``base_seed`` is used.
    parameter_space : mapping, optional
        Passed through to :func:`generate_scenarios` to build a sweep.
    base_seed : int, default 0
        Seed used to derive scenario seeds in sweep mode.
    replicates : int, default 1
        Number of replicates per parameter combination.
    max_runs : int, optional
        Cap on the number of scenarios produced.
    zip_groups : mapping, optional
        Positional-pairing groups, see :func:`generate_scenarios`.
    name : str, optional
        Experiment name (used in logs and output rows).
    result_directory : str or pathlib.Path, optional
        Destination directory for artefacts and the run index.
    debug : bool, default True
        If True, scenario exceptions propagate (useful interactively).
        Must be False when using parallel ``execution`` backends.
    execution : ExecutionConfig or mapping, optional
        Parallelism configuration. ``None`` is sequential. Mappings
        with keys ``backend``, ``n_jobs``, ``chunksize`` are accepted.
        See :class:`ExecutionConfig` for HPC guidance.
    solve_config : SolveConfig, optional
        If provided, the returned experiment solves for the strike
        price that satisfies the target NPV (and optional DSCR) rather
        than evaluating a fixed configuration.
    freeze_simulation_config : SimulationConfig or mapping, optional
        Modules to run once per scenario before iterating. Defaults to
        ``simulation_config``.
    iteration_simulation_config : SimulationConfig or mapping, optional
        Modules to rerun on every bisection step. Default is
        revenue + valuation with ``collect_results=False``.
    final_simulation_config : SimulationConfig or mapping, optional
        Modules executed once at the solved strike price with
        ``collect_results=True`` so artefacts are persisted exactly
        once. Defaults to the iteration config with results collection
        re-enabled.

    Returns
    -------
    Experiment
        A concrete :class:`Experiment` subclass with a ``.run()``
        method that returns a status DataFrame.

    Examples
    --------
    Single run::

        exp = build_experiment(
            library_path="examples",
            base_config_path="Inputs/HKN/Config.yaml",
            simulation_config={"run_valuation": True, "collect_results": True},
            overrides={"Revenue_overrides.strike_price": 65.0},
            name="baseline",
            result_directory="results/baseline",
        )
        df = exp.run()

    Parameter sweep on HPC::

        exp = build_experiment(
            library_path="examples",
            base_config_path="Inputs/HKN/Config.yaml",
            simulation_config=sim_cfg,
            parameter_space={
                "Revenue_overrides.strike_price": [50, 60, 70, 80, 90],
            },
            replicates=5,
            name="strike_sweep",
            result_directory="results/strike_sweep",
            debug=False,
            execution={"backend": "process", "n_jobs": 32},
        )
        df = exp.run()

    Solving the strike price that zeros equity NPV under a DSCR floor::

        exp = build_experiment(
            library_path="examples",
            base_config_path="Inputs/HKN/Config.yaml",
            simulation_config=sim_cfg,
            parameter_space={"CAPEX_overrides.material_price.copper": [7000, 9000]},
            solve_config=SolveConfig(
                target_value=0.0,
                npv_attr="npv_equity",
                mode="min_feasible",
                dscr_min_floor=1.2,
            ),
            name="breakeven_copper",
            result_directory="results/breakeven_copper",
            execution={"backend": "process", "n_jobs": 16},
        )
        df = exp.run()

    See Also
    --------
    generate_scenarios : expose the sweep logic directly.
    SolveConfig, ExecutionConfig : configuration dataclasses.
    SingleExperiment, SweepExperiment, SolveSingleExperiment, SolveSweepExperiment
    """
    library_path = Path(library_path)
    base_config_path = Path(base_config_path)
    sim_cfg = _normalise_sim_cfg(simulation_config)
    exec_cfg = _normalise_execution_config(execution)

    # Initialize logging once for this experiment
    logger = init_experiment_logging(
        result_directory=result_directory or "results",
        name=name or "experiment",
        console=False,
        level=logging.INFO,
    )

    # ---------------------------------------------------------------------
    # Solve-mode config normalization + defaults
    # ---------------------------------------------------------------------
    freeze_cfg: Optional[SimulationConfig] = None
    iter_cfg: Optional[SimulationConfig] = None
    final_cfg: Optional[SimulationConfig] = None

    if solve_config is not None:
        # Freeze cfg default: use sim_cfg unless user supplies an override
        if freeze_simulation_config is None:
            freeze_cfg = sim_cfg
        else:
            freeze_cfg = _normalise_sim_cfg(freeze_simulation_config)

        # Iteration cfg default: revenue+valuation only, do NOT collect
        if iteration_simulation_config is None:
            iter_cfg = SimulationConfig.from_dict({
                "run_marketenv": False,
                "run_metenv": False,
                "run_windfarm": False,
                "run_capex": False,
                "run_opex": False,
                "run_LCA": False,
                "run_lifetime_extension": False,
                "run_curtailment": False,
                "run_revenue": True,
                "run_valuation": True,
                "capex_dashboard": False,
                "opex_dashboard": False,
                "valuation_dashboard": False,
                "collect_results": False,
            })
        else:
            iter_cfg = _normalise_sim_cfg(iteration_simulation_config)

        # Final cfg default: copy iteration but collect_results=True
        if final_simulation_config is None:
            # build from iter_cfg fields (attrs-based class => __dict__ is fine)
            d = dict(iter_cfg.__dict__)
            d["collect_results"] = True
            final_cfg = SimulationConfig.from_dict(d)
        else:
            final_cfg = _normalise_sim_cfg(final_simulation_config)

    # ---------------------------------------------------------------------
    # Case 1: single configuration (no parameter_space)
    # ---------------------------------------------------------------------
    if parameter_space is None:
        scenarios = _build_single_config_scenarios(
            overrides=overrides,
            base_seed=base_seed,
            replicates=replicates,
            explicit_seed=seed,
        )

        # --- SOLVE MODE ---
        if solve_config is not None:
            if freeze_cfg is None or iter_cfg is None:
                raise RuntimeError("Internal error: solve mode configs not initialized.")

            if len(scenarios) == 1:
                return SolveSingleExperiment(
                    library_path=library_path,
                    base_config_path=base_config_path,
                    freeze_simulation_config=freeze_cfg,
                    iteration_simulation_config=iter_cfg,
                    final_simulation_config=final_cfg,
                    execution=exec_cfg,
                    scenario=scenarios[0],
                    solve_config=solve_config,
                    name=name,
                    result_directory=result_directory,
                    debug=debug,
                    logger=logger,
                )
            else:
                return SolveSweepExperiment(
                    library_path=library_path,
                    base_config_path=base_config_path,
                    freeze_simulation_config=freeze_cfg,
                    iteration_simulation_config=iter_cfg,
                    final_simulation_config=final_cfg,
                    execution=exec_cfg,
                    scenarios=scenarios,
                    solve_config=solve_config,
                    name=name,
                    result_directory=result_directory,
                    debug=debug,
                    logger=logger,
                )

        # --- EVAL MODE (existing) ---
        if len(scenarios) == 1:
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

    # ---------------------------------------------------------------------
    # Case 2: parameter sweep (possibly with replicates)
    # ---------------------------------------------------------------------
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
    save_root = Path(result_directory) if result_directory is not None else Path("results")
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

    # --- SOLVE MODE ---
    if solve_config is not None:
        if freeze_cfg is None or iter_cfg is None:
            raise RuntimeError("Internal error: solve mode configs not initialized.")

        return SolveSweepExperiment(
            library_path=library_path,
            base_config_path=base_config_path,
            freeze_simulation_config=freeze_cfg,
            iteration_simulation_config=iter_cfg,
            final_simulation_config=final_cfg,
            execution=exec_cfg,
            scenarios=scenarios,
            solve_config=solve_config,
            name=name,
            result_directory=result_directory,
            debug=debug,
            logger=logger,
        )

    # --- EVAL MODE ---
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
