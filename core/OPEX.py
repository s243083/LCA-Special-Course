from __future__ import annotations

"""
OPEX module – scaffolding for three calculation modes:
  1) 'capex_fraction' – existing behaviour: OpEx as a fraction of CapEx
  2) 'analytic_ctmc' – analytical CTMC expectations (availability, expected counts, OpEx)
  3) 'time_march_ctmc' – event/time-marching CTMC (availability time series + activity log)

This file contains only structure and dummy implementations. Replace TODOs with real logic.

External integration points expected (per existing project):
  - ValueWindEnv (env): provides metocean, calendar, config accessors
  - WindFarm (wf): provides power_records (pd.DataFrame), turbines inventory, farm meta

Keep the public surface compatible with the current code:
  - class OPEX with method calc_OPEX(self) -> dict of artefacts including 'OPEX_records'

Author: MG
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import logging

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import apply_overrides , get_input_parameter
import copy

# Plotly imports for dashboard
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataclasses import asdict, is_dataclass

from scipy.stats import gamma as gamma_dist


# -----------------------------------------------------------------------------
# Data contracts (schemas) – light, minimal versions
# -----------------------------------------------------------------------------

@dataclass
class VesselSpec:
    # Identity / type
    name: str
    capability: str = "CTV"  # class/type label as given in input (e.g., "CTV", "SOV")

    # Ops limits (weather gating)
    Hs_limit_m: float = 1.5                  # from max_waveheight
    wind_limit_ms: Optional[float] = None    # from max_windspeed

    # Transit & logistics
    speed_kn: float = 20.0                   # derived from YAML 'speed' (assumed km/h) -> knots
    base_distance_km: float = 0.0            # from port_distance
    strategy: Literal["scheduled","spot","permanent","campaign","unspecified"] = "unspecified"

    # Costing
    day_rate_eur: float = 0.0                # from equipment_rate
    mobilization_fee_eur: float = 0.0        # from mobilization_cost
    mobilization_days: float = 0.0           # from mobilization_days

    # Optional extras (placeholders for future use)
    crew_capacity: Optional[int] = None
    availability_fraction: float = 1.0


@dataclass
class PMPolicy:
    type: Literal["interval", "reliability", "none"] = "none"
    tau_h: Optional[float] = None  # for 'interval'
    xi_PM: Optional[float] = None  # for 'reliability'

@dataclass
class MaintenanceSpec:
    """
    One maintenance “mode” entry, either a CM failure mode or a PM task.
    - For CM entries, lambda_per_h > 0 and task_type == "CM".
    - For PM entries, lambda_per_h == 0 and task_type == "PM" (use frequency_per_year or tau_h).
    """
    component: str                      # e.g., "electrical_system"
    mode_id: Optional[str]              # e.g., "1", "2" for failures; "PM:0" for maintenance[0]
    task_type: Literal["CM", "PM"]
    # Stochastic/tempo
    lambda_per_h: float                 # CM failure rate per hour; 0 for PM
    lambda_nominal_per_h: Optional[float] = None  # baseline failure rate (before uncertainty/factors)
    shape: Optional[float] = None       # Weibull shape (CM only, if provided)
    scale_years: Optional[float] = None # Weibull scale in years (CM only, if provided)
    frequency_per_year: Optional[float] = None # PM only; if set, tau_h = 8760/freq
    tau_h: Optional[float] = None       # PM interval hours (derived from frequency if provided)
    # Durations & costs
    MTTR_h: float = 0.0
    MTTW_L_h: float = 0.0                 # duration for the task (CM or PM)
    spares_eur: float = 0.0
    labour_rate_eur_h: float = 80.0
    labour_h: float = 0.0               # default to MTTR_h (1-tech equivalent) unless you model crew size
    preferred_vessels: List[str] = field(default_factory=list)
    n_technicians: int = 1              # number of technicians required for the task
    description: Optional[str] = None


@dataclass
class AccessProfile:
    task_type: Literal["CM", "PM"]

    # --- Logistics waiting (MTTL) ---
    # mu_L_per_h: rate of clearing logistics (1 / mean_logistic_wait_h)
    mu_L_per_h: float
    mean_logistic_wait_h: float

    # --- Weather waiting (MTTW) ---
    # mu_A_per_h: rate of getting a suitable weather window (1 / mean_weather_wait_h)
    mu_A_per_h: float
    mean_weather_wait_h: float

    # --- Repair / service (incl. transits) ---
    mu_R_per_h: float
    service_time_h: float

    # Vessel "type" / capability used for this task
    chosen_vessels: List[str] = field(default_factory=list)

@dataclass
class ActivityLogEntry:
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    turbine_id: str
    component: str
    task_type: Literal["CM", "PM"]
    vessels: List[str]
    sailing_h: float
    onsite_h: float
    crew_h: float
    spares_eur: float
    labour_eur: float
    transport_eur: float
    total_cost_eur: float

@dataclass
class AvailabilitySummary:
    component_A: Dict[str, float]  # per component availability (0..1)
    turbine_A: Dict[str, float]    # per turbine availability
    farm_A: float                  # farm availability
    downtime_h: Dict[str, float]   # per component downtime over horizon

@dataclass
class OpExBreakdown:
    fixed_OM_eur: float
    CM_cost_eur: float
    PM_cost_eur: float
    transport_eur: float
    labour_eur: float
    spares_eur: float

# -----------------------------------------------------------------------------
# OPEX Engine – three modes
# -----------------------------------------------------------------------------

class OPEX:
    def __init__(self, env):
        self.config = env.config
        self.env = env  # Access to simulation environment
        self.logger = getattr(env, "logger", logging.getLogger("winpact.opex"))
        self.parameters = load_OMData(env.config)
        self.parameters =  get_input_parameter(self.parameters, 'OM','OM_Process')

        # Mode selection: 'capex_fraction' (default for backward compat), 'analytic_ctmc', 'time_march_ctmc'
        self.mode = get_input_parameter(self.parameters, 'Simulation_Mode')
        self.OM_input_file = get_input_parameter(self.parameters, 'OM_input_file')
        self.OM_vessel_input_file = get_input_parameter(self.parameters, 'OM_Vessel_input_file')

        self.project_start = pd.to_datetime(
            self.env.config.Project_StartDate,
            format="%d.%m.%Y"
        )

        # Apply any overrides from config        
        apply_overrides(self, getattr(self.config, "OPEX_overrides", {}))


        self.OPEX_records = {}


    # --------------------------- Entry point ---------------------------
    def calc_OPEX(
        self,
        window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
        overrides: dict | None = None,
        append: bool = True,
        window_label: str | None = None,
    ) -> None:
        """
        Compute OpEx for a given time window and accumulate results.

        Parameters
        ----------
        window:
            (start_ts, end_ts) timestamps. If None, uses config.WF_OperationsStart_h/End_h as today.
        overrides:
            Nested dict of parameter overrides applied ONLY for this call (e.g. analytic_ctmc.mean_shift).
            Expected to patch self.parameters (YAML parameters dict) not just attributes.
        append:
            If True, accumulate into self.OPEX_records / self.OPEX_records_extras.
            If False, overwrite/reset (useful for first call).
        window_label:
            Optional label stored in extras (e.g., "baseline", "lte_extension", "2029-01").
        """

        # ----------------------------
        # 0) Save state we might override temporarily
        # ----------------------------
        old_start_h = getattr(self.config, "WF_OperationsStart_h", None)
        old_end_h   = getattr(self.config, "WF_OperationsEnd_h", None)

        # parameters dict is what get_input_parameter() reads from
        old_parameters = copy.deepcopy(getattr(self, "parameters", {}))

        # ----------------------------
        # 1) Apply window override via config hours (agreed approach)
        # ----------------------------
        if window is not None:
            w0, w1 = pd.to_datetime(window[0]), pd.to_datetime(window[1])
            if w1 <= w0:
                raise ValueError(f"OPEX window end must be after start. Got {w0=} {w1=}")

            # project_start already computed in __init__ from config.Project_StartDate
            # (keep consistent with existing code)
            start_h = int(round((w0 - self.project_start).total_seconds() / 3600.0))
            end_h   = int(round((w1 - self.project_start).total_seconds() / 3600.0))

            self.config.WF_OperationsStart_h = start_h
            self.config.WF_OperationsEnd_h   = end_h

        # ----------------------------
        # 2) Apply per-call overrides into self.parameters (deep-merge)
        # ----------------------------
        if overrides:
            self._deep_update_dict(self.parameters, overrides)

        # ----------------------------
        # 3) Run the selected mode (unchanged)
        # ----------------------------
        mode = self.mode
        if mode == "capex_fraction":
            opex_df, extras = self._calc_opex_as_fraction_of_capex()
        elif mode == "analytic_ctmc":
            opex_df, extras = self._calc_opex_analytic_ctmc()
        elif mode == "time_march_ctmc":
            opex_df, extras = self._calc_opex_time_march()
        else:
            raise ValueError(f"Unsupported OPEX mode: {mode}")

        # annotate extras with window metadata (useful later)
        extras = extras or {}
        extras["mode"] = mode
        extras["window_label"] = window_label
        extras["window"] = window

        # ----------------------------
        # 4) Accumulate records (OPEX_records) and merge extras
        # ----------------------------
        if not append or not isinstance(getattr(self, "OPEX_records", None), pd.DataFrame) or self.OPEX_records is None or len(self.OPEX_records) == 0:
            # reset/overwrite
            self.OPEX_records = opex_df.copy(deep=True)
            self.OPEX_records_extras = self._init_extras_container()
            self._merge_extras_inplace(self.OPEX_records_extras, extras)
        else:
            # append/accumulate
            self.OPEX_records = self._accumulate_opex_records(self.OPEX_records, opex_df)
            self._merge_extras_inplace(self.OPEX_records_extras, extras)

        self.build_extras_tables()

        # keep the “latest window” artefacts for convenience (optional, preserves old UX)
        self.availability_summary = extras.get("availability_summary")
        self.availability_profile = extras.get("availability_profile")
        self.activity_log = extras.get("activity_log")
        self.OpEx_breakdown = extras.get("OpEx_breakdown")

        # ----------------------------
        # 5) Restore temporary overrides
        # ----------------------------
        self.parameters = old_parameters
        if old_start_h is not None:
            self.config.WF_OperationsStart_h = old_start_h
        if old_end_h is not None:
            self.config.WF_OperationsEnd_h = old_end_h

        return None

    
    # ---------------------------- Plotting Entry Point ----------------------------

    def plot_opex_dashboard(self):
        self.opex_dashboard()
        # optional return of figure object can be added later
        return None
    

    # --------------------------- Shared helpers ---------------------------
    def _load_maintenance_specs(self) -> List[MaintenanceSpec]:
        """
        Parse O&M YAML into a list of MaintenanceSpec where each failure mode (CM)
        and each maintenance entry (PM) becomes one independent item.

        Expected YAML (example):
        Turbine:
        electrical_system:
            name: electrical_system
            maintenance:
            - time: 0
                materials: 0
                service_equipment: CTV
                frequency: 0
            failures:
            1:
                scale: 1.859
                shape: 1
                MTTR: 14
                materials: 1000
                service_equipment: CTV
                description: ...
            2:
                ...

        Assumptions:
        - Weibull scale is in YEARS; shape=k (k=1 exponential). MTBF = scale * Γ(1 + 1/k).
        λ_year = 1/MTBF, λ_per_h = λ_year/8760.
        - PM frequency is events per year; tau_h = 8760 / frequency (if frequency > 0).
        - Labour hours default to MTTR_h at 1 tech equivalent; labour rate from parameters or 80 €/h.
        """
        # Try to load the file the config points to
        try:
            om_data = load_yaml(self.config.valuewind_inputFolder, self.OM_input_file)
        except Exception as e:
            self.logger.error(
                "Could not load OM input file '%s': %s",
                self.OM_input_file,
                e,
            )
            return []

        turbine_block = (om_data or {}).get("Turbine", {})
        if not isinstance(turbine_block, dict) or not turbine_block:
            self.logger.warning("OM input has no 'Turbine' block or it is empty.")
            return []

        # Labour rate from parameters (fallback 80 €/h)
        try:
            labour_rate = float(get_input_parameter(self.parameters, "OM_Costs", "labour_rate_eur_h"))
        except Exception:
            labour_rate = 140.0

        specs: List[MaintenanceSpec] = []

        for comp_key, comp_data in turbine_block.items():
            if not isinstance(comp_data, dict):
                continue
            comp_name = comp_data.get("name", comp_key)

            # ---------- Failures -> CM entries ----------
            failures = comp_data.get("failures", {}) or {}
            for mode_id, f in failures.items():
                if not isinstance(f, dict):
                    continue

                shape = float(f.get("shape", 1.0))
                scale_years = float(f.get("scale", 1.0))

                # --- MTTR lookup with warning logic ---
                if "MTTR" in f:
                    mttr_h = float(f["MTTR"])
                elif "time" in f:
                    mttr_h = float(f["time"])
                else:
                    mttr_h = 8.0
                    self.logger.warning(
                        "Component '%s', failure mode '%s': "
                        "No 'MTTR' or 'time' field provided. Defaulting to MTTR=8.0 h.",
                        comp_name,
                        mode_id,
                    )
                if 'materials' in f:
                    materials_eur = float(f["materials"])
                else:
                    materials_eur = 0.0
                    self.logger.warning(
                        "Component '%s', failure mode '%s': "
                        "No 'materials' field provided. Defaulting to 0.0 EUR.",
                        comp_name,
                        mode_id,
                    )

                if 'MTTW_L' in f:
                    mttwL_h = float(f['MTTW_L'])
                else:
                    mttwL_h = None
                    self.logger.warning(
                        "Component '%s', failure mode '%s': "
                        "No 'MTTW_L' field provided. Defaulting to infinity.",
                        comp_name,
                        mode_id,
                    )
                vessel = f.get("service_equipment")
                desc = f.get("description")

                # Compute λ_per_h from Weibull mean
                try:
                    mtbf_years = scale_years * math.gamma(1.0 + 1.0 / max(shape, 1e-9))
                except ValueError:
                    mtbf_years = scale_years
                rate_per_year = (1.0 / mtbf_years) if mtbf_years > 0 else 0.0
                lambda_per_h = rate_per_year / 8760.0

                preferred_vessels = [vessel] if vessel else ["CTV"]

                # n_technicians field (optional) to scale labour hours
                n_technicians = int(f.get("n_technicians", 1))

                specs.append(
                    MaintenanceSpec(
                        component=comp_name,
                        mode_id=str(mode_id),
                        task_type="CM",
                        lambda_per_h=lambda_per_h,
                        lambda_nominal_per_h=lambda_per_h,
                        shape=shape,
                        scale_years=scale_years,
                        frequency_per_year=None,
                        tau_h=None,
                        MTTR_h=mttr_h,
                        MTTW_L_h=mttwL_h,
                        spares_eur=materials_eur,
                        labour_rate_eur_h=labour_rate,
                        labour_h=mttr_h,
                        preferred_vessels=preferred_vessels,
                        n_technicians=n_technicians,
                        description=desc,
                    )
                )


            # ---------- Maintenance -> PM entries ----------
            maint_list = comp_data.get("maintenance", []) or []
            for i, m in enumerate(maint_list):
                if not isinstance(m, dict):
                    continue

                if "frequency" in m:
                    days_between = float(m["frequency"])
                    if days_between > 0:
                        tau_h = days_between * 24.0
                        freq_per_year = 365.0 / days_between
                    else:
                        tau_h = None
                        freq_per_year = 0.0
                        print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'frequency' field provided. Defaulting to 0.0 (PM disabled)."
                        )
                else:
                    freq_per_year = 0.0
                    tau_h = None
                    print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'frequency' field provided. Defaulting to 0.0 (PM disabled)."
                    )

                # --- time lookup with warning logic ---
                if "time" in m:
                    time_h = float(m["time"])
                else:
                    time_h = 0.0
                    print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'time' field provided. Defaulting to 0.0 h."
                    )

                # --- materials lookup with warning logic ---
                if "materials" in m:
                    materials_pm = float(m["materials"])
                else:
                    materials_pm = 0.0
                    print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'materials' field provided. Defaulting to 0.0 EUR."
                    )

                # --- vessel lookup with warning logic ---
                if "service_equipment" in m:
                    vessel_pm = m["service_equipment"]
                else:
                    vessel_pm = "CTV"
                    print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'service_equipment' field provided. Defaulting to 'CTV'."
                    )

                if 'MTTW_L' in m:
                    mttwL_h = float(m['MTTW_L'])
                else:
                    mttwL_h = None
                    print(
                        f"[OPEX WARNING] Component '{comp_name}', PM entry {i}: "
                        f"No 'MTTW_L' field provided. Defaulting to infinity."
                    )

                desc_pm = m.get("description")

                preferred_vessels_pm = [vessel_pm] if vessel_pm else ["CTV"]

                # n_technicians field (optional) to scale labour hours
                n_technicians = int(m.get("n_technicians", 1))


                specs.append(
                    MaintenanceSpec(
                        component=comp_name,
                        mode_id=f"PM:{i}",
                        task_type="PM",
                        lambda_per_h=0.0,
                        shape=None,
                        scale_years=None,
                        frequency_per_year=freq_per_year if freq_per_year > 0 else None,
                        tau_h=tau_h,
                        MTTR_h=time_h,
                        MTTW_L_h=mttwL_h,
                        spares_eur=materials_pm,
                        labour_rate_eur_h=labour_rate,
                        labour_h=time_h,
                        preferred_vessels=preferred_vessels_pm,
                        n_technicians=n_technicians,
                        description=desc_pm,
                    )
                )

        return specs
    
    def _apply_uncertainty_to_maintenance_specs(self, specs, rng):
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "uncertainty") or {}

        if bool(cfg.get("flag_apply_epistemic_lambda", False)):
            #self._apply_epistemic_uncertainty_failure_rates_gamma(specs, rng)
            self._apply_epistemic_uncertainty_failure_rates_gamma_shared(specs, rng)
            self._apply_epistemic_uncertainty_failure_rates_gamma_hierarchical_budgeted(specs, rng)

        if bool(cfg.get("flag_apply_process", False)):
            self._apply_process_uncertainty_lognormal(specs, rng)


    def _apply_process_uncertainty_lognormal(self, specs, rng):
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "uncertainty") or {}
        mttr_sigma  = float(cfg.get("mttr_sigma", 0.0))
        mttwL_sigma = float(cfg.get("mttwL_sigma", 0.0))
        anchor = str(cfg.get("process_anchor", "mean")).lower()

        def mu_for_anchor(sig: float) -> float:
            if anchor == "mean":
                return -0.5 * sig * sig   # E[f]=1
            elif anchor == "median":
                return 0.0                # median(f)=1
            else:
                raise ValueError(f"Unknown process_anchor={anchor!r}")

        for s in specs:
            if mttr_sigma > 0.0 and s.MTTR_h > 0.0:
                mu = mu_for_anchor(mttr_sigma)
                f = rng.lognormal(mean=mu, sigma=mttr_sigma)
                s.MTTR_h *= f
                if s.labour_h > 0.0:
                    s.labour_h *= f

            if (
                mttwL_sigma > 0.0
                and s.MTTW_L_h is not None
                and math.isfinite(s.MTTW_L_h)
                and s.MTTW_L_h > 0.0
            ):
                mu = mu_for_anchor(mttwL_sigma)
                f = rng.lognormal(mean=mu, sigma=mttwL_sigma)
                s.MTTW_L_h *= f


    @staticmethod
    def _gamma_unit_median(k: float) -> float:
        k = max(float(k), 1e-12)
        return float(gamma_dist.ppf(0.5, a=k, scale=1.0))

    def _apply_epistemic_uncertainty_failure_rates_gamma(self, specs, rng):
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "uncertainty") or {}
        cv = float(cfg.get("lambda_gamma_cv", 0.0))
        anchor = str(cfg.get("process_anchor", "mean")).lower()

        if cv <= 0.0:
            return

        k = 1.0 / max(1e-12, cv*cv)

        for s in specs:
            if s.task_type != "CM":
                continue
            lam0 = s.lambda_per_h
            if lam0 <= 0.0:
                continue

            if anchor == "mean":
                theta = lam0 / k
            elif anchor == "median":
                m1 = self._gamma_unit_median(k)
                theta = lam0 / max(m1, 1e-18)
            else:
                raise ValueError(f"Unknown lambda_gamma_anchor={anchor!r}")

            s.lambda_per_h = rng.gamma(shape=k, scale=theta)

    def _apply_epistemic_uncertainty_failure_rates_gamma_shared(self, specs, rng):
        """
        Apply a SINGLE shared epistemic Gamma factor F to all CM failure rates.

        If a cap quantile is used (here 0.90), we do NOT winsorize (min),
        but instead sample from the Gamma distribution CONDITIONED on F <= F_cap,
        i.e. a truncated/conditioned Gamma. This avoids a point-mass spike at the cap.
        """
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "uncertainty") or {}
        cv = float(cfg.get("lambda_gamma_cv", 0.0))
        anchor = str(cfg.get("process_anchor", "mean")).lower()

        if cv <= 0.0:
            return

        # Gamma shape from CV
        k = 1.0 / max(1e-12, cv * cv)

        # Determine scale so anchor statistic = 1 for the *unconditioned* Gamma
        if anchor == "mean":
            theta = 1.0 / k                      # mean(F)=1
        elif anchor == "median":
            m1 = gamma_dist.ppf(0.5, a=k, scale=1.0)
            theta = 1.0 / max(m1, 1e-18)         # median(F)=1
        else:
            raise ValueError(f"Unknown lambda_gamma_anchor={anchor!r}")

        # --- CONDITIONED / TRUNCATED GAMMA SAMPLING ---
        # Cap quantile (90% here; rename var if you prefer)
        q_cap = 0.90
        F_cap = float(gamma_dist.ppf(q_cap, a=k, scale=theta))

        # Rejection sampling: draw from Gamma until F <= F_cap
        # Acceptance probability is q_cap (e.g. 0.90), so this is fast.
        max_tries = 10_000  # safety guard (should never hit for reasonable q_cap)
        F = None
        for _ in range(max_tries):
            x = float(rng.gamma(shape=k, scale=theta))
            if x <= F_cap:
                F = x
                break
        if F is None:
            # Extremely unlikely fallback: just use the cap (but this should never happen)
            F = F_cap

        # Apply to all CM failure rates
        for s in specs:
            if s.task_type != "CM":
                continue
            if s.lambda_per_h <= 0.0:
                continue
            s.lambda_per_h *= F

    def _apply_epistemic_uncertainty_failure_rates_gamma_hierarchical_budgeted(self, specs, rng):
        """
        Option B: Split a target aggregate CV into:
        - shared/systemic factor F_shared
        - independent factors F_i per CM mode
        such that the aggregate turbine-level CV of sum(lambda_i) matches lambda_gamma_cv (approximately).

        Model: lambda_i = lambda_i0 * F_shared * F_i
        """
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "uncertainty") or {}

        cv_target = float(cfg.get("lambda_gamma_cv", 0.0))
        if cv_target <= 0.0:
            return

        r = float(cfg.get("lambda_gamma_budget_share", 1.0))  # default: all shared (backwards compatible)
        r = float(np.clip(r, 0.0, 1.0))

        anchor = str(cfg.get("process_anchor", "mean")).lower()

        # Optional cap quantile (None/<=0 disables)
        q_cap = float(cfg.get("lambda_gamma_cap_quantile", 0.0))
        if not (0.0 < q_cap < 1.0):
            q_cap = None

        # ---- collect baseline lambdas for weighting (CM only) ----
        cm_specs = [s for s in specs if s.task_type == "CM" and (s.lambda_per_h or 0.0) > 0.0]
        if not cm_specs:
            return

        lam0 = np.array([float(s.lambda_per_h) for s in cm_specs], dtype=float)
        Lam0 = float(lam0.sum())
        if Lam0 <= 0.0:
            return

        p = lam0 / Lam0
        sum_p2 = float(np.sum(p * p))
        sum_p2 = max(sum_p2, 1e-18)

        # ---- compute CV split (Option B) ----
        cv_shared = cv_target * math.sqrt(r)

        # If r==1 => no independent part
        if r >= 1.0:
            cv_ind = 0.0
        else:
            cv_ind = cv_target * math.sqrt(max(0.0, 1.0 - r)) / math.sqrt(sum_p2)

    # ---- helper: build Gamma factor distribution from CV + anchor ----
        def sample_gamma_factor(cv: float) -> float:
            if cv <= 0.0:
                return 1.0

            k = 1.0 / max(1e-12, cv * cv)

            # Choose scale so anchor statistic = 1 for the *unconditioned* gamma
            if anchor == "mean":
                theta = 1.0 / k                       # mean = 1
            elif anchor == "median":
                m1 = float(gamma_dist.ppf(0.5, a=k, scale=1.0))
                theta = 1.0 / max(m1, 1e-18)          # median = 1
            else:
                raise ValueError(f"Unknown process_anchor={anchor!r}")

            # Optional truncation to avoid point-mass spike: draw conditioned on <= cap
            if q_cap is not None:
                cap = float(gamma_dist.ppf(q_cap, a=k, scale=theta))
                # rejection sampling (acceptance prob = q_cap)
                for _ in range(10_000):
                    x = float(rng.gamma(shape=k, scale=theta))
                    if x <= cap:
                        return x
                return cap  # extremely unlikely fallback

            return float(rng.gamma(shape=k, scale=theta))

        # ---- sample shared factor once ----
        F_shared = sample_gamma_factor(cv_shared)

        # ---- sample independent factors and apply ----
        if cv_ind <= 0.0:
            # only shared
            for s in cm_specs:
                s.lambda_per_h *= F_shared
        else:
            for s in cm_specs:
                F_i = sample_gamma_factor(cv_ind)
                s.lambda_per_h *= (F_shared * F_i)




    def _apply_factors_to_maintenance_specs(self, specs: List[MaintenanceSpec]) -> None:
        """
        Deterministic mean-shift factors (scenario assumptions).
        These are NOT tied to scenarios here; orchestration sets them via overrides.
        """
        cfg = get_input_parameter(self.parameters, "analytic_ctmc", "mean_shift") or {}

        lambda_factor = float(cfg.get("lambda_factor", 1.0))
        mttr_factor   = float(cfg.get("mttr_factor", 1.0))
        mttwL_factor  = float(cfg.get("mttwL_factor", 1.0))
        tau_factor    = float(cfg.get("tau_factor", 1.0))

        for s in specs:
            # Failure rate: only meaningful for CM because PM uses tau_h
            if s.task_type == "CM" and s.lambda_nominal_per_h is not None:
                s.lambda_per_h = s.lambda_nominal_per_h * lambda_factor


            # MTTR affects both CM and PM (and keep labour consistent)
            if s.MTTR_h and s.MTTR_h > 0:
                s.MTTR_h *= mttr_factor
                if s.labour_h and s.labour_h > 0:
                    s.labour_h *= mttr_factor

            # Logistics waiting: apply if finite and >0 (both CM/PM if present)
            if s.MTTW_L_h is not None and math.isfinite(s.MTTW_L_h) and s.MTTW_L_h > 0:
                s.MTTW_L_h *= mttwL_factor

            # PM interval: apply only when tau exists (PM only)
            if s.task_type == "PM" and s.tau_h is not None and s.tau_h > 0:
                s.tau_h *= tau_factor
                # keep frequency consistent if you rely on it
                s.frequency_per_year = 8760.0 / s.tau_h


    def _load_vessels(self, *args, **kwargs) -> Dict[str, VesselSpec]:
        """
        Load vessels from the same OM input file the class already points to.
        Expected YAML structure:

        Vessels:
        CTV:
            equipment_rate: 3500          # €/day
            capability: CTV
            speed: 37.04                  # assumed km/h (≈ 20 kn)
            strategy: scheduled
            max_windspeed: 99             # m/s (or sentinel), optional
            max_waveheigh: 2              # typo variant supported
            max_waveheight: 2
            mobilization_cost: 0
            mobilization_days: 0
            port_distance: 30             # km

        Returns a dict: {vessel_name: VesselSpec}
        """
        try:
            vessel_data = load_yaml(self.config.valuewind_inputFolder, self.OM_vessel_input_file)
        except Exception as e:
            print(f"[OPEX] Could not load Vessel input file '{self.OM_vessel_input_file}' for vessels: {e}")
            # Minimal sensible defaults to keep model running
            return {
                "CTV": VesselSpec(name="CTV", capability="CTV", Hs_limit_m=1.5, wind_limit_ms=None,
                                speed_kn=25.0, base_distance_km=40.0, strategy="unspecified",
                                day_rate_eur=15000.0, mobilization_fee_eur=0.0, mobilization_days=0.0)
            }

        vessels_block = (vessel_data or {}).get("Vessels", {})
        if not isinstance(vessels_block, dict) or not vessels_block:
            print("[OPEX] Vessel input has no 'Vessels' block or it is empty; returning default CTV.")
            return {
                "CTV": VesselSpec(name="CTV", capability="CTV", Hs_limit_m=1.5, wind_limit_ms=None,
                                speed_kn=25.0, base_distance_km=40.0, strategy="unspecified",
                                day_rate_eur=15000.0, mobilization_fee_eur=0.0, mobilization_days=0.0)
            }

        out: Dict[str, VesselSpec] = {}
        for v_name, v_data in vessels_block.items():
            if not isinstance(v_data, dict):
                continue

            # Pull with robust defaults
            equipment_rate = float(v_data.get("equipment_rate", 0.0) or 0.0)
            capability = str(v_data.get("capability", "CTV") or "CTV")
            strategy = str(v_data.get("strategy", "unspecified") or "unspecified")

            # Speed handling: assume YAML 'speed' is km/h; convert to knots
            speed_val = float(v_data.get("speed", 0.0) or 0.0)
            speed_kn = speed_val / 1.852 if speed_val > 0 else 0.0

            # Wave height limit: accept both correct and typo keys
            hs_limit = v_data.get("max_waveheight", v_data.get("max_waveheigh", None))
            Hs_limit_m = float(hs_limit) if hs_limit is not None else 1.5

            # Wind limit (optional)
            wind_limit_ms = v_data.get("max_windspeed", None)
            wind_limit_ms = float(wind_limit_ms) if wind_limit_ms is not None else None

            # Costs & logistics
            mobilization_fee = float(v_data.get("mobilization_cost", 0.0) or 0.0)
            mobilization_days = float(v_data.get("mobilization_days", 0.0) or 0.0)
            port_distance_km = float(v_data.get("port_distance", 0.0) or 0.0)

            out[v_name] = VesselSpec(
                name=v_name,
                capability=capability,
                Hs_limit_m=Hs_limit_m,
                wind_limit_ms=wind_limit_ms,
                speed_kn=speed_kn,
                base_distance_km=port_distance_km,
                strategy=strategy,
                day_rate_eur=equipment_rate,
                mobilization_fee_eur=mobilization_fee,
                mobilization_days=mobilization_days,
            )

        return out




    def _compute_access_profiles(
        self,
        maintenance_specs: List[MaintenanceSpec],
        vessels: Dict[str, VesselSpec],
        p_access_hourly: float,   # hardcoded hourly probability of being accessible
    ) -> Dict[Tuple[str, str], AccessProfile]:
        """
        Build access/repair/transit rates per MaintenanceSpec (each failure mode or PM entry).

        Now we distinguish explicitly:
          - Logistics waiting (mean_logistic_wait_h -> mu_L_per_h)
          - Weather waiting (mean_weather_wait_h  -> mu_A_per_h)
          - Repair/service (service_time_h        -> mu_R_per_h)

        Simple heuristics:
          - mean_logistic_wait_h is derived from vessel.mobilization_days if > 0,
            otherwise assumed 0 (no explicit logistics delay).
          - Weather waiting uses a Bernoulli-window approximation:
                p_window = p_access_hourly ** L_hrs
                mu_A_per_h = p_window
            where L_hrs is the required continuous window length (transit + onsite + transit).
        """
        profiles: Dict[Tuple[str, str], AccessProfile] = {}

        for s in maintenance_specs:
            # --- resolve requested vessel CAPABILITY ------------------------------
            if s.preferred_vessels and len(s.preferred_vessels) > 0:
                requested_cap = s.preferred_vessels[0]  # interpret as capability label
            else:
                requested_cap = "CTV"  # sensible default capability

            # find a vessel with matching capability
            matching_vessels = [
                v for v in vessels.values()
                if v.capability.lower() == requested_cap.lower()
            ]

            if not matching_vessels:
                available_caps = sorted({v.capability for v in vessels.values()})
                raise ValueError(
                    f"[OPEX] No vessel available with capability '{requested_cap}' "
                    f"for component '{s.component}', mode '{s.mode_id}'. "
                    f"Available capabilities: {available_caps}"
                )

            # choose first vessel with that capability (can be refined later)
            v = matching_vessels[0]

            # --- transit and service times ---------------------------------------
            speed_kmh = 1.852 * v.speed_kn
            if speed_kmh > 0 and v.base_distance_km > 0:
                t_transit = v.base_distance_km / speed_kmh
            else:
                t_transit = 0.0

            t_transit_out = t_transit
            t_transit_back = t_transit
            t_onsite = max(0.0, s.MTTR_h)

            # total continuous window length needed (hours)
            L = t_transit_out + t_onsite + t_transit_back
            L_hrs = int(math.ceil(max(0.0, L)))
            
            
            # --- Logistics waiting (MTTL) ----------------------------------------
            # Use per-mode MTTW_L_h from MaintenanceSpec if given.
            # Semantics:
            #   - finite, > 0:   real logistics waiting in hours
            #   - None / inf / ≤0: no explicit logistics delay modelled (instantaneous)
            mttwL_h = getattr(s, "MTTW_L_h", None)

            # If missing or inf → model instantaneous logistics clearing
            if mttwL_h is None or not math.isfinite(mttwL_h) or mttwL_h <= 0.0:
                mean_logistic_wait_h = 0.0
                mu_L_per_h = 1e12   # practically instantaneous
            else:
                mean_logistic_wait_h = float(mttwL_h)
                mu_L_per_h = 1.0 / mean_logistic_wait_h


            # --- Weather waiting (MTTW) ------------------------------------------
            # Bernoulli weather window approximation:
            #   p_window = p_access_hourly ** L_hrs
            #   mu_A_per_h = p_window
            # For p_access_hourly in (0,1), this gives small p_window for long jobs.

            #p_window = max(1e-12, p_access_hourly ** L_hrs)

            mu_A_per_h = p_access_hourly
            mean_weather_wait_h = (1.0 / mu_A_per_h) if mu_A_per_h > 0.0 else float("inf")

            # --- Repair / service rate -------------------------------------------
            service_time_h = t_transit_out + t_onsite + t_transit_back
            mu_R_per_h = (1.0 / service_time_h) if service_time_h > 0 else 0.0

            profiles[(s.component, str(s.mode_id))] = AccessProfile(
                task_type=s.task_type,
                mu_L_per_h=mu_L_per_h,
                mean_logistic_wait_h=mean_logistic_wait_h,
                mu_A_per_h=mu_A_per_h,
                mean_weather_wait_h=mean_weather_wait_h,
                mu_R_per_h=mu_R_per_h,
                service_time_h=service_time_h,
                # store capability label, not vessel name
                chosen_vessels=[requested_cap],
            )

        return profiles



    def _project_time_index(self) -> Tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp, float]:
        """
        Compute the operational time window and a monthly DatetimeIndex.

        Returns
        -------
        idx : pd.DatetimeIndex
            Monthly timestamps (month starts) covering the operational window.
        op_start_ts : pd.Timestamp
            Absolute timestamp when the operational phase begins.
        op_end_ts : pd.Timestamp
            Absolute timestamp when the operational phase ends.
        T_h : float
            Total operational duration in hours.
        """
        # --- Retrieve config offsets (hours from project start)
        start_h = getattr(self.env.config, "WF_OperationsStart_h", 0.0)
        end_h   = getattr(self.env.config, "WF_OperationsEnd_h", 0.0)

        # --- Compute absolute timestamps for operational window
        op_start_ts = self.project_start + pd.to_timedelta(start_h, unit="h")
        op_end_ts   = self.project_start + pd.to_timedelta(end_h,   unit="h")

        # --- Total operational duration in hours
        T_h = max(1.0, (op_end_ts - op_start_ts).total_seconds() / 3600.0)

        # --- Snap to month starts and build monthly index
        start_month = op_start_ts.to_period("M").start_time
        end_month   = op_end_ts.to_period("M").start_time
        idx = pd.date_range(start=start_month, end=end_month, freq="MS")

        return idx, op_start_ts, op_end_ts, T_h



    def _even_payment_schedule(self, total_opex_eur: float, idx: pd.DatetimeIndex) -> pd.DataFrame:
        per = np.repeat(total_opex_eur / len(idx), len(idx))
        return pd.DataFrame({"timestamp": idx, "OM_payment": per})

    # --------------------------- Mode 1: CAPEX fraction ---------------------------
    def _calc_opex_as_fraction_of_capex(self) -> tuple[pd.DataFrame, dict]:
        """
        Legacy/simple mode: OpEx as a fraction of CapEx.
        Returns (OPEX_records_df, extras_dict)
        """
        # --- parameters ---
        om_frac = float(get_input_parameter(self.parameters,'capex_fraction','capex_fraction'))  # e.g., 3% of CAPEX

        # --- CAPEX total (must exist) ---
        capex_df = self.env.capex.get_cost_dataframe()
        if capex_df is None or capex_df.empty or "cost" not in capex_df.columns:
            raise ValueError("CAPEX data is missing or invalid — cannot compute OPEX as fraction of CAPEX.")

        capex_total = float(pd.to_numeric(capex_df["cost"], errors="coerce").fillna(0.0).sum())
        total_opex = om_frac * capex_total  # total over entire ops window

        # --- project index using helper ---
        idx, op_start_ts, op_end_ts, T_h = self._project_time_index()

        # --- even payment schedule using helper ---
        # Pass total_opex as negative (outflow convention)
        opex_df = self._even_payment_schedule(total_opex_eur=-abs(total_opex), idx=idx)

        # --- optional: scale production by availability ---
        availability = float(getattr(self, "availability", 0.93))
        self._apply_availability_to_power_window(availability)

        # --- extras payload ---
        availability_summary = AvailabilitySummary(
            component_A={}, turbine_A={}, farm_A=availability, downtime_h={}
        )

        extras = {
            "availability_summary": availability_summary,
            "availability_profile": None,
            "activity_log": None,
            "OpEx_breakdown": OpExBreakdown(
                fixed_OM_eur=total_opex,
                CM_cost_eur=0.0,
                PM_cost_eur=0.0,
                transport_eur=0.0,
                labour_eur=0.0,
                spares_eur=0.0,
            ),
        }

        return opex_df, extras


    # --------------------------- Mode 2: Analytical CTMC ---------------------------
    def _calc_opex_analytic_ctmc(self) -> tuple[pd.DataFrame, dict]:
        """
        Analytical CTMC approach with one unified CTMC per component.

        - For each component, build a CTMC with:
            state 0 = UP
            state i = down in mode i (failure or PM)
        - Transition rates:
            0 -> i : lambda_i  (failure / PM rate)
            i -> 0 : mu_i      (repair+access rate)
        - Component availability = steady-state probability pi[0].
        """
        self.p_access_hourly = float(get_input_parameter(self.parameters, "analytic_ctmc", "p_access_hourly"))
        maint_specs = self._load_maintenance_specs()

        self.process_mean_shift = bool(get_input_parameter(self.parameters, "analytic_ctmc", "mean_shift", "flag_apply"))
        if self.process_mean_shift:
            self._apply_factors_to_maintenance_specs(maint_specs)

            ms_cfg = get_input_parameter(self.parameters, "analytic_ctmc", "mean_shift") or {}
            p_access_factor = float(ms_cfg.get("p_access_factor", 1.0))
            self.p_access_hourly = float(np.clip(self.p_access_hourly * p_access_factor, 0.0, 1.0))

            self._apply_uncertainty_to_maintenance_specs(maint_specs, rng=np.random.default_rng())

        vessels = self._load_vessels()
        access = self._compute_access_profiles(maint_specs, vessels, p_access_hourly=self.p_access_hourly)

        idx, op_start_ts, op_end_ts, T_h = self._project_time_index()
        n_turbines = self.env.windFarm.n_turbines

        # ---- containers -----------------------------------------------------------
        component_costs: Dict[str, Dict[str, float]] = {}
        transport_total = labour_total = spares_total = 0.0
        transport_cm = labour_cm = spares_cm = 0.0
        transport_pm = labour_pm = spares_pm = 0.0
        mode_cost_rows: List[dict] = []


        # --------------------------------------------------------------------------
        # 1) COSTS & EXPECTED EVENT COUNTS (per mode, as before)
        # --------------------------------------------------------------------------
        for s in maint_specs:
            key = (s.component, str(s.mode_id))
            ap = access.get(key)
            if ap is None:
                # shouldn't happen, but be defensive
                continue

            t_service = ap.service_time_h

            # event rate per turbine
            if s.task_type == "CM":
                lam = s.lambda_per_h
                N_per_turbine = lam * T_h
            else:
                lam = (1.0 / s.tau_h) if s.tau_h else 0.0
                N_per_turbine = (T_h / s.tau_h) if s.tau_h else 0.0

            # costs per event
            requested_cap = None
            if ap.chosen_vessels:
                requested_cap = ap.chosen_vessels[0]    # interpreted as capability label
            else:
                requested_cap = "CTV" # default capability
                # print warning
                print(f"[OPEX] Warning: No vessel requested for {s.component} mode {s.mode_id}; defaulting to 'CTV'.")


            # Find a vessel with matching capability
            matching_vessels = [
                v for v in vessels.values()
                if v.capability.lower() == requested_cap.lower()
            ]

            if not matching_vessels:
                available_caps = sorted({v.capability for v in vessels.values()})
                raise ValueError(
                    f"[OPEX] No vessel available with capability '{requested_cap}'. "
                    f"Available capabilities: {available_caps}"
                )

            # select the first matching vessel
            v = matching_vessels[0]

            cost_transport = v.day_rate_eur * (t_service / 24.0)
            cost_labour = s.labour_rate_eur_h * s.labour_h * s.n_technicians
            cost_spares = s.spares_eur

            N_farm = N_per_turbine * n_turbines
            this_event_transport = N_farm * cost_transport
            this_event_labour = N_farm * cost_labour
            this_event_spares = N_farm * cost_spares
            this_event_total = (
                this_event_transport + this_event_labour + this_event_spares
            )

            mode_cost_rows.append(
                {
                    "component": s.component,
                    "mode_id": str(s.mode_id),
                    "task_type": s.task_type,        # "CM" or "PM"
                    "N_interventions": float(N_farm),
                    "transport_eur": float(this_event_transport),
                    "labour_eur": float(this_event_labour),
                    "spares_eur": float(this_event_spares),
                    "fixed_OM_eur": 0.0,            # filled in later or left as 0
                    "total_eur": float(this_event_total),
                }
            )


            # global totals
            transport_total += this_event_transport
            labour_total += this_event_labour
            spares_total += this_event_spares

            if s.task_type == "CM":
                transport_cm += this_event_transport
                labour_cm += this_event_labour
                spares_cm += this_event_spares
            else:
                transport_pm += this_event_transport
                labour_pm += this_event_labour
                spares_pm += this_event_spares

            # per-component accumulation
            cc = component_costs.setdefault(
                s.component,
                {
                    "transport_eur": 0.0,
                    "labour_eur": 0.0,
                    "spares_eur": 0.0,
                    "CM_eur": 0.0,
                    "PM_eur": 0.0,
                    "total_eur": 0.0,
                },
            )
            cc["transport_eur"] += this_event_transport
            cc["labour_eur"] += this_event_labour
            cc["spares_eur"] += this_event_spares
            cc["total_eur"] += this_event_total
            if s.task_type == "CM":
                cc["CM_eur"] += this_event_total
            else:
                cc["PM_eur"] += this_event_total

        # --------------------------------------------------------------------------
        # 2) AVAILABILITY VIA UNIFIED CTMC PER COMPONENT (WITH WAITING & REPAIR STATES)
        # --------------------------------------------------------------------------
        # group specs by component
        comp_to_specs: Dict[str, List[MaintenanceSpec]] = {}
        for s in maint_specs:
            comp_to_specs.setdefault(s.component, []).append(s)

        component_A: Dict[str, float] = {}
        downtime_h: Dict[str, float] = {}

        # NEW: downtime split containers (define ONCE, outside loop)
        downtime_logistics_h: Dict[str, float] = {}
        downtime_weather_h: Dict[str, float] = {}
        downtime_repair_h: Dict[str, float] = {}

        downtime_logistics_fraction: Dict[str, float] = {}
        downtime_weather_fraction: Dict[str, float] = {}
        downtime_repair_fraction: Dict[str, float] = {}

        for comp, specs_list in comp_to_specs.items():
            n_modes = len(specs_list)
            if n_modes == 0:
                component_A[comp] = 1.0
                downtime_h[comp] = 0.0

                downtime_logistics_fraction[comp] = 0.0
                downtime_weather_fraction[comp] = 0.0
                downtime_repair_fraction[comp] = 0.0

                downtime_logistics_h[comp] = 0.0
                downtime_weather_h[comp] = 0.0
                downtime_repair_h[comp] = 0.0
                continue

            # Explicit CTMC structure (with separate logistics & weather waiting):
            #
            #   state 0        = UP
            #   state 3*k+1    = WL_k  (waiting for logistics for mode k)
            #   state 3*k+2    = WW_k  (logistics ready, waiting for weather)
            #   state 3*k+3    = R_k   (under repair / service)
            #
            n_states = 1 + 3 * n_modes
            Q = np.zeros((n_states, n_states), dtype=float)

            lambdas: List[float] = []
            mu_Ls: List[float] = []
            mu_As: List[float] = []
            mu_Rs: List[float] = []

            for local_idx, s in enumerate(specs_list):
                iWL = 3 * local_idx + 1
                iWW = 3 * local_idx + 2
                iR  = 3 * local_idx + 3

                key = (s.component, str(s.mode_id))
                ap = access.get(key)

                lam = mu_L = mu_A = mu_R = 0.0

                if ap is not None:
                    # failure / PM rate from UP to WL
                    if s.task_type == "CM":
                        lam = s.lambda_per_h
                    else:
                        lam = (1.0 / s.tau_h) if s.tau_h else 0.0

                    # logistics rate from WL to WW
                    mu_L = ap.mu_L_per_h if ap.mu_L_per_h > 0 else 0.0
                    # access rate from WW to R (weather window)
                    mu_A = ap.mu_A_per_h if ap.mu_A_per_h > 0 else 0.0
                    # repair rate from R to UP
                    mu_R = ap.mu_R_per_h if ap.mu_R_per_h > 0 else 0.0

                lambdas.append(lam)
                mu_Ls.append(mu_L)
                mu_As.append(mu_A)
                mu_Rs.append(mu_R)

                # 0 -> WL
                Q[0, iWL] += lam
                Q[0, 0]   -= lam

                # WL -> WW
                Q[iWL, iWW] += mu_L
                Q[iWL, iWL] -= mu_L

                # WW -> R
                Q[iWW, iR] += mu_A
                Q[iWW, iWW] -= mu_A

                # R -> 0
                Q[iR, 0] += mu_R
                Q[iR, iR] -= mu_R

            total_lambda = sum(lambdas)

            # degenerate case: no failures / PM and no progress
            if (
                total_lambda == 0.0
                and all(mu == 0.0 for mu in mu_Ls)
                and all(mu == 0.0 for mu in mu_As)
                and all(mu == 0.0 for mu in mu_Rs)
            ):
                A_comp = 1.0

                downtime_logistics_fraction[comp] = 0.0
                downtime_weather_fraction[comp] = 0.0
                downtime_repair_fraction[comp] = 0.0

                downtime_logistics_h[comp] = 0.0
                downtime_weather_h[comp] = 0.0
                downtime_repair_h[comp] = 0.0

            else:
                # solve steady state: pi Q = 0, sum(pi) = 1
                A_mat = Q.T.copy()
                b_vec = np.zeros(n_states)
                A_mat[-1, :] = 1.0
                b_vec[-1] = 1.0

                try:
                    pi = np.linalg.solve(A_mat, b_vec)
                    pi = np.maximum(pi, 0.0)
                    s_sum = pi.sum()
                    if s_sum > 0:
                        pi /= s_sum

                    # availability
                    A_comp = float(pi[0])

                    # downtime fractions by state group
                    wl_idx = []
                    ww_idx = []
                    r_idx  = []

                    for local_idx in range(n_modes):
                        iWL = 3 * local_idx + 1
                        iWW = 3 * local_idx + 2
                        iR  = 3 * local_idx + 3
                        if iWL < len(pi): wl_idx.append(iWL)
                        if iWW < len(pi): ww_idx.append(iWW)
                        if iR  < len(pi): r_idx.append(iR)

                    wl_frac = float(pi[wl_idx].sum()) if wl_idx else 0.0
                    ww_frac = float(pi[ww_idx].sum()) if ww_idx else 0.0
                    r_frac  = float(pi[r_idx].sum())  if r_idx  else 0.0

                    downtime_logistics_fraction[comp] = wl_frac
                    downtime_weather_fraction[comp]   = ww_frac
                    downtime_repair_fraction[comp]    = r_frac

                    downtime_logistics_h[comp] = wl_frac * T_h * n_turbines
                    downtime_weather_h[comp]   = ww_frac * T_h * n_turbines
                    downtime_repair_h[comp]    = r_frac  * T_h * n_turbines

                except np.linalg.LinAlgError:
                    A_comp = 1.0

                    downtime_logistics_fraction[comp] = 0.0
                    downtime_weather_fraction[comp] = 0.0
                    downtime_repair_fraction[comp] = 0.0

                    downtime_logistics_h[comp] = 0.0
                    downtime_weather_h[comp] = 0.0
                    downtime_repair_h[comp] = 0.0

            component_A[comp] = A_comp
            downtime_h[comp] = (1.0 - A_comp) * T_h * n_turbines

        # --------------------------------------------------------------------------
        # 3) TURBINE / FARM AVAILABILITY (still series over components)
        # --------------------------------------------------------------------------
        A_turb = float(np.prod(list(component_A.values()))) if component_A else 1.0
        turbine_A = {f"T{tid}": A_turb for tid in range(n_turbines)}
        farm_A = A_turb

        # --------------------------------------------------------------------------
        # 4) OPEX TOTALS + TIME SERIES
        # --------------------------------------------------------------------------
        OM_overhead = get_input_parameter(self.parameters, 'OM_overhead') * 1000.0  # €/kW-year 
        turbine_rated_power = float(getattr(self.env.windFarm, "turbine_rated_power", 0.0))
        n_turbines = self.env.windFarm.n_turbines

        fixed_om = OM_overhead * turbine_rated_power * n_turbines * (T_h / 8760.0)

        if fixed_om != 0.0:
            mode_cost_rows.append(
                {
                    "component": "Fixed OM",
                    "mode_id": "fixed",
                    "task_type": "fixed",   # or "PM" if you prefer
                    "N_interventions": 0.0,
                    "transport_eur": 0.0,
                    "labour_eur": 0.0,
                    "spares_eur": 0.0,
                    "fixed_OM_eur": float(fixed_om),
                    "total_eur": float(fixed_om),
                }
            )

        cm_cost = transport_cm + labour_cm + spares_cm
        pm_cost = transport_pm + labour_pm + spares_pm
        transport = transport_total
        labour = labour_total
        spares = spares_total
        total = transport + labour + spares + fixed_om
        total_opex_eur = total


        # monthly schedule (even payment)
        opex_df = self._even_payment_schedule(-abs(total), idx)

        # add per-component cost columns (evenly distributed)
        if len(opex_df) > 0:
            for comp, costs in component_costs.items():
                comp_total = costs.get("total_eur", 0.0)
                per_month = comp_total / len(opex_df)
                opex_df[f"{comp}_cost"] = per_month

            opex_df["fixed_OM"] = fixed_om / len(opex_df)
        else:
            opex_df["fixed_OM"] = 0.0

        # -------------------------------------------------------------------------
        # 5) correct production by availability (window-aware)
        # -------------------------------------------------------------------------
        self._apply_availability_to_power_window(farm_A)

        # --------------------------------------------------------------------------
        # 6) EXTRAS PAYLOAD (consistent with _calc_opex_as_fraction_of_capex)
        # --------------------------------------------------------------------------
        extras = {
            "mode": "analytic_ctmc",
            "availability_summary": AvailabilitySummary(
                component_A=component_A,
                turbine_A=turbine_A,
                farm_A=farm_A,
                downtime_h=downtime_h,
            ),
            "availability_profile": self._monthly_availability_profile(idx, farm_A),
            "activity_log": None,
            "OpEx_breakdown": OpExBreakdown(
                fixed_OM_eur=fixed_om,
                CM_cost_eur=cm_cost,
                PM_cost_eur=pm_cost,
                transport_eur=transport,
                labour_eur=labour,
                spares_eur=spares,
            ),
            "component_cost_breakdown": component_costs,
            "mode_cost_breakdown": mode_cost_rows,
            "total_opex_eur": total_opex_eur,
            "ops_horizon": {
                "T_h": T_h,
                "op_start_ts": op_start_ts,
                "op_end_ts": op_end_ts,
                "n_months": len(idx),
            },
        }

        extras["downtime_breakdown"] = {
            "logistics_h": downtime_logistics_h,
            "weather_h": downtime_weather_h,
            "repair_h": downtime_repair_h,
            "logistics_fraction": downtime_logistics_fraction,
            "weather_fraction": downtime_weather_fraction,
            "repair_fraction": downtime_repair_fraction,
        }

        return opex_df, extras



    def _monthly_availability_profile(self, idx: pd.DatetimeIndex, farm_A: float) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": idx, "availability": np.repeat(farm_A, len(idx))})
    

    def _apply_availability_to_power_window(
        self,
        availability: float,
    ) -> None:
        """
        Apply availability scaling to env.windFarm.power_records
        ONLY within the current OPEX window defined by
        config.WF_OperationsStart_h / WF_OperationsEnd_h.

        This function is safe to call multiple times for different windows;
        it will never double-scale outside the active window.
        """

        if availability is None:
            return

        if getattr(self.env, "windFarm", None) is None:
            return

        power_df = getattr(self.env.windFarm, "power_records", None)
        if power_df is None or power_df.empty:
            return

        if "timestamp" not in power_df.columns:
            return

        # Derive window timestamps from config
        project_start = pd.to_datetime(self.config.Project_StartDate)
        op_start_h = int(getattr(self.config, "WF_OperationsStart_h"))
        op_end_h   = int(getattr(self.config, "WF_OperationsEnd_h"))

        op_start_ts = project_start + pd.to_timedelta(op_start_h, unit="h")
        op_end_ts   = project_start + pd.to_timedelta(op_end_h, unit="h")

        ts = pd.to_datetime(power_df["timestamp"])
        mask = (ts >= op_start_ts) & (ts < op_end_ts)

        if not mask.any():
            return

        availability = float(availability)

        # Scale only numeric, non-timestamp columns
        prod_cols = [
            c for c in power_df.columns
            if c != "timestamp" and pd.api.types.is_numeric_dtype(power_df[c])
        ]

        if prod_cols:
            power_df.loc[mask, prod_cols] = power_df.loc[mask, prod_cols] * availability
        else:
            # Fallback: attempt numeric coercion
            cols = [c for c in power_df.columns if c != "timestamp"]
            power_df.loc[mask, cols] = (
                power_df.loc[mask, cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                * availability
            )

        # Explicit assignment for clarity
        self.env.windFarm.power_records = power_df



    # --------------------------- Mode 3: Time-marching CTMC ---------------------------
    def _calc_opex_time_march(self) -> Dict[str, Any]:
        """
        Time-marching CTMC:
        - build per-(turbine, component) CTMCs
        - sample events forward in (continuous) time
        - log activities + costs
        - aggregate to monthly OPEX and availability

        This is a skeleton – fill in TODOs with real sampling, metocean, and dispatch logic.
        """
        # 1) Inputs / setup -----------------------------------------------------------
        maint_specs = self._load_maintenance_specs()
        vessels     = self._load_vessels()

        windFarm = getattr(self.env, "windFarm", None)
        t_ids    = self._turbine_ids(windFarm) if windFarm is not None else ["T01"]

        # project time window
        idx, op_start_ts, op_end_ts, T_h = self._project_time_index()
        sim_end_ts = op_end_ts

        # 2) Build CTMC entities ------------------------------------------------------
        # One entity per (turbine, maintenance spec) for now
        # later you can aggregate per component if needed
        entities = []  # list of dicts or small structs
        for tid in t_ids:
            for s in maint_specs:
                # resolve access profile (gives us mu_A and service_time)
                ap = self._compute_access_profiles([s], vessels).get((s.component, str(s.mode_id)))
                if ap is None:
                    continue

                entity = {
                    "turbine_id": tid,
                    "component": s.component,
                    "mode_id": str(s.mode_id),
                    "task_type": s.task_type,    # "CM" or "PM"

                    # CTMC state
                    "state": "W",                 # W, WA, R
                    "last_change_ts": op_start_ts,

                    # rates (can be made time-dependent later)
                    "lambda_h": s.lambda_per_h if s.task_type == "CM" else (1.0 / s.tau_h if s.tau_h else 0.0),
                    "mu_A_h": ap.mu_A_per_h,
                    "mu_R_h": (1.0 / ap.service_time_h) if ap.service_time_h > 0 else 0.0,

                    # cost params
                    "service_time_h": ap.service_time_h,
                    "preferred_vessels": ap.chosen_vessels or ["CTV"],
                    "labour_rate_eur_h": s.labour_rate_eur_h,
                    "labour_h": s.labour_h,
                    "spares_eur": s.spares_eur,

                    # accounting
                    "time_in_state": {
                        "W": 0.0,
                        "WA": 0.0,
                        "R": 0.0,
                    },
                }
                entities.append(entity)

        # 3) Event list / priority queue ---------------------------------------------
        # For the skeleton we just use a python list; replace with heapq in real code
        # Each event: {"time": ts, "entity_idx": i, "transition": "..."}
        event_queue = []

        # initialize first events per entity (TODO: real sampling)
        for i, ent in enumerate(entities):
            # TODO: sample next event time from current rates and push to queue
            # placeholder: everyone gets an event at op_start_ts
            event_queue.append({
                "time": op_start_ts,
                "entity_idx": i,
                "transition": None,  # to be decided at execution time
            })

        # 4) Activity log (final output) ----------------------------------------------
        activity_rows: List[ActivityLogEntry] = []

        # 5) Turbine-level availability tracking -------------------------------------
        # we track how many components are down per turbine
        turbine_down_counter = {tid: 0 for tid in t_ids}
        turbine_last_change  = {tid: op_start_ts for tid in t_ids}
        turbine_time_up      = {tid: 0.0 for tid in t_ids}
        turbine_time_down    = {tid: 0.0 for tid in t_ids}

        # 6) Main simulation loop -----------------------------------------------------
        current_ts = op_start_ts
        # NOTE: replace this while with proper heap pop logic
        while event_queue and current_ts < sim_end_ts:
            # pick next event (TODO: pop from heap by time)
            evt = event_queue.pop(0)
            evt_ts = evt["time"]
            if evt_ts > sim_end_ts:
                break

            # advance global time
            prev_ts = current_ts
            current_ts = evt_ts
            dt = (current_ts - prev_ts).total_seconds() / 3600.0  # h

            # between prev_ts and current_ts, everyone stayed in their current state
            for ent in entities:
                state = ent["state"]
                ent["time_in_state"][state] += dt

            # also update turbine up/down time
            for tid in t_ids:
                if turbine_down_counter[tid] > 0:
                    turbine_time_down[tid] += dt
                else:
                    turbine_time_up[tid] += dt

            # process this event ------------------------------------------------------
            ent = entities[evt["entity_idx"]]

            # TODO: decide which transition fires, based on ent["state"] and rates
            # For skeleton: force a W -> WA -> R -> W cycle
            old_state = ent["state"]
            if old_state == "W":
                new_state = "WA"
            elif old_state == "WA":
                new_state = "R"
            else:
                new_state = "W"

            ent["state"] = new_state
            ent["last_change_ts"] = current_ts

            # if this transition corresponds to doing maintenance, log activity
            if old_state == "WA" and new_state == "R":
                # build activity row from ent and current_ts
                v_name = ent["preferred_vessels"][0]
                v = vessels.get(v_name, None)
                transport_eur = (v.day_rate_eur * (ent["service_time_h"] / 24.0)) if v else 0.0
                labour_eur    = ent["labour_rate_eur_h"] * ent["labour_h"]
                spares_eur    = ent["spares_eur"]
                total_eur     = transport_eur + labour_eur + spares_eur

                act = ActivityLogEntry(
                    timestamp_start=current_ts,
                    timestamp_end=current_ts + pd.Timedelta(hours=ent["service_time_h"]),
                    turbine_id=ent["turbine_id"],
                    component=ent["component"],
                    task_type=ent["task_type"],
                    vessels=[v_name],
                    sailing_h=0.0,     # TODO
                    onsite_h=ent["service_time_h"],
                    crew_h=ent["labour_h"],
                    spares_eur=spares_eur,
                    labour_eur=labour_eur,
                    transport_eur=transport_eur,
                    total_cost_eur=total_eur,
                )
                activity_rows.append(act)

            # update turbine down-counter if this component changed up/down
            tid = ent["turbine_id"]
            if old_state == "W" and new_state != "W":
                turbine_down_counter[tid] += 1
            elif old_state != "W" and new_state == "W":
                turbine_down_counter[tid] = max(0, turbine_down_counter[tid] - 1)

            # schedule next event for this entity (TODO: real sampling from rates)
            next_evt_ts = current_ts + pd.Timedelta(hours=24.0)  # placeholder
            if next_evt_ts < sim_end_ts:
                event_queue.append({
                    "time": next_evt_ts,
                    "entity_idx": evt["entity_idx"],
                    "transition": None,
                })

        # 7) Build activity log DataFrame --------------------------------------------
        activity_log_df = pd.DataFrame([a.__dict__ for a in activity_rows]) if activity_rows else pd.DataFrame(
            columns=[f.name for f in dataclasses.fields(ActivityLogEntry)]  # type: ignore
        )

        # 8) Aggregate to monthly OPEX ------------------------------------------------
        if not activity_log_df.empty:
            opex_df = (
                activity_log_df
                .assign(month=lambda d: d["timestamp_start"].values.astype("datetime64[M]"))
                .groupby("month", as_index=False)["total_cost_eur"].sum()
                .rename(columns={"month": "timestamp", "total_cost_eur": "OM_payment"})
            )
        else:
            opex_df = pd.DataFrame({"timestamp": idx, "OM_payment": np.zeros(len(idx))})

        # add fixed OM evenly
        fixed_om_total = float(getattr(self.config, "Fixed_OM_Annual", 0.0)) * (len(idx) / 12.0)
        fixed_df = self._even_payment_schedule(fixed_om_total, idx)
        opex_df = (
            pd.concat([opex_df, fixed_df], ignore_index=True)
            .groupby("timestamp", as_index=False)["OM_payment"].sum()
            .sort_values("timestamp")
        )

        # 9) Availability outputs -----------------------------------------------------
        # per component (expected): time_in_state["W"] / total
        component_A = {}
        downtime_h = {}
        total_sim_h = (sim_end_ts - op_start_ts).total_seconds() / 3600.0
        for ent in entities:
            up_h = ent["time_in_state"]["W"]
            A = up_h / total_sim_h if total_sim_h > 0 else 1.0
            component_A[ent["component"]] = max(component_A.get(ent["component"], 0.0), A)
            downtime_h[ent["component"]] = total_sim_h - up_h

        # turbine availability from accumulated times
        turbine_A = {}
        for tid in t_ids:
            up_h = turbine_time_up[tid]
            A = up_h / total_sim_h if total_sim_h > 0 else 1.0
            turbine_A[tid] = A
        farm_A = float(np.mean(list(turbine_A.values()))) if turbine_A else 1.0

        availability_series = pd.DataFrame({"timestamp": idx, "availability": np.repeat(farm_A, len(idx))})

        # 10) Return artefacts --------------------------------------------------------
        return {
            "mode": "time_march_ctmc",
            "OPEX_records": opex_df,
            "availability_summary": AvailabilitySummary(
                component_A=component_A,
                turbine_A=turbine_A,
                farm_A=farm_A,
                downtime_h=downtime_h,
            ),
            "availability_profile": availability_series,
            "activity_log": activity_log_df,
            "OpEx_breakdown": OpExBreakdown(
                fixed_OM_eur=fixed_om_total,
                CM_cost_eur=np.nan,
                PM_cost_eur=np.nan,
                transport_eur=np.nan,
                labour_eur=np.nan,
                spares_eur=np.nan,
            ),
        }


    def _deep_update_dict(self, dst: dict, src: dict) -> dict:
        """Recursive dict merge: dst <- src (in place)."""
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                self._deep_update_dict(dst[k], v)
            else:
                dst[k] = v
        return dst


    def _accumulate_opex_records(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Accumulate two OPEX_records frames safely.
        Requirement: both have ['timestamp','OM_payment'] at minimum.
        """
        if df_old is None or df_old.empty:
            return df_new.copy(deep=True)
        if df_new is None or df_new.empty:
            return df_old.copy(deep=True)

        # Align columns: keep any extra columns but sum OM_payment; others can be kept by last or ignored.
        # For now, only guarantee aggregation of OM_payment (valuation depends on that).
        keep_cols = ["timestamp", "OM_payment"]
        old2 = df_old[keep_cols].copy()
        new2 = df_new[keep_cols].copy()

        out = (
            pd.concat([old2, new2], ignore_index=True)
            .groupby("timestamp", as_index=False)["OM_payment"].sum()
            .sort_values("timestamp")
        )
        return out


    def _init_extras_container(self) -> dict:
        """
        Container for merged extras across many windows.
        - windows: store raw per-window extras for traceability and dashboards
        - full: optional aggregated views
        """
        return {
            "mode": None,          # last mode used
            "windows": [],

            # aggregated / stitched helpers (optional but useful)
            "availability_profile": None,     # concatenated profile
            "activity_log": None,             # concatenated log (if any)
            "OpEx_breakdown": None,           # summed breakdown
            "availability_summary": None,     # aggregated summary (weighted)
            "mode_cost_breakdown": None,      # concatenated list if present (analytic_ctmc dashboard)
        }


    def _merge_extras_inplace(self, dst: dict, new: dict) -> None:
        """
        Merge one window's extras into the accumulated container.

        Policy:
        - Always append raw window extras into dst["windows"].
        - Concatenate availability_profile and activity_log if present.
        - Sum OpEx_breakdown numeric fields across windows.
        - Aggregate availability_summary via weighted average by window duration (if available).
        - Concatenate mode_cost_breakdown lists (analytic_ctmc dashboard expects this). :contentReference[oaicite:1]{index=1}
        """

        if dst is None:
            raise ValueError("Extras container not initialized.")
        
        # propagate mode to top-level for backward compatibility
        m = new.get("mode", None)
        if m is not None:
            dst["mode"] = m
            if "modes" in dst and isinstance(dst["modes"], set):
                dst["modes"].add(m)


        # 1) store raw window extras
        dst["windows"].append(new)

        # 2) availability_profile
        ap = new.get("availability_profile", None)
        if ap is not None and isinstance(ap, pd.DataFrame) and not ap.empty:
            if dst["availability_profile"] is None:
                dst["availability_profile"] = ap.copy(deep=True)
            else:
                dst["availability_profile"] = (
                    pd.concat([dst["availability_profile"], ap], ignore_index=True)
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                )

        # 3) activity_log
        al = new.get("activity_log", None)
        if al is not None and isinstance(al, pd.DataFrame) and not al.empty:
            if dst["activity_log"] is None:
                dst["activity_log"] = al.copy(deep=True)
            else:
                dst["activity_log"] = pd.concat([dst["activity_log"], al], ignore_index=True)

        # 4) OpEx_breakdown: sum numeric attributes when present
        bd = new.get("OpEx_breakdown", None)
        if bd is not None:
            if dst["OpEx_breakdown"] is None:
                dst["OpEx_breakdown"] = bd
            else:
                # dataclass-like: add fields defensively
                for field in ["fixed_OM_eur", "CM_cost_eur", "PM_cost_eur", "transport_eur", "labour_eur", "spares_eur"]:
                    a = getattr(dst["OpEx_breakdown"], field, 0.0)
                    b = getattr(bd, field, 0.0)
                    try:
                        setattr(dst["OpEx_breakdown"], field, float(a) + float(b))
                    except Exception:
                        pass

        # 5) availability_summary: weighted by window duration
        summ = new.get("availability_summary", None)
        if summ is not None:

            # 1) determine weight
            weight_h = None

            w = new.get("window", None)
            if w is not None:
                w0, w1 = w
                weight_h = (pd.to_datetime(w1) - pd.to_datetime(w0)).total_seconds() / 3600.0
                if weight_h <= 0:
                    weight_h = None

            # optional but HIGHLY recommended: use ops_horizon if window is None
            if weight_h is None:
                ops = new.get("ops_horizon", None)
                if isinstance(ops, dict):
                    try:
                        weight_h = float(ops.get("T_h", None))
                        if not (weight_h and weight_h > 0):
                            weight_h = None
                    except Exception:
                        weight_h = None

            if dst["availability_summary"] is None:
                dst["availability_summary"] = {"_weighted_hours": 0.0, "_farmA_x_h": 0.0, "farm_A": None}
            agg = dst["availability_summary"]

            farm_A = getattr(summ, "farm_A", None)

            if weight_h is not None and farm_A is not None:
                # 2) weighted aggregation wins
                agg["_weighted_hours"] += weight_h
                agg["_farmA_x_h"] += float(farm_A) * float(weight_h)
                agg["farm_A"] = agg["_farmA_x_h"] / max(1e-12, agg["_weighted_hours"])
            else:
                # 3) no weight -> last-known fallback ALWAYS updates
                try:
                    agg["farm_A"] = float(farm_A)
                except Exception:
                    pass


            # (Optional extension: merge component_A/turbine_A/downtime_h similarly if you need)
            # For now we preserve detailed per-window summaries in dst["windows"].

        # 6) analytic_ctmc dashboard breakdown list
        mcb = new.get("mode_cost_breakdown", None)
        if mcb is not None:
            if dst["mode_cost_breakdown"] is None:
                dst["mode_cost_breakdown"] = list(mcb) if isinstance(mcb, list) else mcb
            else:
                if isinstance(dst["mode_cost_breakdown"], list) and isinstance(mcb, list):
                    dst["mode_cost_breakdown"].extend(mcb)


    # --------------------------- Extras to df ---------------------------#

    @staticmethod
    def _asdict_safe(x):
        if x is None:
            return None
        if is_dataclass(x):
            return asdict(x)
        return x

    def build_extras_tables(self) -> None:
        ex = getattr(self, "OPEX_records_extras", None)
        if not isinstance(ex, dict):
            self.opex_windows_df = pd.DataFrame()
            self.opex_breakdown_df = pd.DataFrame()
            self.opex_mode_cost_breakdown_df = pd.DataFrame()
            self.opex_component_cost_breakdown_df = pd.DataFrame()
            self.opex_availability_profile_df = pd.DataFrame()
            self.opex_activity_log_df = pd.DataFrame()
            self.opex_downtime_breakdown_df = pd.DataFrame()
            return

        windows = ex.get("windows", []) or []

        # --- 1) windows overview table ---
        rows = []
        for w in windows:
            # window parsing
            ww = w.get("window", None)
            if isinstance(ww, (list, tuple)) and len(ww) == 2:
                w0, w1 = ww
            else:
                w0, w1 = None, None

            summ = self._asdict_safe(w.get("availability_summary"))
            bd   = self._asdict_safe(w.get("OpEx_breakdown"))

            rows.append({
                "mode": w.get("mode"),
                "window_label": w.get("window_label"),
                "window_start": pd.to_datetime(w0) if w0 is not None else pd.NaT,
                "window_end":   pd.to_datetime(w1) if w1 is not None else pd.NaT,
                "farm_A": (summ or {}).get("farm_A", np.nan),
                "fixed_OM_eur": (bd or {}).get("fixed_OM_eur", np.nan),
                "CM_cost_eur":  (bd or {}).get("CM_cost_eur", np.nan),
                "PM_cost_eur":  (bd or {}).get("PM_cost_eur", np.nan),
                "transport_eur":(bd or {}).get("transport_eur", np.nan),
                "labour_eur":   (bd or {}).get("labour_eur", np.nan),
                "spares_eur":   (bd or {}).get("spares_eur", np.nan),
            })
        self.opex_windows_df = pd.DataFrame(rows)

        # --- 2) breakdown per window ---
        if not self.opex_windows_df.empty:
            self.opex_breakdown_df = self.opex_windows_df[
                ["mode","window_label","window_start","window_end",
                "fixed_OM_eur","CM_cost_eur","PM_cost_eur","transport_eur","labour_eur","spares_eur"]
            ].copy()
        else:
            self.opex_breakdown_df = pd.DataFrame()

        # --- 3) mode_cost_breakdown: rows tagged by window ---
        mcb_rows = []
        for w in windows:
            mcb = w.get("mode_cost_breakdown")
            if isinstance(mcb, list):
                for r in mcb:
                    if not isinstance(r, dict):
                        continue
                    rr = dict(r)
                    rr["window_label"] = w.get("window_label")
                    rr["mode"] = w.get("mode")
                    mcb_rows.append(rr)
        self.opex_mode_cost_breakdown_df = pd.DataFrame(mcb_rows)

        # --- 4) component_cost_breakdown: dict-of-dicts ---
        ccb_rows = []
        for w in windows:
            ccb = w.get("component_cost_breakdown")
            if isinstance(ccb, dict):
                for comp, d in ccb.items():
                    if not isinstance(d, dict):
                        continue
                    rr = dict(d)
                    rr["component"] = comp
                    rr["window_label"] = w.get("window_label")
                    rr["mode"] = w.get("mode")
                    ccb_rows.append(rr)
        self.opex_component_cost_breakdown_df = pd.DataFrame(ccb_rows)

        # --- 4b) downtime_breakdown: dict-of-dicts-of-floats (per component) ---
        db_rows = []
        for w in windows:
            db = w.get("downtime_breakdown")
            if not isinstance(db, dict):
                continue

            # window parsing (same pattern as above)
            ww = w.get("window", None)
            if isinstance(ww, (list, tuple)) and len(ww) == 2:
                w0, w1 = ww
            else:
                w0, w1 = None, None

            # pull maps (may be missing)
            Lh = db.get("logistics_h", {}) or {}
            Wh = db.get("weather_h", {}) or {}
            Rh = db.get("repair_h", {}) or {}

            Lf = db.get("logistics_fraction", {}) or {}
            Wf = db.get("weather_fraction", {}) or {}
            Rf = db.get("repair_fraction", {}) or {}

            # union of components present in any map
            comps = set()
            for m in (Lh, Wh, Rh, Lf, Wf, Rf):
                if isinstance(m, dict):
                    comps.update(m.keys())

            for comp in sorted(comps):
                db_rows.append({
                    "mode": w.get("mode"),
                    "window_label": w.get("window_label"),
                    "window_start": pd.to_datetime(w0) if w0 is not None else pd.NaT,
                    "window_end":   pd.to_datetime(w1) if w1 is not None else pd.NaT,
                    "component": comp,

                    "logistics_h": float(Lh.get(comp, np.nan)) if isinstance(Lh, dict) else np.nan,
                    "weather_h":   float(Wh.get(comp, np.nan)) if isinstance(Wh, dict) else np.nan,
                    "repair_h":    float(Rh.get(comp, np.nan)) if isinstance(Rh, dict) else np.nan,

                    "logistics_fraction": float(Lf.get(comp, np.nan)) if isinstance(Lf, dict) else np.nan,
                    "weather_fraction":   float(Wf.get(comp, np.nan)) if isinstance(Wf, dict) else np.nan,
                    "repair_fraction":    float(Rf.get(comp, np.nan)) if isinstance(Rf, dict) else np.nan,
                })

        self.opex_downtime_breakdown_df = pd.DataFrame(db_rows)





        # --- 5) stitched frames for saving ---
        ap = ex.get("availability_profile")
        al = ex.get("activity_log")
        self.opex_availability_profile_df = ap.copy(deep=True) if isinstance(ap, pd.DataFrame) else pd.DataFrame()
        self.opex_activity_log_df = al.copy(deep=True) if isinstance(al, pd.DataFrame) else pd.DataFrame()




    # OPEX Dashboard
    def opex_dashboard(self, drop_zeros: bool = True, show: bool = True):
        """
        OPEX dashboard (CTMC only).

        Preserves legacy layout:
        - KPI panel
        - Sunburst: cost breakdown
        - Sunburst: interventions breakdown

        Aggregates over ALL windows by re-reading extras['windows'][i]['mode_cost_breakdown']
        and summing on (component, task_type, mode_id).
        """

        # -----------------------------
        # 0) Guards: CTMC-only
        # -----------------------------
        extras = getattr(self, "OPEX_records_extras", None)
        if not isinstance(extras, dict):
            raise ValueError("OPEX dashboard unavailable: missing OPEX_records_extras (run OPEX first).")

        windows = extras.get("windows", None)
        if not isinstance(windows, list) or len(windows) == 0:
            raise ValueError("OPEX dashboard unavailable: extras['windows'] is empty. Run CTMC windows first.")

        # This dashboard is only meaningful if we have CTMC breakdown rows
        has_any_ctmc_breakdown = any(
            isinstance(w.get("mode_cost_breakdown", None), list) and len(w["mode_cost_breakdown"]) > 0
            for w in windows
        )
        if not has_any_ctmc_breakdown:
            raise ValueError(
                "OPEX dashboard is available for CTMC (analytic_ctmc / time_march_ctmc) only: "
                "no per-window 'mode_cost_breakdown' found."
            )

        # -----------------------------
        # 1) Total OPEX (use accumulated records)
        # -----------------------------
        df_cash = getattr(self, "OPEX_records", None)
        if not isinstance(df_cash, pd.DataFrame) or df_cash.empty or "OM_payment" not in df_cash.columns:
            raise ValueError("Missing self.OPEX_records with column 'OM_payment' (run OPEX first).")

        df_cash2 = df_cash.copy()
        if "timestamp" in df_cash2.columns:
            df_cash2["timestamp"] = pd.to_datetime(df_cash2["timestamp"])
        df_cash2["OM_payment"] = pd.to_numeric(df_cash2["OM_payment"], errors="coerce").fillna(0.0)

        # Convention in your codebase: OM_payment is negative outflow
        total_opex_eur = float(-df_cash2["OM_payment"].sum())

        # Weighted availability (your merge helper already aggregates this)
        farm_A = np.nan
        av = extras.get("availability_summary", None)
        if isinstance(av, dict):
            try:
                farm_A = float(av.get("farm_A", np.nan))
            except Exception:
                farm_A = np.nan

        # -----------------------------
        # 2) Rebuild & aggregate CTMC mode breakdowns across ALL windows
        # -----------------------------
        rows = []
        for w in windows:
            mcb = w.get("mode_cost_breakdown", None)
            if not isinstance(mcb, list):
                continue
            for r in mcb:
                if isinstance(r, dict):
                    rows.append(dict(r))

        df_modes = pd.DataFrame(rows)
        if df_modes.empty:
            raise ValueError("No CTMC breakdown rows found after scanning windows[].")

        # Ensure required fields exist
        required = [
            "component", "mode_id", "task_type",
            "N_interventions",
            "transport_eur", "labour_eur", "spares_eur", "fixed_OM_eur", "total_eur",
        ]
        for c in required:
            if c not in df_modes.columns:
                df_modes[c] = 0.0

        # Coerce numerics
        num_cols = ["N_interventions", "transport_eur", "labour_eur", "spares_eur", "fixed_OM_eur", "total_eur"]
        df_modes[num_cols] = df_modes[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Optional: drop zeros to keep plots clean
        if drop_zeros:
            df_modes = df_modes.loc[
                (df_modes["total_eur"] != 0.0) | (df_modes["N_interventions"] != 0.0)
            ].copy()

        # Canonical aggregation key across windows
        df_agg = (
            df_modes
            .groupby(["component", "task_type", "mode_id"], dropna=False, as_index=False)[num_cols]
            .sum()
        )

        # -----------------------------
        # 3) Build Sunburst 1: Costs (stable IDs)
        # Hierarchy:
        #   Root
        #     component
        #       task_type
        #         mode_id
        #           transport/labour/spares/fixed  (optional leaf breakdown)
        # -----------------------------
        # If you want the exact “category breakdown rings” like before, we include category leaves.
        categories = [
            ("transport_eur", "Transport"),
            ("labour_eur", "Labour"),
            ("spares_eur", "Spares"),
            ("fixed_OM_eur", "Fixed OM"),
        ]

        ids_cost, parents_cost, labels_cost, values_cost = [], [], [], []

        root_id = "cost_root"
        ids_cost.append(root_id)
        parents_cost.append("")
        labels_cost.append("OPEX Costs")
        values_cost.append(float(df_agg["total_eur"].sum()))

        # Build nodes
        for comp, dfc in df_agg.groupby("component", dropna=False):
            comp_label = str(comp)
            comp_id = f"{root_id}|comp:{comp_label}"
            ids_cost.append(comp_id)
            parents_cost.append(root_id)
            labels_cost.append(comp_label)
            values_cost.append(float(dfc["total_eur"].sum()))

            for task, dft in dfc.groupby("task_type", dropna=False):
                task_label = str(task)
                task_id = f"{comp_id}|task:{task_label}"
                ids_cost.append(task_id)
                parents_cost.append(comp_id)
                labels_cost.append(task_label)
                values_cost.append(float(dft["total_eur"].sum()))

                for mode, dfm in dft.groupby("mode_id", dropna=False):
                    mode_label = str(mode)
                    # IMPORTANT: include task in the ID to avoid collisions
                    mode_id = f"{task_id}|mode:{mode_label}"
                    ids_cost.append(mode_id)
                    parents_cost.append(task_id)
                    labels_cost.append(f"Mode {mode_label}")
                    values_cost.append(float(dfm["total_eur"].sum()))

                    # Category leaves under each mode
                    # (Sum in case dfm has multiple rows, though after groupby it should be 1 row)
                    for col, lab in categories:
                        v = float(dfm[col].sum())
                        if drop_zeros and v == 0.0:
                            continue
                        leaf_id = f"{mode_id}|cat:{col}"
                        ids_cost.append(leaf_id)
                        parents_cost.append(mode_id)
                        labels_cost.append(lab)
                        values_cost.append(v)

        fig_cost = go.Figure(go.Sunburst(
            ids=ids_cost,
            labels=labels_cost,
            parents=parents_cost,
            values=values_cost,
            branchvalues="total",
            hovertemplate="%{label}<br>%{value:,.0f} €<extra></extra>",
        ))
        fig_cost.update_layout(
            title="OPEX Cost Breakdown (CTMC aggregated)",
            height=650
        )

        # -----------------------------
        # 4) Build Sunburst 2: Interventions (stable IDs)
        # Hierarchy:
        #   Root
        #     component
        #       task_type
        #         mode_id
        # -----------------------------
        ids_int, parents_int, labels_int, values_int = [], [], [], []
        root2_id = "int_root"
        ids_int.append(root2_id)
        parents_int.append("")
        labels_int.append("Interventions")
        values_int.append(float(df_agg["N_interventions"].sum()))

        for comp, dfc in df_agg.groupby("component", dropna=False):
            comp_label = str(comp)
            comp_id = f"{root2_id}|comp:{comp_label}"
            ids_int.append(comp_id)
            parents_int.append(root2_id)
            labels_int.append(comp_label)
            values_int.append(float(dfc["N_interventions"].sum()))

            for task, dft in dfc.groupby("task_type", dropna=False):
                task_label = str(task)
                task_id = f"{comp_id}|task:{task_label}"
                ids_int.append(task_id)
                parents_int.append(comp_id)
                labels_int.append(task_label)
                values_int.append(float(dft["N_interventions"].sum()))

                for mode, dfm in dft.groupby("mode_id", dropna=False):
                    mode_label = str(mode)
                    mode_id = f"{task_id}|mode:{mode_label}"
                    ids_int.append(mode_id)
                    parents_int.append(task_id)
                    labels_int.append(f"Mode {mode_label}")
                    values_int.append(float(dfm["N_interventions"].sum()))

        fig_int = go.Figure(go.Sunburst(
            ids=ids_int,
            labels=labels_int,
            parents=parents_int,
            values=values_int,
            branchvalues="total",
            hovertemplate="%{label}<br>%{value:,.2f}<extra></extra>",
        ))
        fig_int.update_layout(
            title="Interventions Breakdown (CTMC aggregated)",
            height=650
        )

        # -----------------------------
        # 5) KPI panel (simple & robust)
        # -----------------------------
        # If your old dashboard had more KPIs (€/MWh etc.), you can add them here,
        # but this keeps it stable and strictly “aggregated over windows”.
        kpi_fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.25
        )

        kpi_fig.add_trace(go.Indicator(
            mode="number",
            value=total_opex_eur / 1e6,
            title={"text": "Total OPEX (all windows)"},
            number={"valueformat": ",.2f", "suffix": " M€"},
        ), row=1, col=1)

        kpi_fig.add_trace(go.Indicator(
            mode="number",
            value=float(df_agg["N_interventions"].sum()),
            title={"text": "Total interventions (all windows)"},
            number={"valueformat": ",.0f"},
        ), row=1, col=2)

        kpi_fig.add_trace(go.Indicator(
            mode="number",
            value=float(100.0 * farm_A) if np.isfinite(farm_A) else 0.0,
            title={"text": "Farm availability (weighted)"},
            number={"valueformat": ".1f", "suffix": " %"},
        ), row=2, col=1)

        kpi_fig.add_trace(go.Indicator(
            mode="number",
            value=float(df_agg["total_eur"].sum()) / 1e6,
            title={"text": "Total CTMC breakdown cost"},
            number={"valueformat": ",.2f", "suffix": " M€"},
        ), row=2, col=2)

        kpi_fig.update_layout(
            title="OPEX KPIs CTMC",
            height=420,
            showlegend=False
        )

        if show:
            kpi_fig.show()
            fig_cost.show()
            fig_int.show()

        return {
            "kpis": kpi_fig,
            "sunburst_cost": fig_cost,
            "sunburst_interventions": fig_int,
            "df_ctmc_breakdown_agg": df_agg,
        }


def load_OMData(config):
    """
    Loads wind farm input parameters from the configuration file.
    """
    OM_inputs = {}
    if hasattr(config, 'Opex_inputFiles'):
        for identifier, file_name in config.Opex_inputFiles.items():
            OM_inputs[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            OM_inputs[identifier] = process_duration_fields(OM_inputs[identifier])
    return OM_inputs
