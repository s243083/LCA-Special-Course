
"""
LTE.py — Lifetime Extension module (draft)
=========================================

Design goals
------------
- Runs AFTER WindFarm has produced response time series (e.g., power_records).
- Extends WindFarm response time series to a new ops end (baseline end + extension).
- Applies IEC AEP haircut as a multiplier on the *extension* slice of the production time series.
- Prepares OPEX analytic CTMC mean-shift overrides for the extension period only.
- Produces LTE-specific cost records (one-offs etc.) to be merged into OPEX records downstream.
- Does NOT run OPEX itself. The environment/orchestrator should:
    1) run WindFarm,
    2) run LTE.apply(),
    3) run OPEX twice (baseline window, extension window) using the mean-shift regime from LTE,
    4) stitch power_records and OPEX_records, and merge LTE.extra_cost_records.

This is intentionally conservative: it avoids double-applying availability multipliers because
the OPEX analytic CTMC path mutates env.windFarm.power_records in-place.

Assumptions
-----------
- env.windFarm.power_records is a pandas DataFrame with:
    - a 'timestamp' column (datetime-like)
    - one or more numeric production columns (e.g., 'Total_Production' in MW)
  (WindFarm.create_fixed_power_timeseries() matches this.)
- env.config contains at least:
    - Project_StartDate (datetime or string parseable by pandas)
    - WF_OperationsStart_h (int hours from Project_StartDate)
    - WF_OperationsEnd_h (int hours from Project_StartDate)
    - resolution (string accepted by pandas date_range / resample, e.g., '1h')
- Orchestrator will overwrite WF_OperationsStart_h/End_h for windowed OPEX runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Callable
import numpy as np
import pandas as pd
import math

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import get_input_parameter, repeat_timeseries_to_duration, apply_overrides


def _to_timestamp(dt: Any) -> pd.Timestamp:
    if isinstance(dt, pd.Timestamp):
        return dt
    return pd.to_datetime(dt)


def _hours_to_timedelta(h: int) -> pd.Timedelta:
    return pd.to_timedelta(int(h), unit="h")


def _get_numeric_cols(df: pd.DataFrame, exclude: Tuple[str, ...] = ("timestamp",)) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def repeat_timeseries_to_end(
    df: pd.DataFrame,
    timestamp_col: str,
    target_end_ts: pd.Timestamp,
    strategy: str = "repeat_full",
) -> pd.DataFrame:
    """
    Extend df to target_end_ts by repeating its own temporal pattern.

    Requires df to contain at least two distinct timestamps.
    """
    if df is None or df.empty:
        raise ValueError("Cannot extend empty time series.")

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    ts = pd.to_datetime(df[timestamp_col])

    if ts.nunique() < 2:
        raise ValueError(
            "repeat_timeseries_to_end requires at least two timestamps "
            "to infer time spacing. Got a single-timestep series."
        )

    t0 = ts.iloc[0]
    t1 = ts.iloc[-1]

    if target_end_ts <= t1:
        return df

    # select base pattern
    if strategy == "repeat_last_year":
        window_start = t1 - pd.Timedelta(days=365)
        base = df[df[timestamp_col] >= window_start].copy()
        if base[timestamp_col].nunique() < 2:
            raise ValueError(
                "repeat_last_year strategy requires at least two timestamps "
                "in the last-year slice."
            )
    elif strategy == "hold_last":
        raise ValueError(
            "hold_last strategy is no longer supported. "
            "Provide a valid time series instead."
        )
    else:
        base = df.copy()

    base = base.sort_values(timestamp_col).reset_index(drop=True)

    base_t0 = pd.to_datetime(base[timestamp_col].iloc[0])
    base_t1 = pd.to_datetime(base[timestamp_col].iloc[-1])
    base_duration = base_t1 - base_t0

    if base_duration <= pd.Timedelta(0):
        raise ValueError(
            "Base pattern has zero duration after preprocessing; "
            "cannot infer repeat period."
        )

    out = df.copy()
    last_ts = t1

    while last_ts < target_end_ts:
        shift = (last_ts - base_t0) + pd.to_timedelta(1, unit="ns")
        block = base.copy()
        block[timestamp_col] = pd.to_datetime(block[timestamp_col]) + shift
        block = block[block[timestamp_col] > last_ts]

        if block.empty:
            raise RuntimeError("Failed to generate non-overlapping extension block.")

        out = pd.concat([out, block], ignore_index=True)
        last_ts = pd.to_datetime(out[timestamp_col].iloc[-1])

    return out[out[timestamp_col] <= target_end_ts].reset_index(drop=True)



def sample_truncated_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> float:
    x = mu if sigma <= 0 else float(rng.normal(mu, sigma))
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return float(x)


@dataclass
class LTEConfig:
    enable_lte: bool = False
    extension_h: int = 0

    # Time-series extension
    tail_strategy: str = "repeat_full"
    timestamp_col: str = "timestamp"

    # AEP haircut (applied as multiplier to production columns during extension window)
    aep_mu: float = 0.0
    aep_sigma: float = 0.0
    aep_min: float = -0.30
    aep_max: float = 0.0

    # OPEX mean shift overrides (analytic_ctmc.mean_shift.*) for extension regime
    lambda_factor: float = 1.0
    mttr_factor: float = 1.0
    mttwL_factor: float = 1.0
    tau_factor: float = 1.0
    p_access_factor: float = 1.0
    p_access_hourly: Optional[float] = None

    # Uncertainty (optional)
    apply_uncertainty: Optional[bool] = None
    lamda_sigma: Optional[float] = None
    mttr_sigma: Optional[float] = None
    mttwL_sigma: Optional[float] = None

    # LTE costs (one-offs at extension start, EUR)
    cost_prelim_analysis_eur: float = 0.0
    cost_inspection_eur: float = 0.0
    cost_detailed_analysis_eur: float = 0.0

    # Refurb uplift: allow either legacy scalar or structured dist spec (sampled in apply)
    refurb_uplift_eur: float = 0.0                 # legacy/simple scalar default
    refurb_uplift_spec: Optional[dict] = None       # e.g. {"dist":"normal_trunc", ...}

    # How to timestamp LTE start costs: "at_base_end" or "first_ext_month"
    lte_start_cost_timing: str = "at_base_end"


class LifetimeExtension:
    """
    Lifetime Extension module.

    Usage pattern in env runner (conceptual):
      - windFarm.start()   # produces power_records to baseline end
      - lte = LifetimeExtension(env)
      - lte_out = lte.apply()  # extends power_records, updates config.WF_OperationsEnd_h, returns regime+costs
      - baseline OPEX run (config window = baseline)
      - extension OPEX run  (config window = extension, apply lte_out["opex_mean_shift_overrides"])
      - stitch / merge
    """
    def __init__(self, env: Any, cfg: Optional["LTEConfig"] = None):
        """
        LTE parameters are read from an input file (YAML) via load_LTEData(config),
        not directly from config. Scenario overrides may then modify the instance.

        Notes:
        - `cfg` can still be provided explicitly for testing; if provided it wins.
        - Overrides are applied *after* YAML load to support scenario variations.
        """
        self.env = env
        self.config = env.config

        # ---- Load LTE inputs from YAML ----
        # Expected to return a parsed dict-like structure
        self.lte_input = load_LTEData(self.config)
        self.lte_input = get_input_parameter(self.lte_input, "LTE")

        # ---- Enable flag from YAML ----
        self.lte_enabled = bool(get_input_parameter(self.lte_input, "LTE", "apply_lte"))

        
        # ---- Apply scenario overrides ----
        apply_overrides(self, getattr(self.config, "LTE_overrides", {}))


        # ---- Build LTEConfig from YAML (unless explicitly provided) ----
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = self._cfg_from_lte_input(self.lte_input)

        # ---- Outputs / state ----
        self.base_end_h: Optional[int] = None
        self.ext_end_h: Optional[int] = None
        self.aep_haircut: float = 0.0
        self.opex_mean_shift_overrides: Dict[str, Any] = {}
        self.cost_records: Optional[pd.DataFrame] = None

        
    def _cfg_from_lte_input(self, lte_input: dict) -> "LTEConfig":
        """
        Map YAML -> LTEConfig.
        Keep this function small and explicit; it’s the contract between YAML and the module.
        """
        L = lte_input.get("LTE", {}) if isinstance(lte_input, dict) else {}

        enable_lte = bool(L.get("apply_lte", False))
        extension_h = int(L.get("extension_h", 0))

        tail_strategy = str(L.get("tail_strategy", "repeat_full"))
        timestamp_col = str(L.get("timestamp_col", "timestamp"))

        aep = L.get("aep_haircut", {}) or {}
        aep_mu = float(aep.get("mu", 0.0))
        aep_sigma = float(aep.get("sigma", 0.0))
        aep_min = float(aep.get("min", -0.30))
        aep_max = float(aep.get("max", 0.0))

        opex = L.get("opex_extension", {}) or {}
        analytic = opex.get("analytic_ctmc", {}) or {}
        ms = analytic.get("mean_shift", {}) or {}

        lambda_factor = float(ms.get("lambda_factor", 1.0))
        mttr_factor = float(ms.get("mttr_factor", 1.0))
        mttwL_factor = float(ms.get("mttwL_factor", 1.0))
        tau_factor = float(ms.get("tau_factor", 1.0))
        p_access_factor = float(ms.get("p_access_factor", 1.0))

        p_access_hourly = analytic.get("p_access_hourly", None)
        if p_access_hourly is not None:
            p_access_hourly = float(p_access_hourly)

        unc = analytic.get("uncertainty", {}) or {}
        apply_uncertainty = unc.get("flag_apply", None)
        lamda_sigma = unc.get("lamda_sigma", None)
        mttr_sigma = unc.get("mttr_sigma", None)
        mttwL_sigma = unc.get("mttwL_sigma", None)

        # Costs block
        costs = L.get("costs", {}) or {}
        cost_prelim = float(costs.get("cost_prelim_analysis_eur", 0.0))
        cost_insp = float(costs.get("cost_inspection_eur", 0.0))
        cost_det = float(costs.get("cost_detailed_analysis_eur", 0.0))
        timing = str(costs.get("lte_start_cost_timing", "at_base_end"))

        # Refurb uplift inputs (stored, sampled later in apply)
        refurb_uplift_eur = float(costs.get("refurb_uplift_eur", 0.0) or 0.0)
        refurb_uplift_spec = costs.get("refurb_uplift", None)
        if not isinstance(refurb_uplift_spec, dict):
            # allow scalar refurb_uplift as shorthand, but normalize to dict None + scalar
            refurb_uplift_spec = None

        return LTEConfig(
            enable_lte=enable_lte,
            extension_h=extension_h,
            tail_strategy=tail_strategy,
            timestamp_col=timestamp_col,
            aep_mu=aep_mu,
            aep_sigma=aep_sigma,
            aep_min=aep_min,
            aep_max=aep_max,
            lambda_factor=lambda_factor,
            mttr_factor=mttr_factor,
            mttwL_factor=mttwL_factor,
            tau_factor=tau_factor,
            p_access_factor=p_access_factor,
            p_access_hourly=p_access_hourly,
            apply_uncertainty=apply_uncertainty,
            lamda_sigma=lamda_sigma,
            mttr_sigma=mttr_sigma,
            mttwL_sigma=mttwL_sigma,
            cost_prelim_analysis_eur=cost_prelim,
            cost_inspection_eur=cost_insp,
            cost_detailed_analysis_eur=cost_det,
            refurb_uplift_eur=refurb_uplift_eur,
            refurb_uplift_spec=refurb_uplift_spec,
            lte_start_cost_timing=timing,
        )


    def _rng(self) -> np.random.Generator:
        c = getattr(self.env, "config", None)
        seed = getattr(c, "seed", None) if c is not None else None
        if seed is None:
            seed = getattr(self.env, "seed", 12345)
        return np.random.default_rng(int(seed))

    def _project_start_ts(self) -> pd.Timestamp:
        c = getattr(self.env, "config", None)
        if c is None:
            raise AttributeError("env.config is required for LTE.")
        return _to_timestamp(getattr(c, "Project_StartDate"))

    def apply(self) -> None:
        """
        Apply LTE transformations to env:
        - extend config.WF_OperationsEnd_h
        - extend windFarm.power_records
        - apply AEP haircut to extension slice
        - prepare OPEX extension overrides
        - create LTE cost_records (separate rows/categories)

        This version reads inputs ONLY from self.cfg (LTEConfig),
        and keeps refurb uplift sampling inside apply().
        """
        cfg = self.cfg

        # No-op if disabled
        if (not cfg.enable_lte) or (cfg.extension_h <= 0):
            self.base_end_h = int(getattr(self.env.config, "WF_OperationsEnd_h"))
            self.ext_end_h = self.base_end_h

            self.aep_haircut = 0.0
            self.refurb_uplift_eur = 0.0
            self.opex_mean_shift_overrides = {}
            self.cost_records = pd.DataFrame(
                columns=[cfg.timestamp_col, "LTE_payment", "LTE_cost_category"]
            )
            return

        c = self.env.config
        rng = self._rng()

        base_start_h = int(getattr(c, "WF_OperationsStart_h"))
        base_end_h = int(getattr(c, "WF_OperationsEnd_h"))
        ext_end_h = base_end_h + int(cfg.extension_h)

        self.base_end_h = base_end_h
        self.ext_end_h = ext_end_h

        # update horizon
        setattr(c, "WF_OperationsEnd_h", int(ext_end_h))

        # SHIFT CAPEX DECOMMISSIONING TO END OF EXTENDED LIFE
        self._shift_capex_decommissioning(extension_h=int(cfg.extension_h))

        # extend power_records
        pr = getattr(self.env.windFarm, "power_records", None)
        if pr is None or not isinstance(pr, pd.DataFrame):
            raise AttributeError("env.windFarm.power_records must exist before LTE.apply().")
        if cfg.timestamp_col not in pr.columns:
            raise KeyError(f"power_records missing required timestamp column '{cfg.timestamp_col}'.")

        project_start = self._project_start_ts()
        target_end_ts = project_start + _hours_to_timedelta(ext_end_h)

        pr_ext = repeat_timeseries_to_end(
            pr,
            timestamp_col=cfg.timestamp_col,
            target_end_ts=target_end_ts,
            strategy=cfg.tail_strategy,
        )

        # extend electricity price records to new horizon (no haircut)
        if getattr(self.env, "MarketEnv", None) is not None:
            price_df = getattr(self.env.MarketEnv, "el_price_records", None)
            if price_df is not None and isinstance(price_df, pd.DataFrame) and (not price_df.empty):
                if cfg.timestamp_col not in price_df.columns:
                    raise KeyError(
                        f"el_price_records missing required timestamp column '{cfg.timestamp_col}'."
                    )

                price_ext = repeat_timeseries_to_end(
                    price_df,
                    timestamp_col=cfg.timestamp_col,
                    target_end_ts=target_end_ts,
                    strategy=cfg.tail_strategy,
                )
                self.env.MarketEnv.el_price_records = price_ext

        # AEP haircut sample + apply only in extension slice
        self.aep_haircut = sample_truncated_normal(
            rng, cfg.aep_mu, cfg.aep_sigma, cfg.aep_min, cfg.aep_max
        )
        self.ext_start_ts = project_start + _hours_to_timedelta(base_end_h)
        self.ext_end_ts = target_end_ts

        ext_mask = pr_ext[cfg.timestamp_col] >= self.ext_start_ts
        num_cols = _get_numeric_cols(pr_ext, exclude=(cfg.timestamp_col,))
        if not num_cols:
            raise ValueError("No numeric production columns found to apply AEP haircut.")
        pr_ext.loc[ext_mask, num_cols] = pr_ext.loc[ext_mask, num_cols] * (1.0 + float(self.aep_haircut))

        self.env.windFarm.power_records = pr_ext

        # Build OPEX overrides
        mean_shift = {
            "flag_apply": True,
            "lambda_factor": float(cfg.lambda_factor),
            "mttr_factor": float(cfg.mttr_factor),
            "mttwL_factor": float(cfg.mttwL_factor),
            "tau_factor": float(cfg.tau_factor),
            "p_access_factor": float(cfg.p_access_factor),
        }
        analytic_ctmc: Dict[str, Any] = {"mean_shift": mean_shift}

        if cfg.p_access_hourly is not None:
            analytic_ctmc["p_access_hourly"] = float(cfg.p_access_hourly)

        if (
            cfg.apply_uncertainty is not None
            or any(x is not None for x in [cfg.lamda_sigma, cfg.mttr_sigma, cfg.mttwL_sigma])
        ):
            uncertainty = {
                "flag_apply": bool(cfg.apply_uncertainty) if cfg.apply_uncertainty is not None else True
            }
            if cfg.lamda_sigma is not None:
                uncertainty["lamda_sigma"] = float(cfg.lamda_sigma)
            if cfg.mttr_sigma is not None:
                uncertainty["mttr_sigma"] = float(cfg.mttr_sigma)
            if cfg.mttwL_sigma is not None:
                uncertainty["mttwL_sigma"] = float(cfg.mttwL_sigma)
            analytic_ctmc["uncertainty"] = uncertainty

        self.opex_mean_shift_overrides = {"analytic_ctmc": analytic_ctmc}

        # -------------------------
        # LTE COST RECORDS (from cfg only; refurb uplift sampled here)
        # -------------------------
        self.refurb_uplift_eur = float(self._sample_refurb_uplift_from_cfg(rng))

        lte_ts = self._lte_cost_timestamp(self.ext_start_ts)
        rows = []

        n_turbines = self.env.windFarm.n_turbines
        PER_TURBINE_CATS = {"refurb_uplift"}  # shoud be defined in config

        def add_cost(cat: str, amount: float):
            amount = float(amount or 0.0)
            if cat in PER_TURBINE_CATS:
                amount *= n_turbines
            if amount != 0.0:
                rows.append(
                    {cfg.timestamp_col: lte_ts, "LTE_payment": -abs(amount), "LTE_cost_category": cat}
                )

        add_cost("prelim_analysis", cfg.cost_prelim_analysis_eur)
        add_cost("inspection", cfg.cost_inspection_eur)
        add_cost("detailed_analysis", cfg.cost_detailed_analysis_eur)
        add_cost("refurb_uplift", self.refurb_uplift_eur)

        self.cost_records = pd.DataFrame(
            rows, columns=[cfg.timestamp_col, "LTE_payment", "LTE_cost_category"]
        )
        return
    

    def _shift_capex_decommissioning(self, extension_h: int) -> None:
        """
        Shift realized CAPEX 'Decommissioning' costs to the end of the extended life
        by moving their timestamps forward by `extension_h` hours.
        """
        if extension_h <= 0:
            return

        capex = getattr(self.env, "capex", None)
        if capex is None:
            return

        df = getattr(capex, "cost_records", None)
        if df is None or df.empty:
            return

        mask = (
            df["phase_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("decommissioning")
        )

        if not mask.any():
            return

        df.loc[mask, "timestamp"] = (
            df.loc[mask, "timestamp"]
            + pd.Timedelta(hours=int(extension_h))
        )




    def _lte_cost_timestamp(self, ext_start_ts: pd.Timestamp) -> pd.Timestamp:
        """Decide timestamp for LTE one-off costs."""
        if self.cfg.lte_start_cost_timing == "first_ext_month":
            return pd.Timestamp(ext_start_ts).to_period("M").to_timestamp()
        return pd.Timestamp(ext_start_ts)


    def _sample_refurb_uplift_from_cfg(self, rng: np.random.Generator) -> float:
        """
        Sample refurbishment uplift using cfg inputs.
        Priority:
        1) cfg.refurb_uplift_spec (dict; supports fixed / normal_trunc)
        2) cfg.refurb_uplift_eur (legacy scalar)
        """
        cfg = self.cfg

        # If a spec dict exists, sample from it
        spec = getattr(cfg, "refurb_uplift_spec", None)
        if isinstance(spec, dict) and spec:
            dist = str(spec.get("dist", "fixed")).lower()

            if dist == "fixed":
                return float(spec.get("value", 0.0) or 0.0)

            if dist in ("normal_trunc", "trunc_normal", "truncated_normal"):
                mu = float(spec.get("mu", 0.0) or 0.0)
                sigma = float(spec.get("sigma", 0.0) or 0.0)
                lo = spec.get("min", None)
                hi = spec.get("max", None)
                lo = float(lo) if lo is not None else None
                hi = float(hi) if hi is not None else None
                return sample_truncated_normal(rng, mu, sigma, lo, hi)

            raise ValueError(f"Unsupported refurb_uplift dist: {dist}")

        # Fall back to legacy scalar
        return float(getattr(cfg, "refurb_uplift_eur", 0.0) or 0.0)


def load_LTEData(config):
    """
    Loads LTE input parameters from the configuration file.
    Returns a dict with LTE parameters.
    """
    lte_input = {}

    if hasattr(config, "LTE_inputFiles"):
        for identifier, file_name in config.LTE_inputFiles.items():
            lte_input[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            lte_input[identifier] = process_duration_fields(lte_input[identifier])

    return lte_input



##################### Backlog to be implemented #####################
def _resolve_factor(self, metrics_df: pd.DataFrame) -> float:
    """
    Read last value of metrics_column.
    If column is missing or value cannot be parsed, raise an error.
    """
    if self.metrics_column not in metrics_df.columns:
        raise KeyError(
            f"Required metrics column '{self.metrics_column}' not found in metrics_records."
        )

    try:
        val = float(metrics_df[self.metrics_column].iloc[-1])
    except Exception as e:
        raise ValueError(
            f"Failed to extract numeric value from '{self.metrics_column}' column."
        ) from e

    if not np.isfinite(val) or val <= 0:
        raise ValueError(
            f"Invalid lifetime extension factor '{val}' in column '{self.metrics_column}'."
        )

    return val

def _compute_total_hours_for_factor(self, power_df: pd.DataFrame, factor: float) -> int:
    """Compute ceil(hours(original_duration * factor))."""
    if power_df.empty:
        raise ValueError("power_records is empty; nothing to extend.")

    ts_col = "timestamp"
    if ts_col not in power_df.columns:
        raise KeyError(f"power_records must contain a '{ts_col}' column.")

    start = pd.to_datetime(power_df[ts_col].iloc[0])
    end = pd.to_datetime(power_df[ts_col].iloc[-1])
    original_duration = (end - start)
    if original_duration <= pd.Timedelta(0):
        raise ValueError("Non-positive original duration; cannot extend.")

    target_duration = original_duration * factor
    return max(int(np.ceil(target_duration.total_seconds() / 3600.0)), 1)