from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import apply_overrides, get_input_parameter


def _get_numeric_cols(df: pd.DataFrame, exclude: Tuple[str, ...] = ("timestamp",)) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


class Curtailment:
    """
    Curtailment module.

    Architecture goals (matching other modules):
    - Has access to env + env.config
    - Reads its own YAML input via config.Curtailment_inputFiles
    - Applies optional scenario overrides from config.Curtailment_overrides
    - Mutates env.windFarm.power_records in-place

    First version:
    - One mode: reduceProduction
      For each month in windFarm.power_records, sample a curtailment fraction from a Gamma distribution.
      Reduce all production values in that month by the sampled fraction.

    Interpretation:
    - We sample a *curtailment fraction* c ~ Gamma(k, theta) and then compute multiplier m = 1 - c.
    - Because Gamma is unbounded, c is truncated/clipped into [c_min, c_max] (defaults [0, 0.95]).
      This ensures the multiplier stays non-negative.

      Author: MG
    """

    def __init__(self, env: Any):
        self.env = env
        self.config = env.config

        # ---- Load Curtailment inputs from YAML ----
        self.curt_input = load_CurtailmentData(self.config)
        self.curt_input = get_input_parameter(self.curt_input,"CU")

        # ---- Apply scenario overrides (if provided) ----
        # This mirrors how WindFarm and LTE accept overrides:
        apply_overrides(self, getattr(self.config, "Curtailment_overrides", {}))

        # ---- Outputs/state ----
        self.curtailment_records: Optional[pd.DataFrame] = None

    def _rng(self) -> np.random.Generator:
        # follow the general approach used elsewhere: prefer config.seed, else env.seed, else default
        c = getattr(self.env, "config", None)
        seed = getattr(c, "seed", None) if c is not None else None
        if seed is None:
            seed = getattr(self.env, "seed", 12345)
        return np.random.default_rng(int(seed))

    def apply(self) -> None:
        """
        Apply curtailment according to YAML mode. Mutates env.windFarm.power_records in-place.
        """

        # -----------------------------
        # High-level bypass flag
        # -----------------------------
        apply_curtailment = bool(
            get_input_parameter(self.curt_input, "Curtailment", "apply_curtailment")
            if get_input_parameter(self.curt_input, "Curtailment", "apply_curtailment") is not None
            else False
        )

        if not apply_curtailment:
            # Explicitly record "no curtailment applied" and exit
            self.curtailment_records = pd.DataFrame(
                columns=["month", "curtailment_fraction", "multiplier"]
            )
            return

        mode = str(get_input_parameter(self.curt_input, "Curtailment", "mode") or "").strip()

        if mode == "" or mode.lower() == "none":
            self.curtailment_records = pd.DataFrame(columns=["month", "curtailment_fraction", "multiplier"])
            return

        if mode != "reduceProduction":
            raise ValueError(f"Curtailment mode '{mode}' not recognized.")

        self._apply_reduce_production()

    def _apply_reduce_production(self) -> None:
        rng = self._rng()

        pr = getattr(self.env.windFarm, "power_records", None)
        if pr is None or not isinstance(pr, pd.DataFrame) or pr.empty:
            raise AttributeError("Curtailment requires env.windFarm.power_records to exist and be non-empty.")

        timestamp_col = str(get_input_parameter(self.curt_input, "Curtailment", "timestamp_col") or "timestamp")
        if timestamp_col not in pr.columns:
            raise KeyError(f"power_records missing required timestamp column '{timestamp_col}'.")

        # -----------------------------
        # Epistemic uncertainty (one draw per simulation)
        # -----------------------------
        apply_epi = bool(
            get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "apply_epistemic_uncertainty") or False
        )

        if apply_epi:
            shape_bounds = get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_shape")
            scale_bounds = get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_scale")

            if not (isinstance(shape_bounds, (list, tuple)) and len(shape_bounds) == 2):
                raise ValueError("Curtailment.reduceProduction.gamma_shape must be a 2-element list [lower, upper].")
            if not (isinstance(scale_bounds, (list, tuple)) and len(scale_bounds) == 2):
                raise ValueError("Curtailment.reduceProduction.gamma_scale must be a 2-element list [lower, upper].")

            k_low, k_high = float(shape_bounds[0]), float(shape_bounds[1])
            th_low, th_high = float(scale_bounds[0]), float(scale_bounds[1])

            if k_low <= 0 or k_high <= 0 or k_low > k_high:
                raise ValueError("gamma_shape bounds must satisfy 0 < lower <= upper.")
            if th_low <= 0 or th_high <= 0 or th_low > th_high:
                raise ValueError("gamma_scale bounds must satisfy 0 < lower <= upper.")

            shape_k = float(rng.uniform(k_low, k_high))
            scale_theta = float(rng.uniform(th_low, th_high))
        else:
            # Existing fixed parameters
            shape_k = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_shape_k"))
            scale_theta = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_scale_theta"))

        # -----------------------------
        # Aleatory uncertainty (monthly draws)
        # -----------------------------
        apply_alea = bool(
            get_input_parameter(
                self.curt_input,
                "Curtailment",
                "reduceProduction",
                "apply_aleatory_uncertainty",
            ) or False
        )

        # Truncation/clipping (because Gamma is unbounded)
        c_min = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "curtailment_min") or 0.0)
        c_max = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "curtailment_max") or 0.95)

        if shape_k <= 0 or scale_theta <= 0:
            raise ValueError("Gamma parameters must be > 0 (gamma_shape_k/gamma_scale_theta or epistemic draws).")
        if not (0.0 <= c_min <= c_max):
            raise ValueError("Curtailment bounds must satisfy 0 <= curtailment_min <= curtailment_max.")

        df = pr.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        prod_cols = _get_numeric_cols(df, exclude=(timestamp_col,))
        if not prod_cols:
            raise ValueError("No numeric production columns found to apply curtailment.")

        month_key = df[timestamp_col].dt.to_period("M")
        months = pd.PeriodIndex(month_key.unique()).sort_values()

        records = []
        month_to_multiplier: Dict[pd.Period, float] = {}

        # Optional: store epistemic draw in diagnostics
        epi_meta = {
            "epistemic_shape_k": shape_k,
            "epistemic_scale_theta": scale_theta,
            "apply_epistemic_uncertainty": apply_epi,
            "apply_aleatory_uncertainty": apply_alea,
        }

        for m in months:
            if apply_alea:
                c = float(rng.gamma(shape=shape_k, scale=scale_theta))
            else:
                # Deterministic monthly curtailment = mean of Gamma (k*theta)
                c = float(shape_k * scale_theta)

            c = float(np.clip(c, c_min, c_max))
            mult = 1.0 - c

            month_to_multiplier[m] = mult
            rec = {
                "month": str(m),
                "curtailment_fraction": c,
                "multiplier": mult,
                **epi_meta,
            }
            records.append(rec)

        multipliers = month_key.map(month_to_multiplier).astype(float).to_numpy()
        df.loc[:, prod_cols] = df.loc[:, prod_cols].multiply(multipliers, axis=0)

        self.env.windFarm.power_records = df
        self.curtailment_records = pd.DataFrame(records)

def load_CurtailmentData(config) -> dict:
    """
    Loads Curtailment input parameters from YAML files specified in config.Curtailment_inputFiles.
    Returns a dict keyed by identifier, matching the pattern used by WindFarm/LTE loaders.
    """
    curt_input: dict = {}

    if hasattr(config, "Curtailment_inputFiles"):
        for identifier, file_name in config.Curtailment_inputFiles.items():
            curt_input[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            curt_input[identifier] = process_duration_fields(curt_input[identifier])

    return curt_input
