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

        # Gamma parameters
        # numpy uses Gamma(shape=k, scale=theta)
        shape_k = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_shape_k"))
        scale_theta = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "gamma_scale_theta"))

        # Truncation/clipping (because Gamma is unbounded)
        c_min = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "curtailment_min") or 0.0)
        c_max = float(get_input_parameter(self.curt_input, "Curtailment", "reduceProduction", "curtailment_max") or 0.95)

        if shape_k <= 0 or scale_theta <= 0:
            raise ValueError("Gamma parameters must be > 0 (gamma_shape_k, gamma_scale_theta).")
        if not (0.0 <= c_min <= c_max):
            raise ValueError("Curtailment bounds must satisfy 0 <= curtailment_min <= curtailment_max.")

        df = pr.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        prod_cols = _get_numeric_cols(df, exclude=(timestamp_col,))
        if not prod_cols:
            raise ValueError("No numeric production columns found to apply curtailment.")

        # Month key: Period('M')
        month_key = df[timestamp_col].dt.to_period("M")

        # Sample one curtailment fraction per month
        months = pd.PeriodIndex(month_key.unique()).sort_values()

        records = []
        month_to_multiplier: Dict[pd.Period, float] = {}

        for m in months:
            c = float(rng.gamma(shape=shape_k, scale=scale_theta))
            c = float(np.clip(c, c_min, c_max))
            mult = 1.0 - c

            month_to_multiplier[m] = mult
            records.append(
                {
                    "month": str(m),  # store as string like '2025-01'
                    "curtailment_fraction": c,
                    "multiplier": mult,
                }
            )

        # Apply multiplier to each row based on its month
        multipliers = month_key.map(month_to_multiplier).astype(float).to_numpy()
        df.loc[:, prod_cols] = df.loc[:, prod_cols].multiply(multipliers, axis=0)

        # Write back mutated production
        self.env.windFarm.power_records = df

        # Store diagnostics
        self.curtailment_records = pd.DataFrame(records, columns=["month", "curtailment_fraction", "multiplier"])


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
