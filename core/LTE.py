import pandas as pd
import numpy as np
from typing import Any

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import get_input_parameter, repeat_timeseries_to_duration, apply_overrides


class LifetimeExtension:
    """
    Extends the wind farm power_records timeline according to a factor found in metrics_records.

    At init:
      - Loads LTE YAML.
      - Extracts the LTE flag with get_input_parameter(..., "LTE").
    At apply():
      - If LTE flag is False, do nothing.
      - If True, extend env.wind_farm.power_records by factor = last metrics_records['farm_mean'].
    """

    def __init__(self, env: Any):
        self.env = env
        self.config = env.config

        # Load inputs
        self.lte_input = load_LTEData(self.config)

        # Pull LTE flag directly
        self.lte_enabled = bool(get_input_parameter(self.lte_input, "LTE"))

        # Allow overrides if scenario provides them
        apply_overrides(self, getattr(self.config, "Lifetime_extension_overrides", {}))

        # Fixed configuration
        self.metrics_column: str = "('farm_mean', '')"
        self.update_windfarm_inplace: bool = True

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

    def apply(self) -> pd.DataFrame:
        """Extend env.wind_farm.power_records if LTE is enabled in config."""
        if not self.lte_enabled:
            return self.env.windFarm.power_records

        if self.env.windFarm.wf_metrics_records is None:
            factor = 1.0
        else:
            factor = self._resolve_factor(self.env.windFarm.wf_metrics_records)
        
        if factor <= 1.0 + 1e-12:
            return self.env.windFarm.power_records

        total_hours = self._compute_total_hours_for_factor(self.env.windFarm.power_records, factor)

        # extend power_records
        extended = repeat_timeseries_to_duration(
            self.env.windFarm.power_records,
            duration=f"{total_hours} hours",
            timestamp_col="timestamp",
            trim_to_duration=True,
        )

        if self.update_windfarm_inplace:
            self.env.windFarm.power_records = extended

        # extend price records
        self.env.MarketEnv.el_price_records = repeat_timeseries_to_duration(
            self.env.MarketEnv.el_price_records,
            duration=f"{total_hours} hours",
            timestamp_col="timestamp",
            trim_to_duration=True,
        )

        # extend OPEX

        self.env.opex.extend_opex_records(total_hours)
        
        # retime decommissioning costs
        self.re_time_decommissioning_costs()

        return None
    
    def re_time_decommissioning_costs(self) -> int:
        """
        Set the timestamp of all rows in env.capex.cost_records where phase_name == 'Decommissioning'
        to the new end-of-operation (last timestamp in env.windFarm.power_records).
        Returns the number of updated rows.
        """
        if not hasattr(self.env, "capex") or self.env.capex is None:
            raise AttributeError("env.capex is missing.")
        cr = self.env.capex.cost_records
        if cr is None or cr.empty:
            return 0
        required_cols = {"timestamp", "phase_name"}
        missing = required_cols.difference(cr.columns)
        if missing:
            raise ValueError(f"cost_records missing columns: {sorted(missing)}")

        pr = getattr(self.env.windFarm, "power_records", None)
        if pr is None or pr.empty or "timestamp" not in pr.columns:
            raise ValueError("power_records is empty or lacks a 'timestamp' column.")
        new_end_ts = pd.to_datetime(pr["timestamp"].iloc[-1])

        mask = cr["phase_name"].astype(str).str.strip().str.lower() == "decommissioning"
        updated = int(mask.sum())
        if updated:
            self.env.capex.cost_records.loc[mask, "timestamp"] = pd.Timestamp(new_end_ts)
            self.env.capex.cost_records.sort_values("timestamp", inplace=True, ignore_index=True)
        return updated



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
