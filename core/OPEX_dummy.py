import numpy as np
import pandas as pd
from core.File_Handling import load_yaml, process_duration_fields


class OPEX: 

    def __init__(self, env):
        self.env = env  # Access to simulation environment
        self.OM_inputs = load_OMData(env.config)
        self.parameters = get_opex_parameter(self.OM_inputs, 'OM', 'DummyModel')

        self.availability = 0.93 # dummy for now

        

        self.OPEX_records = {}

        # Project start (used to convert "project_time_h" to timestamps)
        self.project_start = pd.to_datetime(
            self.env.config.Project_StartDate,
            format="%d.%m.%Y"
        )

    def calc_OPEX(self):
        """Calculate OPEX as a fraction of CAPEX and distribute across equal payments."""

        # Time window (in hours from project start)
        self.start_time = float(self.env.config.WF_OperationsStart_h)
        self.end_time   = float(self.env.config.WF_OperationsEnd_h)
        if self.end_time <= self.start_time:
            raise ValueError("WF_OperationsEnd_h must be greater than WF_OperationsStart_h.")

        # Parameters
        n_yearly_payments = int(self.parameters.get("n_yearly_payments", 12))
        capex_fraction = float(self.parameters.get("capex_fraction", 0.03))  # e.g., 3% of CAPEX

        # CAPEX total via accessor
        capex_df = self.env.capex.get_cost_dataframe()
        if capex_df.empty:
            total_capex = 0.0
        else:
            total_capex = float(pd.to_numeric(capex_df["cost"], errors="coerce").fillna(0).sum())

        # Total OPEX over entire operations window
        total_opex = capex_fraction * total_capex

        # Determine number of payments across full operations period
        total_hours = self.end_time - self.start_time
        total_years = total_hours / (365.25 * 24.0)
        n_periods = max(1, int(np.ceil(total_years * n_yearly_payments)))

        # Equal per-payment amount
        per_payment = total_opex / n_periods if n_periods > 0 else 0.0

        # Evenly spaced payment timestamps (in project hours)
        period_delta = total_hours / n_periods
        rows = []
        for i in range(n_periods):
            project_time_h = self.start_time + i * period_delta
            timestamp = self.project_start + pd.to_timedelta(project_time_h, unit="h")
            rows.append({"timestamp": timestamp, "OM_payment": per_payment})

        df = pd.DataFrame(rows)
        self.OPEX_records = df

        # apply availability on production # dummt for now

        if getattr(self.env.windFarm, "power_records", None) is not None and not self.env.windFarm.power_records.empty:
            prod_cols = [c for c in self.env.windFarm.power_records.columns if c != "timestamp"]
            self.env.windFarm.power_records[prod_cols] = (
                self.env.windFarm.power_records[prod_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                * self.availability
            )


        return df
    

    def extend_opex_records(self, target_total_hours: int) -> pd.DataFrame:
        """
        Extend OPEX_records so that payments continue with the same frequency and amount
        up to the given project duration in hours (from project start).

        Parameters
        ----------
        target_total_hours : int
            Total duration (in hours from project start) to which OPEX should be extended.
            Example: if project_start is 2020-01-01 00:00 and target_total_hours=175200,
            the target end timestamp is 2020-01-01 00:00 + 175200h.

        Returns
        -------
        pd.DataFrame
            Updated OPEX_records including any newly appended rows.
        """
        # Ensure we have baseline OPEX
        if getattr(self, "OPEX_records", None) is None or len(self.OPEX_records) == 0:
            # Build the initial schedule from config if missing
            self.calc_OPEX()

        df = self.OPEX_records.copy()
        if df.empty:
            return df  # nothing we can do

        if "timestamp" not in df.columns or "OM_payment" not in df.columns:
            raise KeyError("OPEX_records must contain 'timestamp' and 'OM_payment' columns.")

        # Ensure timestamps are datetime and sorted
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Determine payment cadence (use median delta for robustness)
        deltas = df["timestamp"].diff().dropna()
        if deltas.empty:
            raise ValueError("Cannot infer OPEX payment cadence from a single record.")

        period = deltas.median()
        if not isinstance(period, pd.Timedelta) or period <= pd.Timedelta(0):
            raise ValueError("Invalid inferred OPEX payment period.")

        # Determine per-payment amount (verify constancy within tolerance)
        per_payment_vals = pd.to_numeric(df["OM_payment"], errors="coerce").fillna(0.0)
        if not np.isfinite(per_payment_vals).all():
            raise ValueError("Non-finite values found in 'OM_payment'.")

        per_payment = float(per_payment_vals.iloc[-1])
        # Optional: check near-constancy (warn silently if drifting)
        if (per_payment_vals - per_payment_vals.iloc[0]).abs().max() > 1e-6:
            # If needed, you could log/print a warning instead of raising.
            per_payment = float(per_payment_vals.iloc[-1])

        # Compute target end timestamp
        target_end_ts = self.project_start + pd.to_timedelta(float(target_total_hours), unit="h")

        last_ts = df["timestamp"].iloc[-1]
        next_ts = last_ts + period

        # Append new rows up to target_end_ts (inclusive if exactly on boundary)
        new_rows = []
        while next_ts <= target_end_ts:
            new_rows.append({"timestamp": next_ts, "OM_payment": per_payment})
            next_ts += period

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Persist & return
        self.OPEX_records = df
        return None
    

    def _parse_components(self, turbine_dict):  
        components = []
        for comp_name, comp_data in turbine_dict.items():
            failures = comp_data.get('failures', {})
            for level, f_data in failures.items():
                components.append({
                    'name': comp_name,
                    'level': level,
                    'scale': f_data['scale'],  # MTTF
                    'time': f_data['time'],    # MTTR
                    'materials': f_data['materials'],
                    'equipment': f_data['service_equipment'],
                    'description': f_data['description'],
                    'replacement': f_data.get('replacement', False),
                    'xi_PM': 0.7,  # Default preventive threshold
                    'MTTW_CM': 12, # Placeholder for now
                    'MTTW_PM': 8,  # Placeholder for now
                    'MTTM': 6,     # Placeholder for now
                    'CM_cost': f_data['materials'],
                    'PM_cost': f_data['materials'] * 0.6  # Heuristic for PM
                })
        return components


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


def get_opex_parameter(opex_inputs, identifier, entry_name, *keys):
    try:
        entries = opex_inputs[identifier]['OM']
    except KeyError as e:
        raise KeyError(f"Missing expected key: {e}")

    for item in entries:
        if item.get('name') == entry_name:
            value = item.get('Parameters', {})
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    return None
                value = value[key]
            return value
    return None


