import pandas as pd
import ast
from pathlib import Path

from core.File_Handling import load_yaml, process_duration_fields
from scipy.stats import weibull_min
import numpy as np
from core.WF_Controller import WF_Controller
from core.utils import apply_overrides , get_input_parameter, repeat_timeseries, gap_fill_timeseries, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration

class WindFarm:
    """
    Represents a wind farm in the simulation environment. Manages turbine data, layout,
    and accesses environmental parameters from MetEnvironment.
    
    Parameters 
     ----------
    env : ValueWindEnv
        The main simulation environment, providing access to MetEnvironment.
    """
    def __init__(self, env):
        self.env = env  # Access to the main simulation environment
        self.wind_farm_input = load_windfarmData(self.env.config)
        self.wind_farm_input = get_input_parameter(self.wind_farm_input, 'WF')
        self.config = env.config

        self.wf_controller = WF_Controller(self.env)

        # Set start and end times from wind_farm_data for the 'WF' identifier
        self.start_time = self.env.config.WF_OperationsStart_h
        self.end_time = self.env.config.WF_OperationsEnd_h
        
        self.power_records = pd.DataFrame()  # Initialize power records DataFrame
        self.wf_metrics_records = pd.DataFrame()  # Initialize metrics records DataFrame

        self.external_response_path = get_input_parameter(self.wind_farm_input,'WF_external_response_path')
        

        # Apply Scenario overrides if provided
        apply_overrides(self, getattr(self.config, "WindFarm_overrides", {}))

    def start(self):
        """Starts the wind farm simulation process."""
        # get mode from config
        mode = get_input_parameter(self.wind_farm_input, 'WindFarm', 'mode')

        if mode == "external":
        # load external response (eg.from VP framework)
            self.load_external()
        elif mode== "PyWake":
            self.power_records = self.calculate_windfarm_response
        elif mode == "fixed":
            self.power_records = self.create_fixed_power_timeseries()
        else:
            raise ValueError(f"WindFarm Response mode '{mode}' not recognized.")


    def load_external(self):
        import ast
        import pandas as pd

        # reference response
        path = self.external_response_path

        reference_df = pd.read_parquet(
            f"{path}/turbine_timeseries_inst.parquet"
        )
        reference_df.columns = [ast.literal_eval(col) for col in reference_df.columns]
        #print(reference_df)

        # keep timestamp + energy cols
        energy_columns = [col for col in reference_df.columns if col[0] == "Energy"]
        timestamp_column = [("timestamp", "")]
        columns_to_keep = timestamp_column + energy_columns

        power_ref_df = reference_df.loc[:, columns_to_keep]

        # add aggregated column
        power_ref_df[("Total_Production", "")] = power_ref_df[energy_columns].sum(axis=1)
        
        # keep only timestamp and Total_Production
        power_ref_df = power_ref_df.loc[:, [("timestamp", ""), ("Total_Production", "")]]

        # rename columns to remove MultiIndex
        power_ref_df.columns = ["timestamp", "Total_Production"]

        # 1) Remove gaps and rebuild a continuous timeline (no interpolation)
        power_ref_df = remove_gaps_rebuild_timestamps(
            power_ref_df,
            timestamp_col="timestamp",
            freq=None,  # infer base step from data
            sort=True,
        )

        # 2) Repeat until desired total duration is reached
        power_ref_df = repeat_timeseries_to_duration(
            power_ref_df,
            duration="20 years",     # extend horizon to 20 years
            timestamp_col="timestamp",
            trim_to_duration=True,
        )

        # Restore MultiIndex timestamp column name and put it first
        #working = working.rename(columns={"timestamp": ("timestamp", "")})
        # Reorder columns to keep timestamp first
        #cols = [("timestamp", "")] + [c for c in working.columns if c != ("timestamp", "")]
        #working = working.loc[:, cols]

        self.power_records = power_ref_df


        # -------------- load metrics data --------------------

        path = self.external_response_path
        file_path = Path(path) / "relative_farm_lifetime.parquet"

        if file_path.exists():
            metrics_df = pd.read_parquet(file_path)
            self.wf_metrics_records = metrics_df
        else:
            self.wf_metrics_records = None

    def calculate_windfarm_response(self,
                                    mode: str = "greedy_timeseries",
                                    wind_ts_file: str = "ResponseFramework/data/timeseries/HKNB_timeseries_full_filled_no_gaps.csv",
                                    dist_file: str = "ResponseFramework/distributions/HKNB_Weib_WSWD_dist_2deg.csv",
                                    use_pywake_farm: bool = True,
                                    wsp_cut_in: float = 3.0,
                                    wsp_cut_off: float = 25.0):
        rf = self.env.response_framework

        rf.parameter_initialization(
            baseline_mode=mode,                 # "greedy_timeseries" or "greedy_distribution"
            wind_price_TS_file=wind_ts_file,   # used for timeseries mode
            distribution_file=dist_file,       # used for distribution mode
            use_pywake_farm=use_pywake_farm,   # True -> HKN farm, False -> single IEA22
            input_resolution="10min",          # kept for simulate_block signature
            wsp_cut_in=wsp_cut_in,
            wsp_cut_off=wsp_cut_off,
        )

        rf.execution()

        
        rf.power_timeseries = rf.power_timeseries[["timestamp", "FarmPower"]].rename(
            columns={"FarmPower": "Total_Production"}
        )

        # Return whichever result is populated
        return rf.power_timeseries if mode == "greedy_timeseries" else rf.power_distribution

    def create_fixed_power_timeseries(self):
        """
        Create a fixed power time series from WF_OperationsStart_h to WF_OperationsEnd_h
        using the configured resolution.

        Returns
        -------
        pd.DataFrame
            Columns:
                - 'timestamp'
                - 'Total_Production'  (constant fixed power in MW)
        """
        cfg = self.config  # same as self.env.config

        # Resolution directly from input, e.g. "10min", "1h", "1d"
        freq = get_input_parameter(self.wind_farm_input, 'WindFarm', 'fixed', 'resolution')

        # Operation start / end in hours (relative to project start)
        start_h = float(getattr(cfg, "WF_OperationsStart_h", 0.0) or 0.0)
        end_h   = float(getattr(cfg, "WF_OperationsEnd_h", 0.0) or 0.0)

        if end_h <= start_h:
            raise ValueError(
                f"WF_OperationsEnd_h ({end_h}) must be greater than "
                f"WF_OperationsStart_h ({start_h})."
            )

        # Project start timestamp
        start_ts = pd.to_datetime(cfg.Project_StartDate, format="%d.%m.%Y")

        op_start_ts = start_ts + pd.to_timedelta(start_h, unit="h")
        op_end_ts   = start_ts + pd.to_timedelta(end_h, unit="h")

        # Build the timestamp index
        timestamps = pd.date_range(
            start=op_start_ts,
            end=op_end_ts,
            freq=freq,
            inclusive="left",   # [start, end)
        )

        # Power response - fixed capacity factor
        rated_power = float(
            get_input_parameter(
                self.wind_farm_input,
                'WindFarm', 'fixed', 'rated_power'
            )
        )  # MW

        capacity_factor = float(
            get_input_parameter(
                self.wind_farm_input,
                'WindFarm', 'fixed', 'capacity_factor'
            )
        )  # 0–1

        fixed_power = rated_power * capacity_factor  # MW

        # Create DataFrame in same format as other options
        power_df = pd.DataFrame({
            "timestamp": timestamps,
            "Total_Production": fixed_power,  # broadcast scalar to all rows
        })

        return power_df



def load_windfarmData(config):
    """
    Loads wind farm input parameters from the configuration file.

    Returns
    -------
    dict
        Dictionary with wind farm parameters.
    """
    wind_farm_data = {}

    if hasattr(config, 'WindFarm_inputFiles'):
        for identifier, file_name in config.WindFarm_inputFiles.items():
            wind_farm_data[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            wind_farm_data[identifier] = process_duration_fields(wind_farm_data[identifier])

    return wind_farm_data