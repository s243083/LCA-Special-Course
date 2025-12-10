import pandas as pd
import numpy as np
from core.File_Handling import load_yaml
from scipy.stats import weibull_min
from core.utils import apply_overrides , get_input_parameter,  remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration


class MetEnvironment:
    def __init__(self, env):
        self.env = env  # Access to the environment
        self.metEnvInput = load_metEnvInput(self.env.config)
        self.metEnvInput = get_input_parameter(self.metEnvInput, 'ME', 'MetEnv')
        
        
        self.environmental_data_ts = pd.DataFrame()


    def create_met_environment(self):

        # get mode from config
        mode = get_input_parameter(self.metEnvInput, 'mode')

        if mode == "external":
            self.load_wind_timeseries()
        elif mode== "fixed":
            self.create_fixed_conditions()
        elif mode == "sampled":
            self.create_sampled_conditions()
        else:
            raise ValueError(f"WindFarm Response mode '{mode}' not recognized.")


    
    def load_wind_timeseries(self):
        """
        Loads wind timeseries data, validates required columns,
        checks the timestamp resolution, and keeps only the expected columns.
        """

        # --- Load configuration ---
        path = get_input_parameter(self.metEnvInput, 'external', 'file')
        resolution = get_input_parameter(self.metEnvInput, 'external', 'resolution')
        expected_cols = get_input_parameter(self.metEnvInput, 'external', 'expected_columns')
        duration = get_input_parameter(self.metEnvInput, 'external', 'target_duration')

        # YAML list → Python set
        if expected_cols is None:
            expected_cols = {"timestamp", "wsp", "TI", "wdir"}
        else:
            expected_cols = set(expected_cols)

        # --- Load CSV ---
        df = pd.read_csv(path)

        # --- Validate required columns ---
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Wind timeseries is missing required columns: {missing}")

        # --- Always keep timestamp ---
        required_cols = {"timestamp"} | expected_cols

        # --- Convert timestamp ---
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # --- Check time resolution ---
        actual_freq = pd.infer_freq(df["timestamp"])
        if actual_freq is None:
            raise ValueError("Could not infer frequency of timestamps in wind data.")

        freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
        normalized = freq_map.get(actual_freq.lower(), actual_freq.lower())

        if normalized != resolution:
            raise ValueError(
                f"Wind timeseries resolution mismatch: expected {resolution}, got {normalized}"
            )

        # --- Keep only required columns ---
        base_df = df[list(required_cols)]


        # --- 1) Remove gaps and rebuild timestamps to be continuous ---
        base_df = remove_gaps_rebuild_timestamps(
            base_df,
            timestamp_col="timestamp",
            freq=None,  # infer from data
            sort=True,
        )

        # --- 2) Repeat until the desired total duration is reached ---
        if duration is not None:
            base_df = repeat_timeseries_to_duration(
                base_df,
                duration=duration,
                timestamp_col="timestamp",
                trim_to_duration=True,
            )

        # Store cleaned dataframe
        self.environmental_data_ts = base_df


def load_metEnvInput(config):
    metEnvData = {}
    
    # Load MetEnv input files
    if hasattr(config, 'MetEnv_inputFiles'):
        for identifier, file_name in config.MetEnv_inputFiles.items():
            metEnvData[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
        #print("Loaded MetEnv data structure:", metEnvData)
    
    return metEnvData
