import pandas as pd
import numpy as np
from core.File_Handling import load_yaml, process_duration_fields, loadcsv
from core.utils import repeat_timeseries, gap_fill_timeseries, get_input_parameter, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration

class MarketEnv:

    def __init__(self,env):
        self.env = env
        self.config = env.config
        self.market_inputs = load_marketInput(self.env.config)
        self.market_inputs = get_input_parameter(self.market_inputs, 'MA')

        self.market_type = get_input_parameter(self.market_inputs, 'Market', 'mode')  

        self.el_price_records = pd.DataFrame()
        

        
    def create_electricityprice(self):
        market_type = self.market_type

        if market_type == 'external':
            self.load_external_price()
        elif market_type == 'fixed':
            self.create_electricityprice_fixed()
        else:
            raise ValueError(f"Market type '{market_type}' not recognized.")
    
    def create_electricityprice_fixed(self): 
        """
        Create a fixed electricity price time series from operations start to operations end.
        The resulting DataFrame is stored in self.el_price_records and has the same format
        as the one created in create_electricityprice_fromEnv, i.e. columns:
            - 'timestamp'
            - 'price'
        """

        cfg = self.env.config

        # resolution directly from input, e.g. "10min", "1h", "1d"
        freq = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'fixed', 'resolution')
        self.resolution = freq

        # Operation start / end in hours (relative)
        start_h = float(cfg.WF_OperationsStart_h)
        end_h   = float(cfg.WF_OperationsEnd_h)

        # Project start TS
        start_ts = pd.to_datetime(cfg.Project_StartDate, format="%d.%m.%Y")

        op_start_ts = start_ts + pd.to_timedelta(start_h, unit="h")
        op_end_ts   = start_ts + pd.to_timedelta(end_h, unit="h")

        # Build timestamps
        timestamps = pd.date_range(
            start=op_start_ts,
            end=op_end_ts,
            freq=freq,
            inclusive="left"
        )

        # Fixed price
        price = float(
            get_input_parameter(
                self.market_inputs,
                'Market', 'timeseries', 'fixed', 'price'
            )
        )

        self.el_price_records = pd.DataFrame({
            "timestamp": timestamps,
            "price": price,
        })

    def load_external_price(self):
        """
        Load external electricity price time series.

        Expects a CSV with at least:
        - 'timestamp'
        - 'price' (by default, or whatever is configured in expected_columns)

        YAML config (example):

        Market:
        timeseries:
            external:
            file: "../examples/Inputs/Market/price_timeseries.csv"
            resolution: "1h"         # or "10min", "1d"
            expected_columns:
                - "price"              # can be extended, e.g. ["price", "price_EUR"]
            target_duration: "8760h" # or whatever format your helper expects
        """

        # --- Load configuration ---
        path = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'file')
        self.resolution = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'resolution')
        resolution = self.resolution
        expected_cols = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'expected_columns')
        duration = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'target_duration')

        # Default required data columns (besides timestamp)
        if expected_cols is None:
            expected_cols = {"price"}
        else:
            expected_cols = set(expected_cols)

        # We always require timestamp as well
        required_cols = {"timestamp"} | expected_cols

        # --- Load CSV ---
        base_df = pd.read_csv(path)

        # --- Check required columns ---
        missing = required_cols - set(base_df.columns)
        if missing:
            raise ValueError(f"Price timeseries is missing required columns: {missing}")

        # --- Keep only timestamp + expected columns (ordered) ---
        ordered_cols = ["timestamp"] + [c for c in expected_cols if c != "timestamp"]
        base_df = base_df[ordered_cols]

        # --- Convert timestamp to datetime ---
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])

        # --- Check time resolution ---
        actual_freq = pd.infer_freq(base_df["timestamp"])
        if actual_freq is None:
            raise ValueError("Could not infer frequency of timestamps in price data.")

        # Map pandas frequency strings to your resolution convention
        freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
        normalized = freq_map.get(actual_freq, actual_freq)
        normalized = normalized.lower()

        if resolution is not None and normalized != resolution.lower():
            raise ValueError(
                f"Price timeseries resolution mismatch: expected {resolution}, got {normalized}"
            )

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

        # Store result
        self.el_price_records = base_df


def load_marketInput(config):
    """
    Loads wind farm input parameters from the configuration file.

    Returns
    -------
    dict
        Dictionary with wind farm parameters.
    """
    market_inputs = {}

    if hasattr(config, 'Market_inputFiles'):
        for identifier, file_name in config.Market_inputFiles.items():
            market_inputs[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            market_inputs[identifier] = process_duration_fields(market_inputs[identifier])

    return market_inputs

