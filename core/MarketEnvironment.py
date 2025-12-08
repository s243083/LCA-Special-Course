import pandas as pd
import numpy as np
from core.File_Handling import load_yaml, process_duration_fields, loadcsv
from core.utils import repeat_timeseries, gap_fill_timeseries, get_input_parameter, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration

class MarketEnv:

    def __init__(self,env):
        self.env = env
        self.config = env.config
        self.market_inputs = load_marketInput(self.env.config)

        self.market_type = get_input_parameter(self.market_inputs, 'MA', 'Market', 'mode')  
        
        self.create_electricityprice()
        
    def create_electricityprice(self):
        market_type = self.market_type

        if market_type == 'fromEnv':
            self.create_electricityprice_fromEnv()
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
        freq = get_input_parameter(self.market_inputs, 'MA', 'Market', 'timeseries', 'fixed', 'resolution')
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
                'MA', 'Market', 'timeseries', 'fixed', 'price'
            )
        )

        self.el_price_records = pd.DataFrame({
            "timestamp": timestamps,
            "price": price,
        })


    def create_electricityprice_fromEnv(self):

        # this is a purpose built function to create electricity price time series from the met environment, it should be replaced by a more general function
        # Build base DataFrame
        base_df = pd.DataFrame({
            "timestamp": self.env.metEnv.environmental_data_ts["timestamp"],
            "price": self.env.metEnv.environmental_data_ts["price"],
        })

        # 1) Remove gaps and rebuild timestamps to be continuous
        base_df = remove_gaps_rebuild_timestamps(
            base_df,
            timestamp_col="timestamp",
            freq=None,   # infer from data
            sort=True,
        )

        # 2) Repeat until the desired total duration is reached
        self.el_price_records = repeat_timeseries_to_duration(
            base_df,
            duration="20 years",     # extend to 20 years
            timestamp_col="timestamp",
            trim_to_duration=True,
        )




    # all functions defining the market behaviour functions should be defined here
    def get_market_condition(self):
        

        return None



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

