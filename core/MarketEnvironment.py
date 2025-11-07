import pandas as pd
import numpy as np
from core.File_Handling import load_yaml, process_duration_fields, loadcsv
from core.utils import repeat_timeseries, gap_fill_timeseries, get_input_parameter, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration

class MarketEnv:

    def __init__(self,env):
        self.env = env
        self.config = env.config
        self.market_inputs = load_marketInput(self.env.config)
        self.resolution = get_input_parameter(self.market_inputs, 'MA', 'Market', 'timeseries', 'resolution')


        self.create_electricityprice()
        
        


    def create_electricityprice(self):
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

