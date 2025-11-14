from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Input loaders (reuse your project utilities)
from core.ResponseFramework_dev.input_handling import (
    load_wind_timeseries,
    load_joint_distribution,
)

# Minimal simulators (the simplified versions you pasted below)
from core.ResponseFramework_dev.simulation_engine import (
    simulate_block,
    simulate_distribution,
)

# Farm setup
from core.ResponseFramework_dev.farm_setup import (
    initialize_HKN_pywake_farm,
    initialize_single_turbine_farm_IEA22,
)


class ResponseFramework:
    """
    Minimal Response Framework

    Supports two modes:
      - 'greedy_timeseries': loads a time series and runs simulate_block (power-only)
      - 'greedy_distribution': loads a joint distribution and runs simulate_distribution (power-only)

    Results are exposed on:
      - self.power_timeseries       (DataFrame or None)
      - self.power_distribution     (DataFrame or None)
      - self.farm                   (farm dict)

    """

    def __init__(self, env):
        self.env = env  # Access to the environment

    
    # -------------------------
    # Parameter initialization
    # -------------------------
    def parameter_initialization(
        self,
        *,
        baseline_mode: str = "greedy_timeseries",            # "greedy_timeseries" or "greedy_distribution"
        wind_price_TS_file: str = "ResponseFramework/data/timeseries/wind_price_timeseries_10min_short.csv",
        distribution_file: str = "ResponseFramework/distributions/HKNB_Weib_WSWD_dist_2deg.csv",
        use_pywake_farm: bool = True,                        # True: HKN PyWake farm, False: single IEA22
        input_resolution: str = "10min",                     # kept for simulate_block signature
        wsp_cut_in: float = 3.0,
        wsp_cut_off: float = 25.0,
    ) -> None:
        # store params
        self.baseline_mode = baseline_mode
        self.wind_price_TS_file = wind_price_TS_file
        self.distribution_file = distribution_file
        self.use_pywake_farm = use_pywake_farm
        self.input_resolution = input_resolution
        self.wsp_cut_in = float(wsp_cut_in)
        self.wsp_cut_off = float(wsp_cut_off)

        # outputs
        self.farm: Optional[Dict[str, Any]] = None
        self.power_timeseries: Optional[pd.DataFrame] = None
        self.power_distribution: Optional[pd.DataFrame] = None

    # -----------------------
    # Execution
    # -----------------------
    def execution(self) -> None:
        # 1) Initialize farm
        if self.use_pywake_farm:
            self.farm = initialize_HKN_pywake_farm()
        else:
            self.farm = initialize_single_turbine_farm_IEA22()

        # 2) Branch by mode (power-only)
        if self.baseline_mode == "greedy_timeseries":
            # Load TS (no price required)
            wind_data = self.env.metEnv.environmental_data_ts

            # Call minimal simulator
            #load from csv
            #self.power_distribution= pd.read_csv("M:\Projects\Cost Model\HiperSim\valuewind\ResponseFramework\repeated_power.csv")
            
            self.power_timeseries = simulate_block(
                wind_data=wind_data,
                farm=self.farm,
                resolution=self.input_resolution,
                use_sector_average=False,
                wsp_min=self.wsp_cut_in,
                wsp_max=self.wsp_cut_off,
                price_series=None,     # ignored by the simplified simulate_block
            )
            self.power_distribution = None

        elif self.baseline_mode == "greedy_distribution":
            # Load distribution (no price required)
            dist_df = load_joint_distribution(
                self.distribution_file,
                require_price=False,
            )

            # Call minimal simulator
            self.power_distribution = simulate_distribution(
                distribution=dist_df,
                farm=self.farm,
                wsp_min=self.wsp_cut_in,
                wsp_max=self.wsp_cut_off,
                use_sector_average=False,
            )
            self.power_timeseries = None

        else:
            raise ValueError("baseline_mode must be 'greedy_timeseries' or 'greedy_distribution'.")
