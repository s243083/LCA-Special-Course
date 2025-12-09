

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
    initialize_pywake_farm,
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
        self.env = env
        self.farm = None
        self.power_timeseries = None
        self.power_distribution = None


    def parameter_initialization(
        self,
        *,
        baseline_mode: str = "timeseries",      # "timeseries" or "distribution"
        use_pywake_farm: bool = True,
        layout_file: str | None = None,
        turbine_model: str = "IEA22",
        distribution_file: str | None = None,
        wsp_cut_in: float = 3.0,
        wsp_cut_off: float = 25.0,
        use_sector_average: bool = False,
        flag_save_power_file: bool = False,
        flag_useprecomputed_file: bool = False,
        precomputed_power_file: str | None = None,
    ) -> None:
        self.baseline_mode          = baseline_mode
        self.use_pywake_farm        = use_pywake_farm
        self.layout_file            = layout_file
        self.turbine_model          = turbine_model
        self.distribution_file      = distribution_file
        self.wsp_cut_in             = float(wsp_cut_in)
        self.wsp_cut_off            = float(wsp_cut_off)
        self.use_sector_average     = bool(use_sector_average)
        self.flag_save_power_file   = bool(flag_save_power_file)
        self.flag_useprecomputed    = bool(flag_useprecomputed_file)
        self.precomputed_power_file = precomputed_power_file

        self.farm = None
        self.power_timeseries = None
        self.power_distribution = None

    # -----------------------
    # Execution
    # -----------------------
    def execution(self) -> None:
        # 1) Initialize farm from config
        self.farm = initialize_pywake_farm(
            use_pywake_farm=self.use_pywake_farm,
            layout_file=self.layout_file,
            turbine_model=self.turbine_model,
        )

        # 2) Timeseries vs distribution
        if self.baseline_mode == "timeseries":
            if self.flag_useprecomputed and self.precomputed_power_file:
                power_df = pd.read_parquet(self.precomputed_power_file)
            else:
                wind_data = self.env.metEnv.environmental_data_ts
                power_df = simulate_block(
                    wind_data=wind_data,
                    farm=self.farm,
                    wsp_min=self.wsp_cut_in,
                    wsp_max=self.wsp_cut_off,
                    use_sector_average=self.use_sector_average,
                )

                if self.flag_save_power_file and self.precomputed_power_file:
                    power_df.to_parquet(self.precomputed_power_file, index=False)

            self.power_timeseries = power_df
            self.power_distribution = None

        elif self.baseline_mode == "distribution":
            if not self.distribution_file:
                raise ValueError("distribution_file must be provided for 'distribution' mode.")

            dist_df = load_joint_distribution(self.distribution_file, require_price=False)

            self.power_distribution = simulate_distribution(
                distribution=dist_df,
                farm=self.farm,
                wsp_min=self.wsp_cut_in,
                wsp_max=self.wsp_cut_off,
                use_sector_average=self.use_sector_average,
            )
            self.power_timeseries = None

        else:
            raise ValueError("baseline_mode must be 'timeseries' or 'distribution'.")

