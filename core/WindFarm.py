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
    # There needs to be a if condition here to determine which process to start.

        # load external response (eg.from VP framework)
        self.load_external()

        # call the response framework
        #self.power_records = self.calculate_windfarm_response()

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








    def run_TS_hourly(self):
        """
        A generator function that triggers the wind farm response calculation every hour
        within the specified operational time window.
        """
        while True:
            if self.start_time <= self.env.now <= self.end_time:
                self.get_WindFarmResponse_timestep()
                #if turbine_surrogate
                #self.get_TurbineResponse()
            yield self.env.timeout(1)

    def run_binWise(self):
        """
        A generator function that triggers the wind farm response calculation for every bin combination of the input distributions.
        """
        #if self.start_time == self.env.now:
            # get the bin combinations of the input distributions

            # get Wind farm response for the bin combinations
            # for each bin get the wind farm response and write it to the turbine level
            # should be the same as the get_WindFarmResponse_timestep function, but for all bins


    def get_WindFarmResponse_timestep(self):
        """
        Returns the wind farm response for the current simulation step.
        """
        # get the control output for this timestep, this is a dummy implementation
        #WF_Controller.compute_turbine_setpoints()
        control_output = self.wf_controller.get_turbine_setpoints()
        response = self.wf_surrogate_query.get_windFarm_response_timestep(control_output)
        # write response to turbine level
        for i, (turbine_name, turbine) in enumerate(self.turbines.items()):
            turbine_response = {}
            turbine_response[f"{'Power'}_{'mean'}"] = response.Power.sel(wt=i) #  This only works if turbines are initialized in the same order as defined in the surrogate model, could be improved
            new_entry = pd.DataFrame({
            'simulation_time': [self.env.now],
            'response': [turbine_response],
            'fatigue_damage': [None]  # Initialize with None, to be updated later
            })
            turbine.response_log = pd.concat([turbine.response_log, new_entry], ignore_index=True)

            # write flow conditions to turbine level
            inflow_conditions = {}
            inflow_conditions[f"{'ws_ambient'}"] = response.WS_eff.sel(wt=i) # same
            inflow_conditions[f"{'ti_ambient'}"] = response.TI_eff.sel(wt=i) # same
            new_entry = pd.DataFrame({
                'simulation_time': [self.env.now],
                'flow_conditions': [inflow_conditions]
            })
            turbine.ambient_inflow = pd.concat(
                [turbine.ambient_inflow, new_entry],
                ignore_index=True
            )
        return None

    def get_turbine_response(self):
        for turbine in self.turbines.values():
            turbine.get_turbine_response()
        return None

    # this is calling pywake for all WS and Wind direction
    def get_WindFarmResponse_global(self):

        # get the control ouptut 
        control_output = self.get_ControlOutput()
        

        # get the reponse of the farm surrogate
        response = self.wf_surrogate_query.get_windFarm_response(control_output)
        return None

    # the fatique analyis is done on turbine level from here
    # turbine object is passed to the function
    def get_FatigueAnalysis(self):    
        for turbine in self.turbines.values():
            self.fatigue_analysis.get_fatigue_analysis(turbine)
        return None

         

    # this needs to be checked
    def get_WindFarmResponse_Reference(self):
        """
        Calculates and logs the wind farm reference response based on Weibull distribution fitting.

        Returns
        -------
        dict of pd.DataFrame
            The wind farm reference response logs, each entry keyed by turbine name.
        """
        TIreq = 10  # Placeholder for turbulence intensity in %
        Preq = 11   # Rated power in MW
        interp_type = 'linear'  # Example interpolation type

        if self.wind_farm_data['WF']['WF_Response_Reference']['flag_fitWB']:
            shape_param, loc, scale_param = self.env.metEnv.fit_weibull_distribution()

            wind_speed_bins = np.linspace(4, 24, 11)
            weibull_pdf = weibull_min.pdf(wind_speed_bins, shape_param, loc, scale_param)
            weibull_pdf = weibull_pdf / np.sum(weibull_pdf)

            for turbine in self.turbines.values():
                turbine.get_turbine_reference_response(
                    self.fatigue_analysis, wind_speed_bins, weibull_pdf, TIreq, Preq, interp_type
                )
        return None


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