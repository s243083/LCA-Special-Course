import pandas as pd
import numpy as np
from core.File_Handling import load_yaml
from scipy.stats import weibull_min

class MetEnvironment:
    def __init__(self, env):
        self.env = env  # Access to the environment
        self.metEnvData = load_metEnvData(self.env.config)
        self.environmental_data_ts = pd.DataFrame()

        
        wind_price_TS_file = r"M:\Projects\Cost Model\HiperSim\WINPACT\core\ResponseFramework_dev\data\timeseries\HKNB_timeseries_full_filled_small_gaps_only.csv"
        input_resolution: str = "10min"                    # kept for simulate_block signature
        self.load_wind_timeseries(wind_price_TS_file, input_resolution)
        



    
    def load_wind_timeseries(self, path, resolution, require_price=False):
        """
        Loads wind timeseries data and optionally checks for price column.

        Parameters:
        - path: path to CSV with columns ['timestamp', 'wsp', 'TI', 'wdir'] (and optionally 'price')
        - resolution: expected resolution, '10min' or '1h'
        - require_price: if True, checks that a 'price' column exists

        Returns:
        - DataFrame with wind (and optionally price) data
        """
        df = pd.read_csv(path)
        expected_cols = {"timestamp", "wsp", "TI", "wdir"}
        if require_price:
            expected_cols.add("price")
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Wind timeseries is missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Resolution check
        actual_freq = pd.infer_freq(df["timestamp"])
        if actual_freq is None:
            raise ValueError("Could not infer frequency of timestamps in wind data.")
        freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
        normalized = freq_map.get(actual_freq.lower(), actual_freq.lower())
        if normalized != resolution:
            raise ValueError(f"Wind timeseries resolution mismatch: expected {resolution}, got {normalized}")

        self.environmental_data_ts = df

    def _initialize_wind_speed_series(self):
        """Initializes the wind speed time series based on configuration data."""
        # Extract Weibull parameters from the 'ME' identifier under 'MetEnv'
        met_env_data = self.metEnvData.get('ME', {}).get('MetEnv', [])
        
        # Find the entry for Wind_speed (assumes there's only one such entry in the config for simplicity)
        for param in met_env_data:
            if param['name'] == "Wind_speed":
                shape = param['Parameters']['Weibull_shape']
                scale = param['Parameters']['Weibull_scale']
                start_hour = self.env.config.WF_OperationsStart_h
                end_hour = self.env.config.WF_OperationsEnd_h
                # Generate the wind speed series with accurate start and end hour range
                return self.generate_wind_speed_series(shape, scale, start_hour, end_hour)

        # Return an empty series if no Wind_speed parameters are found
        return pd.Series(dtype=float)

    def _initialize_TI(self):
        """Initializes the turbulence intensity (TI) based on configuration data."""
        # Access 'MetEnv' data under the 'ME' identifier
        met_env_data = self.metEnvData.get('ME', {}).get('MetEnv', [])
        
        # Find the entry for TI in the parameters (assuming a single entry with 'name': "TI")
        for param in met_env_data:
            if param['name'] == "Turbulence_Intensity":
                # Store and return the TI value for future use
                return param['Parameters'].get('TI', None)
        
        # Return None if TI is not found and log a message
        print("Turbulence Intensity (TI) data not found.")
        return None

    def get_wind_speed(self) -> float:
        """
        Provides wind speed at the current simulation time, interpolating if necessary.

        Returns
        -------
        float
            The wind speed at the current simulation time.
        """
        current_time = self.env.now  # Directly use the simulation hour index
        
        if current_time in self.wind_speed_series.index:
            return self.wind_speed_series.loc[current_time]
        else:
            # Interpolate wind speed if exact time is not in the series
            return np.interp(current_time, self.wind_speed_series.index, self.wind_speed_series.values)
        
    def get_wind_direction(self) -> float:
        """
        Provides wind direction at the current simulation time, interpolating if necessary.

        Returns
        -------
        float
            The wind direction at the current simulation time.
        """
        return np.random.randint(0, 360) # Placeholder for wind direction
    
        
    def get_TI(self) -> float:
        """
        Retrieves the Turbulence Intensity (TI) parameter as a static value.

        Returns
        -------
        float
            The turbulence intensity (TI) value.
        """
        return self.TI

    def fit_weibull_distribution(self):
        """
        Fit a Weibull distribution to the given wind speed series.

        Returns
        -------
        shape_param, loc, scale_param : tuple
            Fitted parameters of the Weibull distribution.
        """
        shape_param, loc, scale_param = weibull_min.fit(self.wind_speed_series, floc=0)
        return shape_param, loc, scale_param

    def generate_wind_speed_series(self, shape: float, scale: float, start_hour: int, end_hour: int) -> pd.Series:
        """
        Generates a time series of hourly wind speed values based on a Weibull distribution.

        Parameters
        ----------
        shape : float
            The shape parameter (k) of the Weibull distribution.
        scale : float
            The scale parameter (λ) of the Weibull distribution.
        start_hour : int
            The start hour for the wind speed time series.
        end_hour : int
            The end hour for the wind speed time series.

        Returns
        -------
        pd.Series
            A time series of wind speed values indexed from start_hour to end_hour.
        """
        # Calculate the number of hours based on the start and end times
        hours = end_hour - start_hour + 1  # inclusive of both start and end hour
        
        # Generate hourly wind speeds from a Weibull distribution
        wind_speeds = scale * np.random.weibull(shape, hours)
        
        # Create a time series indexed by simulation hour from start_hour to end_hour
        time_index = range(start_hour, end_hour + 1)
        wind_speed_series = pd.Series(wind_speeds, index=time_index, name="wind_speed")

        return wind_speed_series


def load_metEnvData(config):
    metEnvData = {}
    
    # Load MetEnv input files
    if hasattr(config, 'MetEnv_inputFiles'):
        for identifier, file_name in config.MetEnv_inputFiles.items():
            metEnvData[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
        print("Loaded MetEnv data structure:", metEnvData)
    
    return metEnvData
