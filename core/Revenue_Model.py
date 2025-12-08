# %%
from core.File_Handling import load_yaml, process_duration_fields
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
from core.utils import apply_overrides , get_input_parameter

class Revenue: 

    def __init__(self,env):
        self.env = env
        self.config = env.config
        self.market_inputs = load_marketInput(self.env.config)
        self.market_inputs = get_input_parameter(self.market_inputs, 'MA', 'Revenue')

        self.scheme_type = get_input_parameter(self.market_inputs, 'marketscheme')
        self.strike_price = get_input_parameter(self.market_inputs, self.scheme_type, 'strike_price') 
        
        apply_overrides(self, getattr(self.config, "Revenue_overrides", {}))
        

        self.revenue_records = []  

    def calc_revenues(self):
            
        #strike_price, scheme_type, one_sided=False, cfd_mode="generation",reference_period="monthly", payment_frequency="monthly"):
        
        power_input = self.env.windFarm.power_records
        #power_input = power_input.loc[:, [("timestamp", ""), ("Total_Production", "")]].copy()
        #power_input.columns = ["timestamp", "Total_Production"]
        # access electrcity price
        price_input= self.env.MarketEnv.el_price_records
        
        strike_price = self.strike_price
        one_sided = get_input_parameter(self.market_inputs, self.scheme_type, 'one_sided')
        cfd_mode = get_input_parameter(self.market_inputs, self.scheme_type, 'cfd_mode')
        reference_freq = get_input_parameter(self.market_inputs, self.scheme_type, 'reference_freq')
        settling_freq = get_input_parameter(self.market_inputs, self.scheme_type, 'settling_freq')
        aap_input = None
        inflation_rate = get_input_parameter(self.market_inputs, 'inflation_rate')
        price_inflation = get_input_parameter(self.market_inputs, 'price_inflation')
        index_strike_price = get_input_parameter(self.market_inputs, 'index_strike_price')

        price_resolution = self.env.MarketEnv.resolution

        cfc = CashflowCalculator(
            simulated_prices = price_input,
            strike_price = strike_price,
            scheme_type = self.scheme_type,
            one_sided = one_sided,
            reference_frequency = reference_freq,
            settling_frequency = settling_freq,
            power_series = power_input[['timestamp', 'Total_Production']],
            available_power_series = aap_input,
            generation_based_cfd = (cfd_mode == "generation"),
            settlement_lag_months = 0,
            inflation_rate_annual = inflation_rate,
            premium = 30.0,
            price_resolution = price_resolution,
            price_inflation = price_inflation,
            index_strike_price = index_strike_price,
        )
        self.revenue_records = cfc.calculate_revenues()



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



class CashflowCalculator:
    def __init__(self,
                 simulated_prices: pd.DataFrame,
                 strike_price: float,
                 scheme_type: str = "CfD",
                 premium: float = 5.0,
                 one_sided: bool = False,
                 generation_based_cfd: bool = True,
                 power_series: pd.Series | pd.DataFrame = None,
                 available_power_series: pd.Series | pd.DataFrame = None,
                 reference_frequency: str = "daily",
                 settling_frequency: str = "monthly",
                 settlement_lag_months: int = 0,
                 index_strike_price: bool = False,
                 inflation_rate_annual: float = 0.02,
                 price_resolution: str = "10min", 
                 price_inflation: bool = False,
                 ):

        self.simulated_prices = simulated_prices
        self.strike_price = strike_price
        self.scheme_type = scheme_type
        self.premium = premium
        self.one_sided = one_sided
        self.generation_based_cfd = generation_based_cfd

        self.power_series = power_series
        self.available_power_series = available_power_series
        self.price_resolution = price_resolution

        self.reference_period = reference_frequency
        self.payment_frequency = settling_frequency
        self.settlement_lag_months = settlement_lag_months
        self.index_strike_price = index_strike_price
        self.inflation_rate_annual = inflation_rate_annual

        self.price_inflation = price_inflation


        # Map to pandas offset aliases
        self.freq_map = {
            "hourly": "H",
            "daily": "D",
            "monthly": "M",      # month end
            "quarterly": "Q",
            "biannual": "2Q",     # two quarters
            "annual": "A"
        }

    def _dt_hours(self) -> float:
        """Return hours represented by each price/power timestamp based on price_resolution."""
        res = str(self.price_resolution).lower()
        if res == "10min":
            return 1.0 / 6.0
        if res == "1h":
            return 1.0
        if res == "1d":
            return 24.0
        raise ValueError(f"Unsupported price_resolution: {self.price_resolution!r}. Use '10min', '1h', or '1d'.")

    def _coerce_inputs(self):
        """Assumes required columns exist. Coerce dtypes, sort, return copies."""
        # Prices
        prices = self.simulated_prices.loc[:, ["timestamp", "price"]].copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="raise")
        prices["price"] = pd.to_numeric(prices["price"], errors="raise")
        prices = prices.sort_values("timestamp").reset_index(drop=True)

        # Power
        power = self.power_series.loc[:, ["timestamp", "Total_Production"]].copy()
        power["timestamp"] = pd.to_datetime(power["timestamp"], errors="raise")
        power["Total_Production"] = pd.to_numeric(power["Total_Production"], errors="raise")
        power = power.sort_values("timestamp").reset_index(drop=True)

        # --- small check: print ranges & compare timestamps ---
        if prices.empty or power.empty:
            print("Warning: prices or power is empty; cannot compare timestamps.")
            return prices, power

        p_ts = prices["timestamp"]
        w_ts = power["timestamp"]

        print(f"[prices] first={p_ts.iloc[0]}  last={p_ts.iloc[-1]}  count={len(p_ts)}")
        print(f"[power ] first={w_ts.iloc[0]}  last={w_ts.iloc[-1]}  count={len(w_ts)}")

        same_timestamps = (len(p_ts) == len(w_ts)) and p_ts.equals(w_ts)
        print(f"Timestamps identical (length & order): {same_timestamps}")

        return prices, power

    def calculate_revenues(self):
        prices, power = self._coerce_inputs()

    

        # Frequencies
        ref_freq = self.freq_map.get(self.reference_period)  # may be None
        pay_freq = self.freq_map.get(self.payment_frequency)
        if pay_freq is None:
            raise ValueError("Invalid payment frequency.")

        # Time-step hours from price resolution (e.g., 10min -> 1/6 h)
        # dt_h = self._dt_hours()
        dt_h = 1

        # Merge inputs
        df = pd.merge(prices, power, on="timestamp", how="inner")

        # Energy per timestamp (MWh if power is MW)
        df["_energy"] = df["Total_Production"] * dt_h

        # Inflate prices
        if self.price_inflation:
            t0 = df["timestamp"].min()
            years = (df["timestamp"] - t0) / np.timedelta64(1, "D") / 365.2425
            df["price"] = df["price"] * (1 + self.inflation_rate_annual) ** years



        # Strike price (optionally indexed)
        if self.scheme_type == "CfD" or self.scheme_type == "FiT":
            if self.index_strike_price:
                t0 = df["timestamp"].min()
                years = (df["timestamp"] - t0) / np.timedelta64(1, "D") / 365.2425
                df["strike_price"] = self.strike_price * (1 + self.inflation_rate_annual) ** years
            else:
                df["strike_price"] = float(self.strike_price)

        energy = df["_energy"]


        # ---------- Market average (time-weighted) + MVF (energy-weighted) ----------
        if ref_freq is not None:
            # make a period key we can join on
            df["__period"] = df["timestamp"].dt.to_period(ref_freq)
            # helper column for energy-weighted realized price
            df["price_energy"] = df["price"] * df["_energy"]

            grp = df.groupby("__period", observed=True)

            tmp = grp.agg(
                energy_sum=("_energy", "sum"),
                price_energy_sum=("price_energy", "sum"),
                market_avg_price=("price", "mean"),   # time-weighted market average
            )
            tmp["realized_price"] = tmp["price_energy_sum"] / tmp["energy_sum"]
            tmp["market_value_factor"] = tmp["realized_price"] / tmp["market_avg_price"]

            # attach back to every row in the period
            df = df.merge(
                tmp[["market_avg_price", "market_value_factor"]],
                left_on="__period", right_index=True, how="left"
            )
        else:
            df["market_value_factor"] = np.nan
            df["market_avg_price"] = np.nan



        # Scheme logic
        if self.scheme_type == "CfD":
            # Market reference = time-weighted market average for the period
            df["market_ref"] = df["market_avg_price"]

            diff = df["strike_price"] - df["market_ref"]
            if self.one_sided:
                diff = diff.clip(lower=0.0)

            df["cfd_cashflow"]   = diff * energy                 # €/MWh * MWh
            df["market_revenue"] = df["price"] * energy
            df["total_revenue"]  = df["market_revenue"] + df["cfd_cashflow"]


        elif self.scheme_type == "FiT":
            df["total_revenue"] = df["strike_price"] * energy

        elif self.scheme_type == "FiP":
            df["total_revenue"] = (df["price"] + float(self.premium)) * energy

        elif self.scheme_type == "Market":
            df["total_revenue"] = df["price"] * energy

        else:
            raise ValueError(f"Unknown scheme_type: {self.scheme_type}")

        # Settlement lag and payment grouping (calendar-aware)
        if self.settlement_lag_months:
            df["settlement_time"] = df["timestamp"] + pd.DateOffset(months=self.settlement_lag_months)
        else:
            df["settlement_time"] = df["timestamp"]


        # Aggregate by settlement period (pay_freq is required/validated)
        revenue_records = (
            df.groupby(pd.Grouper(key="settlement_time", freq=pay_freq))
            .agg(total_revenue=("total_revenue", "sum"))
            .reset_index()
            .rename(columns={"settlement_time": "timestamp"})
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return revenue_records






###################################################################################################################################################
