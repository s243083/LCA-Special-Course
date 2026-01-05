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
        

        self.revenue_records = pd.DataFrame()
        self.revenue_metrics_records = pd.DataFrame()



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
            power_series = power_input,
            available_power_series = aap_input,
            generation_based_cfd = (cfd_mode == "generation"),
            settlement_lag_months = 0,
            inflation_rate_annual = inflation_rate,
            premium = 30.0,
            price_resolution = price_resolution,
            price_inflation = price_inflation,
            index_strike_price = index_strike_price,
        )
        
        revenue_df, metrics_df = cfc.calculate_revenues()
        self.revenue_records = revenue_df
        self.revenue_metrics_records = metrics_df



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

        self.revenue_records = pd.DataFrame()
        self.revenue_metrics_records = pd.DataFrame() 


        # Map to pandas offset aliases
        self.freq_map = {
            "hourly": "H",
            "daily": "D",
            "monthly": "M",      # month end
            "quarterly": "Q",
            "biannual": "2Q",     # two quarters
            "annual": "A"
        }

    def _coerce_inputs(self):
        """
        Assumes required columns exist.
        Coerce dtypes, sort, return copies.
        Keeps '_energy' (MWh per row) intact.
        """

        # ---------------- Prices ----------------
        prices = self.simulated_prices.loc[:, ["timestamp", "price"]].copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="raise")
        prices["price"] = pd.to_numeric(prices["price"], errors="raise")
        prices = prices.sort_values("timestamp").reset_index(drop=True)

        # ---------------- Power / Energy ----------------
        required = {"timestamp", "_energy"}
        missing = required - set(self.power_series.columns)
        if missing:
            raise KeyError(
                f"power_series missing required columns: {sorted(missing)}. "
                "Expected at least ['timestamp', '_energy']."
            )

        # Keep Total_Power if present (useful for diagnostics / plots)
        keep_cols = ["timestamp", "_energy"]
        if "Total_Power" in self.power_series.columns:
            keep_cols.append("Total_Power")

        power = self.power_series.loc[:, keep_cols].copy()
        power["timestamp"] = pd.to_datetime(power["timestamp"], errors="raise")
        power["_energy"] = pd.to_numeric(power["_energy"], errors="raise")

        if "Total_Power" in power.columns:
            power["Total_Power"] = pd.to_numeric(power["Total_Power"], errors="coerce")

        power = power.sort_values("timestamp").reset_index(drop=True)

        # ---------------- Diagnostics ----------------
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

    
    def _align_price_and_power(self, prices, power, target_resolution="price") -> pd.DataFrame:
        prices = prices.copy()
        power  = power.copy()

        # --- basic validation ---
        if "_energy" not in power.columns:
            raise KeyError("power dataframe must contain '_energy' (MWh per row)")
        if "timestamp" not in power.columns:
            raise KeyError("power dataframe must contain 'timestamp'")

        prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="raise")
        power["timestamp"]  = pd.to_datetime(power["timestamp"], errors="raise")

        prices = prices.sort_values("timestamp")
        power  = power.sort_values("timestamp")

        # --- price rule from config ---
        price_rule = {"10min": "10min", "1h": "H", "1d": "D"}.get(
            str(self.price_resolution).lower()
        )
        if price_rule is None:
            raise ValueError(f"Unsupported price_resolution: {self.price_resolution!r}")

        # --- infer power resolution (only used if target_resolution='power') ---
        inferred = pd.infer_freq(power["timestamp"])
        power_rule = inferred or "10min"
        power_rule = {"10T": "10min", "T": "min"}.get(power_rule, power_rule)

        # --- choose target rule ---
        if target_resolution == "price":
            rule = price_rule
        elif target_resolution == "power":
            rule = power_rule
        else:
            rule = {"10min": "10min", "1h": "H", "1d": "D"}[target_resolution]

        # --- resample ---
        energy_target = (
            power.set_index("timestamp")["_energy"]
            .resample(rule)
            .sum()
        )

        price_target = (
            prices.set_index("timestamp")["price"]
            .resample(rule)
            .mean()     # time-weighted market price
        )

        df = (
            pd.concat([price_target, energy_target], axis=1)
            .dropna()
            .reset_index()
        )

        return df

    def calculate_revenues(self):
        prices, power = self._coerce_inputs()

        df = self._align_price_and_power(prices=prices, power=power, target_resolution="price")

        ref_freq = self.freq_map.get(self.reference_period)
        pay_freq = self.freq_map.get(self.payment_frequency)
        if pay_freq is None:
            raise ValueError("Invalid payment frequency.")

        # optional price inflation
        if self.price_inflation and not df.empty:
            t0 = df["timestamp"].min()
            years = (df["timestamp"] - t0) / np.timedelta64(1, "D") / 365.2425
            df["price"] = df["price"] * (1 + self.inflation_rate_annual) ** years

        # strike price
        if self.scheme_type in ("CfD", "FiT"):
            if self.index_strike_price and not df.empty:
                t0 = df["timestamp"].min()
                years = (df["timestamp"] - t0) / np.timedelta64(1, "D") / 365.2425
                df["strike_price"] = self.strike_price * (1 + self.inflation_rate_annual) ** years
            else:
                df["strike_price"] = float(self.strike_price)

        energy = df["_energy"]

        # ---------- MVF metrics ----------
        revenue_metrics_records = self._calc_mvf_metrics(df, ref_freq)

        # Always provide columns for downstream scheme logic
        df["market_avg_price"] = np.nan
        df["market_value_factor"] = np.nan

        if not revenue_metrics_records.empty:
            # map per-period values onto each row
            df["__period"] = df["timestamp"].dt.to_period(ref_freq)

            mavg_map = revenue_metrics_records.set_index("period")["market_avg_price"]
            mvf_map  = revenue_metrics_records.set_index("period")["market_value_factor"]

            df["market_avg_price"] = df["__period"].map(mavg_map)
            df["market_value_factor"] = df["__period"].map(mvf_map)

        # ---------- Scheme logic ----------
        if self.scheme_type == "CfD":
            # if ref_freq is None, market_avg_price stays NaN -> you should decide policy:
            # either raise, or fallback to spot price.
            if ref_freq is None:
                raise ValueError("CfD requires a reference_frequency to compute market reference price.")

            df["market_ref"] = df["market_avg_price"]
            diff = df["strike_price"] - df["market_ref"]
            if self.one_sided:
                diff = diff.clip(lower=0.0)

            df["cfd_cashflow"] = diff * energy
            df["market_revenue"] = df["price"] * energy
            df["total_revenue"] = df["market_revenue"] + df["cfd_cashflow"]

        elif self.scheme_type == "FiT":
            df["total_revenue"] = df["strike_price"] * energy

        elif self.scheme_type == "FiP":
            df["total_revenue"] = (df["price"] + float(self.premium)) * energy

        elif self.scheme_type == "Market":
            df["total_revenue"] = df["price"] * energy

        else:
            raise ValueError(f"Unknown scheme_type: {self.scheme_type}")

        # ---------- Settlement ----------
        if self.settlement_lag_months:
            df["settlement_time"] = df["timestamp"] + pd.DateOffset(months=self.settlement_lag_months)
        else:
            df["settlement_time"] = df["timestamp"]

        revenue_records = (
            df.groupby(pd.Grouper(key="settlement_time", freq=pay_freq))
            .agg(total_revenue=("total_revenue", "sum"))
            .reset_index()
            .rename(columns={"settlement_time": "timestamp"})
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # store on instance + return
        self.revenue_metrics_records = revenue_metrics_records
        return revenue_records, revenue_metrics_records

    
    def _calc_mvf_metrics(self, df: pd.DataFrame, ref_freq: str) -> pd.DataFrame:
        """
        Returns per-reference-period metrics:
          period (Period), timestamp (period start), energy_sum, market_avg_price, realized_price, market_value_factor
        """
        if df.empty or ref_freq is None:
            return pd.DataFrame(
                columns=["timestamp", "period", "energy_sum", "market_avg_price", "realized_price", "market_value_factor"]
            )

        period = df["timestamp"].dt.to_period(ref_freq)

        tmp = (
            df.assign(__period=period, price_energy=df["price"] * df["_energy"])
              .groupby("__period", observed=True)
              .agg(
                  energy_sum=("_energy", "sum"),
                  price_energy_sum=("price_energy", "sum"),
                  market_avg_price=("price", "mean"),  # time-weighted
              )
        )

        tmp["realized_price"] = tmp["price_energy_sum"] / tmp["energy_sum"].replace({0: np.nan})
        tmp["market_value_factor"] = tmp["realized_price"] / tmp["market_avg_price"].replace({0: np.nan})

        metrics = tmp.reset_index().rename(columns={"__period": "period"})
        metrics["timestamp"] = metrics["period"].dt.to_timestamp(how="start")

        return (
            metrics[["timestamp", "period", "energy_sum", "market_avg_price", "realized_price", "market_value_factor"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )







###################################################################################################################################################
