# %%
from core.File_Handling import load_yaml, process_duration_fields
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import get_period_alias
from core.utils import apply_overrides, get_input_parameter


class Revenue:
    """Wind-farm revenue model with configurable market schemes.

    The Revenue module turns a power-production time series and an
    electricity-price time series into a settled cash-flow stream under
    one of several market schemes (merchant, fixed price, two-sided
    Contract for Difference, premium, etc.). It is parametrised by a
    YAML market block plus the ``Revenue_overrides`` dotted-path map.

    The expensive lifting is done by :class:`CashflowCalculator`; this
    class is a thin façade that resolves inputs, applies overrides,
    holds the public attributes, and calls the calculator with the
    farm-level production and price series.

    Parameters
    ----------
    env : ValueWindEnv
        Owning environment. Revenue reads
        ``env.windFarm.power_records`` and
        ``env.MarketEnv.el_price_records`` at call time, so it must be
        invoked after both have been populated.

    Attributes
    ----------
    scheme_type : str
        Active market scheme identifier (``"CfD"``, ``"merchant"``,
        ``"fixed"``, ...).
    strike_price : float
        Strike or fixed price of the active scheme. This is the
        decision variable mutated by :class:`core.Simulation.SolveSweepExperiment`
        when solving for break-even.
    premium : float
        Additional premium paid on top of the reference price (where
        applicable).
    one_sided : bool
        If True, the CfD only pays out when the reference price falls
        below strike (no clawback).
    cfd_mode : {"generation", "production"}
        Whether settlements are based on generation or available power.
    reference_freq, settling_freq : str
        Pandas frequency aliases for the reference and settlement
        windows (e.g. ``"D"``, ``"M"``).
    inflation_rate, price_inflation, index_strike_price : float, bool, bool
        Inflation handling: annual rate, whether the simulated price is
        inflated, and whether the strike price is indexed.
    revenue_records : pandas.DataFrame
        Settled cash flows after :meth:`calc_revenues`.
    revenue_metrics_records, revenue_metrics_records_potential : pandas.DataFrame
        Aggregated revenue/MVF metrics (physical and, optionally,
        potential energy).

    See Also
    --------
    CashflowCalculator : the underlying settlement engine.
    core.MarketEnvironment.MarketEnvironment : produces the price series.
    core.Valuation.Valuation : consumes ``revenue_records``.
    """

    def __init__(self, env):
        self.env = env
        self.config = env.config

        self.market_inputs = load_marketInput(self.env.config)
        self.market_inputs = get_input_parameter(self.market_inputs, "MA", "Revenue")

        self.scheme_type = get_input_parameter(self.market_inputs, "marketscheme")

        self.strike_price = get_input_parameter(self.market_inputs, self.scheme_type, "strike_price")
        self.premium = get_input_parameter(self.market_inputs, self.scheme_type, "premium")
        self.one_sided = get_input_parameter(self.market_inputs, self.scheme_type, "one_sided")
        self.cfd_mode = get_input_parameter(self.market_inputs, self.scheme_type, "cfd_mode")
        self.reference_freq = get_input_parameter(self.market_inputs, self.scheme_type, "reference_freq")
        self.settling_freq = get_input_parameter(self.market_inputs, self.scheme_type, "settling_freq")
        
        self.inflation_rate = get_input_parameter(self.market_inputs, "inflation_rate")
        self.price_inflation = get_input_parameter(self.market_inputs, "price_inflation")
        self.index_strike_price = get_input_parameter(self.market_inputs, "index_strike_price")



        apply_overrides(self, getattr(self.config, "Revenue_overrides", {}))

        self.revenue_records = pd.DataFrame()
        self.revenue_metrics_records = pd.DataFrame()

        # Optional: if you want both physical & potential metric views, CashflowCalculator stores potential metrics too.
        self.revenue_metrics_records_potential = pd.DataFrame()

    def calc_revenues(self):
        power_input = self.env.windFarm.power_records
        price_input = self.env.MarketEnv.el_price_records

        aap_input = None
        price_resolution = self.env.MarketEnv.resolution

        cfc = CashflowCalculator(
            simulated_prices=price_input,
            strike_price=self.strike_price,
            scheme_type=self.scheme_type,
            one_sided=self.one_sided,
            reference_frequency=self.reference_freq,
            settling_frequency= self.settling_freq,
            power_series=power_input,
            available_power_series=aap_input,
            generation_based_cfd=(self.cfd_mode == "generation"),
            settlement_lag_months=0,
            inflation_rate_annual=self.inflation_rate,
            premium=self.premium,
            price_resolution=price_resolution,
            price_inflation=self.price_inflation,
            index_strike_price=self.index_strike_price,
            store_potential_metrics=True,  # keep potential MVF metrics alongside physical
        )

        revenue_df, metrics_physical_df = cfc.calculate_revenues()
        self.revenue_records = revenue_df
        self.revenue_metrics_records = metrics_physical_df

        # Optional potential metrics (if enabled in calculator)
        if hasattr(cfc, "revenue_metrics_records_potential"):
            self.revenue_metrics_records_potential = cfc.revenue_metrics_records_potential


def load_marketInput(config):
    """
    Loads market input parameters from configuration.
    """
    market_inputs = {}

    if hasattr(config, "Market_inputFiles"):
        for identifier, file_name in config.Market_inputFiles.items():
            market_inputs[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            market_inputs[identifier] = process_duration_fields(market_inputs[identifier])

    return market_inputs


class CashflowCalculator:
    """Cash-flow settlement engine for electricity-market schemes.

    Computes period-by-period payments from a price series and a power
    series under merchant, fixed-price, or Contract-for-Difference
    arrangements. Handles the resampling between price resolution,
    reference period (over which the reference price is averaged),
    settlement period, and an optional settlement lag, plus inflation
    indexation of the strike price.

    The calculator is decoupled from WINPACT's environment so it can
    be unit-tested in isolation.

    Parameters
    ----------
    simulated_prices : pandas.DataFrame
        Electricity-price time series at ``price_resolution``.
    strike_price : float
        Strike price (CfD) or fixed price (PPA), in the same currency
        and units as ``simulated_prices``.
    scheme_type : str, default ``"CfD"``
        Market scheme identifier; selects the settlement formula.
    premium : float, default 5.0
        Additional premium added on top of merchant revenue (where
        applicable).
    one_sided : bool, default False
        If True, the CfD only pays the producer when the reference
        price is below strike (no clawback for high prices).
    generation_based_cfd : bool, default True
        If True, settlement uses actually-generated MWh; if False, it
        uses the available (potential) MWh from
        ``available_power_series``.
    power_series : pandas.Series or DataFrame
        Generated power series.
    available_power_series : pandas.Series or DataFrame, optional
        Production-based-CfD reference power.
    reference_frequency, settling_frequency : str
        Pandas frequency aliases for the reference price averaging
        window and the settlement payment window.
    settlement_lag_months : int, default 0
        Lag between settlement period end and payment date.
    index_strike_price : bool, default False
        If True, the strike price is inflated annually at
        ``inflation_rate_annual``.
    inflation_rate_annual : float, default 0.02
        Annual inflation rate (fraction).
    price_resolution : str, default ``"10min"``
        Resolution of the input price series.
    price_inflation : bool, default False
        If True, simulated prices are also inflated.
    store_potential_metrics : bool, default False
        If True, also computes Market Value Factor (MVF) metrics on the
        potential (available) energy series for comparison against
        physical generation.

    See Also
    --------
    Revenue : façade that wires this calculator to the simulation.
    """

    def __init__(
        self,
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
        store_potential_metrics: bool = False,
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
        self.store_potential_metrics = store_potential_metrics

        self.revenue_records = pd.DataFrame()
        self.revenue_metrics_records = pd.DataFrame()

        # Optional additional view
        self.revenue_metrics_records_potential = pd.DataFrame()

        # Map to pandas offset aliases
        self.freq_map = {
            "hourly": "h",
            "daily": "D",
            "monthly": "ME",  # month end
            "quarterly": "Q",
            "biannual": "2Q",
            "annual": "YE",
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
        power = power.copy()

        if "_energy" not in power.columns:
            raise KeyError("power dataframe must contain '_energy' (MWh per row)")
        if "timestamp" not in power.columns:
            raise KeyError("power dataframe must contain 'timestamp'")

        prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="raise")
        power["timestamp"] = pd.to_datetime(power["timestamp"], errors="raise")

        prices = prices.sort_values("timestamp")
        power = power.sort_values("timestamp")

        # price rule from config
        price_rule = {"10min": "10min", "1h": "h", "1d": "D"}.get(str(self.price_resolution).lower())
        if price_rule is None:
            raise ValueError(f"Unsupported price_resolution: {self.price_resolution!r}")

        # infer power resolution (only used if target_resolution='power')
        inferred = pd.infer_freq(power["timestamp"])
        power_rule = inferred or "10min"
        power_rule = {"10T": "10min", "T": "min"}.get(power_rule, power_rule)

        # choose target rule
        if target_resolution == "price":
            rule = price_rule
        elif target_resolution == "power":
            rule = power_rule
        else:
            rule = {"10min": "10min", "1h": "h", "1d": "D"}[target_resolution]

        # resample
        energy_target = power.set_index("timestamp")["_energy"].resample(rule).sum()
        price_target = prices.set_index("timestamp")["price"].resample(rule).mean()  # time-weighted market price

        df = pd.concat([price_target, energy_target], axis=1).dropna().reset_index()
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
        if self.scheme_type in ("CfD", "FiT", "CfD_NoRevenue_at_NegativePrices", "Capability_CfD"):
            if self.index_strike_price and not df.empty:
                t0 = df["timestamp"].min()
                years = (df["timestamp"] - t0) / np.timedelta64(1, "D") / 365.2425
                df["strike_price"] = self.strike_price * (1 + self.inflation_rate_annual) ** years
            else:
                df["strike_price"] = float(self.strike_price)

        # ---------------------------------------------------------------------
        # Explicit energy bases (potential/physical/CfD-basis)
        # ---------------------------------------------------------------------
        df["energy_potential"] = df["_energy"].astype(float)
        df["energy_physical"] = df["energy_potential"].copy()
        df["energy_cfd_basis"] = df["energy_potential"].copy()

        negative_mask = df["price"] < 0
        if len(df) > 0:
            print(f"Share of negative price intervals: {negative_mask.mean():.2%}")
        else:
            print("Warning: df is empty, cannot compute negative price share.")

        # Apply scheme-specific curtailment / basis rules BEFORE metrics
        if self.scheme_type == "CfD_NoRevenue_at_NegativePrices":
            # Physical stops at negative prices, and CfD is paid on curtailed (physical) energy
            df.loc[negative_mask, "energy_physical"] = 0.0
            df["energy_cfd_basis"] = df["energy_physical"]

        elif self.scheme_type == "Capability_CfD":
            # Physical stops at negative prices, but CfD is still paid on potential generation
            df.loc[negative_mask, "energy_physical"] = 0.0
            df["energy_cfd_basis"] = df["energy_potential"]

        else:
            # Default: support paid on physical energy
            df["energy_cfd_basis"] = df["energy_physical"]

        # ---------------------------------------------------------------------
        # MVF metrics computed on PHYSICAL energy (reflects curtailment)
        # ---------------------------------------------------------------------
        metrics_physical = self._calc_mvf_metrics(df, ref_freq, energy_col="energy_physical")
        self.revenue_metrics_records = metrics_physical

        if self.store_potential_metrics:
            self.revenue_metrics_records_potential = self._calc_mvf_metrics(df, ref_freq, energy_col="energy_potential")

        # Map period metrics back onto each row
        df["market_avg_price"] = np.nan
        df["market_value_factor"] = np.nan

        if not metrics_physical.empty:
            period_freq = get_period_alias(ref_freq) or ref_freq
            df["__period"] = df["timestamp"].dt.to_period(period_freq)

            mavg_map = metrics_physical.set_index("period")["market_avg_price"]
            mvf_map = metrics_physical.set_index("period")["market_value_factor"]

            df["market_avg_price"] = df["__period"].map(mavg_map)
            df["market_value_factor"] = df["__period"].map(mvf_map)

        # ---------------------------------------------------------------------
        # Scheme logic + unified support metrics
        # ---------------------------------------------------------------------
        e_phys = df["energy_physical"]
        e_cfd = df["energy_cfd_basis"]

        # Downstream-consistent columns
        df["market_revenue"] = 0.0
        df["cfd_cashflow"] = 0.0
        df["market_ref"] = np.nan

        # NEW: universal support fields
        df["support_payment"] = 0.0         # net support cashflow attributable to scheme
        df["support_energy"] = 0.0          # MWh basis used for support_payment
        df["support_rate"] = np.nan         # support_payment / support_energy (if energy>0)

        # Optional: FiT support accounting mode
        # - "gross": support = total FiT payment, market_revenue forced to 0
        # - "net_topup": support = (strike - price) * energy (optionally one-sided), market_revenue = price*energy
        fit_support_mode = getattr(self, "fit_support_mode", "gross")  # "gross" | "net_topup"

        if self.scheme_type == "CfD":
            if ref_freq is None:
                raise ValueError("CfD requires a reference_frequency to compute market reference price.")

            df["market_ref"] = df["market_avg_price"]
            diff = df["strike_price"] - df["market_ref"]
            if self.one_sided:
                diff = diff.clip(lower=0.0)

            df["market_revenue"] = df["price"] * e_phys
            df["cfd_cashflow"] = diff * e_cfd
            df["total_revenue"] = df["market_revenue"] + df["cfd_cashflow"]

            df["support_payment"] = df["cfd_cashflow"]
            df["support_energy"] = e_cfd

        elif self.scheme_type == "CfD_NoRevenue_at_NegativePrices":
            if ref_freq is None:
                raise ValueError(
                    "CfD_NoRevenue_at_NegativePrices requires a reference_frequency to compute market reference price."
                )

            df["market_ref"] = df["market_avg_price"]
            raw_diff = df["strike_price"] - df["market_ref"]
            diff = raw_diff.clip(lower=0.0) if self.one_sided else raw_diff

            df["market_revenue"] = df["price"] * e_phys
            df["cfd_cashflow"] = diff * e_cfd
            df["total_revenue"] = df["market_revenue"] + df["cfd_cashflow"]

            df["support_payment"] = df["cfd_cashflow"]
            df["support_energy"] = e_cfd

        elif self.scheme_type == "Capability_CfD":
            if ref_freq is None:
                raise ValueError("Capability_CfD requires a reference_frequency to compute market reference price.")

            df["market_ref"] = df["market_avg_price"]
            df.loc[negative_mask, "market_ref"] = 0.0  # capability assumption

            diff = df["strike_price"] - df["market_ref"]
            if self.one_sided:
                diff = diff.clip(lower=0.0)

            df["market_revenue"] = df["price"] * e_phys
            df["cfd_cashflow"] = diff * e_cfd
            df["total_revenue"] = df["market_revenue"] + df["cfd_cashflow"]

            df["support_payment"] = df["cfd_cashflow"]
            df["support_energy"] = e_cfd

        elif self.scheme_type == "FiT":
            # Both options supported via fit_support_mode
            if fit_support_mode not in ("gross", "net_topup"):
                raise ValueError(f"Unsupported fit_support_mode={fit_support_mode!r}. Use 'gross' or 'net_topup'.")

            if fit_support_mode == "gross":
                # Support = total FiT payment, merchant leg is not booked
                df["market_revenue"] = 0.0
                df["total_revenue"] = df["strike_price"] * e_phys

                df["support_payment"] = df["total_revenue"]
                df["support_energy"] = e_phys

            else:  # "net_topup" (ALWAYS one-sided)
                # Book merchant + top-up (synthetic) support, total remains FiT
                df["market_revenue"] = df["price"] * e_phys

                raw_diff = df["strike_price"] - df["price"]
                diff = raw_diff.clip(lower=0.0)  # ALWAYS one-sided for FiT net_topup

                df["support_payment"] = diff * e_phys
                df["support_energy"] = e_phys

                df["total_revenue"] = df["strike_price"] * e_phys

        elif self.scheme_type == "FiP":
            # Total revenue includes merchant + premium; support is premium-only
            df["market_revenue"] = df["price"] * e_phys
            df["total_revenue"] = (df["price"] + float(self.premium)) * e_phys

            df["support_payment"] = float(self.premium) * e_phys
            df["support_energy"] = e_phys

        elif self.scheme_type == "Market":
            df["market_revenue"] = df["price"] * e_phys
            df["total_revenue"] = df["market_revenue"]

            df["support_payment"] = 0.0
            df["support_energy"] = e_phys

        else:
            raise ValueError(f"Unknown scheme_type: {self.scheme_type}")

        # Support rate (safe divide)
        df["support_rate"] = df["support_payment"] / df["support_energy"].replace({0.0: np.nan})

        # ---------- Settlement ----------
        if self.settlement_lag_months:
            df["settlement_time"] = df["timestamp"] + pd.DateOffset(months=self.settlement_lag_months)
        else:
            df["settlement_time"] = df["timestamp"]

        revenue_records = (
            df.groupby(pd.Grouper(key="settlement_time", freq=pay_freq))
            .agg(
                total_revenue=("total_revenue", "sum"),
                market_revenue=("market_revenue", "sum"),
                support_payment=("support_payment", "sum"),
                support_energy=("support_energy", "sum"),
            )
            .reset_index()
            .rename(columns={"settlement_time": "timestamp"})
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Helpful settlement-level rate (optional but consistent)
        revenue_records["support_rate"] = (
            revenue_records["support_payment"] / revenue_records["support_energy"].replace({0.0: np.nan})
        )

        # ---------------------------------------------------------------------
        # FINAL: store scheme financial metrics in revenue_metrics_records
        # (keeps MVF metrics too by merging on "period")
        # ---------------------------------------------------------------------
        if ref_freq is None or df.empty:
            scheme_metrics = pd.DataFrame(
                columns=[
                    "timestamp",
                    "period",
                    "market_revenue",
                    "support_payment",
                    "support_energy",
                    "support_rate",
                    "total_revenue",
                ]
            )
        else:
            period_freq = get_period_alias(ref_freq) or ref_freq
            df["__period2"] = df["timestamp"].dt.to_period(period_freq)

            tmp = (
                df.groupby("__period2", observed=True)
                .agg(
                    market_revenue=("market_revenue", "sum"),
                    support_payment=("support_payment", "sum"),
                    support_energy=("support_energy", "sum"),
                    total_revenue=("total_revenue", "sum"),
                )
                .reset_index()
                .rename(columns={"__period2": "period"})
            )
            tmp["support_rate"] = tmp["support_payment"] / tmp["support_energy"].replace({0.0: np.nan})
            tmp["timestamp"] = tmp["period"].dt.to_timestamp(how="start")

            scheme_metrics = tmp[
                ["timestamp", "period", "market_revenue", "support_payment", "support_energy", "support_rate", "total_revenue"]
            ].sort_values("timestamp").reset_index(drop=True)

        # Merge MVF metrics (physical) + scheme metrics
        if self.revenue_metrics_records is None or self.revenue_metrics_records.empty:
            self.revenue_metrics_records = scheme_metrics
        else:
            # self.revenue_metrics_records has: timestamp, period, energy_sum, market_avg_price, realized_price, market_value_factor
            self.revenue_metrics_records = (
                self.revenue_metrics_records.merge(scheme_metrics, on=["timestamp", "period"], how="outer")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

        return revenue_records, self.revenue_metrics_records

    def _calc_mvf_metrics(self, df: pd.DataFrame, ref_freq: str, energy_col: str = "_energy") -> pd.DataFrame:
        """
        Returns per-reference-period metrics for a chosen energy basis:
          timestamp (period start), period, energy_sum, market_avg_price, realized_price, market_value_factor

        realized_price is energy-weighted: sum(price*energy)/sum(energy)
        market_avg_price is time-weighted: mean(price)
        """
        if df.empty or ref_freq is None:
            return pd.DataFrame(
                columns=["timestamp", "period", "energy_sum", "market_avg_price", "realized_price", "market_value_factor"]
            )

        if energy_col not in df.columns:
            raise KeyError(f"Energy column {energy_col!r} not found in df.")

        period_freq = get_period_alias(ref_freq) or ref_freq
        period = df["timestamp"].dt.to_period(period_freq)

        tmp = (
            df.assign(
                __period=period,
                __energy=df[energy_col],
                price_energy=df["price"] * df[energy_col],
            )
            .groupby("__period", observed=True)
            .agg(
                energy_sum=("__energy", "sum"),
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