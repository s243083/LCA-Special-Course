import pandas as pd
import numpy as np
from core.File_Handling import load_yaml, process_duration_fields

class FINEX:
    def __init__(self, env, capex):
        self.env = env  # Access to the environment for scheduling
        self.capex = capex  # Access to CAPEX instance for cost data
        self.project_start = pd.to_datetime(self.env.config.Project_StartDate, format="%d.%m.%Y")
        self.finex_inputs= load_finex_inputs(self.env.config)
        self.debt_share= get_finex_parameter(self.finex_inputs, 'FI', 'Debt', 'debt_share') /100
        self.interest_rate= get_finex_parameter(self.finex_inputs, 'FI', 'Debt', 'interest_rate') /100
        self.n_yearly_payments= get_finex_parameter(self.finex_inputs, 'FI', 'Debt', 'n_yearly_payments')
        self.n_total_payments = get_finex_parameter(self.finex_inputs, 'FI', 'Debt', 'n_total_payments')
        self.start_debt_service = get_finex_parameter(self.finex_inputs, 'FI', 'Debt', 'start_debt_service_h')

        self.E= get_finex_parameter(self.finex_inputs, 'FI', 'Equity', 'E')/100
        self.flag_CAPM= get_finex_parameter(self.finex_inputs, 'FI', 'Equity', 'equity_costModel', 'flag_CAPM')
        self.rf= get_finex_parameter(self.finex_inputs, 'FI', 'Equity', 'equity_costModel', 'rf')/100
        self.rm= get_finex_parameter(self.finex_inputs, 'FI', 'Equity', 'equity_costModel', 'rm')/100
        self.beta= get_finex_parameter(self.finex_inputs, 'FI', 'Equity', 'equity_costModel', 'beta')
        
        # tax parameters (not yet implemented in calculations)
        self.tax_rate = get_finex_parameter(self.finex_inputs, 'FI', 'Tax', 'tax_rate')/100

        # depreciation parameters
        self.depreciation_method = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'method') # ["SL"]
        self.depreciation_period = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'period') # [years]
        self.depreciation_salvage_value = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'salvage_value') / 100 # [% of initial CAPEX]

        self.finex_records = pd.DataFrame(columns=["timestamp", "debt_payment"])

    def calc_FINEX(self):
            """
            Orchestrates FINEX: computes WACC (for valuation only), builds
            debt amortization schedule (cost_of_debt, not WACC), and depreciation.
            Populates:
                self.finex_records : DataFrame with columns
                    timestamp, payment, interest, principal, outstanding, depreciation
                self.WACC_annual  : float
            Returns self.finex_records for convenience.
            """
            # 1) parameters & timing
            total_cost = float(self.capex.total_cost)
            periods_per_year = int(self.n_yearly_payments)
            n_periods = int(self.n_total_payments)
            start_ts = pd.to_datetime(self.env.config.Project_StartDate, format="%d.%m.%Y")
            first_pay_ts = start_ts + pd.Timedelta(hours=self.start_debt_service)

            # 2) capital structure
            D0 = total_cost * self.debt_share
            E0 = total_cost - D0
            self.D0 = D0
            self.E0 = E0

            i_per = (1 + self.interest_rate) ** (1/periods_per_year) - 1  # periodic cost of debt  # 

            # equity rate (CAPM or fixed), then periodic
            self.equity_annual = (self.rf + self.beta * (self.rm - self.rf)) if self.flag_CAPM else self.E
            re_per = (1 + self.equity_annual) ** (1/periods_per_year) - 1

            # 3) WACC (for discounting; do not use for debt payments)
            Re_annual = self.equity_annual                 # already annual fraction
            Rd_annual = self.interest_rate            # already annual fraction
            V = D0 + E0
            self.WACC_annual = (E0/V)*Re_annual + (D0/V)*Rd_annual*(1 - self.tax_rate)

            # 4) amortization table (annuity by default; can extend to bullet/sculpted)
            payment = self._annuity_payment(D0, i_per, n_periods)  # uses cost_of_debt
            sched = []
            outstanding = D0
            period_delta = pd.to_timedelta(8760/periods_per_year, unit="h")
            for k in range(n_periods):
                ts = first_pay_ts + k*period_delta
                interest = outstanding * i_per
                principal = payment - interest
                outstanding = max(0.0, outstanding - principal)
                sched.append({"timestamp": ts,
                            "payment": -(payment),        # outflow
                            "interest": -(interest),
                            "principal": -(principal),
                            "outstanding": outstanding})

            debt_df = pd.DataFrame(sched)

            # 5) depreciation schedule (you already have this; reuse/extend)
            dep_df = self._build_depreciation_schedule(first_pay_ts, periods_per_year)

            # 6) merge and maintain backward compatibility (debt_payment column)
            finex = (debt_df.merge(dep_df, on="timestamp", how="outer")
                            .sort_values("timestamp")
                            .fillna({"payment":0,"interest":0,"principal":0,"outstanding":0,"depreciation":0}))


            self.finex_records = finex
            return self.finex_records


    def _build_depreciation_schedule(self, first_pay_ts: pd.Timestamp, periods_per_year: int) -> pd.DataFrame:
        """
        Build a straight-line (SL) depreciation schedule as a standalone DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ["timestamp", "depreciation"]
            - 'timestamp': period endpoints spaced evenly by 8760/periods_per_year hours from start_ts
            - 'depreciation': constant per-period charge (negative cashflow convention not applied here
                            since depreciation is non-cash; caller decides sign handling)

        Notes
        -----
        - Pure function: does NOT read or write self.finex_records.
        - Uses inputs from FINEX config:
            * self.depreciation_method  (only "SL" supported)
            * self.depreciation_period  (years, > 0)
            * self.depreciation_salvage_value (fraction of CAPEX, e.g. 0.05)
            * self.capex.total_cost
        - If CAPEX <= 0, returns an empty DataFrame with the expected columns.
        """
        # ---- Validate base inputs ----
        if periods_per_year is None or int(periods_per_year) <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        periods_per_year = int(periods_per_year)

        if not isinstance(first_pay_ts, pd.Timestamp):
            first_pay_ts = pd.to_datetime(first_pay_ts, errors="coerce")
        if pd.isna(first_pay_ts):
            raise ValueError("first_pay_ts must be a valid timestamp.")

        capex_total = float(getattr(self.capex, "total_cost", 0.0))
        if capex_total <= 0.0:
            # Nothing to depreciate → empty schedule with correct columns
            return pd.DataFrame(columns=["timestamp", "depreciation"])

        method = (self.depreciation_method or "SL").upper()
        if method != "SL":
            raise NotImplementedError(
                f"Depreciation method '{self.depreciation_method}' not supported (only 'SL')."
            )

        life_years = float(self.depreciation_period)
        if life_years <= 0.0:
            raise ValueError("depreciation_period must be > 0 years.")

        salvage_ratio = float(self.depreciation_salvage_value or 0.0)  # already a fraction (e.g., 0.05)
        salvage_value = max(0.0, capex_total * salvage_ratio)
        depreciable_base = max(0.0, capex_total - salvage_value)

        n_periods = int(round(life_years * periods_per_year))
        if n_periods <= 0:
            raise ValueError("Computed number of depreciation periods must be > 0.")

        per_period_dep = depreciable_base / n_periods

        # ---- Build schedule (evenly spaced hours; matches your current convention) ----
        period_delta = pd.to_timedelta(8760 / periods_per_year, unit="h")
        dep_rows = (
            {"timestamp": first_pay_ts + i * period_delta, "depreciation": per_period_dep}
            for i in range(n_periods)
        )
        dep_df = pd.DataFrame(dep_rows, columns=["timestamp", "depreciation"])

        # Ensure proper dtypes
        dep_df["timestamp"] = pd.to_datetime(dep_df["timestamp"], errors="coerce")
        dep_df["depreciation"] = dep_df["depreciation"].astype(float)

        return dep_df
    
    def _annuity_payment(self, P, i, n): return P * (i*(1+i)**n) / ((1+i)**n - 1) if n>0 else 0.0


    def get_cost_dataframe(self):
        """Converts the cost records list to a DataFrame for further analysis."""
        return pd.DataFrame(self.finex_records)


def load_finex_inputs(config):
    """
    Loads FINEX input parameters from configuration files, 
    converting any duration parameters to hours.

    Parameters
    ----------
    config : Configuration
        The configuration object containing paths to FINEX input files.

    Returns
    -------
    dict
        Dictionary containing FINEX data, with duration parameters converted to hours where applicable.
    """
    finex_input = {}

    # Load FINEX input files
    if hasattr(config, 'Finex_inputFiles'):
        for identifier, file_name in config.Finex_inputFiles.items():
            finex_input[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            finex_input[identifier] = process_duration_fields(finex_input[identifier])  # Process for duration fields
        print("Loaded Finex data structure:", finex_input)

    return finex_input



def get_finex_parameter(finex_data, identifier, entry_name, *keys):
    try:
        entries = finex_data[identifier]['FINEX']
    except KeyError as e:
        raise KeyError(f"Missing expected key: {e}")

    for item in entries:
        if item.get('name') == entry_name:
            value = item.get('Parameters', {})
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    return None
                value = value[key]
            return value
    return None


