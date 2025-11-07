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
        self.tax_rate = get_finex_parameter(self.finex_inputs, 'FI', 'Tax', 'tax_rate')

        # depreciation parameters
        self.depreciation_method = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'method') # ["SL"]
        self.depreciation_period = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'period') # [years]
        self.depreciation_salvage_value = get_finex_parameter(self.finex_inputs, 'FI', 'Depreciation', 'salvage_value') / 100 # [% of initial CAPEX]

        self.finex_records = pd.DataFrame(columns=["timestamp", "debt_payment"])

    def calc_FINEX(self):
        total_cost = self.capex.total_cost
        n_periods = self.n_total_payments
        periods_per_year = self.n_yearly_payments
        start_debt_service = self.project_start + pd.Timedelta(hours=self.start_debt_service)

        # ---- Cost of Debt ----
        self.loan_amount = total_cost * self.debt_share
        self.cost_of_debt = (1 + self.interest_rate) ** (1 / periods_per_year) - 1  # periodic interest rate

        # ---- Cost of Equity ----
        self.equity_amount = total_cost * (1 - self.debt_share)
        if self.flag_CAPM:
            equity_rate = self.rf + self.beta * (self.rm - self.rf)
        else:
            equity_rate = self.E
        self.cost_of_equity = (1 + equity_rate) ** (1 / periods_per_year) - 1  # periodic equity rate

        # ---- WACC (Weighted Average Cost of Capital) ----
        E = self.equity_amount
        D = self.loan_amount
        V = total_cost
        Re = self.cost_of_equity
        Rd = self.cost_of_debt
        wacc_period = (E / V) * Re + (D / V) * Rd
        self.WACC_annual = (1 + wacc_period) ** periods_per_year - 1

        # ---- Debt Service Calculation (Annuity formula) ----
        annuity_factor = (wacc_period * (1 + wacc_period) ** n_periods) / \
                        ((1 + wacc_period) ** n_periods - 1)
        self.debt_service = self.loan_amount * annuity_factor

        # ---- Generate Payment Records with Timings ----
        period_delta = pd.to_timedelta(8760 / periods_per_year, unit="h")

        rows = []
        for i in range(n_periods):
            timing = start_debt_service + i * period_delta  # Timestamp
            payment = self.debt_service
            rows.append({"timestamp": timing, "debt_payment": payment})

        # Append the new records to the existing DataFrame
        self.finex_records = pd.concat([self.finex_records, pd.DataFrame(rows)], ignore_index=True)

        self.add_depreciation_schedule()

    def add_depreciation_schedule(self):
        """
        Minimal straight-line depreciation schedule based on total CAPEX.
        - Starts at self.project_start
        - Frequency: self.n_yearly_payments periods per year
        - Method: 'SL' only (straight-line)
        - Salvage value: self.depreciation_salvage_value (fraction of CAPEX)
        - Duration: self.depreciation_period (years)

        Side effect:
            Adds/merges a 'depreciation' column into self.finex_records.
        """
        import pandas as pd

        # ---- Baseline & parameters ----
        capex_total = float(getattr(self.capex, "total_cost", 0.0))
        if capex_total <= 0:
            # Nothing to depreciate; ensure column exists and exit
            if "depreciation" not in self.finex_records.columns:
                self.finex_records["depreciation"] = 0.0
            return self.finex_records

        method = (self.depreciation_method or "SL").upper()
        if method != "SL":
            raise NotImplementedError(f"Depreciation method '{self.depreciation_method}' not supported in minimal impl.")

        life_years = float(self.depreciation_period)
        if life_years <= 0:
            raise ValueError("depreciation_period must be > 0 years.")

        salvage_ratio = float(self.depreciation_salvage_value or 0.0)  # already provided as fraction (e.g., 0.05)
        salvage_value = max(0.0, capex_total * salvage_ratio)
        depreciable_base = max(0.0, capex_total - salvage_value)

        periods_per_year = int(self.n_yearly_payments)
        if periods_per_year <= 0:
            raise ValueError("n_yearly_payments must be a positive integer.")

        n_periods = int(round(life_years * periods_per_year))
        if n_periods <= 0:
            raise ValueError("Computed number of depreciation periods must be > 0.")

        per_period_dep = depreciable_base / n_periods

        # ---- Build schedule ----
        period_delta = pd.to_timedelta(8760 / periods_per_year, unit="h")
        start_ts = self.project_start

        dep_rows = [
            {"timestamp": start_ts + i * period_delta, "depreciation": per_period_dep}
            for i in range(n_periods)
        ]
        dep_df = pd.DataFrame(dep_rows)

        # ---- Merge into finex_records ----
        if self.finex_records.empty:
            self.finex_records = dep_df
        else:
            self.finex_records = (
                self.finex_records.merge(dep_df, on="timestamp", how="outer")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            # Fill missing depreciation with 0 for timestamps that have no depreciation
            if "depreciation" in self.finex_records.columns:
                self.finex_records["depreciation"] = self.finex_records["depreciation"].fillna(0.0)
            else:
                self.finex_records["depreciation"] = 0.0

        return self.finex_records


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


#########################################################Backlog###################################################################
""" 
    inflation = (1 + INFLATION_RATE_ANNUAL) \
                ** (np.arange(n_periods) * avg_period_length_years)
    opex_per  = (opex_annual / periods_per_year) * inflation

    depreciation = np.zeros(n_periods)
    book = capex
    for t in range(n_periods):
        dep = 0.15 * book
        depreciation[t] = dep / periods_per_year
        book -= depreciation[t]
        if book < 0:
            book = 0
    if depreciation.sum() > capex:                       
        depreciation *= capex / depreciation.sum() """