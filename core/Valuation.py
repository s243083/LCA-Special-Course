from core.File_Handling import load_yaml, process_duration_fields
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates





# NEW imports for interactivity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.utils import apply_overrides , get_input_parameter

class Valuation:
    def __init__(self, env):
        self.env = env
        self.valuationInput = load_valuationInput(self.env.config)
        self.cf_aggregation_period = get_input_parameter(self.valuationInput, 'VA', 'cf_aggregation_period')

        # Apply any overrides from config to valuationInput
        if hasattr(self.env.config, 'Valuation_overrides'):
            apply_overrides(self.valuationInput, self.env.config.Valuation_overrides)

        # project start retained if needed elsewhere, but aggregation uses timestamps directly
        self.project_start = self.env.config.Project_StartDate

        # Expect these to already be DataFrames
        # capex_costs: columns include ['timestamp','cost','category_name','subcategory_name','item_name', ...]
        # finex_costs: columns include ['timestamp','debt_payment']
        # revenue_records: columns include ['timestamp','revenue']
        self.capex_costs = pd.DataFrame
        self.finex_costs = pd.DataFrame
        self.revenue_records = pd.DataFrame
        self.opex_costs = pd.DataFrame

        self.cashflow_records = pd.DataFrame
        self.cashflow_discounted_records = pd.DataFrame

        self.valuemetrics = pd.DataFrame
    # ------------------------------ Public API ------------------------------

    def project_valuation(self):
            self.capex_costs = self.env.capex.cost_records
            self.finex_records = self.env.finex.finex_records
            self.revenue_records = self.env.RevenueModel.revenue_records
            self.power_records = self.env.windFarm.power_records
            self.opex_costs = self.env.opex.OPEX_records
            self.tax_rate = self.env.finex.tax_rate
            self.discount_rate = self.env.finex.WACC_annual


            # Build and store both undiscounted & discounted cash-flow tables
            self.calculate_cash_flows()

            #print(self.cashflow_discounted_records)  # or self.cashflow_records
            self.valuation()

    # ------------------------------ Cash-flow engine ------------------------------
    def calculate_cash_flows(self):
        """
        Populates:
          - self.unified_records (tall records, undiscounted)
          - self.cashflow_records (aggregated by period, undiscounted)
          - self.cashflow_discounted_records (aggregated + discount columns when rate given)

        Returns the discounted DataFrame if discount_rate is provided,
        otherwise the undiscounted aggregated DataFrame.
        """

        
        # 1) Build unified tall records once
        self.unified_records = pd.concat(
            [self._capex_df(), self._finex_df(), self._revenue_df(), self._opex_df()],
            ignore_index=True
        ).dropna(subset=["timestamp"]).sort_values("timestamp")

        # 2) Aggregate (UNDISCOUNTED) and store
        self.cashflow_records = self.aggregate_cash_flows(self.unified_records)

        # 3) Optionally add discounting and store the discounted table
        if self.discount_rate is not None:
            self.cashflow_discounted_records = self.discount_cash_flows(
                self.cashflow_records
            )

    # ------------------------------ Normalization ------------------------------

    def _ensure_ts(self, df, col="timestamp"):
        out = df.copy()
        if col not in out.columns:
            raise ValueError(f"Expected '{col}' column in DataFrame.")
        out[col] = pd.to_datetime(out[col], errors="coerce")
        return out

    def _capex_df(self):
        df = self._ensure_ts(self.capex_costs)
        # Negative = outflow
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "amount": -df["cost"].astype(float),
            "type": "capex",
            "category": df.get("category_name"),
            "subcategory": df.get("subcategory_name"),
            "label": df.get("item_name")
        })
        return out

    def _finex_df(self):
        """
        Produces tall records from FINEX:
        - 'finex' rows for debt payments (cash outflow)
        - 'depreciation' rows if available (non-cash value preserved)
        """
        df = self._ensure_ts(self.finex_records)

        frames = []

        # Debt payments (cash)
        if "debt_payment" in df.columns:
            frames.append(pd.DataFrame({
                "timestamp": df["timestamp"],
                "amount": -df["debt_payment"].fillna(0.0).astype(float),  # outflow
                "type": "finex",
                "category": None,
                "subcategory": None,
                "label": "debt_payment"
            }))

        # Depreciation (non-cash value preserved; exclude from net cash later)
        if "depreciation" in df.columns:
            frames.append(pd.DataFrame({
                "timestamp": df["timestamp"],
                "amount": -df["depreciation"].fillna(0.0).astype(float),
                "type": "depreciation",
                "category": None,
                "subcategory": None,
                "label": "depreciation"
            }))

        return pd.concat(frames, ignore_index=True)

    def _revenue_df(self):
        df = self._ensure_ts(self.revenue_records)
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "amount": df["total_revenue"].astype(float),  # inflow
            "type": "revenue",
            "category": None,
            "subcategory": None,
            "label": "revenue"
        })

        return out
    
    def _opex_df(self):
        df = self._ensure_ts(self.opex_costs)
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "amount": -df["OM_payment"].astype(float),
            "type": "opex",
            "category": None,
            "subcategory": None,
            "label": "opex"
        })
        return out

    # ------------------------------ Aggregation & Discounting ------------------------------

    def aggregate_cash_flows(self, records_df):
        """
        Aggregates raw records into period cash flows.

        Produces:
        - free_cash_flow: FCFF using taxes on EBIT (EBIT = revenue + opex + depreciation),
        then add back depreciation (non-cash), include capex, exclude finex.
        - net_cash_flow: includes finex (sum of revenue + opex + capex + finex), excludes depreciation.

        Notes:
        - Assumes amounts in records_df['amount'] are already signed:
            revenue > 0, opex/capex/finex < 0, depreciation < 0 (non-cash).
        - Taxes are applied to positive EBIT only (no tax credit on negative EBIT).
        """
        import pandas as pd
        import numpy as np
        period = self.cf_aggregation_period

        freq_map = {"monthly": "M", "quarterly": "Q", "yearly": "A"}
        if period not in freq_map:
            raise ValueError("period must be one of: 'monthly', 'quarterly', 'yearly'.")

        freq = freq_map[period]
        df = records_df.copy().sort_values("timestamp")

        grouped = (
            df.groupby([pd.Grouper(key="timestamp", freq=freq), "type"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )

        # Ensure columns exist even if missing in data
        for col in ("capex", "finex", "opex", "revenue", "depreciation"):
            if col not in grouped.columns:
                grouped[col] = 0.0

        # -------------------- Net cash flow (exclude depreciation) --------------------
        grouped["net_cash_flow"] = grouped[["revenue", "opex", "capex", "finex"]].sum(axis=1)

        # -------------------- Free cash flow (tax on EBIT) ---------------------------
        tax_rate = float(getattr(self, "tax_rate", 0.0) or 0.0)

        # EBIT includes depreciation (non-cash) to capture the tax shield
        ebit = grouped["revenue"] + grouped["opex"] + grouped["depreciation"]

        # Tax only when EBIT > 0 (no immediate tax credit on negative EBIT)
        tax = np.where(ebit > 0.0, ebit * tax_rate, 0.0)

        # NOPAT = EBIT - tax
        nopat = ebit - tax

        # Add back depreciation (it's negative in the table, so add back -depreciation)
        dep_addback = -grouped["depreciation"]

        # FCFF: NOPAT + Depreciation add-back + CapEx (signed; usually negative). Excludes FinEx.
        grouped["free_cash_flow"] = nopat + dep_addback + grouped["capex"]

        # Turn the index back into a column (period end timestamps)
        grouped = grouped.reset_index().rename(columns={"timestamp": "period_end"})
        return grouped




    def discount_cash_flows(self, df):
        """
        Apply discounting to both free_cash_flow and net_cash_flow.

        Args:
            df: DataFrame with at least 'period_end', 'free_cash_flow', 'net_cash_flow'
            discount_rate_annual: annual discount rate (e.g., 0.02 for 2%)
            period: one of {"monthly", "quarterly", "yearly"}
        """
        discount_rate_annual = self.discount_rate
        period = self.cf_aggregation_period

        periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[period]
        r = discount_rate_annual / periods_per_year

        out = df.copy().sort_values("period_end").reset_index(drop=True)
        out["t"] = np.arange(len(out))  # 0,1,2,... per period
        out["discount_factor"] = (1.0 / (1.0 + r)) ** out["t"]

        # Discount both cash flow types
        if "net_cash_flow" in out:
            out["discounted_net_cash_flow"] = out["net_cash_flow"] * out["discount_factor"]
        if "free_cash_flow" in out:
            out["discounted_free_cash_flow"] = out["free_cash_flow"] * out["discount_factor"]

        return out

    def valuation(self):
        """
        Uses already-created inputs:
        - self.cashflow_records (UNDISCOUNTED, with capex/finex/opex/revenue/free_cash_flow/net_cash_flow)
        - self.cashflow_discounted_records (has discount_factor & discounted_* columns)
        - self.power_records (optional, for LCOE)

        Returns dict with NPV (from discounted FREE cash flow), 
        IRR (from UNdiscounted FREE cash flow), 
        LCOE (if computable), and the discounted CF table.
        """
        import pandas as pd

        cf_disc = self.cashflow_discounted_records
        cf_undisc = self.cashflow_records
        aggregation_period = self.cf_aggregation_period
    

        # ---------- NPV (from DISCOUNTED FREE cash flows) ----------
        self.npv = (
            float(cf_disc["discounted_free_cash_flow"].sum())
            if isinstance(cf_disc, pd.DataFrame) and "discounted_free_cash_flow" in cf_disc.columns
            else None
        )
        print(f"NPV (from FCF): {self.npv}")

        # ---------- IRR (from UNdiscounted FREE cash flows) ----------
        self.irr = None
        if isinstance(cf_undisc, pd.DataFrame) and not cf_undisc.empty and "free_cash_flow" in cf_undisc.columns:
            irr_series = (
                cf_undisc.sort_values("period_end")["free_cash_flow"]
                .astype(float)
                .to_numpy()
            )
            self.irr = self._compute_irr(irr_series)
        print(f"IRR (from FCF): {self.irr}")

        # ---------- LCOE (optional; requires self.power_records) ----------
        self.lcoe = None
        if isinstance(getattr(self, "power_records", None), pd.DataFrame):
            base = cf_disc[["period_end", "discount_factor", "capex", "finex", "opex"]].copy()
            base[["capex", "finex", "opex"]] = base[["capex", "finex", "opex"]].fillna(0.0)
            # Treat spends as positive in the numerator
            base["costs"] = -(base["capex"] + base["opex"])

            freq = {"monthly": "M", "quarterly": "Q", "yearly": "A"}[aggregation_period]

            pr = self.power_records.copy()
            pr.columns = ["timestamp", "Total_Production"]
            pr["timestamp"] = pd.to_datetime(pr["timestamp"], errors="coerce")
            pr["Total_Production"] = pd.to_numeric(pr["Total_Production"], errors="coerce")
            pr = (
                pr.dropna(subset=["timestamp", "Total_Production"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            energy_agg = (
                pr.groupby(pd.Grouper(key="timestamp", freq=freq))["Total_Production"]
                .sum()
                .reset_index()
                .rename(columns={"timestamp": "period_end", "Total_Production": "energy"})
            )

            lcoe_df = (
                base.merge(energy_agg, on="period_end", how="left")
                    .fillna({"energy": 0.0})
                    .sort_values("period_end")
                    .reset_index(drop=True)
            )
            lcoe_df["pv_costs"] = lcoe_df["costs"] * lcoe_df["discount_factor"]
            lcoe_df["pv_energy"] = lcoe_df["energy"] * lcoe_df["discount_factor"]

            denom = float(lcoe_df["pv_energy"].sum())
            self.lcoe = float(lcoe_df["pv_costs"].sum() / denom) if denom > 0 else None
            print(f"LCOE: {self.lcoe}")

        # ---------- append to self.valuemetrics ----------
        metrics_row = {
            "npv": self.npv,
            "irr": self.irr,
            "lcoe": self.lcoe,
        }

        row_df = pd.DataFrame([metrics_row])
        if not isinstance(getattr(self, "valuemetrics", None), pd.DataFrame) or self.valuemetrics.empty:
            self.valuemetrics = row_df
        else:
            self.valuemetrics = pd.concat([self.valuemetrics, row_df], ignore_index=True)



    def _compute_irr(self, cashflows):
        """
        IRR via numpy_financial if available; otherwise Newton's method fallback.
        cashflows: 1D array-like of period cash flows (t=0,1,2,...)
        Returns float (periodic IRR), converted to annualized based on aggregation period
        to be comparable with annual discount rates.
        """
        if cashflows is None or len(cashflows) == 0:
            return None

        # quick exit if all same sign (no IRR)
        if np.all(cashflows >= 0) or np.all(cashflows <= 0):
            return None

        # try numpy_financial
        irr = None

        import numpy_financial as npf
        irr_periodic = npf.irr(cashflows)
        irr = self._annualize_rate(irr_periodic)
        return irr

    def _annualize_rate(self, r_periodic):
        if r_periodic is None or not np.isfinite(r_periodic):
            return None
        # map aggregation period to periods/year
        agg = self.cf_aggregation_period
        periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}.get(agg, 12)
        try:
            return (1.0 + r_periodic) ** periods_per_year - 1.0
        except Exception:
            return None
        


    def plot_cash_flows(
        self,
        use_discounted: bool = False,
        include_net_line: bool = True,
        figsize=(12, 6),
        bar_width: float = 0.2,
        title: str | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        interactive: bool = True,
    ):
        # pick source
        df = self.cashflow_discounted_records if use_discounted else self.cashflow_records
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Cash-flow table is empty. Run calculate_cash_flows() first.")

        # ensure required cols
        cols = ["capex", "finex", "opex", "revenue", "net_cash_flow", "period_end"]
        for c in cols:
            if c not in df.columns:
                if c == "period_end":
                    raise ValueError("Expected 'period_end' in cash-flow table.")
                df[c] = 0.0

        # dates
        base = df.copy()
        base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
        if base["period_end"].isna().any():
            raise ValueError("Found invalid 'period_end' timestamps in cash-flow table.")

        # filter if needed
        if start_date or end_date:
            sd = pd.to_datetime(start_date, errors="coerce") if start_date else None
            ed = pd.to_datetime(end_date, errors="coerce") if end_date else None
            mask = pd.Series(True, index=base.index)
            if sd is not None:
                mask &= base["period_end"] >= sd
            if ed is not None:
                mask &= base["period_end"] <= ed
            base = base.loc[mask]
            if base.empty:
                raise ValueError("No cash flows in the selected window.")

        data = base.sort_values("period_end").reset_index(drop=True)
        series_order = ["capex", "finex", "opex", "revenue"]
        plot_title = title or ("Discounted Cash Flows by Period" if use_discounted else "Cash Flows by Period")

        # -------- interactive branch (Plotly) --------
        if interactive:
            fig = make_subplots(specs=[[{"secondary_y": include_net_line}]])
            for i, s in enumerate(series_order):
                fig.add_trace(
                    go.Bar(
                        x=data["period_end"],
                        y=data[s].astype(float),
                        name=s,
                        offsetgroup=str(i),
                    ),
                    secondary_y=False,
                )
            if include_net_line:
                fig.add_trace(
                    go.Scatter(
                        x=data["period_end"],
                        y=data["net_cash_flow"].astype(float),
                        name="net_cash_flow",
                        mode="lines+markers",
                    ),
                    secondary_y=True,
                )

            fig.update_layout(
                barmode="group",
                title=plot_title,
                hovermode="x unified",
                width=int(figsize[0] * 80),
                height=int(figsize[1] * 80),
            )
            fig.update_xaxes(title_text="Period end", tickformat="%Y-%m")
            fig.update_yaxes(title_text="Cash flow", secondary_y=False)
            if include_net_line:
                fig.update_yaxes(title_text="Net cash flow", secondary_y=True)

            fig.show()
            return fig

        # -----------------------
        # Matplotlib fallback
        # -----------------------
        # (Your original logic, unchanged)
        y = [data[s].astype(float).values for s in series_order]
        n = len(series_order)
        group_gap = 0.2
        group_width = n * bar_width
        centers = np.arange(len(data)) * (group_width + group_gap)
        offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width

        fig, ax = plt.subplots(figsize=figsize)
        for i, (name, yi) in enumerate(zip(series_order, y)):
            ax.bar(centers + offsets[i], yi, width=bar_width, label=name)

        ax.axhline(0, linewidth=1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        dates = data["period_end"]
        max_labels = 12
        n_points = len(dates)
        step = max(1, int(np.ceil(n_points / max_labels)))
        tick_idx = np.arange(0, n_points, step)
        if len(tick_idx) == 0 or tick_idx[-1] != n_points - 1:
            tick_idx = np.unique(np.append(tick_idx, n_points - 1))

        ax.set_xticks(centers[tick_idx])
        ax.set_xticklabels([dates.iloc[i].strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")

        if include_net_line:
            ax2 = ax.twinx()
            ax2.plot(centers, data["net_cash_flow"].values, marker="o", linestyle="-", label="net_cash_flow")
            ax2.set_ylabel("Net cash flow")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best")
        else:
            ax.legend(loc="best")

        ax.set_title(plot_title)
        ax.set_xlabel("Period end")
        ax.set_ylabel("Cash flow")

        pad = group_gap / 2
        ax.set_xlim(centers[0] - group_width / 2 - pad, centers[-1] + group_width / 2 + pad)

        fig.tight_layout()
        plt.show()
        return fig, ax

    

    def plot_cash_flows_lines(
        self,
        use_discounted: bool = False,
        include_net_line: bool = True,
        figsize=(12, 6),
        title: str | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ):
        """
        Plot time series lines of capex, finex, opex, revenue per aggregated period.
        Optionally restrict the plot to a date window.

        Parameters
        ----------
        use_discounted : bool
            If True, use self.cashflow_discounted_records; otherwise use self.cashflow_records.
        include_net_line : bool
            If True, overlays a line for net cash flow per period.
        figsize : tuple
            Matplotlib figure size.
        title : str | None
            Optional custom title.
        start_date : str | pd.Timestamp | None
            Inclusive start of the window (compared to 'period_end'). If None, unbounded on the left.
        end_date : str | pd.Timestamp | None
            Inclusive end of the window (compared to 'period_end'). If None, unbounded on the right.

        Returns
        -------
        (fig, ax)
            Matplotlib figure and axis.
        """
        # Pick the source table
        df = self.cashflow_discounted_records if use_discounted else self.cashflow_records
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Cash-flow table is empty. Run calculate_cash_flows() first.")

        # Ensure required columns
        cols = ["capex", "finex", "opex", "revenue", "net_cash_flow", "period_end"]
        for c in cols:
            if c not in df.columns:
                if c == "period_end":
                    raise ValueError("Expected 'period_end' in cash-flow table.")
                df[c] = 0.0

        # Coerce period_end to datetime and filter window
        base = df.copy()
        base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
        if base["period_end"].isna().any():
            raise ValueError("Found invalid 'period_end' timestamps in cash-flow table.")

        if start_date is not None or end_date is not None:
            sd = pd.to_datetime(start_date, errors="coerce") if start_date is not None else None
            ed = pd.to_datetime(end_date, errors="coerce") if end_date is not None else None
            if start_date is not None and pd.isna(sd):
                raise ValueError(f"Could not parse start_date={start_date!r}")
            if end_date is not None and pd.isna(ed):
                raise ValueError(f"Could not parse end_date={end_date!r}")

            mask = pd.Series(True, index=base.index)
            if sd is not None:
                mask &= base["period_end"] >= sd
            if ed is not None:
                mask &= base["period_end"] <= ed

            base = base.loc[mask]
            if base.empty:
                raise ValueError("No cash flows in the selected window.")

        # Sort by time
        data = base.sort_values("period_end").reset_index(drop=True)
        dates = data["period_end"]

        # Series to plot
        series_order = ["capex", "finex", "opex", "revenue"]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=figsize)

        for name in series_order:
            yi = data[name].astype(float).values
            ax.plot(dates, yi, label=name, linewidth=1.8)

        if include_net_line:
            ax.plot(dates, data["net_cash_flow"].values, label="net_cash_flow",
                    linewidth=2.2, linestyle="--", marker=None)

        # Baseline, grid, labels
        ax.axhline(0, linewidth=1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Thin x tick labels (max ~12) for readability
        max_labels = 12
        n_points = len(dates)
        if n_points:
            step = max(1, int(np.ceil(n_points / max_labels)))
            tick_idx = np.arange(0, n_points, step)
            if tick_idx[-1] != n_points - 1:
                tick_idx = np.append(tick_idx, n_points - 1)
            ax.set_xticks(dates.iloc[tick_idx])
            ax.set_xticklabels([dates.iloc[i].strftime("%Y-%m") for i in tick_idx],
                            rotation=45, ha="right")

        # Titles & axes labels
        default_title = "Discounted Cash Flows (lines)" if use_discounted else "Cash Flows (lines)"
        ax.set_title(title or default_title)
        ax.set_xlabel("Period end")
        ax.set_ylabel("Cash flow")

        # Compact legend
        num_lines = len(series_order) + (1 if include_net_line else 0)
        ax.legend(loc="best", ncol=2 if num_lines >= 5 else 1, frameon=True)

        fig.tight_layout()
        plt.show()
        return fig, ax



def load_valuationInput(config):
    """
    Loads wind farm input parameters from the configuration file.

    Returns
    -------
    dict
        Dictionary with wind farm parameters.
    """
    valuationInput = {}

    if hasattr(config, 'Valuation_inputFiles'):
        for identifier, file_name in config.Valuation_inputFiles.items():
            valuationInput[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            valuationInput[identifier] = process_duration_fields(valuationInput[identifier])

    return valuationInput
