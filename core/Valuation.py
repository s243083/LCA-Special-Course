from core.File_Handling import load_yaml, process_duration_fields
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# removed: matplotlib.dates as mdates (unused)

# Interactivity
import plotly.graph_objects as go
from plotly.subplots import make_subplots



from core.utils import apply_overrides, get_input_parameter
from core.CashFlowEngine import CashFlowEngine


class Valuation:
    def __init__(self, env):
        self.env = env
        self.valuationInput = load_valuationInput(self.env.config)
        self.cf_aggregation_period = get_input_parameter(self.valuationInput, 'VA', 'cf_aggregation_period')

        # Apply any overrides from config to valuationInput
        if hasattr(self.env.config, 'Valuation_overrides'):
            apply_overrides(self.valuationInput, self.env.config.Valuation_overrides)

        # project start retained if needed elsewhere
        self.project_start = self.env.config.Project_StartDate

        self.power_records = self.env.windFarm.power_records if hasattr(self.env, "windFarm") else None
        

        # Placeholders (instances, not classes)
        self.capex_costs = pd.DataFrame()
        self.finex_costs = pd.DataFrame()
        self.revenue_records = pd.DataFrame()
        self.opex_costs = pd.DataFrame()

        self.cashflow_records = pd.DataFrame()
        self.cashflow_discounted_records = pd.DataFrame()
        self.valuemetrics = pd.DataFrame()

        self.discount_rate = None
        self.tax_rate = None
        self.power_records = None

    # ------------------------------ Public API ------------------------------
    def project_valuation(self):
        # 1) FINEX
        finex_df = self.env.finex.calc_FINEX()
        self.wacc_annual = self.env.finex.WACC_annual
        self.tax_rate = self.env.finex.cfg.tax_rate or 0.0  # fraction (e.g., 0.22)

        # cost of equity (annual)
        self.cost_of_equity_annual = self.env.finex.equity_annual

        # lte costs (if any)
        lte_df = None
        if hasattr(self.env, "lifetimeExtension") and hasattr(self.env.lifetimeExtension, "cost_records"):
            lte_df = self.env.lifetimeExtension.cost_records



        # Optional power for LCOE
        if hasattr(self.env, "windFarm") and hasattr(self.env.windFarm, "power_records"):
            self.power_records = self.env.windFarm.power_records

        # 2) CashFlowEngine
        start_ts = pd.to_datetime(self.env.config.Project_StartDate, format="%d.%m.%Y")

        # Prefer actual data horizons (covers LTE extensions)
        candidates = []

        def _max_ts(df, col="timestamp"):
            if isinstance(df, pd.DataFrame) and (not df.empty) and (col in df.columns):
                ts = pd.to_datetime(df[col], errors="coerce")
                m = ts.max()
                return m if pd.notna(m) else None
            return None

        # WindFarm (power)
        candidates.append(_max_ts(getattr(self.env.windFarm, "power_records", None)))

        # Revenue
        candidates.append(_max_ts(getattr(self.env.RevenueModel, "revenue_records", None)))

        # OPEX
        candidates.append(_max_ts(getattr(self.env.opex, "OPEX_records", None)))

        # FINEX
        candidates.append(_max_ts(finex_df))

        # CAPEX
        candidates.append(_max_ts(getattr(self.env.capex, "cost_records", None)))


        # LTE costs (one-offs)
        if lte_df is not None:
            candidates.append(_max_ts(lte_df))

        # Choose the latest valid timestamp
        op_end_ts = max([t for t in candidates if t is not None], default=None)

        # Fallback to config-based end if nothing else exists
        if op_end_ts is None:
            end_h = float(getattr(self.env.config, "WF_OperationsEnd_h", 0.0) or 0.0)
            op_end_ts = start_ts + pd.to_timedelta(end_h, unit="h")

        # Month-start calendar up to *actual* end
        calendar = pd.date_range(start=start_ts, end=op_end_ts, freq="MS")


        # pass it to the CashFlowEngine
        cfe = CashFlowEngine(self.env, calendar=calendar)
        wf = cfe.run_waterfall(
            capex_df=self.env.capex.cost_records,
            opex_df=self.env.opex.OPEX_records,
            revenue_df=self.env.RevenueModel.revenue_records,
            finex_df=finex_df,
            tax_rate=self.tax_rate,
            lte_df=lte_df,
        )

        # 3) Aggregate & discount
        self.cashflow_records = self.aggregate_cash_flows(wf)
        self.cashflow_discounted_records = self.discount_cash_flows(self.cashflow_records)

        # 4) KPIs
        self.valuation()
        
        return self.cashflow_discounted_records


    def plot_valuation_results(self):

        # 1) Executive overview dashboard
        self.overview_dashboard(scenario_name="Base")
        # Cash flow plot
        self.plot_cash_flows()

        # 2) NPV contribution bridge (equity and firm views)
        self.pv_bridge(use_equity=True)   # → NPV (Equity)


        self.pv_bridge(use_equity=False)    # → NPV (Firm)


        # 3) DSCR calendar heatmap
        self.dscr_heatmap()

        return None


    # ------------------------------ Aggregation (WIDE) ------------------------------
    def aggregate_cash_flows(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        period = self.cf_aggregation_period
        freq_map = {"monthly": "ME", "quarterly": "Q", "yearly": "A"}
        if period not in freq_map:
            raise ValueError("cf_aggregation_period must be one of: 'monthly', 'quarterly', 'yearly'.")
        freq = freq_map[period]

        df = df_wide.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Sum numeric columns (except DSCR which we average)
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        sum_cols = [c for c in numcols if c != "DSCR"]

        agg_sum = df.groupby(pd.Grouper(key="timestamp", freq=freq))[sum_cols].sum()
        out = agg_sum
        if "DSCR" in df.columns:
            out = out.join(df.groupby(pd.Grouper(key="timestamp", freq=freq))["DSCR"].mean(), how="left")

        out = out.reset_index().rename(columns={"timestamp": "period_end"})

        # Equity "net" line for convenience
        if "Equity_CF" in out.columns:
            out["net_cash_flow"] = out["Equity_CF"]

        # ---- FCFF (for NPV_firm) ----
        needed = {"EBIT", "Tax", "depreciation", "capex"}
        if needed.issubset(out.columns):
            nopat = out["EBIT"] + out["Tax"]              # Tax is negative → EBIT - tax
            dep_addback = -out["depreciation"]            # depreciation stored as negative → add back as positive
            out["FCFF"] = nopat + dep_addback + out["capex"]
        else:
            # If any component is missing, create FCFF with NaNs to avoid silent misuse
            out["FCFF"] = np.nan

        return out


    # ------------------------------ Discounting ------------------------------
    def discount_cash_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("cashflow table is empty. Run project_valuation() first.")

        if self.wacc_annual is None or self.cost_of_equity_annual is None:
            raise ValueError("WACC or cost of equity not set. Ensure project_valuation() computed them.")

        period = self.cf_aggregation_period
        ppy = {"monthly": 12, "quarterly": 4, "yearly": 1}[period]

        r_wacc   = (1.0 + float(self.wacc_annual))**(1.0/ppy)   - 1.0
        r_equity = (1.0 + float(self.cost_of_equity_annual))**(1.0/ppy) - 1.0

        out = df.copy().sort_values("period_end").reset_index(drop=True)
        out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
        if out["period_end"].isna().any():
            raise ValueError("Invalid 'period_end' timestamp(s) after aggregation.")

        out["t"] = np.arange(len(out))

        out["df_wacc"] = (1.0 / (1.0 + r_wacc)) ** out["t"]
        out["df_equity"] = (1.0 / (1.0 + r_equity)) ** out["t"]

        if "Equity_CF" in out.columns:
            out["discounted_equity_cf"] = out["Equity_CF"] * out["df_equity"]
        if "FCFF" in out.columns:
            out["discounted_fcff"] = out["FCFF"] * out["df_wacc"]
        if "CFADS" in out.columns:
            out["discounted_cfads"] = out["CFADS"] * out["df_wacc"]

        return out


    # ------------------------------ Energy helper ------------------------------
    def _total_energy_mwh(self) -> float | None:
        """
        Total produced electricity over the full available lifetime (MWh),
        based on windFarm.power_records['_energy'].
        Returns None if power records are missing/invalid.
        """
        pr = getattr(self, "power_records", None)
        if not isinstance(pr, pd.DataFrame) or pr.empty:
            return None
        if "_energy" not in pr.columns:
            return None

        energy = pd.to_numeric(pr["_energy"], errors="coerce")
        total = float(energy.dropna().sum())
        return total if np.isfinite(total) else None


    # ------------------------------ KPIs ------------------------------
    def valuation(self):
        """
        KPIs:
          - NPV (discounted Equity_CF)
          - IRR (undiscounted Equity_CF series)
          - LCOE (if power records provided) using PV(capex+opex) / PV(energy)
        """
        cf_disc = self.cashflow_discounted_records
        cf_undisc = self.cashflow_records
        # ---- NPVs ----
        self.npv_equity = (
            float(cf_disc["discounted_equity_cf"].sum())
            if isinstance(cf_disc, pd.DataFrame) and "discounted_equity_cf" in cf_disc.columns
            else None
        )
        self.npv_firm = (
            float(cf_disc["discounted_fcff"].sum())
            if isinstance(cf_disc, pd.DataFrame) and "discounted_fcff" in cf_disc.columns
            else None
        )
        #print(f"NPV_equity: {self.npv_equity}")
        #print(f"NPV_firm:   {self.npv_firm}")

        # ---- IRR (Equity) ----
        self.irr = None
        if isinstance(cf_undisc, pd.DataFrame) and not cf_undisc.empty and "Equity_CF" in cf_undisc.columns:
            irr_series = (
                cf_undisc.sort_values("period_end")["Equity_CF"]
                .astype(float)
                .to_numpy()
            )
            self.irr = self._compute_irr(irr_series)
        #print(f"IRR (Equity_CF): {self.irr}")
        # ---- LCOE (optional; excludes financing) ----
        self.lcoe = None
        if isinstance(getattr(self, "power_records", None), pd.DataFrame):
            base_cols = ["period_end", "df_wacc", "capex", "opex"]
            has_lte = isinstance(cf_disc, pd.DataFrame) and ("lte" in cf_disc.columns)

            if has_lte:
                base_cols.append("lte")

            if all(c in cf_disc.columns for c in base_cols):
                base = cf_disc[base_cols].copy().fillna(0.0)

                # capex/opex/lte are expected as cashflows (typically negative for costs)
                # LCOE numerator wants positive costs
                if has_lte:
                    base["costs"] = -(base["capex"] + base["opex"] + base["lte"])
                else:
                    base["costs"] = -(base["capex"] + base["opex"])

                freq = {"monthly": "ME", "quarterly": "Q", "yearly": "A"}[self.cf_aggregation_period]

                pr = self.power_records.copy()

                # Expect internal convention: timestamp + Total_Power + _energy
                if "timestamp" not in pr.columns:
                    raise KeyError("power_records missing required 'timestamp' column for LCOE.")
                if "_energy" not in pr.columns:
                    raise KeyError("power_records missing required '_energy' column (MWh per row) for LCOE.")

                pr["timestamp"] = pd.to_datetime(pr["timestamp"], errors="coerce")
                pr["_energy"] = pd.to_numeric(pr["_energy"], errors="coerce")
                pr = pr.dropna(subset=["timestamp", "_energy"]).sort_values("timestamp")

                # Aggregate energy to the same period frequency as cashflows
                energy_agg = (
                    pr.groupby(pd.Grouper(key="timestamp", freq=freq))["_energy"]
                    .sum()
                    .reset_index()
                    .rename(columns={"timestamp": "period_end", "_energy": "energy"})
                )

                lcoe_df = (
                    base.merge(energy_agg, on="period_end", how="left")
                        .fillna({"energy": 0.0})
                        .sort_values("period_end")
                        .reset_index(drop=True)
                )
                lcoe_df["pv_costs"] = lcoe_df["costs"] * lcoe_df["df_wacc"]
                lcoe_df["pv_energy"] = lcoe_df["energy"] * lcoe_df["df_wacc"]

                denom = float(lcoe_df["pv_energy"].sum())
                self.lcoe = float(lcoe_df["pv_costs"].sum() / denom) if denom > 0 else None

                #print(f"LCOE: {self.lcoe}")

        # ---- Total energy (lifetime, MWh) ----
        self.total_energy_mwh = self._total_energy_mwh()

        # ---- MVF (optional) ----
        self.avg_mvf = None
        mvf_df = None

        # Try to read MVF metrics from RevenueModel (preferred)
        if hasattr(self.env, "RevenueModel") and hasattr(self.env.RevenueModel, "revenue_metrics_records"):
            mvf_df = self.env.RevenueModel.revenue_metrics_records

        if isinstance(mvf_df, pd.DataFrame) and (not mvf_df.empty) and ("market_value_factor" in mvf_df.columns):
            tmp = mvf_df.copy()

            tmp["market_value_factor"] = pd.to_numeric(tmp["market_value_factor"], errors="coerce")
            tmp = tmp.dropna(subset=["market_value_factor"])

            if not tmp.empty:
                # Prefer energy-weighted average if available
                if "energy_sum" in tmp.columns:
                    tmp["energy_sum"] = pd.to_numeric(tmp["energy_sum"], errors="coerce").fillna(0.0)
                    w = tmp["energy_sum"].clip(lower=0.0)
                    denom = float(w.sum())
                    self.avg_mvf = float((tmp["market_value_factor"] * w).sum() / denom) if denom > 0 else float(tmp["market_value_factor"].mean())
                else:
                    # Fallback: simple average over reference periods
                    self.avg_mvf = float(tmp["market_value_factor"].mean())


        metrics_row = {
            "npv_firm": self.npv_firm,
            "npv_equity": self.npv_equity,
            "irr": self.irr,
            "lcoe": self.lcoe,
            "avg_mvf": self.avg_mvf,
            "total_energy_mwh": self.total_energy_mwh,   # <-- NEW
        }
        row_df = pd.DataFrame([metrics_row])
        if not isinstance(getattr(self, "valuemetrics", None), pd.DataFrame) or self.valuemetrics.empty:
            self.valuemetrics = row_df
        else:
            self.valuemetrics = pd.concat([self.valuemetrics, row_df], ignore_index=True)

    # ------------------------------ IRR helpers ------------------------------
    def _compute_irr(self, cashflows):
        if cashflows is None or len(cashflows) == 0:
            return None
        if np.all(cashflows >= 0) or np.all(cashflows <= 0):
            return None
        import numpy_financial as npf
        irr_periodic = npf.irr(cashflows)
        return self._annualize_rate(irr_periodic)

    def _annualize_rate(self, r_periodic):
        if r_periodic is None or not np.isfinite(r_periodic):
            return None
        periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}.get(self.cf_aggregation_period, 12)
        try:
            return (1.0 + r_periodic) ** periods_per_year - 1.0
        except Exception:
            return None

    # ------------------------------ Plotting ------------------------------

    def plot_cash_flows(
        self,
        use_discounted: bool = False,
        include_net_line: bool = True,
        figsize=(18, 6),   # wider figure
        bar_width: float = 0.2,
        title: str | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        interactive: bool = True,
    ):
        # pick source
        df = self.cashflow_discounted_records if use_discounted else self.cashflow_records
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Cash-flow table is empty. Run project_valuation() first.")

        required = ["capex", "DS", "opex", "revenue", "Equity_CF", "period_end"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Expected '{c}' in cash-flow table. Missing column '{c}'.")

        base = df.copy()
        base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
        if base["period_end"].isna().any():
            raise ValueError("Found invalid 'period_end' timestamps in cash-flow table.")

        # optional range filter
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

        # compute FCF (free cash flow)
        if "FCFF" in data.columns:
            data["FCF"] = data["FCFF"]
        else:
            data["FCF"] = data["revenue"] + data["opex"] + data["capex"]

        series_order = ["capex", "DS", "opex", "lte", "revenue"] if "lte" in data.columns else ["capex", "DS", "opex", "revenue"]


        default_title = "Discounted Cash Flows by Period" if use_discounted else "Cash Flows by Period"
        plot_title = title or default_title

        # --------------------------- Plotly Figure ---------------------------
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Bar series
        for i, s in enumerate(series_order):
            fig.add_trace(
                go.Bar(
                    x=data["period_end"],
                    y=data[s].astype(float),
                    name=s,
                    offsetgroup=str(i),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>"
                        + f"{s}: "
                        + "%{y:,.0f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        # Equity line
        fig.add_trace(
            go.Scatter(
                x=data["period_end"],
                y=data["Equity_CF"].astype(float),
                name="Equity_CF",
                mode="lines+markers",
                line=dict(width=2),
                hovertemplate="Equity_CF: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Free cash flow line
        fig.add_trace(
            go.Scatter(
                x=data["period_end"],
                y=data["FCF"].astype(float),
                name="Free Cash Flow",
                mode="lines",
                line=dict(width=2, dash="dot"),
                hovertemplate="FCF: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Layout
        fig.update_layout(
            template="plotly_white",
            barmode="group",
            title=dict(text=plot_title, x=0.01),
            hovermode="x unified",
            width=1400,                   # <<— wider figure
            height=int(figsize[1] * 80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=50, r=50, t=80, b=50),
        )

        fig.update_xaxes(
            title_text="Period end",
            tickformat="%Y-%m",
            showgrid=True,
        )

        fig.update_yaxes(
            title_text="Cash flow",
            zeroline=True,
            zerolinewidth=1,
        )

        if interactive:
            fig.show()

        return fig

    

    def overview_dashboard(self, scenario_name="Base"):
        cf = self.cashflow_discounted_records.copy()
        cf = cf.sort_values("period_end")

        # ----- KPIs
        kpi = {
            "npv_equity": self.npv_equity,
            "npv_firm": self.npv_firm,
            "irr": self.irr,
            "lcoe": self.lcoe,
        }

        # ----- cumulative discounted equity & project (firm) value
        cf["cum_disc_eq"] = cf["discounted_equity_cf"].cumsum()

        if "discounted_fcff" in cf.columns:
            cf["cum_disc_firm"] = cf["discounted_fcff"].cumsum()
        else:
            cf["cum_disc_firm"] = np.nan   # fail-safe

        # ----- Subplots layout
        fig = make_subplots(
            rows=2, cols=4,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"colspan": 4, "type": "xy"}, None, None, None],
            ],
            vertical_spacing=0.12, horizontal_spacing=0.05,
        )

        # ----- KPI cards
        fig.add_trace(go.Indicator(
            mode="number",
            value=float(kpi["npv_equity"] or 0),
            title={"text": "NPV (Equity)"},
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=float(kpi["npv_firm"] or 0),
            title={"text": "NPV (Firm)"},
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number",
            value=float(kpi["irr"] or 0),
            number={"suffix": "%", "valueformat": ".2%"},
            title={"text": "IRR (Equity)"},
        ), row=1, col=3)

        fig.add_trace(go.Indicator(
            mode="number",
            value=float(kpi["lcoe"] or 0),
            title={"text": "LCOE"},
        ), row=1, col=4)

        # ----- PLOT 1: cumulative discounted equity value
        fig.add_trace(go.Scatter(
            x=cf["period_end"],
            y=cf["cum_disc_eq"],
            mode="lines+markers",
            name="Cumulative PV Equity CF",
            line=dict(width=3),
        ), row=2, col=1)

        # ----- PLOT 2: cumulative discounted FCFF (project value)
        fig.add_trace(go.Scatter(
            x=cf["period_end"],
            y=cf["cum_disc_firm"],
            mode="lines",
            name="Cumulative PV Project Value (FCFF)",
            line=dict(width=2, dash="dot"),
        ), row=2, col=1)
        
        # ----- Breakeven annotations (equity + firm)
        def _first_recovery_index(vals: np.ndarray):
            """First index where cum >= 0 *after* having been negative."""
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0 or not np.isfinite(vals).any():
                return None

            neg_mask = vals < 0
            had_negative_before = np.cumsum(neg_mask) > 0

            crossing_mask = (vals >= 0) & had_negative_before
            idxs = np.where(crossing_mask)[0]
            return int(idxs[0]) if idxs.size > 0 else None

        # equity breakeven (cumulative discounted equity CF)
        idx_eq = _first_recovery_index(cf["cum_disc_eq"].values)

        # firm / project breakeven (cumulative discounted FCFF), if available
        idx_firm = None
        if "cum_disc_firm" in cf.columns and np.isfinite(cf["cum_disc_firm"]).any():
            idx_firm = _first_recovery_index(cf["cum_disc_firm"].values)

        # draw equity breakeven marker
        if idx_eq is not None:
            x_eq = cf["period_end"].iloc[idx_eq]
            y_eq = cf["cum_disc_eq"].iloc[idx_eq]

            fig.add_vline(
                x=x_eq,
                line_dash="dash",
                row=2, col=1,
                exclude_empty_subplots=False,
            )
            fig.add_annotation(
                x=x_eq,
                y=y_eq,
                text="Equity Breakeven",
                showarrow=True,
                row=2, col=1,
            )

        # draw firm/project breakeven marker (different dash style)
        if idx_firm is not None:
            x_firm = cf["period_end"].iloc[idx_firm]
            y_firm = cf["cum_disc_firm"].iloc[idx_firm]

            # if both breakevens fall on the same date, nudge the annotation a bit
            same_x = (idx_eq is not None) and (x_firm == cf["period_end"].iloc[idx_eq])

            fig.add_vline(
                x=x_firm,
                line_dash="dot",         # distinguish from equity
                row=2, col=1,
                exclude_empty_subplots=False,
            )
            fig.add_annotation(
                x=x_firm,
                y=y_firm * (1.02 if same_x else 1.0),  # small vertical offset if overlapping
                text="Project Breakeven",
                showarrow=True,
                row=2, col=1,
            )


        # ----- Layout
        fig.update_layout(
            title=f"Project Overview — {scenario_name}",
            hovermode="x unified",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        fig.update_xaxes(title="Period end", row=2, col=1)
        fig.update_yaxes(title="PV (currency)", row=2, col=1)

        fig.show()
        return fig


    def pv_bridge(self, use_equity: bool = True):

        # use raw cashflows (not yet discounted)
        df = self.cashflow_records.copy()

        # bring in discount factors
        disc = self.cashflow_discounted_records[["period_end", "df_equity", "df_wacc"]]
        df = df.merge(disc, on="period_end", how="left")

        # select correct discount factor
        if use_equity:
            df["df"] = df["df_equity"]
        else:
            df["df"] = df["df_wacc"]

        if use_equity:
            # Prefer true equity capex if available; otherwise fall back to full capex
            if "equity_contribution" in df.columns:
                df["equity_capex"] = df["equity_contribution"]
            else:
                df["equity_capex"] = df["capex"]

            pv = {
                "Revenue":       (df["revenue"] * df["df"]).sum(),
                "Opex":          (df["opex"] * df["df"]).sum(),
                "LTE":           (df["lte"] * df["df"]).sum() if "lte" in df.columns else 0.0,
                "Tax":           (df["Tax"] * df["df"]).sum(),
                "Debt Service":  (df["DS"] * df["df"]).sum(),
                "Equity Capex":  (df["equity_capex"] * df["df"]).sum(),
            }

            # NPV to equity = PV of Equity_CF using equity discount factor
            end_total = (df["Equity_CF"] * df["df"]).sum()
            label = "NPV Equity"

        else:
            pv = {
                "Revenue":  (df["revenue"] * df["df"]).sum(),
                "Opex":     (df["opex"] * df["df"]).sum(),
                "LTE":      (df["lte"] * df["df"]).sum() if "lte" in df.columns else 0.0,
                "Tax":      (df["Tax"] * df["df"]).sum(),
                "Capex":    (df["capex"] * df["df"]).sum(),
            }

            # NPV to firm = PV of FCFF using WACC discount factor
            end_total = (df["FCFF"] * df["df"]).sum()
            label = "NPV Firm"

        # plot waterfall
        x = list(pv.keys()) + [label]
        y = list(pv.values()) + [end_total]
        measures = ["relative"] * (len(x) - 1) + ["total"]

        fig = go.Figure(go.Waterfall(x=x, y=y, measure=measures))
        fig.update_layout(title=f"NPV Bridge — {label}")
        fig.show()

        return fig

    

    def dscr_heatmap(self):
        df = self.cashflow_records.copy()
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["DSCR"] = pd.to_numeric(df["DSCR"], errors="coerce")
        df = df.dropna(subset=["period_end", "DSCR"])

        df["year"] = df["period_end"].dt.year.astype(str)
        df["month"] = df["period_end"].dt.month  # 1..12 (NOT locale-dependent)

        pivot = df.pivot_table(index="year", columns="month", values="DSCR", aggfunc="mean")
        pivot = pivot.reindex(columns=range(1, 13))  # ensure Jan..Dec present

        # Convert to numeric array explicitly (protect against object dtype)
        z = pivot.to_numpy(dtype=float)

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=month_labels,
                y=pivot.index.tolist(),
                colorbar_title="DSCR",
            )
        )
        fig.update_layout(title="DSCR Calendar Heatmap")
        fig.show()
        return fig


def load_valuationInput(config):
    """
    Loads wind farm input parameters from the configuration file.
    """
    valuationInput = {}
    if hasattr(config, 'Valuation_inputFiles'):
        for identifier, file_name in config.Valuation_inputFiles.items():
            valuationInput[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            valuationInput[identifier] = process_duration_fields(valuationInput[identifier])
    return valuationInput
