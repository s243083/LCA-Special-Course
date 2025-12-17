import pandas as pd
import numpy as np



class CashFlowEngine:
    def __init__(self, env,calendar: pd.DatetimeIndex | None = None):
        self.calendar = calendar  # optional canonical index
        self.env = env
        self.D0 = self.env.finex.D0
        self.E0 = self.env.finex.E0


    def run_waterfall(self, *, capex_df, opex_df, revenue_df, finex_df, tax_rate: float, lte_df = None) -> pd.DataFrame:
        """
        Orchestrates the waterfall for one pass.
        Inputs (required columns):
          capex_df:   ["timestamp","cost"] (cost<0)
          opex_df:    ["timestamp","OM_payment"] (outflow<0)
          revenue_df: ["timestamp","total_revenue"] (inflow>0)
          finex_df:   ["timestamp","interest","principal","payment","depreciation", ...]
        Returns a wide DataFrame indexed by timestamp with columns:
          revenue, opex, capex, depreciation, interest, principal, payment, outstanding,
          EBITDA, EBIT, EBT, Tax, LCF, CFADS, DS, Equity_CF, DSCR
        """
        # Build cash flow base table

        df = self._align(capex_df, opex_df, revenue_df, finex_df, lte_df=lte_df)
        if "lte" not in df.columns:
            df["lte"] = 0.0


        # operating lines
        df["EBITDA"] = df["revenue"] + df["opex"]
        df["EBIT"]   = df["EBITDA"] + df["depreciation"]         # dep is negative → subtraction
        df["EBT"]    = df["EBIT"] + df["interest"]               # interest is negative → subtraction

        # tax with loss-carry-forward ledger
        df["LCF"] = 0.0
        lcf = 0.0
        taxes = []
        for _, r in df.iterrows():
            taxable = r["EBT"] + min(0.0, lcf)  # apply positive LCF (lcf is <=0)
            tax = tax_rate * taxable if taxable > 0 else 0.0
            lcf = (taxable - tax) if taxable < 0 else 0.0  # update carry-forward
            taxes.append(-tax)  # outflow
        df["Tax"] = taxes
        df["LCF"] = lcf  # optionally keep last or store running via second pass

        # CFADS & debt service
        # CFADS & debt service
        df["CFADS"] = df["EBITDA"] + df["Tax"]          # interest excluded by design
        df["DS"]    = df["interest"] + df["principal"]  # both negative

        # --- Equity cash flows (true equity view, using D0/E0 from FINEX) ---
        total_capital = float(self.D0) + float(self.E0)

        if total_capital > 0.0:
            equity_share = float(self.E0) / total_capital
        else:
            # Fallback: if something is off, assume 100% equity
            equity_share = 1.0

        # Clamp to [0, 1] for safety
        equity_share = min(1.0, max(0.0, equity_share))

        # Equity contributions follow the CAPEX profile, scaled by equity share
        # capex < 0 → equity_contribution < 0 in construction periods
        df["equity_contribution"] = df["capex"] * equity_share

        # Equity cash flow to *equity holders*:
        #   CFADS        : operating cash after tax (pre-interest)
        #   DS           : debt service (interest + principal, negative)
        #   equity_contr.: equity injections (negative in build years)
        df["Equity_CF"] = df["CFADS"] + df["DS"] + df["equity_contribution"] + df["lte"]

        # coverage
        ds_pos = df["DS"].abs()  # DS magnitude
        df["DSCR"] = np.where(ds_pos > 0, df["CFADS"] / ds_pos, np.nan)


        return df

    # --- helper ---
    def _align(self, capex_df, opex_df, revenue_df, finex_df, lte_df = None) -> pd.DataFrame:
        # --- helpers
        def _normalize_to_calendar(ts: pd.Series) -> pd.Series:
            ts = pd.to_datetime(ts, errors="coerce")

            if self.calendar is None:
                return ts

            fs = getattr(self.calendar, "freqstr", None)

            # 1) If calendar is month-start, normalize explicitly to month-start
            if fs == "MS":
                return ts.dt.to_period("M").dt.to_timestamp()

            # 2) Otherwise try floor for fixed freqs
            freq = getattr(self.calendar, "freq", None) or fs
            try:
                return ts.dt.floor(freq)
            except Exception:
                return ts



        def _prep(df, rename_map, cols, ensure_negative_for=None):
            ensure_negative_for = ensure_negative_for or []

            out = df.copy()
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
            out = out.dropna(subset=["timestamp"])

            # snap timestamps to calendar grid if provided
            out["timestamp"] = _normalize_to_calendar(out["timestamp"])

            if rename_map:
                out = out.rename(columns=rename_map)

            for c in cols:
                if c not in out.columns:
                    out[c] = 0.0
                out[c] = pd.to_numeric(out[c], errors="coerce")

            for c in ensure_negative_for:
                if c in out.columns:
                    out[c] = -out[c].abs()

            # keep needed cols and drop rows where all values are NaN
            out = out[["timestamp"] + cols]
            out = out[out[cols].notna().any(axis=1)]

            # sum duplicates per (normalized) timestamp
            out = out.groupby("timestamp", as_index=False)[cols].sum()
            return out

        capex = _prep(capex_df, {"cost": "capex"}, ["capex"], ensure_negative_for=["capex"])
        opex  = _prep(opex_df, {"OM_payment": "opex"}, ["opex"])
        rev   = _prep(revenue_df, {"total_revenue": "revenue"}, ["revenue"])
        fin   = _prep(finex_df, None, ["payment","interest","principal","outstanding","depreciation"])
        lte = _prep(lte_df, {"LTE_payment": "lte"}, ["lte"]) if lte_df is not None else None

        df = (
            rev.merge(opex, on="timestamp", how="outer")
            .merge(capex, on="timestamp", how="outer")
            .merge(fin, on="timestamp", how="outer")
        )

        if lte is not None:
            df = df.merge(lte, on="timestamp", how="outer")

        df = df.sort_values("timestamp").fillna(0.0)


        if self.calendar is not None:
            df = (df.set_index("timestamp")
                    .reindex(self.calendar)
                    .fillna(0.0)
                    .reset_index()
                    .rename(columns={"index": "timestamp"}))

        return df
