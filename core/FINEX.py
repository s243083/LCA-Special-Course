"""
FINEX.py — Financing & Depreciation module (LTE-style overrides, no extra helpers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict

import pandas as pd

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import get_input_parameter, apply_overrides


# -----------------------------
# Normalized config (resolved)
# -----------------------------
@dataclass(frozen=True)
class FinexConfig:
    """Resolved financing & depreciation configuration.

    A flat, immutable representation of the FINEX section of the input
    YAML after duration-field processing and override application. All
    rates are annual fractions in the range [0, 1] unless noted.

    Parameters
    ----------
    debt_share : float
        Share of CAPEX financed by debt (e.g. ``0.7`` for 70/30 D/E).
    interest_rate : float
        Annual interest rate on the debt tranche.
    n_yearly_payments : int
        Debt-service payments per year (e.g. 2 for semi-annual).
    n_total_payments : int
        Total number of scheduled debt-service payments over the loan
        tenor.
    start_debt_service_h : int
        Hour offset from project start at which the first debt-service
        payment is due.
    E : float
        Cost of equity (annual fraction) when ``flag_CAPM`` is False.
    flag_CAPM : bool
        If True, the cost of equity is recomputed via CAPM
        (``rf + beta * (rm - rf)``) and ``E`` is ignored.
    rf, rm, beta : float
        Risk-free rate, market return, and equity beta used by CAPM.
    tax_rate : float
        Corporate tax rate, used in the after-tax WACC formula.
    depreciation_method : {"SL"}
        Depreciation scheme. Currently only straight-line is supported.
    depreciation_period : float
        Depreciation horizon in years.
    depreciation_salvage_value : float
        Salvage value as a fraction of CAPEX.
    flag_fixed_wacc : bool
        If True, ``WACC_annual_input`` is used directly and the D/E +
        cost-of-equity inputs are ignored.
    WACC_annual_input : float
        Externally supplied WACC, honoured only when
        ``flag_fixed_wacc`` is True.
    """

    # Debt
    debt_share: float                 # fraction (0..1)
    interest_rate: float              # annual fraction (0..1)
    n_yearly_payments: int
    n_total_payments: int
    start_debt_service_h: int

    # Equity
    E: float                          # annual fraction (0..1)
    flag_CAPM: bool
    rf: float                         # annual fraction (0..1)
    rm: float                         # annual fraction (0..1)
    beta: float

    # Tax (used for WACC only)
    tax_rate: float                   # fraction (0..1)

    # Depreciation
    depreciation_method: str          # "SL"
    depreciation_period: float        # years
    depreciation_salvage_value: float # fraction of CAPEX (0..1)


    flag_fixed_wacc: bool
    WACC_annual_input: float


class FINEX:
    """Project financing and depreciation engine.

    FINEX produces the time-resolved schedule of debt service, interest,
    principal, outstanding balance, and depreciation that the
    cash-flow engine consumes alongside CAPEX, OPEX, and revenue. It
    also computes the WACC (or accepts a fixed one) and the cost of
    equity used by :class:`core.Valuation.Valuation`.

    Parameters
    ----------
    env : ValueWindEnv
        Owning environment. FINEX reads ``env.config`` and the
        ``FINEX_overrides`` mapping for scenario variations.
    capex : CAPEX
        CAPEX module instance, used to determine the financing base.
    cfg : FinexConfig, optional
        Pre-built immutable config. If provided, FINEX skips its YAML
        load and uses this directly — primarily a test/debug hook.

    Attributes
    ----------
    cfg : FinexConfig
        Resolved configuration after YAML load + overrides.
    D0, E0 : float
        Initial debt and equity amounts derived from the financing
        base and ``cfg.debt_share``.
    WACC_annual : float
        After-tax weighted-average cost of capital used as the discount
        rate by Valuation.
    equity_annual : float
        Cost of equity (input or CAPM).
    finex_records : pandas.DataFrame
        Per-period schedule with columns ``timestamp``, ``payment``,
        ``interest``, ``principal``, ``outstanding``, ``depreciation``,
        plus any extras added by sub-routines.

    Notes
    -----
    Overrides are applied *before* the resolved :class:`FinexConfig` is
    built, so dotted-path parameters in ``FINEX_overrides`` may target
    either the YAML-shaped fields or the FINEX instance directly.

    See Also
    --------
    FinexConfig : the resolved configuration schema.
    core.CAPEX.CAPEX : provides the financing base.
    core.Valuation.Valuation : consumes ``WACC_annual`` and ``finex_records``.
    """

    def __init__(self, env: Any, capex: Any, cfg: Optional[FinexConfig] = None):
        self.env = env
        self.config = env.config
        self.capex = capex

        # ---- Load FINEX inputs from YAML ----
        self.finex_input = load_finex_inputs(self.config)
        self.finex_input = get_input_parameter(self.finex_input, "FI")

        # ---- Apply scenario overrides ----
        apply_overrides(self, getattr(self.config, "FINEX_overrides", {}))

        # ---- Build FinexConfig from YAML (unless explicitly provided) ----
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = self._cfg_from_finex_input(self.finex_input)

        # ---- Derived / outputs ----
        self.project_start = pd.to_datetime(self.config.Project_StartDate, format="%d.%m.%Y")

        self.D0: float = 0.0
        self.E0: float = 0.0
        self.WACC_annual: float = 0.0
        self.equity_annual: float = 0.0

        self.finex_records = pd.DataFrame(
            columns=[
                "timestamp",
                "payment",
                "interest",
                "principal",
                "outstanding",
                "depreciation",
                "debt_payment",  # legacy alias
            ]
        )

    def _cfg_from_finex_input(self, finex_input: Dict[str, Any]) -> FinexConfig:
        """
        Map YAML -> FinexConfig (normalized only; no extra coercion/validation).
        """
        debt_share = get_input_parameter(finex_input, "FINEX", "Debt", "debt_share") / 100.0
        interest_rate = get_input_parameter(finex_input, "FINEX", "Debt", "interest_rate") / 100.0
        n_yearly_payments = int(get_input_parameter(finex_input, "FINEX", "Debt", "n_yearly_payments"))
        n_total_payments = int(get_input_parameter(finex_input, "FINEX", "Debt", "n_total_payments"))
        start_debt_service_h = int(get_input_parameter(finex_input, "FINEX", "Debt", "start_debt_service_h"))

        E = get_input_parameter(finex_input, "FINEX", "Equity", "E") / 100.0
        flag_CAPM = bool(get_input_parameter(finex_input, "FINEX", "Equity", "equity_costModel", "flag_CAPM"))
        rf = get_input_parameter(finex_input, "FINEX", "Equity", "equity_costModel", "rf") / 100.0
        rm = get_input_parameter(finex_input, "FINEX", "Equity", "equity_costModel", "rm") / 100.0
        beta = get_input_parameter(finex_input, "FINEX", "Equity", "equity_costModel", "beta")


        flag_fixed_wacc = bool(get_input_parameter(finex_input, "FINEX", "WACC", "flag_fixed_WACC"))
        WACC_annual_input = get_input_parameter(finex_input, "FINEX", "WACC", "WACC_annual")


        tax_rate = get_input_parameter(finex_input, "FINEX", "Tax", "tax_rate") / 100.0

        depreciation_method = get_input_parameter(finex_input, "FINEX", "Depreciation", "method")
        depreciation_period = get_input_parameter(finex_input, "FINEX", "Depreciation", "period")
        depreciation_salvage_value = get_input_parameter(finex_input, "FINEX", "Depreciation", "salvage_value") / 100.0

        return FinexConfig(
            debt_share=float(debt_share),
            interest_rate=float(interest_rate),
            n_yearly_payments=int(n_yearly_payments),
            n_total_payments=int(n_total_payments),
            start_debt_service_h=int(start_debt_service_h),
            E=float(E),
            flag_CAPM=bool(flag_CAPM),
            rf=float(rf),
            rm=float(rm),
            beta=float(beta),
            tax_rate=float(tax_rate),
            depreciation_method=str(depreciation_method),
            depreciation_period=float(depreciation_period),
            depreciation_salvage_value=float(depreciation_salvage_value),
            flag_fixed_wacc=flag_fixed_wacc,
            WACC_annual_input=WACC_annual_input,
        )

    def calc_FINEX(self) -> pd.DataFrame:
        """
        Computes WACC (for valuation), debt amortization schedule, and depreciation schedule.
        Populates self.finex_records and self.WACC_annual.
        """
        cfg = self.cfg

        total_cost = float(self.capex.total_cost)
        periods_per_year = int(cfg.n_yearly_payments)
        n_periods = int(cfg.n_total_payments)

        start_ts = pd.to_datetime(self.env.config.Project_StartDate, format="%d.%m.%Y")
        first_pay_ts = start_ts + pd.Timedelta(hours=int(cfg.start_debt_service_h))

        # capital structure
        D0 = total_cost * float(cfg.debt_share)
        E0 = total_cost - D0
        self.D0 = D0
        self.E0 = E0

        # periodic cost of debt
        i_per = (1.0 + float(cfg.interest_rate)) ** (1.0 / periods_per_year) - 1.0

        # equity annual rate
        self.equity_annual = (float(cfg.rf) + float(cfg.beta) * (float(cfg.rm) - float(cfg.rf))) if cfg.flag_CAPM else float(cfg.E)

        # WACC annual (used for discounting)
        if cfg.flag_fixed_wacc:
            self.WACC_annual = float(cfg.WACC_annual_input)
        else:
            Re_annual = float(self.equity_annual)
            Rd_annual = float(cfg.interest_rate)
            V = D0 + E0
            self.WACC_annual = (E0 / V) * Re_annual + (D0 / V) * Rd_annual * (1.0 - float(cfg.tax_rate))


        # amortization
        payment = self._annuity_payment(D0, i_per, n_periods)
        sched = []
        outstanding = float(D0)
        period_delta = pd.to_timedelta(8760 / periods_per_year, unit="h")

        for k in range(n_periods):
            ts = first_pay_ts + k * period_delta
            interest = outstanding * i_per
            principal = payment - interest
            outstanding = max(0.0, outstanding - principal)
            sched.append(
                {
                    "timestamp": ts,
                    "payment": -float(payment),
                    "interest": -float(interest),
                    "principal": -float(principal),
                    "outstanding": float(outstanding),
                }
            )

        debt_df = pd.DataFrame(sched)

        # depreciation
        dep_df = self._build_depreciation_schedule(first_pay_ts, periods_per_year)

        finex = (
            debt_df.merge(dep_df, on="timestamp", how="outer")
            .sort_values("timestamp")
            .fillna(
                {
                    "payment": 0.0,
                    "interest": 0.0,
                    "principal": 0.0,
                    "outstanding": 0.0,
                    "depreciation": 0.0,
                }
            )
        )

        # legacy alias
        finex["debt_payment"] = finex["payment"]

        self.finex_records = finex
        return self.finex_records

    def _build_depreciation_schedule(self, first_pay_ts: pd.Timestamp, periods_per_year: int) -> pd.DataFrame:
        """
        Straight-line depreciation schedule.
        Returns DataFrame columns: ["timestamp", "depreciation"].
        """
        cfg = self.cfg

        capex_total = float(getattr(self.capex, "total_cost", 0.0))
        if capex_total <= 0.0:
            return pd.DataFrame(columns=["timestamp", "depreciation"])

        method = (cfg.depreciation_method or "SL").upper()
        if method != "SL":
            raise NotImplementedError(f"Depreciation method '{cfg.depreciation_method}' not supported (only 'SL').")

        life_years = float(cfg.depreciation_period)
        salvage_ratio = float(cfg.depreciation_salvage_value or 0.0)

        salvage_value = max(0.0, capex_total * salvage_ratio)
        depreciable_base = max(0.0, capex_total - salvage_value)

        n_periods = int(round(life_years * periods_per_year))
        per_period_dep = depreciable_base / n_periods

        period_delta = pd.to_timedelta(8760 / periods_per_year, unit="h")
        dep_rows = ({"timestamp": first_pay_ts + i * period_delta, "depreciation": per_period_dep} for i in range(n_periods))
        dep_df = pd.DataFrame(dep_rows, columns=["timestamp", "depreciation"])

        dep_df["timestamp"] = pd.to_datetime(dep_df["timestamp"], errors="coerce")
        dep_df["depreciation"] = dep_df["depreciation"].astype(float)
        return dep_df

    @staticmethod
    def _annuity_payment(P: float, i: float, n: int) -> float:
        return P * (i * (1 + i) ** n) / ((1 + i) ** n - 1) if n > 0 else 0.0

    def get_cost_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.finex_records)


def load_finex_inputs(config: Any) -> Dict[str, Any]:
    """
    Loads FINEX input parameters from configuration files,
    converting any duration parameters to hours.
    """
    finex_input: Dict[str, Any] = {}

    if hasattr(config, "Finex_inputFiles"):
        for identifier, file_name in config.Finex_inputFiles.items():
            finex_input[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            finex_input[identifier] = process_duration_fields(finex_input[identifier])

    return finex_input
