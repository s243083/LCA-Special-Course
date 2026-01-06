import logging
from core.CAPEX import CAPEX
from core.FINEX import FINEX
from core.OPEX import OPEX
from core.MetEnvironment import MetEnvironment
from core.WindFarm import WindFarm
from core.Valuation import Valuation
from core.MarketEnvironment import MarketEnv
from core.Revenue_Model import Revenue
#from core.ResponseFramework_dev.Response_Framework import ResponseFramework
from core.ResultsCollector import ResultsCollector
from core.LTE import LifetimeExtension
from core.Curtailment import Curtailment
from core.SimulationConfig import SimulationConfig
from core.utils import check_time_series_alignment
import pandas as pd


class ValueWindEnv():
    def __init__(self, config, simulation_config: SimulationConfig, logger: logging.Logger | None = None):
        self.config = config
        self.simulation_config = simulation_config
        self.logger = logger or logging.getLogger("winpact.env")

        self.metEnv = MetEnvironment(self)
        self.windFarm = WindFarm(self)
        self.capex = CAPEX(self)
        self.finex = FINEX(self, self.capex)
        self.opex = OPEX(self)
        self.MarketEnv = MarketEnv(self)
        self.RevenueModel = Revenue(self)
        self.valuation = Valuation(self)
#        self.response_framework = ResponseFramework(self)
        self.lifetimeExtension = LifetimeExtension(self)
        self.curtailment = Curtailment(self)
        self.results_collector = ResultsCollector(self)

    def run_simulation(self, until=None):
        cfg = self.simulation_config  # shorthand

        # Market Environment
        if cfg.run_marketenv:
            self.MarketEnv.create_electricityprice()

        # Met Environment
        if cfg.run_metenv:
            self.metEnv.create_met_environment()

        if cfg.run_marketenv and cfg.run_metenv:
            check_time_series_alignment(self)

        # CAPEX
        if cfg.run_capex:
            self.capex.start()
            if cfg.capex_dashboard:
                self.capex.plot_capex_dashboard(turbine_id=1)

        # Wind Farm
        if cfg.run_windfarm:
            self.windFarm.start()

        # Curtailment
        if cfg.run_curtailment:
            self.curtailment.apply()

        # -------------------------
        # LTE (optional) BEFORE OPEX
        # -------------------------
        if cfg.run_lifetime_extension:
            # LTE.apply() mutates:
            # - config.WF_OperationsEnd_h (extended horizon)
            # - windFarm.power_records (extended + AEP haircut in extension)
            # and sets attributes used later by OPEX/Valuation
            self.lifetimeExtension.apply()

        # -------------------------
        # OPEX (windowed + internal accumulation)
        # -------------------------
        if cfg.run_opex:
            # Determine LTE status from LTE object attributes (no return values)
            lte = self.lifetimeExtension
            lte_enabled = getattr(lte.cfg, "enable_lte", False)
            base_end_h = getattr(lte, "base_end_h", None)
            ext_end_h = getattr(lte, "ext_end_h", None)

            do_two_pass = bool(
                cfg.run_lifetime_extension
                and lte_enabled
                and isinstance(base_end_h, int)
                and isinstance(ext_end_h, int)
                and (ext_end_h > base_end_h)
            )

            if not do_two_pass:
                # Single pass over full horizon (baseline only or LTE disabled)
                self.opex.calc_OPEX(window=None, overrides=None, append=False, window_label="full")
            else:
                project_start = pd.to_datetime(self.config.Project_StartDate)

                base_start_h = int(self.config.WF_OperationsStart_h)
                base_end_h = int(base_end_h)
                ext_end_h = int(ext_end_h)

                ops_start_ts = project_start + pd.to_timedelta(base_start_h, unit="h")
                base_end_ts  = project_start + pd.to_timedelta(base_end_h, unit="h")
                ext_end_ts   = project_start + pd.to_timedelta(ext_end_h, unit="h")

                # 1) Baseline window (reset accumulator)
                self.opex.calc_OPEX(
                    window=(ops_start_ts, base_end_ts),
                    overrides=None,
                    append=False,
                    window_label="baseline",
                )

                # 2) Extension window (append, with LTE mean-shift overrides from LTE object)
                self.opex.calc_OPEX(
                    window=(base_end_ts, ext_end_ts),
                    overrides=getattr(lte, "opex_mean_shift_overrides", {}) or {},
                    append=True,
                    window_label="lte_extension",
                )

            if cfg.opex_dashboard:
                self.opex.plot_opex_dashboard()

        # Revenues
        if cfg.run_revenue:
            self.RevenueModel.calc_revenues()

        # Valuation
        if cfg.run_valuation:
            # Valuation pulls LTE cost records from env.lifetimeExtension.cost_records
            self.valuation.project_valuation()
            if cfg.valuation_dashboard:
                self.valuation.plot_valuation_results()

        # Results Collector
        if cfg.collect_results:
            self.results_collector.collect_df(
                attr_map={
                    "valuation_metrics": "valuation.valuemetrics",
                    "capex": "capex.cost_records",

                    "opex_records": "opex.OPEX_records",
                    "opex_windows": "opex.opex_windows_df",
                    "opex_breakdown": "opex.opex_breakdown_df",
                    "opex_mode_cost_breakdown": "opex.opex_mode_cost_breakdown_df",
                    "opex_component_cost_breakdown": "opex.opex_component_cost_breakdown_df",
                    # stitched frames already exist in extras container if you prefer:
                    "opex_availability_profile": "opex.OPEX_records_extras.availability_profile",  # only works if it's an attribute; see note below
                    "opex_activity_log": "opex.OPEX_records_extras.activity_log",


                }
            )

