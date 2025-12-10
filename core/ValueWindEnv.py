import logging
from core.CAPEX import CAPEX
from core.FINEX import FINEX
from core.OPEX import OPEX
from core.MetEnvironment import MetEnvironment
from core.WindFarm import WindFarm
from core.Valuation import Valuation
from core.MarketEnvironment import MarketEnv
from core.Revenue_Model import Revenue
from core.ResponseFramework_dev.Response_Framework import ResponseFramework
from core.ResultsCollector import ResultsCollector
from core.LTE import LifetimeExtension
from core.SimulationConfig import SimulationConfig
from core.utils import check_time_series_alignment


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
        self.response_framework = ResponseFramework(self)
        self.lifetimeExtension = LifetimeExtension(self)
        self.results_collector = ResultsCollector(self)



    def run_simulation(self, until=None):
        cfg = self.simulation_config  # shorthand

        # Market Environment
        if cfg.run_marketenv:
            self.MarketEnv.create_electricityprice()
        
        # Met Environment
        if cfg.run_metenv:
            self.metEnv.create_met_environment()

        # Alignment check
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

        # OPEX
        if cfg.run_opex:
            self.opex.calc_OPEX()
            if cfg.opex_dashboard:
                self.opex.plot_opex_dashboard()

        # Lifetime Extension
        if cfg.run_lifetime_extension:
            self.lifetimeExtension.apply()

        # Revenues
        if cfg.run_revenue:
            self.RevenueModel.calc_revenues()

        # Valuation
        if cfg.run_valuation:
            self.valuation.project_valuation()
            if cfg.valuation_dashboard:
                self.valuation.plot_valuation_results()

        # Results Collector
        if cfg.collect_results:
            self.results_collector.collect_df(
                attr_map={
                    "valuation_metrics": "valuation.valuemetrics",
                    "capex": "capex.cost_records",
                }
            )


        