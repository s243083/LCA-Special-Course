import simpy
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

class ValueWindEnv(simpy.Environment):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.metEnv= MetEnvironment(self)
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
        # Start the CAPEX process in the environment
        self.capex.start()
        self.capex.plot_cost_pies(turbine_id=1)

        
        # Start the wind farm process in the environment
        self.windFarm.start()

        # Calculate OPEX
        self.opex.calc_OPEX()
        


        # Apply Lifetime Extension if enabled
        # self.lifetimeExtension.apply()



        # calculate Revenues
        self.RevenueModel.calc_revenues()


        # calculate Valuation
        self.valuation.project_valuation()
        

        # Call Reuslts Collector
        self.results_collector.collect_df()


        