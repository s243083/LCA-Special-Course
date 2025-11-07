import py_wake
import numpy as np

class PyWakeModel:
    def __init__(self):
        # Initialize PyWake model
        # this should not be hardcoded, but read from Windfarm configuration
        from py_wake.examples.data.lillgrund import LillgrundSite
        from py_wake.examples.data.lillgrund import LillgrundSWT23

        from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

        self.windTurbines = LillgrundSWT23()
        self.site = LillgrundSite()

        self.x, self.y = self.site.initial_position.T

        self.wf_model = Bastankhah_PorteAgel_2014(self.site, self.windTurbines, k=0.0324555)


    def get_windFarm_response_global(self):
        sim_res = self.wf_model(self.x, self.y,     # wind turbine positions
                   h=None,   # wind turbine heights (defaults to the heights defined in windTurbines)
                   type=0,   # Wind turbine types
                   wd=None,  # Wind direction
                   ws=None,  # Wind speed
                  )
        return sim_res
    

    def get_windFarm_response_timestep(self, ws, wd, ti, control_output):
        #operating = np.ones((len(self.x), 1))
        sim_res = self.wf_model(self.x, self.y,     # wind turbine positions
                   wd=wd,  # Wind direction
                   ws=ws,  # Wind speed
                   time = 0, 
                   TI=ti,  # Turbulence intensity
                   #operating = operating # this can interact with the control_output
                  )
        return sim_res
    
    
    def get_turbine_inflow(self, turbine, control_output):
        return {'Vreq': 10, 'TIreq': 0.1}   
