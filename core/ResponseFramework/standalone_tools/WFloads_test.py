import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import sys
sys.path.insert(0, os.getcwd())
# os.chdir('/groups/SUDOCO/Task23/sudoco_task2.3/')
from surrogates_interface.transformers import Transformer
import matplotlib.pyplot as plt
from surrogates_interface.surrogates import TensorFlowModel

import numpy as np
import pandas as pd
import xarray as xr
from py_wake.site import UniformSite
# from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.examples.data.hornsrev1 import V80
# from dependencies import compute_sector_average, PropagateDownwindNoSelfInduction, predict_loads_sector_average
from wind_farm_loads.py_wake import  PropagateDownwindNoSelfInduction, predict_loads_sector_average, predict_loads_rotor_average
from wind_farm_loads.tool_agnostic import compute_sector_average
from abc import ABC
from py_wake.deficit_models import ZongGaussianDeficit, NOJDeficit
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.site.shear import PowerShear
from py_wake.turbulence_models import CrespoHernandez, GCLTurbulence,STF2017TurbulenceModel
# from py_wake.deficit_models.gaussian import GaussianDeficit
from py_wake.site import UniformSite
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides INFO and WARNING messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')         # Hides tf.function retracing warnings
# from iea3_4_pywake_openfast_1 import iea3_4
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine, WindTurbines
from data.turbine.iea_22s import IEA22s
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.superposition_models import WeightedSum, SqrMaxSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.site._site import UniformSite
from py_wake.deficit_models.utils import ct2a_mom1d
from data.turbine.iea_22s import IEA22s
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from py_wake.rotor_avg_models import RotorCenter, GridRotorAvg, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg, PolarGridRotorAvg, PolarRotorAvg, polar_gauss_quadrature, GaussianOverlapAvgModel
from py_wake.flow_map import HorizontalGrid


# Updated list of available surrogate channels based on files you have
channels = [
    "SA_blade_root_ip", "SA_blade_root_oop","SA_blade_root_projected", "SA_shaft_oop", "SA_shaft_yaw",
    "SA_tbss", "SA_tbfa", "SA_tower_top_fa", "SA_tower_torsion","SA_tower_top_ss","SA_tower_base_projected"
]

channels_RA = [
    "RA_blade_root_ip", "RA_blade_root_oop","RA_blade_root_projected", "RA_shaft_oop", "RA_shaft_yaw",
    "RA_tbss", "RA_tbfa", "RA_tower_top_fa", "RA_tower_torsion","RA_tower_top_ss","RA_tower_base_projected"
]

surrogates = {}
for ch in channels:
    model_path = os.path.join("models", f"{ch}.keras")
    scaler_path = os.path.join("models", f"scaler_{ch}.h5")
    # surrogates[ch] = TensorFlowModel(model_path, scaler_path)
    surrogates[ch] = TensorFlowModel.load_h5(
    model_path=model_path, extra_data_path=scaler_path)


surrogates_rotor = {}
for ch in channels_RA:
    model_path = os.path.join("models/RA_surrogate", f"{ch}.keras")
    scaler_path = os.path.join("models/RA_surrogate", f"scaler_{ch}.h5")
    # surrogates[ch] = TensorFlowModel(model_path, scaler_path)
    surrogates_rotor[ch] = TensorFlowModel.load_h5(
    model_path=model_path, extra_data_path=scaler_path)

# Example with IEA22 turbines in the flow and IEA 3.4 for the surrogates

# %%prun
wt = IEA22s()

# First approach of creating two turbine types one shut and one the normal turbine
u = [0,3,12,25,30]
ct = [0,0,0,0, 0]
power = [0,0,0,0,0]

wt2 = WindTurbine(name='MyWT',
                    diameter=wt.diameter(),
                    hub_height=wt.hub_height(),
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))
wts = WindTurbines.from_WindTurbine_lst([wt2, wt])

# Second approach of creating a single turbine with operating modes
wt.powerCtFunction = PowerCtFunctionList(
    key="operating",
    powerCtFunction_lst=[
        PowerCtTabular(
            ws=[0, 100], power=[0, 0], power_unit="w", ct=[0, 0]
        ),  # 0=No power and ct
        IEA22s().powerCtFunction,
    ],  # 1=Normal operation
    default_value=1,
)
# Then to run it I need something like this
# # sim_res_zong_weighted = wf_model(
#     x, y, wd=270, ws=7, TI=0.06, yaw=0, tilt=0, operating=np.array([0, 0, 1, 1])
# )


# Wind farm layout
D = wt.diameter()


# models pywake
deficit_model = ZongGaussianDeficit(
    a=[0.38, 4e-3],
    deltawD=1.0 / np.sqrt(2),
    eps_coeff=0.35,
    lam=7.5,
    B=3,
    rotorAvgModel=CGIRotorAvg(21),
    groundModel=None,
    use_effective_ws=True,
    use_effective_ti=True,
)

turbulence_model = CrespoHernandez(
    ct2a=ct2a_mom1d,
    c=[0.73, 0.83, 0.03, -0.32],
    addedTurbulenceSuperpositionModel=SqrMaxSum(),
)

site = UniformSite(shear=PowerShear(h_ref=wt.hub_height(), alpha=0.2))


# Wind farm model (no self induction)
wfm = PropagateDownwindNoSelfInduction(
    site=site,
    windTurbines=wts,
    wake_deficitModel=deficit_model,
    superpositionModel=WeightedSum(),
    deflectionModel=JimenezWakeDeflection(),
    turbulenceModel=turbulence_model,
)


# site.ds
layout_subset = pd.read_csv(r"data/HKN_layout_subset_with_scaled.csv")
x_scaled = layout_subset["x_scaled"]
y_scaled = layout_subset["y_scaled"]
yaw_ang= np.full(34, 0)
power_level=np.full(34, 100)

import time
start = time.time()


sim_res = wfm(
    x=x_scaled,
    y=y_scaled,
    type=np.full(34, True),  # Only first three turbines are active
    TI=np.full(50000, 0.1),            # single turbulence intensity
    wd=np.full(50000, 210),              # single wind direction
    ws=np.full(50000, 8),              # single wind speed
    yaw=yaw_ang,        # array yaw values
    tilt=np.full(34, 0),      # array of tilt values
    time=True,
    # power_demand=power_level,     # array of power level values works only if the turbine has power levels defined
)
# import cProfile
# import pstats
# def run_sector_average():
#     sa = compute_sector_average(
#         sim_res, 
#         n_radius=10,
#         n_azimuth=73,
#         look="downwind",
#     )
#     return sa

# profiler = cProfile.Profile()
# profiler.enable()

# sa = run_sector_average()

# profiler.disable()
# #Write results to a text file
# with open("profile_sector_average.txt", "w") as f:
#     ps = pstats.Stats(profiler, stream=f)
#     ps.strip_dirs()
#     ps.sort_stats("cumtime")  # sort by cumulative time spent in function
#     ps.print_stats()
# import time
# start = time.time()
# sa = compute_sector_average(
#         sim_res, 
#         n_radius=10,
#         n_azimuth=73,
#         look="downwind")
# end = time.time()
# print(f"Elapsed time: {end - start:.2f} s")


loads_rotor = predict_loads_rotor_average(
    surrogates_rotor,
    sim_res,                      # the PyWake SimulationResult
    yaw_ang,              # yaw array (example)
    power_level,             # power demand array (example)
    ti_in_percent=True,           # same flag as sector-averaged
    # dtype=np.float32              # output dtype
)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")

# # Predict loads using sector-averaged inflow
# loads_sector = predict_loads_sector_average(
#     surrogates,   # dictionary of TensorflowSurrogate objects
#     # sim_res,      # simulation result from PyWake
#     sa,           # sector-averaged inflow (the variable `sa`)
#     yaw_ang,              # yaw array (example)
#     power_level,             # power demand array (example)
#     ti_in_percent=True,
# )