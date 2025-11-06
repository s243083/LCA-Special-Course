import numpy as np
import pandas as pd
import os
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
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine, WindTurbines
from data.turbine.iea_22s import IEA22s
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.superposition_models import WeightedSum, SqrMaxSum, LinearSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.site._site import UniformSite
from py_wake.deficit_models.utils import ct2a_mom1d
from data.turbine.iea_22s import IEA22s
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from py_wake.rotor_avg_models import RotorCenter, GridRotorAvg, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg, PolarGridRotorAvg, PolarRotorAvg, polar_gauss_quadrature, GaussianOverlapAvgModel
from py_wake.flow_map import HorizontalGrid


def initialize_HKN_pywake_farm(farm_file):
    """
    Initializes the wind farm layout, turbine model, and PyWake flow model
    for the HKN subset case.

    Returns
    -------
    dict
        {
            "layout_df": pd.DataFrame with turbine layout and IDs,
            "flow_model": PyWake WindFarmModel object,
            "flow_type": "pywake",
            "turbine": WindTurbines object,
            "rotor_model": rotor averaging model config (dict),
            "setup_summary": dictionary with flow setup metadata for settings.json
        }
    """
    # === Load layout ===
    farm_folder = 'data'
    # farm_file = 'layout/HKN_layout_subset_with_scaled.csv'
    layout_df = pd.read_csv(os.path.join(farm_folder, farm_file))
    layout_df["id"] = np.arange(len(layout_df))  # Ensure unique turbine IDs
    x = layout_df["x_scaled"].values
    y = layout_df["y_scaled"].values

    # === Define turbine model with two modes ===
    # Mode 0: full shutdown (0 power, 0 Ct)
    # Mode 1: default IEA22s curve
    wt_base = IEA22s()
    wt = IEA22s()
    wt.powerCtFunction = PowerCtFunctionList(
        key="operating",  # used later as mode switch per turbine
        powerCtFunction_lst=[
            PowerCtTabular(
                ws=[0, 100],
                power=[0, 0],
                ct=[0, 0],
                power_unit="w"
            ),
            wt_base.powerCtFunction
        ],
        default_value=1
    )

    # === Rotor averaging model ===
    rotor_model = CGIRotorAvg(n=21)  # Could be parametrized later

    # === Wake deficit model ===
    deficit_model = ZongGaussianDeficit(
        a=[0.38, 4e-3],
        deltawD=1.0 / np.sqrt(2),
        eps_coeff=0.35,
        lam=7.5,
        B=3,
        rotorAvgModel=rotor_model,
        groundModel=None,
        use_effective_ws=True,
        use_effective_ti=True,
    )

    # === Turbulence model ===
    turbulence_model = CrespoHernandez(
        ct2a=ct2a_mom1d,
        c=[0.73, 0.83, 0.03, -0.32],
        addedTurbulenceSuperpositionModel=SqrMaxSum()
    )

    # === Site model ===
    site = UniformSite(shear=PowerShear(h_ref=wt.hub_height(), alpha=0.2))

    # === Assemble PyWake model ===
    wfm = PropagateDownwindNoSelfInduction(
        site=site,
        windTurbines=wt,
        wake_deficitModel=deficit_model,
        superpositionModel=LinearSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=turbulence_model
    )

    return {
        "layout_df": layout_df,
        "flow_model": wfm,
        "x": x,
        "y": y,
        "flow_type": "pywake",
        "turbine": wt,
        "rotor_model": {"type": "CGI", "n_points": 21},
        "setup_summary": {
            "farm_definition_file": farm_file,
            "farm_name": "HKN_subset",
            "flow_model": "PyWake.ZongGaussian + CrespoHernandez",
            "turbine_model": "IEA22",
            "rotor_avg": "CGI(21)",
            "site_model": "UniformSite(alpha=0.2)"
        }
    }

def initialize_single_turbine_farm_IEA22():
    """
    Initializes the wind farm layout, turbine model, and PyWake flow model
    for the HKN subset case.

    Returns
    -------
    dict
        {
            "layout_df": pd.DataFrame with turbine layout and IDs,
            "flow_model": PyWake WindFarmModel object,
            "flow_type": "pywake",
            "turbine": WindTurbines object,
            "rotor_model": rotor averaging model config (dict),
            "setup_summary": dictionary with flow setup metadata for settings.json
        }
    """

    x, y = 0.0, 0.0
    layout_df= pd.DataFrame({"id": [0], "x": [x], "y": [y]})
    x= np.array([x])
    y= np.array([y])

    # === Define turbine model with two modes ===
    # Mode 0: full shutdown (0 power, 0 Ct)
    # Mode 1: default IEA22s curve
    wt_base = IEA22s()
    wt = IEA22s()
    wt.powerCtFunction = PowerCtFunctionList(
        key="operating",  # used later as mode switch per turbine
        powerCtFunction_lst=[
            PowerCtTabular(
                ws=[0, 100],
                power=[0, 0],
                ct=[0, 0],
                power_unit="w"
            ),
            wt_base.powerCtFunction
        ],
        default_value=1
    )

    # === Rotor averaging model ===
    rotor_model = CGIRotorAvg(n=21)  # Could be parametrized later

    # === Wake deficit model ===
    deficit_model = ZongGaussianDeficit(
        a=[0.38, 4e-3],
        deltawD=1.0 / np.sqrt(2),
        eps_coeff=0.35,
        lam=7.5,
        B=3,
        rotorAvgModel=rotor_model,
        groundModel=None,
        use_effective_ws=True,
        use_effective_ti=True,
    )

    # === Turbulence model ===
    turbulence_model = CrespoHernandez(
        ct2a=ct2a_mom1d,
        c=[0.73, 0.83, 0.03, -0.32],
        addedTurbulenceSuperpositionModel=SqrMaxSum()
    )

    # === Site model ===
    site = UniformSite(shear=PowerShear(h_ref=wt.hub_height(), alpha=0.2))

    # === Assemble PyWake model ===
    wfm = PropagateDownwindNoSelfInduction(
        site=site,
        windTurbines=wt,
        wake_deficitModel=deficit_model,
        superpositionModel=WeightedSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=turbulence_model
    )

    return {
        "layout_df": layout_df,
        "flow_model": wfm,
        "x": x,
        "y": y,
        "flow_type": "pywake",
        "turbine": wt,
        "rotor_model": {"type": "CGI", "n_points": 21},
        "setup_summary": {
            "farm_definition_file": None,
            "farm_name": "HKN_subset",
            "flow_model": "PyWake.ZongGaussian + CrespoHernandez",
            "turbine_model": "IEA22",
            "rotor_avg": "CGI(21)",
            "site_model": "UniformSite(alpha=0.2)"
        }
    }


def compute_effective_inflow(
    wsp,
    TI,
    wdir,
    x,
    y,
    yaw,
    tilt,
    operating,
    flow_model,
    flow_type="pywake",
    use_sector_average=False,
    n_radius=10,
    n_azimuth=73,
    dtype=np.float64
):
    """
    Computes turbine-level inflow and power using PyWake or floris.

    Parameters
        wsp : float or array-like, shape (n_times,)
            Free-stream wind speed (m/s).
        TI : float or array-like, shape (n_times,)
            Turbulence intensity.
        wdir : float or array-like, shape (n_times,)
            Wind direction (degrees).
        x, y : array-like, shape (n_turbs,)
            Turbine layout coordinates (meters).
        yaw, tilt : array-like, shape (n_turbs, n_times)
            Control input arrays: yaw and tilt in degrees.
        operating : array-like, shape (n_turbs, n_times)
            Binary array: 1 for active turbine, 0 for shut down.
        flow_model : PyWake PropagateDownwind object
            Preconfigured PyWake flow model.
        flow_type : str, default="pywake"
            Only "pywake" currently supported.
        use_sector_average : bool, default=False
            If True, compute 4-sector WS/TI per turbine.
        n_radius : int, default=10
            Number of radial divisions for sector averaging.
        n_azimuth : int, default=73
            Number of azimuthal divisions for sector averaging.
        dtype : np.dtype
            Desired output precision (np.float32 or np.float64).

    Returns
    -------
        sim_res : PyWake SimulationResult
            Full simulation result object from PyWake.
        sa : xarray.DataArray or None
            Sector-averaged inflow quantities if enabled; otherwise None.
        inflow_df : pd.DataFrame
            Per-timestep inflow DataFrame:
            - If rotor-averaged: columns `wsp_idX`, `TI_idX`
            - If sector-averaged: columns `wspSAk_idX`, `tiSAk_idX` for k in 1..4
        power_df : pd.DataFrame
            Per-timestep power output in MW with columns `Power_idX`
    """
    if flow_type != "pywake":
        raise NotImplementedError("Only PyWake is currently supported.")

    # === Run PyWake model ===
    sim_res = flow_model(
        x=x,
        y=y,
        ws=np.array(wsp),
        wd=np.array(wdir),
        TI=np.array(TI),
        yaw=yaw,
        tilt=tilt,
        operating=operating,
        time=True
    )

    # === Power output ===
    power_MW = sim_res.Power.values.astype(dtype) / 1e6  # shape: (n_turbs, n_times)
    power_df = pd.DataFrame(power_MW.T, columns=[f"Power_id{i}" for i in range(power_MW.shape[0])])

    # === Rotor-averaged inflow ===
    if not use_sector_average:
        ws_eff = sim_res["WS_eff"].values.astype(dtype)  # (n_turbs, n_times)
        ti_eff = sim_res["TI_eff"].values.astype(dtype)

        inflow_rows = []
        for t in range(ws_eff.shape[1]):
            row = {f"wsp_id{i}": ws_eff[i, t] for i in range(ws_eff.shape[0])}
            row.update({f"TI_id{i}": ti_eff[i, t] for i in range(ti_eff.shape[0])})
            inflow_rows.append(row)

        inflow_df = pd.DataFrame(inflow_rows)
        return sim_res, None, inflow_df, power_df

    # === Sector-averaged inflow ===
    sa = compute_sector_average(
        sim_res,
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        look="downwind",
        use_single_precision=True
    )

    # shape: (wt, time, quantity, direction)
    ws = sa.sel(quantity="WS_eff").values.astype(dtype)  # (n_turbs, n_times, 4)
    ti = sa.sel(quantity="TI_eff").values.astype(dtype)

    n_turbs, n_times, n_sectors = ws.shape

    inflow_rows = []
    for t in range(n_times):
        row = {}
        for i in range(n_turbs):
            for s in range(n_sectors):
                row[f"wspSA{s+1}_id{i}"] = ws[i, t, s]
                row[f"tiSA{s+1}_id{i}"] = ti[i, t, s]
        inflow_rows.append(row)

    inflow_df = pd.DataFrame(inflow_rows)
    return sim_res, sa, inflow_df, power_df

