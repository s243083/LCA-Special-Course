# Author: [Vasilis Pettas, Moritz Gräfe]

import numpy as np
import pandas as pd

from wind_farm_loads.py_wake import PropagateDownwindNoSelfInduction
from wind_farm_loads.tool_agnostic import compute_sector_average

from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.superposition_models import WeightedSum, SqrMaxSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site._site import UniformSite
from py_wake.site.shear import PowerShear
from py_wake.rotor_avg_models import CGIRotorAvg

from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from core.ResponseFramework.data.turbine.iea_22s import IEA22s

def initialize_pywake_farm(
    *,
    use_pywake_farm: bool,
    layout_file: str | None,
    turbine_model: str = "IEA22",
):
    """
    Generic PyWake farm initializer controlled by config.

    - If use_pywake_farm is True:
        - load layout from layout_file
    - If False:
        - single turbine at (0,0)
    - turbine_model: currently supports "IEA22" mapped to IEA22s()
    """
    # 1) Turbine model selection (can be extended later)
    if turbine_model == "IEA22":
        wt_base = IEA22s()
    else:
        raise NotImplementedError(f"Turbine model '{turbine_model}' not implemented yet.")

    wt = IEA22s()
    wt.powerCtFunction = PowerCtFunctionList(
        key="operating",
        powerCtFunction_lst=[
            PowerCtTabular(ws=[0, 100], power=[0, 0], ct=[0, 0], power_unit="w"),
            wt_base.powerCtFunction,
        ],
        default_value=1,
    )

    # 2) Layout
    if use_pywake_farm:
        if layout_file is None:
            raise ValueError("layout_file must be provided when use_pywake_farm=True")

        layout_df = pd.read_csv(layout_file)

        # Flexible column support: x_scaled/y_scaled or x/y
        if {"x_scaled", "y_scaled"}.issubset(layout_df.columns):
            x = layout_df["x_scaled"].to_numpy()
            y = layout_df["y_scaled"].to_numpy()
        elif {"x", "y"}.issubset(layout_df.columns):
            x = layout_df["x"].to_numpy()
            y = layout_df["y"].to_numpy()
        else:
            raise ValueError("Layout file must contain either (x, y) or (x_scaled, y_scaled) columns.")

        layout_df = layout_df.copy()
        if "id" not in layout_df.columns:
            layout_df["id"] = np.arange(len(layout_df))

    else:
        # Single turbine at (0,0)
        x = np.array([0.0])
        y = np.array([0.0])
        layout_df = pd.DataFrame({"id": [0], "x": [0.0], "y": [0.0]})

    # 3) Rotor / wake / turbulence / site: as in your current setup
    rotor_model = CGIRotorAvg(n=21)
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
    turbulence_model = CrespoHernandez(
        ct2a=ct2a_mom1d,
        c=[0.73, 0.83, 0.03, -0.32],
        addedTurbulenceSuperpositionModel=SqrMaxSum(),
    )
    site = UniformSite(shear=PowerShear(h_ref=wt.hub_height(), alpha=0.2))

    wfm = PropagateDownwindNoSelfInduction(
        site=site,
        windTurbines=wt,
        wake_deficitModel=deficit_model,
        superpositionModel=WeightedSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=turbulence_model,
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
            "farm_definition_file": layout_file,
            "farm_name": "custom_pywake_farm" if use_pywake_farm else "single_turbine",
            "flow_model": "PyWake.ZongGaussian + CrespoHernandez",
            "turbine_model": turbine_model,
            "rotor_avg": "CGI(21)",
            "site_model": "UniformSite(alpha=0.2)",
        },
    }


import numpy as np
import pandas as pd


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
    dtype=np.float64,
    chunk_size=20000,            # <-- NEW: max number of timesteps per PyWake call
):
    """
    Computes turbine-level inflow and power using PyWake or floris, with optional
    time-windowed execution to reduce memory usage.

    Parameters
    ----------
    wsp : array-like, shape (n_times,)
        Free-stream wind speed (m/s).
    TI : array-like, shape (n_times,)
        Turbulence intensity.
    wdir : array-like, shape (n_times,)
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
    chunk_size : int or None, optional
        If given and smaller than the number of timesteps, the simulation is
        split into time windows of at most `chunk_size` steps and run in
        multiple calls to `flow_model` to reduce memory.

    Returns
    -------
    sim_res : None or PyWake SimulationResult
        - If `chunk_size` is None or >= n_times: full SimulationResult.
        - If chunked: currently returns None (numeric results are concatenated).
          You can change this to return a list of per-chunk SimulationResults.
    sa : None or list or xarray-object
        - If `use_sector_average` is False: None.
        - If True and not chunked: sector-averaged inflow object.
        - If True and chunked: None (numeric outputs concatenated). You can
          adapt this to return a list of per-chunk sector-averaged objects.
    inflow_df : pd.DataFrame
        Per-timestep inflow DataFrame.
        - If rotor-averaged: columns `wsp_idX`, `TI_idX`.
        - If sector-averaged: columns `wspSAk_idX`, `tiSAk_idX` for k in 1..4.
    power_df : pd.DataFrame
        Per-timestep power output in MW with columns `Power_idX`.
    """
    if flow_type != "pywake":
        raise NotImplementedError("Only PyWake is currently supported.")

    # Normalize inputs to arrays
    wsp = np.asarray(wsp)
    TI = np.asarray(TI)
    wdir = np.asarray(wdir)
    yaw = np.asarray(yaw)
    tilt = np.asarray(tilt)
    operating = np.asarray(operating)

    n_times = wsp.shape[0]

    # === Case 1: No chunking needed -> original behavior ===
    if chunk_size is None or chunk_size >= n_times:
        # --- Run PyWake model ---
        sim_res = flow_model(
            x=x,
            y=y,
            ws=wsp,
            wd=wdir,
            TI=TI,
            yaw=yaw,
            tilt=tilt,
            operating=operating,
            time=True,
        )

        # --- Power output ---
        power_MW = sim_res.Power.values.astype(dtype) / 1e6  # (n_turbs, n_times)
        power_df = pd.DataFrame(
            power_MW.T,
            columns=[f"Power_id{i}" for i in range(power_MW.shape[0])]
        )

        if not use_sector_average:
            # Rotor-averaged inflow
            ws_eff = sim_res["WS_eff"].values.astype(dtype)  # (n_turbs, n_times)
            ti_eff = sim_res["TI_eff"].values.astype(dtype)

            inflow_rows = []
            for t in range(ws_eff.shape[1]):
                row = {f"wsp_id{i}": ws_eff[i, t] for i in range(ws_eff.shape[0])}
                row.update({f"TI_id{i}": ti_eff[i, t] for i in range(ti_eff.shape[0])})
                inflow_rows.append(row)

            inflow_df = pd.DataFrame(inflow_rows)
            return sim_res, None, inflow_df, power_df

        # Sector-averaged inflow
        sa = compute_sector_average(
            sim_res,
            n_radius=n_radius,
            n_azimuth=n_azimuth,
            look="downwind",
            use_single_precision=True,
        )

        ws = sa.sel(quantity="WS_eff").values.astype(dtype)  # (n_turbs, n_times, n_sectors)
        ti = sa.sel(quantity="TI_eff").values.astype(dtype)

        n_turbs, n_times_sa, n_sectors = ws.shape

        inflow_rows = []
        for t in range(n_times_sa):
            row = {}
            for i in range(n_turbs):
                for s in range(n_sectors):
                    row[f"wspSA{s+1}_id{i}"] = ws[i, t, s]
                    row[f"tiSA{s+1}_id{i}"] = ti[i, t, s]
            inflow_rows.append(row)

        inflow_df = pd.DataFrame(inflow_rows)
        return sim_res, sa, inflow_df, power_df

    # === Case 2: Chunked execution over time ===

    power_chunks = []
    ws_eff_chunks = []
    ti_eff_chunks = []
    sa_ws_chunks = []
    sa_ti_chunks = []

    # If you want to keep sim_res/sa per chunk, you can store them:
    # sim_res_chunks = []
    # sa_chunks = []

    for start in range(0, n_times, chunk_size):
        end = min(start + chunk_size, n_times)

        wsp_c = wsp[start:end]
        TI_c = TI[start:end]
        wdir_c = wdir[start:end]

        yaw_c = yaw[:, start:end] if yaw.ndim == 2 else yaw
        tilt_c = tilt[:, start:end] if tilt.ndim == 2 else tilt
        operating_c = operating[:, start:end] if operating.ndim == 2 else operating

        # --- Run PyWake on this window ---
        sim_res_c = flow_model(
            x=x,
            y=y,
            ws=wsp_c,
            wd=wdir_c,
            TI=TI_c,
            yaw=yaw_c,
            tilt=tilt_c,
            operating=operating_c,
            time=True,
        )
        # sim_res_chunks.append(sim_res_c)

        # --- Power chunk ---
        power_c = sim_res_c.Power.values.astype(dtype)  # (n_turbs, n_times_chunk)
        power_chunks.append(power_c)

        if not use_sector_average:
            # Rotor-averaged inflow chunk
            ws_eff_c = sim_res_c["WS_eff"].values.astype(dtype)
            ti_eff_c = sim_res_c["TI_eff"].values.astype(dtype)
            ws_eff_chunks.append(ws_eff_c)
            ti_eff_chunks.append(ti_eff_c)
        else:
            # Sector-averaged inflow chunk
            sa_c = compute_sector_average(
                sim_res_c,
                n_radius=n_radius,
                n_azimuth=n_azimuth,
                look="downwind",
                use_single_precision=True,
            )
            # sa_chunks.append(sa_c)
            ws_c = sa_c.sel(quantity="WS_eff").values.astype(dtype)
            ti_c = sa_c.sel(quantity="TI_eff").values.astype(dtype)
            sa_ws_chunks.append(ws_c)
            sa_ti_chunks.append(ti_c)

    # --- Concatenate power over time ---
    power_MW = np.concatenate(power_chunks, axis=1) / 1e6  # (n_turbs, n_times_total)
    power_df = pd.DataFrame(
        power_MW.T,
        columns=[f"Power_id{i}" for i in range(power_MW.shape[0])]
    )

    # --- Build inflow_df from concatenated inflow quantities ---
    if not use_sector_average:
        ws_eff = np.concatenate(ws_eff_chunks, axis=1)  # (n_turbs, n_times_total)
        ti_eff = np.concatenate(ti_eff_chunks, axis=1)

        n_turbs, n_times_total = ws_eff.shape

        inflow_rows = []
        for t in range(n_times_total):
            row = {f"wsp_id{i}": ws_eff[i, t] for i in range(n_turbs)}
            row.update({f"TI_id{i}": ti_eff[i, t] for i in range(n_turbs)})
            inflow_rows.append(row)

        inflow_df = pd.DataFrame(inflow_rows)

        # We don't attempt to concatenate SimulationResult objects here
        # (PyWake-specific). If you need them, return sim_res_chunks instead.
        sim_res = None
        sa = None
        return sim_res, sa, inflow_df, power_df

    else:
        ws = np.concatenate(sa_ws_chunks, axis=1)  # (n_turbs, n_times_total, n_sectors)
        ti = np.concatenate(sa_ti_chunks, axis=1)

        n_turbs, n_times_total, n_sectors = ws.shape

        inflow_rows = []
        for t in range(n_times_total):
            row = {}
            for i in range(n_turbs):
                for s in range(n_sectors):
                    row[f"wspSA{s+1}_id{i}"] = ws[i, t, s]
                    row[f"tiSA{s+1}_id{i}"] = ti[i, t, s]
            inflow_rows.append(row)

        inflow_df = pd.DataFrame(inflow_rows)

        sim_res = None
        sa = None   # or sa_chunks if you want the full detail
        return sim_res, sa, inflow_df, power_df


