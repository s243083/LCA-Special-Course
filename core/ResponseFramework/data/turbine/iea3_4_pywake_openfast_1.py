# -*- coding: utf-8 -*-
"""
PyWake model of the IEA 3.4 MW.

@author: ricriv
"""

# %% Import.

import os

import numpy as np
import xarray as xr
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import (
    PowerCtXr,
)

# Code that runs upon module import.


# %% PyWake model of the IEA 3.4 using data from Adrien.
#    This model includes curtailment.

# Load Look Up Table made by Adrien based on steady wind OpenFAST simulations.
# Rename some keys because they are hard-coded in PyWake.
# Reverse one of the dimension because PyWake GridInterpolator wants them in ascending order.
# Solution from https://stackoverflow.com/a/72970086/3676517
iea3_4_openfast_lut_with_curtailment = (
    xr.load_dataset(f"{os.path.dirname(__file__)}/LUT_IEA3_4MW_WithAllCurtailment.nc")
    .rename(
        {
            "WindSpeeds": "ws",
            "PowerDemand_elec_PercentOfRated": "power_demand",
            "Ct": "ct",
            "Power_elec": "power",
        }
    )
    .isel(power_demand=slice(None, None, -1))
)

iea3_4 = WindTurbine(
    name="IEA 3.4 MW OpenFAST",
    diameter=130.0,
    hub_height=110.0,
    powerCtFunction=PowerCtXr(
        iea3_4_openfast_lut_with_curtailment,
        power_unit="W",
        # additional_models=[],
    ),
)


# %% Main.

if __name__ == "__main__":

    # Get performance.
    ws = np.linspace(3.0, 20.0, 171)
    power_demand = np.arange(100, 5, -10)

    power = np.full((ws.size, power_demand.size), np.nan)
    ct = power.copy()

    for j in range(power_demand.size):
        power[:, j] = iea3_4.power(ws, power_demand=power_demand[j])
        ct[:, j] = iea3_4.ct(ws, power_demand=power_demand[j])

    pitch = (
        iea3_4_openfast_lut_with_curtailment["Pitch"]
        .interp(
            coords={
                "ws": ws,
                "power_demand": power_demand,
            }
        )
        .to_numpy()
    )

    rotor_speed = (
        iea3_4_openfast_lut_with_curtailment["RotorSpeed"]
        .interp(
            coords={
                "ws": ws,
                "power_demand": power_demand,
            }
        )
        .to_numpy()
    )

    # Plot performance.
    import matplotlib.pyplot as plt

    plt.close("all")
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("OpenFAST")
    ax.set_xlabel("Wind speed")
    ax.set_ylabel("Power [kW]")
    ax.grid(True)
    for i in range(power_demand.size):
        ax.plot(ws, power[:, i] / 1000.0, label=f"{power_demand[i]}%")
    ax.legend(loc="lower right", title="Power demand")

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("OpenFAST")
    ax.set_xlabel("Wind speed")
    ax.set_ylabel("Thrust coefficient [-]")
    ax.grid(True)
    for i in range(power_demand.size):
        ax.plot(ws, ct[:, i], label=f"{power_demand[i]}%")
    ax.legend(loc="upper right", title="Power demand")

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("OpenFAST")
    ax.set_xlabel("Wind speed")
    ax.set_ylabel("Rotor speed [rpm]")
    ax.grid(True)
    for i in range(power_demand.size):
        ax.plot(ws, rotor_speed[:, i], label=f"{power_demand[i]}%")
    ax.legend(loc="lower right", title="Power demand")

    fig, ax = plt.subplots(dpi=300)
    ax.set_title("OpenFAST")
    ax.set_xlabel("Wind speed")
    ax.set_ylabel("Pitch [deg]")
    ax.grid(True)
    for i in range(power_demand.size):
        ax.plot(ws, pitch[:, i], label=f"{power_demand[i]}%")
    ax.legend(loc="upper left", title="Power demand")
