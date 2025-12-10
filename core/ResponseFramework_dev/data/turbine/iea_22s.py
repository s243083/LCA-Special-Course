# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


DATA_PATH = Path(__file__).parent


# IEA 22 MW reference turbine based on HAWC2 run (@vaspas)
IEA22s_data = pd.read_csv(DATA_PATH / "IEA22_operational_curves_HAWC2_smooth.csv", sep=",")
IEA22s_power_curve = np.array([IEA22s_data["WindSpeed"], IEA22s_data["PowerAc"]]).T
IEA22s_ct_curve = np.array(
    [IEA22s_data["WindSpeed"], IEA22s_data["Ct"]]
).T


class IEA22s(WindTurbine):
    def __init__(self, method="pchip"):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(
            self,
            name="IEA22s",
            diameter=284,
            hub_height=170,
            powerCtFunction=PowerCtTabular(
                IEA22s_power_curve[:, 0],
                IEA22s_power_curve[:, 1],
                "W",
                IEA22s_ct_curve[:, 1],
                method="pchip",
                ws_cutin=3,
                ws_cutout=25,
                ct_idle=0.0,
            ),
        )


def main():
    wt = IEA22s()
    print("Diameter", wt.diameter())
    print("Hub height", wt.hub_height())
    ws = np.arange(0, 25)
    import matplotlib.pyplot as plt

    plt.plot(ws, wt.power(ws), ".-", label="power [W]")
    c = plt.plot([], label="ct")[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), ".-", color=c)
    plt.show()
    print('d')


if __name__ == "__main__":
    main()
