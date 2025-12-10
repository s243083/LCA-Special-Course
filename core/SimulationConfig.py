from __future__ import annotations
from typing import Any, Mapping
from attrs import define

@define(auto_attribs=True)
class SimulationConfig:
    """
    Controls which modules run and which plots are produced in ValueWindEnv.run_simulation.
    """
    # High-level module switches
    run_capex: bool = True
    run_marketenv: bool = True
    run_metenv: bool = True
    capex_dashboard: bool = True
    run_opex: bool = True
    opex_dashboard: bool = True
    run_windfarm: bool = True
    run_opex: bool = True
    run_lifetime_extension: bool = False
    run_revenue: bool = True
    run_valuation: bool = True
    valuation_dashboard: bool = False
    collect_results: bool = True



    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SimulationConfig":
        """
        Convenience constructor to build SimulationConfig from a plain dict.
        Extra keys are ignored.
        """
        allowed = cls.__annotations__.keys()
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)
