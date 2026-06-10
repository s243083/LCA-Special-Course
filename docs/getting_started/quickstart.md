# Quickstart

A minimal end-to-end WINPACT run.

```python
from core.SimulationConfig import SimulationConfig
from core.Simulation import Simulation

cfg = SimulationConfig.from_yaml("examples/Inputs/example_config.yaml")
sim = Simulation(cfg)
results = sim.run()

print(results.summary())
```

For a full walk-through, see [the WINPACT API tutorial](../tutorials/winpact_api).

```{note}
Stub â€” replace example paths with a committed minimal input set in
`examples/Inputs/` so the snippet is runnable as-is.
```
