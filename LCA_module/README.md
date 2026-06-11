# Wind Farm LCA Module

## Setup

1. Install dependencies:
   ```
   pip install pandas openpyxl matplotlib
   ```

2. Place your ecoinvent Excel file in this folder and rename it to:
   ```
   ecoinvent_database.xlsx
   ```
   The code reads from the sheet named **"Ecoinvent data (original)"**,
   skipping the first 3 metadata rows.

3. Run the module:
   ```
   python main.py
   ```

---

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — run this to start the analysis |
| `config.py` | Project variables (turbine count, lifetime, file paths) |
| `inventory_codes.py` | Maps each component to an ecoinvent Process # |
| `inventory_masses.py` | Mass/quantity for each component (per turbine + full farm) |
| `lca_engine.py` | All calculation logic (do not edit unless extending) |
| `results.py` | Terminal output and chart generation |

---

## How to Adapt to a New Project

1. **Update `config.py`** — change turbine count, lifetime, capacity, etc.
2. **Update `inventory_codes.py`** — swap or add ecoinvent process codes.
3. **Update `inventory_masses.py`** — update masses and quantities.
4. Place the new ecoinvent database Excel in the folder.
5. Run `python main.py`.

---

## Adding a New Life Stage

In `inventory_codes.py`, add a new top-level key:
```python
"New Stage Name": {
    "Component Group": {
        "Component A": 12345,   # ecoinvent Process #
        "Component B": 67890,
    }
}
```

Do the same in `inventory_masses.py` with quantities:
```python
"New Stage Name": {
    "Component Group": {
        "Component A": {
            "per_turbine": (1500.0, "kg"),
            "full_farm":   (51000.0, "kg"),
        },
        "Component B": {
            "per_turbine": (200.0, "kg"),
            "full_farm":   (6800.0, "kg"),
        },
    }
}
```

---

## Outputs

Results are saved in the `outputs/` folder:
- `lca_results_per_turbine.csv`
- `lca_results_full_farm.csv`
- `gwp_by_stage_per_turbine.png`
- `gwp_by_stage_full_farm.png`
- `relative_impact_per_turbine.png`
- `relative_impact_full_farm.png`
- `gwp_pie_per_turbine.png`
- `gwp_pie_full_farm.png`
