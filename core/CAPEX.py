import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from core.File_Handling import load_yaml, process_duration_fields


# TOPFARM Cost Model import
# FROM Mads M. Pedersen, Mikkel Friis-Møller, Pierre-Elouan Réthoré, Ernestas Simutis, Riccardo Riva, Julian Quick, Nikolay Krasimirov Dimitrov, Jenni Rinker, & Katherine Dykes. (2025). DTUWindEnergy/TopFarm2: Release of v2.6.1 (v2.6.1). Zenodo. https://doi.org/10.5281/zenodo.17540961
from core.DTU_Cost_Model.dtu_wind_cm_main import economic_evaluation
from core.utils import apply_overrides

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Blade Mass estimator implementation

from maesopt.structure.mcsprop import BladeMassEstimator




class CAPEX:
    """
    CAPEX calculator without SimPy.
    - Extracts a cost schedule from input data.
    - Computes all costs immediately in a deterministic loop.
    - Stores detailed cost records in a DataFrame (self.cost_records).
    - NEW: Precomputes material unit prices once (self.material_unit_prices) and reuses them.
    """

    def __init__(self, env):
        self.rng = np.random.default_rng()
        self.env = env  # Access to configuration and (optionally) logging, randoms, etc.
        self.config = env.config

        # Apply overrides early (before reading CAPEX file)
        apply_overrides(self.config, getattr(self.config, "CAPEX_overrides", {}))

        # Load input data
        self.capex_data, self.material_data = load_capex_data(self.env.config)

        # Extract the "schedule" as-is (timing, category)
        self.cost_items = self.extract_cost_schedule()

        # Project start (used to convert "project_time_h" to timestamps)
        self.project_start = pd.to_datetime(
            self.env.config.Project_StartDate,
            format="%d.%m.%Y"
        )

        # Storage for detailed cost records
        self.cost_records = pd.DataFrame(
            columns=[
                "timestamp",
                "phase_name",          # NEW
                "item_name",
                "category_name",
                "subcategory_name",
                "subsubcategory_name",
                "cost",
                "turbine_id",
                "per_turbine",         #
            ]
        )


        # Optional: track material records if you want to persist masses, etc.
        self.material_records = []  # list of dicts; adapt to your needs

        # Initialize scaling model
        self.scaling_model = economic_evaluation()

        # Aggregate total
        self.total_cost = 0.0

        # NEW: cache for precomputed material unit prices
        # { material_name: unit_price_float }
        self.material_unit_prices = {}

        # Apply overrides again, but now on *the model object*
        apply_overrides(self, getattr(self.config, "CAPEX_overrides", {}))


        # Precompute once at construction (you can move to start() if you prefer)
        self.precompute_material_prices()

        # Blade Mass Estimator implementation
        self.blade_mass_results = None          # raw xarray dataset from estimator
        self.blade_total_mass = None            # kg, per blade
        self.blade_material_mass = {}           # {material_key: mass_kg_per_blade}
        self.n_blades_per_turbine = 3           # or take from config

    # ---------------------------
    # Data extraction
    # ---------------------------
    def extract_cost_schedule(self):
        cost_items = []
        target_phase_names = {
            "Production & Acquisition",
            "Project & Balance of Plant",
            "Balance of Plant",
            "Project & BoP",
            "Decommissioning",
        }

        for _, capex_file_data in self.capex_data.items():
            phases = capex_file_data.get("Phase", [])
            matched_phases = [p for p in phases if p.get("name") in target_phase_names]
            if not matched_phases:
                continue

            for phase in matched_phases:
                phase_name = phase.get("name", "")
                for category in phase.get("categories", []):
                    project_time_h = category.get("project_time_h")
                    per_turbine = category.get("per_turbine", False)
                    if project_time_h is None:
                        continue
                    # include phase_name
                    cost_items.append((phase_name, project_time_h, category, per_turbine))

        return cost_items



    # ---------------------------
    # NEW: Precompute material prices once
    # ---------------------------
    def precompute_material_prices(self):
        """
        Build {material_name: unit_price} from self.material_data.

        Supports:
          - GBM (flag_GBM)
          - Jump-Diffusion (flag_JumpDif)
          - OU on log-price (flag_OU)
        and allows correlated shocks across materials when a correlation
        matrix is provided in the configuration.

        Expected optional config (example):

            self.env.config.MaterialCorrelation = {
                "materials": ["Steel", "Copper", "Aluminium"],
                "matrix": [
                    [1.0, 0.8, 0.6],
                    [0.8, 1.0, 0.5],
                    [0.6, 0.5, 1.0],
                ],
            }

        Any materials not listed in "materials" are treated as
        uncorrelated with others (off-diagonal entries remain 0).
        """
        prices: dict[str, float] = {}

        # ------------------------------------------------------------------
        # 1) Flatten all commodities into a list of (name, params)
        # ------------------------------------------------------------------
        commodity_entries: list[tuple[str, dict]] = []

        for _, material_file_data in self.material_data.items():
            commodities = material_file_data.get("Commodity", {})
            if not isinstance(commodities, dict):
                continue

            for name, node in commodities.items():
                if not name:
                    continue
                # "first wins" behaviour: if name already seen, skip
                if name in prices or any(name == n for n, _ in commodity_entries):
                    continue

                params = (node.get("CostParameters") or {})
                commodity_entries.append((str(name), params))

        if not commodity_entries:
            self.material_unit_prices = {}
            return

        n = len(commodity_entries)

        # ------------------------------------------------------------------
        # 2) Build full correlation matrix for these commodities
        #    Default: independent shocks (identity matrix)
        # ------------------------------------------------------------------
        corr_full = np.eye(n, dtype=float)

        corr_cfg = material_file_data.get("MaterialCorrelation", {})
        
        if corr_cfg is not None:
            corr_materials = list(corr_cfg.get("materials", []))
            corr_matrix_raw = np.array(corr_cfg.get("matrix", []), dtype=float)

            # basic checks
            if (
                corr_matrix_raw.ndim == 2
                and corr_matrix_raw.shape[0] == corr_matrix_raw.shape[1]
                and corr_matrix_raw.shape[0] == len(corr_materials)
            ):
                idx_corr = {m: i for i, m in enumerate(corr_materials)}

                # map correlation entries into full matrix
                for i, (name_i, _) in enumerate(commodity_entries):
                    if name_i not in idx_corr:
                        continue
                    i_corr = idx_corr[name_i]
                    for j, (name_j, _) in enumerate(commodity_entries):
                        if name_j not in idx_corr:
                            continue
                        j_corr = idx_corr[name_j]
                        corr_full[i, j] = corr_matrix_raw[i_corr, j_corr]

        # make sure diagonal is exactly 1
        np.fill_diagonal(corr_full, 1.0)

        # ------------------------------------------------------------------
        # 3) Draw one vector of correlated N(0,1) shocks
        # ------------------------------------------------------------------
        Z = self.sample_correlated_normals(corr_full)  # shape (n,)

        # ------------------------------------------------------------------
        # 4) Compute unit prices using these shocks
        # ------------------------------------------------------------------
        for i, (name, params) in enumerate(commodity_entries):
            base = float(params.get("material_cost", 0) or 0)

            flag_gbm = bool(params.get("flag_GBM", False))
            flag_jump_diff = bool(params.get("flag_JumpDif", False))
            flag_ou = bool(params.get("flag_OU", False))

            mu = float(params.get("mu", 0) or 0)
            sigma = float(params.get("sigma", 0) or 0)

            lambda_jump = float(params.get("lambda_jump", 0) or 0)
            sigma_jump = float(params.get("sigma_jump", 0) or 0)

            kappa = float(params.get("kappa", 0) or 0)
            theta = float(params.get("theta", 0) or 0)
            # allow dedicated sigma_ou, fallback to sigma
            sigma_ou = float(params.get("sigma_ou", params.get("sigma", 0)) or 0)

            timing_hours = float(params.get("prediction_horizon_h", 1) or 1)
            timing_years = timing_hours / 8760.0

            z_i = float(Z[i])  # correlated standard normal shock for this material

            unit_price = base
            if flag_ou:
                unit_price = self.ou_logprice(base, kappa, theta, sigma_ou, timing_years, z=z_i)
            elif flag_gbm:
                unit_price = self.geometric_brownian_motion(base, mu, sigma, timing_years, z=z_i)
            elif flag_jump_diff:
                unit_price = self.jump_diffusion(base, mu, sigma, timing_years, lambda_jump, sigma_jump, z=z_i)

            prices[name] = float(unit_price)

        self.material_unit_prices = prices



    # ---------------------------
    # Public entrypoint
    # ---------------------------
    def start(self):
        """
        Public entrypoint retained for API compatibility.
        Immediately computes all capital costs for the extracted schedule.
        """
        # Reset accumulators for a fresh run
        self.total_cost = 0.0
        self.cost_records = self.cost_records.iloc[0:0]  # clear but keep columns
        self.material_records = []

        # Ensure prices exist (in case caller changed materials/config at runtime)
        if not self.material_unit_prices:
            self.precompute_material_prices()

        self.calculate_capital_costs_for_schedule()
        return None

    # ---------------------------
    # Core calculation loop
    # ---------------------------
    def calculate_capital_costs_for_schedule(self):
        """
        Iterate the extracted schedule (as-is) and compute costs for each item.
        """

        # call external Cost model here
        n_turbines = self.env.WindFarm.n_turbines

        rated_rpm = np.full(n_turbines, 12.0)
        rotor_diameter = np.full(n_turbines, 120.0)
        rated_power = np.full(n_turbines, 3.0)
        hub_height = np.full(n_turbines, 100.0)
        water_depth = np.full(n_turbines, 30.0)
        cabling_cost = 1_000_000.0

        # Evaluate scaling model once per category (as in your original)
        self.scaling_model.calculate_capex(
            rated_rpm, rotor_diameter, rated_power,
            hub_height, water_depth, cabling_cost
        )
        # Call blade mass estimator once
        self.run_blade_mass_estimator()

        for phase_name, timing_hours, category, per_turbine in self.cost_items:
            timestamp = self.project_start + pd.Timedelta(hours=float(timing_hours))
            if per_turbine:
                for turbine_id in range(1, n_turbines + 1):
                    self.calculate_capital_cost(phase_name, category, timestamp, turbine_id, per_turbine=True)
            else:
                self.calculate_capital_cost(phase_name, category, timestamp, None, per_turbine=False)


    # ---------------------------
    # Category-level calculation
    # ---------------------------
    def calculate_capital_cost(self, phase_name, category, timestamp, turbine_id, *, per_turbine: bool):
        """
        Compute and record all costs for a single category at a given timestamp.
        Adds `phase_name` and `per_turbine` to every cost record row.
        """
        item_cost = 0.0
        category_name = category.get("name", "")
        rows = []
        ts = pd.Timestamp(timestamp)

        # Base fields shared by all appended rows
        base_row = {
            "timestamp": ts,
            "phase_name": phase_name,
            "category_name": category_name,
            "turbine_id": turbine_id,
            "per_turbine": per_turbine,
        }

        for subcategory in category.get("subcategories", []) or []:
            subcategory_name = subcategory.get("name", "")
            subsubcategory_name = subcategory_name  # for the subcategory-level row
            total = float(subcategory.get("fixed_cost", 0) or 0)
            beta_markup = float(subcategory.get("beta_markup", 0) or 0)

            # Flags
            flag_material_cost_sub = bool(subcategory.get("flag_material_cost", False))
            use_bme = bool(subcategory.get("use_blade_mass_estimator", False))

            # Normalize materials at subcategory level
            materials_sub = subcategory.get("material", [])
            if isinstance(materials_sub, dict):
                materials_sub = [materials_sub]

            # ---- Subcategory-level material costs ----
            if use_bme:
                # Blade Mass Estimator path (only when explicitly requested)
                # self.run_blade_mass_estimator() no need to call for each turbine

                # Build a lookup: material_name -> CF from YAML
                cf_by_name = {
                    m.get("name"): float(m.get("CF", 1) or 1)
                    for m in materials_sub
                    if m.get("name") is not None
                }

                # Loop over estimator material masses directly
                for material_name, mass_per_blade in self.blade_material_mass.items():
                    # per turbine: multiply by number of blades, convert to tonnes
                    n_blades = self.n_blades_per_turbine
                    total_mass_for_turbine = mass_per_blade * n_blades / 1000.0  # kg → t

                    unit_price = float(self.material_unit_prices.get(material_name, 0.0) or 0.0)
                    cf = cf_by_name.get(material_name, 1.0)

                    ext = total_mass_for_turbine * unit_price * cf
                    total += ext

                    self.material_records.append({
                        "timestamp": ts,
                        "phase_name": phase_name,
                        "category_name": category_name,
                        "subcategory_name": subcategory_name,
                        "subsubcategory_name": subsubcategory_name,
                        "material_name": material_name,
                        "mass": total_mass_for_turbine,
                        "unit_cost": unit_price,
                        "CF": cf,
                        "extended_cost": ext,
                        "turbine_id": turbine_id,
                        "per_turbine": per_turbine,
                        "source": "blade_mass_estimator",
                    })

            elif flag_material_cost_sub:
                # Standard subcategory material costing (no blade estimator)
                for material in materials_sub:
                    material_name = material.get("name")
                    material_cost_per_mass = float(self.material_unit_prices.get(material_name, 0.0) or 0.0)
                    mass = float(material.get("mass", 0) or 0)
                    cf = float(material.get("CF", 1) or 1)

                    ext = mass * material_cost_per_mass * cf
                    total += ext

                    self.material_records.append({
                        "timestamp": ts,
                        "phase_name": phase_name,
                        "category_name": category_name,
                        "subcategory_name": subcategory_name,
                        "subsubcategory_name": subsubcategory_name,
                        "material_name": material_name,
                        "mass": mass,
                        "unit_cost": material_cost_per_mass,
                        "CF": cf,
                        "extended_cost": ext,
                        "turbine_id": turbine_id,
                        "per_turbine": per_turbine,
                    })

            # ---- Subcategory-level scaling model ----
            if subcategory.get("scaling_models", {}).get("flag_DTU_scaling_model", False):
                comp_name = str(subcategory_name).lower()
                scaling_costs = self.scaling_model.turbine_component_costs.get(comp_name, {})

                if isinstance(scaling_costs, dict):
                    for v in scaling_costs.values():
                        if isinstance(v, (np.ndarray, pd.Series)) and turbine_id is not None:
                            total += float(v[turbine_id - 1])
                        else:
                            total += float(v)
                elif isinstance(scaling_costs, (float, int, np.floating)):
                    total += float(scaling_costs)
                elif isinstance(scaling_costs, (np.ndarray, pd.Series)) and turbine_id is not None:
                    total += float(scaling_costs[turbine_id - 1])

            # ---- Append subcategory-level row ----
            total *= (1 + beta_markup)
            rows.append({
                **base_row,
                "item_name": f"{subcategory_name}_cost_item",
                "subcategory_name": subcategory_name,
                "subsubcategory_name": subsubcategory_name,
                "cost": float(total),
            })
            item_cost += float(total)

            # ---- Subsubcategory-level costs ----
            for item in subcategory.get("subsubcategories", []) or []:
                subsubcategory_name = item.get("name", "")
                total = float(item.get("fixed_cost", 0) or 0)

                flag_material_cost_subsub = bool(item.get("flag_material_cost", False))

                # Materials at subsubcategory level
                if flag_material_cost_subsub:
                    materials = item.get("material", [])
                    if isinstance(materials, dict):
                        materials = [materials]

                    for material in materials:
                        material_name = material.get("name")
                        material_cost_per_mass = float(self.material_unit_prices.get(material_name, 0.0) or 0.0)
                        mass = float(material.get("mass", 0) or 0)
                        cf = float(material.get("CF", 1) or 1)

                        ext = mass * material_cost_per_mass * cf
                        total += ext

                        self.material_records.append({
                            "timestamp": ts,
                            "phase_name": phase_name,
                            "category_name": category_name,
                            "subcategory_name": subcategory_name,
                            "subsubcategory_name": subsubcategory_name,
                            "material_name": material_name,
                            "mass": mass,
                            "unit_cost": material_cost_per_mass,
                            "CF": cf,
                            "extended_cost": ext,
                            "turbine_id": turbine_id,
                            "per_turbine": per_turbine,
                        })

                # Scaling at subsubcategory level
                if item.get("scaling_models", {}).get("flag_DTU_scaling_model", False):
                    parent_comp = str(subcategory_name).lower()
                    scaling_costs = self.scaling_model.turbine_component_costs.get(parent_comp, {})
                    if isinstance(scaling_costs, dict):
                        val = scaling_costs.get(subsubcategory_name, 0)
                        if isinstance(val, (np.ndarray, pd.Series)) and turbine_id is not None:
                            total += float(val[turbine_id - 1])
                        else:
                            total += float(val)

                # Append subsubcategory-level row
                total *= (1 + beta_markup)
                rows.append({
                    **base_row,
                    "item_name": f"{subsubcategory_name}_cost_item",
                    "subcategory_name": subcategory_name,
                    "subsubcategory_name": subsubcategory_name,
                    "cost": float(total),
                })
                item_cost += float(total)

        # Batch-append to the records DataFrame
        if rows:
            self.cost_records = pd.concat([self.cost_records, pd.DataFrame(rows)], ignore_index=True)

        self.total_cost += float(item_cost)


        #print(f"Total capital cost for category '{category_name}': {item_cost}")

        # Placeholders for future categories:
        # - Installation & Commissioning (IC)
        # - Development & Consenting (DC)
        # - Decommissioning & Disposal (DD)

    def run_blade_mass_estimator(self):
        # 1) Load windIO turbine data (path or object comes from env.config)
        
        from maesopt.structure.mcsprop import BladeMassEstimator
        from windIO import load_yaml as load_yaml
        import windIO.examples.turbine as wio_turb
        from pathlib import Path
        wio_data = load_yaml(Path(wio_turb.__file__).parent/"IEA-22-280-RWT.yaml")

        # 2) Build and populate the estimator
        bme = BladeMassEstimator()
        bme.from_windIO(wio_data)

        # 3) Run the estimator and store results
        out = bme.run(return_flag=3)  # xarray.Dataset

        self.blade_mass_results = out
        self.blade_total_mass = float(out["mass"])  # kg, per blade

        # 4) Extract a per-material mass breakdown
        # depends on how the estimator names dims, but conceptually:
        material_names = list(out.material_names.values)     # ["glass", "carbon", ...]
        material_masses = list(out.material_mass.values)     # [kg]

        self.blade_material_mass = dict(zip(material_names, material_masses))


    def get_cost_dataframe(self):
        """Converts the cost records list to a DataFrame for further analysis."""
        return pd.DataFrame(self.cost_records)
    


    def plot_cost_pies(self, turbine_id: int, **kwargs):
        """
        Wrapper around `plot_capex_cost_pies` using this instance's cost_records.
        """
        # Plot the dashboard first
        capex_dashboard(self, turbine_id=turbine_id)
        #material_dashboard(self, turbine_id=turbine_id)
        #plot_capex_cost_pies(self.cost_records, turbine_id, **kwargs)

        return None
    
    def sample_correlated_normals(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Draw a vector Z ~ N(0, corr_matrix) using Cholesky decomposition.

        corr_matrix: (M x M) symmetric positive definite correlation matrix.
        Returns:
            np.ndarray of shape (M,)
        """
        M = corr_matrix.shape[0]
        # Cholesky factorisation: corr = L L^T
        L = np.linalg.cholesky(corr_matrix)
        z_indep = self.rng.normal(0.0, 1.0, size=M)
        return L @ z_indep


    
    def geometric_brownian_motion(self, S0, mu, sigma, T, z=None):
        """
        Calculate a random realization of Geometric Brownian Motion.

        If z is provided, it is treated as a standard normal shock (N(0,1))
        and scaled by sqrt(T). Otherwise, a new normal is drawn internally.
        """
        if T <= 0:
            return float(S0)

        if z is None:
            # draw internally (independent)
            W = self.rng.normal(0.0, np.sqrt(T))
        else:
            # external standard normal shock → Brownian increment over [0,T]
            W = np.sqrt(T) * float(z)

        return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W)

    def jump_diffusion(self, S0, mu, sigma, T, lambda_jump, sigma_jump, z=None):
        """
        Calculate a random realization of Jump Diffusion process.

        The diffusion part can use a correlated normal shock `z` if provided.
        Jumps are kept independent across commodities (for now).
        """
        # Diffusion (GBM) component
        S_t = self.geometric_brownian_motion(S0, mu, sigma, T, z=z)

        # Number of jumps in the time interval (independent across materials)
        if T > 0 and lambda_jump > 0:
            num_jumps = self.rng.poisson(lambda_jump * T)
        else:
            num_jumps = 0

        # Apply jump diffusion adjustments
        for _ in range(num_jumps):
            jump_size = self.rng.normal(0.0, sigma_jump)
            S_t *= np.exp(jump_size)

        return float(S_t)

    def ou_logprice(self, S0, kappa, theta, sigma, T, z=None):
        """
        One-step Ornstein–Uhlenbeck (OU) evolution on log-price.

        OU on X_t = ln S_t:
            dX_t = kappa * (theta - X_t) dt + sigma dW_t

        Exact solution over horizon T:
            X_T ~ N(m(T), v(T)) with:
                m(T) = theta + (X0 - theta) * exp(-kappa * T)
                v(T) = (sigma^2 / (2*kappa)) * (1 - exp(-2 * kappa * T))

        If z is provided, it is treated as N(0,1) and used as the OU shock.
        Otherwise, a new normal is drawn internally.
        """
        if T <= 0:
            return float(S0)

        X0 = np.log(S0)

        if kappa <= 0:
            # Fallback: treat as GBM on log-price with external z if given
            if z is None:
                Z = self.rng.normal(0.0, 1.0)
            else:
                Z = float(z)
            X_T = X0 + (theta - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
            return float(np.exp(X_T))

        # Mean and variance of X_T
        exp_term = np.exp(-kappa * T)
        m_T = theta + (X0 - theta) * exp_term
        var_T = (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * T))

        if var_T < 0:
            var_T = 0.0

        # Sample from Normal(m_T, var_T)
        if z is None:
            Z = self.rng.normal(0.0, 1.0)
        else:
            Z = float(z)

        X_T = m_T + np.sqrt(var_T) * Z

        return float(np.exp(X_T))



def load_capex_data(config):
    """
    Loads CAPEX and Material input parameters from configuration files, 
    converting any nested duration parameters to hours.

    Parameters
    ----------
    config : Configuration
        The configuration object containing paths to CAPEX and Material input files.

    Returns
    -------
    tuple
        Two dictionaries containing CAPEX data and Material data, respectively.
    """
    capex_data = {}
    material_data = {}

    # Load CAPEX input files
    if hasattr(config, 'Capex_inputFiles'):
        for identifier, file_name in config.Capex_inputFiles.items():
            capex_data[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            capex_data[identifier] = process_duration_fields(capex_data[identifier])  # Process for duration fields

        #print("Loaded CAPEX data structure:", capex_data)

    # Load Material input files
    if hasattr(config, 'Material_inputFiles'):
        for identifier, file_name in config.Material_inputFiles.items():
            material_data[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            material_data[identifier] = process_duration_fields(material_data[identifier])  # Process for duration fields

        #print("Loaded Material data structure:", material_data)

    return capex_data, material_data





def plot_capex_cost_pies(cost_records: pd.DataFrame,
                         turbine_id: int,
                         *,
                         figsize_turbine=(8, 8),
                         figsize_overall=(6, 6),
                         ring_width=0.28,
                         drop_zeros=True,
                         textsize_outer=7,
                         textsize_middle=8,
                         textsize_inner=9,
                         textsize_overall=10,
                         show=True):
    """
    Draws:
      1) Per-turbine nested donut: inner=Category, middle=Subcategory, outer=Subsubcategory.
      2) Project-level pie by Category: (sum of all per-turbine rows) + (all per_turbine=False / BoP rows).
    """
    required = {"category_name", "subcategory_name", "subsubcategory_name", "cost", "turbine_id"}
    missing = required.difference(cost_records.columns)
    if missing:
        raise ValueError(f"cost_records is missing required columns: {sorted(missing)}")

    df = cost_records.copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    if drop_zeros:
        df = df[df["cost"] > 0]

    def _labelize(x):
        if pd.isna(x) or (isinstance(x, str) and x.strip() == ""):
            return "Unspecified"
        return str(x)

    def make_autopct(values):
        values = np.asarray(values, dtype=float)
        total = values.sum() if values.size else 0.0
        def _inner(pct):
            if total <= 0:
                return ""
            val = pct * total / 100.0
            return f"{val:,.0f}\n({pct:.1f}%)"
        return _inner

    # ---------- Robust turbine selection ----------
    # mask that matches both numeric and string reps of turbine_id
    def _mask_for_tid(series, tid):
        num_match = pd.to_numeric(series, errors="coerce") == tid
        str_match = series.astype(str) == str(tid)
        return (num_match.fillna(False)) | (str_match.fillna(False))

    dft = df[_mask_for_tid(df["turbine_id"], turbine_id)]

    if dft.empty:
        # try fallback: pick the first available turbine id (if any)
        available = df["turbine_id"].dropna().unique()
        if available.size > 0:
            fallback_id = available[0]
            #print(f"plot_capex_cost_pies: turbine_id={turbine_id} not found; using turbine_id={fallback_id} instead.")
            dft = df[_mask_for_tid(df["turbine_id"], fallback_id)]
            used_turbine_id = fallback_id
        else:
            used_turbine_id = None
    else:
        used_turbine_id = turbine_id

    # ---------- Per-turbine nested donut ----------
    if used_turbine_id is None or dft.empty:
        # No per-turbine data available -> render a placeholder figure
        fig_turbine, ax = plt.subplots(figsize=figsize_turbine)
        ax.text(0.5, 0.5, "No per-turbine costs available", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        cat_sum = (dft.groupby("category_name", dropna=False)["cost"].sum().sort_values(ascending=False))
        subcat_sum = (dft.groupby(["category_name", "subcategory_name"], dropna=False)["cost"].sum())
        subsub_sum = (dft.groupby(["category_name", "subcategory_name", "subsubcategory_name"], dropna=False)["cost"].sum())

        inner_labels, inner_sizes = [], []
        middle_labels, middle_sizes = [], []
        outer_labels, outer_sizes = [], []

        for cat in cat_sum.index:
            inner_labels.append(_labelize(cat))
            inner_sizes.append(float(cat_sum.loc[cat]))

            # Subcategories
            try:
                sc_series = subcat_sum.loc[cat]
                if isinstance(sc_series, pd.Series):
                    for subcat, sc_val in sc_series.items():
                        middle_labels.append(_labelize(subcat))
                        middle_sizes.append(float(sc_val))

                        # Subsubcategories
                        try:
                            ssub_series = subsub_sum.loc[(cat, subcat)]
                            if isinstance(ssub_series, pd.Series):
                                for ssub, ss_val in ssub_series.items():
                                    outer_labels.append(_labelize(ssub))
                                    outer_sizes.append(float(ss_val))
                            else:
                                outer_labels.append("Unspecified")
                                outer_sizes.append(float(ssub_series))
                        except KeyError:
                            pass
                else:
                    middle_labels.append("Unspecified")
                    middle_sizes.append(float(sc_series))
            except KeyError:
                pass

        fig_turbine, ax = plt.subplots(figsize=figsize_turbine)
        startangle = 90

        # OUTER ring (Subsubcategory)
        if len(outer_sizes) and np.sum(outer_sizes) > 0:
            ax.pie(
                outer_sizes,
                labels=outer_labels,
                autopct=make_autopct(outer_sizes),
                pctdistance=0.85,
                labeldistance=1.08,
                radius=1.0,
                startangle=startangle,
                wedgeprops=dict(width=ring_width),
                textprops=dict(fontsize=textsize_outer),
            )

        # MIDDLE ring (Subcategory)
        if len(middle_sizes) and np.sum(middle_sizes) > 0:
            ax.pie(
                middle_sizes,
                labels=middle_labels,
                autopct=make_autopct(middle_sizes),
                pctdistance=0.75,
                labeldistance=0.95,
                radius=1.0 - ring_width,
                startangle=startangle,
                wedgeprops=dict(width=ring_width),
                textprops=dict(fontsize=textsize_middle),
            )

        # INNER ring (Category)
        if len(inner_sizes) and np.sum(inner_sizes) > 0:
            ax.pie(
                inner_sizes,
                labels=inner_labels,
                autopct=make_autopct(inner_sizes),
                pctdistance=0.65,
                labeldistance=0.85,
                radius=1.0 - 2 * ring_width,
                startangle=startangle,
                wedgeprops=dict(width=ring_width),
                textprops=dict(fontsize=textsize_inner, weight="bold"),
            )

        ax.set(aspect="equal")
        ax.set_title(f"Turbine {used_turbine_id} — Cost distribution\n(Category → Subcategory → Subsubcategory)")
        plt.tight_layout()

    # ---------- Project-level pie (sum of all turbines + BoP/project) ----------
    if "per_turbine" in df.columns:
        turbine_rows = df[df["per_turbine"] == True]
        project_rows = df[df["per_turbine"] == False].copy()
        # collapse duplicated project rows if they appear once per turbine_id
        if not project_rows.empty and project_rows["turbine_id"].notna().any() and project_rows["turbine_id"].nunique() > 1:
            project_rows = (
                project_rows
                .groupby(["category_name", "subcategory_name", "subsubcategory_name"], dropna=False, as_index=False)["cost"]
                .mean()
            )
    else:
        # heuristic: project rows have missing turbine_id
        project_rows = df[df["turbine_id"].isna()].copy()
        turbine_rows = df[df["turbine_id"].notna()].copy()
        if not project_rows.empty and project_rows["turbine_id"].nunique() > 1:
            project_rows = (
                project_rows
                .groupby(["category_name", "subcategory_name", "subsubcategory_name"], dropna=False, as_index=False)["cost"]
                .mean()
            )

    overall_parts = []
    if not turbine_rows.empty:
        overall_parts.append(
            turbine_rows.groupby("category_name", dropna=False)["cost"].sum().rename("cost")
        )
    if not project_rows.empty:
        overall_parts.append(
            project_rows.groupby("category_name", dropna=False)["cost"].sum().rename("cost")
        )

    if overall_parts:
        overall_cat = pd.concat(overall_parts, axis=0).groupby(level=0).sum().sort_values(ascending=False)
    else:
        overall_cat = pd.Series(dtype=float)

    overall_labels = [_labelize(c) for c in overall_cat.index]
    overall_sizes = overall_cat.values

    fig_overall, ax2 = plt.subplots(figsize=figsize_overall)
    if overall_sizes.size and np.sum(overall_sizes) > 0:
        ax2.pie(
            overall_sizes,
            labels=overall_labels,
            autopct=make_autopct(overall_sizes),
            pctdistance=0.7,
            labeldistance=1.05,
            startangle=90,
            textprops=dict(fontsize=textsize_overall),
        )
    else:
        ax2.text(0.5, 0.5, "No project costs to display", ha="center", va="center", fontsize=12)
        ax2.axis("off")

    ax2.set(aspect="equal")
    ax2.set_title("Project cost distribution — (All turbines + Balance of Plant) by Category")
    plt.tight_layout()

    if show:
        plt.show()

    return fig_turbine, fig_overall



def capex_dashboard(
    self,
    turbine_id: int = 1,
    phase_filter: str | None = None,
    drop_zeros: bool = True,
    show: bool = True,
):
    """
    Interactive CAPEX dashboard using Plotly.

    KPIs:
      - Total Project CAPEX
      - Turbine <turbine_id> CAPEX
      - Turbine <turbine_id> Rotor Cost (3_blades)
      - Share of Turbine CAPEX (all turbines) from Total CAPEX

    Figure 2: Sunburst (per-turbine breakdown) + Sunburst (project breakdown)
    """
    if not isinstance(self.cost_records, pd.DataFrame) or self.cost_records.empty:
        raise ValueError("No CAPEX cost_records available. Run CAPEX.start() first.")

    df = self.cost_records.copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)

    if drop_zeros:
        df = df[df["cost"] > 0]

    if phase_filter is not None:
        df = df[df["phase_name"] == phase_filter]

    if df.empty:
        raise ValueError("No CAPEX data available after filtering (phase_filter / drop_zeros).")

    # -------- Helpers --------
    def _labelize(x: str | None, fallback: str = "Unspecified"):
        if pd.isna(x):
            return fallback
        if isinstance(x, str) and x.strip() == "":
            return fallback
        return str(x)

    def _mask_for_tid(series, tid):
        """Match both numeric and string representations of turbine_id."""
        num_match = pd.to_numeric(series, errors="coerce") == tid
        str_match = series.astype(str) == str(tid)
        return (num_match.fillna(False)) | (str_match.fillna(False))

    # -------- Basic splits --------
    has_per_turbine = "per_turbine" in df.columns

    if has_per_turbine:
        turbine_rows = df[df["per_turbine"] == True]
        project_rows = df.copy()  # project view uses all rows
    else:
        # Heuristic fallback: turbine rows = those with a turbine_id
        turbine_rows = df[df["turbine_id"].notna()]
        project_rows = df.copy()

    # Number of turbines present
    if not turbine_rows.empty:
        n_turbines = turbine_rows["turbine_id"].dropna().nunique()
    else:
        n_turbines = 0

    # Raw totals in base currency (e.g. EUR)
    total_project_capex = float(project_rows["cost"].sum()) if not project_rows.empty else 0.0

    # -------- Per-turbine dataframe (selected turbine) --------
    dft = pd.DataFrame(columns=df.columns)
    used_turbine_id = None

    if not turbine_rows.empty:
        dft = turbine_rows[_mask_for_tid(turbine_rows["turbine_id"], turbine_id)]
        if dft.empty:
            # fallback: first turbine we find
            available = turbine_rows["turbine_id"].dropna().unique()
            if available.size > 0:
                used_turbine_id = available[0]
                dft = turbine_rows[_mask_for_tid(turbine_rows["turbine_id"], used_turbine_id)]
            else:
                used_turbine_id = None
        else:
            used_turbine_id = turbine_id

    # ===========================
    #          KPIs
    # ===========================
    # 1) Total project CAPEX (M€)
    kpi_total_capex_m = total_project_capex / 1e6

    # 2) Turbine CAPEX (selected turbine)
    selected_turbine_capex_raw = float(dft["cost"].sum()) if not dft.empty else 0.0
    selected_turbine_capex_m = selected_turbine_capex_raw / 1e6

    # 3) Turbine Rotor Cost (3_blades) for selected turbine
    if not dft.empty:
        mask_3_blades = (
            (dft["category_name"] == "3_blades") |
            (dft["subcategory_name"] == "3_blades") |
            (dft["subsubcategory_name"] == "3_blades")
        )
        rotor_cost_raw = float(dft.loc[mask_3_blades, "cost"].sum())
    else:
        rotor_cost_raw = 0.0
    rotor_cost_m = rotor_cost_raw / 1e6

    # 4) Share of Turbine CAPEX (all turbines) from total CAPEX
    if not turbine_rows.empty:
        per_turbine_capex_total_raw = float(turbine_rows["cost"].sum())
    else:
        per_turbine_capex_total_raw = 0.0

    share_turbine_capex_total = (
        per_turbine_capex_total_raw / total_project_capex
        if total_project_capex > 0 else 0.0
    )

    # -------- Sunburst builders --------
    def build_sunburst_data(df_base: pd.DataFrame):
        """
        Build labels / parents / values / ids for a 3-level sunburst:
        Category → Subcategory → Subsubcategory
        Values are in M€.
        """
        if df_base.empty:
            return ["No data"], [""], [1.0], ["root"]

        cat_sum = (
            df_base
            .groupby("category_name", dropna=False)["cost"]
            .sum()
            .sort_values(ascending=False)
        )
        subcat_sum = (
            df_base
            .groupby(["category_name", "subcategory_name"], dropna=False)["cost"]
            .sum()
        )
        subsub_sum = (
            df_base
            .groupby(["category_name", "subcategory_name", "subsubcategory_name"], dropna=False)["cost"]
            .sum()
        )

        labels = []
        parents = []
        values = []
        ids = []

        # categories (top ring)
        for cat, v_cat in cat_sum.items():
            clab = _labelize(cat, "Unspecified category")
            cid = f"cat:{clab}"
            labels.append(clab)
            parents.append("")
            values.append(float(v_cat) / 1e6)   # M€
            ids.append(cid)

        # subcategories
        for (cat, subcat), v_sc in subcat_sum.items():
            clab = _labelize(cat, "Unspecified category")
            slab = _labelize(subcat, "Unspecified subcategory")
            cid = f"cat:{clab}"
            sid = f"{cid}|sub:{slab}"
            labels.append(slab)
            parents.append(cid)
            values.append(float(v_sc) / 1e6)    # M€
            ids.append(sid)

        # subsubcategories
        for (cat, subcat, subsub), v_ss in subsub_sum.items():
            clab = _labelize(cat, "Unspecified category")
            slab = _labelize(subcat, "Unspecified subcategory")
            ssub_lab = _labelize(subsub, "Unspecified subsubcategory")
            cid = f"cat:{clab}"
            sid = f"{cid}|sub:{slab}"
            ssid = f"{sid}|ssub:{ssub_lab}"
            labels.append(ssub_lab)
            parents.append(sid)
            values.append(float(v_ss) / 1e6)    # M€
            ids.append(ssid)

        return labels, parents, values, ids

    labels_t, parents_t, values_t, ids_t = build_sunburst_data(dft)
    labels_p, parents_p, values_p, ids_p = build_sunburst_data(project_rows)

    # =====================================================================
    #  FIGURE 1: KPI INDICATORS
    # =====================================================================
    fig_kpi = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"} for _ in range(4)]],
        horizontal_spacing=0.06,
    )

    # KPI 1: Total project CAPEX
    fig_kpi.add_trace(
        go.Indicator(
            mode="number",
            value=float(kpi_total_capex_m),
            title={"text": "Total Project CAPEX"},
            number={"valueformat": ",.1f", "suffix": " M€"},
        ),
        row=1, col=1,
    )

    # KPI 2: Turbine CAPEX (selected turbine)
    fig_kpi.add_trace(
        go.Indicator(
            mode="number",
            value=float(selected_turbine_capex_m),
            title={"text": f"Turbine {used_turbine_id if used_turbine_id is not None else turbine_id} CAPEX"},
            number={"valueformat": ",.1f", "suffix": " M€"},
        ),
        row=1, col=2,
    )

    # KPI 3: Turbine Rotor Cost (3_blades)
    fig_kpi.add_trace(
        go.Indicator(
            mode="number",
            value=float(rotor_cost_m),
            title={"text": f"Turbine {used_turbine_id if used_turbine_id is not None else turbine_id} Rotor Cost<br><sub>3_blades</sub>"},
            number={"valueformat": ",.2f", "suffix": " M€"},
        ),
        row=1, col=3,
    )

    # KPI 4: Share of Turbine CAPEX (all turbines) from total CAPEX
    fig_kpi.add_trace(
        go.Indicator(
            mode="number",
            value=float(share_turbine_capex_total),
            title={"text": "Share of Turbine CAPEX<br><sub>All turbines / Total CAPEX</sub>"},
            number={"valueformat": ".1%"},
        ),
        row=1, col=4,
    )

    title_suffix = f" — Phase: {phase_filter}" if phase_filter is not None else ""
    fig_kpi.update_layout(
        title=dict(
            text=f"CAPEX Overview — KPIs{title_suffix}",
            font=dict(size=26),
            x=0.01,
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=20),
        height=250,
    )

    # =====================================================================
    #  FIGURE 2: SUNBURSTS (big, clean layout)
    # =====================================================================
    fig_sun = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.08,
    )

    fig_sun.add_trace(
        go.Sunburst(
            labels=labels_t,
            parents=parents_t,
            values=values_t,
            ids=ids_t,
            branchvalues="total",
            hovertemplate=(
                "%{label}<br>"
                "Cost: %{value:,.2f} M€<br>"
                "Share: %{percentParent:.1%} of parent<br>"
                "Global: %{percentRoot:.1%} of total<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    fig_sun.add_trace(
        go.Sunburst(
            labels=labels_p,
            parents=parents_p,
            values=values_p,
            ids=ids_p,
            branchvalues="total",
            hovertemplate=(
                "%{label}<br>"
                "Cost: %{value:,.2f} M€<br>"
                "Share: %{percentParent:.1%} of parent<br>"
                "Global: %{percentRoot:.1%} of total<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    fig_sun.update_layout(
        title=dict(
            text=f"CAPEX Breakdown — Turbine vs Project{title_suffix}",
            font=dict(size=26),
            x=0.01,
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
        height=700,
        annotations=[
            dict(
                text=f"Turbine {used_turbine_id if used_turbine_id is not None else turbine_id}",
                x=0.19, y=0.02, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            ),
            dict(
                text="Project Total",
                x=0.81, y=0.02, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            ),
        ],
    )

    if show:
        fig_kpi.show()
        fig_sun.show()

    return fig_kpi, fig_sun
