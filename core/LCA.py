# core/LCA.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from core.File_Handling import load_yaml, process_duration_fields
from core.utils import apply_overrides, get_input_parameter


class LCA:
    """
    Life Cycle Assessment (LCA) module scaffold.

    Current supported mode(s):
      - "material_inventory": reads env.capex.material_records and computes CO2e
        using Material input YAML factors: Commodity.<material>.LCAParameters.tCO2e_per_t
    """

    def __init__(self, env, logger: Optional[logging.Logger] = None):
        self.env = env
        self.config = env.config
        self.logger = logger or getattr(env, "logger", None) or logging.getLogger("winpact.lca")

        # Optional overrides (consistent pattern with other modules)
        apply_overrides(self, getattr(self.config, "LCA_overrides", {}) or {})

        # ---- Load inputs (module + material inputs are loaded by this module) ----
        self.parameters = self._load_lca_input(self.config)
        self.parameters = get_input_parameter(self.parameters,"LC")
        self.material_input = self._load_material_data(self.config)

        # ---- Read LCA mode + mode-specific settings from LCA input YAML ----
        # Your YAML uses: LCA: { LCA_mode: "material_inventory", material_inventory: {...} }
        self.mode = get_input_parameter(self.parameters, "LCA", "LCA_mode")

        # Mode settings (for now only material_inventory exists)
        self.mass_unit = get_input_parameter(self.parameters, "LCA", self.mode, "mass_unit")
        self.ef_unit = get_input_parameter(self.parameters, "LCA", self.mode, "ef_unit")


        # Emission factor map: {material_name: factor_float}
        # Assumes Material YAML schema:
        # Commodity.<Material>.LCAParameters.tCO2e_per_t: <float>
        self.emission_factors = self._extract_emission_factors(self.material_input)

        # ---- Public outputs ----
        self.lca_records = pd.DataFrame(
            columns=[
                "timestamp",
                "phase_name",
                "category_name",
                "subcategory_name",
                "subsubcategory_name",
                "material_name",
                "mass",
                "mass_unit",
                "emission_factor",
                "emission_factor_unit",
                "co2e",
                "co2e_unit",
                "turbine_id",
                "per_turbine",
            ]
        )

        self.lca_summary_by_material = pd.DataFrame()
        self.lca_summary_total = pd.DataFrame()


    # -------------------------
    # Public API
    # -------------------------
    def start(self) -> None:
        """
        Entry point. Clears outputs and dispatches based on LCA.LCA_mode.

        LCA YAML example:
          LCA:
            LCA_mode: "material_inventory"
            material_inventory:
              mass_unit: "t"
              ef_unit: "tCO2eq_per_t"
        """
        # Clear for repeatability
        self.lca_records = self.lca_records.iloc[0:0]
        self.lca_summary_by_material = pd.DataFrame()
        self.lca_summary_total = pd.DataFrame()

        

        if self.mode == "material_inventory":
            self.calc_material_inventory()
        else:
            raise ValueError(f"Unknown LCA_mode: {self.mode!r}")

        self._build_summaries()

    # -------------------------
    # Mode implementations
    # -------------------------
    def calc_material_inventory(self) -> None:
        capex = getattr(self.env, "capex", None)
        if capex is None or not hasattr(capex, "material_records"):
            self.logger.warning("CAPEX material_records not available. Run CAPEX.start() before LCA.")
            return

        mat_records = capex.material_records
        if not mat_records:
            self.logger.info("No CAPEX material_records found. LCA results will be empty.")
            return

        df = pd.DataFrame(mat_records).copy()

        # Required fields
        for col in ("material_name", "mass"):
            if col not in df.columns:
                raise KeyError(f"CAPEX material_records missing required field: {col}")

        # Optional fields
        optional_cols = [
            "timestamp",
            "phase_name",
            "category_name",
            "subcategory_name",
            "subsubcategory_name",
            "turbine_id",
            "per_turbine",
        ]
        for col in optional_cols:
            if col not in df.columns:
                df[col] = None

        df["mass"] = pd.to_numeric(df["mass"], errors="coerce").fillna(0.0)
        df["mass_unit"] = self.mass_unit

        # ---- EF lookup ----
        df["emission_factor"] = df["material_name"].map(self.emission_factors)
        df["emission_factor_unit"] = self.ef_unit

        # ---- SIMPLE RULE: skip rows with missing EF ----
        missing_mask = df["emission_factor"].isna()
        if missing_mask.any():
            missing_materials = sorted(
                df.loc[missing_mask, "material_name"].dropna().unique().tolist()
            )
            self.logger.warning(
                "Skipping materials with no emission factors: %s", missing_materials
            )
            df = df.loc[~missing_mask].copy()

        # ---- CO2e calculation ----
        df["co2e"] = df["mass"] * pd.to_numeric(df["emission_factor"], errors="coerce").fillna(0.0)
        df["co2e_unit"] = "tCO2e"

        self.lca_records = df[
            [
                "timestamp",
                "phase_name",
                "category_name",
                "subcategory_name",
                "subsubcategory_name",
                "material_name",
                "mass",
                "mass_unit",
                "emission_factor",
                "emission_factor_unit",
                "co2e",
                "co2e_unit",
                "turbine_id",
                "per_turbine",
            ]
        ].reset_index(drop=True)

    # -------------------------
    # Summaries
    # -------------------------
    def _build_summaries(self) -> None:
        if self.lca_records is None or self.lca_records.empty:
            self.lca_summary_by_material = pd.DataFrame(
                columns=["material_name", "co2e", "co2e_unit", "mass", "mass_unit"]
            )
            self.lca_summary_total = pd.DataFrame(columns=["co2e", "co2e_unit"])
            return

        df = self.lca_records.copy()

        self.lca_summary_by_material = (
            df.groupby(["material_name", "co2e_unit", "mass_unit"], dropna=False)[["co2e", "mass"]]
            .sum()
            .reset_index()
            .sort_values("co2e", ascending=False)
        )

        total = float(df["co2e"].sum())
        unit = df["co2e_unit"].iloc[0]
        self.lca_summary_total = pd.DataFrame([{"co2e": total, "co2e_unit": unit}])

    # -------------------------
    # Emission factors
    # -------------------------
    def _extract_emission_factors(self, material_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract emission factors using the *current* assumed schema:

          Commodity:
            Steel:
              LCAParameters:
                tCO2e_per_t: 2.0

        Returns:
          { "Steel": 2.0, ... }
        """
        factors: Dict[str, float] = {}

        for _, mat_file_data in (material_data or {}).items():
            commodities = (mat_file_data or {}).get("Commodity", {})
            if not isinstance(commodities, dict):
                continue

            for material_name, node in commodities.items():
                if material_name in factors:
                    continue

                node = node or {}
                lca_params = node.get("LCAParameters", {}) or {}

                # assume only this key for now (as per your schema)
                ef = lca_params.get("tCO2e_per_t", None)
                if ef is None:
                    continue

                try:
                    factors[str(material_name)] = float(ef)
                except Exception:
                    self.logger.warning(
                        "Non-numeric tCO2e_per_t for %s: %r (skipping)", material_name, ef
                    )

        return factors


    # -------------------------
    # Input loading
    # -------------------------
    def _load_lca_input(self, config) -> Dict[str, Any]:
        """
        Loads the LCA input YAML(s) listed in config.LCA_inputFiles.
        Returns a dict keyed by identifier, same pattern as other modules.
        """
        data: Dict[str, Any] = {}

        files = getattr(config, "LCA_inputFiles", None)
        if not files:
            self.logger.warning("No LCA_inputFiles found on config. LCA will run with defaults.")
            return data

        for identifier, file_name in files.items():
            raw = load_yaml(config.valuewind_inputFolder, file_name)
            data[identifier] = process_duration_fields(raw)

        return data

    def _load_material_data(self, config) -> Dict[str, Any]:
        """
        Loads the Material input YAML(s) listed in config.Material_inputFiles.
        LCA loads these itself to be self-sufficient (not dependent on CAPEX internals).
        """
        data: Dict[str, Any] = {}

        files = getattr(config, "Material_inputFiles", None)
        if not files:
            self.logger.warning("No Material_inputFiles found on config. LCA cannot compute emissions.")
            return data

        for identifier, file_name in files.items():
            raw = load_yaml(config.valuewind_inputFolder, file_name)
            data[identifier] = process_duration_fields(raw)

        return data
    



    # -------------------------# Dashboards
    # ------------------------- 
    def plot_LCA_dashboard(self, show: bool = True):
        """
        LCA dashboard (basic).

        Preserves dashboard style:
        - KPI panel (Plotly indicator)

        KPI:
        - Total Life Cycle Emissions
        """

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # -----------------------------
        # 0) Guards
        # -----------------------------
        df_total = getattr(self, "lca_summary_total", None)
        if not isinstance(df_total, pd.DataFrame) or df_total.empty:
            raise ValueError("LCA dashboard unavailable: missing lca_summary_total (run LCA.start() first).")

        if "co2e" not in df_total.columns:
            raise ValueError("LCA dashboard unavailable: lca_summary_total missing column 'co2e'.")
        if "co2e_unit" not in df_total.columns:
            # keep it robust: default unit if absent
            unit = "tCO2e"
        else:
            unit = str(df_total["co2e_unit"].iloc[0])

        total_co2e = float(pd.to_numeric(df_total["co2e"].iloc[0], errors="coerce") or 0.0)

        # -----------------------------
        # 1) KPI panel
        # -----------------------------
        kpi_fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "indicator"}]],
        )

        kpi_fig.add_trace(go.Indicator(
            mode="number",
            value=total_co2e,
            title={"text": "Total LCA Emissions"},
            number={"valueformat": ",.2f", "suffix": f" {unit}"},
        ), row=1, col=1)

        kpi_fig.update_layout(
            title="LCA KPIs",
            height=260,
            showlegend=False
        )

        if show:
            kpi_fig.show()

        return {"kpis": kpi_fig}
