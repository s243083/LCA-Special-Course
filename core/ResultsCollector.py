# -------------- Result Collector ---------------------#
from __future__ import annotations
from pathlib import Path

import pandas as pd
import itertools, time, hashlib, gc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, List
from datetime import datetime
import re  # ensure this is imported


class ResultsCollector:
    """
    Collects record DataFrames from modules on the env and stores them
    in a single combined Parquet file at
      <results_root[/experiment_name]>/records.parquet.
    All paths and identifiers are read from env.config if available:
      - result_directory (preferred) or results_root (fallback)
      - experiment_name (optional subfolder)
      - scenario_id, scenario_label, seed (metadata columns)
    """

    def __init__(self, env) -> None:
        self.env = env
        config = self.env.config

        # ---- Direct access to config fields ----
        result_directory = config.result_directory          # required
        self.experiment_name = config.experiment_name       # may be None/""

        # Build final root path (optionally nest by experiment)
        base_root = Path(result_directory)
        self.results_root = base_root / self.experiment_name if self.experiment_name else base_root
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.out_path = self.results_root / "records.parquet"

        # Cache common metadata from config for convenience
        self.cfg_scenario_id = str(config.scenario_id)
        self.cfg_label = config.scenario_label
        self.cfg_seed = config.seed


    def _get(self, obj: Any, path: str) -> Any:
        cur = obj
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur

    def _coerce_df(self, value: Any) -> Optional[pd.DataFrame]:
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, (list, tuple)):
            try:
                return pd.DataFrame(value) if value else None
            except Exception:
                return None
        if isinstance(value, dict):
            try:
                return pd.DataFrame([value])
            except Exception:
                return None
        return None



    def collect_df(
        self,
        scenario_id: Optional[str] = None,
        label: Optional[str] = None,                 # kept for API compatibility; unused
        attr_map: Optional[Mapping[str, str]] = None # {name: dotted_path}
    ) -> pd.DataFrame:
        """
        Save each DataFrame specified in attr_map to its own Parquet file:
            <results_root[/experiment_name]>/<name>_df_<scenario_id>.parquet

        - Does NOT add scenario/experiment columns to the data.
        - Returns a summary DataFrame with file paths and row counts.
        """
        if attr_map is None:
            attr_map = {"valuation_metrics": "valuation.valuemetrics"}

        # Determine scenario id for filenames
        sid = scenario_id or self.cfg_scenario_id or "noid"
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid)).strip("_") or "noid"

        saved_rows: List[dict] = []
        for name, dotted in attr_map.items():
            val = self._get(self.env, dotted)
            df = self._coerce_df(val)
            if df is None or df.empty:
                continue

            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "table"
            out_path = self.results_root / f"{safe_name}_df_{safe_sid}.parquet"

            df.to_parquet(out_path, index=False)
            saved_rows.append({
                "name": name,
                "path": str(out_path),
                "rows": len(df)
            })

        return None
