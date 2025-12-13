# -------------- Result Collector ---------------------#
from __future__ import annotations
from pathlib import Path

import pandas as pd
import itertools, time, hashlib, gc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, List
from datetime import datetime
import re  # ensure this is imported
import logging



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
        self.logger = getattr(env, "logger", logging.getLogger("winpact.results"))

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
        self.logger.info(
            "[collector:init] results_root=%s out_path=%s scenario_id=%s",
            self.results_root, self.out_path, self.cfg_scenario_id
        )


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
        label: Optional[str] = None,
        attr_map: Optional[Mapping[str, str]] = None
    ) -> Optional[pd.DataFrame]:

        if attr_map is None:
            attr_map = {"valuation_metrics": "valuation.valuemetrics"}

        sid = scenario_id or self.cfg_scenario_id or "noid"
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid)).strip("_") or "noid"

        self.logger.info(
            "ResultsCollector started | scenario_id=%s | results_root=%s",
            safe_sid,
            self.results_root,
        )

        wrote_anything = False

        for name, dotted in attr_map.items():
            self.logger.info("Collecting result '%s' from '%s'", name, dotted)

            val = self._get(self.env, dotted)

            if val is None:
                self.logger.warning(
                    "Result '%s' resolved to None (path='%s') — skipping",
                    name,
                    dotted,
                )
                continue

            df = self._coerce_df(val)

            if df is None:
                self.logger.warning(
                    "Result '%s' could not be coerced to DataFrame (type=%s) — skipping",
                    name,
                    type(val),
                )
                continue

            if df.empty:
                self.logger.warning(
                    "Result '%s' DataFrame is EMPTY — skipping write",
                    name,
                )
                continue

            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "table"
            out_path = self.results_root / f"{safe_name}_df_{safe_sid}.parquet"

            self.logger.info(
                "Writing %d rows to %s",
                len(df),
                out_path,
            )

            df.to_parquet(out_path, index=False)
            wrote_anything = True

        if not wrote_anything:
            self.logger.warning(
                "ResultsCollector finished — NO result files were written for scenario %s",
                safe_sid,
            )
        else:
            self.logger.info(
                "ResultsCollector finished successfully for scenario %s",
                safe_sid,
            )

        return None

