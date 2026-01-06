#!/usr/bin/env python3
"""
Combined NPV Histograms Script
------------------------------

Creates combined histograms of NPV (equity and firm) across all macro scenarios
(Scenario.name), overlaying each macro scenario.

Outputs (for each metric):
1) Filled histogram + optional distribution fit curves (Gaussian or Lognormal)
2) Same figure but WITHOUT fit curves

Enhancements:
- Transparent filled histograms + crisp outlines for readability.
- Optional density normalization (recommended for comparing shapes).
- Fit curves plotted consistently with counts or density.
- Deterministic macro scenarios shown as vertical lines.

Assumptions:
- scenarios.json exists in RESULTS_FOLDER.
- For each scenario_id SID there is a parquet file:
    {TABLE_NAME}_df_{SID}.parquet
  with columns including "npv_equity", "npv_firm".
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# --------------------------------------------------------------------
# 1) Configuration
# --------------------------------------------------------------------
TABLE_NAME = "valuation_metrics"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
# RESULTS_FOLDER still points to results/LTE_Experiment
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Reliability_Uncertainty"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

# NEW: figures go to results/Figures/Experiment
FIG_FOLDER = PROJECT_ROOT / "results" / "Figures" / "OPEX_Reliability_Uncertainty"
FIG_FOLDER.mkdir(parents=True, exist_ok=True)


SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"


def get_macro_name(sc: dict, default: str) -> str:
    overrides = sc.get("overrides") or {}
    return str(
        overrides.get("Scenario.name")
        or sc.get("Scenario.name")
        or sc.get("label")
        or default
    )

def load_table(table: str, sid: str) -> pd.DataFrame:
    p = RESULTS_FOLDER / f"{table}_df_{sid}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

def weighted_farm_A_from_windows(df_win: pd.DataFrame) -> float:
    """
    Returns a single farm availability per SID.
    Weighted by window duration if possible; otherwise mean of farm_A.
    """
    if df_win is None or df_win.empty or "farm_A" not in df_win.columns:
        return np.nan

    d = df_win.copy()
    d["farm_A"] = pd.to_numeric(d["farm_A"], errors="coerce")

    if "window_start" in d.columns and "window_end" in d.columns:
        d["window_start"] = pd.to_datetime(d["window_start"], errors="coerce")
        d["window_end"] = pd.to_datetime(d["window_end"], errors="coerce")
        dt_h = (d["window_end"] - d["window_start"]).dt.total_seconds() / 3600.0
        d["dt_h"] = pd.to_numeric(dt_h, errors="coerce")

        ok = d["farm_A"].notna() & d["dt_h"].notna() & (d["dt_h"] > 0)
        if ok.any():
            return float((d.loc[ok, "farm_A"] * d.loc[ok, "dt_h"]).sum() / d.loc[ok, "dt_h"].sum())

    # fallback
    if d["farm_A"].notna().any():
        return float(d["farm_A"].mean())
    return np.nan

def interventions_from_mode_breakdown(df_mcb: pd.DataFrame) -> tuple[float, float]:
    """
    Returns (total_interventions, cm_interventions) per SID.
    """
    if df_mcb is None or df_mcb.empty:
        return (np.nan, np.nan)

    d = df_mcb.copy()
    if "N_interventions" not in d.columns:
        return (np.nan, np.nan)

    d["N_interventions"] = pd.to_numeric(d["N_interventions"], errors="coerce").fillna(0.0)
    total_int = float(d["N_interventions"].sum())

    if "task_type" in d.columns:
        cm_int = float(d.loc[d["task_type"].astype(str).str.upper() == "CM", "N_interventions"].sum())
    else:
        cm_int = np.nan

    return (total_int, cm_int)

def npv_firm_from_valuation(df_val: pd.DataFrame) -> float:
    if df_val is None or df_val.empty or "npv_firm" not in df_val.columns:
        return np.nan
    s = pd.to_numeric(df_val["npv_firm"], errors="coerce").dropna()
    # Usually valuation_metrics has one row; if multiple, take mean (or sum if that's your meaning)
    return float(s.mean()) if len(s) else np.nan


# -----------------------------
# Load scenarios & group by macro name
# -----------------------------
with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
    scenarios = json.load(f)
if isinstance(scenarios, dict) and "scenarios" in scenarios:
    scenarios = scenarios["scenarios"]

macro_to_sids = {}
for sc in scenarios:
    sid = str(sc.get("scenario_id", "")).strip()
    if not sid:
        continue
    macro = get_macro_name(sc, default=sid)
    macro_to_sids.setdefault(macro, []).append(sid)

macro_names = list(macro_to_sids.keys())
macro_names.sort()


# -----------------------------
# Build per-SID metrics and bucket by macro scenario
# -----------------------------
farmA_by_macro = {m: [] for m in macro_names}
cmint_by_macro = {m: [] for m in macro_names}
scatter_rows = []  # for plot 4

for macro, sids in macro_to_sids.items():
    for sid in sids:
        df_win = load_table("opex_windows", sid)
        df_mcb = load_table("opex_mode_cost_breakdown", sid)
        df_val = load_table("valuation_metrics", sid)

        farm_A = weighted_farm_A_from_windows(df_win)
        total_int, cm_int = interventions_from_mode_breakdown(df_mcb)
        npv_firm = npv_firm_from_valuation(df_val)

        if np.isfinite(farm_A):
            farmA_by_macro[macro].append(farm_A)

        if np.isfinite(cm_int):
            cmint_by_macro[macro].append(cm_int)

        if np.isfinite(npv_firm) and np.isfinite(total_int):
            scatter_rows.append({"macro": macro, "sid": sid, "npv_firm": npv_firm, "total_interventions": total_int})

df_scatter = pd.DataFrame(scatter_rows)


# -----------------------------
# Plot 1: Violin farm availability
# -----------------------------
data1 = [farmA_by_macro[m] for m in macro_names]
plt.figure(figsize=(11, 5), dpi=200)
ax = plt.gca()
parts = ax.violinplot(data1, showmeans=True, showmedians=False, showextrema=True)
ax.set_title("Farm availability by macro scenario (violin)")
ax.set_ylabel("Farm availability (farm_A)")
ax.set_xticks(np.arange(1, len(macro_names) + 1))
ax.set_xticklabels(macro_names, rotation=20, ha="right")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_FOLDER / "violin_farm_availability.png")
plt.close()


# -----------------------------
# Plot 2: Violin CM interventions
# -----------------------------
data2 = [cmint_by_macro[m] for m in macro_names]
plt.figure(figsize=(11, 5), dpi=200)
ax = plt.gca()
parts = ax.violinplot(data2, showmeans=True, showmedians=False, showextrema=True)
ax.set_title("Corrective maintenance interventions by macro scenario (violin)")
ax.set_ylabel("CM interventions (sum N_interventions where task_type=='CM')")
ax.set_xticks(np.arange(1, len(macro_names) + 1))
ax.set_xticklabels(macro_names, rotation=20, ha="right")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_FOLDER / "violin_cm_interventions.png")
plt.close()


# -----------------------------
# Plot 4: Scatter NPV_firm vs total interventions (colored by macro)
# -----------------------------
plt.figure(figsize=(10, 6), dpi=200)
ax = plt.gca()

# use matplotlib default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
macro_to_color = {m: colors[i % len(colors)] for i, m in enumerate(macro_names)}

for macro in macro_names:
    dd = df_scatter.loc[df_scatter["macro"] == macro]
    if dd.empty:
        continue
    ax.scatter(dd["total_interventions"], dd["npv_firm"], label=macro, alpha=0.75)

ax.set_title("NPV (firm) vs total interventions")
ax.set_xlabel("Total interventions (sum N_interventions)")
ax.set_ylabel("NPV_firm [currency]")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=False, fontsize=9, ncol=2 if len(macro_names) > 6 else 1)
plt.tight_layout()
plt.savefig(FIG_FOLDER / "scatter_npv_firm_vs_total_interventions.png")
plt.close()

print("Saved plots to:", FIG_FOLDER)
