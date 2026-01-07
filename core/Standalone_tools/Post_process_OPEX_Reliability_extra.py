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
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Reliability"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

# NEW: figures go to results/Figures/Experiment
FIG_FOLDER = PROJECT_ROOT / "results" / "Figures" / "OPEX_Reliability"
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
# A4-like landscape strip: two violins side-by-side
# Target size: 200 mm x 80 mm
# -----------------------------

MM_TO_INCH = 1.0 / 25.4
FIGSIZE_IN = (200 * MM_TO_INCH, 80 * MM_TO_INCH)

data1 = [farmA_by_macro[m] for m in macro_names]
data2 = [cmint_by_macro[m] for m in macro_names]

# Explicit, user-editable x tick labels
xtick_labels = [
    "R1",
    "R2",
    "R3",
    "R0",
]

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=FIGSIZE_IN,
    dpi=300,
    gridspec_kw={
        "left": 0.08,
        "right": 0.98,
        "top": 0.90,
        "bottom": 0.28,  # extra room for labels in short figure
        "wspace": 0.30,
    },
)

# -----------------------------
# (a) Farm availability
# -----------------------------
ax1.violinplot(
    data1,
    showmeans=True,
    showmedians=False,
    showextrema=True,
)

ax1.set_ylabel("Farm availability", fontsize=10)
ax1.set_xticks(np.arange(1, len(xtick_labels) + 1))
ax1.set_xticklabels(xtick_labels, fontsize=9)
ax1.tick_params(axis="y", labelsize=8)

ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# -----------------------------
# (b) CM interventions
# -----------------------------
ax2.violinplot(
    data2,
    showmeans=True,
    showmedians=False,
    showextrema=True,
)

ax2.set_ylabel("CM interventions", fontsize=10)
ax2.set_xticks(np.arange(1, len(xtick_labels) + 1))
ax2.set_xticklabels(xtick_labels, fontsize=9)
ax2.tick_params(axis="y", labelsize=8)

ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.savefig(
    FIG_FOLDER / "violin_availability_and_cm_interventions_200x80mm.png",
    dpi=300,
)
plt.close()

# -----------------------------
# Compact scatter: 100 x 50 mm
#   - explicit plot order (plot_order)
#   - independent legend order + labels (legend_order, legend_labels)
#   - controllable dot transparency (DOT_ALPHA)
#   - NO median annotations
# -----------------------------
MM_TO_INCH = 1.0 / 25.4
FIGSIZE_IN = (100 * MM_TO_INCH, 50 * MM_TO_INCH)

DOT_ALPHA = 0.75  # <-- control transparency here (0.0..1.0)

fig, ax = plt.subplots(
    1, 1,
    figsize=FIGSIZE_IN,
    dpi=300,
    gridspec_kw={
        "left": 0.16,
        "right": 0.98,
        "top": 0.92,
        "bottom": 0.32,
    },
)

# Matplotlib default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Canonical mapping: macro_names is already sorted earlier
macro_to_r = {m: f"R{i}" for i, m in enumerate(macro_names)}  # R0, R1, ...
r_to_macro = {r: m for m, r in macro_to_r.items()}
macro_to_color = {m: colors[i % len(colors)] for i, m in enumerate(macro_names)}

# ---- USER CONTROL ----
plot_order = ["R2", "R1", "R0", "R3"]      # controls draw order
legend_order = [ "R0", "R1", "R2", "R3"]    # controls legend order

legend_labels = {
    "R0": "R1",
    "R1": "R2",
    "R2": "R3",
    "R3": "R0",
}
# ----------------------

# Plot points in explicit order; keep a handle per scenario for legend
handles_by_r = {}

for r_id in plot_order:
    macro = r_to_macro.get(r_id)
    if macro is None:
        continue

    dd = df_scatter.loc[df_scatter["macro"] == macro]
    if dd.empty:
        continue

    h = ax.scatter(
        dd["total_interventions"],
        dd["npv_firm"],
        s=18,
        alpha=DOT_ALPHA,
        color=macro_to_color[macro],
        edgecolors="none",
        zorder=10,
    )
    handles_by_r[r_id] = h

ax.set_xlabel("Total interventions", fontsize=8)
ax.set_ylabel("NPV_firm [€]", fontsize=8)

ax.tick_params(axis="both", labelsize=7)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend in explicit order, with explicit labels
legend_handles = []
legend_texts = []
for r_id in legend_order:
    h = handles_by_r.get(r_id)
    if h is None:
        continue
    legend_handles.append(h)
    legend_texts.append(legend_labels.get(r_id, r_id))

ax.legend(
    handles=legend_handles,
    labels=legend_texts,
    loc="upper right",
    fontsize=7,
    frameon=True,
    borderpad=0.3,
    handletextpad=0.4,
    labelspacing=0.3,
)

plt.savefig(
    FIG_FOLDER / "scatter_npv_firm_vs_total_interventions_100x50mm.png",
    dpi=300,
)
plt.close()




print("Saved plot to:", FIG_FOLDER)
