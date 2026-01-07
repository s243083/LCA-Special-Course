#!/usr/bin/env python3
"""
Post-process OPEX_Efficiency experiment
--------------------------------------

Figure 1 (two subplots):
  (a) Violin plot: Farm availability over scenarios (P0..P4)
  (b) Grouped violins: total downtime hours split into
      - weather waiting (weather_h)
      - logistics waiting (logistics_h)
      - repair time (repair_h)
      grouped by scenario.

Inputs (per scenario_id SID):
  - opex_windows_df_{SID}.parquet
  - opex_component_downtime_breakdown_df_{SID}.parquet
  - scenarios.json (in RESULTS_FOLDER)

Output:
  - results/Figures/OPEX_Efficiency/violin_availability_and_downtime_split_200x80mm.png
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# 1) Configuration
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # examples/HPC -> examples -> repo root
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Efficiency"
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

FIG_FOLDER = PROJECT_ROOT / "results" / "Figures" / "OPEX_Efficiency"
FIG_FOLDER.mkdir(parents=True, exist_ok=True)

SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"


# --------------------------------------------------------------------
# 2) Helpers
# --------------------------------------------------------------------
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

    if d["farm_A"].notna().any():
        return float(d["farm_A"].mean())

    return np.nan

def totals_from_downtime_breakdown(df_db: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns (total_logistics_h, total_weather_h, total_repair_h) per SID.

    df_db is expected to be the opex_component_downtime_breakdown DF:
      columns include: logistics_h, weather_h, repair_h
      may include multiple windows; multiple components.

    We:
      - coerce to numeric
      - sum across components AND windows (simple sum)

    Note: your *_h values are already farm-level hours per component (multiplied by n_turbines),
    so summing over components yields farm totals.
    """
    if df_db is None or df_db.empty:
        return (np.nan, np.nan, np.nan)

    d = df_db.copy()
    for c in ("logistics_h", "weather_h", "repair_h"):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    tot_L = float(np.nansum(d["logistics_h"].values))
    tot_W = float(np.nansum(d["weather_h"].values))
    tot_R = float(np.nansum(d["repair_h"].values))
    return (tot_L, tot_W, tot_R)


# --------------------------------------------------------------------
# 3) Load scenarios & group by macro scenario name
# --------------------------------------------------------------------
with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
    scenarios = json.load(f)
if isinstance(scenarios, dict) and "scenarios" in scenarios:
    scenarios = scenarios["scenarios"]

macro_to_sids: Dict[str, List[str]] = {}
for sc in scenarios:
    sid = str(sc.get("scenario_id", "")).strip()
    if not sid:
        continue
    macro = get_macro_name(sc, default=sid)
    macro_to_sids.setdefault(macro, []).append(sid)

macro_names = sorted(macro_to_sids.keys())


# --------------------------------------------------------------------
# 4) Build per-SID metrics and bucket by macro scenario
# --------------------------------------------------------------------
farmA_by_macro = {m: [] for m in macro_names}
L_by_macro = {m: [] for m in macro_names}  # logistics waiting hours
W_by_macro = {m: [] for m in macro_names}  # weather waiting hours
R_by_macro = {m: [] for m in macro_names}  # repair hours

for macro, sids in macro_to_sids.items():
    for sid in sids:
        df_win = load_table("opex_windows", sid)
        df_db  = load_table("opex_component_downtime_breakdown", sid)

        farm_A = weighted_farm_A_from_windows(df_win)
        tot_L, tot_W, tot_R = totals_from_downtime_breakdown(df_db)

        if np.isfinite(farm_A):
            farmA_by_macro[macro].append(farm_A)
        if np.isfinite(tot_L):
            L_by_macro[macro].append(tot_L)
        if np.isfinite(tot_W):
            W_by_macro[macro].append(tot_W)
        if np.isfinite(tot_R):
            R_by_macro[macro].append(tot_R)


# --------------------------------------------------------------------
# 5) Plot: 2-panel figure (A4-like strip)
# --------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
FIGSIZE_IN = (200 * MM_TO_INCH, 80 * MM_TO_INCH)

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=FIGSIZE_IN,
    dpi=300,
    gridspec_kw={
        "left": 0.07,
        "right": 0.99,
        "top": 0.90,
        "bottom": 0.28,
        "wspace": 0.25,
    },
)

# -----------------------------
# (a) Farm availability violin
# -----------------------------
data_A = [farmA_by_macro[m] for m in macro_names]
pos_A = np.arange(1, len(macro_names) + 1)

ax1.violinplot(
    data_A,
    positions=pos_A,
    showmeans=True,
    showmedians=False,
    showextrema=True,
)
ax1.set_ylabel("Farm availability", fontsize=10)
ax1.set_xticks(pos_A)
ax1.set_xticklabels(macro_names, fontsize=8, rotation=20, ha="right")
ax1.tick_params(axis="y", labelsize=8)
ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ---------------------------------------------
# (b) Grouped violins: downtime split (hours)
# ---------------------------------------------
# For each scenario group at x=i, plot 3 violins:
#   logistics at i - dx, weather at i, repair at i + dx
dx = 0.22
positions_L = pos_A - dx
positions_W = pos_A
positions_R = pos_A + dx

data_L = [L_by_macro[m] for m in macro_names]
data_W = [W_by_macro[m] for m in macro_names]
data_R = [R_by_macro[m] for m in macro_names]

vL = ax2.violinplot(data_L, positions=positions_L, widths=0.18, showmeans=True, showmedians=False, showextrema=True)
vW = ax2.violinplot(data_W, positions=positions_W, widths=0.18, showmeans=True, showmedians=False, showextrema=True)
vR = ax2.violinplot(data_R, positions=positions_R, widths=0.18, showmeans=True, showmedians=False, showextrema=True)

# Make the three categories distinguishable without manually setting colors:
# - Different edge styles
for body in vL.get("bodies", []):
    body.set_alpha(0.35)
    body.set_linewidth(0.8)
for body in vW.get("bodies", []):
    body.set_alpha(0.35)
    body.set_linewidth(0.8)
for body in vR.get("bodies", []):
    body.set_alpha(0.35)
    body.set_linewidth(0.8)

ax2.set_ylabel("Total downtime [h]\n(farm-level, summed over components)", fontsize=10)
ax2.set_xticks(pos_A)
ax2.set_xticklabels(macro_names, fontsize=8, rotation=20, ha="right")
ax2.tick_params(axis="y", labelsize=8)
ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Simple legend (text-only, no colors assumed)
ax2.text(0.02, 0.98, "Grouped violins:\nL = Logistics\nW = Weather\nR = Repair",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8)

# --------------------------------------------------------------------
# 6) Save
# --------------------------------------------------------------------
out_path = FIG_FOLDER / "violin_availability_and_downtime_split_200x80mm.png"
plt.savefig(out_path, dpi=300)
plt.close()

print("Saved plot to:", out_path)
