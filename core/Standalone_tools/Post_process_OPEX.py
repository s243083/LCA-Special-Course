#!/usr/bin/env python3
"""
Create histograms of total project OPEX and annualised OPEX
for each macro scenario (Scenario.name) and combined across scenarios.

Definitions:
- "Macro scenario" = Scenario.name (e.g. "Best-practice O&M", "Harsh env + immature O&M", ...)
- Each macro scenario has multiple replicates; each replicate is a scenario entry
  with its own scenario_id and parquet file.

Assumptions:
- scenarios.json exists in the RESULTS_FOLDER and contains a list of scenarios
  with at least:
      - "scenario_id"
      - "overrides" dict that includes "Scenario.name" (preferred)
- For each scenario_id SID there is a parquet file named:
      {TABLE_NAME}_df_{SID}.parquet
- Each parquet file contains OPEX records with at least:
      - "timestamp" (datetime-like)
      - "OM_payment" (numeric, usually negative for outflow)

For each macro scenario S:
- We compute per replicate:
    - total project OPEX (sum of OM_payment, converted to positive EUR)
    - annualised OPEX (total / years over which timestamps span)

Outputs (in RESULTS_FOLDER):
- Per macro scenario:
    - histogram_total_opex_<ScenarioNameSanitized>.png
    - histogram_annual_opex_<ScenarioNameSanitized>.png
- Combined (all macro scenarios overlaid, colored with legend):
    - histogram_total_opex_all_scenarios.png
    - histogram_annual_opex_all_scenarios.png
"""

from pathlib import Path
import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1) Configuration
# --------------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Uncertainty"

TABLE_NAME = "opex"  # prefix of parquet files: opex_df_<scenario_id>.parquet

RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"
assert SCENARIOS_PATH.exists(), f"scenarios.json not found at: {SCENARIOS_PATH}"

# --------------------------------------------------------------------
# 2) Load scenarios.json
# --------------------------------------------------------------------
with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

# Might be {"scenarios": [...]} or just [...]
if isinstance(scenarios, dict) and "scenarios" in scenarios:
    scenarios = scenarios["scenarios"]

assert isinstance(scenarios, list), "Expected a list of scenarios in scenarios.json"
print(f"Loaded {len(scenarios)} scenario entries from {SCENARIOS_PATH}")

# --------------------------------------------------------------------
# 3) Helpers
# --------------------------------------------------------------------
def get_macro_name(sc: Dict[str, Any], default: str) -> str:
    """
    Extract the macro scenario name from a scenario entry.

    Priority:
    1. sc["overrides"]["Scenario.name"]  (if present)
    2. sc.get("Scenario.name")
    3. sc.get("label")
    4. fallback: default (e.g. scenario_id)
    """
    overrides = sc.get("overrides") or {}
    name = overrides.get("Scenario.name") or sc.get("Scenario.name") \
           or sc.get("label") or default
    return str(name)


def _sanitize_name(name: str) -> str:
    """Make a string safe for filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _maybe_millions_single(ax, values: np.ndarray, label: str) -> np.ndarray:
    """Scale a single array to millions if needed, and set x-axis label."""
    if values.size == 0:
        ax.set_xlabel(label)
        return values
    vmax = float(np.max(values))
    if vmax >= 1e6:
        ax.set_xlabel(label + " [million]")
        return values / 1e6
    else:
        ax.set_xlabel(label)
        return values


def _maybe_millions_multi(ax, arrays: List[np.ndarray], label: str) -> List[np.ndarray]:
    """Scale multiple arrays jointly based on global max."""
    nonempty = [arr for arr in arrays if arr.size > 0]
    if not nonempty:
        ax.set_xlabel(label)
        return arrays
    all_vals = np.concatenate(nonempty)
    vmax = float(np.max(all_vals))
    if vmax >= 1e6:
        ax.set_xlabel(label + " [million]")
        return [arr / 1e6 for arr in arrays]
    else:
        ax.set_xlabel(label)
        return arrays


def _total_and_annual_opex(df: pd.DataFrame) -> tuple[float, float]:
    """
    Given an OPEX_records DataFrame with columns:
        - 'timestamp'
        - 'OM_payment' (usually negative for cash outflow)

    Returns:
        (total_opex_eur_positive, annualised_opex_eur_per_year_positive)
    """
    if "OM_payment" not in df.columns:
        raise ValueError("Expected 'OM_payment' column in OPEX parquet.")

    df = df.copy()
    df["OM_payment"] = pd.to_numeric(df["OM_payment"], errors="coerce").fillna(0.0)

    total = float(df["OM_payment"].sum())

    # Interpret OM_payment as outflow; ensure positive magnitude
    total_opex = -total if total <= 0.0 else total

    # Annualise based on timestamp span (if timestamps exist)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = ts.dropna()
        if len(ts) >= 2:
            dt_years = (ts.max() - ts.min()).days / 365.25
            years = dt_years if dt_years > 0 else 1.0
        else:
            years = 1.0
    else:
        years = 1.0

    annual_opex = total_opex / years
    return total_opex, annual_opex

# --------------------------------------------------------------------
# 4) Group scenario_ids by macro scenario name
# --------------------------------------------------------------------
macro_to_entries: Dict[str, Dict[str, Any]] = {}  # macro_name -> sid -> scenario_entry

for sc in scenarios:
    sid = str(sc.get("scenario_id", "")).strip()
    if not sid:
        continue
    macro_name = get_macro_name(sc, default=sid)
    # deduplicate by scenario_id within each macro scenario
    macro_to_entries.setdefault(macro_name, {})
    macro_to_entries[macro_name][sid] = sc

print("\nMacro scenarios found (unique scenario_ids):")
for macro, entries_by_sid in macro_to_entries.items():
    print(f"  {macro}: {len(entries_by_sid)} replicate(s)")

if not macro_to_entries:
    raise RuntimeError("No macro scenarios could be identified; check Scenario.name in scenarios.json")

# --------------------------------------------------------------------
# 5) Load parquet data and aggregate per macro scenario
# --------------------------------------------------------------------
macro_data: Dict[str, Dict[str, Any]] = {}
missing_files: List[Any] = []

for macro_name, entries_by_sid in macro_to_entries.items():
    total_opex_list: List[float] = []    # one value per replicate
    annual_opex_list: List[float] = []   # one value per replicate

    for sid, sc in entries_by_sid.items():
        parquet_path = RESULTS_FOLDER / f"{TABLE_NAME}_df_{sid}.parquet"
        if not parquet_path.exists():
            missing_files.append((macro_name, sid, str(parquet_path), "Missing file"))
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            missing_files.append((macro_name, sid, str(parquet_path), f"Read error: {e}"))
            continue

        try:
            total_opex, annual_opex = _total_and_annual_opex(df)
        except Exception as e:
            print(
                f"WARNING: macro '{macro_name}', scenario_id {sid} could not compute OPEX: {e}"
            )
            continue

        total_opex_list.append(total_opex)
        annual_opex_list.append(annual_opex)

    total_opex_arr = np.array(total_opex_list, dtype=float)
    annual_opex_arr = np.array(annual_opex_list, dtype=float)

    if total_opex_arr.size or annual_opex_arr.size:
        macro_data[macro_name] = {
            "total_opex": total_opex_arr,
            "annual_opex": annual_opex_arr,
        }

        print(
            f"\nMacro scenario '{macro_name}': "
            f"{total_opex_arr.size} replicate total-OPEX values, "
            f"{annual_opex_arr.size} annualised OPEX values"
        )

print(f"\nCollected OPEX data for {len(macro_data)} macro scenario(s).")
if missing_files:
    print("\nFiles not loaded:")
    for m in missing_files:
        print(f" - macro='{m[0]}', scenario_id={m[1]}, path={m[2]}, reason={m[3]}")

if not macro_data:
    raise RuntimeError("No usable macro scenario OPEX data; cannot plot.")

# --------------------------------------------------------------------
# 6) Per-macro-scenario histograms (with Gaussian curve fit)
# --------------------------------------------------------------------
print("\nCreating per-macro-scenario OPEX histograms...")

for macro_name, data in macro_data.items():
    total_opex_arr = data["total_opex"]
    annual_opex_arr = data["annual_opex"]

    safe_name = _sanitize_name(macro_name)

    # 6a) Histogram of total project OPEX across replicates
    if total_opex_arr.size > 0:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        vals = _maybe_millions_single(ax, total_opex_arr, "Total project OPEX [EUR]")

        bins = min(30, max(1, vals.size))

        # histogram
        n, bin_edges, _ = ax.hist(
            vals,
            bins=bins,
            edgecolor="C0",
            alpha=0.6,
            linewidth=1.2,
            label="Histogram",
        )

        # Gaussian fit
        if vals.size > 1:
            mu = float(vals.mean())
            sigma = float(vals.std(ddof=1))
            if sigma > 0:
                x = np.linspace(bin_edges[0], bin_edges[-1], 200)
                bin_width = bin_edges[1] - bin_edges[0]
                y = (
                    1.0
                    / (sigma * np.sqrt(2.0 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    * vals.size
                    * bin_width
                )
                ax.plot(x, y, color="C0", linewidth=1.8, label="Gaussian fit")

        ax.set_ylabel("Frequency")
        ax.set_title(f"Total project OPEX\nScenario: {macro_name}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        out_path = RESULTS_FOLDER / f"histogram_total_opex_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved total OPEX histogram for '{macro_name}' to: {out_path}")

    # 6b) Histogram of annualised OPEX across replicates
    if annual_opex_arr.size > 0:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        vals = _maybe_millions_single(ax, annual_opex_arr, "Annualised OPEX [EUR/year]")

        bins = min(30, max(1, vals.size))

        # histogram
        n, bin_edges, _ = ax.hist(
            vals,
            bins=bins,
            edgecolor="C1",
            alpha=0.6,
            linewidth=1.2,
            label="Histogram",
        )

        # Gaussian fit
        if vals.size > 1:
            mu = float(vals.mean())
            sigma = float(vals.std(ddof=1))
            if sigma > 0:
                x = np.linspace(bin_edges[0], bin_edges[-1], 200)
                bin_width = bin_edges[1] - bin_edges[0]
                y = (
                    1.0
                    / (sigma * np.sqrt(2.0 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    * vals.size
                    * bin_width
                )
                ax.plot(x, y, color="C1", linewidth=1.8, label="Gaussian fit")

        ax.set_ylabel("Frequency")
        ax.set_title(f"Annualised OPEX\nScenario: {macro_name}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        out_path = RESULTS_FOLDER / f"histogram_annual_opex_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved annual OPEX histogram for '{macro_name}' to: {out_path}")

# --------------------------------------------------------------------
# 7) Combined histograms (all macro scenarios overlaid, with fits)
# --------------------------------------------------------------------
print("\nCreating combined OPEX histograms for all macro scenarios...")

macro_names = list(macro_data.keys())
total_arrays = [macro_data[m]["total_opex"] for m in macro_names]
annual_arrays = [macro_data[m]["annual_opex"] for m in macro_names]

# use matplotlib default color cycle
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 7a) Combined total project OPEX histogram
nonempty_total = [a for a in total_arrays if a.size > 0]
if nonempty_total:
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    scaled_total_arrays = _maybe_millions_multi(ax, total_arrays, "Total project OPEX [EUR]")

    nonempty_scaled = [a for a in scaled_total_arrays if a.size > 0]
    all_vals = np.concatenate(nonempty_scaled)
    bins = min(30, max(5, len(all_vals)))
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins)

    for i, (macro_name, arr) in enumerate(zip(macro_names, scaled_total_arrays)):
        if arr.size == 0:
            continue

        color = color_cycle[i % len(color_cycle)]

        # histogram (step)
        n, _, _ = ax.hist(
            arr,
            bins=bin_edges,
            histtype="step",
            linewidth=1.5,
            alpha=0.9,
            label=macro_name,
            color=color,
        )

        # Gaussian fit
        if arr.size > 1:
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))
            if sigma > 0:
                x = np.linspace(bin_edges[0], bin_edges[-1], 200)
                bin_width = bin_edges[1] - bin_edges[0]
                y = (
                    1.0
                    / (sigma * np.sqrt(2.0 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    * arr.size
                    * bin_width
                )
                ax.plot(x, y, color=color, linestyle="-", linewidth=1.5)

    ax.set_ylabel("Frequency")
    ax.set_title("Total project OPEX\n(all macro scenarios)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    out_path = RESULTS_FOLDER / "histogram_total_opex_all_scenarios.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined total OPEX histogram to: {out_path}")
else:
    print("No total project OPEX data available; skipping combined total OPEX plot.")

# 7b) Combined annualised OPEX histogram
nonempty_ann = [a for a in annual_arrays if a.size > 0]
if nonempty_ann:
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    scaled_ann_arrays = _maybe_millions_multi(ax, annual_arrays, "Annualised OPEX [EUR/year]")

    nonempty_scaled_ann = [a for a in scaled_ann_arrays if a.size > 0]
    all_vals_ann = np.concatenate(nonempty_scaled_ann)
    bins_ann = min(30, max(5, len(all_vals_ann)))
    bin_edges_ann = np.linspace(all_vals_ann.min(), all_vals_ann.max(), bins_ann)

    for i, (macro_name, arr) in enumerate(zip(macro_names, scaled_ann_arrays)):
        if arr.size == 0:
            continue

        color = color_cycle[i % len(color_cycle)]

        # histogram (step)
        n, _, _ = ax.hist(
            arr,
            bins=bin_edges_ann,
            histtype="step",
            linewidth=1.5,
            alpha=0.9,
            label=macro_name,
            color=color,
        )

        # Gaussian fit
        if arr.size > 1:
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))
            if sigma > 0:
                x = np.linspace(bin_edges_ann[0], bin_edges_ann[-1], 200)
                bin_width = bin_edges_ann[1] - bin_edges_ann[0]
                y = (
                    1.0
                    / (sigma * np.sqrt(2.0 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    * arr.size
                    * bin_width
                )
                ax.plot(x, y, color=color, linestyle="-", linewidth=1.5)

    ax.set_ylabel("Frequency")
    ax.set_title("Annualised OPEX\n(all macro scenarios)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    out_path = RESULTS_FOLDER / "histogram_annual_opex_all_scenarios.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined annual OPEX histogram to: {out_path}")
else:
    print("No annualised OPEX data available; skipping combined annual OPEX plot.")

print("\nDone.")







################################NPV Histograms Script################################

#!/usr/bin/env python3
"""
Create histograms of NPV (equity and firm) for each macro scenario (Scenario.name)
and combined across scenarios.

Curve fitting options:
- FIT_DISTRIBUTION = "gaussian"  -> normal fit in linear space
- FIT_DISTRIBUTION = "lognormal" -> normal fit in log space, only on positive samples

Deterministic handling:
- If all samples in a macro scenario are (near) identical, no histogram/fit is drawn.
  Instead a single vertical line is shown (per-macro and combined plots).

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
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"
assert SCENARIOS_PATH.exists(), f"scenarios.json not found at: {SCENARIOS_PATH}"

# Choose: "gaussian" or "lognormal"
FIT_DISTRIBUTION = "lognormal"

# Deterministic detection tolerance
DET_ATOL = 1e-12
DET_RTOL = 1e-10


# --------------------------------------------------------------------
# 2) Load scenarios.json
# --------------------------------------------------------------------
with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

if isinstance(scenarios, dict) and "scenarios" in scenarios:
    scenarios = scenarios["scenarios"]

assert isinstance(scenarios, list), "Expected a list of scenarios in scenarios.json"
print(f"Loaded {len(scenarios)} scenario entries from {SCENARIOS_PATH}")


# --------------------------------------------------------------------
# 3) Helpers
# --------------------------------------------------------------------
def get_macro_name(sc: Dict[str, Any], default: str) -> str:
    overrides = sc.get("overrides") or {}
    name = (
        overrides.get("Scenario.name")
        or sc.get("Scenario.name")
        or sc.get("label")
        or default
    )
    return str(name)


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _is_deterministic(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    v0 = float(values[0])
    return bool(np.allclose(values, v0, rtol=DET_RTOL, atol=DET_ATOL))


def _scale_factor_for(values_list: List[np.ndarray], extra_points: Optional[List[float]] = None) -> float:
    maxima: List[float] = []
    for arr in values_list:
        if arr.size > 0:
            maxima.append(float(np.nanmax(arr)))
    if extra_points:
        maxima.extend([float(v) for v in extra_points if np.isfinite(v)])
    if not maxima:
        return 1.0
    vmax = float(np.nanmax(np.array(maxima, dtype=float)))
    return 1e6 if vmax >= 1e6 else 1.0


def _set_x_label(ax, base_label: str, scale: float) -> None:
    ax.set_xlabel(base_label + (" [million]" if scale == 1e6 else ""))


def _gaussian_fit_curve(x: np.ndarray, mu: float, sigma: float, n_samples: int, bin_width: float) -> np.ndarray:
    """Scaled Gaussian curve to match histogram counts."""
    return (
        1.0
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        * n_samples
        * bin_width
    )


def _lognormal_fit_curve(
    x: np.ndarray, mu_log: float, sigma_log: float, n_samples: int, bin_width: float
) -> np.ndarray:
    """Scaled lognormal curve to match histogram counts. Assumes x > 0."""
    return (
        1.0
        / (x * sigma_log * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((np.log(x) - mu_log) / sigma_log) ** 2)
        * n_samples
        * bin_width
    )


def _plot_fit_curve(
    ax,
    values: np.ndarray,
    bin_edges: np.ndarray,
    color: str,
    label: str,
    fit_distribution: str,
) -> None:
    """
    Plot either Gaussian or Lognormal fit (scaled to histogram counts).

    Notes:
    - Gaussian uses all samples.
    - Lognormal uses only strictly positive samples; if none, no fit is drawn.
    """
    if values.size <= 1:
        return

    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.0
    if bin_width <= 0:
        return

    dist = fit_distribution.strip().lower()

    if dist == "gaussian":
        mu = float(values.mean())
        sigma = float(values.std(ddof=1))
        if sigma <= 0:
            return
        x = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 300)
        y = _gaussian_fit_curve(x, mu, sigma, int(values.size), bin_width)
        ax.plot(x, y, color=color, linewidth=1.8, label=label)
        return

    if dist == "lognormal":
        pos = values[values > 0]
        if pos.size <= 1:
            return
        logv = np.log(pos)
        mu_log = float(logv.mean())
        sigma_log = float(logv.std(ddof=1))
        if sigma_log <= 0:
            return
        x = np.linspace(float(pos.min()), float(pos.max()), 300)
        y = _lognormal_fit_curve(x, mu_log, sigma_log, int(pos.size), bin_width)
        ax.plot(x, y, color=color, linewidth=1.8, label=label)
        return

    raise ValueError(f"Unknown FIT_DISTRIBUTION='{fit_distribution}'. Use 'gaussian' or 'lognormal'.")


def _split_det_nondet(
    mnames: List[str], arrays: List[np.ndarray]
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, float]]]:
    nondet: List[Tuple[str, np.ndarray]] = []
    det: List[Tuple[str, float]] = []
    for name, arr in zip(mnames, arrays):
        if arr.size == 0:
            continue
        if _is_deterministic(arr):
            det.append((name, float(arr[0])))
        else:
            nondet.append((name, arr))
    return nondet, det


# --------------------------------------------------------------------
# 4) Group scenario_ids by macro scenario name
# --------------------------------------------------------------------
macro_to_entries: Dict[str, Dict[str, Any]] = {}  # macro_name -> sid -> scenario_entry

for sc in scenarios:
    sid = str(sc.get("scenario_id", "")).strip()
    if not sid:
        continue
    macro_name = get_macro_name(sc, default=sid)
    macro_to_entries.setdefault(macro_name, {})
    macro_to_entries[macro_name][sid] = sc

print("\nMacro scenarios found (unique scenario_ids):")
for macro, entries_by_sid in macro_to_entries.items():
    print(f"  {macro}: {len(entries_by_sid)} replicate(s)")

if not macro_to_entries:
    raise RuntimeError("No macro scenarios could be identified; check Scenario.name in scenarios.json")


# --------------------------------------------------------------------
# 5) Load valuation_metrics and aggregate per macro scenario
# --------------------------------------------------------------------
macro_data: Dict[str, Dict[str, Any]] = {}
missing_files: List[Any] = []

for macro_name, entries_by_sid in macro_to_entries.items():
    npv_equity_list: List[float] = []
    npv_firm_list: List[float] = []

    for sid, sc in entries_by_sid.items():
        parquet_path = RESULTS_FOLDER / f"{TABLE_NAME}_df_{sid}.parquet"
        if not parquet_path.exists():
            missing_files.append((macro_name, sid, str(parquet_path), "Missing file"))
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            missing_files.append((macro_name, sid, str(parquet_path), f"Read error: {e}"))
            continue

        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"WARNING: macro '{macro_name}', scenario_id {sid} has empty valuation_metrics; skipping.")
            continue

        if "npv_equity" in df.columns:
            vals = pd.to_numeric(df["npv_equity"], errors="coerce").dropna()
            npv_equity_list.extend(float(v) for v in vals)
        else:
            print(f"WARNING: macro '{macro_name}', scenario_id {sid} has no 'npv_equity' column.")

        if "npv_firm" in df.columns:
            vals_f = pd.to_numeric(df["npv_firm"], errors="coerce").dropna()
            npv_firm_list.extend(float(v) for v in vals_f)
        else:
            print(f"WARNING: macro '{macro_name}', scenario_id {sid} has no 'npv_firm' column.")

    npv_equity_arr = np.array(npv_equity_list, dtype=float)
    npv_firm_arr = np.array(npv_firm_list, dtype=float)

    if npv_equity_arr.size or npv_firm_arr.size:
        macro_data[macro_name] = {"npv_equity": npv_equity_arr, "npv_firm": npv_firm_arr}
        print(
            f"\nMacro scenario '{macro_name}': "
            f"{npv_equity_arr.size} NPV_equity sample(s), "
            f"{npv_firm_arr.size} NPV_firm sample(s)"
        )

print(f"\nCollected valuation metrics for {len(macro_data)} macro scenario(s).")
if missing_files:
    print("\nFiles not loaded:")
    for m in missing_files:
        print(f" - macro='{m[0]}', scenario_id={m[1]}, path={m[2]}, reason={m[3]}")

if not macro_data:
    raise RuntimeError("No usable macro scenario valuation data; cannot plot.")


# --------------------------------------------------------------------
# 6) Per-macro plots
# --------------------------------------------------------------------
print("\nCreating per-macro-scenario NPV plots...")

for macro_name, data in macro_data.items():
    safe_name = _sanitize_name(macro_name)

    # ---- equity
    npv_equity_arr = data["npv_equity"]
    if npv_equity_arr.size > 0:
        det = _is_deterministic(npv_equity_arr)
        scale = _scale_factor_for([npv_equity_arr])
        vals = npv_equity_arr / scale

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        _set_x_label(ax, "NPV (Equity) [currency]", scale)
        ax.set_ylabel("Frequency")
        ax.set_title(f"NPV (Equity)\nScenario: {macro_name}")
        ax.grid(True, linestyle="--", alpha=0.3)

        if det:
            ax.axvline(float(vals[0]), linewidth=2.0, linestyle="-", label="Deterministic value")
            ax.legend()
        else:
            bins = min(30, max(1, vals.size))
            n, bin_edges, _ = ax.hist(
                vals, bins=bins, edgecolor="C0", alpha=0.6, linewidth=1.2, label="Histogram"
            )
            _plot_fit_curve(
                ax=ax,
                values=vals,
                bin_edges=bin_edges,
                color="C0",
                label="Gaussian fit" if FIT_DISTRIBUTION.lower() == "gaussian" else "Lognormal fit",
                fit_distribution=FIT_DISTRIBUTION,
            )
            ax.legend()

        out_path = RESULTS_FOLDER / f"histogram_npv_equity_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved NPV (Equity) plot for '{macro_name}' to: {out_path}")

    # ---- firm
    npv_firm_arr = data["npv_firm"]
    if npv_firm_arr.size > 0:
        det = _is_deterministic(npv_firm_arr)
        scale = _scale_factor_for([npv_firm_arr])
        vals = npv_firm_arr / scale

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        _set_x_label(ax, "NPV (Firm) [currency]", scale)
        ax.set_ylabel("Frequency")
        ax.set_title(f"NPV (Firm)\nScenario: {macro_name}")
        ax.grid(True, linestyle="--", alpha=0.3)

        if det:
            ax.axvline(float(vals[0]), linewidth=2.0, linestyle="-", label="Deterministic value")
            ax.legend()
        else:
            bins = min(30, max(1, vals.size))
            n, bin_edges, _ = ax.hist(
                vals, bins=bins, edgecolor="C1", alpha=0.6, linewidth=1.2, label="Histogram"
            )
            _plot_fit_curve(
                ax=ax,
                values=vals,
                bin_edges=bin_edges,
                color="C1",
                label="Gaussian fit" if FIT_DISTRIBUTION.lower() == "gaussian" else "Lognormal fit",
                fit_distribution=FIT_DISTRIBUTION,
            )
            ax.legend()

        out_path = RESULTS_FOLDER / f"histogram_npv_firm_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved NPV (Firm) plot for '{macro_name}' to: {out_path}")


# --------------------------------------------------------------------
# 7) Combined plots
# --------------------------------------------------------------------
print("\nCreating combined NPV plots for all macro scenarios...")

macro_names = list(macro_data.keys())
equity_arrays = [macro_data[m]["npv_equity"] for m in macro_names]
firm_arrays = [macro_data[m]["npv_firm"] for m in macro_names]
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ---- combined equity
nondet_eq, det_eq = _split_det_nondet(macro_names, equity_arrays)
if nondet_eq or det_eq:
    scale_eq = _scale_factor_for([a for _, a in nondet_eq], extra_points=[v for _, v in det_eq])

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    _set_x_label(ax, "NPV (Equity) [currency]", scale_eq)
    ax.set_ylabel("Frequency")
    ax.set_title("NPV (Equity)\n(all macro scenarios)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # histograms + fits (non-deterministic)
    if nondet_eq:
        all_vals = np.concatenate([arr for _, arr in nondet_eq]) / scale_eq
        bins_eq = min(30, max(5, len(all_vals)))
        bin_edges_eq = np.linspace(float(all_vals.min()), float(all_vals.max()), bins_eq)

        for i, (macro_name, arr_raw) in enumerate(nondet_eq):
            arr = arr_raw / scale_eq
            color = color_cycle[i % len(color_cycle)]

            ax.hist(
                arr,
                bins=bin_edges_eq,
                histtype="step",
                linewidth=1.5,
                alpha=0.9,
                label=macro_name,
                color=color,
            )

            _plot_fit_curve(
                ax=ax,
                values=arr,
                bin_edges=bin_edges_eq,
                color=color,
                label=None,  # no separate legend entry for the fit (keeps legend clean)
                fit_distribution=FIT_DISTRIBUTION,
            )

    # deterministic vertical lines
    base_idx = len(nondet_eq)
    for j, (macro_name, v_raw) in enumerate(det_eq):
        color = color_cycle[(base_idx + j) % len(color_cycle)]
        ax.axvline(v_raw / scale_eq, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    # if only deterministic points, set sensible x-limits
    if not nondet_eq and det_eq:
        xs = np.array([v / scale_eq for _, v in det_eq], dtype=float)
        x_min, x_max = float(xs.min()), float(xs.max())
        if np.isclose(x_min, x_max):
            pad = 0.05 * (abs(x_min) + 1.0)
            ax.set_xlim(x_min - pad, x_max + pad)
        else:
            pad = 0.05 * (x_max - x_min)
            ax.set_xlim(x_min - pad, x_max + pad)

    ax.legend()
    out_path = RESULTS_FOLDER / "histogram_npv_equity_all_scenarios.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined NPV (Equity) plot to: {out_path}")
else:
    print("No NPV (Equity) data available; skipping combined equity NPV plot.")

# ---- combined firm
nondet_f, det_f = _split_det_nondet(macro_names, firm_arrays)
if nondet_f or det_f:
    scale_f = _scale_factor_for([a for _, a in nondet_f], extra_points=[v for _, v in det_f])

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    _set_x_label(ax, "NPV (Firm) [currency]", scale_f)
    ax.set_ylabel("Frequency")
    ax.set_title("NPV (Firm)\n(all macro scenarios)")
    ax.grid(True, linestyle="--", alpha=0.3)

    if nondet_f:
        all_vals = np.concatenate([arr for _, arr in nondet_f]) / scale_f
        bins_f = min(30, max(5, len(all_vals)))
        bin_edges_f = np.linspace(float(all_vals.min()), float(all_vals.max()), bins_f)

        for i, (macro_name, arr_raw) in enumerate(nondet_f):
            arr = arr_raw / scale_f
            color = color_cycle[i % len(color_cycle)]

            ax.hist(
                arr,
                bins=bin_edges_f,
                histtype="step",
                linewidth=1.5,
                alpha=0.9,
                label=macro_name,
                color=color,
            )

            _plot_fit_curve(
                ax=ax,
                values=arr,
                bin_edges=bin_edges_f,
                color=color,
                label=None,
                fit_distribution=FIT_DISTRIBUTION,
            )

    base_idx = len(nondet_f)
    for j, (macro_name, v_raw) in enumerate(det_f):
        color = color_cycle[(base_idx + j) % len(color_cycle)]
        ax.axvline(v_raw / scale_f, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    if not nondet_f and det_f:
        xs = np.array([v / scale_f for _, v in det_f], dtype=float)
        x_min, x_max = float(xs.min()), float(xs.max())
        if np.isclose(x_min, x_max):
            pad = 0.05 * (abs(x_min) + 1.0)
            ax.set_xlim(x_min - pad, x_max + pad)
        else:
            pad = 0.05 * (x_max - x_min)
            ax.set_xlim(x_min - pad, x_max + pad)

    ax.legend()
    out_path = RESULTS_FOLDER / "histogram_npv_firm_all_scenarios.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined NPV (Firm) plot to: {out_path}")
else:
    print("No NPV (Firm) data available; skipping combined firm NPV plot.")

print("\nDone.")
