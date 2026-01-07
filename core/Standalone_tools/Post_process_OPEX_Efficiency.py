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
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Efficiency"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

# NEW: figures go to results/Figures/Experiment
RESULTS_FOLDER_FIG = PROJECT_ROOT / "results" / "Figures" / "OPEX_Efficiency"
RESULTS_FOLDER_FIG.mkdir(parents=True, exist_ok=True)


SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"
if not SCENARIOS_PATH.is_file():
    raise FileNotFoundError(f"scenarios.json not found at: {SCENARIOS_PATH}")

# Choose: "gaussian" or "lognormal"
FIT_DISTRIBUTION = "lognormal"

# Deterministic detection tolerance
DET_ATOL = 1e-12
DET_RTOL = 1e-10

# Histogram binning
N_BINS = 90        # increase for finer bins (e.g. 50–150)
MIN_BINS = 10      # minimum bin count when there are enough samples

# Plot styling
FILL_ALPHA = 0.20          # transparency of filled histogram
EDGE_LW = 1.6              # outline width
DRAW_OUTLINE = True        # outline over the fill
USE_DENSITY = True         # True -> density; False -> counts
FIGSIZE = (10, 6)
DPI = 200


# --------------------------------------------------------------------
# 2) Load scenarios.json
# --------------------------------------------------------------------
with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

if isinstance(scenarios, dict) and "scenarios" in scenarios:
    scenarios = scenarios["scenarios"]

if not isinstance(scenarios, list):
    raise TypeError("Expected a list of scenarios in scenarios.json")

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


def _plot_fit_curve(
    ax,
    values: np.ndarray,
    bin_edges: np.ndarray,
    color: str,
    fit_distribution: str,
    density: bool,
) -> None:
    """
    Plot either Gaussian or Lognormal fit.

    - If density=True, fit is plotted as a PDF (area=1).
    - If density=False, fit is scaled to histogram counts (n_samples * bin_width).
    """
    if values.size <= 1 or len(bin_edges) <= 1:
        return

    bin_width = float(bin_edges[1] - bin_edges[0])
    if bin_width <= 0:
        return

    dist = fit_distribution.strip().lower()

    if dist == "gaussian":
        mu = float(values.mean())
        sigma = float(values.std(ddof=1))
        if sigma <= 0:
            return

        x = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 400)
        y = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        if not density:
            y = y * int(values.size) * bin_width

        ax.plot(x, y, color=color, linewidth=1.8)
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

        x = np.linspace(float(pos.min()), float(pos.max()), 400)
        y = (1.0 / (x * sigma_log * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((np.log(x) - mu_log) / sigma_log) ** 2
        )

        if not density:
            y = y * int(pos.size) * bin_width

        ax.plot(x, y, color=color, linewidth=1.8)
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


def _make_bin_edges(values: np.ndarray) -> np.ndarray:
    """Shared bin edges for combined plots, based on pooled data range."""
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if np.isclose(vmin, vmax):
        pad = 0.05 * (abs(vmin) + 1.0)
        vmin -= pad
        vmax += pad

    nbins = min(N_BINS, max(MIN_BINS, int(values.size)))
    return np.linspace(vmin, vmax, nbins + 1)


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

if not macro_to_entries:
    raise RuntimeError("No macro scenarios could be identified; check Scenario.name in scenarios.json")

print("\nMacro scenarios found (unique scenario_ids):")
for macro, entries_by_sid in macro_to_entries.items():
    print(f"  {macro}: {len(entries_by_sid)} replicate(s)")


# --------------------------------------------------------------------
# 5) Load valuation_metrics and aggregate per macro scenario
# --------------------------------------------------------------------
macro_data: Dict[str, Dict[str, np.ndarray]] = {}
missing_files: List[Any] = []

for macro_name, entries_by_sid in macro_to_entries.items():
    npv_equity_list: List[float] = []
    npv_firm_list: List[float] = []

    for sid in entries_by_sid:
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
            print(f"WARNING: macro '{macro_name}', scenario_id {sid} has empty {TABLE_NAME}; skipping.")
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

    eq = np.array(npv_equity_list, dtype=float)
    fm = np.array(npv_firm_list, dtype=float)

    if eq.size or fm.size:
        macro_data[macro_name] = {"npv_equity": eq, "npv_firm": fm}
        print(
            f"\nMacro scenario '{macro_name}': "
            f"{eq.size} NPV_equity sample(s), "
            f"{fm.size} NPV_firm sample(s)"
        )

print(f"\nCollected valuation metrics for {len(macro_data)} macro scenario(s).")
if missing_files:
    print("\nFiles not loaded:")
    for m in missing_files:
        print(f" - macro='{m[0]}', scenario_id={m[1]}, path={m[2]}, reason={m[3]}")

if not macro_data:
    raise RuntimeError("No usable macro scenario valuation data; cannot plot.")


# --------------------------------------------------------------------
# 6) Combined plots (with & without fits)
# --------------------------------------------------------------------
print("\nCreating combined NPV plots for all macro scenarios...")

macro_names = list(macro_data.keys())
equity_arrays = [macro_data[m]["npv_equity"] for m in macro_names]
firm_arrays = [macro_data[m]["npv_firm"] for m in macro_names]
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_combined(
    arrays: List[np.ndarray],
    title: str,
    xlabel: str,
    out_name: str,
    show_fits: bool = True,
) -> None:
    nondet, det = _split_det_nondet(macro_names, arrays)
    if not nondet and not det:
        print(f"No data available for '{title}'; skipping.")
        return

    scale = _scale_factor_for([a for _, a in nondet], extra_points=[v for _, v in det])

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    _set_x_label(ax, xlabel, scale)
    ax.set_ylabel("Density" if USE_DENSITY else "Frequency")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if nondet:
        pooled = np.concatenate([arr for _, arr in nondet]) / scale
        bin_edges = _make_bin_edges(pooled)

        for i, (macro_name, arr_raw) in enumerate(nondet):
            arr = arr_raw / scale
            color = color_cycle[i % len(color_cycle)]

            # Filled histogram
            ax.hist(
                arr,
                bins=bin_edges,
                histtype="stepfilled",
                alpha=FILL_ALPHA,
                density=USE_DENSITY,
                color=color,
                label=macro_name,
            )

            # Crisp outline
            if DRAW_OUTLINE:
                ax.hist(
                    arr,
                    bins=bin_edges,
                    histtype="step",
                    linewidth=EDGE_LW,
                    density=USE_DENSITY,
                    color=color,
                )

            # Optional fit curve
            if show_fits:
                _plot_fit_curve(
                    ax=ax,
                    values=arr,
                    bin_edges=bin_edges,
                    color=color,
                    fit_distribution=FIT_DISTRIBUTION,
                    density=USE_DENSITY,
                )

    # Deterministic scenarios as vertical lines
    base_idx = len(nondet)
    for j, (macro_name, v_raw) in enumerate(det):
        color = color_cycle[(base_idx + j) % len(color_cycle)]
        ax.axvline(
            v_raw / scale,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=macro_name,
        )

    # If only deterministic points, set sensible x-limits
    if not nondet and det:
        xs = np.array([v / scale for _, v in det], dtype=float)
        x_min, x_max = float(xs.min()), float(xs.max())
        if np.isclose(x_min, x_max):
            pad = 0.05 * (abs(x_min) + 1.0)
            ax.set_xlim(x_min - pad, x_max + pad)
        else:
            pad = 0.05 * (x_max - x_min)
            ax.set_xlim(x_min - pad, x_max + pad)

    n_items = len(nondet) + len(det)
    ax.legend(frameon=False, fontsize=9, ncol=2 if n_items > 6 else 1)

    out_path = RESULTS_FOLDER_FIG / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


# ---- Equity: with fits + without fits
plot_combined(
    arrays=equity_arrays,
    title="NPV (Equity)\n(all macro scenarios)",
    xlabel="NPV (Equity) [currency]",
    out_name="histogram_npv_equity_all_scenarios.png",
    show_fits=True,
)

plot_combined(
    arrays=equity_arrays,
    title="NPV (Equity)\n(all macro scenarios) — no fit",
    xlabel="NPV (Equity) [currency]",
    out_name="histogram_npv_equity_all_scenarios_nofit.png",
    show_fits=False,
)

# ---- Firm: with fits + without fits
plot_combined(
    arrays=firm_arrays,
    title="NPV (Firm)\n(all macro scenarios)",
    xlabel="NPV (Firm) [currency]",
    out_name="histogram_npv_firm_all_scenarios.png",
    show_fits=True,
)

plot_combined(
    arrays=firm_arrays,
    title="NPV (Firm)\n(all macro scenarios) — no fit",
    xlabel="NPV (Firm) [currency]",
    out_name="histogram_npv_firm_all_scenarios_nofit.png",
    show_fits=False,
)

print("\nDone.")
