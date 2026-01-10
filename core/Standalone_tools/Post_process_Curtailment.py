#!/usr/bin/env python3
"""
NPV + Lost Energy (Curtailment) — Combined 1x2 Subplot Histograms
------------------------------------------------------------------

Creates 1x2 subplot figures across all macro scenarios (Scenario.name):

LEFT  subplot: NPV distribution (Equity or Firm) across macro scenarios
RIGHT subplot: Lost Energy due to Curtailment = (Baseline energy reference - Scenario total_energy_mwh)

Legend is placed below both subplots.

Outputs (for each NPV metric):
1) Figure with optional fit curves on NPV (Gaussian or Lognormal)
2) Same figure WITHOUT fit curves

Assumptions:
- scenarios.json exists in RESULTS_FOLDER
- For each scenario_id SID there is a parquet file:
    {TABLE_NAME}_df_{SID}.parquet
  with columns including:
    - npv_equity
    - npv_firm
    - total_energy_mwh   (as produced by Valuation.valuation())
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

# RESULTS folder (Curtailment experiment)
RESULTS_FOLDER = PROJECT_ROOT / "results" / "Curtailment"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

# Figures output folder
RESULTS_FOLDER_FIG = PROJECT_ROOT / "results" / "Figures" / "Curtailment"
RESULTS_FOLDER_FIG.mkdir(parents=True, exist_ok=True)

SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"
if not SCENARIOS_PATH.is_file():
    raise FileNotFoundError(f"scenarios.json not found at: {SCENARIOS_PATH}")

# Choose: "gaussian" or "lognormal" (fits are only applied to NPV subplot)
FIT_DISTRIBUTION = "lognormal"

# Deterministic detection tolerance
DET_ATOL = 1e-12
DET_RTOL = 1e-10

# Histogram binning
N_BINS = 90
MIN_BINS = 10

# Plot styling
FILL_ALPHA = 0.20
EDGE_LW = 1.6
DRAW_OUTLINE = True
USE_DENSITY = True
DPI = 200


# --- Print layout: 200 mm x 80 mm (1x2 panels) ---
MM_TO_INCH = 1.0 / 25.4
FIGSIZE_200x80 = (200 * MM_TO_INCH, 80 * MM_TO_INCH)

# Font sizes tuned for 200x80 mm figure
FS_TITLE = 9      # was 8
FS_LABEL = 8      # was 7
FS_TICK = 7       # was 6
FS_LEGEND = 7     # was 6


# Lost energy construction
BASELINE_MATCH = "C0"      # substring used to identify baseline macro scenario name
BASELINE_STAT = "median"   # "mean" or "median" baseline energy reference statistic

# Column for energy in valuation metrics parquet
ENERGY_COL = "total_energy_mwh"


# --- Short legend labels ---
LEGEND_LABELS = {
    "C0 — Reference (no curtailment)": "C0",
    "C1 — Low transmission constraints": "C1",
    "C2 — Medium transmission constraints": "C2",
    "C3 — High transmission constraints": "C3",
    "C4 — Very high market curtailment occurrence": "C4",
    "C5 — Storage solutions development": "C5",
}



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
    ax.set_xlabel(base_label + (" [million]" if scale == 1e6 else ""), fontsize=FS_LABEL)


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
    - If density=True, fit plotted as a PDF (area=1).
    - If density=False, fit scaled to histogram counts (n_samples * bin_width).
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


def _style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK)


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


# Identify baseline macro name (C0)
baseline_candidates = [m for m in macro_to_entries.keys() if BASELINE_MATCH in m]
if not baseline_candidates:
    raise RuntimeError(f"No baseline macro found containing '{BASELINE_MATCH}' in macro names.")
if len(baseline_candidates) > 1:
    print(f"WARNING: multiple baseline candidates found: {baseline_candidates}. Using first.")
baseline_macro = baseline_candidates[0]
print(f"\nBaseline macro detected: {baseline_macro}")


# --------------------------------------------------------------------
# 5) Load valuation_metrics and aggregate per macro scenario
# --------------------------------------------------------------------
macro_data: Dict[str, Dict[str, np.ndarray]] = {}
missing_files: List[Any] = []

for macro_name, entries_by_sid in macro_to_entries.items():
    npv_equity_list: List[float] = []
    npv_firm_list: List[float] = []
    energy_total_list: List[float] = []

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

        # NPVs: keep all numeric samples if present
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

        # Total energy (MWh): typically one value per scenario run, but we accept arrays
        if ENERGY_COL in df.columns:
            e = pd.to_numeric(df[ENERGY_COL], errors="coerce").dropna()
            energy_total_list.extend(float(v) for v in e)
        else:
            print(f"WARNING: macro '{macro_name}', scenario_id {sid} has no '{ENERGY_COL}' column.")

    eq = np.array(npv_equity_list, dtype=float)
    fm = np.array(npv_firm_list, dtype=float)
    en = np.array(energy_total_list, dtype=float)

    if eq.size or fm.size or en.size:
        macro_data[macro_name] = {"npv_equity": eq, "npv_firm": fm, ENERGY_COL: en}
        print(
            f"\nMacro scenario '{macro_name}': "
            f"{eq.size} NPV_equity sample(s), "
            f"{fm.size} NPV_firm sample(s), "
            f"{en.size} total_energy_mwh sample(s)"
        )

print(f"\nCollected valuation metrics for {len(macro_data)} macro scenario(s).")
if missing_files:
    print("\nFiles not loaded:")
    for m in missing_files:
        print(f" - macro='{m[0]}', scenario_id={m[1]}, path={m[2]}, reason={m[3]}")

if not macro_data:
    raise RuntimeError("No usable macro scenario valuation data; cannot plot.")


# Baseline energy reference
baseline_energy_arr = macro_data.get(baseline_macro, {}).get(ENERGY_COL, np.array([], dtype=float))
baseline_energy_arr = baseline_energy_arr[np.isfinite(baseline_energy_arr)]
if baseline_energy_arr.size == 0:
    raise RuntimeError(f"Baseline macro '{baseline_macro}' has no valid '{ENERGY_COL}' samples.")

if BASELINE_STAT == "mean":
    baseline_energy_ref = float(np.nanmean(baseline_energy_arr))
elif BASELINE_STAT == "median":
    baseline_energy_ref = float(np.nanmedian(baseline_energy_arr))
else:
    raise ValueError("BASELINE_STAT must be 'mean' or 'median'.")

print(f"\nBaseline energy reference ({BASELINE_STAT}): {baseline_energy_ref:,.0f} MWh")


# --------------------------------------------------------------------
# 6) Prepare arrays for plotting
# --------------------------------------------------------------------
macro_names = list(macro_data.keys())

equity_arrays = [macro_data[m]["npv_equity"] for m in macro_names]
firm_arrays = [macro_data[m]["npv_firm"] for m in macro_names]

energy_arrays = [macro_data[m].get(ENERGY_COL, np.array([], dtype=float)) for m in macro_names]
lost_energy_arrays: List[np.ndarray] = []
for arr in energy_arrays:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        lost_energy_arrays.append(np.array([], dtype=float))
    else:
        lost_energy_arrays.append(baseline_energy_ref - arr)

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# --------------------------------------------------------------------
# 7) Plotting: 1x2 figure (NPV left, Lost Energy right) with shared legend below
# --------------------------------------------------------------------
def plot_two_panel(
    arrays_left: List[np.ndarray],
    arrays_right: List[np.ndarray],
    left_xlabel: str,
    right_xlabel: str,
    out_name: str,
    show_fits_left: bool = True,
) -> None:
    """
    Creates a 1x2 subplot:
      - Left: histogram overlay for arrays_left (NPV)
      - Right: histogram overlay for arrays_right (Lost Energy)
    Deterministic macro scenarios are drawn as vertical lines.
    Legend appears below both subplots.
    """
    # Split deterministic vs nondet per panel using the same macro_names ordering
    nondet_L, det_L = _split_det_nondet(macro_names, arrays_left)
    nondet_R, det_R = _split_det_nondet(macro_names, arrays_right)

    if (not nondet_L and not det_L) and (not nondet_R and not det_R):
        print(f"No data available; skipping {out_name}")
        return

    # Scaling per panel (kept independent, like your original script)
    scale_L = _scale_factor_for([a for _, a in nondet_L], extra_points=[v for _, v in det_L])
    scale_R = _scale_factor_for([a for _, a in nondet_R], extra_points=[v for _, v in det_R])

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=FIGSIZE_200x80, dpi=DPI, constrained_layout=False
    )

    # Reserve space for legend below both panelsfig.subplots_adjust(bottom=0.26, wspace=0.25)
    fig.subplots_adjust(bottom=0.24, wspace=0.25)

    # Style + labels
    _set_x_label(axL, left_xlabel, scale_L)
    _style_axes(axL, "Density" if USE_DENSITY else "Frequency")

    _set_x_label(axR, right_xlabel, scale_R)
    _style_axes(axR, "Density" if USE_DENSITY else "Frequency")

    # --- LEFT PANEL (NPV) ---
    if nondet_L:
        pooled_L = np.concatenate([arr for _, arr in nondet_L]) / scale_L
        bin_edges_L = _make_bin_edges(pooled_L)

        for i, (macro_name, arr_raw) in enumerate(nondet_L):
            arr = arr_raw / scale_L
            color = color_cycle[i % len(color_cycle)]

            axL.hist(
                arr,
                bins=bin_edges_L,
                histtype="stepfilled",
                alpha=FILL_ALPHA,
                density=USE_DENSITY,
                color=color,
                label=macro_name,
            )

            if DRAW_OUTLINE:
                axL.hist(
                    arr,
                    bins=bin_edges_L,
                    histtype="step",
                    linewidth=EDGE_LW,
                    density=USE_DENSITY,
                    color=color,
                )

            if show_fits_left:
                _plot_fit_curve(
                    ax=axL,
                    values=arr,
                    bin_edges=bin_edges_L,
                    color=color,
                    fit_distribution=FIT_DISTRIBUTION,
                    density=USE_DENSITY,
                )

    # Deterministic left as vertical lines
    base_idx_L = len(nondet_L)
    for j, (macro_name, v_raw) in enumerate(det_L):
        color = color_cycle[(base_idx_L + j) % len(color_cycle)]
        axL.axvline(v_raw / scale_L, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    # Sensible x-limits if only deterministic left
    if not nondet_L and det_L:
        xs = np.array([v / scale_L for _, v in det_L], dtype=float)
        x_min, x_max = float(xs.min()), float(xs.max())
        if np.isclose(x_min, x_max):
            pad = 0.05 * (abs(x_min) + 1.0)
            axL.set_xlim(x_min - pad, x_max + pad)
        else:
            pad = 0.05 * (x_max - x_min)
            axL.set_xlim(x_min - pad, x_max + pad)

    # --- RIGHT PANEL (Lost Energy) ---
    if nondet_R:
        pooled_R = np.concatenate([arr for _, arr in nondet_R]) / scale_R
        bin_edges_R = _make_bin_edges(pooled_R)

        for i, (macro_name, arr_raw) in enumerate(nondet_R):
            arr = arr_raw / scale_R
            color = color_cycle[i % len(color_cycle)]

            axR.hist(
                arr,
                bins=bin_edges_R,
                histtype="stepfilled",
                alpha=FILL_ALPHA,
                density=USE_DENSITY,
                color=color,
                label=macro_name,
            )

            if DRAW_OUTLINE:
                axR.hist(
                    arr,
                    bins=bin_edges_R,
                    histtype="step",
                    linewidth=EDGE_LW,
                    density=USE_DENSITY,
                    color=color,
                )

    # Deterministic right as vertical lines
    base_idx_R = len(nondet_R)
    for j, (macro_name, v_raw) in enumerate(det_R):
        color = color_cycle[(base_idx_R + j) % len(color_cycle)]
        axR.axvline(v_raw / scale_R, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    # Sensible x-limits if only deterministic right
    if not nondet_R and det_R:
        xs = np.array([v / scale_R for _, v in det_R], dtype=float)
        x_min, x_max = float(xs.min()), float(xs.max())
        if np.isclose(x_min, x_max):
            pad = 0.05 * (abs(x_min) + 1.0)
            axR.set_xlim(x_min - pad, x_max + pad)
        else:
            pad = 0.05 * (x_max - x_min)
            axR.set_xlim(x_min - pad, x_max + pad)

    # --- Shared legend below both plots ---
    handles, labels = axL.get_legend_handles_labels()
    if not handles:
        handles, labels = axR.get_legend_handles_labels()

    # Replace long macro names by short labels
    labels = [LEGEND_LABELS.get(lab, lab) for lab in labels]

    # Deduplicate while preserving order
    seen = set()
    handles_u = []
    labels_u = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)



    n_items = len(labels_u)
    ncol = min(n_items, 6)  # one row up to 6 items

    fig.legend(
        handles_u,
        labels_u,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        frameon=False,
        fontsize=FS_LEGEND,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    out_path = RESULTS_FOLDER_FIG / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------
# 8) Generate figures
# --------------------------------------------------------------------
print("\nCreating combined 1x2 NPV + Lost Energy plots for all macro scenarios...")

# --- Firm NPV + Lost Energy (with fits on NPV)
plot_two_panel(
    arrays_left=firm_arrays,
    arrays_right=lost_energy_arrays,
    left_xlabel="NPV [€]",
    right_xlabel="Lost Energy [MWh]",
    out_name="npv_firm_and_lost_energy_all_scenarios.png",
    show_fits_left=True,
)

# --- Firm NPV + Lost Energy (no fits)
plot_two_panel(
    arrays_left=firm_arrays,
    arrays_right=lost_energy_arrays,
    left_xlabel="NPV [€]",
    right_xlabel="Lost Energy [MWh]",
    out_name="npv_firm_and_lost_energy_all_scenarios_nofit.png",
    show_fits_left=False,
)

# --- Equity NPV + Lost Energy (with fits on NPV)
plot_two_panel(
    arrays_left=equity_arrays,
    arrays_right=lost_energy_arrays,
    left_xlabel="NPV (Equity) [€]",
    right_xlabel="Lost Energy [MWh]",
    out_name="npv_equity_and_lost_energy_all_scenarios.png",
    show_fits_left=True,
)

# --- Equity NPV + Lost Energy (no fits)
plot_two_panel(
    arrays_left=equity_arrays,
    arrays_right=lost_energy_arrays,
    left_xlabel="NPV (Equity) [€]",
    right_xlabel="Lost Energy [MWh]",
    out_name="npv_equity_and_lost_energy_all_scenarios_nofit.png",
    show_fits_left=False,
)

print("\nDone.")
