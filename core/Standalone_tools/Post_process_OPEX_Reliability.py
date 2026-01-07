#!/usr/bin/env python3
"""
Combined NPV + Total OPEX Histograms Script
------------------------------------------

Creates combined histograms of:
  - NPV (equity and firm)  [LEFT subplot]
  - Total OPEX (sum of OM_payment) [RIGHT subplot]
across all macro scenarios (Scenario.name), overlaying each macro scenario.

Outputs (for each NPV metric):
1) Two-panel figure WITH optional distribution fit curves on NPV
2) Two-panel figure WITHOUT fit curves on NPV

Notes:
- Total OPEX is computed per scenario_id (SID) as:
      total_opex_eur = -sum(OM_payment)
  assuming OM_payment is negative outflow in the simulation.
- Histograms are transparent filled + crisp outlines.
- Optional density normalization (recommended for comparing shapes).
- Deterministic scenarios are plotted as vertical lines.
- Uses shared bin edges per panel (per metric) to compare macro overlays.

Assumptions:
- scenarios.json exists in RESULTS_FOLDER.
- For each scenario_id SID there is a parquet file:
    valuation_metrics_df_{SID}.parquet  with columns including "npv_equity", "npv_firm"
    opex_records_df_{SID}.parquet       with columns including "OM_payment"
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
VAL_TABLE_NAME = "valuation_metrics"
OPEX_TABLE_NAME = "opex_records"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
RESULTS_FOLDER = PROJECT_ROOT / "results" / "OPEX_Reliability"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
assert RESULTS_FOLDER.exists(), f"Results folder not found: {RESULTS_FOLDER}"

RESULTS_FOLDER_FIG = PROJECT_ROOT / "results" / "Figures" / "OPEX_Reliability"
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
FIGSIZE = (4, 2.5)
DPI = 200

# Total OPEX sign convention:
# If OM_payment is negative outflow, total OPEX = -sum(OM_payment)
OPEX_NEGATIVE_OUTFLOW = True

# --- Print layout: 200 mm x 80 mm ---
MM_TO_INCH = 1.0 / 25.4
FIGSIZE_200x80 = (200 * MM_TO_INCH, 80 * MM_TO_INCH)

# --- Font sizes tuned for 200x80 mm ---
FS_TITLE = 9
FS_LABEL = 8
FS_TICK = 7
FS_LEGEND = 7


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
    ax.set_xlabel(
        base_label + (" [million]" if scale == 1e6 else ""),
        fontsize=FS_LABEL,
    )

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


def _read_valuation_metrics(sid: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (npv_equity_values, npv_firm_values). Potentially multiple samples per SID."""
    parquet_path = RESULTS_FOLDER / f"{VAL_TABLE_NAME}_df_{sid}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(str(parquet_path))
    df = pd.read_parquet(parquet_path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    eq = np.array([], dtype=float)
    fm = np.array([], dtype=float)

    if "npv_equity" in df.columns:
        vals = pd.to_numeric(df["npv_equity"], errors="coerce").dropna()
        eq = vals.to_numpy(dtype=float)

    if "npv_firm" in df.columns:
        vals = pd.to_numeric(df["npv_firm"], errors="coerce").dropna()
        fm = vals.to_numpy(dtype=float)

    return eq, fm


def _read_total_opex(sid: str) -> float:
    """Return one scalar total OPEX per SID."""
    parquet_path = RESULTS_FOLDER / f"{OPEX_TABLE_NAME}_df_{sid}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(str(parquet_path))
    df = pd.read_parquet(parquet_path)
    if not isinstance(df, pd.DataFrame) or df.empty or "OM_payment" not in df.columns:
        return float("nan")

    pay = pd.to_numeric(df["OM_payment"], errors="coerce").fillna(0.0)
    s = float(pay.sum())
    return float(-s) if OPEX_NEGATIVE_OUTFLOW else float(s)


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
# 5) Load valuation_metrics + total OPEX and aggregate per macro scenario
# --------------------------------------------------------------------
macro_data: Dict[str, Dict[str, np.ndarray]] = {}
missing_files: List[Any] = []

for macro_name, entries_by_sid in macro_to_entries.items():
    npv_equity_list: List[float] = []
    npv_firm_list: List[float] = []
    opex_total_list: List[float] = []

    for sid in entries_by_sid:
        # valuation metrics (may provide multiple rows / samples)
        try:
            eq_arr, fm_arr = _read_valuation_metrics(sid)
            npv_equity_list.extend([float(v) for v in eq_arr])
            npv_firm_list.extend([float(v) for v in fm_arr])
        except FileNotFoundError as e:
            missing_files.append((macro_name, sid, str(e), "Missing valuation file"))
        except Exception as e:
            missing_files.append((macro_name, sid, f"{VAL_TABLE_NAME}_df_{sid}.parquet", f"Valuation read error: {e}"))

        # total OPEX (one scalar per SID)
        try:
            total_opex = _read_total_opex(sid)
            if np.isfinite(total_opex):
                opex_total_list.append(float(total_opex))
        except FileNotFoundError as e:
            missing_files.append((macro_name, sid, str(e), "Missing opex_records file"))
        except Exception as e:
            missing_files.append((macro_name, sid, f"{OPEX_TABLE_NAME}_df_{sid}.parquet", f"OPEX read error: {e}"))

    eq = np.array(npv_equity_list, dtype=float)
    fm = np.array(npv_firm_list, dtype=float)
    ox = np.array(opex_total_list, dtype=float)

    if eq.size or fm.size or ox.size:
        macro_data[macro_name] = {"npv_equity": eq, "npv_firm": fm, "opex_total": ox}
        print(
            f"\nMacro scenario '{macro_name}': "
            f"{eq.size} NPV_equity sample(s), "
            f"{fm.size} NPV_firm sample(s), "
            f"{ox.size} total OPEX sample(s)"
        )

print(f"\nCollected data for {len(macro_data)} macro scenario(s).")
if missing_files:
    print("\nFiles not loaded:")
    for m in missing_files:
        print(f" - macro='{m[0]}', scenario_id={m[1]}, path={m[2]}, reason={m[3]}")

if not macro_data:
    raise RuntimeError("No usable macro scenario data; cannot plot.")


# --------------------------------------------------------------------
# 6) Two-panel combined plots (NPV left, Total OPEX right)
# --------------------------------------------------------------------
print("\nCreating combined two-panel plots for all macro scenarios...")

macro_names = list(macro_data.keys())
equity_arrays = [macro_data[m].get("npv_equity", np.array([], dtype=float)) for m in macro_names]
firm_arrays = [macro_data[m].get("npv_firm", np.array([], dtype=float)) for m in macro_names]
opex_arrays = [macro_data[m].get("opex_total", np.array([], dtype=float)) for m in macro_names]
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_two_panel(
    arrays_left: List[np.ndarray],
    left_title: str,
    left_xlabel: str,
    arrays_right: List[np.ndarray],
    right_title: str,
    right_xlabel: str,
    out_name: str,
    show_fits_left: bool = True,
    show_fits_right: bool = False,
) -> None:
    nondet_L, det_L = _split_det_nondet(macro_names, arrays_left)
    nondet_R, det_R = _split_det_nondet(macro_names, arrays_right)

    if (not nondet_L and not det_L) and (not nondet_R and not det_R):
        print(f"No data available for '{out_name}'; skipping.")
        return

    fig, (axL, axR) = plt.subplots(
        1, 2,
        figsize=FIGSIZE_200x80,
        dpi=DPI,
        gridspec_kw={
            "left": 0.08,
            "right": 0.98,
            "top": 0.88,
            "bottom": 0.35,  # room for centered legend
            "wspace": 0.25,
        },
    )

    for ax in (axL, axR):
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_ylabel("Density" if USE_DENSITY else "Frequency", fontsize=FS_LABEL)

    axL.set_title(left_title, fontsize=FS_TITLE)
    axR.set_title(right_title, fontsize=FS_TITLE)

    # ---------------- Left panel ----------------
    scale_L = _scale_factor_for([a for _, a in nondet_L], extra_points=[v for _, v in det_L])
    _set_x_label(axL, left_xlabel, scale_L)

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

    base_idx_L = len(nondet_L)
    for j, (macro_name, v_raw) in enumerate(det_L):
        color = color_cycle[(base_idx_L + j) % len(color_cycle)]
        axL.axvline(v_raw / scale_L, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    # ---------------- Right panel ----------------
    scale_R = _scale_factor_for([a for _, a in nondet_R], extra_points=[v for _, v in det_R])
    _set_x_label(axR, right_xlabel, scale_R)

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
            if show_fits_right:
                _plot_fit_curve(
                    ax=axR,
                    values=arr,
                    bin_edges=bin_edges_R,
                    color=color,
                    fit_distribution=FIT_DISTRIBUTION,
                    density=USE_DENSITY,
                )

    base_idx_R = len(nondet_R)
    for j, (macro_name, v_raw) in enumerate(det_R):
        color = color_cycle[(base_idx_R + j) % len(color_cycle)]
        axR.axvline(v_raw / scale_R, color=color, linewidth=2.0, linestyle="--", label=macro_name)

    # ---- Combined legend (centered below both subplots) ----
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()

    # de-duplicate while preserving order
    seen = set()
    handles, labels = [], []
    for h, lab in list(zip(hL, lL)) + list(zip(hR, lR)):
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(h)
        labels.append(lab)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=FS_LEGEND,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    out_path = RESULTS_FOLDER_FIG / out_name
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


# ---- Equity: with fits + without fits
plot_two_panel(
    arrays_left=equity_arrays,
    left_title="NPV (Equity)\n(all macro scenarios)",
    left_xlabel="NPV (Equity) [currency]",
    arrays_right=opex_arrays,
    right_title="Total OPEX\n(all macro scenarios)",
    right_xlabel="Total OPEX [currency]",
    out_name="histogram_npv_equity_and_total_opex_all_scenarios.png",
    show_fits_left=True,
    show_fits_right=False,
)

plot_two_panel(
    arrays_left=equity_arrays,
    left_title="NPV (Equity)\n(all macro scenarios) — no fit",
    left_xlabel="NPV (Equity) [currency]",
    arrays_right=opex_arrays,
    right_title="Total OPEX\n(all macro scenarios)",
    right_xlabel="Total OPEX [currency]",
    out_name="histogram_npv_equity_and_total_opex_all_scenarios_nofit.png",
    show_fits_left=False,
    show_fits_right=False,
)

# ---- Firm: with fits + without fits
plot_two_panel(
    arrays_left=firm_arrays,
    left_title="NPV",
    left_xlabel="NPV  [€]",
    arrays_right=opex_arrays,
    right_title="Total OPEX",
    right_xlabel="Total OPEX [€]",
    out_name="histogram_npv_firm_and_total_opex_all_scenarios.png",
    show_fits_left=True,
    show_fits_right=False,
)

plot_two_panel(
    arrays_left=firm_arrays,
    left_title="NPV",
    left_xlabel="NPV [€]",
    arrays_right=opex_arrays,
    right_title="Total OPEX",
    right_xlabel="Total OPEX [€]",
    out_name="histogram_npv_firm_and_total_opex_all_scenarios_nofit.png",
    show_fits_left=False,
    show_fits_right=False,
)

print("\nDone.")
