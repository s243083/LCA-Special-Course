#!/usr/bin/env python3
"""
Post-processing: histograms + scatters (NPV, strike, support totals, support rate)
--------------------------------------------------------------------------------

Produces (overlayed by "case"):

  1) Histogram of NPV (npv_firm) per case
  2) Scatter: share of negative price hours vs. strike price per case
  3) Histogram of total support payment per case
  4) Histogram of strike price per case

  5) Histogram of support_rate_lifetime per case                          (NEW)
  6) Scatter: support_rate_lifetime vs. strike price per case              (NEW)
  7) Scatter: support_rate_lifetime vs. negative price share per case      (NEW)

Legend naming / grouping:
  - Uses ONLY scenario overrides["name"] as case name (fallback: sc["name"], sc["label"], scenario_id)
  - Optional manual mapping: LEGEND_NAME_MAP maps derived case name -> legend label

Inputs (expected in RESULTS_FOLDER):
  - scenarios.json
  - valuation_metrics_df_{SID}.parquet
  - market_statistics_summary_records_df_{SID}.parquet   (Option A summary table)

Assumptions:
  - Strike price is stored in valuation metrics as column: "strike_price"
  - support_rate_lifetime is stored in valuation metrics as column: "support_rate_lifetime"
  - Negative-price share is stored in market statistics summary as:
        "neg_share" (fraction in [0,1]) preferred
    with optional fallbacks:
        "neg_share_pct" or ("neg_hours_total","hours_total")
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# 1) Configuration
# --------------------------------------------------------------------
VAL_TABLE_NAME = "valuation_metrics"
MKTSTAT_SUMMARY_TABLE_NAME = "market_statistics_records"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed

RESULTS_FOLDER = PROJECT_ROOT / "results" / "DoE_SolveStrike_NPV0"
RESULTS_FOLDER = Path(RESULTS_FOLDER)
if not RESULTS_FOLDER.exists():
    raise FileNotFoundError(
        f"Results folder not found: {RESULTS_FOLDER}\n"
        f"Edit RESULTS_FOLDER at the top of this script to match your output folder."
    )

RESULTS_FOLDER_FIG = PROJECT_ROOT / "results" / "Figures" / "DoE_SolveStrike_NPV0"
RESULTS_FOLDER_FIG.mkdir(parents=True, exist_ok=True)

SCENARIOS_PATH = RESULTS_FOLDER / "scenarios.json"
if not SCENARIOS_PATH.is_file():
    raise FileNotFoundError(f"scenarios.json not found at: {SCENARIOS_PATH}")

# --- Legend label mapping (EDIT THIS) --------------------------------
# Keys are derived case names (overrides["name"]).
LEGEND_NAME_MAP: Dict[str, str] = {
    # "CfD_Capability": "Capability CfD",
}
# If True: cases not in LEGEND_NAME_MAP keep their derived case name in the legend.
# If False: cases not in LEGEND_NAME_MAP are excluded from plotting.
KEEP_UNMAPPED_CASES = True
# --------------------------------------------------------------------


# Deterministic detection tolerance (for histogram "vertical line" behavior)
DET_ATOL = 1e-12
DET_RTOL = 1e-10

# Histogram binning
N_BINS = 120
MIN_BINS = 10

# Plot styling
FILL_ALPHA = 0.20
EDGE_LW = 1.6
DRAW_OUTLINE = True
USE_DENSITY = True
DPI = 200

# --- Print layout: 200 mm x 80 mm ---
MM_TO_INCH = 1.0 / 25.4
FIGSIZE_200x80 = (200 * MM_TO_INCH, 80 * MM_TO_INCH)

# Font sizes tuned for 200x80 mm
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
def get_case_name(sc: Dict[str, Any], sid: str) -> str:
    """
    Case name used for grouping and legend.

    Uses ONLY:
        overrides["name"]

    Fallbacks:
        sc["name"]
        sc["label"]
        scenario_id
    """
    overrides = sc.get("overrides") or {}
    name = (
        overrides.get("name")
        or sc.get("name")
        or sc.get("label")
        or sid
    )
    return str(name)


def legend_label_for_case(case_name: str) -> Optional[str]:
    """
    Apply manual legend rename mapping to a derived case name.
    """
    if case_name in LEGEND_NAME_MAP:
        return str(LEGEND_NAME_MAP[case_name])
    return case_name if KEEP_UNMAPPED_CASES else None


def _read_parquet_table(table_name: str, sid: str) -> pd.DataFrame:
    """Read {table_name}_df_{sid}.parquet from RESULTS_FOLDER."""
    p = RESULTS_FOLDER / f"{table_name}_df_{sid}.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return df


def _is_deterministic(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    v0 = float(values[0])
    return bool(np.allclose(values, v0, rtol=DET_RTOL, atol=DET_ATOL))


def _split_det_nondet(
    labels: List[str], arrays: List[np.ndarray]
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, float]]]:
    nondet: List[Tuple[str, np.ndarray]] = []
    det: List[Tuple[str, float]] = []
    for label, arr in zip(labels, arrays):
        if arr.size == 0:
            continue
        if _is_deterministic(arr):
            det.append((label, float(arr[0])))
        else:
            nondet.append((label, arr))
    return nondet, det


def _make_bin_edges(values: np.ndarray) -> np.ndarray:
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


def _scale_factor_for(values_list: List[np.ndarray], extra_points: Optional[List[float]] = None) -> float:
    maxima: List[float] = []
    for arr in values_list:
        if arr.size > 0:
            maxima.append(float(np.nanmax(np.abs(arr))))
    if extra_points:
        maxima.extend([float(abs(v)) for v in extra_points if np.isfinite(v)])
    if not maxima:
        return 1.0
    vmax = float(np.nanmax(np.array(maxima, dtype=float)))
    return 1e6 if vmax >= 1e6 else 1.0


def _set_x_label(ax, base_label: str, scale: float, fs: int) -> None:
    ax.set_xlabel(base_label + (" [million]" if scale == 1e6 else ""), fontsize=fs)


def _col_to_arr(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return vals.to_numpy(dtype=float)


def _read_valuation_metrics(sid: str) -> Dict[str, np.ndarray]:
    """
    Returns:
      - npv_firm: np.ndarray
      - support_total: np.ndarray (support_total_undisc preferred; abs fallback; sum support_payment fallback)
      - strike_price: np.ndarray (expects column "strike_price")
      - support_rate_lifetime: np.ndarray (expects column "support_rate_lifetime")
    """
    df = _read_parquet_table(VAL_TABLE_NAME, sid)
    if df.empty:
        return {
            "npv_firm": np.array([], float),
            "support_total": np.array([], float),
            "strike_price": np.array([], float),
            "support_rate_lifetime": np.array([], float),
        }

    npv = _col_to_arr(df, "npv_firm")

    # support total selection / fallback
    support = _col_to_arr(df, "support_total_undisc")
    if support.size == 0:
        support = _col_to_arr(df, "support_total_abs_undisc")
    if support.size == 0 and "support_payment" in df.columns:
        vals = pd.to_numeric(df["support_payment"], errors="coerce").dropna().to_numpy(dtype=float)
        support = np.array([float(np.nansum(vals))], dtype=float) if vals.size else np.array([], float)

    strike = _col_to_arr(df, "strike_price")
    supp_rate = _col_to_arr(df, "support_rate_lifetime")

    return {
        "npv_firm": npv,
        "support_total": support,
        "strike_price": strike,
        "support_rate_lifetime": supp_rate,
    }


def _read_negative_price_share(sid: str) -> Optional[float]:
    """
    Reads negative price share from Option A summary table:
      market_statistics_summary_records_df_{sid}.parquet

    Returns fraction in [0,1] or None.

    Preferred:
      - neg_share (fraction)
    Fallbacks:
      - neg_share_pct (percent)
      - neg_hours_total / hours_total
    """
    try:
        df = _read_parquet_table(MKTSTAT_SUMMARY_TABLE_NAME, sid)
    except FileNotFoundError:
        return None

    if df.empty:
        return None

    row = df.iloc[0]

    if "neg_share" in df.columns:
        v = pd.to_numeric(row["neg_share"], errors="coerce")
        return float(v) if pd.notna(v) else None

    if "neg_share_pct" in df.columns:
        v = pd.to_numeric(row["neg_share_pct"], errors="coerce")
        return float(v) / 100.0 if pd.notna(v) else None

    if {"neg_hours_total", "hours_total"}.issubset(df.columns):
        neg = pd.to_numeric(row["neg_hours_total"], errors="coerce")
        tot = pd.to_numeric(row["hours_total"], errors="coerce")
        if pd.notna(neg) and pd.notna(tot) and float(tot) > 0.0:
            return float(neg) / float(tot)

    return None


def _pick_scalar(arr: np.ndarray) -> Optional[float]:
    """Pick a single scalar from an array (deterministic metric)."""
    if arr.size == 0:
        return None
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return None
    return float(a[0])


def _plot_hist_overlay(
    *,
    arrays: List[np.ndarray],
    labels: List[str],
    xlabel: str,
    out_name: str,
    title: Optional[str] = None,
) -> None:
    nondet, det = _split_det_nondet(labels, arrays)
    if not nondet and not det:
        print(f"No data for {out_name}; skipping.")
        return

    fig, ax = plt.subplots(
        1, 1,
        figsize=FIGSIZE_200x80,
        dpi=DPI,
        gridspec_kw={"left": 0.08, "right": 0.98, "top": 0.90, "bottom": 0.35},
    )

    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_ylabel("Density" if USE_DENSITY else "Frequency", fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    scale = _scale_factor_for([a for _, a in nondet], extra_points=[v for _, v in det])
    _set_x_label(ax, xlabel, scale, FS_LABEL)

    pooled_parts: List[np.ndarray] = []
    pooled_parts += [arr for _, arr in nondet]
    pooled_parts += [np.array([v], dtype=float) for _, v in det]
    pooled = np.concatenate(pooled_parts) / scale
    bin_edges = _make_bin_edges(pooled)

    for i, (lab, arr_raw) in enumerate(nondet):
        arr = arr_raw / scale
        color = color_cycle[i % len(color_cycle)]
        ax.hist(
            arr,
            bins=bin_edges,
            histtype="stepfilled",
            alpha=FILL_ALPHA,
            density=USE_DENSITY,
            color=color,
            label=lab,
        )
        if DRAW_OUTLINE:
            ax.hist(
                arr,
                bins=bin_edges,
                histtype="step",
                linewidth=EDGE_LW,
                density=USE_DENSITY,
                color=color,
            )

    base_idx = len(nondet)
    for j, (lab, v_raw) in enumerate(det):
        color = color_cycle[(base_idx + j) % len(color_cycle)]
        ax.axvline(v_raw / scale, color=color, linewidth=2.0, linestyle="--", label=lab)

    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            ncol=min(4, len(leg_labels)),
            fontsize=FS_LEGEND,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    out_path = RESULTS_FOLDER_FIG / out_name
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


def _plot_scatter_two_vars(
    *,
    points_by_label: Dict[str, List[Tuple[float, float]]],
    xlabel: str,
    ylabel: str,
    y_is_fraction_to_pct: bool,
    out_name: str,
    title: Optional[str] = None,
) -> None:
    npts = sum(len(v) for v in points_by_label.values())
    if npts == 0:
        print(f"No points for {out_name}; skipping.")
        return

    fig, ax = plt.subplots(
        1, 1,
        figsize=FIGSIZE_200x80,
        dpi=DPI,
        gridspec_kw={"left": 0.10, "right": 0.98, "top": 0.90, "bottom": 0.35},
    )

    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK)

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = list(points_by_label.keys())

    for i, lab in enumerate(labels):
        pts = points_by_label[lab]
        if not pts:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if y_is_fraction_to_pct:
            ys = ys * 100.0

        color = color_cycle[i % len(color_cycle)]
        ax.scatter(xs, ys, s=18, alpha=0.8, color=color, label=lab)

        # label mean marker
        ax.scatter([float(np.mean(xs))], [float(np.mean(ys))], s=55, marker="x", color=color)

    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            ncol=min(4, len(leg_labels)),
            fontsize=FS_LEGEND,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    out_path = RESULTS_FOLDER_FIG / out_name
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------
# 4) Group scenario_ids by case name (overrides["name"])
# --------------------------------------------------------------------
case_to_sids: Dict[str, List[str]] = {}

for sc in scenarios:
    sid = str(sc.get("scenario_id", "")).strip()
    if not sid:
        continue

    case = get_case_name(sc, sid)
    case_to_sids.setdefault(case, [])
    if sid not in case_to_sids[case]:
        case_to_sids[case].append(sid)

if not case_to_sids:
    raise RuntimeError("No cases could be identified; check scenarios.json content")

print("\nCases found (unique scenario_ids):")
for case, sids in case_to_sids.items():
    lab = legend_label_for_case(case)
    note = "" if (lab is not None) else "  (SKIPPED: not in LEGEND_NAME_MAP)"
    if lab is not None and lab != case:
        note = f"  (legend: '{lab}')"
    print(f"  {case}: {len(sids)} replicate(s){note}")


# --------------------------------------------------------------------
# 5) Load per-case data
# --------------------------------------------------------------------
missing: List[Tuple[str, str, str, str]] = []
case_data: Dict[str, Dict[str, Any]] = {}

for case, sids in case_to_sids.items():
    legend_case = legend_label_for_case(case)
    if legend_case is None:
        continue

    npv_list: List[float] = []
    support_list: List[float] = []
    strike_list: List[float] = []
    supp_rate_list: List[float] = []

    # points
    pts_negshare_vs_strike: List[Tuple[float, float]] = []
    pts_supp_rate_vs_strike: List[Tuple[float, float]] = []
    pts_supp_rate_vs_negshare: List[Tuple[float, float]] = []

    for sid in sids:
        # valuation metrics (npv + support + strike + support_rate_lifetime)
        try:
            dval = _read_valuation_metrics(sid)

            npv_list.extend([float(v) for v in dval["npv_firm"] if np.isfinite(v)])
            support_list.extend([float(v) for v in dval["support_total"] if np.isfinite(v)])
            strike_list.extend([float(v) for v in dval["strike_price"] if np.isfinite(v)])
            supp_rate_list.extend([float(v) for v in dval["support_rate_lifetime"] if np.isfinite(v)])

            strike_scalar = _pick_scalar(dval["strike_price"])
            supp_rate_scalar = _pick_scalar(dval["support_rate_lifetime"])
        except FileNotFoundError as e:
            missing.append((case, sid, str(e), "Missing valuation_metrics parquet"))
            strike_scalar = None
            supp_rate_scalar = None
        except Exception as e:
            missing.append((case, sid, f"{VAL_TABLE_NAME}_df_{sid}.parquet", f"Valuation read error: {e}"))
            strike_scalar = None
            supp_rate_scalar = None

        # market stats summary (negative share)
        try:
            neg_share = _read_negative_price_share(sid)
        except Exception as e:
            missing.append((case, sid, f"{MKTSTAT_SUMMARY_TABLE_NAME}_df_{sid}.parquet", f"Market stats read error: {e}"))
            neg_share = None

        # scatter points
        if strike_scalar is not None and neg_share is not None:
            pts_negshare_vs_strike.append((float(strike_scalar), float(neg_share)))

        if strike_scalar is not None and supp_rate_scalar is not None:
            pts_supp_rate_vs_strike.append((float(strike_scalar), float(supp_rate_scalar)))

        if neg_share is not None and supp_rate_scalar is not None:
            pts_supp_rate_vs_negshare.append((float(neg_share), float(supp_rate_scalar)))

    npv = np.array(npv_list, dtype=float)
    support = np.array(support_list, dtype=float)
    strike = np.array(strike_list, dtype=float)
    supp_rate = np.array(supp_rate_list, dtype=float)

    if npv.size or support.size or strike.size or supp_rate.size or pts_negshare_vs_strike or pts_supp_rate_vs_strike or pts_supp_rate_vs_negshare:
        case_data[legend_case] = {
            "npv": npv,
            "support_total": support,
            "strike_price": strike,
            "support_rate_lifetime": supp_rate,
            "pts_negshare_vs_strike": pts_negshare_vs_strike,
            "pts_supp_rate_vs_strike": pts_supp_rate_vs_strike,
            "pts_supp_rate_vs_negshare": pts_supp_rate_vs_negshare,
        }

print(f"\nCollected data for {len(case_data)} plotted case(s).")

if missing:
    print("\nFiles/issues:")
    for (case, sid, path, reason) in missing:
        print(f" - case='{case}', scenario_id={sid}, path={path}, reason={reason}")

if not case_data:
    raise RuntimeError("No usable case data; cannot plot.")


# --------------------------------------------------------------------
# 6) Build arrays + make plots
# --------------------------------------------------------------------
case_labels = list(case_data.keys())

npv_arrays = [case_data[c]["npv"] for c in case_labels]
support_arrays = [case_data[c]["support_total"] for c in case_labels]
strike_arrays = [case_data[c]["strike_price"] for c in case_labels]
supp_rate_arrays = [case_data[c]["support_rate_lifetime"] for c in case_labels]

points_negshare_vs_strike = {c: case_data[c]["pts_negshare_vs_strike"] for c in case_labels}
points_supp_rate_vs_strike = {c: case_data[c]["pts_supp_rate_vs_strike"] for c in case_labels}
points_supp_rate_vs_negshare = {c: case_data[c]["pts_supp_rate_vs_negshare"] for c in case_labels}

print("\nCreating plots...")

_plot_hist_overlay(
    arrays=npv_arrays,
    labels=case_labels,
    xlabel="NPV [€]",
    out_name="histogram_npv_per_case.png",
)

_plot_scatter_two_vars(
    points_by_label=points_negshare_vs_strike,
    xlabel="Strike price [€ / MWh]",
    ylabel="Negative price share [% of hours]",
    y_is_fraction_to_pct=True,
    out_name="scatter_neg_price_share_vs_strike_per_case.png",
)

_plot_hist_overlay(
    arrays=support_arrays,
    labels=case_labels,
    xlabel="Total support payment [€]",
    out_name="histogram_total_support_payment_per_case.png",
)

_plot_hist_overlay(
    arrays=strike_arrays,
    labels=case_labels,
    xlabel="Strike price [€ / MWh]",
    out_name="histogram_strike_price_per_case.png",
)

# NEW: support_rate_lifetime histogram
_plot_hist_overlay(
    arrays=supp_rate_arrays,
    labels=case_labels,
    xlabel="Support rate (lifetime) [€ / MWh]",
    out_name="histogram_support_rate_lifetime_per_case.png",
)

# NEW: support_rate_lifetime vs strike
_plot_scatter_two_vars(
    points_by_label=points_supp_rate_vs_strike,
    xlabel="Strike price [€ / MWh]",
    ylabel="Support rate (lifetime) [€ / MWh]",
    y_is_fraction_to_pct=False,
    out_name="scatter_support_rate_lifetime_vs_strike_per_case.png",
)

# NEW: support_rate_lifetime vs negative price share
_plot_scatter_two_vars(
    points_by_label=points_supp_rate_vs_negshare,
    xlabel="Negative price share [% of hours]",
    ylabel="Support rate (lifetime) [€ / MWh]",
    y_is_fraction_to_pct=False,
    out_name="scatter_support_rate_lifetime_vs_neg_price_share_per_case.png",
)

print("\nDone.")