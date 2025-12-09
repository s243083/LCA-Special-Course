import pandas as pd
import numpy as np
from typing import Optional
from core.ResponseFramework_dev.farm_setup import compute_effective_inflow

def simulate_block(
    wind_data: pd.DataFrame,
    farm: dict,
    wsp_min: float | None = 3.0,
    wsp_max: float | None = 25.0,
    use_sector_average: bool = False,
) -> pd.DataFrame:
    """
    Minimal power-only simulation using constant control.

    Inputs
    ------
    wind_data : DataFrame with columns ['timestamp', 'wsp', 'TI', 'wdir']
    farm      : dict with keys ['layout_df', 'x', 'y', 'flow_model', 'flow_type'?]

    Returns
    -------
    power_df : DataFrame with columns:
        ['timestamp', 'Power_id0', 'Power_id1', ..., 'FarmPower']  (MW)
    """
    # --- Validate & prepare inputs ---
    required = ["timestamp", "wsp", "TI", "wdir"]
    missing = [c for c in required if c not in wind_data.columns]
    if missing:
        raise ValueError(f"wind_data missing columns: {missing}")

    # drop rows with NaNs in required columns (power returns only on valid rows)
    valid_mask = ~wind_data[["wsp", "TI", "wdir"]].isna().any(axis=1)
    wind = wind_data.loc[valid_mask, required].copy()

    # arrays for compute_effective_inflow
    timestamps = wind["timestamp"].to_numpy()
    wsp = wind["wsp"].to_numpy()
    TI = wind["TI"].to_numpy()
    wdir = wind["wdir"].to_numpy()

    # --- Constant control (vectorized) ---
    n_turbs = len(farm["layout_df"])
    n_times = len(wind)

    # yaw=0°, tilt=0° everywhere
    yaw = np.zeros((n_turbs, n_times), dtype=float)
    tilt = np.zeros_like(yaw)

    # operating flag: 1 everywhere, optionally 0 outside [wsp_min, wsp_max]
    if (wsp_min is not None) or (wsp_max is not None):
        lo = -np.inf if wsp_min is None else float(wsp_min)
        hi = +np.inf if wsp_max is None else float(wsp_max)
        op_ts = ((wsp >= lo) & (wsp <= hi)).astype(int)          # shape (n_times,)
    else:
        op_ts = np.ones(n_times, dtype=int)

    operating = np.tile(op_ts, (n_turbs, 1))                     # shape (n_turbs, n_times)

    # --- Flow model call (PyWake) ---

    sim_res, sa, inflow_df, power_core_df = compute_effective_inflow(
        wsp=wsp,
        TI=TI,
        wdir=wdir,
        x=farm["x"],
        y=farm["y"],
        yaw=yaw,
        tilt=tilt,
        operating=operating,
        flow_model=farm["flow_model"],
        flow_type=farm.get("flow_type", "pywake"),
        use_sector_average=use_sector_average,
        dtype=np.float64,
    )

    # --- Output: power only ---
    power_df = power_core_df.copy()
    power_df.insert(0, "timestamp", timestamps)
    power_cols = [c for c in power_df.columns if c.startswith("Power_id")]
    power_df["FarmPower"] = power_df[power_cols].sum(axis=1)
    power_df.drop(columns=power_cols, inplace=True)

    return power_df


def simulate_distribution(
    distribution: pd.DataFrame,
    farm: dict,
    surrogate=None,                    # unused (kept for signature compatibility)
    control_fn=None,                   # unused
    channel_processing_specs=None,     # unused
    lifetime_years: int = 20,          # unused
    wsp_min: float = 3.0,
    wsp_max: float = 25.0,
    supports_operating_modes: bool = False,  # unused
    use_sector_average: bool = False,
    use_prices: bool = False,          # unused
    price_type: str | None = None,     # unused
    fixed_price_value: float | None = None,  # unused
    return_bin_outputs: bool = False   # unused
) -> pd.DataFrame:
    """
    Minimal power-only distribution simulation.

    Parameters
    ----------
    distribution : pd.DataFrame
        Required columns: ['wsp', 'TI', 'wdir', 'probability'].
    farm : dict
        Must contain: ['layout_df', 'x', 'y', 'flow_model'] and optional 'flow_type'.

    Returns
    -------
    pd.DataFrame
        Columns: ['wsp','TI','wdir','probability','Power_id0',...,'FarmPower'] (MW).
        One row per distribution bin (after dropping NaNs).
    """
    # --- Validate inputs ---
    req_cols = ["wsp", "TI", "wdir", "probability"]
    missing = [c for c in req_cols if c not in distribution.columns]
    if missing:
        raise ValueError(f"distribution missing columns: {missing}")

    # Drop bins with NaNs in required inflow columns
    valid = ~distribution[["wsp", "TI", "wdir", "probability"]].isna().any(axis=1)
    dist = distribution.loc[valid, req_cols].reset_index(drop=True)

    # Extract arrays for model call
    wsp = dist["wsp"].to_numpy()
    TI = dist["TI"].to_numpy()
    wdir = dist["wdir"].to_numpy()
    prob = dist["probability"].to_numpy()  # kept only to pass through to output

    # --- Constant control (yaw=0, tilt=0), operating mask from wsp range ---
    n_turbs = len(farm["layout_df"])
    n_bins = len(dist)

    yaw = np.zeros((n_turbs, n_bins), dtype=float)
    tilt = np.zeros_like(yaw)

    # Operating flag per bin, broadcast to all turbines
    lo = -np.inf if wsp_min is None else float(wsp_min)
    hi = +np.inf if wsp_max is None else float(wsp_max)
    op_bins = ((wsp >= lo) & (wsp <= hi)).astype(int)         # (n_bins,)
    operating = np.tile(op_bins, (n_turbs, 1))                # (n_turbs, n_bins)

    # --- PyWake call ---
    _, _, _, power_core_df = compute_effective_inflow(
        wsp=wsp,
        TI=TI,
        wdir=wdir,
        x=farm["x"],
        y=farm["y"],
        yaw=yaw,
        tilt=tilt,
        operating=operating,
        flow_model=farm["flow_model"],
        flow_type=farm.get("flow_type", "pywake"),
        use_sector_average=use_sector_average,
        dtype=np.float64,
    )

    # --- Build output: per-bin power + farm total ---
    out = pd.concat(
        [
            dist[["wsp", "TI", "wdir", "probability"]].reset_index(drop=True),
            power_core_df.reset_index(drop=True),
        ],
        axis=1,
    )
    power_cols = [c for c in out.columns if c.startswith("Power_id")]
    out["FarmPower"] = out[power_cols].sum(axis=1)

    return out

#expected farm power (MW) across the distribution:
# expected_farm_power = (out["FarmPower"] * out["probability"]).sum()