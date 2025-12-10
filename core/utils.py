import pandas as pd
import numpy as np
from typing import Optional, Sequence, Union
import re


import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import logging


import logging
from pathlib import Path

def init_experiment_logging(
    result_directory: str | Path,
    name: str,
    *,
    console: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialise a logger that writes to:

        <result_directory>/<name>/<name>.log

    Returns a logger (not the root logger).
    """

    # Build directory
    result_directory = Path(result_directory)
    log_dir = result_directory / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Final log file path
    log_file = log_dir / f"{name}.log"

    # Create logger
    logger = logging.getLogger(f"winpact.{name}")
    logger.setLevel(level)

    # Remove old handlers (important when running in notebooks)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (optional)
    if console:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    logger.propagate = False  # prevent double logging

    logger.info(f"Logger initialised. Writing to {log_file}")
    return logger



def _deep_update(dst: dict, src: dict) -> None:
    """Recursively update dict dst with dict src."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v

def apply_overrides(obj: object, overrides: dict[str, object]) -> None:
    """
    Apply overrides from a dict to attributes of an object.

    Rules:
    - If target attr is a dict and override is a dict -> deep-merge (no full replace).
    - Else if attr exists (non-dict) -> replace value.
    - Else skip (avoids typos).
    """
    if not overrides:
        return

    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue

        current = getattr(obj, key)

        # Deep-merge dictionaries (critical for material_data)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_update(current, value)
        else:
            setattr(obj, key, value)



def get_input_parameter(market_inputs, *keys):
    try:
        value = market_inputs
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value
    except Exception:
        return None
    


# external timeseries looping and gap filling


TimedeltaLike = Union[pd.Timedelta, np.timedelta64, int, float]

def repeat_timeseries(
    df: pd.DataFrame,
    n_repeat: int,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Repeat a time-series DataFrame `n_repeat` times while extending timestamps
    so cadence continues seamlessly after each block.

    Assumptions:
      - `df[timestamp_col]` is datetime-like and regularly spaced (or close).
      - Other columns are repeated as-is.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime-like column.
    n_repeat : int
        Number of times to repeat (n>=1). If 1, returns a copy of `df`.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        Repeated DataFrame with extended timestamps and reset index.
    """
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")

    out = df.copy()

    # Ensure datetime
    if not np.issubdtype(out[timestamp_col].dtype, np.datetime64):
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=False)

    # Infer step (prefer mode of diffs; fallback to median)
    diffs = out[timestamp_col].diff().dropna()
    if diffs.empty:
        # Single-row case: assume 1 step of 1 second to allow extension
        step = pd.Timedelta(seconds=1)
    else:
        step = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()

    base_len = len(out)
    if n_repeat == 1 or base_len == 0:
        return out.reset_index(drop=True)

    # Build repeated blocks with timestamp offsets
    blocks = []
    base_ts = out[timestamp_col].to_numpy()
    for i in range(n_repeat):
        block = out.copy()
        offset = i * base_len * step
        block[timestamp_col] = base_ts + offset
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def gap_fill_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    data_cols: Optional[Sequence[str]] = None,
    freq: Optional[TimedeltaLike] = None,
    method: str = "linear",
    jitter: bool = True,
    jitter_scale: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Gap-fill a time-series by:
      1) Creating a complete timestamp index at the detected (or provided) frequency
      2) Interpolating numeric columns
      3) (Optional) Adding small Gaussian noise to values created by interpolation,
         scaled by each column's std dev.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing `timestamp_col` and one or more data columns.
    timestamp_col : str
        Name of the timestamp column.
    data_cols : sequence of str, optional
        Which columns to fill. Defaults to all numeric columns.
    freq : Timedelta-like, optional
        Desired frequency (e.g., '15min', pd.Timedelta('1h')). If None, inferred
        from the median step in the input.
    method : {'linear','time','nearest','slinear','quadratic','cubic', ...}
        Pandas interpolation method (for numeric columns).
    jitter : bool
        If True, add noise to *only* the points that were filled (not the originals).
    jitter_scale : float
        Multiplier for each column's std used as noise std (e.g., 0.05 => 5% of std).
    seed : int, optional
        Random seed for reproducibility of jitter.

    Returns
    -------
    pd.DataFrame
        Gap-filled DataFrame with the same columns and a complete timestamp coverage.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Ensure datetime
    if not np.issubdtype(out[timestamp_col].dtype, np.datetime64):
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=False)

    out = out.sort_values(timestamp_col)
    out = out.set_index(timestamp_col)

    # Choose columns
    if data_cols is None:
        data_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in data_cols if c not in out.columns]
    if missing_cols:
        raise ValueError(f"data_cols not found in df: {missing_cols}")

    # Infer frequency if not provided
    if freq is None:
        diffs = out.index.to_series().diff().dropna()
        if diffs.empty:
            # Single-row case: default to 1 second to permit reindex
            freq = pd.Timedelta(seconds=1)
        else:
            # Use median spacing to be robust to occasional outliers
            freq = diffs.median()

    # Build complete index
    full_index = pd.date_range(start=out.index.min(), end=out.index.max(), freq=freq)
    original_mask = pd.Series(True, index=out.index)
    filled = out.reindex(full_index)

    # Interpolate numeric columns
    for col in data_cols:
        filled[col] = filled[col].interpolate(
            method=method,
            limit_direction="both",
        )

    # Optional: jitter only the points that were created by reindex (i.e., were NaN)
    if jitter and jitter_scale > 0:
        rng = np.random.default_rng(seed)
        new_points_mask = ~filled.index.isin(original_mask.index)
        if new_points_mask.any():
            # Compute per-column std on original data (avoid NaNs)
            col_stds = {
                col: out[col].dropna().std(ddof=1) if out[col].dropna().size > 1 else 0.0
                for col in data_cols
            }
            for col in data_cols:
                std = col_stds[col]
                if std and np.isfinite(std):
                    noise = rng.normal(0.0, jitter_scale * std, new_points_mask.sum())
                    # Add noise only to the newly created/interpolated timestamps
                    col_vals = filled.loc[new_points_mask, col].to_numpy()
                    filled.loc[new_points_mask, col] = col_vals + noise

    filled = filled.reset_index().rename(columns={"index": timestamp_col})

    # Preserve any non-numeric columns by forward/back filling where sensible
    non_numeric = [c for c in out.columns if c not in data_cols]
    if non_numeric:
        # Reindex those too, then fill
        aux = out[non_numeric].reindex(full_index)
        aux = aux.ffill().bfill().reset_index(drop=True)
        for c in non_numeric:
            filled[c] = aux[c]

    return filled


def _infer_step(index: pd.Series) -> pd.Timedelta:
    """Infer the base step from a datetime-like Series of timestamps."""
    diffs = index.sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]  # ignore zero/negative diffs if any
    if diffs.empty:
        return pd.Timedelta(seconds=1)
    mode = diffs.mode()
    return (mode.iloc[0] if not mode.empty else diffs.median())


def _coerce_datetime(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, utc=False)
    return s


def remove_gaps_rebuild_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    freq: Optional[pd.Timedelta] = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Remove gaps by *compressing time* (no interpolation):
      - Keep rows and values as-is (order-preserving).
      - Rebuild a continuous timestamp column that starts at the first timestamp.
      - Step is inferred (most common diff) unless `freq` is given.

    Parameters
    ----------
    df : DataFrame
        Input with a datetime-like `timestamp_col`.
    timestamp_col : str
        Name of the timestamp column.
    freq : pd.Timedelta, optional
        Base frequency to enforce. If None, inferred from data.
    sort : bool
        If True (default), sort by timestamp before rebuilding.

    Returns
    -------
    DataFrame
        Same rows/values, but timestamps are rebuilt to be continuous.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out[timestamp_col] = _coerce_datetime(out[timestamp_col])
    if sort:
        out = out.sort_values(timestamp_col).reset_index(drop=True)

    step = pd.Timedelta(freq) if isinstance(freq, (pd.Timedelta, pd._libs.tslibs.timedeltas.Timedelta)) else (freq or None)
    if step is None:
        step = _infer_step(out[timestamp_col])

    start = out.loc[0, timestamp_col]
    new_ts = pd.Series(start + np.arange(len(out)) * step, name=timestamp_col)
    out[timestamp_col] = new_ts.to_numpy()
    return out


def _parse_duration_to_target_end(
    start: pd.Timestamp,
    duration: Union[str, pd.Timedelta, pd.DateOffset]
) -> pd.Timestamp:
    """
    Convert a duration spec to a concrete target end timestamp relative to `start`.
    Accepts:
      - pd.Timedelta
      - pd.DateOffset (supports months/years)
      - strings like '730 days', '18 months', '20 years', '24h', '1D', '90min'
    """
    if isinstance(duration, pd.Timestamp):
        # If user accidentally passes a timestamp, treat it as absolute end.
        return duration

    if isinstance(duration, pd.Timedelta):
        return start + duration

    if isinstance(duration, pd.DateOffset):
        return start + duration

    if isinstance(duration, str):
        s = duration.strip().lower()

        # Try standard Timedelta parsing first
        try:
            td = pd.to_timedelta(s)
            return start + td
        except Exception:
            pass

        # Lightweight parsing for months/years
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(year|years|yr|y)\s*$", s)
        if m:
            years = float(m.group(1))
            whole = int(years)
            frac = years - whole
            target = start + pd.DateOffset(years=whole)
            if frac:  # approximate fractional year as 365.2425 days
                target = target + pd.to_timedelta(frac * 365.2425, unit="D")
            return target

        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(month|months|mo|m)\s*$", s)
        if m:
            months = float(m.group(1))
            whole = int(months)
            frac = months - whole
            target = start + pd.DateOffset(months=whole)
            if frac:  # approximate fractional month as 30.44 days
                target = target + pd.to_timedelta(frac * 30.44, unit="D")
            return target

        raise ValueError(
            "Unrecognized duration string. Try formats like '20 years', '18 months', "
            "'730 days', '24h', or pass a pd.Timedelta / pd.DateOffset."
        )

    raise TypeError("`duration` must be str, pd.Timedelta, or pd.DateOffset.")


def repeat_timeseries_to_duration(
    df: pd.DataFrame,
    duration: Union[str, pd.Timedelta, pd.DateOffset],
    timestamp_col: str = "timestamp",
    trim_to_duration: bool = True,
) -> pd.DataFrame:
    """
    Repeat a time-series *until* a target overall duration is reached.

    - Keeps cadence seamless between blocks (no gaps, no overlap).
    - Computes as many whole repeats as needed; optionally trims the tail so
      the final timestamp does not exceed the target end.

    Parameters
    ----------
    df : DataFrame
        Input with a datetime-like `timestamp_col`.
    duration : str | pd.Timedelta | pd.DateOffset
        Desired total span from the *first* timestamp to the final timestamp.
        Examples: '20 years', pd.DateOffset(years=20), '730 days', '24h'.
    timestamp_col : str
        Name of the timestamp column.
    trim_to_duration : bool
        If True (default), drop trailing rows whose timestamps exceed the target end.

    Returns
    -------
    DataFrame
        Repeated DataFrame with extended timestamps and reset index.
    """
    if df.empty:
        return df.copy()

    base = df.copy()
    base[timestamp_col] = _coerce_datetime(base[timestamp_col])
    base = base.sort_values(timestamp_col).reset_index(drop=True)

    base_len = len(base)
    if base_len == 1:
        # Single-row case: extend with a 1-second cadence
        step = pd.Timedelta(seconds=1)
    else:
        step = _infer_step(base[timestamp_col])

    start = base.loc[0, timestamp_col]
    target_end = _parse_duration_to_target_end(start, duration)

    # Fast path when target already satisfied
    if base_len > 1 and base.loc[base_len - 1, timestamp_col] >= target_end:
        out = base.copy()
        if trim_to_duration:
            out = out[out[timestamp_col] <= target_end]
        return out.reset_index(drop=True)

    # Prepare numpy timestamps for fast offsetting
    base_ts = base[timestamp_col].to_numpy()

    blocks = []
    i = 0
    last_ts = base_ts[-1]
    # Keep adding blocks until we reach/past target_end
    while last_ts < target_end:
        offset = i * base_len * step
        block = base.copy()
        block[timestamp_col] = base_ts + offset
        blocks.append(block)
        i += 1
        last_ts = block[timestamp_col].iloc[-1]

        # Safety guard for pathological step/length
        if i > 1_000_000:
            raise RuntimeError("Too many repeats—check base step or duration.")

    out = pd.concat(blocks, ignore_index=True)

    if trim_to_duration:
        out = out[out[timestamp_col] <= target_end].reset_index(drop=True)

    return out


def check_time_series_alignment(self) -> None:
    """
    Ensure that MetEnvironment and MarketEnv time series are aligned:

    - both non-empty
    - same number of time steps
    - same timestamps (same start, end and intermediate points)

    If everything is consistent, prints and logs a summary.
    """

    logger = self.logger or logging.getLogger("winpact.env")

    met_df = self.metEnv.environmental_data_ts
    price_df = self.MarketEnv.el_price_records

    # 1) Basic sanity
    if met_df is None or met_df.empty:
        raise ValueError("MetEnvironment.environmental_data_ts is empty; cannot run coupled simulation.")

    if price_df is None or price_df.empty:
        raise ValueError("MarketEnv.el_price_records is empty; cannot run coupled simulation.")

    # 2) Normalize timestamp dtypes
    met_ts = pd.to_datetime(met_df["timestamp"])
    price_ts = pd.to_datetime(price_df["timestamp"])

    # 3) Check length
    if len(met_ts) != len(price_ts):
        raise ValueError(
            f"Time series length mismatch between MetEnv and MarketEnv: "
            f"{len(met_ts)} vs {len(price_ts)} rows."
        )

    # 4) Check exact timestamp alignment
    if not met_ts.equals(price_ts):
        msg_parts = []

        if met_ts.iloc[0] != price_ts.iloc[0]:
            msg_parts.append(
                f"start timestamps differ: "
                f"MetEnv={met_ts.iloc[0]!r}, MarketEnv={price_ts.iloc[0]!r}"
            )

        if met_ts.iloc[-1] != price_ts.iloc[-1]:
            msg_parts.append(
                f"end timestamps differ: "
                f"MetEnv={met_ts.iloc[-1]!r}, MarketEnv={price_ts.iloc[-1]!r}"
            )

        met_freq = pd.infer_freq(met_ts)
        price_freq = pd.infer_freq(price_ts)
        if met_freq != price_freq:
            msg_parts.append(
                f"inferred frequencies differ: MetEnv={met_freq}, MarketEnv={price_freq}"
            )

        details = " | ".join(msg_parts) if msg_parts else "timestamps are not identical."
        raise ValueError(f"MetEnv and MarketEnv time series are not aligned: {details}")

    # ----------------------------------------------------------------------------------
    # If we reach here, everything is OK → log and print summary
    # ----------------------------------------------------------------------------------

    freq = pd.infer_freq(met_ts)
    msg = (
        f"Time series alignment OK:\n"
        f"  • n_steps: {len(met_ts)}\n"
        f"  • start:   {met_ts.iloc[0]}\n"
        f"  • end:     {met_ts.iloc[-1]}\n"
        f"  • freq:    {freq}\n"
        f"  • sources: MetEnv + MarketEnv\n"
    )

    print(msg)            # console output
    logger.info(msg)      # logfile output


def save_sceanarios(
    scenarios: Sequence[Mapping[str, Any]],
    result_directory: Union[str, Path],
    name: Optional[str] = None,
    filename: str = "scenarios.json",
) -> Path:
    root = Path(result_directory)
    if name:
        root = root / name
    root.mkdir(parents=True, exist_ok=True)

    def _json_safe(x: Any) -> Any:
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, Mapping):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [_json_safe(v) for v in x]
        # last resort stringification
        try:
            return str(x)
        except Exception:
            return repr(x)

    payload = {
        "experiment_name": name,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "count": len(list(scenarios)),
        "scenarios": [_json_safe(dict(s)) for s in scenarios],
    }

    out_path = root / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path