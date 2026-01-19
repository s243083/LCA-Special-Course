#!/usr/bin/env python3
"""
Calibrate GBM (mu, sigma), mean-reverting OU (kappa, theta, sigma),
and simple Jump-Diffusion (lambda_jump, sigma_jump)
from historical price/index time series stored in a CSV.

Usage examples
--------------
Single series (column 'Price'):
    python calibrate_commodities.py data.csv --date-col Date --price-cols Price

Multiple materials in same file (columns 'Steel','Copper',...):
    python calibrate_commodities.py materials.csv --date-col Date --price-cols Steel Copper

If --price-cols is omitted, all numeric columns except the date column are used.
"""

import argparse
import sys
import csv
import numpy as np
import pandas as pd


def detect_delimiter(csv_path: str) -> str:
    """Auto-detect the delimiter used in a CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
    except Exception:
        # Default to comma if detection fails
        return ','


def normalize_number(value) -> float:
    """Convert European format numbers (period as thousands separator) to float.
    
    Examples:
        '11.580.921' -> 11580921.0
        '123,45' -> 123.45
        '123.45' -> 123.45 (ambiguous, assumes decimal point)
    """
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).strip()
    
    # Try to detect format based on position of separators
    # If last char is comma, it's likely the decimal separator (European: 123,45)
    # If last char is period, check if it looks like thousands separator (European: 1.234.567,89)
    
    # Replace European thousands separator (period) with empty string, but preserve decimal comma
    # Count periods and commas
    period_count = value.count('.')
    comma_count = value.count(',')
    
    if period_count > 0 and comma_count == 0:
        # Only periods: could be thousands separator or decimal point
        last_period = value.rfind('.')
        last_period_pos = len(value) - last_period - 1
        
        if last_period_pos == 3:
            # Period is 3 positions from end (likely thousands separator)
            value = value.replace('.', '')
        # else: period is decimal point, leave as is
    elif period_count > 0 and comma_count > 0:
        # Both periods and commas: European format with comma as decimal
        # Remove periods (thousands separators), keep comma
        value = value.replace('.', '').replace(',', '.')
    elif comma_count > 0 and period_count == 0:
        # Only commas: likely European decimal separator
        value = value.replace(',', '.')
    
    try:
        return float(value)
    except ValueError:
        return np.nan


def infer_dt_years(dates: pd.Series) -> float:
    """Infer time step Δt in years from a pandas datetime series or year integers."""
    # Remember original dtype before any processing
    is_numeric_orig = pd.api.types.is_numeric_dtype(dates)
    
    # Remove NaN and duplicates, then sort
    dates_clean = dates.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    
    # If originally numeric, keep as numeric
    if is_numeric_orig:
        year_diffs = dates_clean.diff().dropna().values
        
        if len(year_diffs) == 0:
            raise ValueError("Not enough unique date values to infer time step (need at least two unique timestamps).")
        
        median_years = np.median(year_diffs)
        std_years = np.std(year_diffs)
        
        if std_years > 0.1 * median_years:
            print(
                f"WARNING: time step is irregular (median={median_years:.2f} years, std={std_years:.2f} years). "
                "Using median step for Δt.",
                file=sys.stderr,
            )
        
        dt_years = float(median_years)
    else:
        # Treat as datetime strings
        dates_clean = pd.to_datetime(dates_clean)
        diffs = dates_clean.diff().dropna().dt.days.values
        
        if len(diffs) == 0:
            raise ValueError("Not enough unique date values to infer time step (need at least two unique timestamps).")
        
        median_days = np.median(diffs)
        std_days = np.std(diffs)
        
        if std_days > 0.1 * median_days:
            print(
                f"WARNING: time step is irregular (median={median_days:.2f} days, std={std_days:.2f} days). "
                "Using median step for Δt.",
                file=sys.stderr,
            )
        
        dt_years = median_days / 365.25
    
    if dt_years <= 0:
        raise ValueError("Inferred non-positive time step.")
    return dt_years


def calibrate_gbm(prices: pd.Series, dt_years: float):
    """
    Calibrate GBM parameters (mu, sigma) from a price series.

    GBM: dS = μ S dt + σ S dW
    Discrete log-returns r_t = ln(S_t / S_{t-1}):
        r_t ~ N( (μ - 0.5 σ^2) Δt, σ^2 Δt )
    """
    # Normalize prices (handle European format numbers)
    prices = prices.apply(normalize_number).astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 2:
        raise ValueError("Not enough data points for GBM calibration (need at least 3 prices).")

    r_bar = log_returns.mean()
    var_r = log_returns.var(ddof=1)

    sigma = np.sqrt(var_r / dt_years)
    mu = r_bar / dt_years + 0.5 * sigma ** 2

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "r_bar": float(r_bar),
        "var_r": float(var_r),
    }


def calibrate_ou_logprice(prices: pd.Series, dt_years: float):
    """
    Calibrate OU parameters (kappa, theta, sigma) on log-prices.

    OU on X_t = ln S_t:
        dX = κ(θ - X) dt + σ dW

    Discrete form:
        X_{t+Δt} = a + b X_t + ε_t
    where:
        b = exp(-κ Δt)
        a = θ (1 - b)
    """
    # Normalize prices (handle European format numbers)
    prices = prices.apply(normalize_number).astype(float)
    X = np.log(prices)
    X_t = X.shift(1).dropna()
    X_tp = X.loc[X_t.index]  # aligned X_{t+Δt}

    if len(X_t) < 2:
        raise ValueError("Not enough data points for OU calibration (need at least 3 prices).")

    # OLS regression: X_tp = a + b * X_t + eps
    X_mean = X_t.mean()
    Y_mean = X_tp.mean()

    cov = ((X_t - X_mean) * (X_tp - Y_mean)).sum()
    var_X = ((X_t - X_mean) ** 2).sum()

    b = cov / var_X
    a = Y_mean - b * X_mean

    # residuals
    residuals = X_tp - (a + b * X_t)
    var_eps = residuals.var(ddof=2)  # N-2 dof for regression

    if b <= 0 or b >= 1:
        print(
            f"WARNING: estimated b={b:.4f} is outside (0,1); OU mean reversion may be a poor fit.",
            file=sys.stderr,
        )

    # Convert to continuous-time OU parameters
    kappa = -np.log(b) / dt_years
    theta = a / (1.0 - b)

    sigma = np.sqrt(2 * kappa * var_eps / (1.0 - b ** 2))

    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "a": float(a),
        "b": float(b),
        "var_eps": float(var_eps),
    }


def calibrate_jump_diffusion(prices: pd.Series, dt_years: float, threshold_std: float = 2.5):
    """
    Rough calibration of jump-diffusion parameters (lambda_jump, sigma_jump)
    from log-return data.

    We:
      1) compute log-returns r_t
      2) compute their std dev, s
      3) mark 'jumps' as those with |r_t| > threshold_std * s
      4) set:
            lambda_jump ≈ N_jumps / (N_steps * Δt)   [jumps per year]
            sigma_jump  ≈ std dev of jump sizes (log space)

    Notes:
      - This is a simple heuristic, not full MLE.
      - μ and σ for the diffusion part remain those from calibrate_gbm().
      - In your CAPEX model, jumps are drawn as:
            jump_size ~ N(0, sigma_jump),
            num_jumps ~ Poisson(lambda_jump * T)
        and applied multiplicatively: S *= exp(jump_size).
    """
    # Normalize prices (handle European format numbers)
    prices = prices.apply(normalize_number).astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 2:
        raise ValueError("Not enough data points for jump calibration (need at least 3 prices).")

    r_std = log_returns.std(ddof=1)
    if r_std <= 0:
        raise ValueError("Zero standard deviation in log-returns; cannot detect jumps.")

    # Identify jumps by magnitude
    jump_mask = np.abs(log_returns) > threshold_std * r_std
    jumps = log_returns[jump_mask]
    n_jumps = jumps.shape[0]
    n_steps = log_returns.shape[0]

    total_time_years = n_steps * dt_years

    if total_time_years <= 0:
        raise ValueError("Non-positive total time for jump calibration.")

    if n_jumps == 0:
        # no jumps detected; return zero-intensity, zero-size
        lambda_jump = 0.0
        sigma_jump = 0.0
    else:
        lambda_jump = n_jumps / total_time_years
        sigma_jump = float(jumps.std(ddof=1))

    return {
        "lambda_jump": float(lambda_jump),
        "sigma_jump": float(sigma_jump),
        "n_jumps": int(n_jumps),
        "n_steps": int(n_steps),
        "threshold_std": float(threshold_std),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate GBM, OU, and Jump-Diffusion parameters from CSV price series.")
    parser.add_argument("csv_path", help="Path to CSV file with a date column and one or more price/index columns.")
    parser.add_argument(
        "--date-col",
        required=True,
        help="Name of the date/time column in the CSV."
    )
    parser.add_argument(
        "--price-cols",
        nargs="+",
        help="Names of price/index columns. If omitted, all numeric columns except the date column are used."
    )
    parser.add_argument(
        "--date-format",
        default=None,
        help="Optional explicit date format (passed to pandas.to_datetime)."
    )
    parser.add_argument(
        "--jump-threshold-std",
        type=float,
        default=2.5,
        help="Std-dev multiple used to classify jumps in log-returns (default: 2.5)."
    )

    args = parser.parse_args()

    # Load data with auto-detected delimiter
    try:
        delimiter = detect_delimiter(args.csv_path)
        df = pd.read_csv(args.csv_path, sep=delimiter)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.date_col not in df.columns:
        print(f"Date column '{args.date_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Parse dates only if not already numeric (e.g., Year column)
    if not pd.api.types.is_numeric_dtype(df[args.date_col]):
        try:
            df[args.date_col] = pd.to_datetime(df[args.date_col], format=args.date_format)
        except Exception as e:
            print(f"Error parsing dates: {e}", file=sys.stderr)
            sys.exit(1)

    df = df.sort_values(args.date_col).reset_index(drop=True)

    # Infer Δt in years
    try:
        dt_years = infer_dt_years(df[args.date_col])
    except Exception as e:
        print(f"Error inferring time step: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Inferred time step dt ≈ {dt_years:.6f} years\n")

    # Determine which price columns to use
    if args.price_cols is not None:
        price_cols = args.price_cols
    else:
        # Use all numeric columns except the date column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        price_cols = [c for c in numeric_cols if c != args.date_col]

    if not price_cols:
        print("No price columns specified or found as numeric columns.", file=sys.stderr)
        sys.exit(1)

    for col in price_cols:
        if col not in df.columns:
            print(f"Price column '{col}' not found in CSV, skipping.", file=sys.stderr)
            continue

        series = df[col].dropna()
        if len(series) < 3:
            print(f"Not enough data for column '{col}' (need at least 3 non-NaN rows), skipping.", file=sys.stderr)
            continue

        print("=" * 80)
        print(f"Series: {col}")

        # GBM calibration
        try:
            gbm_params = calibrate_gbm(series, dt_years)
            print("\nGBM parameters (annualised):")
            print(f"  mu    = {gbm_params['mu']:.6f}")
            print(f"  sigma = {gbm_params['sigma']:.6f}")
        except Exception as e:
            print(f"  GBM calibration failed: {e}", file=sys.stderr)

        # OU calibration
        try:
            ou_params = calibrate_ou_logprice(series, dt_years)
            print("\nOU (mean-reverting on log-price) parameters (annualised):")
            print(f"  kappa = {ou_params['kappa']:.6f}")
            print(f"  theta = {ou_params['theta']:.6f}")
            print(f"  sigma = {ou_params['sigma']:.6f}")
        except Exception as e:
            print(f"  OU calibration failed: {e}", file=sys.stderr)

        # Jump-Diffusion calibration
        try:
            jd_params = calibrate_jump_diffusion(series, dt_years, threshold_std=args.jump_threshold_std)
            print("\nJump-Diffusion parameters (rough, annualised):")
            print(f"  lambda_jump = {jd_params['lambda_jump']:.6f}  # expected jumps per year")
            print(f"  sigma_jump  = {jd_params['sigma_jump']:.6f}  # std of jump log-size")
            print(f"  n_jumps     = {jd_params['n_jumps']} (out of {jd_params['n_steps']} steps, threshold={jd_params['threshold_std']}σ)")
        except Exception as e:
            print(f"  Jump-Diffusion calibration failed: {e}", file=sys.stderr)

        print()

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
